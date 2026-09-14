"""FrozenPolicy: restore an OGBench impls checkpoint as a frozen, queryable policy.

Wraps the impls restore pattern (rebuild an identically-configured agent, then
`flax.serialization.from_state_dict` via `restore_agent`) behind a uniform
interface:

    policy = FrozenPolicy.load(run_dir, epoch)
    action = policy.act(ob, goal)                # deterministic (temperature=0)
    action = policy.act(ob, goal, seed=key, temperature=1.0)

Run dirs are the ones written by impls/main.py: they must contain flags.json
and params_<epoch>.pkl. Requires OGBENCH_IMPLS on sys.path (impls modules
import each other by top-level name).
"""

import json
import os
import sys

import jax
import numpy as np

IMPLS = os.environ.get('OGBENCH_IMPLS', '/path/to/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

from agents import agents  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


class FrozenPolicy:
    def __init__(self, agent, agent_name, env_name, seed):
        self._agent = agent
        self.agent_name = agent_name
        self.env_name = env_name
        self.train_seed = seed

    @classmethod
    def load(cls, run_dir, epoch, ob_dim=None, action_dim=None):
        with open(os.path.join(run_dir, 'flags.json')) as f:
            flags = json.load(f)
        config = flags['agent']
        agent_name = config['agent_name']

        if ob_dim is None or action_dim is None:
            ob_dim, action_dim = _example_dims(flags['env_name'])
        ex_observations = np.zeros((1, ob_dim), dtype=np.float32)
        ex_actions = np.zeros((1, action_dim), dtype=np.float32)

        agent_class = agents[agent_name]
        agent = agent_class.create(flags['seed'], ex_observations, ex_actions, config)
        agent = restore_agent(agent, run_dir, epoch)
        return cls(agent, agent_name, flags['env_name'], flags['seed'])

    def act(self, observations, goals, seed=None, temperature=0.0):
        # sample_actions always draws from the actor distribution, so a PRNG
        # key is mandatory even at temperature=0 (where the draw is
        # deterministic for const_std actors, and HIQL splits the key
        # internally).
        if seed is None:
            seed = jax.random.PRNGKey(0)
        actions = self._agent.sample_actions(
            observations=observations, goals=goals, seed=seed, temperature=temperature
        )
        return np.asarray(actions)

    def value(self, observations, goals):
        """This agent's OWN goal-conditioned state value V(s, g) on raw observations.

        Exists for value-based learners (GCIQL, GCIVL, HIQL define a 'value'
        module); GCBC has no critic and CRL/QRL parameterize Q/d differently.
        Used as an alternative rollout scorer (eval_planner --critic_scorer):
        one member's critic applied identically to every candidate's imagined
        states, so scores are comparable across the bank. Returns (N,).
        """
        try:
            v = self._agent.network.select('value')(
                np.asarray(observations, dtype=np.float32), np.asarray(goals, dtype=np.float32))
        except (KeyError, AttributeError, TypeError) as e:
            raise ValueError(f'{self.agent_name} exposes no goal-conditioned value network') from e
        v = np.asarray(v)
        # Ensembled heads (if any) -> conservative min, matching IQL's target.
        return v.min(axis=0) if v.ndim == 2 else v

    def q_min(self, observations, goals, actions):
        """This agent's OWN goal-conditioned action value min_j Q_j(s, a, g) on raw observations.

        GCIQL trains two Q heads (an ensemblized GCValue over [s; g; a]); the
        minimum over heads is the agent's own conservative critic, the same
        reduction it uses for its value and actor targets. Used by the
        model-free selector (eval_planner --critic_select_commit), which scores
        every bank member's proposed action at the current state without any
        imagined rollout. Returns (N,).
        """
        try:
            q = self._agent.network.select('critic')(
                np.asarray(observations, dtype=np.float32), np.asarray(goals, dtype=np.float32),
                np.asarray(actions, dtype=np.float32))
        except (KeyError, AttributeError, TypeError) as e:
            raise ValueError(f'{self.agent_name} exposes no goal-conditioned action-value network') from e
        q = np.asarray(q)
        return q.min(axis=0) if q.ndim == 2 else q


def _example_dims(env_name):
    """Observation/action dims from the env registration, without datasets."""
    import gymnasium
    import ogbench  # noqa: F401

    # Dataset names embed the dataset type as the second-to-last token
    # (e.g. antmaze-large-navigate-v0 -> antmaze-large-v0), matching
    # ogbench.make_env_and_datasets.
    tokens = env_name.split('-')
    env_id = '-'.join(tokens[:-2] + tokens[-1:]) if len(tokens) > 2 else env_name
    env = gymnasium.make(env_id)
    dims = (env.observation_space.shape[-1], env.action_space.shape[-1])
    env.close()
    return dims
