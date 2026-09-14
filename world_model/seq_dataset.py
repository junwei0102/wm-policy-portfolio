"""Multi-step window sampling + goal-conditioned value batches for WM training.

Wraps an OGBench compact `Dataset` (observations/actions/terminals/valids).
Compact semantics (ogbench/utils.py:60-73): for a trajectory s0..s4,
terminals = [0,0,0,1,1] (last transition start AND final state) and
valids = [1,1,1,1,0]. Hence for any valid start index t,
`terminal_locs[searchsorted(terminal_locs, t)]` is the last valid transition
start of t's trajectory, and that index + 1 is the trajectory's final state.

Value-head batches reuse GCDataset's goal-sampling machinery verbatim: with
gc_negative=False, batch['rewards'] equals the goal-reached indicator that the
LAVL expectile-TD head consumes as a 0/-1 reward.
"""

import os
import sys

import numpy as np

IMPLS = os.environ.get('OGBENCH_IMPLS', '/path/to/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

from utils.datasets import GCDataset  # noqa: E402


def compute_norm_stats(dataset, eps=1e-6):
    """Fixed per-dimension stats from the train split: obs and one-step deltas."""
    obs = np.asarray(dataset['observations'], dtype=np.float64)
    if 'valids' in dataset:
        (starts,) = np.nonzero(np.asarray(dataset['valids']) > 0)
    else:
        starts = np.arange(len(obs) - 1)
    deltas = obs[starts + 1] - obs[starts]
    return dict(
        obs_mean=obs.mean(axis=0).astype(np.float32),
        obs_std=np.maximum(obs.std(axis=0), eps).astype(np.float32),
        delta_mean=deltas.mean(axis=0).astype(np.float32),
        delta_std=np.maximum(deltas.std(axis=0), eps).astype(np.float32),
    )


class WMSequenceDataset:
    """Samples H-step windows plus goal-conditioned value-head batches."""

    def __init__(
        self,
        dataset,
        horizon,
        discount=0.99,
        p_curgoal=0.2,
        p_trajgoal=0.5,
        p_randomgoal=0.3,
        geom_sample=True,
        frame_skip=1,
    ):
        if frame_skip != 1:
            # Humanoid k-step jump models need action blocks per jump; the
            # sampler below assumes single-step transitions.
            raise NotImplementedError('frame_skip > 1 not implemented yet')
        self.dataset = dataset
        self.horizon = horizon
        self.discount = discount
        self.p_curgoal = p_curgoal
        self.p_trajgoal = p_trajgoal
        self.p_randomgoal = p_randomgoal
        self.geom_sample = geom_sample
        gc_config = dict(
            discount=discount,
            value_p_curgoal=p_curgoal,
            value_p_trajgoal=p_trajgoal,
            value_p_randomgoal=p_randomgoal,
            value_geom_sample=geom_sample,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=None,
            frame_stack=None,
        )
        self.gc_dataset = GCDataset(dataset=dataset, config=gc_config)
        (self.terminal_locs,) = np.nonzero(np.asarray(dataset['terminals']) > 0)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        H = self.horizon
        final = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        offs = np.arange(H)
        action_pos = np.minimum(idxs[:, None] + offs[None, :], final[:, None])
        step_valid = (idxs[:, None] + offs[None, :] <= final[:, None]).astype(np.float32)
        obs_pos = np.minimum(idxs[:, None] + np.arange(H + 1)[None, :], (final + 1)[:, None])

        reach_goal_idx, reach_targets = self._sample_reach_goals(idxs, final)

        gc_batch = self.gc_dataset.sample(len(idxs), idxs=idxs)
        return dict(
            obs_seq=self.dataset['observations'][obs_pos],
            action_seq=self.dataset['actions'][action_pos],
            step_valid=step_valid,
            observations=gc_batch['observations'],
            next_observations=gc_batch['next_observations'],
            value_goals=gc_batch['value_goals'],
            # With gc_negative=False: rewards = success in {0,1}, masks = 1-success.
            # The LAVL TD head consumes rewards-1 (0/-1 convention) and masks.
            rewards=gc_batch['rewards'],
            masks=gc_batch['masks'],
            # Diagnostic only (validation/value_rank_corr): goals with known
            # temporal offsets and gamma^D targets; not used by any loss.
            reach_goals=self.dataset['observations'][reach_goal_idx],
            reach_targets=reach_targets,
            # Uniform random goals for LAVL's local-smoothness regularizer.
            random_goals=self.dataset['observations'][self.dataset.get_random_idxs(len(idxs))],
        )

    def _sample_reach_goals(self, idxs, final):
        """Goals with KNOWN temporal offsets (validation rank-correlation diagnostic).

        Monte-Carlo discounted-reachability targets, no task-dim knowledge:
          curgoal (p_curgoal):    goal = s_t itself,           target = 1
          trajgoal (p_trajgoal):  goal = s_{t+D}, D geometric  target = gamma^D
                                  (clipped to the trajectory's final state)
          randomgoal (p_random):  uniform dataset state,       target = 0
        The random-goal target-0 is the standard unreachable-negative
        approximation; test-time ranking compares states against a FIXED goal,
        so its bias is shared across candidates.
        """
        n = len(idxs)
        final_state = final + 1  # index of the trajectory's final state
        remaining = final_state - idxs  # >= 1 for every valid start
        if self.geom_sample:
            dist = np.random.geometric(p=1 - self.discount, size=n)
        else:
            dist = 1 + np.floor(np.random.rand(n) * remaining).astype(np.int64)
        dist = np.minimum(dist, remaining)

        r = np.random.rand(n)
        is_cur = r < self.p_curgoal
        is_traj = ~is_cur & (r < self.p_curgoal + self.p_trajgoal)
        rand_idx = self.dataset.get_random_idxs(n)
        goal_idx = np.where(is_cur, idxs, np.where(is_traj, idxs + dist, rand_idx))
        targets = np.where(
            is_cur, 1.0, np.where(is_traj, self.discount**dist, 0.0)
        ).astype(np.float32)
        return goal_idx, targets
