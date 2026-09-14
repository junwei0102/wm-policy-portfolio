"""Oracle-dynamics WMPP: the RolloutRanker arbitration with the k-step branches
executed in the REAL simulator instead of the learned world model.

At every arbitration boundary the live env is snapshotted (interfaces.sim_branch.
SimBranch, bit-exact within one reset), every bank policy is rolled out k steps
from that state in the simulator, the visited real states are scored with the
SAME value head / ensemble reduction / horizon aggregation as RolloutRanker, and
the env is restored before the chosen policy acts. This isolates what the
learned dynamics cost: WMPP - SimWMPP = model-induced ranking loss; SimWMPP -
best policy = what the value head + switching logic deliver with perfect
dynamics. The transition counter records SIMULATOR steps (P*k per decision).
"""

import time

import numpy as np

from interfaces.sim_branch import SimBranch
from planners.portfolio import ENS_AGGS, _EpisodeLog, _ens_reduce, _lexi_argmax


class SimRolloutRanker:
    def __init__(self, env, wm, bank, counter, horizon, replan_every=None, score_agg='max', ens_agg='mean',
                 critic_fn=None):
        # critic_fn(states, goals) -> (N,): score the REAL branches with a bank member's own value
        # (the direct value of the paper) instead of the world model's metric head.
        self.critic_fn = critic_fn
        assert critic_fn is not None or wm.config['value_head'], 'SimRolloutRanker scores with the value head'
        assert score_agg in ('max', 'mean', 'last'), score_agg
        assert ens_agg in ENS_AGGS, ens_agg
        self.env = env
        self.u = env.unwrapped
        self.sim = SimBranch(env)
        self.wm = wm
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.counter = counter
        self.horizon = int(horizon)
        self.replan_every = int(replan_every or horizon)
        self.score_agg = score_agg
        self.ens_agg = ens_agg
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_replan = None
        self._current = None
        self._log = _EpisodeLog()
        self._sim_steps = 0

    def episode_info(self):
        info = self._log.info()
        info['sim_steps'] = self._sim_steps
        return info

    def _agg_horizon(self, scores_hp):
        if self.score_agg == 'mean':
            return scores_hp.mean(axis=0)
        if self.score_agg == 'last':
            return scores_hp[-1]
        return scores_hp.max(axis=0)

    def _branch(self, policy, ob, goal):
        """Roll `policy` for k real steps from the current (restored) state; returns (k, d) states."""
        states = []
        x = ob
        for _ in range(self.horizon):
            a = np.clip(np.asarray(policy.act(x, goal)), -1, 1)
            x, _, terminated, truncated, _ = self.u.step(a)
            x = np.asarray(x, dtype=np.float32)
            states.append(x.copy())
            self._sim_steps += 1
            self.counter.add(1)
            if terminated or truncated:
                break
        while len(states) < self.horizon:  # pad a terminated branch with its final state
            states.append(states[-1].copy())
        return np.stack(states)

    def _replan(self, ob, goal):
        t0 = time.perf_counter()
        snap = self.sim.snapshot()
        trajs = []
        for name in self.names:
            self.sim.restore(snap)
            trajs.append(self._branch(self.bank[name], ob, goal))
        self.sim.restore(snap)  # the real episode continues exactly where it was
        traj = np.stack(trajs, axis=1)  # (H, P, d)
        H, P = traj.shape[:2]
        E = self.wm.config['num_members']
        if self.critic_fn is not None:
            flat = traj.reshape(H * P, -1)
            vals = np.asarray(self.critic_fn(flat, np.broadcast_to(goal, flat.shape[:-1] + goal.shape[-1:])))
            vals = np.broadcast_to(vals.reshape(1, H * P), (E, H * P))  # one critic; ensemble axis is a no-op
            self._log.val_calls += H * P
        else:
            traj_e = np.broadcast_to(traj.reshape(1, H * P, -1), (E, H * P, traj.shape[-1]))
            goals_e = np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])
            vals = np.asarray(self.wm.value_score(traj_e, goals_e))  # (E, H*P): every head scores the same real state
            self._log.val_calls += H * E * P
        scores = self._agg_horizon(_ens_reduce(vals.reshape(E, H, P), self.ens_agg))
        self.counter.mark_decision()
        self._current = self.names[_lexi_argmax(scores, np.arange(P, 0, -1.0))]
        self._steps_since_replan = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if self._steps_since_replan is None or self._steps_since_replan >= self.replan_every:
            self._replan(ob, goal)
        self._steps_since_replan += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class SimOracleArbiter:
    """Dynamic simulator oracle (privileged; evaluation only).

    At every commitment boundary the live env is snapshotted, EVERY bank policy is
    rolled out in the real simulator for the remaining episode budget, and the env
    is restored. Rule: keep the incumbent if its branch succeeds; otherwise switch to
    the succeeding policy with the fewest steps-to-success (ties -> name order); if
    no branch succeeds keep the incumbent. The initial incumbent is `fallback` (the
    best fixed policy), so the oracle is "best policy + perfect-foresight switching
    every c steps": an upper bound for any arbiter with commitment >= c on this bank.
    OGBench episodes terminate at success, so a branch's final success is its
    ever-success. The counter records simulator steps (P x remaining budget worst case).
    """

    def __init__(self, env, bank, counter, commit, episode_len, fallback):
        assert fallback in bank, (fallback, sorted(bank))
        self.env = env
        self.u = env.unwrapped
        self.sim = SimBranch(env)
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.counter = counter
        self.commit = int(commit)
        self.episode_len = int(episode_len)
        self.fallback = fallback
        self.reset_episode()

    def reset_episode(self):
        self._t = 0
        self._steps_since = None
        self._current = self.fallback
        self._log = _EpisodeLog()
        self._sim_steps = 0
        self._boundaries = []
        self._n_success_candidates = 0

    def episode_info(self):
        info = self._log.info()
        info['sim_steps'] = self._sim_steps
        info['oracle_boundaries'] = '|'.join(self._boundaries)
        info['n_success_candidates'] = self._n_success_candidates
        return info

    def _branch(self, policy, ob, goal, budget):
        """Roll `policy` from the current (restored) state; returns (success, steps)."""
        x, steps, success = ob, 0, 0.0
        for _ in range(int(budget)):
            a = np.clip(np.asarray(policy.act(x, goal)), -1, 1)
            x, _, terminated, truncated, info = self.u.step(a)
            steps += 1
            self._sim_steps += 1
            self.counter.add(1)
            success = float(info['success'])
            if terminated or truncated or success > 0:
                break
        return success, steps

    def _choose(self, outcomes):
        winners = [n for n in self.names if outcomes[n][0] > 0]
        self._n_success_candidates += len(winners)
        if self._current in winners:
            return self._current
        if winners:
            return min(winners, key=lambda n: (outcomes[n][1], n))
        return self._current

    def _replan(self, ob, goal):
        t0 = time.perf_counter()
        snap = self.sim.snapshot()
        budget = self.episode_len - self._t
        outcomes = {}
        for name in self.names:
            self.sim.restore(snap)
            outcomes[name] = self._branch(self.bank[name], ob, goal, budget)
        self.sim.restore(snap)  # the real episode continues exactly where it was
        choice = self._choose(outcomes)
        self._boundaries.append(
            f't{self._t}:' + ','.join(f'{n}={int(outcomes[n][0])}@{outcomes[n][1]}' for n in self.names) + f'->{choice}')
        self._current = choice
        self._steps_since = 0
        self._log.n_plans += 1
        self.counter.mark_decision()
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if self._steps_since is None or self._steps_since >= self.commit:
            self._replan(ob, goal)
        self._steps_since += 1
        self._t += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)
