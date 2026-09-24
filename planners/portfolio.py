"""Planner ladder rungs 1-2: one-step chooser and whole-policy rollout ranker.

Both implement the FrozenPolicy `.act(ob, goal, temperature)` protocol, so
`evaluation.paired_eval.evaluate_paired` runs them unchanged; both consume a
TransitionCounter through `wm_step`/`imagine_policy_rollout` only.

Budget parity: the chooser replans every env step at cost P*E; the ranker
replans every R=horizon env steps at cost P*E*horizon — identical per-env-step
budget. `mark_decision()` is called at every replan so the eval harness can
assert this.

Deterministic tie-breaking: policies are iterated in sorted-name order and
argmax breaks ties toward the earlier name (np.argmax convention on exact
ties; scores are compared lexicographically: success score, then negative
normalized goal distance).
"""

import time

import numpy as np

from world_model.rollout import imagine_bank_rollout, wm_step


class _EpisodeLog:
    """Per-episode decision record shared by both planners.

    The chain collapses consecutive steps under the same policy into
    'name:steps' segments, e.g. 'hiql-sd0:100>gciql-sd0:200>crl-sd0:37'.
    """

    def __init__(self):
        self.segments = []  # list of [name, steps]
        self.plan_seconds = 0.0
        self.n_plans = 0
        # Per-episode model-call accounting (mirrors TransitionCounter, but
        # per episode so episodes.csv carries it): dynamics = imagined
        # transitions (one per ensemble member per branch per step); value =
        # value-head evaluations (one per imagined state per member).
        self.dyn_calls = 0
        self.val_calls = 0

    def record_step(self, name):
        if self.segments and self.segments[-1][0] == name:
            self.segments[-1][1] += 1
        else:
            self.segments.append([name, 1])

    def info(self):
        return dict(
            policy_chain='>'.join(f'{n}:{s}' for n, s in self.segments),
            n_switches=max(len(self.segments) - 1, 0),
            n_plans=self.n_plans,
            plan_ms_total=1000.0 * self.plan_seconds,
            wm_transitions=self.dyn_calls,
            value_evals=self.val_calls,
        )

class _DecisionLog:
    """Optional per-decision record (state, goal, scores, winner) for analysis
    figures. Only allocated when a planner is built with log_decisions=True;
    never touches the decision itself or any RNG."""

    def __init__(self):
        self.rows = []

    def record(self, t, ob, goal, scores, winner, explore=False):
        self.rows.append((int(t), np.asarray(ob, dtype=np.float32).copy(),
                          np.asarray(goal, dtype=np.float32).copy(),
                          np.asarray(scores, dtype=np.float32).copy(), int(winner), bool(explore)))

    def pop(self):
        rows, self.rows = self.rows, []
        if not rows:
            return None
        return dict(
            t=np.array([r[0] for r in rows], dtype=np.int32),
            obs=np.stack([r[1] for r in rows]),
            goal=np.stack([r[2] for r in rows]),
            scores=np.stack([r[3] for r in rows]),
            winner=np.array([r[4] for r in rows], dtype=np.int32),
            explore=np.array([r[5] for r in rows], dtype=bool),
        )


def _lexi_argmax(primary, secondary):
    """Argmax of primary; exact ties broken by secondary; then lowest index."""
    best = np.flatnonzero(primary == primary.max())
    if len(best) == 1:
        return int(best[0])
    return int(best[np.argmax(secondary[best])])


ENS_AGGS = ('mean', 'min', 'lcb')


def _ens_reduce(vals_e, ens_agg):
    """Collapse the ensemble axis of an (E, ...) value array.

    mean — ensemble mean (paper default, eq. 12);
    min  — pessimistic: worst member;
    lcb  — mean minus one std over members (disagreement-penalised).
    """
    if ens_agg == 'mean':
        return vals_e.mean(axis=0)
    if ens_agg == 'min':
        return vals_e.min(axis=0)
    if ens_agg == 'lcb':
        return vals_e.mean(axis=0) - vals_e.std(axis=0)
    raise ValueError(ens_agg)


class OneStepChooser:
    def __init__(self, wm, bank, counter, progress_fn=None):
        self.wm = wm
        self.bank = dict(sorted(bank.items()))
        self.counter = counter
        self.names = list(self.bank)
        self.progress_fn = progress_fn
        self.reset_episode()

    def reset_episode(self):
        self._log = _EpisodeLog()

    def episode_info(self):
        return self._log.info()

    def act(self, ob, goal, temperature=0.0):
        t0 = time.perf_counter()
        E = self.wm.config['num_members']
        P = len(self.names)
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        actions = np.stack(
            [np.clip(np.asarray(p.act(ob, goal)), -1, 1) for p in self.bank.values()]
        )  # (P, da)
        obs_e = np.broadcast_to(ob, (E, P, ob.shape[-1]))
        acts_e = np.broadcast_to(actions, (E, P, actions.shape[-1]))
        next_e = wm_step(self.wm, obs_e, acts_e, self.counter)  # (E, P, d)

        if self.progress_fn is not None:
            goals_p = np.broadcast_to(goal, (P, goal.shape[-1]))
            prog = np.stack(
                [self.progress_fn(next_e[e], goals_p) for e in range(E)]
            ).mean(axis=0)  # (P,)
        else:
            obs_std = np.asarray(self.wm.normalizer['obs_std'])
            prog = -np.linalg.norm((next_e - goal) / obs_std, axis=-1).mean(axis=0)

        self.counter.mark_decision()
        self._log.dyn_calls += E * P
        winner = _lexi_argmax(prog, np.arange(P, 0, -1.0))
        self._log.record_step(self.names[winner])
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0
        return actions[winner]


class RolloutRanker:
    """Imagine-then-commit planner: the general (k, c) cell of the 2x2.

    `horizon` (k) = scoring/imagination horizon; `replan_every` (c) =
    commitment: the winner runs closed-loop on real observations for c env
    steps before re-arbitration. The paper's ablation cells are
    (k=1, c=1), (k=1, c=C), (k=H, c=1), (k=H, c=C). Budget per env step is
    P*E*k/c — annotated in results, deliberately NOT equalized across cells
    (the ablation isolates which knob carries the gain).
    """

    def __init__(
        self,
        wm,
        bank,
        counter,
        horizon,
        replan_every=None,
        progress_fn=None,
        score_mode='progress',
        score_agg='max',
        explore_every=None,
        ens_agg='mean',
        critic_fn=None,
        log_decisions=False,
        native_fns=None,
    ):
        self.wm = wm
        self.bank = dict(sorted(bank.items()))
        self.counter = counter
        self.horizon = horizon
        self._dec = _DecisionLog() if log_decisions else None
        self.replan_every = replan_every or horizon
        self.names = list(self.bank)
        self.progress_fn = progress_fn
        # Ensemble reduction BEFORE the horizon aggregation (eq. 12 uses the mean).
        assert ens_agg in ENS_AGGS, ens_agg
        self.ens_agg = ens_agg
        # Scheduled exploration: at a replan boundary reached >= explore_every
        # env steps after the last exploration, commit to the LEAST-USED policy
        # of the episode (ties among least-used broken by the model score)
        # instead of the argmax. None = plain WMPA.
        self.explore_every = int(explore_every) if explore_every else None
        # Horizon aggregation of the per-step (H, P) score matrix:
        #   max  — best state visited within k (optimistic; default)
        #   mean — average along the imagined path (rewards steady progress)
        #   last — state the policy ENDS in after k steps (matches commit-k
        #          semantics: the next replan starts from wherever it leaves you)
        assert score_agg in ('max', 'mean', 'last'), score_agg
        self.score_agg = score_agg
        # 'progress': task-relevant progress (privileged task dims; diagnostics
        # only). 'value': the learned LAVL value head ALONE — score is the best
        # (over horizon, ensemble-mean) -d(s, g) toward the goal; no
        # task-dimension knowledge anywhere.
        # 'critic': a bank member's own V(s, g) (e.g. GCIQL's IQL value) applied
        # to every candidate's imagined raw states -- an ablation of the LAVL
        # metric head that keeps dynamics, horizon, and commitment identical.
        # 'native': policy i's imagined states are scored by policy i's OWN goal-conditioned
        # value network (native_fns[name](states, goals) -> (N,)); the scores of different
        # members then come from different critics -- the own-value comparison of the appendix.
        assert score_mode in ('progress', 'value', 'critic', 'native'), score_mode
        if score_mode == 'value':
            assert wm.config['value_head'], 'WM was trained without a value head'
        if score_mode == 'critic':
            assert critic_fn is not None, 'score_mode=critic needs critic_fn(states, goals) -> (N,)'
        if score_mode == 'native':
            assert native_fns is not None and all(n in native_fns for n in self.names), 'score_mode=native needs a value fn per member'
        self.native_fns = native_fns
        self.critic_fn = critic_fn
        self.score_mode = score_mode
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_replan = None
        self._current = None
        self._log = _EpisodeLog()
        self._usage = {n: 0 for n in self.names}
        self._steps_since_explore = 0
        self._n_explore = 0
        self._t = 0  # env steps taken in this episode (decision log only)
        if self._dec is not None:
            self._dec.rows = []

    def episode_info(self):
        info = self._log.info()
        info['n_explore'] = self._n_explore
        return info

    def pop_decisions(self):
        return self._dec.pop() if self._dec is not None else None

    def _agg_horizon(self, scores_hp):
        """(H, P) per-step ensemble-mean scores -> (P,) via self.score_agg."""
        if self.score_agg == 'mean':
            return scores_hp.mean(axis=0)
        if self.score_agg == 'last':
            return scores_hp[-1]
        return scores_hp.max(axis=0)

    def _replan(self, ob, goal):
        t0 = time.perf_counter()
        obs_std = np.asarray(self.wm.normalizer['obs_std'])
        roll = imagine_bank_rollout(
            self.wm, [self.bank[n] for n in self.names], ob, goal, self.horizon, self.counter
        )
        traj = roll['obs_traj'][1:]  # (H, E, P, d)
        H, E, P = traj.shape[:3]
        self._log.dyn_calls += H * E * P
        if self.score_mode == 'value':
            traj_e = np.moveaxis(traj, 1, 0).reshape(E, H * P, -1)  # (E, H*P, d)
            goals_e = np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])
            vals = np.asarray(self.wm.value_score(traj_e, goals_e))  # (E, H*P)
            self._log.val_calls += H * E * P
            scores = self._agg_horizon(_ens_reduce(vals.reshape(E, H, P), self.ens_agg))  # (P,)
        elif self.score_mode == 'critic':
            flat = traj.reshape(H * E * P, -1)  # raw imagined states
            goals_flat = np.broadcast_to(goal, flat.shape[:-1] + goal.shape[-1:])
            vals = np.asarray(self.critic_fn(flat, goals_flat)).reshape(H, E, P)
            self._log.val_calls += H * E * P
            scores = self._agg_horizon(_ens_reduce(np.moveaxis(vals, 1, 0), self.ens_agg))  # (P,)
        elif self.score_mode == 'native':
            vals = np.empty((H, E, P), dtype=np.float64)
            goals_flat = np.broadcast_to(goal, (H * E,) + goal.shape[-1:])
            for p, name in enumerate(self.names):
                vals[:, :, p] = np.asarray(self.native_fns[name](traj[:, :, p, :].reshape(H * E, -1), goals_flat)).reshape(H, E)
            self._log.val_calls += H * E * P
            scores = self._agg_horizon(_ens_reduce(np.moveaxis(vals, 1, 0), self.ens_agg))  # (P,)
        elif self.progress_fn is not None:
            flat = traj.reshape(H * E * P, -1)
            goals_flat = np.broadcast_to(goal, flat.shape)
            scores = self._agg_horizon(
                self.progress_fn(flat, goals_flat).reshape(H, E, P).mean(axis=1)
            )  # (P,)
        else:
            scores = self._agg_horizon(
                -np.linalg.norm((traj - goal) / obs_std, axis=-1).mean(axis=1)
            )
        self.counter.mark_decision()
        if self.explore_every and self._steps_since_explore >= self.explore_every:
            least = min(self._usage.values())
            cand = np.array([i for i, n in enumerate(self.names) if self._usage[n] == least])
            winner = int(cand[_lexi_argmax(scores[cand], np.arange(len(cand), 0, -1.0))])
            self._steps_since_explore = 0
            self._n_explore += 1
            explored = True
        else:
            winner = _lexi_argmax(scores, np.arange(len(self.names), 0, -1.0))
            explored = False
        self._current = self.names[winner]
        self._steps_since_replan = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0
        if self._dec is not None:
            self._dec.record(self._t, ob, goal, scores, winner, explored)

    def act(self, ob, goal, temperature=0.0):
        if self._steps_since_replan is None or self._steps_since_replan >= self.replan_every:
            self._replan(np.asarray(ob, dtype=np.float32), np.asarray(goal, dtype=np.float32))
        self._steps_since_replan += 1
        self._steps_since_explore += 1
        self._t += 1
        self._usage[self._current] += 1
        self._log.record_step(self._current)
        # Winner runs closed-loop on the REAL observation.
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class RandomArbiter:
    """Random-arbitration control: same commitment schedule as WMPA, no model.

    At every arbitration boundary (every `commit` real env steps, starting at
    the first step of the episode) a policy is drawn UNIFORMLY from the full
    bank, independently of the previous choice (the active policy is a valid
    draw — no forced switch), and executed closed-loop on real observations for
    `commit` steps. Uses no world model and no value head: dynamics/value call
    counts are exactly zero. Isolates the value of model-based *selection*
    from the value of the commitment schedule itself.

    Reproducibility: the draw stream is seeded per episode from `seed` and the
    episode's first (observation, goal) pair, so the same episode key always
    yields the same schedule regardless of evaluation order.
    """

    def __init__(self, bank, commit, seed=0, log_decisions=False):
        assert commit >= 1, commit
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.commit = int(commit)
        self.seed = int(seed)
        self._dec = _DecisionLog() if log_decisions else None
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_draw = None
        self._current = None
        self._rng = None
        self._log = _EpisodeLog()
        self._t = 0
        if self._dec is not None:
            self._dec.rows = []

    def episode_info(self):
        return self._log.info()

    def pop_decisions(self):
        return self._dec.pop() if self._dec is not None else None

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def _draw(self):
        t0 = time.perf_counter()
        self._current = self.names[int(self._rng.integers(len(self.names)))]
        self._steps_since_draw = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        if self._rng is None:
            self._seed_rng(ob, goal)
        if self._steps_since_draw is None or self._steps_since_draw >= self.commit:
            self._draw()
            if self._dec is not None:
                self._dec.record(self._t, ob, goal, np.full(len(self.names), np.nan),
                                 self.names.index(self._current), False)
        self._steps_since_draw += 1
        self._t += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class NativeSelectArbiter:
    """Model-free arbitration by each member's OWN critic: at every arbitration boundary
    (every `commit` real env steps) member i scores the CURRENT state with its own
    goal-conditioned value network, i* = argmax_i V_i(s, g), and pi_{i*} runs closed-loop
    for `commit` steps. No imagined states, no shared scorer: the values of different
    members come from different estimators. Dynamics-call count is zero; value calls are
    P per decision.
    """

    def __init__(self, bank, native_fns, commit, seed=0):
        assert commit >= 1, commit
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        assert all(n in native_fns for n in self.names), 'every member needs its own value fn'
        self.native_fns = native_fns
        self.commit = int(commit)
        self.seed = int(seed)  # unused (deterministic)
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_select = None
        self._current = None
        self._log = _EpisodeLog()

    def episode_info(self):
        return self._log.info()

    def _select(self, ob, goal):
        t0 = time.perf_counter()
        ob = np.asarray(ob, dtype=np.float32)[None]
        goal = np.asarray(goal, dtype=np.float32)[None]
        v = np.array([float(np.asarray(self.native_fns[n](ob, goal)).reshape(-1)[0]) for n in self.names])
        self._log.val_calls += len(self.names)
        self._current = self.names[_lexi_argmax(v, np.arange(len(self.names), 0, -1.0))]
        self._steps_since_select = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        if self._steps_since_select is None or self._steps_since_select >= self.commit:
            self._select(ob, goal)
        self._steps_since_select += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class CriticSelectArbiter:
    """Model-free arbitration by an action-value critic: no imagined states.

    At every arbitration boundary (every `commit` real env steps, starting at
    the first step) each bank member proposes its action a_i = pi_i(s, g) at
    the CURRENT state, one critic scores them, i* = argmax_i q_fn(s, a_i, g),
    and pi_{i*} runs closed-loop for `commit` steps. With GCIQL's twin heads
    q_fn = min_j Q_j (FrozenPolicy.q_min) this is the zero-rollout counterpart
    of the direct-value WMPA cell: same critic family, same candidates, same
    commitment, no dynamics model. Dynamics-call count is exactly zero; value
    calls are P per decision.
    """

    def __init__(self, bank, q_fn, commit, seed=0):
        assert commit >= 1, commit
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.q_fn = q_fn
        self.commit = int(commit)
        self.seed = int(seed)  # unused (deterministic); kept for interface symmetry
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_select = None
        self._current = None
        self._log = _EpisodeLog()

    def episode_info(self):
        return self._log.info()

    def _select(self, ob, goal):
        t0 = time.perf_counter()
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        acts = np.stack([np.clip(np.asarray(self.bank[n].act(ob, goal)), -1, 1) for n in self.names])  # (P, da)
        P = len(self.names)
        q = np.asarray(self.q_fn(np.broadcast_to(ob, (P,) + ob.shape), np.broadcast_to(goal, (P,) + goal.shape), acts),
                       dtype=np.float64).reshape(P)
        self._log.val_calls += P
        self._current = self.names[_lexi_argmax(q, np.arange(P, 0, -1.0))]
        self._steps_since_select = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        if self._steps_since_select is None or self._steps_since_select >= self.commit:
            self._select(ob, goal)
        self._steps_since_select += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class PolicyMPC:
    """Sampling-MPC baseline on ONE fixed policy (the action prior).

    Isolates world-model *search* from portfolio *selection*: the candidates
    are not different policies but N Gaussian perturbations of a single
    policy's action. At every arbitration boundary (every `replan_every`=c real
    env steps, starting at the first step): candidate 0 is the policy mean
    mu(s,g); candidates 1..N-1 are clip(mu + sigma*eps_i, -1, 1) with
    eps_i ~ N(0, I) — i.e. samples from the OGBench actor distribution at
    temperature=sigma (all OGBench actors are N(mu, (1*temperature)^2 I)).
    Each candidate is imagined for k steps in the ensemble WM (first step =
    the candidate action, then the policy mean closed-loop per member), scored
    with the SAME value head / horizon aggregation as RolloutRanker, and the
    winner's first action is executed in the real env; for c > 1 the policy
    mean is executed closed-loop for the remaining c-1 steps (matching the
    imagined branch). Budget per env step is N*E*k/c.

    Exact score ties go to the lowest index, i.e. the unperturbed policy —
    the baseline can only leave the fixed policy when the model prefers it.

    Reproducibility: the noise stream is seeded per episode from `seed` and
    the episode's first (observation, goal), like RandomArbiter.
    """

    def __init__(self, wm, policy, policy_name, counter, n_samples, sigma,
                 horizon, replan_every=1, score_agg='max', seed=0, candidates='gauss', ens_agg='mean',
                 critic_fn=None):
        assert n_samples >= 1 and horizon >= 1 and replan_every >= 1
        # critic_fn(states, goals) -> (N,): score imagined states with a bank member's own value
        # (the paper's direct value) instead of the world model's metric head.
        self.critic_fn = critic_fn
        assert critic_fn is not None or wm.config['value_head'], 'PolicyMPC scores with the value head'
        assert score_agg in ('max', 'mean', 'last'), score_agg
        assert ens_agg in ENS_AGGS, ens_agg
        self.ens_agg = ens_agg
        # candidates='gauss': candidate 0 = policy mean, others = clip(mu + sigma*eps).
        # candidates='samples': the policy proposes its own candidates via
        # policy.sample_candidates(ob, goal, n, rng) (candidate 0 = its
        # deterministic decode, others = draws from its action distribution).
        # Scoring, commit and budget are identical in both modes.
        assert candidates in ('gauss', 'samples'), candidates
        if candidates == 'samples':
            assert hasattr(policy, 'sample_candidates'), 'policy must implement sample_candidates(ob, goal, n, rng)'
        self.candidates = candidates
        self.wm = wm
        self.policy = policy
        self.policy_name = policy_name
        self.counter = counter
        self.n_samples = int(n_samples)
        self.sigma = float(sigma)
        self.horizon = int(horizon)
        self.replan_every = int(replan_every)
        self.score_agg = score_agg
        self.seed = int(seed)
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_replan = None
        self._committed = None  # first action of the winning candidate
        self._rng = None
        self._log = _EpisodeLog()
        self._nonmean_picks = 0
        self._winner_idx = []

    def episode_info(self):
        info = self._log.info()
        info['mpc_nonmean_picks'] = self._nonmean_picks
        info['mpc_mean_winner_idx'] = float(np.mean(self._winner_idx)) if self._winner_idx else 0.0
        return info

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def _agg_horizon(self, scores_hn):
        if self.score_agg == 'mean':
            return scores_hn.mean(axis=0)
        if self.score_agg == 'last':
            return scores_hn[-1]
        return scores_hn.max(axis=0)

    def _replan(self, ob, goal):
        from world_model.rollout import imagine_mpc_rollout

        t0 = time.perf_counter()
        if self.candidates == 'samples':
            cands = np.clip(np.asarray(self.policy.sample_candidates(ob, goal, self.n_samples, self._rng),
                                       dtype=np.float32), -1, 1)  # (N, da), candidate 0 = deterministic decode
            assert cands.shape[0] == self.n_samples, cands.shape
        else:
            mean = np.clip(np.asarray(self.policy.act(ob[None], goal[None]))[0], -1, 1)  # (da,)
            eps = self._rng.standard_normal((self.n_samples - 1, mean.shape[-1])).astype(np.float32)
            cands = np.concatenate([mean[None], np.clip(mean[None] + self.sigma * eps, -1, 1)], axis=0)  # (N, da)
        roll = imagine_mpc_rollout(self.wm, self.policy, ob, goal, cands, self.horizon, self.counter)
        traj = roll['obs_traj'][1:]  # (H, E, N, d)
        H, E, N = traj.shape[:3]
        self._log.dyn_calls += H * E * N
        if self.critic_fn is not None:
            flat = traj.reshape(H * E * N, -1)
            vals = np.asarray(self.critic_fn(flat, np.broadcast_to(goal, flat.shape[:-1] + goal.shape[-1:])))
            vals = np.moveaxis(vals.reshape(H, E, N), 1, 0)  # (E, H, N)
        else:
            traj_e = np.moveaxis(traj, 1, 0).reshape(E, H * N, -1)
            goals_e = np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])
            vals = np.asarray(self.wm.value_score(traj_e, goals_e)).reshape(E, H, N)
        self._log.val_calls += H * E * N
        scores = self._agg_horizon(_ens_reduce(vals, self.ens_agg))  # (N,)
        self.counter.mark_decision()
        winner = _lexi_argmax(scores, np.arange(N, 0, -1.0))
        self._committed = cands[winner]
        self._nonmean_picks += int(winner != 0)
        self._winner_idx.append(winner)
        self._steps_since_replan = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if self._rng is None:
            self._seed_rng(ob, goal)
        if self._steps_since_replan is None or self._steps_since_replan >= self.replan_every:
            self._replan(ob, goal)
        first = self._steps_since_replan == 0
        self._steps_since_replan += 1
        self._log.record_step(self.policy_name)
        if first:
            return self._committed
        # Remaining commit steps: the policy mean closed-loop on the REAL observation.
        return np.clip(np.asarray(self.policy.act(ob[None], goal[None]))[0], -1, 1)


class LeastUsedArbiter:
    """Deterministic round-robin control: every `commit` env steps switch to the
    policy with the least usage so far in the episode (ties in sorted-name
    order). No model, no randomness — isolates the effect of *switching* from
    the effect of *random* switching (RandomArbiter) at the same interval.
    """

    def __init__(self, bank, commit, order='name', seed=0):
        assert commit >= 1, commit
        assert order in ('name', 'reverse', 'shuffle'), order
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.commit = int(commit)
        # Cycle order: sorted names, reversed names, or a fresh random
        # permutation for every round (seeded per episode like RandomArbiter)
        # — the latter isolates "systematic coverage" from "lucky order".
        self.order = order
        self.seed = int(seed)
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_draw = None
        self._current = None
        self._usage = {n: 0 for n in self.names}
        self._rng = None
        self._round = []
        self._log = _EpisodeLog()

    def episode_info(self):
        return self._log.info()

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def _draw(self):
        t0 = time.perf_counter()
        if self.order == 'shuffle':
            if not self._round:
                self._round = [self.names[i] for i in self._rng.permutation(len(self.names))]
            self._current = self._round.pop(0)
        else:
            least = min(self._usage.values())
            names = self.names if self.order == 'name' else self.names[::-1]
            self._current = next(n for n in names if self._usage[n] == least)
        self._steps_since_draw = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        if self._rng is None:
            self._seed_rng(ob, goal)
        if self._steps_since_draw is None or self._steps_since_draw >= self.commit:
            self._draw()
        self._steps_since_draw += 1
        self._usage[self._current] += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class NoisyPolicyArbiter:
    """No-model control for PolicyMPC: one fixed policy whose action is
    perturbed, a = clip(mu + sigma*eps), every `every` env steps (starting at
    the first step) and left at the mean otherwise — i.e. exactly PolicyMPC's
    candidate distribution with a UNIFORMLY RANDOM pick instead of the
    value-head argmax. Zero model calls. Noise stream seeded per episode like
    RandomArbiter.
    """

    def __init__(self, policy, policy_name, sigma, every, seed=0):
        assert every >= 1, every
        self.policy = policy
        self.policy_name = policy_name
        self.sigma = float(sigma)
        self.every = int(every)
        self.seed = int(seed)
        self.reset_episode()

    def reset_episode(self):
        self._t = 0
        self._rng = None
        self._log = _EpisodeLog()

    def episode_info(self):
        return self._log.info()

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def act(self, ob, goal, temperature=0.0):
        if self._rng is None:
            self._seed_rng(ob, goal)
        mean = np.clip(np.asarray(self.policy.act(ob, goal)), -1, 1)
        if self._t % self.every == 0:
            t0 = time.perf_counter()
            mean = np.clip(mean + self.sigma * self._rng.standard_normal(mean.shape).astype(np.float32), -1, 1)
            self._log.n_plans += 1
            self._log.plan_seconds += time.perf_counter() - t0
        self._t += 1
        self._log.record_step(self.policy_name)
        return mean


class PortfolioMPC:
    """Sampling MPC over the WHOLE bank: candidates = every policy's mean
    action plus n_per-1 Gaussian perturbations of it (P*n_per branches);
    each branch is imagined k steps (first step = candidate action, then its
    own policy's mean closed-loop per member), scored with the value head like
    RolloutRanker/PolicyMPC; the winner's first action is executed and its
    policy then runs closed-loop for the remaining c-1 steps. Budget per env
    step P*n_per*E*k/c. Ties -> lowest index (policy order, mean first).
    """

    def __init__(self, wm, bank, counter, n_per, sigma, horizon, replan_every=1,
                 score_agg='max', seed=0, ens_agg='mean'):
        assert n_per >= 1 and horizon >= 1 and replan_every >= 1
        assert wm.config['value_head'], 'PortfolioMPC scores with the value head'
        assert score_agg in ('max', 'mean', 'last'), score_agg
        assert ens_agg in ENS_AGGS, ens_agg
        self.ens_agg = ens_agg
        self.wm = wm
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.counter = counter
        self.n_per = int(n_per)
        self.sigma = float(sigma)
        self.horizon = int(horizon)
        self.replan_every = int(replan_every)
        self.score_agg = score_agg
        self.seed = int(seed)
        self.reset_episode()

    def reset_episode(self):
        self._steps_since_replan = None
        self._current = None
        self._committed = None
        self._rng = None
        self._log = _EpisodeLog()
        self._nonmean_picks = 0

    def episode_info(self):
        info = self._log.info()
        info['mpc_nonmean_picks'] = self._nonmean_picks
        return info

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def _agg_horizon(self, scores_hn):
        if self.score_agg == 'mean':
            return scores_hn.mean(axis=0)
        if self.score_agg == 'last':
            return scores_hn[-1]
        return scores_hn.max(axis=0)

    def _replan(self, ob, goal):
        from world_model.rollout import imagine_portfolio_mpc_rollout

        t0 = time.perf_counter()
        policies = [self.bank[n] for n in self.names]
        means = np.stack([np.clip(np.asarray(p.act(ob[None], goal[None]))[0], -1, 1) for p in policies])  # (P, da)
        P, da = means.shape
        eps = self._rng.standard_normal((P, self.n_per - 1, da)).astype(np.float32)
        cands = np.concatenate([means[:, None], np.clip(means[:, None] + self.sigma * eps, -1, 1)], axis=1)  # (P, N, da)
        roll = imagine_portfolio_mpc_rollout(self.wm, policies, ob, goal, cands, self.horizon, self.counter)
        traj = roll['obs_traj'][1:]  # (H, E, P*N, d)
        H, E, B = traj.shape[:3]
        self._log.dyn_calls += H * E * B
        traj_e = np.moveaxis(traj, 1, 0).reshape(E, H * B, -1)
        goals_e = np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])
        vals = np.asarray(self.wm.value_score(traj_e, goals_e))  # (E, H*B)
        self._log.val_calls += H * E * B
        scores = self._agg_horizon(_ens_reduce(vals.reshape(E, H, B), self.ens_agg))  # (B,)
        self.counter.mark_decision()
        winner = _lexi_argmax(scores, np.arange(B, 0, -1.0))
        i, j = divmod(winner, self.n_per)
        self._current = self.names[i]
        self._committed = cands[i, j]
        self._nonmean_picks += int(j != 0)
        self._steps_since_replan = 0
        self._log.n_plans += 1
        self._log.plan_seconds += time.perf_counter() - t0

    def act(self, ob, goal, temperature=0.0):
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if self._rng is None:
            self._seed_rng(ob, goal)
        if self._steps_since_replan is None or self._steps_since_replan >= self.replan_every:
            self._replan(ob, goal)
        first = self._steps_since_replan == 0
        self._steps_since_replan += 1
        self._log.record_step(self._current)
        if first:
            return self._committed
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)


class StallRestartArbiter:
    """No-model restart heuristic (reviewer-requested control): keep the current
    policy while the state keeps moving; when the normalised displacement over
    the last `window` env steps falls below `eps`, switch to a uniformly random
    OTHER bank policy (restart). Displacement is measured in the world model's
    normalised observation space (per-dimension std of the dataset), so `eps`
    is dimensionless. Starts with a uniform draw; zero model / value calls;
    RNG seeded per episode like RandomArbiter.
    """

    def __init__(self, bank, obs_mean, obs_std, window, eps=0.05, seed=0):
        assert window >= 1 and eps >= 0
        self.bank = dict(sorted(bank.items()))
        self.names = list(self.bank)
        self.obs_mean = np.asarray(obs_mean, np.float32)
        self.obs_std = np.asarray(obs_std, np.float32)
        self.window = int(window)
        self.eps = float(eps)
        self.seed = int(seed)
        self.reset_episode()

    def reset_episode(self):
        self._current = None
        self._rng = None
        self._hist = []
        self._steps_since_switch = 0
        self._n_restarts = 0
        self._log = _EpisodeLog()

    def episode_info(self):
        info = self._log.info()
        info['n_restarts'] = self._n_restarts
        return info

    def _seed_rng(self, ob, goal):
        import hashlib

        h = hashlib.sha256()
        h.update(np.int64(self.seed).tobytes())
        h.update(np.asarray(ob, dtype=np.float32).tobytes())
        h.update(np.asarray(goal, dtype=np.float32).tobytes())
        self._rng = np.random.default_rng(int.from_bytes(h.digest()[:8], 'little'))

    def _stalled(self, z):
        if len(self._hist) < self.window or self._steps_since_switch < self.window:
            return False
        return float(np.linalg.norm(z - self._hist[-self.window])) < self.eps

    def act(self, ob, goal, temperature=0.0):
        ob = np.asarray(ob, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if self._rng is None:
            self._seed_rng(ob, goal)
            self._current = self.names[int(self._rng.integers(len(self.names)))]
        z = (ob - self.obs_mean) / self.obs_std
        if self._stalled(z):
            t0 = time.perf_counter()
            others = [n for n in self.names if n != self._current]
            self._current = others[int(self._rng.integers(len(others)))]
            self._steps_since_switch = 0
            self._n_restarts += 1
            self._log.n_plans += 1
            self._log.plan_seconds += time.perf_counter() - t0
        self._hist.append(z)
        self._steps_since_switch += 1
        self._log.record_step(self._current)
        return np.clip(np.asarray(self.bank[self._current].act(ob, goal)), -1, 1)
