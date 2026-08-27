"""Imagination API + unified transition-call accounting.

`wm_step` is the ONLY sanctioned entry point for imagined stepping in planner
code: every call increments the TransitionCounter by E*B (one query per
ensemble member per batch row). Diagnostics use their own counter (reported,
never budget-constrained — documented exemption).
"""

import numpy as np


class TransitionCounter:
    def __init__(self):
        self.total = 0
        self.per_decision = []
        self._last_mark = 0

    def add(self, n):
        self.total += int(n)

    def mark_decision(self):
        cost = self.total - self._last_mark
        self.per_decision.append(cost)
        self._last_mark = self.total
        return cost


def wm_step(wm, obs_e, actions_e, counter):
    """(E, B, d) x (E, B, da) -> (E, B, d). Counts E*B transitions."""
    E, B = obs_e.shape[0], obs_e.shape[1]
    counter.add(E * B)
    return np.asarray(wm.predict_next(obs_e, actions_e))


def imagine_policy_rollout(wm, policy, obs0, goal, horizon, counter):
    """Roll `policy` closed-loop in the world model from a single real state.

    Each ensemble member evolves its own trajectory; the policy is re-queried
    on every member's imagined observation (batched over members).

    Returns dict with:
      obs_traj (H+1, E, d), actions (H, E, da)
    """
    E = wm.config['num_members']
    obs0 = np.asarray(obs0, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    goals_e = np.tile(goal, (E, 1))

    x = np.tile(obs0, (E, 1))  # (E, d)
    obs_traj, acts = [x.copy()], []
    for _ in range(horizon):
        actions = np.clip(np.asarray(policy.act(x, goals_e)), -1, 1)  # (E, da)
        x = wm_step(wm, x[:, None, :], actions[:, None, :], counter)[:, 0, :]
        obs_traj.append(x.copy())
        acts.append(actions)
    return dict(obs_traj=np.stack(obs_traj), actions=np.stack(acts))


def imagine_bank_rollout(wm, policies, obs0, goal, horizon, counter):
    """Roll EVERY bank policy closed-loop simultaneously, batching the WM step.

    policies: ordered list of FrozenPolicy. State batch is (E, P, d): each
    ensemble member evolves its own trajectory per policy; each policy is
    re-queried on its own imagined observations (one E-batched act call per
    policy per step), then all P candidate branches advance in a single
    wm_step of batch (E, P) — P times fewer WM dispatches than rolling
    policies one at a time, identical transition accounting.

    Returns dict(obs_traj (H+1, E, P, d)).
    """
    E = wm.config['num_members']
    P = len(policies)
    obs0 = np.asarray(obs0, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    goals_e = np.tile(goal, (E, 1))

    x = np.broadcast_to(obs0, (E, P, obs0.shape[-1])).copy()
    obs_traj = [x.copy()]
    for _ in range(horizon):
        actions = np.stack(
            [np.clip(np.asarray(p.act(x[:, i], goals_e)), -1, 1) for i, p in enumerate(policies)],
            axis=1,
        )  # (E, P, da)
        x = wm_step(wm, x, actions, counter)
        obs_traj.append(x.copy())
    return dict(obs_traj=np.stack(obs_traj))


def open_loop_error(wm, obs0, action_seq, real_obs_seq):
    """Diagnostics: replay real actions open-loop, compare against real states.

    obs0 (d,), action_seq (T, da), real_obs_seq (T+1, d) with real_obs_seq[0]=obs0.
    Returns per-step (T,) arrays: normalized MSE of the ensemble mean, and
    ensemble disagreement (mean per-dim std across members, normalized space).
    """
    traj = np.asarray(wm.imagine_openloop(obs0[None], action_seq[None]))  # (E, 1, T+1, d)
    traj = traj[:, 0]  # (E, T+1, d)
    obs_std = np.asarray(wm.normalizer['obs_std'])
    norm_traj = traj / obs_std
    norm_real = np.asarray(real_obs_seq) / obs_std
    mean_pred = norm_traj.mean(axis=0)  # (T+1, d)
    mse = ((mean_pred[1:] - norm_real[1:]) ** 2).mean(axis=-1)  # (T,)
    disagreement = norm_traj.std(axis=0).mean(axis=-1)[1:]  # (T,)
    return mse, disagreement


def imagine_mpc_rollout(wm, policy, obs0, goal, first_actions, horizon, counter):
    """Sampling-MPC branches: N candidate FIRST actions, then the policy mean.

    Branch i executes `first_actions[i]` at imagined step 0 and follows
    `policy` (its mean, temperature 0) closed-loop on its own imagined
    observations for the remaining horizon-1 steps — one E*N-batched act call
    per step. State batch is (E, N, d): each ensemble member evolves its own
    trajectory per candidate. Transition accounting is identical to
    `imagine_bank_rollout` with P -> N.

    Returns dict(obs_traj (H+1, E, N, d)).
    """
    E = wm.config['num_members']
    obs0 = np.asarray(obs0, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    first_actions = np.asarray(first_actions, dtype=np.float32)  # (N, da)
    N = first_actions.shape[0]
    goals_flat = np.broadcast_to(goal, (E * N, goal.shape[-1]))

    x = np.broadcast_to(obs0, (E, N, obs0.shape[-1])).copy()
    obs_traj = [x.copy()]
    for t in range(horizon):
        if t == 0:
            actions = np.broadcast_to(first_actions, (E, N, first_actions.shape[-1]))
        else:
            flat = policy.act(x.reshape(E * N, -1), goals_flat)
            actions = np.clip(np.asarray(flat), -1, 1).reshape(E, N, -1)
        x = wm_step(wm, x, actions, counter)
        obs_traj.append(x.copy())
    return dict(obs_traj=np.stack(obs_traj))


def imagine_portfolio_mpc_rollout(wm, policies, obs0, goal, first_actions, horizon, counter):
    """Portfolio-MPC branches: for every policy i, N candidate FIRST actions,
    then policy i's mean closed-loop. `first_actions` is (P, N, da); branch
    (i, j) executes first_actions[i, j] at step 0 and follows policies[i] on
    its own imagined observations afterwards (one E*N-batched act call per
    policy per step). State batch (E, P*N, d); accounting = bank rollout with
    P -> P*N. Returns dict(obs_traj (H+1, E, P*N, d))."""
    E = wm.config['num_members']
    obs0 = np.asarray(obs0, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    first_actions = np.asarray(first_actions, dtype=np.float32)
    P, N, da = first_actions.shape
    goals_flat = np.broadcast_to(goal, (E * N, goal.shape[-1]))

    x = np.broadcast_to(obs0, (E, P * N, obs0.shape[-1])).copy()
    obs_traj = [x.copy()]
    for t in range(horizon):
        if t == 0:
            actions = np.broadcast_to(first_actions.reshape(1, P * N, da), (E, P * N, da))
        else:
            acts = []
            for i, p in enumerate(policies):
                flat = p.act(x[:, i * N:(i + 1) * N].reshape(E * N, -1), goals_flat)
                acts.append(np.clip(np.asarray(flat), -1, 1).reshape(E, N, da))
            actions = np.concatenate(acts, axis=1)  # (E, P*N, da)
        x = wm_step(wm, x, actions, counter)
        obs_traj.append(x.copy())
    return dict(obs_traj=np.stack(obs_traj))
