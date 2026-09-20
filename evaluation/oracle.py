"""Dynamic-oracle machinery: decision-state harvest + real-simulator branches.

From states visited by a base (best-fixed) policy, branch every bank policy in
the REAL simulator to episode end. Snapshots are only valid within one reset's
compiled model (manipspace recompiles per reset), so each episode's states are
branched immediately after the episode, before the next reset.

Branches step `env.unwrapped` directly (the TimeLimit wrapper's counter is not
part of the snapshot); the remaining budget `episode_len - t` is enforced
manually, preserving the official episode budget.
"""

import numpy as np

from evaluation.paired_eval import _ensure_seedable_action_space, episode_seed
from interfaces.sim_branch import SimBranch


def _seed_env(env, seed):
    _ensure_seedable_action_space()
    np.random.seed(seed)
    env.unwrapped.np_random = np.random.default_rng(seed)
    env.action_space.seed(int(seed))


def collect_and_branch_episode(
    env,
    env_name,
    task_id,
    episode_idx,
    base_policy_name,
    bank,
    episode_len,
    snapshot_every,
    states_per_ep,
    save_horizon,
):
    """One base episode: harvest snapshots, then branch every policy from each.

    Returns (state_rows, branch_rows): lists of dicts.
    """
    seed = episode_seed(env_name, f'task{task_id}', episode_idx)
    _seed_env(env, seed)
    # The trajectory generator is either a bank member (by name) or any controller
    # exposing act(ob, goal) — e.g. the WMPA arbiter or Random-Switch — so decision
    # states can be harvested from the distribution the arbiter itself induces.
    base_policy = bank[base_policy_name] if isinstance(base_policy_name, str) else base_policy_name
    if hasattr(base_policy, 'reset_episode'):
        base_policy.reset_episode()
    ob, info = env.reset(options=dict(task_id=task_id))
    goal = np.asarray(info['goal'])
    sim = SimBranch(env)

    # --- base episode with periodic snapshots ---
    snapshots = []  # (t, ob, snap)
    t = 0
    done = False
    while not done and t < episode_len:
        if t % snapshot_every == 0 and t > 0:
            snapshots.append((t, np.asarray(ob).copy(), sim.snapshot()))
        action = np.clip(base_policy.act(ob, goal), -1, 1)
        ob, _, terminated, truncated, info = env.step(action)
        t += 1
        done = terminated or truncated
    base_outcome = dict(success=float(info['success']), steps=t)

    # Evenly-spaced subsample stratifies early/mid/late automatically.
    if len(snapshots) > states_per_ep:
        keep = np.linspace(0, len(snapshots) - 1, states_per_ep).round().astype(int)
        snapshots = [snapshots[i] for i in sorted(set(keep))]

    state_rows, branch_rows = [], []
    for snap_t, snap_ob, snap in snapshots:
        state_row = dict(
            task_id=task_id,
            episode_idx=episode_idx,
            reset_seed=seed,
            t=snap_t,
            frac=snap_t / episode_len,
            obs=snap_ob,
            goal=goal,
            base_success=base_outcome['success'],
        )
        state_rows.append(state_row)
        budget = episode_len - snap_t
        for name, policy in sorted(bank.items()):
            sim.restore(snap)
            branch_rows.append(
                _roll_branch(env, policy, snap_ob, goal, budget, save_horizon)
                | dict(policy=name, task_id=task_id, episode_idx=episode_idx, t=snap_t)
            )
    return state_rows, branch_rows, base_outcome


def _roll_branch(env, policy, ob, goal, budget, save_horizon):
    u = env.unwrapped
    obs_hist = [np.asarray(ob).copy()]
    act_hist = []
    success, ever_success, steps = 0.0, 0.0, 0
    for _ in range(budget):
        action = np.clip(policy.act(ob, goal), -1, 1)
        ob, _, terminated, truncated, info = u.step(action)
        steps += 1
        success = float(info['success'])
        ever_success = max(ever_success, success)
        if len(act_hist) < save_horizon:
            act_hist.append(np.asarray(action).copy())
            obs_hist.append(np.asarray(ob).copy())
        if terminated or truncated:
            break
    return dict(
        success=success,
        ever_success=ever_success,
        steps=steps,
        branch_obs=np.stack(obs_hist),
        branch_actions=np.stack(act_hist) if act_hist else np.zeros((0, env.action_space.shape[0])),
    )


def pack_results(state_rows, branch_rows, save_horizon, obs_dim, act_dim):
    """Assemble npz-ready dicts; branch obs/actions zero-padded to save_horizon."""
    S = len(state_rows)
    states = dict(
        state_id=np.arange(S),
        task_id=np.array([r['task_id'] for r in state_rows]),
        episode_idx=np.array([r['episode_idx'] for r in state_rows]),
        reset_seed=np.array([r['reset_seed'] for r in state_rows], dtype=np.int64),
        t=np.array([r['t'] for r in state_rows]),
        frac=np.array([r['frac'] for r in state_rows], dtype=np.float32),
        obs=np.stack([r['obs'] for r in state_rows]).astype(np.float32),
        goal=np.stack([r['goal'] for r in state_rows]).astype(np.float32),
        base_success=np.array([r['base_success'] for r in state_rows], dtype=np.float32),
    )
    key_to_id = {
        (r['task_id'], r['episode_idx'], r['t']): i for i, r in enumerate(state_rows)
    }
    N = len(branch_rows)
    policies = sorted({r['policy'] for r in branch_rows})
    pol_to_idx = {p: i for i, p in enumerate(policies)}
    b_obs = np.zeros((N, save_horizon + 1, obs_dim), dtype=np.float32)
    b_act = np.zeros((N, save_horizon, act_dim), dtype=np.float32)
    b_len = np.zeros(N, dtype=np.int64)
    for i, r in enumerate(branch_rows):
        L = min(len(r['branch_actions']), save_horizon)
        b_obs[i, : L + 1] = r['branch_obs'][: L + 1]
        b_act[i, :L] = r['branch_actions'][:L]
        b_len[i] = L
    branches = dict(
        state_id=np.array([key_to_id[(r['task_id'], r['episode_idx'], r['t'])] for r in branch_rows]),
        policy_idx=np.array([pol_to_idx[r['policy']] for r in branch_rows]),
        success=np.array([r['success'] for r in branch_rows], dtype=np.float32),
        ever_success=np.array([r['ever_success'] for r in branch_rows], dtype=np.float32),
        steps=np.array([r['steps'] for r in branch_rows]),
        branch_obs=b_obs,
        branch_actions=b_act,
        branch_len=b_len,
        policies=np.array(policies),
    )
    return states, branches


def headroom(branches, best_fixed_name, policies=None, n_boot=10000, seed=0):
    """State-level oracle headroom over a fixed policy, paired bootstrap CI.

    branches: dict from a loaded branches.npz (or pack_results output).
    Returns dict(headroom, ci_lo, ci_hi, oracle_mean, best_fixed_mean,
    switch_rate).
    """
    names = [str(p) for p in branches['policies']]
    if policies is None:
        policies = names
    pol_idxs = [names.index(p) for p in policies]
    best_idx = names.index(best_fixed_name)

    state_ids = branches['state_id']
    S = int(state_ids.max()) + 1
    success = np.full((S, len(names)), np.nan)
    success[state_ids, branches['policy_idx']] = branches['success']
    sub = success[:, pol_idxs]
    valid_states = ~np.isnan(sub).any(axis=1)
    sub = sub[valid_states]
    best_col = success[valid_states, best_idx]

    per_state_max = sub.max(axis=1)
    deltas = per_state_max - best_col
    rng = np.random.default_rng(seed)
    boot = np.array(
        [deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)]
    )
    winners = sub.argmax(axis=1)
    best_pos = pol_idxs.index(best_idx) if best_idx in pol_idxs else -1
    return dict(
        n_states=int(len(deltas)),
        oracle_mean=float(per_state_max.mean()),
        best_fixed_mean=float(best_col.mean()),
        headroom=float(deltas.mean()),
        ci_lo=float(np.percentile(boot, 2.5)),
        ci_hi=float(np.percentile(boot, 97.5)),
        switch_rate=float((winners != best_pos).mean()) if best_pos >= 0 else np.nan,
    )
