"""Oracle invariant: the base policy's own branch from a snapshot must
reproduce the base episode's continuation bitwise (same observations, same
outcome) — the paired 'continue best-fixed' branch is exact, not approximate."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium
import numpy as np
import ogbench  # noqa: F401

from evaluation.oracle import _roll_branch, _seed_env
from evaluation.paired_eval import episode_seed
from interfaces.policy_bank import load_bank
from interfaces.sim_branch import SimBranch

POLICY_ROOT = '/scratch/jwquan/wmpp/policies/p1/OGBench/p1-pilot'


def check(env_id, env_name, policy_name, snap_t=40, horizon=60):
    bank = load_bank(env_name, POLICY_ROOT, 1000000, seeds=[0])
    policy = bank[policy_name]
    env = gymnasium.make(env_id)
    seed = episode_seed(env_name, 'task1', 0)

    # Base episode: record post-snapshot continuation.
    _seed_env(env, seed)
    ob, info = env.reset(options=dict(task_id=1))
    goal = np.asarray(info['goal'])
    sim = SimBranch(env)
    snap, snap_ob = None, None
    base_obs = []
    for t in range(snap_t + horizon):
        if t == snap_t:
            snap = sim.snapshot()
            snap_ob = np.asarray(ob).copy()
        action = np.clip(policy.act(ob, goal), -1, 1)
        ob, _, terminated, truncated, info = env.step(action)
        if t >= snap_t:
            base_obs.append(np.asarray(ob).copy())
        if terminated or truncated:
            break

    # Branch: restore and roll the same policy.
    sim.restore(snap)
    branch = _roll_branch(env, policy, snap_ob, goal, len(base_obs), save_horizon=horizon)
    branch_obs = branch['branch_obs'][1 : len(base_obs) + 1]

    ok = all(np.array_equal(a, b) for a, b in zip(base_obs, branch_obs))
    env.close()
    return ok, len(base_obs)


if __name__ == '__main__':
    all_ok = True
    for env_id, env_name, pol in [
        ('antmaze-large-v0', 'antmaze-large-navigate-v0', 'hiql-sd0'),
        ('cube-double-v0', 'cube-double-play-v0', 'gciql-sd0'),
    ]:
        ok, n = check(env_id, env_name, pol)
        all_ok &= ok
        print(f'{"PASS" if ok else "FAIL"}  {env_name} ({pol}): {n} continuation steps bitwise')
    sys.exit(0 if all_ok else 1)
