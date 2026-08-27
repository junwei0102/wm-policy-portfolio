"""P0 hard gate: bitwise snapshot -> replay determinism on all main-tier envs.

Protocol per env:
  reset(task_id=1) -> warm up W steps -> snapshot -> roll N fixed actions
  recording observations (branch A) -> restore -> roll the same N actions
  (branch B) -> assert every observation array is bitwise identical.

Run standalone (prints a table) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium
import numpy as np
import ogbench  # noqa: F401  (registers env IDs)

from interfaces.sim_branch import SimBranch

ENV_IDS = [
    'antmaze-large-v0',
    'antmaze-giant-v0',
    'humanoidmaze-giant-v0',
    'cube-double-v0',
    'cube-triple-v0',
    'cube-quadruple-v0',
    'scene-v0',
    'puzzle-4x4-v0',
]

WARMUP_STEPS = 50
BRANCH_STEPS = 100


def _rollout(env, actions):
    obs_seq = []
    for a in actions:
        ob, _, terminated, truncated, _ = env.step(a)
        obs_seq.append(np.asarray(ob).copy())
        if terminated or truncated:
            break
    return obs_seq


def check_env(env_id, seed=0):
    env = gymnasium.make(env_id, terminate_at_goal=False)
    env.reset(seed=seed, options=dict(task_id=1))
    rng = np.random.default_rng(seed)
    act_dim = env.action_space.shape[0]
    warmup = rng.uniform(-1, 1, size=(WARMUP_STEPS, act_dim))
    branch = rng.uniform(-1, 1, size=(BRANCH_STEPS, act_dim))

    _rollout(env, warmup)
    sim = SimBranch(env)
    snap = sim.snapshot()
    obs_a = _rollout(env, branch)
    sim.restore(snap)
    obs_b = _rollout(env, branch)
    env.close()

    if len(obs_a) != len(obs_b):
        return False, f'length mismatch {len(obs_a)} vs {len(obs_b)}'
    for t, (a, b) in enumerate(zip(obs_a, obs_b)):
        if not np.array_equal(a, b):
            return False, f'first divergence at step {t}, max|diff|={np.max(np.abs(a - b)):.3e}'
    return True, f'{len(obs_a)} steps bitwise identical'


def test_snapshot_determinism():
    failures = []
    for env_id in ENV_IDS:
        ok, msg = check_env(env_id)
        if not ok:
            failures.append(f'{env_id}: {msg}')
    assert not failures, '\n'.join(failures)


if __name__ == '__main__':
    all_ok = True
    for env_id in ENV_IDS:
        ok, msg = check_env(env_id)
        all_ok &= ok
        print(f'{"PASS" if ok else "FAIL"}  {env_id:28s} {msg}')
    sys.exit(0 if all_ok else 1)
