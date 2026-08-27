"""Paired-episode evaluation harness.

Every episode is identified by the key (env_name, task_key, episode_idx). The
reset seed is a deterministic hash of that key, so every method/policy
evaluated against the same key sees the identical initial state and goal —
paired comparisons come from matching keys, never from averaging unmatched
runs. (Training seed enters through the policy checkpoint, not the key.)

Records one row per episode: success, steps, control effort, and the episode
key fields needed to join across methods.
"""

import csv
import hashlib
import os
import time

import numpy as np


def _ensure_seedable_action_space():
    """Make manipspace action_space seedable by caching it per instance.

    ManipSpaceEnv.action_space is a property that constructs a NEW Box on
    every access, so seeding it has no effect and the goal-phase stabilization
    steps (`self.action_space.sample()` inside reset) draw from a fresh
    entropy-seeded space each time — the goal observation becomes unpairable.
    The bounds are constants, so caching one Box per instance changes nothing
    semantically while making `env.action_space.seed(...)` stick.
    """
    from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

    if getattr(ManipSpaceEnv, '_paired_space_patch', False):
        return
    orig_fget = ManipSpaceEnv.action_space.fget

    def fget(self):
        if not hasattr(self, '_cached_action_space'):
            self._cached_action_space = orig_fget(self)
        return self._cached_action_space

    ManipSpaceEnv.action_space = property(fget)
    ManipSpaceEnv._paired_space_patch = True


def episode_seed(env_name, task_key, episode_idx):
    """Deterministic, process-independent seed for an episode key."""
    key = f'{env_name}|{task_key}|{episode_idx}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')


def run_episode(env, policy, task_id, seed, temperature=0.0):
    # reset(seed=...) is NOT sufficient for paired resets in OGBench: locomaze
    # draws its init/goal noise from the GLOBAL np.random (maze.py add_noise)
    # BEFORE the seed is applied inside reset, and the stabilization steps use
    # action_space.sample(), whose RNG reset() never seeds. Seed all three
    # streams up front instead.
    _ensure_seedable_action_space()
    np.random.seed(seed)
    env.unwrapped.np_random = np.random.default_rng(seed)
    env.action_space.seed(int(seed))
    if hasattr(policy, 'reset_episode'):
        policy.reset_episode()
    ob, info = env.reset(options=dict(task_id=task_id))
    goal = info['goal']
    steps, effort, success = 0, 0.0, 0.0
    act_seconds = 0.0
    done = False
    while not done:
        t0 = time.perf_counter()
        action = np.clip(policy.act(ob, goal, temperature=temperature), -1, 1)
        act_seconds += time.perf_counter() - t0
        ob, _, terminated, truncated, info = env.step(action)
        steps += 1
        effort += float(np.sum(np.square(action)))
        success = float(info['success'])
        done = terminated or truncated
    result = dict(
        success=success,
        steps=steps,
        control_effort=effort,
        act_ms_per_step=1000.0 * act_seconds / max(steps, 1),
    )
    # Planners expose per-episode decision info (policy chain, switch counts,
    # planning overhead); plain policies don't.
    if hasattr(policy, 'episode_info'):
        result.update(policy.episode_info())
    return result


def evaluate_paired(
    env,
    env_name,
    policies,
    task_ids,
    episodes_per_task,
    out_csv=None,
    temperature=0.0,
):
    """Evaluate {name: FrozenPolicy} on paired episodes; returns list of rows."""
    rows = []
    for name, policy in policies.items():
        for task_id in task_ids:
            for ep in range(episodes_per_task):
                seed = episode_seed(env_name, f'task{task_id}', ep)
                result = run_episode(env, policy, task_id, seed, temperature)
                rows.append(
                    dict(
                        env_name=env_name,
                        policy=name,
                        task_id=task_id,
                        episode_idx=ep,
                        reset_seed=seed,
                        **result,
                    )
                )
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        # Planner rows carry extra fields (chain, overhead); take the union.
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows)
    return rows


def success_matrix(rows):
    """policy x task mean-success dict from evaluate_paired rows."""
    from collections import defaultdict

    acc = defaultdict(list)
    for r in rows:
        acc[(r['policy'], r['task_id'])].append(r['success'])
    return {k: float(np.mean(v)) for k, v in acc.items()}
