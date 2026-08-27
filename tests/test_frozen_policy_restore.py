"""Validate FrozenPolicy restore against a real completed P1 checkpoint.

Restores the run, rolls eval episodes with the official reset conventions
(task_id loop, temperature=0), and reports success — it should be in the same
range as the run's own eval.csv for the same tasks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium
import numpy as np
import ogbench  # noqa: F401

from interfaces.frozen_policy import FrozenPolicy

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else None
EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
EPISODES_PER_TASK = int(sys.argv[3]) if len(sys.argv) > 3 else 5

policy = FrozenPolicy.load(RUN_DIR, EPOCH)
print(f'restored {policy.agent_name} for {policy.env_name} (train seed {policy.train_seed})')

tokens = policy.env_name.split('-')
env_id = '-'.join(tokens[:-2] + tokens[-1:])
env = gymnasium.make(env_id)

successes = []
for task_id in range(1, len(env.unwrapped.task_infos) + 1):
    task_successes = []
    for ep in range(EPISODES_PER_TASK):
        ob, info = env.reset(seed=1000 * task_id + ep, options=dict(task_id=task_id))
        goal = info['goal']
        done = False
        success = 0.0
        while not done:
            action = np.clip(policy.act(ob, goal), -1, 1)
            ob, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']
        task_successes.append(success)
    successes.append(np.mean(task_successes))
    print(f'  task {task_id}: success {np.mean(task_successes):.2f}')
env.close()
print(f'overall: {np.mean(successes):.3f}')
