"""Generate the P2 main-bank manifest from the official impls/hyperparameters.sh.

8 main-tier datasets x 5 bank algorithms x 3 seeds = 120 rows. Flags are taken
verbatim from the official per-env commands (alphas, discounts, subgoal_steps,
stitch goal-sampling), so the bank reproduces the benchmark's own settings.
"""

import csv
import os
import re

HP = '/project/6067317/jwquan/ogbench/impls/hyperparameters.sh'
OUT = os.path.join(os.path.dirname(__file__), 'p2_manifest.csv')

DATASETS = [
    'antmaze-giant-navigate-v0',
    'antmaze-giant-stitch-v0',
    'humanoidmaze-giant-navigate-v0',
    'cube-triple-play-v0',
    'cube-triple-noisy-v0',
    'cube-quadruple-play-v0',
    'scene-play-v0',
    'puzzle-4x4-play-v0',
]
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl']
SEEDS = [0, 1, 2]

commands = {}
with open(HP) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'python main\.py --env_name=(\S+) (.*)', line)
        if not m:
            continue
        env, rest = m.groups()
        am = re.search(r'--agent=agents/(\w+)\.py', rest)
        if not am:
            continue
        # Keep only agent flags; harness flags (eval_episodes etc.) are set by
        # the sbatch template.
        flags = ' '.join(tok for tok in rest.split() if tok.startswith('--agent'))
        commands[(env, am.group(1))] = flags

rows, missing = [], []
idx = 0
for env in DATASETS:
    for algo in ALGOS:
        key = (env, algo)
        if key not in commands:
            missing.append(key)
            continue
        for seed in SEEDS:
            idx += 1
            rows.append([idx, env, algo, seed, commands[key]])

with open(OUT, 'w', newline='') as f:
    # csv.writer's default lineterminator is \r\n; a trailing \r on the flags
    # column corrupts --agent=<path> rows (path becomes "gcbc.py\r").
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['idx', 'env_name', 'algo', 'seed', 'flags'])
    w.writerows(rows)

print(f'{len(rows)} rows -> {OUT}')
for key in missing:
    print('MISSING in hyperparameters.sh:', key)
