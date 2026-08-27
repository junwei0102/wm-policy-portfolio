"""Manifest for the manipulation-pilot extension: 5 datasets x 6 algos x seed 0.

Adds scene-play/noisy, puzzle-4x4-play/noisy, cube-double-noisy pilot banks
(6 algorithms including HIQL, seed 0) with official hyperparameters, mirroring
the cube-double-play pilot design. scene-play/puzzle-4x4-play 5-algo rows
duplicate upcoming P2 seed-0 runs deliberately — the P2 queue is deep and the
pilot pipeline needs banks now.
"""

import csv
import os
import re

HP = '/project/6067317/jwquan/ogbench/impls/hyperparameters.sh'
OUT = os.path.join(os.path.dirname(__file__), 'mpilot_manifest.csv')

DATASETS = [
    'cube-double-noisy-v0',
    'scene-play-v0',
    'scene-noisy-v0',
    'puzzle-4x4-play-v0',
    'puzzle-4x4-noisy-v0',
]
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
SEEDS = [0]

commands = {}
with open(HP) as f:
    for line in f:
        m = re.match(r'python main\.py --env_name=(\S+) (.*)', line.strip())
        if not m:
            continue
        env, rest = m.groups()
        am = re.search(r'--agent=agents/(\w+)\.py', rest)
        if not am:
            continue
        flags = ' '.join(tok for tok in rest.split() if tok.startswith('--agent'))
        commands[(env, am.group(1))] = flags

rows, missing = [], []
idx = 0
for env in DATASETS:
    for algo in ALGOS:
        if (env, algo) not in commands:
            missing.append((env, algo))
            continue
        for seed in SEEDS:
            idx += 1
            rows.append([idx, env, algo, seed, commands[(env, algo)]])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['idx', 'env_name', 'algo', 'seed', 'flags'])
    w.writerows(rows)

print(f'{len(rows)} rows -> {OUT}')
for key in missing:
    print('MISSING in hyperparameters.sh:', key)
