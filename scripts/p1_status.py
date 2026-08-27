"""Tally P1 pilot-bank runs: env x algo x seed -> final overall success."""

import csv
import glob
import json
import os

BASE = '/scratch/jwquan/wmpp/policies/p1/OGBench/p1-pilot'

rows = []
for run_dir in sorted(glob.glob(os.path.join(BASE, '*'))):
    flags_path = os.path.join(run_dir, 'flags.json')
    eval_path = os.path.join(run_dir, 'eval.csv')
    if not os.path.exists(flags_path):
        continue
    with open(flags_path) as f:
        flags = json.load(f)
    done = bool(glob.glob(os.path.join(run_dir, f"params_{flags['train_steps']}.pkl")))
    success, step = None, None
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = list(csv.DictReader(f))
        if data:
            success = float(data[-1]['evaluation/overall_success'])
            step = int(float(data[-1]['step']))
    rows.append(
        dict(
            env=flags['env_name'],
            algo=flags['agent']['agent_name'],
            seed=flags['seed'],
            status='done' if done else f'step {step}' if step else 'starting',
            success=success,
        )
    )

rows.sort(key=lambda r: (r['env'], r['algo'], r['seed']))
print(f"{'env':30s} {'algo':7s} sd status      last_success")
for r in rows:
    s = f"{r['success']:.3f}" if r['success'] is not None else '-'
    print(f"{r['env']:30s} {r['algo']:7s} {r['seed']:2d} {r['status']:11s} {s}")
print(f'\n{sum(r["status"] == "done" for r in rows)}/36 done, {len(rows)} started')
