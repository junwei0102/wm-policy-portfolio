"""Collect the LAVL puzzle hyperparameter search.

Reads eval.csv of every run under the sweep root and prints the end-of-training
success of each cell, so the winning (high_alpha, low_alpha, discount, expectile,
smoothness) combination can be applied to the eight puzzle datasets.

  python scripts/report_lavl_sweep.py [--root /scratch/jwquan/wmpp/policies/lavl_sweep]
"""
import argparse
import csv
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/scratch/jwquan/wmpp/policies/lavl_sweep')
args = ap.parse_args()

rows = []
for flags_path in sorted(glob.glob(os.path.join(args.root, '*', 'LAVL', '*', '*', 'flags.json'))):
    run = os.path.dirname(flags_path)
    f = json.load(open(flags_path))
    a = f['agent']
    ev = os.path.join(run, 'eval.csv')
    succ, step = None, None
    if os.path.exists(ev):
        for r in csv.DictReader(open(ev)):
            if r.get('evaluation/overall_success') not in (None, ''):
                succ, step = float(r['evaluation/overall_success']), int(float(r['step']))
    rows.append(dict(env=f['env_name'], ha=a['high_alpha'], la=a['low_alpha'], g=a['discount'],
                     k=a['expectile'], sm=a['smoothness_weight'], succ=succ, step=step,
                     done=os.path.exists(os.path.join(run, 'params_1000000.pkl'))))

rows.sort(key=lambda r: (-1 if r['succ'] is None else -r['succ']))
print(f"{'high_a':>6s} {'low_a':>6s} {'gamma':>6s} {'kappa':>5s} {'smooth':>6s} {'success':>8s} {'step':>9s}  ckpt")
for r in rows:
    s = '--' if r['succ'] is None else f"{100 * r['succ']:.1f}"
    print(f"{r['ha']:6.1f} {r['la']:6.1f} {r['g']:6.3f} {r['k']:5.1f} {r['sm']:6.1f} {s:>8s} "
          f"{r['step'] if r['step'] else '--':>9}  {'yes' if r['done'] else 'no'}")
done = [r for r in rows if r['succ'] is not None]
if done:
    b = done[0]
    print(f"\nbest so far ({len(done)}/{len(rows)} cells finished): high_alpha={b['ha']}, low_alpha={b['la']}, "
          f"discount={b['g']}, expectile={b['k']}, smoothness={b['sm']}  ->  {100 * b['succ']:.1f}%")
