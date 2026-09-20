"""Does the offline criterion pick the value head that actually wins?

Joins scripts/value_head_criterion.py outputs with the online ground truth of
scripts/report_mechanism.py (metric vs direct success at the reported cell, paired
bootstrap) and reports, per variant, on how many datasets the criterion agrees with the
better head and how many success points are lost when it does not.

  python scripts/report_value_head_criterion.py
"""
import json
import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import FAMILIES  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('crit_dir', '/scratch/jwquan/wmpp/planner_eval/value_head_criterion', 'Criterion JSONs.')
flags.DEFINE_string('mechanism', '/scratch/jwquan/wmpp/planner_eval/mechanism_report.json', 'Online ground truth.')
flags.DEFINE_string('variants', 'traj,imagined,neighbor', 'Variants to report.')


def main(_):
    mech = json.load(open(FLAGS.mechanism))
    envs = [e for _, es in FAMILIES for e in es]
    truth = {}
    for env in envs:
        a = mech.get(env, {}).get('A')
        if not a or not a.get('other_scorer'):
            continue
        # 'wmpa' is the reported head; 'other_scorer' the other one at the same cell.
        rep, oth = 100 * a['wmpa']['mean'], 100 * a['other_scorer']['mean']
        metric, direct = (rep, oth) if a['scorer'] == 'metric' else (oth, rep)
        # d_wmpa is (other - reported); sign it as (direct - metric)
        d = a['other_scorer']['d_wmpa']
        delta = 100 * d['delta'] * (1 if a['scorer'] == 'metric' else -1)
        truth[env] = dict(metric=metric, direct=direct, delta=delta, significant=bool(d['significant']))

    for variant in FLAGS.variants.split(','):
        rows, agree, ties, regret = [], 0, 0, []
        for env in envs:
            p = os.path.join(FLAGS.crit_dir, f'{env[:-3]}.json')
            if env not in truth or not os.path.exists(p):
                continue
            c = json.load(open(p))[variant]
            m = c['metric']['score']
            dsc = float(np.mean([v['score'] for k, v in c.items() if k.startswith('direct/')]))
            pick = 'direct' if dsc > m else 'metric'
            t = truth[env]
            best = 'direct' if t['delta'] > 0 else ('metric' if t['delta'] < 0 else 'tie')
            ok = best == 'tie' or pick == best
            agree += int(ok and best != 'tie'); ties += int(best == 'tie')
            lost = 0.0 if ok else abs(t['delta'])
            regret.append(lost)
            rows.append((env[:-3], m, dsc, dsc - m, pick, t['metric'], t['direct'], t['delta'], t['significant'], ok))
        n_dec = len(rows) - ties
        print(f"\n=== variant: {variant}   ({len(rows)} datasets, {ties} with no real difference)")
        print(f"{'dataset':26s} {'A_metric':>8s} {'A_direct':>8s} {'diff':>7s}  {'pick':>6s} | "
              f"{'metric':>6s} {'direct':>6s} {'truth':>7s}  agree")
        for env, m, d, df, pick, tm, td, dl, sig, ok in rows:
            mark = '' if dl == 0 else ('*' if sig else '')
            print(f"{env:26s} {m:8.3f} {d:8.3f} {df:+7.3f}  {pick:>6s} | {tm:6.1f} {td:6.1f} {dl:+6.1f}{mark:1s}  "
                  f"{'yes' if ok else 'NO'}")
        print(f"agreement: {agree}/{n_dec} datasets with a real difference"
              f"   mean regret when wrong: {np.mean([r for r in regret if r > 0]) if any(regret) else 0:.1f} points"
              f"   total points lost: {np.sum(regret):.0f}")


if __name__ == '__main__':
    app.run(main)
