"""Report-only: on every dataset, the metric-value WMPA of the paper, the direct-value WMPA that scores
imagined states with the bank's GCIQL V, and the model-free Q-select that ranks each member's proposed
action with the same agent's twin-Q critic (c=1 and c=k) -- against the best policy and against each other,
paired hierarchical bootstrap on the official-protocol episodes. Reads existing logs (Q-select: og50qsel,
og50qsel2, og50qsel3; direct value: og50cr, og50qv, og50qv2, og50qv3), uses its own rng, writes nothing
into the paper.
  python scripts/report_qsel_all.py [--out /scratch/jwquan/wmpp/planner_eval/qsel_all_report.json]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import FAMILIES, contrast, rows_by_policy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--eval_root', default='/scratch/jwquan/wmpp/planner_eval')
ap.add_argument('--out', default='/scratch/jwquan/wmpp/planner_eval/qsel_all_report.json')
args = ap.parse_args()
K = {'maze': 1, 'cube': 5, 'scene': 10, 'puzzle': 10}
QTAGS = ['og50qsel', 'og50qsel2', 'og50qsel3']
DTAGS = ['og50cr', 'og50qv', 'og50qv2', 'og50qv3']  # direct-value WMPA at the reported cell
MAIN_TAGS, SEEDS = ['og50', 'og50k5', 'og50r1', 'og50cr'], [0, 1, 2]
rng = np.random.default_rng(20260918)


def ci(x):
    return f"{x['delta']*100:+4.0f} [{x['ci_lo']*100:+3.0f},{x['ci_hi']*100:+3.0f}]{'*' if x['significant'] else ' '}"


def mean(rows):
    return 100 * np.mean([np.mean([float(r['success']) for r in rows[s]]) for s in SEEDS])


def first(env_dir, seed, tags, var):
    got = None
    for t in tags:
        got = rows_by_policy(env_dir, seed, t).get(var) or got
    return got


out = {}
print(f"{'dataset':26s} {'Best':>4s} {'WMPA':>4s} | {'Q c=1':>5s} {'vs Best':>17s} {'vs WMPA':>17s} | {'Q c=k':>5s} {'vs Best':>17s} {'vs WMPA':>17s}")
for fam, envs in FAMILIES:
    k = K[fam.lower()]
    for env in envs:
        ed = os.path.join(args.eval_root, env)
        fx = {s: rows_by_policy(ed, s, 'og50fx') for s in SEEDS}
        algs = sorted({p.rsplit('-sd', 1)[0] for p in fx[0]})
        best = max(algs, key=lambda a: np.mean([np.mean([float(r['success']) for r in fx[s][f'{a}-sd{s}']]) for s in SEEDS]))
        B = {s: fx[s][f'{best}-sd{s}'] for s in SEEDS}
        wvar = f'critic{k}_commit{k}' if env.startswith('puzzle') else f'score{k}_commit{k}'
        W = {s: first(ed, s, MAIN_TAGS, wvar) for s in SEEDS}
        # Direct-value WMPA at the same cell. On the puzzle family it IS the reported head (og50cr).
        D = {s: first(ed, s, DTAGS, f'critic{k}_commit{k}') for s in SEEDS}
        has_D = all(D[s] is not None and len(D[s]) == 250 for s in SEEDS)
        row, cells = dict(k=k, best=best, best_sr=mean(B), wmpa=mean(W),
                          direct=mean(D) if has_D else None,
                          d_direct_best=contrast(D, B, 10000, rng) if has_D else None,
                          d_direct_wmpa=contrast(D, W, 10000, rng) if has_D else None), []
        for name, c in (('c1', 1), ('ck', k)):
            Q = {s: first(ed, s, QTAGS, f'qsel_commit{c}') for s in SEEDS}
            if not all(Q[s] is not None and len(Q[s]) == 250 for s in SEEDS):
                row[name] = None
                cells.append(f"{'--':>5s} {'(incomplete)':>17s} {'':17s}")
                continue
            e = dict(mean=mean(Q), d_best=contrast(Q, B, 10000, rng), d_wmpa=contrast(Q, W, 10000, rng))
            # Same-agent contrast: direct-value WMPA minus Q-select at matched commitment.
            if has_D:
                e['d_direct_minus_q'] = contrast(D, Q, 10000, rng)
            row[name] = e
            cells.append(f"{e['mean']:5.0f} {ci(e['d_best']):>17s} {ci(e['d_wmpa']):>17s}")
        out[env] = row
        print(f"{env[:-3]:26s} {row['best_sr']:4.0f} {row['wmpa']:4.0f} | " + " | ".join(cells))
for name in ('c1', 'ck'):
    have = [e for e in out if out[e][name]]
    sig = lambda key, sign: sum(out[e][name][key]['significant'] and sign * out[e][name][key]['delta'] > 0 for e in have)
    print(f"{name}: n={len(have)}  vs Best: up {sig('d_best', 1)} / down {sig('d_best', -1)}, mean {np.mean([out[e][name]['d_best']['delta'] for e in have])*100:+.1f}"
          f" | vs WMPA: up {sig('d_wmpa', 1)} / down {sig('d_wmpa', -1)}, mean {np.mean([out[e][name]['d_wmpa']['delta'] for e in have])*100:+.1f}"
          f" | mean success: Q {np.mean([out[e][name]['mean'] for e in have]):.1f}, Best {np.mean([out[e]['best_sr'] for e in have]):.1f}, WMPA {np.mean([out[e]['wmpa'] for e in have]):.1f}")
print()
print(f"{'dataset':26s} {'Best':>4s} {'metric':>6s} {'direct':>6s} {'Q c=k':>5s} | {'direct vs metric':>17s} {'direct vs Best':>17s} {'direct - Q(c=k)':>17s}")
hd = [e for e in out if out[e]['direct'] is not None]
for e in hd:
    r = out[e]
    dq = r['ck']['d_direct_minus_q'] if r['ck'] and 'd_direct_minus_q' in r['ck'] else None
    print(f"{e[:-3]:26s} {r['best_sr']:4.0f} {r['wmpa']:6.0f} {r['direct']:6.0f} {(r['ck']['mean'] if r['ck'] else float('nan')):5.0f} | "
          f"{ci(r['d_direct_wmpa']):>17s} {ci(r['d_direct_best']):>17s} {(ci(dq) if dq else '--'):>17s}")
sig = lambda key, sign: sum(out[e][key]['significant'] and sign * out[e][key]['delta'] > 0 for e in hd)
print(f"direct value: n={len(hd)}  vs metric WMPA: up {sig('d_direct_wmpa', 1)} / down {sig('d_direct_wmpa', -1)}, mean {np.mean([out[e]['d_direct_wmpa']['delta'] for e in hd])*100:+.1f}"
      f" | vs Best: up {sig('d_direct_best', 1)} / down {sig('d_direct_best', -1)}, mean {np.mean([out[e]['d_direct_best']['delta'] for e in hd])*100:+.1f}"
      f" | mean success {np.mean([out[e]['direct'] for e in hd]):.1f}")
dq = [e for e in hd if out[e]['ck'] and 'd_direct_minus_q' in out[e]['ck']]
u = sum(out[e]['ck']['d_direct_minus_q']['significant'] and out[e]['ck']['d_direct_minus_q']['delta'] > 0 for e in dq)
d = sum(out[e]['ck']['d_direct_minus_q']['significant'] and out[e]['ck']['d_direct_minus_q']['delta'] < 0 for e in dq)
print(f"same-agent direct - Q-select(c=k): n={len(dq)}  direct better on {u}, worse on {d}, mean {np.mean([out[e]['ck']['d_direct_minus_q']['delta'] for e in dq])*100:+.1f}")
json.dump(out, open(args.out, 'w'), indent=1, default=float)
print('wrote', args.out)
