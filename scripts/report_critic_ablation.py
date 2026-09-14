"""Report the critic-scorer ablation (tag og50qv): GCIQL's own V(s, g) vs the LAVL metric head.

For every dataset with critic{k}_commit{k} rows (eval_planner --critic_scorer),
compares, on the same test episodes and bank seeds:
  * best fixed policy (og50fx)            -- 3-seed mean
  * WMPP with the LAVL head at the family k (og50, score{k}_commit{k})
  * WMPP with the GCIQL critic at every evaluated k (og50qv, critic{k}_commit{k})
and gives the paired per-episode difference critic - LAVL at the family k with a
hierarchical bootstrap (bank seeds, then episodes; percentile 95% interval).

Stdlib only (login-node safe). Usage:
  python scripts/report_critic_ablation.py [--eval_root ...] [--tag og50qv]
"""

import argparse
import csv
import json
import os
import random

SEEDS = [0, 1, 2]
KS = [1, 5, 10, 25, 50, 100]


def load_rows(path):
    if not os.path.exists(path):
        return {}
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r['policy'], []).append(r)
    return by


def episode_key(r):
    return (r.get('task_id', r.get('task')), r.get('episode', r.get('episode_idx')), r.get('reset_seed'))


def paired_delta(a_rows, b_rows):
    """Per-episode success difference a - b, aligned on (task, episode, reset_seed)."""
    b = {episode_key(r): float(r['success']) for r in b_rows}
    out = []
    for r in a_rows:
        k = episode_key(r)
        if k in b:
            out.append(float(r['success']) - b[k])
    return out


def hier_boot(per_seed, n_boot=2000, seed=1):
    """per_seed: list (seeds) of lists (episodes) of paired deltas -> (mean, lo, hi)."""
    rng = random.Random(seed)
    per_seed = [d for d in per_seed if d]
    if not per_seed:
        return None
    mean = sum(sum(d) for d in per_seed) / sum(len(d) for d in per_seed)
    boots = []
    S = len(per_seed)
    for _ in range(n_boot):
        tot, n = 0.0, 0
        for _ in range(S):
            d = per_seed[rng.randrange(S)]
            m = len(d)
            tot += sum(d[rng.randrange(m)] for _ in range(m))
            n += m
        boots.append(tot / n)
    boots.sort()
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_root', default='/scratch/jwquan/wmpp/planner_eval')
    ap.add_argument('--tag', default='og50qv')
    ap.add_argument('--cells', default='/project/6067317/jwquan/wm-policy-portfolio/manifests/selected_cells.json')
    args = ap.parse_args()
    cells = json.load(open(args.cells))

    envs = sorted(e for e in os.listdir(args.eval_root)
                  if e.endswith('-v0') and os.path.exists(f'{args.eval_root}/{e}/bank_sd0_{args.tag}/summary.json'))
    print(f'critic-scorer ablation ({args.tag}): {len(envs)} datasets\n')
    hdr = f'{"dataset":24s} {"k*":>4s} {"best":>6s} {"LAVL":>6s} ' + ' '.join(f'{"crit"+str(k):>7s}' for k in KS) + '   critic-LAVL @k* [95% CI]  scorer'
    print(hdr)
    for e in envs:
        kstar = cells.get(e, {}).get('k')
        best, lavl, crit = [], [], {k: [] for k in KS}
        deltas, scorer, complete = [], None, True
        for s in SEEDS:
            fx = f'{args.eval_root}/{e}/bank_sd{s}_og50fx/summary.json'
            sw = f'{args.eval_root}/{e}/bank_sd{s}_og50/summary.json'
            qv = f'{args.eval_root}/{e}/bank_sd{s}_{args.tag}/summary.json'
            if not all(os.path.exists(p) for p in (fx, sw, qv)):
                complete = False
                continue
            fxs = json.load(open(fx))['success']
            best.append(max(fxs.values()))
            sws = json.load(open(sw))['success']
            qvj = json.load(open(qv))
            qvs = qvj['success']
            scorer = qvj.get('critic_scorer', scorer)
            if kstar is not None and f'score{kstar}_commit{kstar}' in sws:
                lavl.append(sws[f'score{kstar}_commit{kstar}'])
            for k in KS:
                if f'critic{k}_commit{k}' in qvs:
                    crit[k].append(qvs[f'critic{k}_commit{k}'])
            a = load_rows(f'{args.eval_root}/{e}/bank_sd{s}_{args.tag}/episodes.csv').get(f'critic{kstar}_commit{kstar}', [])
            b = load_rows(f'{args.eval_root}/{e}/bank_sd{s}_og50/episodes.csv').get(f'score{kstar}_commit{kstar}', [])
            deltas.append(paired_delta(a, b))
        m = lambda v: f'{100 * sum(v) / len(v):6.1f}' if v else '    --'
        cb = hier_boot(deltas)
        ci = f'{100 * cb[0]:+6.1f} [{100 * cb[1]:+5.1f},{100 * cb[2]:+5.1f}]' + ('*' if cb and (cb[1] > 0 or cb[2] < 0) else ' ') if cb else '        --'
        star = ' (incomplete)' if not complete else ''
        print(f'{e.replace("-v0", ""):24s} {str(kstar):>4s} {m(best)} {m(lavl)} ' + ' '.join(f'{m(crit[k]):>7s}' for k in KS) + f'   {ci}  {scorer}{star}')


if __name__ == '__main__':
    main()
