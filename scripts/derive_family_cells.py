"""Derive the per-family commitment interval from the held-out episodes.

The paper tunes exactly one quantity per task family: the interval k=c, used by
WMPA as its imagination horizon and commitment and by Random-Switch as its
re-draw interval (the two are always matched -- see report_og50 --family_c).

It is selected on the held-out episodes 50-99 (tag og50val), never on the test
episodes 0-49 that every reported number uses:

    for each dataset : 3-seed mean success of score{k}_commit{k}
    for each family  : mean over that family's datasets
    k*               : argmax over k in {1,5,10,25,50,100}, ties -> smallest k

Prints the family x k table and the exact --family_k/--family_c flags for
report_og50.py. Stdlib only, so it runs on the login node.

Usage:
  python scripts/derive_family_cells.py [--eval_root ...] [--tag og50val]
"""

import argparse
import json
import os

KS = [1, 5, 10, 25, 50, 100]
SEEDS = [0, 1, 2]
FAMILIES = [
    ('maze', ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0']),
    ('cube', ['cube-single-play-v0', 'cube-single-noisy-v0', 'cube-double-play-v0', 'cube-double-noisy-v0',
              'cube-triple-play-v0', 'cube-triple-noisy-v0']),
    ('scene', ['scene-play-v0', 'scene-noisy-v0']),
    ('puzzle', ['puzzle-3x3-play-v0', 'puzzle-3x3-noisy-v0', 'puzzle-4x4-play-v0', 'puzzle-4x4-noisy-v0',
                'puzzle-4x5-play-v0', 'puzzle-4x5-noisy-v0', 'puzzle-4x6-play-v0', 'puzzle-4x6-noisy-v0']),
]


def dataset_diagonal(eval_root, env, tag, prefix='score'):
    """{k: 3-seed mean success} for {prefix}{k}_commit{k}; {} if any seed is missing."""
    per_k = {}
    for k in KS:
        vals = []
        for s in SEEDS:
            p = os.path.join(eval_root, env, f'bank_sd{s}_{tag}', 'summary.json')
            if not os.path.exists(p):
                return {}
            su = json.load(open(p)).get('success', {})
            if f'{prefix}{k}_commit{k}' not in su:
                return {}
            vals.append(su[f'{prefix}{k}_commit{k}'])
        per_k[k] = 100 * sum(vals) / len(vals)
    return per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_root', default='/path/to/planner_eval')
    ap.add_argument('--tag', default='og50val', help='held-out sweep tag (episodes 50-99)')
    ap.add_argument('--json', default=None)
    ap.add_argument('--scorer', default='', help='per-family scorer override, e.g. "puzzle=og50crval:critic": '
                    'derive that family\'s k from tag og50crval and variants critic{k}_commit{k}')
    args = ap.parse_args()
    scorer = {}
    for item in (x for x in args.scorer.split(',') if x):
        fam, spec = item.split('=', 1)
        tag, prefix = spec.split(':')
        scorer[fam] = (tag, prefix)

    chosen, incomplete = {}, []
    print(f'held-out ({args.tag}) 3-seed mean success by k, averaged over each family\n')
    print(f'{"family":8s} {"n":>3s} ' + ' '.join(f'{"k=" + str(k):>7s}' for k in KS) + '   k*')
    for fam, envs in FAMILIES:
        tag, prefix = scorer.get(fam, (args.tag, 'score'))
        if fam in scorer:
            # held-out justification of the scorer choice: LAVL family mean alongside
            lav = {e: dataset_diagonal(args.eval_root, e, args.tag) for e in envs}
            good = {e: d for e, d in lav.items() if d}
            if good:
                lm = {k: sum(d[k] for d in good.values()) / len(good) for k in KS}
                print(f'[{fam}: LAVL head, {args.tag}] ' + ' '.join(f'k={k}:{lm[k]:.1f}' for k in KS)
                      + f'  -> best {max(lm.values()):.1f}')
        per_env = {e: dataset_diagonal(args.eval_root, e, tag, prefix) for e in envs}
        missing = [e for e, d in per_env.items() if not d]
        incomplete += missing
        good = {e: d for e, d in per_env.items() if d}
        if not good:
            print(f'{fam:8s}   0  (no complete datasets)')
            continue
        fam_mean = {k: sum(d[k] for d in good.values()) / len(good) for k in KS}
        best = max(KS, key=lambda k: (fam_mean[k], -k))
        chosen[fam] = best
        print(f'{fam:8s} {len(good):3d} ' + ' '.join(f'{fam_mean[k]:7.1f}' for k in KS) + f'   {best}'
              + (f'   [{prefix} scorer, {tag}]' if fam in scorer else ''))
        for e in sorted(good):
            bk = max(KS, key=lambda k: (good[e][k], -k))
            print(f'    {e.replace("-v0", ""):28s} ' + ' '.join(f'{good[e][k]:7.1f}' for k in KS) + f'   ({bk})')
        if missing:
            print(f'    !! excluded (incomplete {args.tag}): {", ".join(m.replace("-v0", "") for m in missing)}')

    flag = ','.join(f'{f}:{chosen[f]}' for f, _ in FAMILIES if f in chosen)
    print('\n--- flags for report_og50.py (family_c is always equal to family_k) ---')
    print(f'  --select_rule=family --family_k={flag} --family_c={flag}')
    if incomplete:
        print(f'\nWARNING: {len(incomplete)} dataset(s) lacked a complete {args.tag} sweep and were '
              f'excluded from their family mean: {", ".join(sorted(set(incomplete)))}')
    if args.json:
        json.dump(dict(chosen=chosen, flag=flag, incomplete=sorted(set(incomplete))), open(args.json, 'w'), indent=1)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
