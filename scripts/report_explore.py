"""Report scheduled-exploration (WMPP + least-used) and LeastUsed round-robin
variants against the plain WMPP cell, the best fixed policy (og50fx) and
Random-Switch at the same c — paired per episode, hierarchical bootstrap.

Usage:
  python scripts/report_explore.py --env_name=puzzle-3x3-play-v0 \
      --extra_tags=og50r1,og50k5,og50x5,og50x10,og50x50 --out=/scratch/.../explore_3x3
"""

import json
import os
import re
import sys

import numpy as np
from absl import app, flags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from report_og50 import hier_boot, load_fixed, load_seed, paired, paired_rows  # noqa: E402

FLAGS = flags.FLAGS
# eval_root/seeds/extra_tags/fixed_tag/n_boot/out are inherited from report_og50's flag definitions.
flags.DEFINE_string('env_name', None, 'Dataset.', required=True)


def fmt_ci(c):
    return f"{100 * c['delta']:+.1f} [{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}]{'*' if c['significant'] else ''}"


def main(_):
    seeds = [int(s) for s in FLAGS.seeds.split(',')]
    env_dir = os.path.join(FLAGS.eval_root, FLAGS.env_name)
    extra = [t for t in FLAGS.extra_tags.split(',') if t]
    data = {s: load_seed(env_dir, s, 'og50', extra) for s in seeds}
    fixed = {s: load_fixed(env_dir, s, FLAGS.fixed_tag) for s in seeds}
    fams = sorted(set.intersection(*[set(fixed[s][0]) for s in seeds]))
    fixed_mean = {f: float(np.mean([np.mean([float(r['success']) for r in fixed[s][0][f]]) for s in seeds])) for f in fams}
    bp = max(fams, key=lambda f: (fixed_mean[f], f))
    spec = data[seeds[0]][0]['variants']
    rng = np.random.default_rng(0)

    def mean_succ(v):
        return float(np.mean([data[s][0]['success'][v] for s in seeds]))

    def chain(v, key):
        return float(np.mean([data[s][0]['policy_chains'][v][key] for s in seeds]))

    def explore_per_ep(v):
        return float(np.mean([np.mean([int(r.get('n_explore') or 0) for r in data[s][1][v]]) for s in seeds]))

    targets = sorted((v for v in spec if '_explore' in v or v.startswith('leastused')),
                     key=lambda v: (spec[v]['commit'], spec[v]['imagine'], v))
    out = []
    for v in targets:
        k, c = spec[v]['imagine'], spec[v]['commit']
        m = re.search(r'_explore(\d+)$', v)
        base = f'score{k}_commit{c}' if m else None
        rand = f'random_commit{c}'
        row = dict(variant=v, imagine=k, commit=c, explore_every=(int(m.group(1)) if m else None),
                   success=mean_succ(v), per_seed={str(s): data[s][0]['success'][v] for s in seeds},
                   switches_per_episode=chain(v, 'mean_switches_per_episode'),
                   explore_per_episode=explore_per_ep(v) if m else None,
                   vs_best_policy=hier_boot({s: paired_rows(data[s][1][v], fixed[s][0][bp]) for s in seeds}, FLAGS.n_boot, rng))
        if base and base in spec:
            row['plain_success'] = mean_succ(base)
            row['vs_plain'] = hier_boot({s: paired(data[s][1], v, base) for s in seeds}, FLAGS.n_boot, rng)
        if rand in spec:
            row['random_success'] = mean_succ(rand)
            row['vs_random'] = hier_boot({s: paired(data[s][1], v, rand) for s in seeds}, FLAGS.n_boot, rng)
        out.append(row)

    L = [f'## {FLAGS.env_name}: scheduled exploration / round-robin (5x50 eps x {len(seeds)} seeds; paired, hierarchical bootstrap; * = CI excludes 0)\n',
         f'Best fixed policy ({FLAGS.fixed_tag}): {bp} {100 * fixed_mean[bp]:.1f}. Plain WMPP cells: ' +
         ', '.join(f"({spec[v]['imagine']},{spec[v]['commit']}) {100 * mean_succ(v):.1f}" for v in sorted(
             (v for v in spec if re.fullmatch(r'score(\d+)_commit\1', v)), key=lambda v: spec[v]['imagine'])) +
         '. Random: ' + ', '.join(f"c={spec[v]['commit']} {100 * mean_succ(v):.1f}" for v in sorted(
             (v for v in spec if v.startswith('random_commit')), key=lambda v: spec[v]['commit'])) + '\n',
         '| variant | (k,c) | explore every | success | per seed | Δ vs best policy | Δ vs plain WMPP cell | Δ vs Random(c) | switches/ep | explorations/ep |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in out:
        plain = fmt_ci(r['vs_plain']) + ' (plain %.1f)' % (100 * r['plain_success']) if 'vs_plain' in r else '—'
        rand = fmt_ci(r['vs_random']) + ' (Random %.1f)' % (100 * r['random_success']) if 'vs_random' in r else '—'
        expl = '—' if r['explore_per_episode'] is None else '%.1f' % r['explore_per_episode']
        per_seed = '/'.join('%.1f' % (100 * v) for v in r['per_seed'].values())
        L.append('| %s | (%d,%d) | %s | %.1f | %s | %s | %s | %s | %.1f | %s |' % (
            r['variant'], r['imagine'], r['commit'], r['explore_every'] or '—', 100 * r['success'], per_seed,
            fmt_ci(r['vs_best_policy']), plain, rand, r['switches_per_episode'], expl))
    md = '\n'.join(L) + '\n'
    print(md)
    if FLAGS.out:
        with open(FLAGS.out + '.json', 'w') as f:
            json.dump(dict(env_name=FLAGS.env_name, best_policy=bp, best_policy_success=fixed_mean[bp], rows=out), f, indent=1, default=float)
        with open(FLAGS.out + '.md', 'w') as f:
            f.write(md)
        print('wrote', FLAGS.out + '.{md,json}')


if __name__ == '__main__':
    app.run(main)
