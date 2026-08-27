"""Aggregate planner evals over bank seeds (hierarchical bootstrap).

Protocol (paper-locked): one seed per algorithm per bank; replicate banks over
seeds; per-episode paired deltas are computed against THAT bank's best-fixed
policy; aggregation bootstraps bank seeds (outer cluster) then episodes within
each bank (inner), 10k resamples, 95% CI.

Usage:
  python aggregate_planner_eval.py --env_name=cube-double-play-v0 --seeds=0,1,2
"""

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'OGBench dataset name.', required=True)
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds to aggregate.')
flags.DEFINE_integer('n_boot', 10000, 'Bootstrap resamples.')
flags.DEFINE_string('dir_tag', None, 'Suffix of per-seed eval dirs (e.g. hsweep).')


def load_seed(env_dir, seed):
    tag = f'_{FLAGS.dir_tag}' if FLAGS.dir_tag else ''
    d = os.path.join(env_dir, f'bank_sd{seed}{tag}')
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    rows = list(csv.DictReader(open(os.path.join(d, 'episodes.csv'))))
    by_method = {}
    for r in rows:
        by_method.setdefault(r['policy'], []).append(r)
    return summary, by_method


def paired_deltas(by_method, variant, best_fixed):
    key = lambda r: (r['task_id'], r['episode_idx'])
    base = {key(r): float(r['success']) for r in by_method[best_fixed]}
    return np.array([float(r['success']) - base[key(r)] for r in by_method[variant]])


def main(_):
    env_dir = os.path.join(FLAGS.eval_root, FLAGS.env_name)
    seeds = [int(s) for s in FLAGS.seeds.split(',')]

    per_seed = {}
    variants = None
    for s in seeds:
        summary, by_method = load_seed(env_dir, s)
        vnames = list(summary['paired_vs_best_fixed'])
        variants = vnames if variants is None else variants
        assert set(vnames) == set(variants), (s, vnames, variants)
        per_seed[s] = {
            v: paired_deltas(by_method, v, summary['best_fixed']) for v in variants
        }

    rng = np.random.default_rng(0)
    out = dict(env_name=FLAGS.env_name, bank_seeds=seeds, n_boot=FLAGS.n_boot)
    for v in variants:
        point = float(np.mean([per_seed[s][v].mean() for s in seeds]))
        boot = np.empty(FLAGS.n_boot)
        for b in range(FLAGS.n_boot):
            picked = rng.choice(seeds, len(seeds), replace=True)
            means = [
                per_seed[s][v][rng.integers(0, len(per_seed[s][v]), len(per_seed[s][v]))].mean()
                for s in picked
            ]
            boot[b] = np.mean(means)
        out[v] = dict(
            delta=point,
            ci_lo=float(np.percentile(boot, 2.5)),
            ci_hi=float(np.percentile(boot, 97.5)),
            per_seed={str(s): float(per_seed[s][v].mean()) for s in seeds},
        )

    agg_name = f'aggregate_{FLAGS.dir_tag}.json' if FLAGS.dir_tag else 'aggregate.json'
    with open(os.path.join(env_dir, agg_name), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    app.run(main)
