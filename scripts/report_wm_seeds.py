"""Auxiliary-model seed variance (reviewer W8, Q5): WMPP re-evaluated with independently
retrained world-model + value ensembles (seeds 1 and 2) on three representative datasets,
same banks, episodes, and protocol. Random-Switch has no model and is unchanged.

Outputs tables/wm_seeds.tex + macros in tables/numbers_wmseeds.tex; markdown to stdout.
"""
import json
import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import fmt, fmt_ci, hier_boot3, macro_key, paired_rows, rows_by_policy, tex_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON.')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('envs', 'auto', 'Envs with WM-seed reruns (auto = every env whose og50wm1/og50wm2 dirs exist for all bank seeds).')
flags.DEFINE_string('fixed_tag', 'og50fx', 'Fixed-policy runs (paired best-policy rows).')
flags.DEFINE_integer('n_boot', 10000, 'Bootstrap resamples.')
flags.DEFINE_string('main_tags', 'og50,og50k5,og50r1', 'Tags holding the paper (WM seed 0) runs.')
flags.DEFINE_string('wm_tags', 'og50wm1,og50wm1f,og50wm1f2;og50wm2,og50wm2f,og50wm2f2', 'Tag groups of the WM-seed reruns: one group per WM seed, tags in a group separated by commas (first tag holding the cell wins).')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'Paper root.')

SEEDS = (0, 1, 2)


def variant_means(env_dir, tags, variant):
    """Per-bank-seed mean success of `variant` over the first tag dir that has it, or None."""
    out = []
    for s in SEEDS:
        rows = None
        for t in tags:
            rows = rows_by_policy(env_dir, s, t).get(variant)
            if rows:
                break
        if not rows:
            return None
        out.append(np.mean([float(r['success']) for r in rows]))
    return np.array(out)


def main(_):
    report = {x['env_name']: x for x in json.load(open(FLAGS.report))}
    macros, lines, lines3 = {}, [], []
    rng = np.random.default_rng(0)
    if FLAGS.envs == 'auto':
        envs = [e for e in sorted(report) if all(
            any(os.path.exists(os.path.join(FLAGS.eval_root, e, f'bank_sd{s}_{t}', 'episodes.csv')) for t in grp.split(','))
            for grp in FLAGS.wm_tags.split(';') for s in SEEDS)]
    else:
        envs = FLAGS.envs.split(',')
    n_sig3 = 0
    for env in envs:
        x = report[env]
        k = x['selected_k']
        env_dir = os.path.join(FLAGS.eval_root, env)
        var = f'score{k}_commit{k}'
        per_wm = [variant_means(env_dir, FLAGS.main_tags.split(','), var)]
        for grp in FLAGS.wm_tags.split(';'):
            per_wm.append(variant_means(env_dir, grp.split(','), var))
        if any(v is None for v in per_wm):
            print(f'[wmseed] {env}: reruns pending '
                  f'({[i for i, v in enumerate(per_wm) if v is None]} missing)')
            continue
        wm_means = np.array([100 * v.mean() for v in per_wm])
        bank_std = 100 * per_wm[0].std()
        key = macro_key(env)
        macros[f'WmSeedMean{key}'] = f'{wm_means.mean():.0f}'
        macros[f'WmSeedStd{key}'] = f'{wm_means.std():.0f}'
        macros[f'WmSeedSpread{key}'] = f'{wm_means.max() - wm_means.min():.0f}'
        macros[f'BankSeedStd{key}'] = f'{bank_std:.0f}'
        best = 100 * np.array([x['best_policy_seeds'][str(s)] for s in SEEDS]).mean()
        # Crossed three-level bootstrap (WM seed x bank seed x episode) of the paired WMPP - best contrast.
        tags = [FLAGS.main_tags.split(',')] + [grp.split(',') for grp in FLAGS.wm_tags.split(';')]
        per_cell = {}
        for w, tg in enumerate(tags):
            for sd in SEEDS:
                rows_w = next((rows_by_policy(env_dir, sd, t).get(var) for t in tg if rows_by_policy(env_dir, sd, t).get(var)), None)
                rows_b = rows_by_policy(env_dir, sd, FLAGS.fixed_tag).get(f"{x['best_policy']}-sd{sd}")
                assert rows_w and rows_b, (env, w, sd)
                per_cell[(w, sd)] = paired_rows(rows_w, rows_b)
        c3 = hier_boot3(per_cell, FLAGS.n_boot, rng)
        n_sig3 += int(c3['significant'] and c3['delta'] > 0)
        macros[f'WmSeedDelta{key}'] = f"{100 * c3['delta']:+.0f}"
        macros[f'WmSeedCI{key}'] = f"[{100 * c3['ci_lo']:+.0f}, {100 * c3['ci_hi']:+.0f}]"
        macros[f'WmSeedP{key}'] = '$<10^{-4}$' if c3['p_value'] <= 1e-4 else f"{c3['p_value']:.3g}"
        d2 = x['contrasts']['WMPP_vs_best_policy']
        lines3.append(' & '.join([tex_env(env), f'$({k},{k})$', fmt_ci(d2), fmt_ci(c3)]) + ' \\\\')
        lines.append(' & '.join(
            [tex_env(env), f'$({k},{k})$', fmt(best)]
            + [fmt(m) for m in wm_means]
            + [f'{wm_means.mean():.0f} $\\pm$ {wm_means.std():.0f}', fmt(bank_std)]) + ' \\\\')
        print(f'[wmseed] {env}: WM seeds {np.round(wm_means, 1)} (std {wm_means.std():.2f}) '
              f'vs bank-seed std {bank_std:.2f}; best {best:.0f}')
    if lines:
        tdir = os.path.join(FLAGS.out_dir, 'tables')
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, 'wm_seeds.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcc|ccc|cc}\n\\toprule\n'
                    ' & & & \\multicolumn{3}{c|}{\\wmpp{} by auxiliary seed} & & bank-seed \\\\\n'
                    'Dataset & $(k,c)$ & Best & seed 0 & seed 1 & seed 2 & mean $\\pm$ std & std \\\\\n\\midrule\n'
                    + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
        with open(os.path.join(tdir, 'numbers_wmseeds.tex'), 'w') as f:
            f.write('% Auto-generated by scripts/report_wm_seeds.py -- do not edit.\n')
            for kk, vv in macros.items():
                f.write(f'\\newcommand{{\\{kk}}}{{{vv}}}\n')
        for k in ('ScenePlay', 'CubeDoublePlay', 'PuzzleFourbyFourNoisy'):
            for pre in ('WmSeedStd', 'BankSeedStd', 'WmSeedMean', 'WmSeedDelta', 'WmSeedCI'):
                macros.setdefault(f'{pre}{k}', '--')
        macros['NumSigUpWmSeed'] = str(n_sig3)
        macros['NumEnvsWmSeed'] = str(len(lines3))
        with open(os.path.join(tdir, 'wm_seeds_contrast.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcrr}\n\\toprule\n'
                    'Dataset & $(k,c)$ & $\\Delta$ vs best, bank seeds $\\times$ episodes & $\\Delta$ vs best, WM seeds $\\times$ bank seeds $\\times$ episodes \\\\\n\\midrule\n'
                    + '\n'.join(lines3) + '\n\\bottomrule\n\\end{tabular}\n')
        with open(os.path.join(tdir, 'numbers_wmseeds.tex'), 'w') as f:
            f.write('% Auto-generated by scripts/report_wm_seeds.py -- do not edit.\n')
            for kk, vv in macros.items():
                f.write(f'\\newcommand{{\\{kk}}}{{{vv}}}\n')
        print('[wmseed] wrote wm_seeds.tex, wm_seeds_contrast.tex +', len(macros), 'macros; 3-level significant gains:', n_sig3, '/', len(lines3))


if __name__ == '__main__':
    app.run(main)
