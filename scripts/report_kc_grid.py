"""Sensitivity of WMPA to the imagination horizon k and the commitment c on the high-gain datasets.

Grid (k, c) with k >= c in {1, 5, 10}: a fixed k is a row, a fixed c a column, and k = c the diagonal,
so one lower-triangular block per dataset shows all three slices. Reports success (mean +- half-width of
the 95% hierarchical-bootstrap interval), policy switches per episode, and median wall-clock per
environment step, each at the dataset's own value head (metric, or the direct head on the puzzles).

Cells: the diagonal is in the main sweeps (og50 / og50k5 / og50cr), (k,1) and (1,k) in og50kc / og50kc2,
the remaining cells in og50kc3. Writes tables/kc_grid.tex and tables/numbers_kcgrid.tex.

  python scripts/report_kc_grid.py
"""
import json
import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import contrast, hier_boot, macro_key, rows_by_policy, tex_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('envs', 'cube-double-play-v0,scene-play-v0,puzzle-4x4-play-v0', 'datasets of the grid.')
flags.DEFINE_string('ks', '1,5,10', 'values of k and c (cells with k >= c are reported).')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', '')
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON (paper cell, scorer, best policy).')
flags.DEFINE_string('tags', 'og50,og50k5,og50cr,og50kc,og50kc2,og50kc3', 'dir tags searched for each cell, in order.')
flags.DEFINE_string('fixed_tag', 'og50fx', '')
flags.DEFINE_integer('n_boot', 10000, '')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'paper root (tables/ inside); "" = print only.')
flags.DEFINE_string('out_json', '/scratch/jwquan/wmpp/planner_eval/kc_grid_report.json', '')

SEEDS = (0, 1, 2)
NUM = {1: 'One', 5: 'Five', 10: 'Ten'}


def cell_rows(env, var, tags):
    out = {}
    for s in SEEDS:
        r = next((rr[var] for rr in (rows_by_policy(os.path.join(FLAGS.eval_root, env), s, t) for t in tags)
                  if var in rr and len(rr[var]) == 250), None)
        if r is None:
            return None
        out[s] = r
    return out


def summarize(rb):
    per = {s: np.array([float(r['success']) for r in v]) for s, v in rb.items()}
    d = hier_boot(per, FLAGS.n_boot, np.random.default_rng(7))
    col = lambda key, f: float(f([float(r.get(key) or 0) for v in rb.values() for r in v]))
    return dict(mean=100 * d['delta'], hw=100 * (d['ci_hi'] - d['ci_lo']) / 2, switches=col('n_switches', np.mean),
                decisions=col('n_plans', np.mean), ms=col('act_ms_per_step', np.median))


def main(_):
    rep = {e['env_name']: e for e in json.load(open(FLAGS.report))}
    ks = [int(x) for x in FLAGS.ks.split(',')]
    cells = [(k, c) for k in ks for c in ks if k >= c]
    tags = FLAGS.tags.split(',')
    rng = np.random.default_rng(20260921)  # own stream
    out, body, macros = {}, [], []
    for env in FLAGS.envs.split(','):
        r = rep[env]
        pref = 'critic' if r['selected_variant'].startswith('critic') else 'score'
        paper = (r['selected_k'], r['selected_c'])
        best = {s: rows_by_policy(os.path.join(FLAGS.eval_root, env), s, FLAGS.fixed_tag)[f"{r['best_policy']}-sd{s}"] for s in SEEDS}
        bsum = summarize(best)
        data = {kc: cell_rows(env, f'{pref}{kc[0]}_commit{kc[1]}', tags) for kc in cells}
        e = dict(best=bsum['mean'], best_hw=bsum['hw'], paper=list(paper), head='direct' if pref == 'critic' else 'metric', cells={})
        for kc, rb in data.items():
            if rb is None:
                e['cells'][f'{kc[0]},{kc[1]}'] = None
                continue
            c = summarize(rb)
            g = contrast(rb, best, FLAGS.n_boot, rng)
            c['gain'] = dict(delta=100 * g['delta'], lo=100 * g['ci_lo'], hi=100 * g['ci_hi'], significant=g['significant'])
            if kc != paper and data.get(paper) is not None:
                p = contrast(rb, data[paper], FLAGS.n_boot, rng)
                c['vs_paper'] = dict(delta=100 * p['delta'], lo=100 * p['ci_lo'], hi=100 * p['ci_hi'], significant=p['significant'])
            e['cells'][f'{kc[0]},{kc[1]}'] = c
        have = [c for c in e['cells'].values() if c]
        e['spread'] = max(c['mean'] for c in have) - min(c['mean'] for c in have)
        e['min_gain'] = min(c['gain']['delta'] for c in have)
        e['min_gain_lo'] = min(c['gain']['lo'] for c in have)
        e['complete'] = len(have) == len(cells)
        out[env] = e

        # ---- markdown
        print(f"\n## {env}  ({e['head']} head)   Best {e['best']:.1f}   paper cell {paper}   spread {e['spread']:.1f}   smallest gain {e['min_gain']:+.1f}")
        print('| (k,c) | calls/step | success | - Best [95% CI] | - paper cell [95% CI] | switches/ep | decisions/ep | ms/step |')
        print('|---|---|---|---|---|---|---|---|')
        for kc in cells:
            c = e['cells'][f'{kc[0]},{kc[1]}']
            if c is None:
                print(f'| {kc} | {18 * kc[0] / kc[1]:g} | pending | | | | | |'); continue
            f = lambda d: f"{d['delta']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" + ('*' if d['significant'] else '')
            print(f"| {kc} | {18 * kc[0] / kc[1]:g} | {c['mean']:.1f} ± {c['hw']:.1f} | {f(c['gain'])} | "
                  f"{'(paper cell)' if kc == paper else f(c['vs_paper'])} | {c['switches']:.1f} | {c['decisions']:.1f} | {c['ms']:.1f} |")

        # ---- tex block: rows = k, three column groups (success | switches | latency), columns = c
        def grp(key, fmt_):
            return lambda k, c_: ('' if c_ > k else '--' if e['cells'][f'{k},{c_}'] is None else fmt_(e['cells'][f'{k},{c_}'], (k, c_)))
        succ = grp('mean', lambda c, kc: ('\\underline{' if kc == paper else '') + f"${c['mean']:.0f} \\pm {c['hw']:.0f}$" + ('}' if kc == paper else ''))
        sw = grp('switches', lambda c, kc: f"${c['switches']:.0f}$")
        ms = grp('ms', lambda c, kc: f"${c['ms']:.1f}$")
        for i, k in enumerate(ks):
            # the best bank policy of Table 2 (same column name, same episodes) is the reference for the gain
            lead = [f"\\multirow{{{len(ks)}}}{{*}}{{{tex_env(env)}}}", f"\\multirow{{{len(ks)}}}{{*}}{{${e['best']:.0f} \\pm {e['best_hw']:.0f}$}}"] if i == 0 else ['', '']
            body.append(' & '.join(lead + [f'${k}$'] + [f_(k, c_) for f_ in (succ, sw, ms) for c_ in ks]) + ' \\\\')
        body.append('\\midrule')
        mk = macro_key(env)
        macros += [f'\\newcommand{{\\KcGridSpread{mk}}}{{{e["spread"]:.0f}}}', f'\\newcommand{{\\KcGridMinGain{mk}}}{{{e["min_gain"]:.0f}}}']
    body = body[:-1]
    n = len(ks)
    head = ('\\begin{tabular}{lcl|' + 'c' * n + '|' + 'c' * n + '|' + 'c' * n + '}\n\\toprule\n'
            f' & & & \\multicolumn{{{n}}}{{c|}}{{success rate (\\%)}} & \\multicolumn{{{n}}}{{c|}}{{switches per episode}} & \\multicolumn{{{n}}}{{c}}{{latency (ms per step)}} \\\\\n'
            'Dataset & \\bestfixed{} & $k$ & ' + ' & '.join(' & '.join(f'$c{{=}}{c}$' for c in ks) for _ in range(3)) + ' \\\\\n\\midrule\n')
    tex = head + '\n'.join(body) + '\n\\bottomrule\n\\end{tabular}\n'
    complete = all(e['complete'] for e in out.values())
    macros += [f'\\newcommand{{\\KcGridMaxSpread}}{{{max(e["spread"] for e in out.values()):.0f}}}',
               f'\\newcommand{{\\KcGridMinGain}}{{{min(e["min_gain"] for e in out.values()):.0f}}}',
               f'\\newcommand{{\\KcGridMinGainLo}}{{{min(e["min_gain_lo"] for e in out.values()):.0f}}}',
               f'\\newcommand{{\\KcGridNumCells}}{{{sum(len([c for c in e["cells"].values() if c]) for e in out.values())}}}']
    print('\n' + tex)
    print('\n'.join(macros))
    json.dump(out, open(FLAGS.out_json, 'w'), indent=1)
    if FLAGS.out_dir and complete:
        open(os.path.join(FLAGS.out_dir, 'tables', 'kc_grid.tex'), 'w').write(tex)
        open(os.path.join(FLAGS.out_dir, 'tables', 'numbers_kcgrid.tex'), 'w').write(
            '% Auto-generated by scripts/report_kc_grid.py -- do not edit.\n' + '\n'.join(macros) + '\n')
        print(f'\nwrote tables/kc_grid.tex and tables/numbers_kcgrid.tex under {FLAGS.out_dir}')
    elif not complete:
        print('\n[kc_grid] some cells are still pending: nothing written to the paper directory')


if __name__ == '__main__':
    app.run(main)
