"""Mechanism report on the five gain-carrying datasets (report to the author; no paper tables).

  A. scorer x selector at the reported cell (k,k): Best | WMPA (reported scorer) | the other
     scorer at the same cell | Q-select at c=1, c=k and c=c* (c* = per-dataset held-out argmax
     over {1,5,10,25,50,100} on episodes 50-99, ties -> smallest c), with paired contrasts vs
     WMPA and vs Best; plus the full held-out / test curves of Q-select by c.
  B. k/c decoupling (metric-value families only): the 2x2 grid {1,k}^2 with success and
     dynamics/value calls per environment step, and the four paired contrasts
     (horizon at fixed c, commitment at fixed k).
  C. Table 5's best-policy MPC rows restated for the same datasets.

Rows come from episodes.csv of bank_sd<seed>_<tag>/; held-out rows (og50qselval) are only used for
the argmax and are never merged with test rows.

Usage (compute node, venv):
  python scripts/report_mechanism.py [--envs=...] [--out=/scratch/jwquan/wmpp/planner_eval/mechanism_report]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from absl import app, flags

from paper_common import KS, contrast, fmt_ci, fmt_pm_ci, macro_key, near_top, rows_by_policy, rows_hw, seed_mean, short, tex_env

FLAGS = flags.FLAGS
flags.DEFINE_string('envs', 'cube-double-play-v0,scene-play-v0,scene-noisy-v0,puzzle-3x3-play-v0,puzzle-4x4-play-v0', '')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', '')
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON (selected cell, scorer, best policy).')
flags.DEFINE_string('main_tags', 'og50,og50cr', 'sweep tags (metric diagonal; critic diagonal on puzzles).')
flags.DEFINE_string('qv_tags', 'og50qv,og50qv2', 'direct-value (critic) runs on metric-value datasets.')
flags.DEFINE_string('qsel_tags', 'og50qsel,og50qsel2', 'Q-select test runs (qsel_commit{c}).')
flags.DEFINE_string('qsel_val_tag', 'og50qselval', 'Q-select held-out runs (episodes 50-99).')
flags.DEFINE_string('kc_tag', 'og50kc', 'off-diagonal (1,k)/(k,1) runs.')
flags.DEFINE_string('fixed_tag', 'og50fx', '')
flags.DEFINE_string('mpc_tag', 'og50mpcf', '')
flags.DEFINE_string('out', '/scratch/jwquan/wmpp/planner_eval/mechanism_report', 'output stem (.md/.json).')
flags.DEFINE_string('paper_dir', '/project/6067317/jwquan/wm-policy-portfolio/WMPP_ICLR2027', 'paper root; writes tables/scorer_selector.tex + tables/numbers_mechanism.tex (empty string = skip).')
flags.DEFINE_integer('n_boot', 10000, '')

SEEDS = (0, 1, 2)


def merged(env_dir, s, tags):
    out = {}
    for t in tags:
        for k, v in rows_by_policy(env_dir, s, t).items():
            out.setdefault(k, v)
    return out


def all_seeds(fn, n_ref=None):
    """{seed: rows} if every seed has rows (and, when n_ref is given, exactly n_ref of them), else None."""
    out = {}
    for s in SEEDS:
        r = fn(s)
        if not r or (n_ref is not None and len(r) != n_ref):
            return None
        out[s] = r
    return out


def summary(env_dir, s, tag):
    p = os.path.join(env_dir, f'bank_sd{s}_{tag}', 'summary.json')
    return json.load(open(p)) if os.path.exists(p) else None


def calls(env_dir, tag, var):
    """Mean over seeds of dynamics/value calls per environment step of `var` under `tag`; None if missing."""
    d, v = [], []
    for s in SEEDS:
        j = summary(env_dir, s, tag)
        if not j or var not in j.get('model_calls', {}):
            return None
        d.append(j['model_calls'][var]['dynamics_per_env_step'])
        v.append(j['model_calls'][var]['value_per_env_step'])
    return dict(dyn=float(np.mean(d)), val=float(np.mean(v)))


def pm(rows):
    return f'{100 * seed_mean(rows):.0f} ± {rows_hw(rows):.0f}'


def main(_):
    rng = np.random.default_rng(0)
    report = {x['env_name']: x for x in json.load(open(FLAGS.report))}
    envs = FLAGS.envs.split(',')
    out, md = {}, []
    md.append('# Mechanism report (5 datasets)\n')
    md.append('Cells: mean success ± half-width of the 95% hierarchical-bootstrap interval (bank seeds, then episodes). '
              'Δ: paired per-episode contrast with its 95% interval; * = interval excludes zero. '
              'Test episodes 0–49 everywhere; held-out episodes 50–99 only for c*.\n')

    # ---------------- A. scorer x selector ----------------
    md.append('## A. Scorer × selector at the reported cell\n')
    md.append('| dataset | (k,c) | Best | WMPA (reported scorer) | other scorer @ (k,k) | Q-select c=1 | Q-select c=k | Q-select c=c* (held-out) |')
    md.append('|---|---|---|---|---|---|---|---|')
    deltas_md = ['\n**Paired contrasts (points):**\n', '| dataset | other scorer − WMPA | Qsel c=1 − WMPA | Qsel c=k − WMPA | Qsel c* − WMPA | Qsel c=1 − Best | Qsel c=k − Best | Qsel c* − Best |', '|---|---|---|---|---|---|---|---|']
    curves_md = ['\n**Q-select by c (3-seed mean success):**\n', '| dataset | split | c=1 | c=5 | c=10 | c=25 | c=50 | c=100 | c* |', '|---|---|---|---|---|---|---|---|---|']
    for env in envs:
        x = report[env]
        k = x['selected_k']
        env_dir = os.path.join(FLAGS.eval_root, env)
        scorer = (x.get('validation') or {}).get('scorer')
        pref = 'critic' if scorer == 'critic' else 'score'
        other_pref = 'score' if pref == 'critic' else 'critic'
        main_rows = {s: merged(env_dir, s, FLAGS.main_tags.split(',')) for s in SEEDS}
        W = all_seeds(lambda s: main_rows[s].get(f'{pref}{k}_commit{k}'))
        assert W, f'{env}: WMPA rows missing'
        n_ref = len(W[0])
        fx = {s: rows_by_policy(env_dir, s, FLAGS.fixed_tag) for s in SEEDS}
        B = all_seeds(lambda s: fx[s].get(f"{x['best_policy']}-sd{s}"), n_ref)
        if pref == 'critic':
            O = all_seeds(lambda s: main_rows[s].get(f'{other_pref}{k}_commit{k}'), n_ref)
        else:
            qv = {s: merged(env_dir, s, FLAGS.qv_tags.split(',')) for s in SEEDS}
            O = all_seeds(lambda s: qv[s].get(f'{other_pref}{k}_commit{k}'), n_ref)
        qs = {s: merged(env_dir, s, FLAGS.qsel_tags.split(',')) for s in SEEDS}
        qv_rows = {s: rows_by_policy(env_dir, s, FLAGS.qsel_val_tag) for s in SEEDS}
        Q = {c: all_seeds(lambda s, c=c: qs[s].get(f'qsel_commit{c}'), n_ref) for c in KS}
        QV = {c: all_seeds(lambda s, c=c: qv_rows[s].get(f'qsel_commit{c}')) for c in KS}
        for s in SEEDS:  # held-out rows must be the held-out split
            j = summary(env_dir, s, FLAGS.qsel_val_tag)
            if j is not None:
                assert list(j.get('episode_range') or []) == [50, 100], (env, s, j.get('episode_range'))
        val_mean = {c: 100 * seed_mean(QV[c]) for c in KS if QV[c]}
        test_mean = {c: 100 * seed_mean(Q[c]) for c in KS if Q[c]}
        c_star = max(KS, key=lambda c: (val_mean[c], -c)) if len(val_mean) == len(KS) else None
        e = dict(k=k, scorer='direct' if scorer == 'critic' else 'metric', best=x['best_policy'], n_episodes=n_ref,
                 wmpa=dict(mean=seed_mean(W), hw=rows_hw(W)),
                 best_fixed=dict(mean=seed_mean(B), hw=rows_hw(B)) if B else None,
                 other_scorer=dict(mean=seed_mean(O), hw=rows_hw(O), d_wmpa=contrast(O, W, FLAGS.n_boot, rng)) if O else None,
                 qsel_test_by_c=test_mean, qsel_val_by_c=val_mean, c_star=c_star, qsel={})
        for label, c in (('c1', 1), ('ck', k), ('cstar', c_star)):
            if c is None or not Q[c]:
                e['qsel'][label] = None
                continue
            e['qsel'][label] = dict(c=c, mean=seed_mean(Q[c]), hw=rows_hw(Q[c]),
                                    d_wmpa=contrast(Q[c], W, FLAGS.n_boot, rng),
                                    d_best=contrast(Q[c], B, FLAGS.n_boot, rng) if B else None)
        # same-agent comparison: direct-value WMPA (rollouts scored by the GCIQL member's V) vs Q-select (its twin-Q, no rollout)
        D = W if pref == 'critic' else O
        e['direct_wmpa'] = dict(mean=seed_mean(D), hw=rows_hw(D)) if D else None
        e['direct_vs_qsel'] = {lab: (contrast(D, Q[c], FLAGS.n_boot, rng) if (D and Q[c]) else None) for lab, c in (('c1', 1), ('ck', k))}
        out[env] = dict(A=e)
        cell = lambda d: '--' if d is None else f"{100 * d['mean']:.0f} ± {d['hw']:.0f}"
        qcell = lambda d: '--' if d is None else f"{cell(d)} (c={d['c']})"
        md.append(f"| {short(env)} | ({k},{k}) {e['scorer']} | {cell(e['best_fixed'])} | {cell(e['wmpa'])} | "
                  f"{cell(e['other_scorer'])} ({other_pref}) | {cell(e['qsel']['c1'])} | {cell(e['qsel']['ck'])} | {qcell(e['qsel']['cstar'])} |")
        dc = lambda d, key: '--' if d is None or d.get(key) is None else fmt_ci(d[key])
        deltas_md.append(f"| {short(env)} | {dc(e['other_scorer'], 'd_wmpa')} | {dc(e['qsel']['c1'], 'd_wmpa')} | {dc(e['qsel']['ck'], 'd_wmpa')} | "
                         f"{dc(e['qsel']['cstar'], 'd_wmpa')} | {dc(e['qsel']['c1'], 'd_best')} | {dc(e['qsel']['ck'], 'd_best')} | {dc(e['qsel']['cstar'], 'd_best')} |")
        for split, m in (('held-out', val_mean), ('test', test_mean)):
            curves_md.append(f"| {short(env)} | {split} | " + ' | '.join(f'{m[c]:.1f}' if c in m else '--' for c in KS)
                             + f" | {c_star if c_star else '--'} |")
    deltas_md.append('\n**Same-agent contrasts (direct-value WMPA − Q-select of the same GCIQL agent, matched commitment):**\n')
    deltas_md.append('| dataset | direct WMPA | Qsel c=1 | Qsel c=k | direct − Qsel(c=1) | direct − Qsel(c=k) |')
    deltas_md.append('|---|---|---|---|---|---|')
    for env in envs:
        e = out[env]['A']
        cell = lambda d: '--' if d is None else f"{100 * d['mean']:.0f} ± {d['hw']:.0f}"
        dc = lambda d: '--' if d is None else fmt_ci(d)
        deltas_md.append(f"| {short(env)} | {cell(e['direct_wmpa'])} | {cell(e['qsel']['c1'])} | {cell(e['qsel']['ck'])} | "
                         f"{dc(e['direct_vs_qsel']['c1'])} | {dc(e['direct_vs_qsel']['ck'])} |")
    md += deltas_md + curves_md

    # ---------------- B. k/c decoupling ----------------
    md.append('\n## B. Imagination horizon k vs commitment c (metric-value datasets)\n')
    md.append('Budget is NOT equalized across cells: dynamics calls per environment step = M·E·k/c (18·k/c). '
              'Success ± hw; calls per env step in parentheses (dynamics / value).\n')
    md.append('| dataset | (1,1) | (1,k) | (k,1) | (k,k) | horizon @ c=k: (k,k)−(1,k) | horizon @ c=1: (k,1)−(1,1) | commitment @ k: (k,k)−(k,1) | commitment @ k=1: (1,k)−(1,1) |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for env in envs:
        x = report[env]
        k = x['selected_k']
        if (x.get('validation') or {}).get('scorer') == 'critic':
            continue  # direct-value families: grid not requested
        env_dir = os.path.join(FLAGS.eval_root, env)
        main_rows = {s: merged(env_dir, s, FLAGS.main_tags.split(',')) for s in SEEDS}
        kc = {s: rows_by_policy(env_dir, s, FLAGS.kc_tag) for s in SEEDS}
        cells = {
            (1, 1): (all_seeds(lambda s: main_rows[s].get('score1_commit1')), 'og50'),
            (k, k): (all_seeds(lambda s: main_rows[s].get(f'score{k}_commit{k}')), 'og50'),
            (1, k): (all_seeds(lambda s: kc[s].get(f'score1_commit{k}')), FLAGS.kc_tag),
            (k, 1): (all_seeds(lambda s: kc[s].get(f'score{k}_commit1')), FLAGS.kc_tag),
        }
        n_ref = len(cells[(k, k)][0][0])
        g = {}
        for (kk, cc), (rows, tag) in cells.items():
            if rows is None or any(len(rows[s]) != n_ref for s in SEEDS):
                g[f'{kk},{cc}'] = None
                continue
            g[f'{kk},{cc}'] = dict(mean=seed_mean(rows), hw=rows_hw(rows), calls=calls(env_dir, tag, f'score{kk}_commit{cc}'),
                                   expected_dyn=18.0 * kk / cc)
        def con(a, b):
            ra, rb = cells[a][0], cells[b][0]
            if ra is None or rb is None or g[f'{a[0]},{a[1]}'] is None or g[f'{b[0]},{b[1]}'] is None:
                return None
            return contrast(ra, rb, FLAGS.n_boot, rng)
        cons = {'horizon_at_ck': con((k, k), (1, k)), 'horizon_at_c1': con((k, 1), (1, 1)),
                'commit_at_kk': con((k, k), (k, 1)), 'commit_at_k1': con((1, k), (1, 1))}
        out[env]['B'] = dict(k=k, grid=g, contrasts=cons)
        gc = lambda key: '--' if g.get(key) is None else (f"{100 * g[key]['mean']:.0f} ± {g[key]['hw']:.0f}" +
                                                          (f" ({g[key]['calls']['dyn']:.1f}/{g[key]['calls']['val']:.1f})" if g[key]['calls'] else ''))
        cc_ = lambda key: '--' if cons[key] is None else fmt_ci(cons[key])
        md.append(f"| {short(env)} (k={k}) | {gc('1,1')} | {gc(f'1,{k}')} | {gc(f'{k},1')} | {gc(f'{k},{k}')} | "
                  f"{cc_('horizon_at_ck')} | {cc_('horizon_at_c1')} | {cc_('commit_at_kk')} | {cc_('commit_at_k1')} |")

    # ---------------- C. MPC restated ----------------
    md.append('\n## C. Best-policy WM action search (Table 5 rows, restated)\n')
    md.append('| dataset | (k,c) | Best | WMPA | MPC on Best (N=32, σ=0.2) | MPC − WMPA |')
    md.append('|---|---|---|---|---|---|')
    for env in envs:
        x = report[env]
        k = x['selected_k']
        env_dir = os.path.join(FLAGS.eval_root, env)
        pref = 'critic' if (x.get('validation') or {}).get('scorer') == 'critic' else 'score'
        main_rows = {s: merged(env_dir, s, FLAGS.main_tags.split(',')) for s in SEEDS}
        W = all_seeds(lambda s: main_rows[s].get(f'{pref}{k}_commit{k}'))
        pat = re.compile(r'mpc\d+_s0\.2_%s%d_commit%d$' % (pref, k, k))
        M = all_seeds(lambda s: next((v for nm, v in rows_by_policy(env_dir, s, FLAGS.mpc_tag).items() if pat.fullmatch(nm)), None), len(W[0]))
        e = out[env]['A']
        m = dict(mean=seed_mean(M), hw=rows_hw(M), d_wmpa=contrast(M, W, FLAGS.n_boot, rng)) if M else None
        out[env]['C'] = m
        cell = lambda d: '--' if d is None else f"{100 * d['mean']:.0f} ± {d['hw']:.0f}"
        md.append(f"| {short(env)} | ({k},{k}) | {cell(e['best_fixed'])} | {cell(e['wmpa'])} | {cell(m)} | {'--' if m is None else fmt_ci(m['d_wmpa'])} |")

    # ---------------- paper table + macros (scorer swap and Q-select at c=1 / c=k; no c*) ----------------
    if FLAGS.paper_dir:
        tdir = os.path.join(FLAGS.paper_dir, 'tables')
        os.makedirs(tdir, exist_ok=True)
        lines, macros = [], {}
        sig_down = lambda d: d is not None and d.get('significant') and d['delta'] < 0
        sig_up = lambda d: d is not None and d.get('significant') and d['delta'] > 0
        counts = {c: dict(n=0, down=0, up=0, above_best=0, deltas=[]) for c in ('Other', 'QselOne', 'QselK', 'DirectQselOne', 'DirectQselK')}
        for env in envs:
            e = out[env]['A']
            metric = e['wmpa'] if e['scorer'] == 'metric' else e['other_scorer']
            direct = e['other_scorer'] if e['scorer'] == 'metric' else e['wmpa']
            ents = {'Other': e['other_scorer'], 'QselOne': e['qsel']['c1'], 'QselK': e['qsel']['ck']}
            top = max([100 * v['mean'] for v in [e['best_fixed'], e['wmpa'], e['other_scorer'], e['qsel']['c1'], e['qsel']['ck']] if v])
            tcell = lambda v, mark='': '--' if v is None else fmt_pm_ci(100 * v['mean'], v['hw'], bold=near_top(100 * v['mean'], top)) + mark
            dag_m = '$^{\\dagger}$' if e['scorer'] == 'metric' else ''
            dag_d = '$^{\\dagger}$' if e['scorer'] == 'direct' else ''
            dk = e['direct_vs_qsel']['ck']
            lines.append(' & '.join([tex_env(env), f"$({e['k']},{e['k']})$", tcell(e['best_fixed']), tcell(metric, dag_m), tcell(direct, dag_d),
                                     tcell(e['qsel']['c1']), tcell(e['qsel']['ck']), '--' if dk is None else fmt_ci(dk)]) + ' \\\\')
            key = macro_key(env)
            for c, v in ents.items():
                if v is None:
                    continue
                macros[f'Mech{c}{key}'] = f"{100 * v['mean']:.0f}"
                macros[f'MechDelta{c}{key}'] = fmt_ci(v['d_wmpa'])
                if v.get('d_best'):
                    macros[f'MechDeltaBest{c}{key}'] = fmt_ci(v['d_best'])
                counts[c]['n'] += 1; counts[c]['deltas'].append(100 * v['d_wmpa']['delta'])
                counts[c]['down'] += int(sig_down(v['d_wmpa'])); counts[c]['up'] += int(sig_up(v['d_wmpa']))
                counts[c]['above_best'] += int(sig_up(v.get('d_best')))
            if e['direct_wmpa']:
                macros[f'MechDirect{key}'] = f"{100 * e['direct_wmpa']['mean']:.0f}"
            for lab, c in (('DirectQselOne', 'c1'), ('DirectQselK', 'ck')):
                d = e['direct_vs_qsel'][c]
                if d is None:
                    continue
                macros[f'Mech{lab}{key}'] = fmt_ci(d)  # direct-value WMPA minus Q-select (positive: rollouts help)
                counts[lab]['n'] += 1; counts[lab]['deltas'].append(100 * d['delta'])
                counts[lab]['down'] += int(sig_down(d)); counts[lab]['up'] += int(sig_up(d))
        for c, d in counts.items():
            macros[f'MechNumEnvs{c}'] = str(d['n']); macros[f'MechNumDown{c}'] = str(d['down']); macros[f'MechNumUp{c}'] = str(d['up'])
            macros[f'MechNumAboveBest{c}'] = str(d['above_best'])
            macros[f'MechMeanDelta{c}'] = f"{np.mean(d['deltas']):+.0f}" if d['deltas'] else '--'
        with open(os.path.join(tdir, 'scorer_selector.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcc|cc|cc|r}\n\\toprule\n'
                    ' & & & \\multicolumn{2}{c|}{\\wmpp{} (imagined states)} & \\multicolumn{2}{c|}{Q-select (no rollout)} & \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
                    'Dataset & $(k,c)$ & Best & metric value & direct value & $c{=}1$ & $c{=}k$ & $\\Delta$ (direct $-$ Q-select, $c{=}k$) \\\\\n\\midrule\n'
                    + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
        with open(os.path.join(tdir, 'numbers_mechanism.tex'), 'w') as f:
            f.write('% Auto-generated by scripts/report_mechanism.py -- do not edit.\n')
            for kk, vv in macros.items():
                f.write(f'\\newcommand{{\\{kk}}}{{{vv}}}\n')
        print(f'wrote {tdir}/scorer_selector.tex + numbers_mechanism.tex ({len(macros)} macros)')

    def clean(o):
        if isinstance(o, dict):
            return {str(kk): clean(v) for kk, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(FLAGS.out + '.json', 'w') as f:
        json.dump(clean(out), f, indent=1)
    with open(FLAGS.out + '.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nwrote {FLAGS.out}.md / .json')


if __name__ == '__main__':
    app.run(main)
