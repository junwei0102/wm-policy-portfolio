"""Generate every table/figure of the ICLR draft from the final og50 report.

Inputs (all produced by earlier stages, see RUNBOOK.md):
  * --report      og50_final_sweep.json from scripts/report_og50.py
                  (--onestep_as_cell --extra_tags=og50r1,og50k5)
  * --env_config  manifests/wmpp_env_config.json (eval.csv baselines per seed)
  * --eval_root   planner_eval root; per-seed WMPP/Random episode files are read
                  from bank_sd<s>_{og50,og50k5,og50r1}/ and the fixed-policy
                  per-episode successes from bank_sd<s>_abl/ (20-episode
                  ablation runs, the only runs that executed every bank member on
                  identical episodes -> static per-episode oracle).

Outputs (written under --out_dir, default WMPP_ICLR2027/):
  tables/main_results.tex     Family / Dataset / Best policy / SR / Random / WMPP
  tables/kc_sweep.tex         diagonal k=c sweep, selected cell in bold
  tables/random_by_c.tex      Random-Switch at every commitment interval
  tables/bank_policies.tex    fixed-policy family means (official protocol, paired episode seeds)
  tables/runtime.tex          switching, model calls, latency
  tables/bank_stats.tex       bank statistics + static per-episode oracle
  tables/oracle_panel.tex     compact Best / U / WMPP panel (all datasets) next to figures/regimes.pdf
  tables/numbers.tex          \\newcommand macros used in the running text
  figures/gain_ci.pdf         gain over best policy with hierarchical-bootstrap CIs
  figures/regimes.pdf         gain vs complementarity headroom
  figures/kc_profiles.pdf     success vs k=c for every environment

Usage:
  python scripts/make_paper_assets.py
"""

import csv
import glob
import json
import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paper_common import bh, fmt_ci, fmt_pm_ci, half_width, hier_boot, holm, macro_key, near_top, paired_rows, rows_by_policy, wm_val_rank_corr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON.')
flags.DEFINE_string('env_config', os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'), 'Env config.')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'Paper root (tables/, figures/).')
flags.DEFINE_string('main_tags', 'og50,og50k5,og50r1,og50cr', 'Dir tags holding the official-protocol runs.')
flags.DEFINE_string('lavl_family_k', 'maze:1,cube:5,scene:10,puzzle:100',
                    'Held-out-selected interval of the LAVL head per family (the LAVL-only rule); used for the '
                    'scorer-ablation table where a family reports the critic scorer instead.')
flags.DEFINE_string('critic_tag', 'og50cr', 'Tag of the critic-scorer diagonal on the test episodes (puzzles).')
flags.DEFINE_string('critic_ctrl_tag', 'og50qv', 'Tag holding critic-scorer rows on the control datasets.')
flags.DEFINE_string('critic_ctrl_envs', 'scene-play-v0,cube-double-play-v0', 'Control datasets for the scorer ablation.')
flags.DEFINE_string('oracle_tag', 'og50fx', 'Dir tag whose runs include every fixed bank policy on the og50 '
                    'episode set (5x50 per seed).')
flags.DEFINE_string('oracle_fallback_tag', 'abl', 'Fallback tag (5x20 per seed) used per env while og50fx runs are missing.')
flags.DEFINE_string('extras_dir', '/scratch/jwquan/wmpp/planner_eval/paper_extras',
                    'Where the full k=c sweep table and profile figure are kept for the record; '
                    'the draft itself carries only the settings table (tables/settings.tex).')

NAME = {'hiql': 'HIQL', 'gciql': 'GCIQL', 'gcivl': 'GCIVL', 'crl': 'CRL', 'gcbc': 'GCBC', 'qrl': 'QRL'}
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
FAMILIES = [
    ('Maze', ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0']),
    ('Cube', ['cube-single-play-v0', 'cube-single-noisy-v0', 'cube-double-play-v0', 'cube-double-noisy-v0',
              'cube-triple-play-v0', 'cube-triple-noisy-v0']),  # cube-quadruple-{play,noisy} dropped from the paper (2026-09-16): every method at 0
    ('Scene', ['scene-play-v0', 'scene-noisy-v0']),
    ('Puzzle', ['puzzle-3x3-play-v0', 'puzzle-3x3-noisy-v0', 'puzzle-4x4-play-v0', 'puzzle-4x4-noisy-v0',
                'puzzle-4x5-play-v0', 'puzzle-4x5-noisy-v0', 'puzzle-4x6-play-v0', 'puzzle-4x6-noisy-v0']),
]
KS = [1, 5, 10, 25, 50, 100]
# Reference categorical palette (dataviz skill, slots 1-3: validated all-pairs).
C_WMPP, C_RAND, C_BEST = '#2a78d6', '#eb6834', '#1baf7a'
C_NAVY = '#1E3765'
# Family colours for the regimes scatter (U of T palette): navy, teal, warm red, purple.
C_FAMILY = {'Cube': '#1E3765', 'Puzzle': '#007FA3', 'Scene': '#DC4633', 'Maze': '#6D247A'}
ENV_FAMILY = {e: fam for fam, fenvs in FAMILIES for e in fenvs}


def short(env):
    return env.replace('-v0', '')


def tex_env(env):
    return '\\texttt{' + short(env) + '}'


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def per_seed_means(eval_root, env, variant, tags, seeds=(0, 1, 2)):
    out = []
    for sd in seeds:
        rows = []
        for tag in tags:
            p = os.path.join(eval_root, env, f'bank_sd{sd}_{tag}', 'episodes.csv')
            if os.path.exists(p):
                rows += [float(r['success']) for r in read_rows(p) if r['policy'] == variant]
        out.append(np.mean(rows) if rows else np.nan)
    return np.array(out)


def static_oracle(eval_root, env, tag, seeds=(0, 1, 2)):
    """Fixed-policy stats on identical episodes: per-family mean and the
    'best policy per episode' union success (static per-episode oracle)."""
    fam_means, unions, union_by_ep, fam_eps = {}, [], {}, {}
    for sd in seeds:
        p = os.path.join(eval_root, env, f'bank_sd{sd}_{tag}', 'episodes.csv')
        if not os.path.exists(p):
            return None
        by_ep = {}
        fam_rows = {}
        for r in read_rows(p):
            pol = r['policy']
            if pol.startswith('score') or pol.startswith('random'):
                continue
            fam = pol.split('-sd')[0]
            key = (r['task_id'], r['episode_idx'])
            by_ep.setdefault(key, []).append(float(r['success']))
            fam_rows.setdefault(fam, []).append(float(r['success']))
        unions.append(np.mean([max(v) for v in by_ep.values()]))
        union_by_ep[sd] = {k: max(v) for k, v in by_ep.items()}
        for fam, v in fam_rows.items():
            fam_means.setdefault(fam, []).append(np.mean(v))
            fam_eps.setdefault(fam, {})[sd] = np.array(v)
    fam_means = {f: float(np.mean(v)) for f, v in fam_means.items()}
    best_fam = max(fam_means, key=fam_means.get)
    return dict(union=float(np.mean(unions)), fam_means=fam_means, best=best_fam, best_mean=fam_means[best_fam],
                n_episodes_per_seed=len(by_ep), union_by_ep=union_by_ep, fam_eps=fam_eps)


def per_episode_success(eval_root, env, variant, tags, seeds=(0, 1, 2)):
    """{seed: {(task_id, episode_idx): success}} of one planner variant across the main tag dirs."""
    out = {}
    for sd in seeds:
        d = {}
        for tag in tags:
            p = os.path.join(eval_root, env, f'bank_sd{sd}_{tag}', 'episodes.csv')
            if os.path.exists(p):
                for r in read_rows(p):
                    if r['policy'] == variant:
                        d[(r['task_id'], r['episode_idx'])] = float(r['success'])
        out[sd] = d
    return out


def fmt_pm(mean, std, bold=False):
    s = f'{mean:.0f} \\pm {std:.0f}'
    return f'$\\mathbf{{{s}}}$' if bold else f'${s}$'



def fmt(x, bold=False, nd=0):
    s = f'{x:.{nd}f}'
    return f'$\\mathbf{{{s}}}$' if bold else f'${s}$'


def main(_):
    report = json.load(open(FLAGS.report))
    envcfg = json.load(open(FLAGS.env_config))
    byenv = {x['env_name']: x for x in report}
    tags = FLAGS.main_tags.split(',')
    tdir, fdir = os.path.join(FLAGS.out_dir, 'tables'), os.path.join(FLAGS.out_dir, 'figures')
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(FLAGS.extras_dir, exist_ok=True)
    for stale in (os.path.join(FLAGS.extras_dir, 'kc_sweep.tex'), os.path.join(fdir, 'kc_profiles.pdf')):
        if os.path.exists(stale):
            os.remove(stale)
            print('[assets] removed from draft (kept in extras_dir):', stale)

    envs = [e for _, es in FAMILIES for e in es if e in byenv]
    missing = [e for _, es in FAMILIES for e in es if e not in byenv]
    if missing:
        print('[assets] environments not in report (pending):', missing)

    # ---- per-env numbers ----------------------------------------------------
    R = {}
    for env in envs:
        x = byenv[env]
        bp = x['best_policy']
        if 'best_policy_seeds' in x:  # paired og50fx baselines (report_og50.py --fixed_tag)
            bp_seeds = np.array([x['best_policy_seeds'][str(s)] for s in (0, 1, 2)])
        else:  # legacy: each checkpoint's own eval.csv
            bp_seeds = np.array([envcfg[env]['ogbench_test_success'][f'{bp}-sd{s}'] for s in (0, 1, 2)])
        w = per_seed_means(FLAGS.eval_root, env, x['methods']['WMPP']['variant'], tags)
        r = per_seed_means(FLAGS.eval_root, env, x['methods']['Random']['variant'], tags)
        assert abs(w.mean() - x['methods']['WMPP']['success']) < 1e-6, env
        assert abs(r.mean() - x['methods']['Random']['success']) < 1e-6, env
        _pref = 'critic' if (x.get('validation') or {}).get('scorer') == 'critic' else 'score'
        R[env] = dict(
            best=bp, best_seeds=bp_seeds, wmpp_seeds=w, rand_seeds=r,
            k=x['selected_k'], c=x['methods']['Random']['commit'],
            dW=x['contrasts']['WMPP_vs_best_policy'], dR=x['contrasts']['Random_vs_best_policy'],
            dWR=x['contrasts']['WMPP_vs_Random'],
            p_best=x['contrasts']['WMPP_vs_best_policy'].get('p_value'),
            p_rand=x['contrasts']['WMPP_vs_Random'].get('p_value'),
            select_rule=x.get('select_rule', 'test'), validation=x.get('validation'),
            scorer=(x.get('validation') or {}).get('scorer', 'lavl'),
            # the reported scorer's own diagonal (LAVL: score{k}_commit{k}; critic: critic{k}_commit{k})
            diag={int(k.split('_commit')[0].replace(_pref, '')): v for k, v in x['diagonal'].items()
                  if k.startswith(_pref)},
            rand_c={int(k.replace('random_commit', '')): v for k, v in x['random_by_commit'].items()},
            fam=x['fixed_test_success'], m=x['methods'],
            official_fam=x.get('official_fixed_test_success', envcfg[env]['ogbench_test_family_mean']),
            official_best=x.get('official_best_policy', envcfg[env]['best_policy_ogbench_test']),
        )
        R[env]['oracle'] = static_oracle(FLAGS.eval_root, env, FLAGS.oracle_tag)
        R[env]['oracle_tag'] = FLAGS.oracle_tag
        if R[env]['oracle'] is None:
            R[env]['oracle'] = static_oracle(FLAGS.eval_root, env, FLAGS.oracle_fallback_tag)
            R[env]['oracle_tag'] = FLAGS.oracle_fallback_tag
            print(f'[assets] {env}: oracle from fallback tag {FLAGS.oracle_fallback_tag} (og50fx runs not complete)')
        assert R[env]['oracle'] is not None, f'{env}: no fixed-policy runs under {FLAGS.oracle_tag} or {FLAGS.oracle_fallback_tag}'

    # ---- main table: every bank policy (family mean, paired episodes) + WMPP ----
    lines = []
    col_vals = {a: [] for a in ALGOS}
    wmpp_vals, best_vals = [], []
    for fam, fenvs in FAMILIES:
        fenvs = [e for e in fenvs if e in R]
        if not fenvs:
            continue
        for i, env in enumerate(fenvs):
            d = R[env]
            vals = {a: (100 * d['fam'][a] if a in d['fam'] else None) for a in ALGOS}
            w_mean = 100 * d['wmpp_seeds'].mean()
            top = max([v for v in vals.values() if v is not None] + [w_mean])
            famcell = f'\\multirow{{{len(fenvs)}}}{{*}}{{{fam}}}' if i == 0 else ''
            fam_eps = d['oracle'].get('fam_eps', {})
            cells, hw = [], {}
            for a in ALGOS:
                if vals[a] is None:
                    cells.append('--')
                elif a in fam_eps and len(fam_eps[a]) == 3:
                    hw[a] = half_width(fam_eps[a])
                    cells.append(fmt_pm_ci(vals[a], hw[a], bold=near_top(vals[a], top)))
                else:
                    cells.append(fmt(vals[a], bold=near_top(vals[a], top)))
            # NB: use THIS env's reported variant (d['m']); `x` here is the stale row of the last env.
            wep = per_episode_success(FLAGS.eval_root, env, d['m']['WMPP']['variant'], tags)
            assert all(len(v) == d['oracle']['n_episodes_per_seed'] for v in wep.values()), \
                (env, d['m']['WMPP']['variant'], {sd: len(v) for sd, v in wep.items()}, 'WMPP rows missing for the CI')
            w_half = half_width({sd: np.array(list(v.values())) for sd, v in wep.items()})
            d['w_half'] = w_half
            d['b_half'] = hw.get(d['best'])
            # best fixed policy of this dataset (the report's best family; its mean is \MeanBest's input)
            b_name = d['best']; b_mean = 100 * float(d['best_seeds'].mean())
            if b_name in hw:  # same bootstrap draw as the policy's own cell, so the two columns agree
                cells.append(fmt_pm_ci(b_mean, hw[b_name], bold=near_top(b_mean, top)))
            else:
                cells.append(fmt(b_mean, bold=near_top(b_mean, top)))
            cells.append(fmt_pm_ci(w_mean, w_half, bold=near_top(w_mean, top)))
            lines.append(' & '.join([famcell, tex_env(env)] + cells) + ' \\\\')
            for a in ALGOS:
                if vals[a] is not None:
                    col_vals[a].append(vals[a])
            best_vals.append(b_mean)
            wmpp_vals.append(w_mean)
        lines.append('\\midrule')
    avg = {a: np.mean(col_vals[a]) for a in ALGOS if col_vals[a]}
    avg['wmpp'] = np.mean(wmpp_vals)
    topavg = max(avg.values())
    avg_cells = ['--' if a not in avg else fmt(avg[a], bold=near_top(avg[a], topavg)) for a in ALGOS]
    avg_cells.append(fmt(np.mean(best_vals), bold=near_top(np.mean(best_vals), topavg)))
    avg_cells.append(fmt(avg['wmpp'], bold=near_top(avg['wmpp'], topavg)))
    lines.append(' & '.join([f'\\multicolumn{{2}}{{l}}{{Average ({len(wmpp_vals)} datasets)}}'] + avg_cells) + ' \\\\')
    with open(os.path.join(tdir, 'main_results.tex'), 'w') as f:
        f.write('\\begin{tabular}{ll' + 'c' * len(ALGOS) + 'cc}\n\\toprule\n'
                'Family & Dataset & ' + ' & '.join(NAME[a] for a in ALGOS) + ' & \\bestfixed{} & \\wmpp{} \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    # macro-averages of the three headline quantities (best family per dataset, Random at its family interval, WMPP)
    avg = {k: np.mean([R[e][k + '_seeds'].mean() * 100 for e in envs]) for k in ('best', 'rand', 'wmpp')}
    macros_hiql_n = len(col_vals['hiql'])

    # ---- k=c sweep table -------------------------------------------------------
    rows = []
    for env in envs:
        d = R[env]
        cells = []
        for k in KS:
            v = d['diag'].get(k)
            cells.append('--' if v is None else fmt(v * 100, bold=(k == d['k'])))
        rows.append(' & '.join([tex_env(env)] + cells + [f'$({d["k"]},{d["k"]})$']) + ' \\\\')
    with open(os.path.join(tdir, 'kc_sweep.tex'), 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(KS) + 'c}\n\\toprule\nDataset & '
                + ' & '.join(f'$k{{=}}c{{=}}{k}$' for k in KS) + ' & selected \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- random at every c -----------------------------------------------------
    rows = []
    n_rand_beats = 0
    for env in envs:
        d = R[env]
        wm = d['wmpp_seeds'].mean() * 100
        cells = []
        best_rand = max((v for v in d['rand_c'].values() if v == v), default=np.nan) * 100
        # OGBench convention across Random at every c AND WMPP: bold >= 95% of
        # the row maximum; the interval paired with WMPP in the main table
        # (the selected c) is underlined.
        if best_rand > wm + 1e-9:
            n_rand_beats += 1
        # Only the interval actually used (the family value, shared with WMPP) is
        # reported; the full sweep over c is computed but not tabulated.
        rv = d['rand_c'].get(d['c'])
        rv = rv * 100 if rv is not None and rv == rv else None
        top = max([wm] + ([rv] if rv is not None else []))
        rep_ = per_episode_success(FLAGS.eval_root, env, d['m']['Random']['variant'], tags)
        r_half = half_width({sd: np.array(list(v.values())) for sd, v in rep_.items()}) if all(rep_.values()) else None
        cells = [f"${d['c']}$",
                 '--' if rv is None else (fmt_pm_ci(rv, r_half, bold=near_top(rv, top)) if r_half is not None else fmt(rv, bold=near_top(rv, top))),
                 fmt_pm_ci(wm, d['w_half'], bold=near_top(wm, top))]
        rows.append(' & '.join([tex_env(env)] + cells) + ' \\\\')
    with open(os.path.join(tdir, 'random_by_c.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc}\n\\toprule\n'
                'Dataset & $c$ & \\randomswitch & \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- scorer ablation: LAVL metric head vs a bank member's own critic ----------
    # Puzzle datasets report the critic in the main table (their family's value head);
    # here both scorers are shown at their own held-out-selected interval, with the
    # paired per-episode difference. Two geometric-manipulation datasets serve as
    # controls (critic at the family interval, from the critic sweep tag).
    lavl_k = {a: int(b) for a, b in (x.split(':') for x in FLAGS.lavl_family_k.split(','))}
    ctrl = [e for e in FLAGS.critic_ctrl_envs.split(',') if e in R]
    fam_of = {e: f.lower() for f, es in FAMILIES for e in es}
    scorer_macros, srows, n_sig_puz, deltas_puz, n_worse_ctrl = {}, [], 0, [], 0
    rng_sc = np.random.default_rng(3)
    for env in [e for e in envs if R[e]['scorer'] == 'critic'] + ctrl:
        env_dir = os.path.join(FLAGS.eval_root, env)
        is_ctrl = env in ctrl
        kl = lavl_k[fam_of[env]]
        kc = R[env]['k']                       # critic k* (puzzles) / family k (controls)
        ctag = FLAGS.critic_ctrl_tag if is_ctrl else FLAGS.critic_tag
        per_seed, lv, cv = {}, [], []
        for sd in (0, 1, 2):
            a = rows_by_policy(env_dir, sd, ctag).get(f'critic{kc}_commit{kc}')
            b = rows_by_policy(env_dir, sd, 'og50').get(f'score{kl}_commit{kl}')
            if not a or not b:
                continue
            per_seed[sd] = paired_rows(a, b)
            cv.append(np.mean([float(r['success']) for r in a])); lv.append(np.mean([float(r['success']) for r in b]))
        best = 100 * float(R[env]['best_seeds'].mean())  # R[env]['best'] is the policy NAME
        if len(per_seed) == 3:
            c = hier_boot(per_seed, 2000, rng_sc)
            key = macro_key(env)
            scorer_macros[f'ScorerDelta{key}'] = f'{100 * c["delta"]:+.0f}'
            scorer_macros[f'ScorerDeltaCI{key}'] = fmt_ci(c)
            scorer_macros[f'ScorerLavl{key}'] = f'{100 * np.mean(lv):.0f}'
            scorer_macros[f'ScorerCritic{key}'] = f'{100 * np.mean(cv):.0f}'
            if is_ctrl:
                n_worse_ctrl += int(c['significant'] and c['delta'] < 0)
            else:
                deltas_puz.append(100 * c['delta']); n_sig_puz += int(c['significant'] and c['delta'] > 0)
            top = max(best, 100 * np.mean(lv), 100 * np.mean(cv))
            srows.append(' & '.join([tex_env(env) + (' (control)' if is_ctrl else ''), fmt(best, bold=near_top(best, top)),
                                     f'{kl}', fmt(100 * np.mean(lv), bold=near_top(100 * np.mean(lv), top)),
                                     f'{kc}', fmt(100 * np.mean(cv), bold=near_top(100 * np.mean(cv), top)),
                                     fmt_ci(c)]) + ' \\\\')
        else:
            srows.append(' & '.join([tex_env(env) + (' (control)' if is_ctrl else ''), fmt(best), f'{kl}', '--', f'{kc}', '--', '--']) + ' \\\\')
    with open(os.path.join(tdir, 'scorer_ablation.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccccc r}\n\\toprule\n'
                'Dataset & Best & $k$ & \\wmpp{} (metric value) & $k$ & \\wmpp{} (direct value) & direct $-$ metric \\\\\n\\midrule\n'
                + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n')
    scorer_macros['ScorerNumSigPuzzle'] = str(n_sig_puz)
    scorer_macros['ScorerNumPuzzle'] = str(len(deltas_puz))
    scorer_macros['ScorerMeanDeltaPuzzle'] = f'{np.mean(deltas_puz):+.0f}' if deltas_puz else '--'
    scorer_macros['ScorerNumWorseCtrl'] = str(n_worse_ctrl)
    scorer_macros['ScorerNumCtrl'] = str(len(ctrl))

    # ---- bank policies table ---------------------------------------------------
    rows = []
    for env in envs:
        d = R[env]
        vals = {a: d['fam'].get(a) for a in ALGOS}
        wm = d['wmpp_seeds'].mean()
        top = max([v for v in vals.values() if v is not None] + [wm])
        cells = ['--' if vals[a] is None else fmt(vals[a] * 100, bold=near_top(vals[a], top)) for a in ALGOS]
        rows.append(' & '.join([tex_env(env)] + cells + [fmt(wm * 100, bold=near_top(wm, top))]) + ' \\\\')
    with open(os.path.join(tdir, 'bank_policies.tex'), 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(ALGOS) + 'c}\n\\toprule\nDataset & '
                + ' & '.join(NAME[a] for a in ALGOS) + ' & \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- runtime / selection behaviour ----------------------------------------
    rows = []
    for env in envs:
        d = R[env]
        w, r = d['m']['WMPP'], d['m']['Random']
        rows.append(' & '.join([
            tex_env(env), f'$({d["k"]},{d["k"]})$',
            f'{w["episode_length"]:.0f}',
            f'{w["switches_per_episode"]:.1f}', f'{r["switches_per_episode"]:.1f}',
            f'{w["decisions_per_episode"]:.1f}',
            f'{w["dynamics_calls_per_env_step"]:.1f}',
            f'{w["usage_entropy_bits"]:.2f}',
            f'{w["act_ms_per_step"]:.2f}', f'{r["act_ms_per_step"]:.2f}', f'{w["fixed_ms_anchor"]:.2f}',
        ]) + ' \\\\')
    with open(os.path.join(tdir, 'runtime.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccccccc}\n\\toprule\n'
                ' & & & \\multicolumn{2}{c}{switches/ep} & & & & \\multicolumn{3}{c}{ms/step (CPU)} \\\\\n'
                '\\cmidrule(lr){4-5}\\cmidrule(lr){9-11}\n'
                'Dataset & $(k,c)$ & ep.\\ len & \\wmpp & Rand. & decisions/ep & WM calls/step & usage $H$ (bits) & \\wmpp & Rand. & fixed \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- bank statistics + static oracle --------------------------------------
    rows = []
    stats = {}
    for env in envs:
        d = R[env]
        J = sorted([v for v in d['fam'].values()], reverse=True)
        bmax, j2 = J[0], (J[1] if len(J) > 1 else 0.0)
        n_half = sum(1 for v in J if v >= 0.5 * bmax and v > 0)
        o = d['oracle']
        dW = d['dW']['delta'] * 100
        stats[env] = dict(bmax=bmax * 100, gap=(bmax - j2) * 100, n_half=n_half,
                          union=o['union'] * 100 if o else np.nan,
                          obest=o['best_mean'] * 100 if o else np.nan, dW=dW,
                          wmpp=d['wmpp_seeds'].mean() * 100)
        # Paired hierarchical bootstrap of WMPP - U on the same episodes (reviewer W7/Q6).
        if o and 'union_by_ep' in o:
            wep = per_episode_success(FLAGS.eval_root, env, d['m']['WMPP']['variant'], tags)
            per_seed = {}
            for sd, ue in o['union_by_ep'].items():
                keys = sorted(set(ue) & set(wep.get(sd, {})))
                if keys:
                    per_seed[sd] = np.array([wep[sd][k] - ue[k] for k in keys])
            if per_seed:
                stats[env]['dU'] = hier_boot(per_seed, 10000, np.random.default_rng(0))
        s = stats[env]
        if abs(s['obest'] - s['bmax']) > 0.05:
            print(f'[assets] WARNING {env}: oracle best {s["obest"]:.1f} != table best {s["bmax"]:.1f} '
                  '(oracle tag and report baselines come from different runs)')
        head = s['union'] - s['obest']
        u_half = None
        if o and 'union_by_ep' in o:
            u_half = half_width({sd: np.array(list(ue.values())) for sd, ue in o['union_by_ep'].items()})
        # Bold rule (decided 2026-09-12): OGBench convention over the three success columns
        # B_max, U (hindsight oracle), and WMPP -- every entry within 95% of their row maximum.
        s['u_half'] = u_half
        top = float(np.nanmax([s['bmax'], s['union'], s['wmpp']]))
        b_cell = fmt_pm_ci(s['bmax'], d['b_half'], bold=near_top(s['bmax'], top)) if d.get('b_half') is not None else fmt(s['bmax'], bold=near_top(s['bmax'], top))
        u_bold = (not np.isnan(s['union'])) and near_top(s['union'], top)
        u_cell = fmt_pm_ci(s['union'], u_half, bold=u_bold) if u_half is not None else fmt(s['union'], bold=u_bold)
        rows.append(' & '.join([
            tex_env(env), b_cell, fmt(s['gap']),
            u_cell, fmt(head),
            fmt_pm_ci(s['wmpp'], d['w_half'], bold=near_top(s['wmpp'], top)), fmt_ci(d['dW']),
            # paired per-episode contrast WMPA - U (hierarchical bootstrap over bank seeds, then episodes)
            fmt_ci(s['dU']) if 'dU' in s else '--', 'PADJ_' + env,
        ]) + ' \\\\')
    # Multiplicity: Holm (FWER) and Benjamini-Hochberg (FDR) over the per-dataset WMPP-vs-best and
    # WMPP-vs-Random bootstrap p-values (reviewer W10).
    def fmt_p(p):
        return '--' if p is None else ('$<10^{-4}$' if p <= 1e-4 else f'{p:.3g}')
    penv = [e for e in envs if R[e]['p_best'] is not None]
    adj = {}
    if penv:
        pb = np.array([R[e]['p_best'] for e in penv]); pr = np.array([R[e]['p_rand'] for e in penv if R[e]['p_rand'] is not None])
        hb, bb = holm(pb), bh(pb)
        adj = {e: dict(holm=float(hb[i]), bh=float(bb[i])) for i, e in enumerate(penv)}
        renv = [e for e in penv if R[e]['p_rand'] is not None]
        hr, br = (holm(pr), bh(pr)) if len(pr) else (np.array([]), np.array([]))
        for i, e in enumerate(renv):
            adj[e].update(holm_rand=float(hr[i]), bh_rand=float(br[i]))
    rows = [r.replace(' & PADJ_' + e, '') for r, e in zip(rows, envs)]  # p_Holm column dropped from the table (counts stay in the text)
    with open(os.path.join(tdir, 'bank_stats.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccccc rr}\n\\toprule\n'
                'Dataset & $B_{\\max}$ & $D_{\\mathrm{gap}}$ & $U$ (oracle) & $U-B_{\\max}$ & \\wmpp & $\\Delta$\\wmpp{} [95\\% CI] & \\wmpp$-U$ [95\\% CI] \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- oracle panel (Fig. "when", panel b): every dataset, family order --------
    prow = []
    for fam, fenvs in FAMILIES:
        fenvs = [e for e in fenvs if e in R]
        if not fenvs:
            continue
        for env in fenvs:
            s = stats[env]
            # Uparrow: WMPP point estimate above the best policy on these episodes.
            # Bold: WMPP point estimate above the oracle U (exact number, no threshold).
            above_best = s['wmpp'] > s['obest'] + 1e-9
            above_union = s['wmpp'] > s['union'] + 1e-9
            wcell = fmt(s['wmpp'], bold=above_union) + ('$^{\\uparrow}$' if above_best else '')
            prow.append(' & '.join([tex_env(env), fmt(s['obest']), fmt(s['union']), wcell]) + ' \\\\')
        prow.append('\\midrule')
    prow = prow[:-1]
    with open(os.path.join(tdir, 'oracle_panel.tex'), 'w') as f:
        f.write('\\begin{tabular}{lrrr}\n\\toprule\nDataset & Best & $U$ & \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(prow) + '\n\\bottomrule\n\\end{tabular}\n')


    # ---- per-dataset settings table (replaces the k=c sweep in the draft) ------
    srows = []
    for env in envs:
        d = R[env]
        wmcfg = {}
        try:
            with open(os.path.join(envcfg[env]['wm_dir'], 'flags.json')) as f:
                wmcfg = json.load(f).get('wm', {})
        except FileNotFoundError:
            print('[assets] WARNING: no flags.json under wm_dir for', env)
        rc = wm_val_rank_corr(envcfg[env]['wm_dir'], envcfg[env].get('wm_epoch'))
        gamma = wmcfg.get('discount')
        srows.append(' & '.join([
            tex_env(env), str(len(d['fam'])), f'$({d["k"]},{d["k"]})$', str(d['c']),
            str(wmcfg.get('horizon', '--')),
            '--' if gamma is None else f'{gamma:.3f}'.rstrip('0').rstrip('.'),
            '--' if 'lavl_expectile' not in wmcfg else f"{wmcfg['lavl_expectile']:.1f}",
            f"{wmcfg.get('lavl_smoothness_weight', 0.0):g}",  # flag absent in older runs = model default 0
            f"{envcfg[env].get('wm_epoch', 0) // 1000}k",
            '--' if rc is None else f'{rc:.2f}',
        ]) + ' \\\\')
    # Compact per-family settings table (what the paper uses): k=c, Random c, WM settings.
    frows = []
    for famname, fenvs in FAMILIES:
        fenvs = [e for e in fenvs if e in R]
        if not fenvs:
            continue
        cfgs = []
        for e in fenvs:
            try:
                cfgs.append(json.load(open(os.path.join(envcfg[e]['wm_dir'], 'flags.json'))).get('wm', {}))
            except FileNotFoundError:
                cfgs.append({})
        def setstr(key, fmt):
            vals = sorted({fmt(c[key]) for c in cfgs if key in c}, key=lambda x: float(x) if x.replace('.', '').isdigit() else 0, reverse=True)
            return '/'.join(vals) if vals else '--'
        ks = sorted({R[e]['k'] for e in fenvs}); cs = sorted({R[e]['c'] for e in fenvs})
        # Random-Switch runs at the SAME interval as WMPP (one value per family);
        # fail loudly if the selection rule ever separates them again.
        assert ks == cs, f'{famname}: Random c {cs} != WMPP k=c {ks} (rule must keep them matched)'
        scorers = {R[e]['scorer'] for e in fenvs}
        assert len(scorers) == 1, f'{famname}: mixed scorers {scorers} (the value head is chosen per family)'
        head = {'lavl': 'metric', 'critic': 'direct'}[scorers.pop()]
        if head == 'direct':
            # The scorer is the bank's GCIQL state value: report ITS gamma/kappa (read from the
            # member's training flags), not the metric head's; the smoothness term does not apply.
            dcfgs = []
            for e in fenvs:
                for fp in sorted(glob.glob(os.path.join(envcfg[e]['policy_root'], '*', 'flags.json'))):
                    try:
                        fl = json.load(open(fp))
                    except json.JSONDecodeError:
                        continue
                    if fl.get('env_name') == e and (fl.get('agent') or {}).get('agent_name') == 'gciql':
                        dcfgs.append(fl['agent']); break
            assert len(dcfgs) == len(fenvs), f'{famname}: GCIQL flags not found for every dataset'
            def dstr(key, fmt):
                return '/'.join(sorted({fmt(c[key]) for c in dcfgs if key in c}, reverse=True))
            gamma_s = dstr('discount', lambda v: f'{v:.3f}'.rstrip('0').rstrip('.'))
            kappa_s, smooth_s = dstr('expectile', lambda v: f'{v:.1f}'), '--'
        else:
            gamma_s = setstr('discount', lambda v: f'{v:.3f}'.rstrip('0').rstrip('.'))
            kappa_s = setstr('lavl_expectile', lambda v: f'{v:.1f}')
            smooth_s = setstr('lavl_smoothness_weight', lambda v: f'{v:g}')
        frows.append(' & '.join([
            famname, str(len(fenvs)), head, '/'.join(str(k) for k in ks),
            setstr('horizon', lambda v: str(v)), gamma_s, kappa_s, smooth_s,
        ]) + ' \\\\')
    with open(os.path.join(tdir, 'settings_family.tex'), 'w') as f:
        f.write('\\begin{tabular}{lclccccc}\n\\toprule\n'
                'Family & datasets & value head & $k{=}c$ & WM $H$ & $\\gamma$ & $\\kappa$ & $\\lambda_{\\mathrm{smooth}}$ \\\\\n\\midrule\n'
                + '\n'.join(frows) + '\n\\bottomrule\n\\end{tabular}\n')
    with open(os.path.join(tdir, 'settings.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccccccccc}\n\\toprule\n'
                'Dataset & $P$ & $(k,c)$ & Random $c$ & WM $H$ & $\\gamma$ & $\\kappa$ & $\\lambda_{\\mathrm{smooth}}$'
                ' & WM ckpt & val.\\ rank corr. \\\\\n\\midrule\n'
                + '\n'.join(srows) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- numbers for the running text -----------------------------------------
    n_up = sum(1 for e in envs if R[e]['dW']['significant'] and R[e]['dW']['delta'] > 0)
    n_down = sum(1 for e in envs if R[e]['dW']['significant'] and R[e]['dW']['delta'] < 0)
    n_wr = sum(1 for e in envs if R[e]['dWR']['significant'] and R[e]['dWR']['delta'] > 0)
    n_wr_neg = sum(1 for e in envs if R[e]['dWR']['significant'] and R[e]['dWR']['delta'] < 0)
    n_rand_up = sum(1 for e in envs if R[e]['dR']['significant'] and R[e]['dR']['delta'] > 0)
    n_rand_down = sum(1 for e in envs if R[e]['dR']['significant'] and R[e]['dR']['delta'] < 0)
    # 'exceeds the static oracle' is counted with a 1-point margin so that floor /
    # saturation datasets (0.8 vs 0.1, 99.9 vs 99.7) do not inflate the count.
    n_above_union = sum(1 for e in envs if stats[e]['wmpp'] >= stats[e]['union'] + 1.0)
    n_above_union_any = sum(1 for e in envs if stats[e]['wmpp'] > stats[e]['union'] + 1e-9)
    # Pre-specified criterion: the paired 95% interval of WMPP - U excludes zero (positive / negative).
    n_above_union_sig = sum(1 for e in envs if 'dU' in stats[e] and stats[e]['dU']['ci_lo'] > 0)
    n_below_union_sig = sum(1 for e in envs if 'dU' in stats[e] and stats[e]['dU']['ci_hi'] < 0)
    print(f'[assets] WMPP - U paired CI: {n_above_union_sig} datasets significantly above U, {n_below_union_sig} below')
    n_above_best_paired = sum(1 for e in envs if stats[e]['wmpp'] > stats[e]['obest'] + 1e-9)
    # Agreement between each checkpoint's own eval.csv and its paired re-evaluation (all bank members).
    base_diffs = [abs(R[e]['fam'][a] - R[e]['official_fam'][a]) * 100
                  for e in envs for a in R[e]['fam'] if a in R[e]['official_fam']]
    best_changed = [e for e in envs if R[e]['best'] != R[e]['official_best']]
    print(f'[assets] eval.csv vs paired re-evaluation: max |diff| {max(base_diffs):.1f}, mean {np.mean(base_diffs):.1f} '
          f'over {len(base_diffs)} (env, family) pairs; best family differs on {best_changed}')
    n_better_mean = sum(1 for e in envs if R[e]['wmpp_seeds'].mean() > R[e]['best_seeds'].mean() + 1e-9)
    n_equal_mean = sum(1 for e in envs if abs(R[e]['wmpp_seeds'].mean() - R[e]['best_seeds'].mean()) <= 1e-9)
    mean_dWR = np.mean([R[e]['dWR']['delta'] for e in envs]) * 100
    macros = {
        'NumEnvs': str(len(envs)), 'NumEnvsPending': str(len(missing)),
        'MeanWMPP': f'{avg["wmpp"]:.0f}', 'MeanBest': f'{avg["best"]:.0f}', 'MeanRandom': f'{avg["rand"]:.0f}',
        'MeanGain': f'{avg["wmpp"] - avg["best"]:.0f}', 'MeanWMPPminusRandom': f'{mean_dWR:.0f}',
        'NumEnvsHiql': str(macros_hiql_n),
        'NumSigUp': str(n_up), 'NumSigDown': str(n_down), 'NumNS': str(len(envs) - n_up - n_down),
        'NumWMPPvsRandomSig': str(n_wr), 'NumWMPPvsRandomNeg': str(n_wr_neg),
        'NumRandomSigUp': str(n_rand_up), 'NumRandomSigDown': str(n_rand_down),
        'NumRandomBestCBeats': str(n_rand_beats), 'NumAboveUnion': str(n_above_union), 'NumAboveUnionAny': str(n_above_union_any), 'NumAboveUnionSig': str(n_above_union_sig), 'NumBelowUnionSig': str(n_below_union_sig), 'NumAboveBestPaired': str(n_above_best_paired),
        'MaxAbsBaselineDiff': f'{max(base_diffs):.0f}', 'MeanAbsBaselineDiff': f'{np.mean(base_diffs):.1f}',
        'NumBestFamilySame': str(len(envs) - len(best_changed)),
        'NumBetterMean': str(n_better_mean), 'NumEqualMean': str(n_equal_mean),
        'NumSigUpHolm': str(sum(1 for e in adj if adj[e]['holm'] < 0.05 and R[e]['dW']['delta'] > 0)),
        'NumSigDownHolm': str(sum(1 for e in adj if adj[e]['holm'] < 0.05 and R[e]['dW']['delta'] < 0)),
        'NumSigUpBH': str(sum(1 for e in adj if adj[e]['bh'] < 0.05 and R[e]['dW']['delta'] > 0)),
        'NumSigDownBH': str(sum(1 for e in adj if adj[e]['bh'] < 0.05 and R[e]['dW']['delta'] < 0)),
        'NumWMPPvsRandomSigHolm': str(sum(1 for e in adj if adj[e].get('holm_rand', 1) < 0.05 and R[e]['dWR']['delta'] > 0)),
        'NumWMPPvsRandomNegHolm': str(sum(1 for e in adj if adj[e].get('holm_rand', 1) < 0.05 and R[e]['dWR']['delta'] < 0)),
        'NumWMPPvsRandomSigBH': str(sum(1 for e in adj if adj[e].get('bh_rand', 1) < 0.05 and R[e]['dWR']['delta'] > 0)),
        'NumWMPPvsRandomNegBH': str(sum(1 for e in adj if adj[e].get('bh_rand', 1) < 0.05 and R[e]['dWR']['delta'] < 0)),
        'SelectRule': next(iter({R[e]['select_rule'] for e in envs}), 'test').replace('_', '-'),
        'NumCneqK': str(sum(1 for e in envs if R[e]['c'] != R[e]['k'])),
    }
    macros.update(scorer_macros)
    print(f"[assets] multiplicity: Holm up {macros['NumSigUpHolm']} / BH up {macros['NumSigUpBH']} (uncorrected {n_up}); "
          f"vs Random Holm {macros['NumWMPPvsRandomSigHolm']}+/{macros['NumWMPPvsRandomNegHolm']}-; selection rule {macros['SelectRule']}")
    n_ep = sorted({R[e]['oracle']['n_episodes_per_seed'] for e in envs})
    n_fallback = sum(1 for e in envs if R[e]['oracle_tag'] != FLAGS.oracle_tag)
    macros['OracleEpsPerGoal'] = '/'.join(str(n // 5) for n in n_ep)
    macros['OracleEpisodesPerSeed'] = '/'.join(str(n) for n in n_ep)
    macros['OracleEpisodes'] = '/'.join(str(3 * n) for n in n_ep)
    macros['NumOracleFallback'] = str(n_fallback)
    print(f'[assets] oracle: episodes/seed {n_ep}; {n_fallback} envs on fallback tag')
    for env in envs:
        key = ''.join(w.capitalize() for w in short(env).replace('x', 'by').split('-'))
        for dgt, word in (('3', 'Three'), ('4', 'Four'), ('5', 'Five'), ('6', 'Six')):
            key = key.replace(dgt, word)  # LaTeX macro names cannot contain digits
        d = R[env]
        macros[f'Gain{key}'] = f'{d["dW"]["delta"] * 100:+.0f}'
        macros[f'Wmpp{key}'] = f'{d["wmpp_seeds"].mean() * 100:.0f}'
        macros[f'Best{key}'] = f'{d["best_seeds"].mean() * 100:.0f}'
        J = sorted(d['fam'].values(), reverse=True)
        j2 = J[1] if len(J) > 1 else 0.0
        macros[f'Second{key}'] = f'{j2 * 100:.0f}'          # runner-up family
        macros[f'Dgap{key}'] = f'{(J[0] - j2) * 100:.0f}'    # dominance gap
        macros[f'Rand{key}'] = f'{d["rand_seeds"].mean() * 100:.0f}'
        macros[f'GainRand{key}'] = f'{d["dR"]["delta"] * 100:+.0f}'
        macros[f'WvsR{key}'] = f'{d["dWR"]["delta"] * 100:+.0f}'
        macros[f'Kc{key}'] = str(d['k'])
        macros[f'RandC{key}'] = str(d['c'])
        macros[f'Pval{key}'] = fmt_p(d['p_best'])
        if env in adj:
            macros[f'PadjHolm{key}'] = fmt_p(adj[env]['holm'])
            macros[f'PadjBH{key}'] = fmt_p(adj[env]['bh'])
        macros[f'Union{key}'] = f'{stats[env]["union"]:.0f}'
        if 'dU' in stats[env]:
            du = stats[env]['dU']
            macros[f'UnionDelta{key}'] = f'{100 * du["delta"]:+.0f}'
            macros[f'UnionCI{key}'] = f'[{100 * du["ci_lo"]:+.0f}, {100 * du["ci_hi"]:+.0f}]'
        macros[f'Obest{key}'] = f'{stats[env]["obest"]:.0f}'
        macros[f'Headroom{key}'] = f'{stats[env]["union"] - stats[env]["obest"]:.0f}'
        rc = {c: v for c, v in d['rand_c'].items() if v == v}
        cbest = max(rc, key=rc.get)
        macros[f'RandBestC{key}'] = f'{rc[cbest] * 100:.0f}'
        macros[f'RandBestCArg{key}'] = str(cbest)
    with open(os.path.join(tdir, 'numbers.tex'), 'w') as f:
        f.write('% Auto-generated by scripts/make_paper_assets.py -- do not edit.\n')
        for k, v in macros.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

    # ---- figures -----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': '#222', 'xtick.color': '#555',
                         'ytick.color': '#555', 'pdf.fonttype': 42})

    # (1) gain over the best policy with CIs, WMPP vs Random, sorted by WMPP gain
    order = sorted(envs, key=lambda e: R[e]['dW']['delta'])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = np.arange(len(order))
    h = 0.38
    for off, key, col, lab in [(+h / 2, 'dW', C_WMPP, '\\textsc{WMPP}'), (-h / 2, 'dR', C_RAND, 'Random-Switch')]:
        vals = np.array([R[e][key]['delta'] for e in order]) * 100
        lo = np.array([R[e][key]['ci_lo'] for e in order]) * 100
        hi = np.array([R[e][key]['ci_hi'] for e in order]) * 100
        ax.barh(y + off, vals, height=h, color=col, alpha=0.9, label=lab.replace('\\textsc{WMPP}', 'WMPP'))
        ax.errorbar(vals, y + off, xerr=[vals - lo, hi - vals], fmt='none', ecolor='#333', elinewidth=0.7, capsize=1.5)
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short(e) for e in order])
    ax.set_xlabel('Success-rate gain over the best fixed policy (points)')
    ax.grid(axis='x', color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'gain_ci.pdf'))
    plt.close(fig)

    # (2) regimes: gain vs complementarity headroom of the static oracle
    fig, ax = plt.subplots(figsize=(5.5, 2.25))  # full text width, flat: placed with width=\linewidth
    xs = np.array([stats[e]['union'] - stats[e]['obest'] for e in envs])
    ys = np.array([stats[e]['dW'] for e in envs])
    ax.plot([0, max(xs.max(), ys.max()) + 2], [0, max(xs.max(), ys.max()) + 2], ls='--', lw=0.8, color='#9a9a9a',
            label=r'$y=x$: gain explained by per-episode selection')
    for fam, _ in FAMILIES:
        idx = [i for i, e in enumerate(envs) if ENV_FAMILY[e] == fam]
        if idx:
            ax.scatter(xs[idx], ys[idx], s=22, color=C_FAMILY[fam], edgecolor='white', linewidth=0.6,
                       zorder=3, label=fam)
    # boxed legend so its markers are not read as data points near the y=x line
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.5, frameon=True, fancybox=False, framealpha=1.0,
              edgecolor='#bfbfbf', facecolor='white', handletextpad=0.3, borderaxespad=0.3, borderpad=0.4, ncol=5,
              columnspacing=0.9, handlelength=1.6).get_frame().set_linewidth(0.6)  # above the axes: never overlaps a point or the y=x line
    labels_on = {e for e in envs if abs(stats[e]['dW']) > 4 or (stats[e]['union'] - stats[e]['obest']) > 12}
    # (dx pt, dy pt, ha): the crowded low-headroom cluster is labelled to the LEFT of the y axis
    # (xlim starts at -6.5 to make room) so no label covers a marker or another label.
    nudge = {'cube-double-play-v0': (3, -8, 'left'), 'cube-single-play-v0': (3, 3, 'left'),
             'puzzle-3x3-play-v0': (-3, 3, 'right'), 'cube-triple-noisy-v0': (-3, -8, 'right'),
             'puzzle-4x5-play-v0': (3, -5, 'left'), 'puzzle-4x6-play-v0': (3, 4, 'left')}
    for e in envs:
        if e in labels_on:
            dx, dy, ha = nudge.get(e, (3, 2, 'left'))
            ax.annotate(short(e).replace('-navigate', ''), (stats[e]['union'] - stats[e]['obest'], stats[e]['dW']),
                        xytext=(dx, dy), textcoords='offset points', fontsize=6, color='#333', ha=ha)
    ax.set_xlim(-6.5, max(xs.max(), ys.max()) + 2)
    ax.axhline(0, color='#555', lw=0.6)
    ax.set_xlabel('Per-episode oracle headroom over best policy (points)')
    ax.set_ylabel('WMPA gain (points)')
    ax.grid(color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'regimes.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # (3) k=c profiles, small multiples
    ncol = 5
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 1.75 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, env in zip(axes, envs):
        d = R[env]
        ks = [k for k in KS if k in d['diag']]
        ax.plot(ks, [d['diag'][k] * 100 for k in ks], color=C_WMPP, lw=1.6, marker='o', ms=3, label='WMPP')
        rk = [k for k in KS if k in d['rand_c'] and d['rand_c'][k] == d['rand_c'][k]]
        ax.plot(rk, [d['rand_c'][k] * 100 for k in rk], color=C_RAND, lw=1.4, marker='s', ms=2.6, label='Random-Switch')
        ax.axhline(d['best_seeds'].mean() * 100, color=C_BEST, lw=1.2, ls='--', label='best policy')
        ax.plot([d['k']], [d['diag'][d['k']] * 100], marker='o', ms=6, mfc='white', mec=C_WMPP, mew=1.4)
        if max(list(d['diag'].values()) + [v for v in d['rand_c'].values() if v == v] + [d['best_seeds'].mean()]) == 0:
            ax.set_ylim(-0.05, 1.0)  # all-zero floor environment: avoid a 1e-17 axis offset
        ax.set_xscale('log')
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_title(short(env).replace('-navigate', ''), fontsize=7.5, pad=3)
        ax.grid(color='#e6e6e6', lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
    for ax in axes[len(envs):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.99, 0.02))
    fig.supxlabel('scoring horizon $k$ = commitment $c$', fontsize=8, y=0.01)
    fig.supylabel('success rate (%)', fontsize=8, x=0.005)
    fig.tight_layout(rect=(0.01, 0.03, 1, 1))
    fig.savefig(os.path.join(FLAGS.extras_dir, 'kc_profiles.pdf'))
    plt.close(fig)

    print(json.dumps({k: v for k, v in macros.items() if not k[0:4] in ('Gain', 'Wmpp', 'Best', 'Rand', 'WvsR', 'KcAn', 'KcCu', 'KcHu', 'KcPu', 'KcSc')}, indent=1))
    print('[assets] bank stats:')
    for env in envs:
        s = stats[env]
        print(f'  {short(env):32s} Bmax {s["bmax"]:5.1f} gap {s["gap"]:5.1f} N1/2 {s["n_half"]} | obest {s["obest"]:5.1f} U {s["union"]:5.1f} head {s["union"] - s["obest"]:5.1f} | WMPP {s["wmpp"]:5.1f} dW {s["dW"]:+5.1f}')
    print('[assets] wrote', tdir, fdir)


if __name__ == '__main__':
    app.run(main)
