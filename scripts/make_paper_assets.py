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
import json
import os

import numpy as np
from absl import app, flags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON.')
flags.DEFINE_string('env_config', os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'), 'Env config.')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'Paper root (tables/, figures/).')
flags.DEFINE_string('main_tags', 'og50,og50k5,og50r1', 'Dir tags holding the official-protocol runs.')
flags.DEFINE_string('oracle_tag', 'og50fx', 'Dir tag whose runs include every fixed bank policy on the og50 '
                    'episode set (5x50 per seed).')
flags.DEFINE_string('oracle_fallback_tag', 'abl', 'Fallback tag (5x20 per seed) used per env while og50fx runs are missing.')

NAME = {'hiql': 'HIQL', 'gciql': 'GCIQL', 'gcivl': 'GCIVL', 'crl': 'CRL', 'gcbc': 'GCBC', 'qrl': 'QRL'}
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
FAMILIES = [
    ('Maze', ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0', 'humanoidmaze-giant-navigate-v0']),
    ('Cube', ['cube-single-play-v0', 'cube-single-noisy-v0', 'cube-double-play-v0', 'cube-double-noisy-v0',
              'cube-triple-play-v0', 'cube-triple-noisy-v0', 'cube-quadruple-play-v0', 'cube-quadruple-noisy-v0']),
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
    fam_means, unions = {}, []
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
        for fam, v in fam_rows.items():
            fam_means.setdefault(fam, []).append(np.mean(v))
    fam_means = {f: float(np.mean(v)) for f, v in fam_means.items()}
    best_fam = max(fam_means, key=fam_means.get)
    return dict(union=float(np.mean(unions)), fam_means=fam_means, best=best_fam, best_mean=fam_means[best_fam],
                n_episodes_per_seed=len(by_ep))


def fmt_pm(mean, std, bold=False):
    s = f'{mean:.1f} \\pm {std:.1f}'
    return f'$\\mathbf{{{s}}}$' if bold else f'${s}$'


def near_top(v, top, frac=0.95):
    """OGBench convention: bold every entry at or above `frac` of the row maximum."""
    return top > 0 and v >= frac * top - 1e-9


def fmt(x, bold=False, nd=1):
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
        R[env] = dict(
            best=bp, best_seeds=bp_seeds, wmpp_seeds=w, rand_seeds=r,
            k=x['selected_k'], c=x['methods']['Random']['commit'],
            dW=x['contrasts']['WMPP_vs_best_policy'], dR=x['contrasts']['Random_vs_best_policy'],
            dWR=x['contrasts']['WMPP_vs_Random'],
            diag={int(k.split('score')[1].split('_')[0]): v for k, v in x['diagonal'].items()},
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

    # ---- main table ------------------------------------------------------------
    lines = []
    for fam, fenvs in FAMILIES:
        fenvs = [e for e in fenvs if e in R]
        if not fenvs:
            continue
        for i, env in enumerate(fenvs):
            d = R[env]
            vals = {k: (d[k + '_seeds'].mean() * 100, d[k + '_seeds'].std() * 100) for k in ('best', 'rand', 'wmpp')}
            top = max(v[0] for v in vals.values())
            # Per-dataset significance of the WMPP gain is reported in the running text (numbers.tex macros), not in the table.
            famcell = f'\\multirow{{{len(fenvs)}}}{{*}}{{{fam}}}' if i == 0 else ''
            cells = [fmt_pm(*vals[k], bold=near_top(vals[k][0], top)) for k in ('best', 'rand', 'wmpp')]
            lines.append(' & '.join([famcell, tex_env(env), NAME[d['best']]] + cells) + ' \\\\')
        lines.append('\\midrule')
    avg = {k: np.mean([R[e][k + '_seeds'].mean() * 100 for e in envs]) for k in ('best', 'rand', 'wmpp')}
    topavg = max(avg.values())
    lines.append(' & '.join([f'\\multicolumn{{3}}{{l}}{{Average ({len(envs)} datasets)}}']
                            + [fmt(avg[k], bold=near_top(avg[k], topavg)) for k in ('best', 'rand', 'wmpp')]) + ' \\\\')
    with open(os.path.join(tdir, 'main_results.tex'), 'w') as f:
        f.write('\\begin{tabular}{llcccc}\n\\toprule\n'
                'Family & Dataset & Best Policy & Success Rate & Random Switch & WMPP (Ours) \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

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
        top = max([v * 100 for v in d['rand_c'].values() if v == v] + [wm])
        if best_rand > wm + 1e-9:
            n_rand_beats += 1
        for k in KS:
            v = d['rand_c'].get(k)
            if v is None or v != v:
                cells.append('--')
            else:
                cell = fmt(v * 100, bold=near_top(v * 100, top))
                cells.append(f'\\underline{{{cell}}}' if k == d['c'] else cell)
        rows.append(' & '.join([tex_env(env)] + cells + [fmt(wm, bold=near_top(wm, top)), fmt(best_rand - wm, nd=1)]) + ' \\\\')
    with open(os.path.join(tdir, 'random_by_c.tex'), 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(KS) + 'cc}\n\\toprule\nDataset & '
                + ' & '.join(f'$c{{=}}{k}$' for k in KS)
                + ' & \\wmpp & best Random $-$ \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

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
        s = stats[env]
        if abs(s['obest'] - s['bmax']) > 0.05:
            print(f'[assets] WARNING {env}: oracle best {s["obest"]:.1f} != table best {s["bmax"]:.1f} '
                  '(oracle tag and report baselines come from different runs)')
        head = s['union'] - s['obest']
        rows.append(' & '.join([
            tex_env(env), fmt(s['bmax']), fmt(s['gap']), str(n_half),
            fmt(s['union']), fmt(head), fmt(s['wmpp']), fmt(dW, nd=1),
        ]) + ' \\\\')
    with open(os.path.join(tdir, 'bank_stats.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccrr}\n\\toprule\n'
                'Dataset & $B_{\\max}$ & $D_{\\mathrm{gap}}$ & $N_{1/2}$ & $U$ (oracle) & $U-B_{\\max}$ & \\wmpp & $\\Delta$\\wmpp \\\\\n\\midrule\n'
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
        'MeanWMPP': f'{avg["wmpp"]:.1f}', 'MeanBest': f'{avg["best"]:.1f}', 'MeanRandom': f'{avg["rand"]:.1f}',
        'MeanGain': f'{avg["wmpp"] - avg["best"]:.1f}', 'MeanWMPPminusRandom': f'{mean_dWR:.1f}',
        'NumSigUp': str(n_up), 'NumSigDown': str(n_down), 'NumNS': str(len(envs) - n_up - n_down),
        'NumWMPPvsRandomSig': str(n_wr), 'NumWMPPvsRandomNeg': str(n_wr_neg),
        'NumRandomSigUp': str(n_rand_up), 'NumRandomSigDown': str(n_rand_down),
        'NumRandomBestCBeats': str(n_rand_beats), 'NumAboveUnion': str(n_above_union), 'NumAboveUnionAny': str(n_above_union_any), 'NumAboveBestPaired': str(n_above_best_paired),
        'MaxAbsBaselineDiff': f'{max(base_diffs):.1f}', 'MeanAbsBaselineDiff': f'{np.mean(base_diffs):.1f}',
        'NumBestFamilySame': str(len(envs) - len(best_changed)),
        'NumBetterMean': str(n_better_mean), 'NumEqualMean': str(n_equal_mean),
    }
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
        macros[f'Gain{key}'] = f'{d["dW"]["delta"] * 100:+.1f}'
        macros[f'Wmpp{key}'] = f'{d["wmpp_seeds"].mean() * 100:.1f}'
        macros[f'Best{key}'] = f'{d["best_seeds"].mean() * 100:.1f}'
        J = sorted(d['fam'].values(), reverse=True)
        j2 = J[1] if len(J) > 1 else 0.0
        macros[f'Second{key}'] = f'{j2 * 100:.1f}'          # runner-up family
        macros[f'Dgap{key}'] = f'{(J[0] - j2) * 100:.1f}'    # dominance gap
        macros[f'Rand{key}'] = f'{d["rand_seeds"].mean() * 100:.1f}'
        macros[f'GainRand{key}'] = f'{d["dR"]["delta"] * 100:+.1f}'
        macros[f'WvsR{key}'] = f'{d["dWR"]["delta"] * 100:+.1f}'
        macros[f'Kc{key}'] = str(d['k'])
        macros[f'Union{key}'] = f'{stats[env]["union"]:.1f}'
        macros[f'Obest{key}'] = f'{stats[env]["obest"]:.1f}'
        macros[f'Headroom{key}'] = f'{stats[env]["union"] - stats[env]["obest"]:.1f}'
        rc = {c: v for c, v in d['rand_c'].items() if v == v}
        cbest = max(rc, key=rc.get)
        macros[f'RandBestC{key}'] = f'{rc[cbest] * 100:.1f}'
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
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    xs = np.array([stats[e]['union'] - stats[e]['obest'] for e in envs])
    ys = np.array([stats[e]['dW'] for e in envs])
    ax.plot([0, max(xs.max(), ys.max()) + 2], [0, max(xs.max(), ys.max()) + 2], ls='--', lw=0.8, color='#9a9a9a')
    for fam, _ in FAMILIES:
        idx = [i for i, e in enumerate(envs) if ENV_FAMILY[e] == fam]
        if idx:
            ax.scatter(xs[idx], ys[idx], s=22, color=C_FAMILY[fam], edgecolor='white', linewidth=0.6,
                       zorder=3, label=fam)
    ax.legend(loc='lower right', fontsize=6, frameon=False, handletextpad=0.3, borderaxespad=0.4)
    labels_on = {e for e in envs if abs(stats[e]['dW']) > 4 or (stats[e]['union'] - stats[e]['obest']) > 12}
    for e in envs:
        if e in labels_on:
            # Nudge labels that would otherwise collide with a neighbour.
            xy_off = {'cube-double-play-v0': (3, -8), 'cube-single-play-v0': (3, -8)}.get(e, (3, 2))
            ax.annotate(short(e).replace('-navigate', ''), (stats[e]['union'] - stats[e]['obest'], stats[e]['dW']),
                        xytext=xy_off, textcoords='offset points', fontsize=6, color='#333')
    ax.axhline(0, color='#555', lw=0.6)
    ax.set_xlabel('Static per-episode oracle headroom $U-B_{\\max}$ (points)')
    ax.set_ylabel('WMPP gain over best policy (points)')
    ax.grid(color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'regimes.pdf'))
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
    fig.savefig(os.path.join(fdir, 'kc_profiles.pdf'))
    plt.close(fig)

    print(json.dumps({k: v for k, v in macros.items() if not k[0:4] in ('Gain', 'Wmpp', 'Best', 'Rand', 'WvsR', 'KcAn', 'KcCu', 'KcHu', 'KcPu', 'KcSc')}, indent=1))
    print('[assets] bank stats:')
    for env in envs:
        s = stats[env]
        print(f'  {short(env):32s} Bmax {s["bmax"]:5.1f} gap {s["gap"]:5.1f} N1/2 {s["n_half"]} | obest {s["obest"]:5.1f} U {s["union"]:5.1f} head {s["union"] - s["obest"]:5.1f} | WMPP {s["wmpp"]:5.1f} dW {s["dW"]:+5.1f}')
    print('[assets] wrote', tdir, fdir)


if __name__ == '__main__':
    app.run(main)
