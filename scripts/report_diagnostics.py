"""Ranking-diagnostics table across environment families (App. H; reviewer W6, Q4, Q7).

Reads <oracle_dir>/<env>/diagnostics_lavl.json (run_wm_diagnostics.py) and meta.json
(oracle-collection settings), the report JSON for the selected k, and the WM training log
for the validation rank correlation. Horizon 25 is mapped to the nearest collected horizon (20).

Outputs: tables/diagnostics.tex, tables/numbers_diagnostics.tex, --out_json, markdown to stdout.
"""
import json
import os
import re
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import ENV_FAMILY, FAMILIES, macro_key, tex_env, wm_val_rank_corr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('oracle_dir', '/scratch/jwquan/wmpp/oracle_best', 'Oracle root for the horizon profile (best-fixed-policy states; '
                    'falls back to the legacy /scratch/jwquan/wmpp/oracle when absent).')
flags.DEFINE_string('roots', 'best:/scratch/jwquan/wmpp/oracle_best,wmpp:/scratch/jwquan/wmpp/oracle_wmpp,random:/scratch/jwquan/wmpp/oracle_random',
                    'label:root list of decision-state distributions (trajectory generator) for the acc@k* columns.')
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON (selected k).')
flags.DEFINE_string('env_config', os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'), 'Manifest (wm_dir per env).')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'Paper root.')
flags.DEFINE_string('out_json', '/scratch/jwquan/wmpp/planner_eval/diagnostics_report.json', 'JSON output.')
flags.DEFINE_string('small_envs', 'cube-triple-play-v0,scene-play-v0', 'Rows of the compact two-dataset table (tables/diagnostics_small.tex).')

SHOW_H = [1, 5, 10, 20, 50, 100]


def main(_):
    report = {x['env_name']: x for x in json.load(open(FLAGS.report))}
    envcfg = json.load(open(FLAGS.env_config))
    roots = [tuple(r.split(':', 1)) for r in FLAGS.roots.split(',') if r]
    legacy = '/scratch/jwquan/wmpp/oracle'

    def profile_root(e):
        for r in (FLAGS.oracle_dir, legacy):
            if os.path.exists(os.path.join(r, e, 'diagnostics_lavl.json')):
                return r
        return None
    envs = [e for _, es in FAMILIES for e in es if profile_root(e)]
    macros, out, lines = {}, {}, []
    dist_acc = {lab: [] for lab, _ in roots}
    for env in envs:
        root = profile_root(env)
        d = json.load(open(os.path.join(root, env, 'diagnostics_lavl.json')))
        meta = {}
        mp = os.path.join(root, env, 'meta.json')
        if os.path.exists(mp):
            meta = json.load(open(mp))
        vr = d.get('value_ranking') or {}
        vt = d.get('value_top1') or {}
        if not vr:
            print(f'[diag] {env}: no value_ranking block, skipped')
            continue
        k = report[env]['selected_k'] if env in report else None
        hs = k if k is not None and str(k) in vr else {25: 20}.get(k)
        key = macro_key(env)
        first_h = next(iter(vr))
        n_states = vr[first_h]['n_states']
        n_pairs = vr[first_h]['n_discordant_pairs']
        pool = len(meta.get('policies', {}))
        acc = {int(h): vr[h]['accuracy'] for h in vr}
        row = dict(env=env, k=k, h_star=hs, n_states=n_states, n_pairs=n_pairs, pool=pool,
                   base=meta.get('base_policy'), acc=acc,
                   spearman=d.get('disagreement_error_spearman'))
        cells = [tex_env(env), str(pool), str(n_states), str(n_pairs)]
        for h in SHOW_H:
            v = acc.get(h)
            b = hs is not None and h == hs
            cells.append('--' if v is None else (f'\\textbf{{{v:.2f}}}' if b else f'{v:.2f}'))
        if hs is not None:
            a = vr[str(hs)]
            cells.append(f"{a['accuracy']:.2f} [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}]")
            macros[f'DiagAccSel{key}'] = f"{a['accuracy']:.2f}"
            macros[f'DiagAccSelCI{key}'] = f"[{a['ci_lo']:.2f}, {a['ci_hi']:.2f}]"
            row['acc_sel'] = a
            t = vt.get(str(hs))
            if t:
                cells.append(f"{t['delta']:+.2f}")
                macros[f'DiagTopOneDelta{key}'] = f"{t['delta']:+.2f}"
                row['top1_sel'] = t
            else:
                cells.append('--')
            for agg, mac in (('value_ranking_mean', 'MeanAgg'), ('value_ranking_last', 'LastAgg')):
                aa = (d.get(agg) or {}).get(str(hs))
                if aa:
                    macros[f'DiagAcc{mac}{key}'] = f"{aa['accuracy']:.2f}"
                    row[f'acc_{mac}'] = aa['accuracy']
        else:
            cells += ['--', '--']
        # acc@k* and the free 'some pool member succeeds from this boundary' panel per state distribution
        row['dist'] = {}
        for lab, r in roots:
            dj = os.path.join(r, env, 'diagnostics_lavl.json')
            bj = os.path.join(r, env, 'branches.npz')
            entry = {}
            if os.path.exists(dj) and hs is not None:
                vd = json.load(open(dj)).get('value_ranking') or {}
                a2 = vd.get(str(hs))
                if a2:
                    entry.update(acc=a2['accuracy'], ci_lo=a2['ci_lo'], ci_hi=a2['ci_hi'], n_pairs=a2['n_discordant_pairs'],
                                 n_states=a2['n_states'])
                    macros[f'DiagAccSel{lab.capitalize()}{key}'] = f"{a2['accuracy']:.2f}" if np.isfinite(a2['accuracy']) else '--'
                    macros[f'DiagPairs{lab.capitalize()}{key}'] = str(a2['n_discordant_pairs'])
                    if a2['accuracy'] == a2['accuracy']:
                        dist_acc[lab].append(a2['accuracy'])
            if os.path.exists(bj):
                b = np.load(bj, allow_pickle=False)
                S = int(b['state_id'].max()) + 1
                succ = np.zeros((S, len(b['policies'])))
                succ[b['state_id'], b['policy_idx']] = b['success']
                entry['any_success'] = float(succ.max(axis=1).mean())
                macros[f'OraclePanel{lab.capitalize()}{key}'] = f"{100 * entry['any_success']:.1f}"
            row['dist'][lab] = entry
            cells.append('--' if 'acc' not in entry or not np.isfinite(entry['acc']) else f"{entry['acc']:.2f} ({entry['n_pairs']})")
        sp = d.get('disagreement_error_spearman')
        cells.append('--' if sp is None else f'{sp:.2f}')
        rc = None
        if env in envcfg:
            rc = wm_val_rank_corr(envcfg[env]['wm_dir'], envcfg[env].get('wm_epoch'))
        cells.append('--' if rc is None else f'{rc:.2f}')
        row['val_rank_corr'] = rc
        if sp is not None:
            macros[f'DiagSpearman{key}'] = f'{sp:.2f}'
        if rc is not None:
            macros[f'ValRankCorr{key}'] = f'{rc:.2f}'
        macros[f'DiagStates{key}'] = str(n_states)
        macros[f'DiagPairs{key}'] = str(n_pairs)
        for h, mac in ((1, 'One'), (100, 'Hundred')):
            if h in acc:
                macros[f'DiagAcc{mac}{key}'] = f'{acc[h]:.2f}'
        out[env] = row
        # floor datasets have no discordant pairs -> undefined accuracies; print them as dashes, never 'nan'
        cells = [re.sub(r'\bnan\b', '--', c) for c in cells]
        lines.append(' & '.join(cells) + ' \\\\')
    sels = [out[e]['acc_sel']['accuracy'] for e in out if 'acc_sel' in out[e]]
    ones = [out[e]['acc'].get(1) for e in out if out[e]['acc'].get(1) is not None]
    hund = [out[e]['acc'].get(100) for e in out if out[e]['acc'].get(100) is not None]
    macros['DiagNumEnvs'] = str(len(out))
    for lab, accs in dist_acc.items():
        accs = [a for a in accs if a == a]  # drop NaN (datasets with no discordant pairs)
        if accs:
            macros[f'DiagMeanAccSel{lab.capitalize()}'] = f'{np.nanmean(accs):.2f}'  # floor datasets: 0 discordant pairs -> nan
            macros[f'DiagNumEnvs{lab.capitalize()}'] = str(len(accs))
            macros[f'DiagNumBelowChance{lab.capitalize()}'] = str(sum(1 for e in out if out[e]['dist'].get(lab, {}).get('ci_hi', 1) < 0.5))
    if sels:
        macros['DiagMeanAccSel'] = f'{np.nanmean(sels):.2f}'
        macros['DiagMinAccSel'] = f'{np.nanmin(sels):.2f}'
        macros['DiagMaxAccSel'] = f'{max(sels):.2f}'
        macros['DiagNumAccSelAboveChance'] = str(sum(
            1 for e in out if 'acc_sel' in out[e] and out[e]['acc_sel']['ci_lo'] > 0.5))
    if ones:
        macros['DiagMeanAccOne'] = f'{np.mean(ones):.2f}'
    if hund:
        macros['DiagMeanAccHundred'] = f'{np.mean(hund):.2f}'
    rcs = {f: [out[e]['val_rank_corr'] for e in out if ENV_FAMILY.get(e) == f and out[e]['val_rank_corr'] is not None]
           for f in ('Maze', 'Cube', 'Scene', 'Puzzle')}
    manip = rcs['Cube'] + rcs['Scene'] + rcs['Puzzle']
    if manip:
        macros['MeanValRankCorrManip'] = f'{np.mean(manip):.2f}'
    if rcs['Maze']:
        macros['MeanValRankCorrMaze'] = f'{np.mean(rcs["Maze"]):.2f}'

    for nm in ['DiagMeanAccSelBest', 'DiagMeanAccSelWmpp', 'DiagMeanAccSelRandom', 'DiagNumBelowChanceWmpp', 'DiagNumEnvsWmpp',
               'DiagAccSelBestCubeDoublePlay', 'DiagAccSelWmppCubeDoublePlay', 'DiagAccSelRandomCubeDoublePlay',
               'DiagAccSelBestSceneNoisy', 'DiagAccSelWmppSceneNoisy', 'DiagAccSelPointmazeMediumNavigate', 'DiagMeanAccSel',
               'DiagAccOneCubeDoublePlay', 'DiagAccHundredCubeDoublePlay', 'DiagNumEnvs']:
        macros.setdefault(nm, '--')
    tdir = os.path.join(FLAGS.out_dir, 'tables')
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, 'diagnostics.tex'), 'w') as f:
        dcols = ' & '.join(f'{lab}' for lab, _ in roots)
        f.write('\\begin{tabular}{lccc|cccccc|cc|' + 'c' * len(roots) + '|cc}\n\\toprule\n'
                ' & & & & \\multicolumn{6}{c|}{pairwise ranking accuracy at horizon} & '
                'acc.\\ at $k^{*}$ & top-1 & \\multicolumn{' + str(len(roots)) + '}{c|}{acc.\\ at $k^{*}$ by state distribution (pairs)} & & val.\\\\\n'
                'Dataset & pool & states & pairs & 1 & 5 & 10 & 20 & 50 & 100 & [95\\% CI] & '
                '$\\Delta$ & ' + dcols + ' & $\\rho_{\\mathrm{dis}}$ & r.c.\\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    # ---- compact table: two contrasting datasets (one loss, one gain) ------------------------
    small = [e for e in FLAGS.small_envs.split(',') if e in out]
    if small:
        rows_s = []
        for env in small:
            r = out[env]; x = report[env]
            best_sr = 100 * x['fixed_test_success'][x['best_policy']]
            rnd_sr = 100 * x['methods']['Random']['success']; w_sr = 100 * x['methods']['WMPP']['success']
            a = r.get('acc_sel', {}); t = r.get('top1_sel', {})
            dd = r['dist']
            def dcell(lab):
                e = dd.get(lab, {})
                return '--' if 'acc' not in e or not np.isfinite(e['acc']) else f"{e['acc']:.2f} [{e['ci_lo']:.2f}, {e['ci_hi']:.2f}] ({e['n_pairs']})"
            def anycell(lab):
                e = dd.get(lab, {})
                return '--' if 'any_success' not in e else f"{100 * e['any_success']:.0f}"
            rows_s.append(' & '.join([
                tex_env(env), f"$({r['k']},{r['k']})$", f"{best_sr:.0f} / {rnd_sr:.0f} / {w_sr:.0f}",
                dcell('best'), dcell('wmpp'), dcell('random'),
                ('--' if not t else f"{t['delta']:+.2f} [{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]"),
                ' / '.join(anycell(lab) for lab in ('best', 'wmpp', 'random'))]) + ' \\\\')
            key = macro_key(env)
            macros[f'DiagAnyBest{key}'] = anycell('best'); macros[f'DiagAnyWmpp{key}'] = anycell('wmpp'); macros[f'DiagAnyRandom{key}'] = anycell('random')
            macros[f'DiagTopOneCI{key}'] = '--' if not t else f"[{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]"
            for lab in ('best', 'wmpp', 'random'):
                e = dd.get(lab, {})
                if 'acc' in e and np.isfinite(e['acc']):
                    macros[f'DiagAccSelCI{lab.capitalize()}{key}'] = f"[{e['ci_lo']:.2f}, {e['ci_hi']:.2f}]"
        with open(os.path.join(tdir, 'diagnostics_small.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcc|ccc|c|c}\n\\toprule\n'
                    ' & & Best / Rand. / \\wmpp & \\multicolumn{3}{c|}{pairwise ranking accuracy at $k^{*}$ [95\\% CI] (pairs), by decision-state distribution} & top-1 & some member succeeds (\\%)\\\\\n'
                    'Dataset & $(k,c)$ & success (\\%) & best-policy states & \\wmpp{} states & \\randomswitch{} states & $\\Delta$ vs chance [95\\% CI] & best / \\wmpp{} / random\\\\\n\\midrule\n'
                    + '\n'.join(rows_s) + '\n\\bottomrule\n\\end{tabular}\n')
    with open(os.path.join(tdir, 'numbers_diagnostics.tex'), 'w') as f:
        f.write('% Auto-generated by scripts/report_diagnostics.py -- do not edit.\n')
        for kk, vv in macros.items():
            f.write(f'\\newcommand{{\\{kk}}}{{{"--" if "nan" in str(vv) else vv}}}\n')
    with open(FLAGS.out_json, 'w') as f:
        json.dump(dict(rows=out, macros=macros), f, indent=1, default=float)

    print('| dataset | k* | states | pairs | ' + ' | '.join(f'acc@{h}' for h in SHOW_H) + ' | acc@k* | top1 d | spearman | valrc |')
    print('|' + '---|' * (13))
    for env, r in out.items():
        print(f'| {env} | {r["k"]} | {r["n_states"]} | {r["n_pairs"]} | '
              + ' | '.join('--' if r['acc'].get(h) is None else f'{r["acc"][h]:.2f}' for h in SHOW_H)
              + ' | ' + (f"{r['acc_sel']['accuracy']:.2f}" if 'acc_sel' in r else '--')
              + ' | ' + (f"{r['top1_sel']['delta']:+.2f}" if 'top1_sel' in r else '--')
              + ' | ' + ('--' if r['spearman'] is None else f"{r['spearman']:.2f}")
              + ' | ' + ('--' if r['val_rank_corr'] is None else f"{r['val_rank_corr']:.2f}") + ' |')
    print('[diag] wrote', tdir, 'and', FLAGS.out_json)


if __name__ == '__main__':
    app.run(main)
