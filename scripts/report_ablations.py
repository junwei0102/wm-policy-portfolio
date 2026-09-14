"""Ablation and control tables for the WMPP paper revision (reviewer W1-W3, W5, W9, W10).

Sections (each skipped gracefully while its runs are still pending):
  E1  scoring ablations at the selected cell   tag og50abl    score{k}_commit{k}_agg{last,mean} / _ens{min,lcb}
  E3  stall-restart controller (no model)      tag og50abl    stall_w{5,10,25,50}
  E2  true-simulator rollouts                  tag og50sim    sim_score{k}_commit{k} (+ sim_score1_commit1)
  E4  WM search without a portfolio            tags og50mpc{k}/og50pmpc{k}  (mpc32_s0.2_*, pmpc6_s0.2_*)
  E8  bank composition                         tags og50bank_{lbo,top2,dup} + bankseed_og50bank_seed
  E9  leakage-free horizon selection           og50_final_sweep{,_loo,_global}.json

Every row follows the official OGBench protocol (5 tasks x 50 episodes x 3 bank seeds) on the
shared episode seeds; contrasts are paired per episode with the hierarchical bootstrap
(bank seeds, then episodes within seed).

Outputs under --out_dir/tables: ablations.tex, stall.tex, search.tex, bank_composition.tex,
selection_rules.tex, numbers_ablations.tex (macros); full JSON at --out_json; markdown to stdout.
"""
import json
import os
import re
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import (FAMILIES, NAME, contrast, family, fmt, fmt_ci, fmt_pm_ci, hier_boot, macro_key,  # noqa: E402
                          near_top, paired_rows, rows_by_policy, rows_hw, seed_mean, tex_env)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'test-rule report_og50 JSON.')
flags.DEFINE_string('report_test', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep_test.json', 'per-dataset test-selected report (post-hoc reference).')
flags.DEFINE_string('report_loo', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep_loo.json', 'loo_family report.')
flags.DEFINE_string('report_global', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep_global.json', 'global-k report.')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('lavl_family_k', 'maze:1,cube:5,scene:10,puzzle:100', 'Metric-value interval per family; used as the ablation reference cell for families that report the direct value.')
flags.DEFINE_string('main_tags', 'og50,og50k5,og50r1,og50cr', 'Dir tags holding the official-protocol sweep runs.')
flags.DEFINE_string('fixed_tag', 'og50fx', 'Dir tag with every fixed bank policy on the same episode seeds.')
flags.DEFINE_string('abl_tag', 'og50abl,og50abl2,og50abl3,og50abl4', 'Comma list of dir tags of the E1/E3 ablation runs (later tags override).')
flags.DEFINE_string('sim_tag', 'og50sim,og50sim1,og50sim3,og50sim4', 'Comma list of dir tags of the true-simulator rollout runs.')
flags.DEFINE_string('simor_tag', 'og50simor', 'Dir tag of the dynamic simulator oracle runs (sim_oracle_commit25).')
flags.DEFINE_string('cls_tag', 'og50dscls,og50dscls2,og50dscls3', 'Comma list of dir tags of the learned gating-selector runs (cls-k*-c*).')
flags.DEFINE_string('bankx_tag', 'og50bankx,og50bankxf,og50bankxf2', 'Comma list of dir tags of the pooled play+noisy bank runs (bankx_sd<s> / bankxfx_sd<s> labels).')
flags.DEFINE_integer('n_boot', 10000, 'Bootstrap resamples.')
flags.DEFINE_string('envs', 'all', 'Comma-separated env subset, or "all".')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), 'Paper root (tables/ inside).')
flags.DEFINE_string('out_json', '/scratch/jwquan/wmpp/planner_eval/ablations_report.json', 'Full JSON output.')

SEEDS = (0, 1, 2)
ABLVARS = [('AggLast', '_agglast'), ('AggMean', '_aggmean'), ('EnsMin', '_ensmin'), ('EnsLcb', '_enslcb')]
STALLS = [(5, 'StallFive'), (10, 'StallTen'), (25, 'StallTwentyFive'), (50, 'StallFifty')]
BANKS = [('full', 'Full bank'), ('lbo', 'Leave-best-out'), ('top2', 'Top-2'), ('dup', 'Best duplicated'), ('seed', 'Seed bank (best algo $\\times$ 3)'),
         ('pooled', 'Pooled play+noisy')]


def merged_rows(env_dir, seed, tags):
    out = {}
    for t in tags:
        for kk, vv in rows_by_policy(env_dir, seed, t).items():
            out.setdefault(kk, vv)
    return out


def all_seeds(fn):
    """{seed: rows} if fn(seed) yields rows for every seed, else None (runs pending)."""
    out = {}
    for s in SEEDS:
        r = fn(s)
        if not r:
            return None
        out[s] = r
    return out


def cell(entry, top=None):
    """Table cell: mean +- 95% half-width (bank seeds, then episodes); bold at >= 95% of the row max `top`."""
    if entry is None:
        return '--'
    v = 100 * entry['mean']
    bold = top is not None and near_top(v, top)
    return fmt_pm_ci(v, entry['hw'], bold=bold) if entry.get('hw') is not None else fmt(v, bold=bold)


def row_top(*entries):
    """Row maximum over present entries (dicts with 'mean' or plain fractions)."""
    vals = [100 * (e['mean'] if isinstance(e, dict) else e) for e in entries if e is not None]
    return max(vals) if vals else 0.0


def md_cell(entry):
    if entry is None:
        return '--'
    if entry.get('same_as_wmpp'):
        return f"{100 * entry['mean']:.0f} (=WMPP)"
    s = f"{100 * entry['mean']:.0f}"
    if entry.get('d_wmpp'):
        s += f" ({fmt_ci(entry['d_wmpp'])})"
    return s



def _member_name(bn):
    """'gciql' -> 'GCIQL'; 'gciql-noisy' (a member pooled in from the sibling dataset) -> 'GCIQL (noisy)'."""
    if bn in NAME:
        return NAME[bn]
    algo, _, suf = str(bn).partition('-')
    return f'{NAME.get(algo, algo)} ({suf})' if suf else str(bn)

def main(_):
    rng = np.random.default_rng(0)
    report = {x['env_name']: x for x in json.load(open(FLAGS.report))}
    reps = {}
    for nm, path in (('test', FLAGS.report_test), ('loo', FLAGS.report_loo), ('glob', FLAGS.report_global)):
        reps[nm] = {x['env_name']: x for x in json.load(open(path))} if os.path.exists(path) else {}
    envs = [e for _, es in FAMILIES for e in es if e in report]
    if FLAGS.envs != 'all':
        keep = set(FLAGS.envs.split(','))
        envs = [e for e in envs if e in keep]
    tags = FLAGS.main_tags.split(',')
    tdir = os.path.join(FLAGS.out_dir, 'tables')
    os.makedirs(tdir, exist_ok=True)

    R = {}
    for env in envs:
        x = report[env]
        k = x['selected_k']
        env_dir = os.path.join(FLAGS.eval_root, env)
        wvar = x['methods']['WMPP']['variant']
        rvar = x['methods']['Random']['variant']
        pref = 'critic' if (x.get('validation') or {}).get('scorer') == 'critic' else 'score'  # variant prefix of the reported scorer
        best = x['best_policy']
        main_rows = {s: merged_rows(env_dir, s, tags) for s in SEEDS}
        fx = {s: rows_by_policy(env_dir, s, FLAGS.fixed_tag) for s in SEEDS}
        W = {s: main_rows[s][wvar] for s in SEEDS}
        B = {s: fx[s].get(f'{best}-sd{s}') for s in SEEDS}
        have_fx = all(B.values())
        RND = {s: main_rows[s].get(rvar) for s in SEEDS}
        r = dict(k=k, best=best, wmpp=seed_mean(W), wmpp_hw=rows_hw(W),
                 best_sr=seed_mean(B) if have_fx else None, best_hw=rows_hw(B) if have_fx else None,
                 rand_hw=rows_hw(RND) if all(RND.values()) else None,
                 rand=x['methods']['Random']['success'],
                 rand_c={int(kk.replace('random_commit', '')): vv for kk, vv in x['random_by_commit'].items()},
                 rows={}, banks={})

        def complete(rows_by_seed):
            """Rows exist for every seed with the full episode set (runs that flush incrementally leave partial CSVs)."""
            return rows_by_seed is not None and all(len(rows_by_seed[s]) == len(W[s]) for s in SEEDS)

        def add(name, rows_by_seed, vs_w=True):
            if not complete(rows_by_seed):
                if rows_by_seed is not None:
                    print(f'[abl] {env}: {name} incomplete ({[len(rows_by_seed[s]) for s in SEEDS]} rows), treated as pending')
                r['rows'][name] = None
                return
            e = dict(mean=seed_mean(rows_by_seed), hw=rows_hw(rows_by_seed))
            if vs_w:
                e['d_wmpp'] = contrast(rows_by_seed, W, FLAGS.n_boot, rng)
            if have_fx:
                e['d_best'] = contrast(rows_by_seed, B, FLAGS.n_boot, rng)
            r['rows'][name] = e

        abl = {}
        for s in SEEDS:  # several ablation tags may coexist (re-runs at a re-selected cell use og50abl2)
            abl[s] = {}
            for t in FLAGS.abl_tag.split(','):
                abl[s].update(rows_by_policy(env_dir, s, t))
        for name, suf in ABLVARS:
            add(name, all_seeds(lambda s: abl[s].get(f'{pref}{k}_commit{k}{suf}')))
        if k == 1:
            r['rows']['OneStep'] = dict(mean=r['wmpp'], hw=r['wmpp_hw'], same_as_wmpp=True)
        else:
            add('OneStep', all_seeds(lambda s: main_rows[s].get(f'{pref}1_commit1')))  # same scorer as the reported cell
        for m, name in STALLS:
            add(name, all_seeds(lambda s, m=m: abl[s].get(f'stall_w{m}')))
        # model-free Q-select control (puzzle family): the direct value's own twin-Q critic ranks proposed actions, no rollout
        add('QselOne', all_seeds(lambda s: abl[s].get('qsel_commit1')))
        add('QselSel', all_seeds(lambda s: abl[s].get(f'qsel_commit{k}')))
        sim = {}
        for s in SEEDS:  # true-sim rows may be split over tags (og50sim = selected cell, og50sim1 = one-step re-runs)
            sim[s] = {}
            for t in FLAGS.sim_tag.split(','):
                sim[s].update(rows_by_policy(env_dir, s, t))
        add('SimSel', all_seeds(lambda s: sim[s].get(f'sim_{pref}{k}_commit{k}')))
        if k != 1:
            add('SimOne', all_seeds(lambda s: sim[s].get('sim_score1_commit1')))

        def mpc_rows(prefix):
            cands = [f'og50{prefix}{k}', f'og50{prefix}']
            cands += ['og50mpcf', 'og50mpckc'] if prefix == 'mpc' else ['og50pmpc1']  # og50mpcf: family cell, sigma 0.2, all datasets
            for sig in ('0\\.2', '0\\.05'):  # (1,1) runs used sigma=0.05
                pat = re.compile(prefix + r'\d+_s%s_%s%d_commit%d$' % (sig, pref, k, k))
                for t in cands:
                    got = all_seeds(lambda s, t=t: next(
                        (vv for nm2, vv in rows_by_policy(env_dir, s, t).items() if pat.fullmatch(nm2)), None))
                    if got:
                        return got
            return None
        add('PolicyMpc', mpc_rows('mpc'))
        add('PortfolioMpc', mpc_rows('pmpc'))
        # dynamic simulator oracle (privileged) and WMPP at the matched (25,25) cell
        add('SimOracle', all_seeds(lambda s: rows_by_policy(env_dir, s, FLAGS.simor_tag).get('sim_oracle_commit25')))
        if r['rows'].get('SimOracle'):
            add('WmppTwentyFive', all_seeds(lambda s: main_rows[s].get('score25_commit25')))
        # learned gating selector: best of the trained cls-k*-c* variants (post-hoc, favourable to the baseline)
        cls = {}
        for s in SEEDS:
            cls[s] = {}
            for t in FLAGS.cls_tag.split(','):
                cls[s].update(rows_by_policy(env_dir, s, t))
        cls_vars = sorted(set.intersection(*[{re.sub(r'-sd\d+$', '', v) for v in cls[s] if v.startswith('cls-')} for s in SEEDS])) if all(cls.values()) else []
        best_cls, best_mean = None, -1.0
        for cv in cls_vars:
            rows_c = all_seeds(lambda s, cv=cv: cls[s].get(f'{cv}-sd{s}'))
            if rows_c and seed_mean(rows_c) > best_mean:
                best_cls, best_mean = cv, seed_mean(rows_c)
        if best_cls:
            add('Selector', all_seeds(lambda s: cls[s].get(f'{best_cls}-sd{s}')))
            r['rows']['Selector']['variant'] = best_cls

        # E8 bank composition
        fams = sorted(x['fixed_test_success'], key=x['fixed_test_success'].get, reverse=True)
        second = fams[1] if len(fams) > 1 else None
        for kind, bbest, P in (('lbo', second, len(fams) - 1), ('top2', best, 2), ('dup', best, len(fams) + 1)):
            rows_w = all_seeds(lambda s, kind=kind: rows_by_policy(env_dir, s, f'og50bank_{kind}').get(f'score{k}_commit{k}'))
            if not complete(rows_w):
                r['banks'][kind] = None
                continue
            rows_r = all_seeds(lambda s, kind=kind: rows_by_policy(env_dir, s, f'og50bank_{kind}').get(rvar))
            bb = {s: fx[s].get(f'{bbest}-sd{s}') for s in SEEDS}
            e = dict(P=P, best_name=bbest, wmpp=seed_mean(rows_w), wmpp_hw=rows_hw(rows_w),
                     rand=seed_mean(rows_r) if rows_r else None, rand_hw=rows_hw(rows_r) if rows_r else None,
                     best_sr=seed_mean(bb) if all(bb.values()) else None, best_hw=rows_hw(bb) if all(bb.values()) else None)
            if all(bb.values()):
                e['d_best'] = contrast(rows_w, bb, FLAGS.n_boot, rng)
            e['d_full'] = contrast(rows_w, W, FLAGS.n_boot, rng)
            r['banks'][kind] = e
        # pooled play+noisy bank: WMPP/Random from bankx_sd<s>, extra members' fixed rows from bankxfx_sd<s>
        def bankx_rows(s, label):
            out = {}
            for t in FLAGS.bankx_tag.split(','):
                out.update(rows_by_policy(env_dir, None, t, label=label))
            return out
        px = all_seeds(lambda s: bankx_rows(s, f'bankx_sd{s}').get(f'{pref}{k}_commit{k}'))  # pooled bank at the reported scorer's cell
        if complete(px):
            prand = all_seeds(lambda s: bankx_rows(s, f'bankx_sd{s}').get(rvar))
            fam_rows = {}
            for s in SEEDS:
                for src in (fx[s], bankx_rows(s, f'bankxfx_sd{s}')):
                    for pol, rows_p in src.items():
                        if pol.startswith(('score', 'random', 'sim', 'mpc', 'pmpc', 'stall', 'cls')):
                            continue
                        fam_rows.setdefault(family(pol), {})[s] = rows_p
            fam_rows = {f: v for f, v in fam_rows.items() if len(v) == len(SEEDS) and complete(v)}
            pbest = max(fam_rows, key=lambda f: (seed_mean(fam_rows[f]), f))
            e = dict(P=len(fam_rows) if any('-noisy-' in f or '-play-' in f for f in fam_rows) else len(fam_rows), best_name=pbest,
                     wmpp=seed_mean(px), rand=seed_mean(prand) if prand else None, best_sr=seed_mean(fam_rows[pbest]),
                     wmpp_hw=rows_hw(px), rand_hw=rows_hw(prand) if prand else None, best_hw=rows_hw(fam_rows[pbest]),
                     d_best=contrast(px, fam_rows[pbest], FLAGS.n_boot, rng), d_full=contrast(px, W, FLAGS.n_boot, rng),
                     members=sorted(fam_rows))
            r['banks']['pooled'] = e
        else:
            r['banks']['pooled'] = None
        sb = rows_by_policy(env_dir, None, 'og50bank_seed', label='bankseed')
        rows_w = sb.get(f'score{k}_commit{k}')
        if rows_w and have_fx and len(rows_w) == len(W[SEEDS[0]]):
            per_seed = {s: paired_rows(rows_w, B[s]) for s in SEEDS}
            rr = sb.get(rvar)
            r['banks']['seed'] = dict(
                P=3, best_name=best, wmpp=float(np.mean([float(q['success']) for q in rows_w])),
                rand=float(np.mean([float(q['success']) for q in rr])) if rr else None,
                best_sr=r['best_sr'], d_best=hier_boot(per_seed, FLAGS.n_boot, rng))
        else:
            r['banks']['seed'] = None
        R[env] = r
        done = sorted(nm for nm, v in r['rows'].items() if v)
        print(f'[abl] {env}: k={k} present={done} banks={sorted(nm for nm, v in r["banks"].items() if v)}')

    # E9 selection rules
    sel = {}
    for env in envs:
        e = {'test': (report[env]['selected_k'], report[env]['methods']['WMPP']['success'],
                      report[env]['contrasts']['WMPP_vs_best_policy'])}
        for nm in ('test', 'loo', 'glob'):
            if env in reps[nm]:
                y = reps[nm][env]
                e[nm] = (y['selected_k'], y['methods']['WMPP']['success'], y['contrasts']['WMPP_vs_best_policy'])
        sel[env] = e

    macros = {}

    def best_cell(r, top):
        if r['best_sr'] is None:
            return '--'
        v = 100 * r['best_sr']
        return fmt_pm_ci(v, r['best_hw'], bold=near_top(v, top)) if r.get('best_hw') is not None else fmt(v, bold=near_top(v, top))

    def wmpp_cell(r, top):
        v = 100 * r['wmpp']
        return fmt_pm_ci(v, r['wmpp_hw'], bold=near_top(v, top))

    # ---- ablations.tex (E1 + one-step + sim) -----------------------------------
    cols = ['AggLast', 'AggMean', 'EnsMin', 'EnsLcb']  # the (1,1) chooser is a (k,c) setting, not a baseline: kept in macros only (user, 2026-09-11)
    lines = []
    for env in envs:
        r = R[env]
        top = row_top(r['best_sr'], r['wmpp'], *[r['rows'].get(c) for c in cols])
        lines.append(' & '.join(
            [tex_env(env), f'$({r["k"]},{r["k"]})$', best_cell(r, top), wmpp_cell(r, top)]
            + [cell(r['rows'].get(c), top) for c in cols]) + ' \\\\')
    for c in cols + ['OneStep', 'PolicyMpc', 'PortfolioMpc', 'Selector', 'SimSel', 'SimOne', 'SimOracle', 'QselOne', 'QselSel']:
        pres = [e for e in envs if R[e]['rows'].get(c) and not R[e]['rows'][c].get('same_as_wmpp')]  # a (1,1) row on a k=1 dataset IS WMPP: no contrast
        vals = [100 * R[e]['rows'][c]['mean'] for e in pres]
        deltas = [100 * R[e]['rows'][c]['d_wmpp']['delta'] for e in pres if 'd_wmpp' in R[e]['rows'][c]]
        macros[f'AblMean{c}'] = f'{np.mean(vals):.0f}' if vals else '--'
        macros[f'AblMeanDelta{c}'] = f'{np.mean(deltas):+.0f}' if deltas else '--'
        macros[f'AblNumDown{c}'] = str(sum(1 for e in pres if R[e]['rows'][c].get('d_wmpp', {}).get('significant') and R[e]['rows'][c]['d_wmpp']['delta'] < 0))
        macros[f'AblNumUp{c}'] = str(sum(1 for e in pres if R[e]['rows'][c].get('d_wmpp', {}).get('significant') and R[e]['rows'][c]['d_wmpp']['delta'] > 0))
        macros[f'AblNumEnvs{c}'] = str(len(pres))
        macros[f'AblNumBelowBest{c}'] = str(sum(1 for e in pres if R[e]['rows'][c].get('d_best', {}).get('significant') and R[e]['rows'][c]['d_best']['delta'] < 0))
        macros[f'AblNumAboveBest{c}'] = str(sum(1 for e in pres if R[e]['rows'][c].get('d_best', {}).get('significant') and R[e]['rows'][c]['d_best']['delta'] > 0))
        for e in pres:
            macros[f'Abl{c}{macro_key(e)}'] = f"{100 * R[e]['rows'][c]['mean']:.0f}"
            if 'd_wmpp' in R[e]['rows'][c]:
                macros[f'AblDelta{c}{macro_key(e)}'] = fmt_ci(R[e]['rows'][c]['d_wmpp'])
    pres_all = [e for e in envs if all(R[e]['rows'].get(c) for c in cols[:4])]
    macros['AblMeanWmpp'] = f'{np.mean([100 * R[e]["wmpp"] for e in pres_all]):.0f}' if pres_all else '--'
    macros['AblNumEnvs'] = str(len(pres_all))
    with open(os.path.join(tdir, 'ablations.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc|cccc}\n\\toprule\n'
                ' & & & & \\multicolumn{2}{c}{horizon agg.} & \\multicolumn{2}{c}{ensemble agg.} \\\\\n'
                '\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n'
                'Dataset & $(k,c)$ & Best & \\wmpp & last & mean & min & LCB \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- stall.tex (E3) --------------------------------------------------------
    lines = []
    n_stall_beats, best_stall_gaps = 0, []
    for env in envs:
        r = R[env]
        rc = {c: v for c, v in r['rand_c'].items() if v == v}
        cbest = max(rc, key=rc.get) if rc else None
        stall_means = {nm: r['rows'][nm]['mean'] for _, nm in STALLS if r['rows'].get(nm)}
        if stall_means:
            sbest = max(stall_means.values())
            best_stall_gaps.append(100 * (sbest - r['wmpp']))
            if sbest > r['wmpp'] + 1e-9:
                n_stall_beats += 1
            macros[f'StallBest{macro_key(env)}'] = f'{100 * sbest:.0f}'
        top = row_top(r['best_sr'], r['wmpp'], *[r['rows'].get(nm) for _, nm in STALLS])
        lines.append(' & '.join(
            [tex_env(env), best_cell(r, top)]
            + [cell(r['rows'].get(nm), top) for _, nm in STALLS]
            + [wmpp_cell(r, top)]) + ' \\\\')
    macros['NumStallBeatsWmpp'] = str(n_stall_beats)
    macros['MeanBestStallMinusWmpp'] = f'{np.mean(best_stall_gaps):+.0f}' if best_stall_gaps else '--'
    macros['NumEnvsStall'] = str(len(best_stall_gaps))
    with open(os.path.join(tdir, 'stall.tex'), 'w') as f:
        f.write('\\begin{tabular}{lc|cccc|c}\n\\toprule\n'
                ' & & \\multicolumn{4}{c|}{stall-restart, window $m$} & \\\\\n'
                'Dataset & Best & $m{=}5$ & $10$ & $25$ & $50$ & \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- search.tex (E2 + E4) --------------------------------------------------
    scols = ['PolicyMpc']  # the paper's Table: WM action search on the best policy vs arbitration (other controls dropped)
    senvs = [e for e in envs if any(R[e]['rows'].get(c) for c in scols)]
    lines = []
    for env in senvs:
        r = R[env]
        top = row_top(r['best_sr'], r['wmpp'], *[r['rows'].get(c) for c in scols])
        lines.append(' & '.join(
            [tex_env(env), f'$({r["k"]},{r["k"]})$', best_cell(r, top)]
            + [cell(r['rows'].get(c), top) for c in scols]
            + [wmpp_cell(r, top)]) + ' \\\\')
        for c in scols + ['WmppTwentyFive']:
            if R[env]['rows'].get(c):
                macros[f'Abl{c}{macro_key(env)}'] = f"{100 * R[env]['rows'][c]['mean']:.0f}"
                if 'd_wmpp' in R[env]['rows'][c]:
                    macros[f'AblDelta{c}{macro_key(env)}'] = fmt_ci(R[env]['rows'][c]['d_wmpp'])
    with open(os.path.join(tdir, 'search.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc|c}\n\\toprule\n'
                'Dataset & $(k,c)$ & Best & WM action search on Best & \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    # true-simulator rollouts on every dataset (W9)
    lines = []
    for env in envs:
        r = R[env]
        e = r['rows'].get('SimSel')
        if not e:
            continue
        top = row_top(r['best_sr'], r['wmpp'], e)
        lines.append(' & '.join([tex_env(env), f'$({r["k"]},{r["k"]})$', best_cell(r, top),
                                 wmpp_cell(r, top), cell(e, top), fmt_ci(e['d_wmpp'])]) + ' \\\\')
    with open(os.path.join(tdir, 'sim_all.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccc r}\n\\toprule\n'
                'Dataset & $(k,c)$ & Best & \\wmpp & true-sim.\\ branches & $\\Delta$ (sim $-$ \\wmpp) \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # model-free Q-select control (puzzle family)
    lines = []
    for env in envs:
        r = R[env]
        e = r['rows'].get('QselSel')
        if not e:
            continue
        top = row_top(r['best_sr'], r['wmpp'], e)  # the c=1 variant (QselOne) stays in the JSON/macros only
        lines.append(' & '.join([tex_env(env), f'$({r["k"]},{r["k"]})$', best_cell(r, top), wmpp_cell(r, top),
                                 cell(e, top), fmt_ci(e['d_wmpp'])]) + ' \\\\')
    with open(os.path.join(tdir, 'qsel.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc|c r}\n\\toprule\n'
                'Dataset & $(k,c)$ & Best & \\wmpp & Q-select & $\\Delta$ (Q-select $-$ \\wmpp) \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- bank_composition.tex (E8) ---------------------------------------------
    benvs = [e for e in envs if any(R[e]['banks'].get(kk) for kk, _ in BANKS if kk != 'full')]
    lines = []
    for env in benvs:
        r = R[env]
        fullP = len(report[env]['fixed_test_success'])
        blocks = [('full', dict(P=fullP, best_name=r['best'], wmpp=r['wmpp'], rand=r['rand'], best_sr=r['best_sr'],
                                wmpp_hw=r['wmpp_hw'], rand_hw=r['rand_hw'], best_hw=r['best_hw']))]
        blocks += [(kk, r['banks'].get(kk)) for kk, _ in BANKS if kk != 'full']
        first = True
        for kk, b in blocks:
            label = dict(BANKS)[kk]
            if b is None:
                continue
            dcell = fmt_ci(b['d_best']) if 'd_best' in b else ('--' if kk != 'full' else '')
            bn = b.get('best_name')
            top = row_top(b.get('best_sr'), b.get('rand'), b['wmpp'])
            def pm(v, hwk):
                v = 100 * v
                return fmt_pm_ci(v, b[hwk], bold=near_top(v, top)) if b.get(hwk) is not None else fmt(v, bold=near_top(v, top))
            lines.append(' & '.join([
                f'\\multirow{{{sum(1 for _, bb in blocks if bb)}}}{{*}}{{{tex_env(env)}}}' if first else '',
                label, str(b['P']),
                ('--' if b.get('best_sr') is None else f"{_member_name(bn)} ({pm(b['best_sr'], 'best_hw')})"),
                '--' if b.get('rand') is None else pm(b['rand'], 'rand_hw'),
                pm(b['wmpp'], 'wmpp_hw'), dcell,
                fmt_ci(b['d_full']) if 'd_full' in b else '']) + ' \\\\')
            first = False
            key = {'lbo': 'Lbo', 'top2': 'TopTwo', 'dup': 'Dup', 'seed': 'Seed', 'full': 'Full', 'pooled': 'Pooled'}[kk]
            macros[f'Bank{key}{macro_key(env)}'] = f"{100 * b['wmpp']:.0f}"
            if b.get('best_sr') is not None:
                macros[f'Bank{key}Best{macro_key(env)}'] = f"{100 * b['best_sr']:.0f}"
            if 'd_best' in b:
                macros[f'BankDelta{key}{macro_key(env)}'] = fmt_ci(b['d_best'])
            if 'd_full' in b:
                macros[f'BankDeltaFull{key}{macro_key(env)}'] = fmt_ci(b['d_full'])
        lines.append('\\midrule')
    if lines and lines[-1] == '\\midrule':
        lines = lines[:-1]
    with open(os.path.join(tdir, 'bank_composition.tex'), 'w') as f:
        f.write('\\begin{tabular}{llccccrr}\n\\toprule\n'
                'Dataset & Bank & $P$ & Best member & Random & \\wmpp & $\\Delta$ vs best member & $\\Delta$ vs six-member \\wmpp \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- selection_rules.tex (E9) ----------------------------------------------
    lines = []
    main_rule = next(iter({report[e].get('select_rule', 'test') for e in envs}), 'test')
    main_is_val = main_rule == 'validation'
    selcols = [('main', 'validation' if main_is_val else 'per-dataset (test)')]
    selcols += [('test', 'per-dataset test (post hoc)')] if main_is_val and reps['test'] else []
    selcols += [('loo', 'leave-one-out family'), ('glob', 'single global $k$')]
    for env in envs:
        sel[env]['main'] = sel[env].pop('test') if not main_is_val else sel[env]['main'] if 'main' in sel[env] else None
    for env in envs:
        e = sel[env]
        if main_is_val:
            e['main'] = (report[env]['selected_k'], report[env]['methods']['WMPP']['success'], report[env]['contrasts']['WMPP_vs_best_policy'])
    have_all = [e for e in envs if all(c in sel[e] and sel[e][c] for c, _ in selcols)]
    for env in envs:
        e = sel[env]
        cells = [tex_env(env), '--' if R[env]['best_sr'] is None else fmt(100 * R[env]['best_sr'])]
        if main_is_val:
            cells.append(str(report[env].get('selected_c', '--')))
        for nm, _ in selcols:
            if e.get(nm):
                kk, sr, d = e[nm]
                star = '$^{*}$' if d['significant'] and d['delta'] > 0 else ('$^{\\downarrow}$' if d['significant'] else '')
                cells += [str(kk), fmt(100 * sr, bold=(nm == 'main')) + star]
            else:
                cells += ['--', '--']
        lines.append(' & '.join(cells) + ' \\\\')
    for nm, mac in (('main', 'Val' if main_is_val else 'Test'), ('test', 'Test'), ('loo', 'Loo'), ('glob', 'Global')):
        if nm == 'test' and not main_is_val:
            continue
        vals = [100 * sel[e][nm][1] for e in have_all if sel[e].get(nm)]
        macros[f'MeanWmpp{mac}'] = f'{np.mean(vals):.0f}' if vals else '--'
        macros[f'NumSigUp{mac}'] = str(sum(1 for e in have_all if sel[e].get(nm) and sel[e][nm][2]['significant'] and sel[e][nm][2]['delta'] > 0))
        macros[f'NumSigDown{mac}'] = str(sum(1 for e in have_all if sel[e].get(nm) and sel[e][nm][2]['significant'] and sel[e][nm][2]['delta'] < 0))
        if nm != 'main':
            macros[f'NumKChanged{mac}'] = str(sum(1 for e in have_all if sel[e].get(nm) and sel[e][nm][0] != sel[e]['main'][0]))
    if main_is_val:
        macros['NumCneqKVal'] = str(sum(1 for e in envs if report[e].get('selected_c') != report[e]['selected_k']))
        macros['MeanRandVal'] = f"{np.mean([100 * report[e]['methods']['Random']['success'] for e in envs]):.0f}"
    macros['NumEnvsSel'] = str(len(have_all))
    head = ' & & ' + ('$c^{*}$ & ' if main_is_val else '') + ' & '.join(f'\\multicolumn{{2}}{{c{"|" if i < len(selcols) - 1 else ""}}}{{{lab}}}' for i, (_, lab) in enumerate(selcols)) + ' \\\\\n'
    sub = 'Dataset & Best & ' + ('Random & ' if main_is_val else '') + ' & '.join('$k$ & \\wmpp' for _ in selcols) + ' \\\\\n'
    with open(os.path.join(tdir, 'selection_rules.tex'), 'w') as f:
        f.write('\\begin{tabular}{lc' + ('c' if main_is_val else '') + '|' + '|'.join('cc' for _ in selcols) + '}\n\\toprule\n'
                + head + sub + '\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    R5 = ['ScenePlay', 'CubeDoublePlay', 'PuzzleFourbyFourNoisy', 'AntmazeLargeNavigate', 'CubeSinglePlay']
    expected = [f'{p}{k}' for p in ('AblSelector', 'AblSimOracle', 'AblDeltaSelector', 'AblDeltaSimOracle', 'AblSimSel', 'AblSimOne', 'AblDeltaSimSel', 'AblWmppTwentyFive') for k in R5]
    expected += [f'{p}{c}' for p in ('AblMean', 'AblMeanDelta', 'AblNumDown', 'AblNumUp', 'AblNumEnvs')
                 for c in ('Selector', 'SimOracle', 'SimSel', 'PolicyMpc', 'PortfolioMpc', 'QselOne', 'QselSel')]
    expected += [f'{p}{c}' for p in ('AblNumBelowBest', 'AblNumAboveBest') for c in ('QselOne', 'QselSel')]
    expected += [f'{p}{k}' for p in ('AblQselSel', 'AblQselOne', 'AblDeltaQselSel', 'AblDeltaQselOne')
                 for k in ('PuzzleThreebyThreePlay', 'PuzzleFourbyFourPlay', 'PuzzleFourbyFourNoisy')]
    expected += [f'{p}{k}' for p in ('BankPooled', 'BankPooledDelta', 'BankPooledBest') for k in ('ScenePlay', 'CubeDoublePlay', 'PuzzleFourbyFourPlay')]
    expected += ['AblDeltaAggMeanPuzzleThreebyThreeNoisy', 'AblEnsLcbCubeSinglePlay', 'AblDeltaEnsLcbCubeSinglePlay',
                 'AblOneStepPuzzleThreebyThreeNoisy', 'AblDeltaSimSelAntmazeLargeNavigate', 'AblSimOnePuzzleFourbyFourNoisy',
                 'AblSimSelPuzzleFourbyFourNoisy', 'AblPolicyMpcScenePlay', 'AblDeltaPolicyMpcScenePlay', 'AblPortfolioMpcPuzzleFourbyFourNoisy']
    expected += ['MeanWmppVal', 'MeanWmppTest', 'MeanWmppLoo', 'MeanWmppGlobal', 'NumSigUpVal', 'NumSigUpTest', 'NumSigUpLoo', 'NumSigUpGlobal',
                 'NumKChangedTest', 'NumKChangedLoo', 'NumKChangedGlobal', 'NumCneqKVal', 'MeanRandVal', 'NumEnvsSel']
    for nm in expected:
        macros.setdefault(nm, '--')
    with open(os.path.join(tdir, 'numbers_ablations.tex'), 'w') as f:
        f.write('% Auto-generated by scripts/report_ablations.py -- do not edit.\n')
        for kk, vv in macros.items():
            f.write(f'\\newcommand{{\\{kk}}}{{{vv}}}\n')

    def clean(o):
        if isinstance(o, dict):
            return {kk: clean(vv) for kk, vv in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(vv) for vv in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(FLAGS.out_json, 'w') as f:
        json.dump(dict(rows={e: clean(R[e]) for e in envs}, selection=clean(sel), macros=macros), f, indent=1)

    # ---- markdown --------------------------------------------------------------
    print('\n## E1/E2/E3/E4 ablations (mean success %, paired delta vs WMPP in parentheses)')
    print('| dataset | (k,c) | best | WMPP | ' + ' | '.join(cols + scols + [nm for _, nm in STALLS]) + ' |')
    print('|' + '---|' * (4 + len(cols) + len(scols) + len(STALLS)))
    for env in envs:
        r = R[env]
        print(f'| {env} | ({r["k"]},{r["k"]}) | ' +
              ('--' if r['best_sr'] is None else f'{100 * r["best_sr"]:.0f}') + f' | **{100 * r["wmpp"]:.0f}** | ' +
              ' | '.join(md_cell(r['rows'].get(c)) for c in cols + scols + [nm for _, nm in STALLS]) + ' |')
    print('\n## E8 bank composition')
    for env in benvs:
        for kk, _ in BANKS:
            b = R[env]['banks'].get(kk) if kk != 'full' else None
            if b:
                rnd = '--' if b.get('rand') is None else f"{100 * b['rand']:.0f}"
                bsr = '--' if b.get('best_sr') is None else f"{100 * b['best_sr']:.0f}"
                dbc = fmt_ci(b['d_best']) if 'd_best' in b else '--'
                print(f'| {env} | {kk} | P={b["P"]} | best {b["best_name"]} {bsr} | rand {rnd} | WMPP {100 * b["wmpp"]:.0f} | {dbc} |')
    print('\n## E9 selection rules')
    for env in envs:
        e = sel[env]
        row = [env]
        for nm in ('test', 'loo', 'glob'):
            row.append(f'{nm}: k={e[nm][0]} {100 * e[nm][1]:.0f} ({fmt_ci(e[nm][2])})' if nm in e else f'{nm}: --')
        print('| ' + ' | '.join(row) + ' |')
    print('\n[abl] macros:', json.dumps(macros, indent=0)[:2000])
    print('[abl] wrote', tdir, 'and', FLAGS.out_json)


if __name__ == '__main__':
    app.run(main)
