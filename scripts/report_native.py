"""Native-critic comparison on the three-member bank {GCIQL, GCIVL, HIQL} (tag og50nat).

Rows per dataset, all on the same bank and episodes (official protocol, 3 bank seeds):
  Best-of-3        best of the three fixed members (three-seed mean, og50fx rows)
  NativeSel(c)     model-free, i* = argmax_i V_i(s, g) with each member's OWN value network
  Q-select(c)      model-free, GCIQL's twin-Q ranks the three proposals (shared critic)
  WMPA-native      rollouts scored by each member's OWN value network
  WMPA-shared      rollouts scored by the paper's shared head (metric; direct on the puzzles)
plus the paper's six-member WMPA for reference. Contrasts are paired per episode (hierarchical
bootstrap over bank seeds then episodes). Writes tables/native_critic.tex + tables/numbers_native.tex.

  python scripts/report_native.py
"""
import json
import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import FAMILIES, contrast, fmt_ci, hier_boot, macro_key, rows_by_policy, tex_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGS = flags.FLAGS
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', '')
flags.DEFINE_string('report', '/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json', 'report_og50 JSON (paper cell per dataset).')
flags.DEFINE_string('tag', 'og50nat,og50natb', 'comma list of dir tags of the three-member runs (later tags fill variants missing from earlier ones).')
flags.DEFINE_string('main_tags', 'og50,og50k5,og50r1,og50cr', 'tags holding the paper\'s six-member cell.')
flags.DEFINE_string('fixed_tag', 'og50fx', '')
flags.DEFINE_string('envs', 'all', '')
flags.DEFINE_integer('n_boot', 10000, '')
flags.DEFINE_string('out_dir', os.path.join(ROOT, 'WMPP_ICLR2027'), '"" = print only.')
flags.DEFINE_string('out_json', '/scratch/jwquan/wmpp/planner_eval/native_report.json', '')

SEEDS = (0, 1, 2)
MEMBERS = ('gciql', 'gcivl', 'hiql')


def all_seeds(fn):
    out = {}
    for s in SEEDS:
        r = fn(s)
        if not r or len(r) != 250:
            return None
        out[s] = r
    return out


def mean_hw(rb):
    per = {s: np.array([float(r['success']) for r in v]) for s, v in rb.items()}
    d = hier_boot(per, 2000, np.random.default_rng(7))
    return 100 * d['delta'], 100 * (d['ci_hi'] - d['ci_lo']) / 2


def main(_):
    rep = {e['env_name']: e for e in json.load(open(FLAGS.report))}
    envs = [e for _, es in FAMILIES for e in es] if FLAGS.envs == 'all' else FLAGS.envs.split(',')
    rng = np.random.default_rng(20260923)
    out, body, macros = {}, [], []
    tallies = {k: [] for k in ('nsel_k', 'qsel_k', 'native', 'shared')}
    for env in envs:
        r = rep[env]
        k = r['selected_k']
        pref = 'critic' if r['selected_variant'].startswith('critic') else 'score'
        env_dir = os.path.join(FLAGS.eval_root, env)
        nat = {}
        for s in SEEDS:
            nat[s] = {}
            for t in FLAGS.tag.split(','):
                for v, rr in rows_by_policy(env_dir, s, t).items():
                    if len(rr) == 250 and v not in nat[s]:
                        nat[s][v] = rr
        fx = {s: rows_by_policy(env_dir, s, FLAGS.fixed_tag) for s in SEEDS}
        # Best-of-3: the member with the highest three-seed mean among the three
        member_rows = {m: all_seeds(lambda s, m=m: fx[s].get(f'{m}-sd{s}')) for m in MEMBERS}
        member_rows = {m: v for m, v in member_rows.items() if v is not None}
        if len(member_rows) < 3:
            print(f'[native] {env}: fixed rows missing, skipped'); continue
        best_name = max(member_rows, key=lambda m: mean_hw(member_rows[m])[0])
        rows = {
            'best3': member_rows[best_name],
            'nsel_1': all_seeds(lambda s: nat[s].get('nsel_commit1')),
            'nsel_k': all_seeds(lambda s: nat[s].get(f'nsel_commit{k}')),
            'qsel_1': all_seeds(lambda s: nat[s].get('qsel_commit1')),
            'qsel_k': all_seeds(lambda s: nat[s].get(f'qsel_commit{k}')),
            'native': all_seeds(lambda s: nat[s].get(f'native{k}_commit{k}')),
            'shared': all_seeds(lambda s: nat[s].get(f'{pref}{k}_commit{k}')),
            'paper6': all_seeds(lambda s: next((rr[f'{pref}{k}_commit{k}'] for rr in (rows_by_policy(env_dir, s, t) for t in FLAGS.main_tags.split(','))
                                               if f'{pref}{k}_commit{k}' in rr), None)),
        }
        if rows['shared'] is None or rows['native'] is None:
            print(f'[native] {env}: three-member runs pending, skipped'); continue
        e = dict(k=k, head=pref, best3=best_name, cells={}, vs_shared={}, vs_best3={})
        for name, rb in rows.items():
            if rb is None:
                e['cells'][name] = None; continue
            m, hw = mean_hw(rb)
            e['cells'][name] = dict(mean=m, hw=hw)
            if name != 'shared':
                e['vs_shared'][name] = contrast(rb, rows['shared'], FLAGS.n_boot, rng)
            if name != 'best3':
                e['vs_best3'][name] = contrast(rb, rows['best3'], FLAGS.n_boot, rng)
        # how often the native comparison collapses onto one member: from the per-episode chains (episodes.csv,
        # available for every run), the share of episodes that never switch and the member picked first
        usage = {}
        for key_, var in (('native', f'native{k}_commit{k}'), ('nsel_k', f'nsel_commit{k}')):
            rb = rows.get(key_)
            if not rb:
                usage[var] = None; continue
            eps = [r for v in rb.values() for r in v]
            first = {}
            for r in eps:
                m = r['policy_chain'].split('>')[0].rsplit('-sd', 1)[0]
                first[m] = first.get(m, 0) + 1 / len(eps)
            usage[var] = dict(single_policy_episodes=float(np.mean([int(float(r['n_switches'])) == 0 for r in eps])), first_pick=first)
        e['usage'] = usage
        out[env] = e
        for key in tallies:
            c = e['vs_best3'].get(key)
            if c:
                tallies[key].append(c)
        c = e['cells']
        vs = lambda n: '--' if n not in e['vs_shared'] else fmt_ci(e['vs_shared'][n])
        cell = lambda n: '--' if c.get(n) is None else f"${c[n]['mean']:.0f} \\pm {c[n]['hw']:.0f}$"
        top = max(c[n]['mean'] for n in ('best3', 'native', 'shared') if c.get(n))
        bold = lambda n: '--' if c.get(n) is None else (f"$\\mathbf{{{c[n]['mean']:.0f} \\pm {c[n]['hw']:.0f}}}$" if c[n]['mean'] >= 0.95 * top - 1e-9 and top > 0 else cell(n))
        body.append(f"{tex_env(env)} & $({k},{k})$ & {bold('best3')} & {bold('native')} & {bold('shared')} & {vs('native')} \\\\")
        un = usage[f'native{k}_commit{k}']
        dom = max(un['first_pick'].values()) if un else float('nan')
        single = un['single_policy_episodes'] if un else float('nan')
        mv = lambda n: c[n]['mean'] if c.get(n) else float('nan')
        print(f"{env[:-3]:26s} ({k},{k}) best3={best_name:5s} {mv('best3'):5.1f} | nsel(1) {mv('nsel_1'):5.1f} nsel(k) {mv('nsel_k'):5.1f} | "
              f"qsel(1) {mv('qsel_1'):5.1f} qsel(k) {mv('qsel_k'):5.1f} | native {mv('native'):5.1f} shared {mv('shared'):5.1f} | "
              f"paper6 {mv('paper6'):5.1f} | native-shared {vs('native')}  nsel-shared {vs('nsel_k')} | native: first pick share {dom:.2f}, single-policy episodes {single:.2f}")
        mk = macro_key(env)
        macros += [f"\\newcommand{{\\NatNativeMinusShared{mk}}}{{{vs('native')}}}",
                   f"\\newcommand{{\\NatNselMinusShared{mk}}}{{{vs('nsel_k')}}}"]
    if not out:
        return
    for key, lab in (('nsel_k', 'NselK'), ('qsel_k', 'QselK'), ('native', 'Native'), ('shared', 'Shared')):
        cs = tallies[key]
        macros += [f"\\newcommand{{\\NatNumUp{lab}}}{{{sum(1 for c in cs if c['significant'] and c['delta'] > 0)}}}",
                   f"\\newcommand{{\\NatNumDown{lab}}}{{{sum(1 for c in cs if c['significant'] and c['delta'] < 0)}}}",
                   f"\\newcommand{{\\NatMeanDelta{lab}}}{{{100 * np.mean([c['delta'] for c in cs]):+.0f}}}"]
        print(f"{lab:8s} vs Best-of-3: sig up {sum(1 for c in cs if c['significant'] and c['delta'] > 0)}, sig down {sum(1 for c in cs if c['significant'] and c['delta'] < 0)}, mean {100 * np.mean([c['delta'] for c in cs]):+.1f} (n={len(cs)})")
    macros.append(f"\\newcommand{{\\NatNumEnvs}}{{{len(out)}}}")
    tex = ('\\begin{tabular}{lcccc r}\n\\toprule\n'
           'Dataset & $(k,c)$ & Best member & \\wmpa{} native & \\wmpa{} shared & native $-$ shared [95\\% CI] \\\\\n\\midrule\n'
           + '\n'.join(body) + '\n\\bottomrule\n\\end{tabular}\n')
    print('\n' + tex)
    json.dump({e: {kk: (vv if kk != 'vs_shared' and kk != 'vs_best3' else {n: {a: b for a, b in c.items() if a != 'per_seed'} for n, c in vv.items()}) for kk, vv in v.items()} for e, v in out.items()},
              open(FLAGS.out_json, 'w'), indent=1)
    if FLAGS.out_dir:
        os.makedirs(os.path.join(FLAGS.out_dir, 'tables'), exist_ok=True)
        open(os.path.join(FLAGS.out_dir, 'tables', 'native_critic.tex'), 'w').write(tex)
        open(os.path.join(FLAGS.out_dir, 'tables', 'numbers_native.tex'), 'w').write('% Auto-generated by scripts/report_native.py -- do not edit.\n' + '\n'.join(macros) + '\n')
        print('wrote tables/native_critic.tex and tables/numbers_native.tex')


if __name__ == '__main__':
    app.run(main)
