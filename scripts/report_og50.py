"""Report the official-protocol (5 tasks x 50 episodes) arbitration results.

Reads per env `bank_sd<seed>_og50/` (planner variants only: diagonal (k,k)
cells, score1_commit1, random_commit{10,25,50,100}); fixed-policy baselines
come either from `bank_sd<seed>_og50fx/` (--fixed_tag=og50fx: every bank
policy re-evaluated on the planners' episode seeds -> paired contrasts; the
paper's setting since 2026-08-25) or, with --fixed_tag empty, from each run's
OGBench eval.csv recorded in manifests/wmpp_env_config.json (unpaired offset).

Per env it reports:
  * the full diagonal (k,k) sweep and the re-selected best cell (ties ->
    smallest k);
  * WMPP (selected k=c), One-Step (1,1) and Random (same c) absolute success,
    delta vs the best policy family over 3 seeds (eval.csv mean, treated as
    the reported baseline value; CI from hierarchical bootstrap of the
    method's episodes) and paired method-vs-method contrasts;
  * switching, model calls, usage entropy and act-ms overhead (fixed-policy
    ms anchor comes from the same-bank 20-episode ablation runs, recorded in
    ablation_abl.json, since og50 jobs do not re-run fixed policies).

Usage:
  python scripts/report_og50.py --envs=all
"""

import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import hier_boot, paired_rows  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('envs', 'all', 'Comma-separated env names, or "all".')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('dir_tag', 'og50', 'Suffix of per-seed eval dirs.')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds to aggregate.')
flags.DEFINE_integer('n_boot', 10000, 'Bootstrap resamples.')
flags.DEFINE_string('env_config', os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'),
                    'Per-env eval.csv baselines (OGBench test, 250 eps per seed).')
flags.DEFINE_string('abl_json', '/scratch/jwquan/wmpp/planner_eval/ablation_abl.json',
                    'Prior ablation report (fixed-policy act-ms anchor).')
flags.DEFINE_string('out', None, 'Output prefix (default: <eval_root>/og50_report).')
flags.DEFINE_bool('onestep_as_cell', False, 'Treat (1,1) as one more k=c cell of the hyperparameter sweep '
                  '(selected like any other); drop the separate One-Step control and report Random only.')
flags.DEFINE_string('family_scorer', '', 'Per-family rollout scorer, e.g. "puzzle:critic": families listed use the '
                    'critic{k}_commit{k} variants (a bank member\'s own V, eval_planner --critic_scorer) as WMPP; '
                    'all other families use the LAVL head (score{k}_commit{k}). The critic rows must be merged '
                    'via --extra_tags (e.g. og50cr).')
flags.DEFINE_string('extra_tags', '', 'Comma-separated extra dir tags whose variants are merged into the main '
                    'tag (e.g. og50r1 = Random at c=1).')
flags.DEFINE_enum('select_rule', 'test', ['test', 'loo_family', 'global', 'validation', 'family'],
                  'How the reported k=c cell is chosen: test = best mean on this dataset (paper); loo_family = the k with the best '
                  'mean over the OTHER datasets of the same family (leave-one-dataset-out); global = the k with the best macro-average '
                  'over all other datasets. The two leakage-free rules never look at the dataset being reported.')
flags.DEFINE_string('family_k', 'maze:1,cube:5,scene:10,puzzle:10', 'family:k list for --select_rule=family (WMPP k=c per task family; '
                    'values chosen on the validation episodes 50..99, never on the reported ones).')
flags.DEFINE_string('family_c', 'maze:1,cube:5,scene:10,puzzle:10', 'family:c list for --select_rule=family (Random-Switch interval per family).')
flags.DEFINE_string('val_tag', 'og50val', 'Dir tag of the validation sweep (episodes 50..99) used by --select_rule=validation: '
                    'k* = best mean WMPP diagonal cell and c* = best mean Random interval, both chosen there and never on the '
                    'reported episodes.')
flags.DEFINE_string('selected_cells_out', os.path.join(ROOT, 'manifests', 'selected_cells.json'),
                    'Where --select_rule=validation writes {env: {k, c, ...}} for downstream launchers.')
flags.DEFINE_string('fixed_tag', '', 'Dir tag holding every fixed bank policy evaluated on the SAME episode seeds '
                    'as the planners (og50fx). When set, the best policy and all fixed baselines come from these '
                    'runs and every delta-vs-best is a paired per-episode contrast; when empty, the eval.csv '
                    'baselines from --env_config are used (unpaired, offset only).')


def family(name):
    return name.rsplit('-sd', 1)[0] if '-sd' in name else name


def load_seed(env_dir, seed, tag, extra_tags=()):
    d = os.path.join(env_dir, f'bank_sd{seed}_{tag}')
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    rows = list(csv.DictReader(open(os.path.join(d, 'episodes.csv'))))
    by_method = {}
    for r in rows:
        by_method.setdefault(r['policy'], []).append(r)
    for et in extra_tags:
        de = os.path.join(env_dir, f'bank_sd{seed}_{et}')
        if not os.path.exists(os.path.join(de, 'summary.json')):
            continue
        with open(os.path.join(de, 'summary.json')) as f:
            se = json.load(f)
        for key in ('variants', 'success', 'policy_chains', 'act_ms_per_step', 'model_calls'):
            summary.setdefault(key, {}).update(se.get(key, {}))
        for r in csv.DictReader(open(os.path.join(de, 'episodes.csv'))):
            by_method.setdefault(r['policy'], []).append(r)
    return summary, by_method


def succ_arr(by_method, m):
    return np.array([float(r['success']) for r in by_method[m]])


def paired(by_method, a, b):
    key = lambda r: (r['task_id'], r['episode_idx'])
    base = {key(r): float(r['success']) for r in by_method[b]}
    return np.array([float(r['success']) - base[key(r)] for r in by_method[a]])


def load_fixed(env_dir, seed, tag):
    """Fixed-policy rows of bank_sd<seed>_<tag>/ grouped by algorithm family, plus act-ms per policy."""
    d = os.path.join(env_dir, f'bank_sd{seed}_{tag}')
    rows = list(csv.DictReader(open(os.path.join(d, 'episodes.csv'))))
    by_fam = {}
    for r in rows:
        if r['policy'].startswith(('score', 'random')):
            continue
        by_fam.setdefault(family(r['policy']), []).append(r)
    with open(os.path.join(d, 'summary.json')) as f:
        summ = json.load(f)
    act_ms = {family(k): float(v) for k, v in summ.get('act_ms_per_step', {}).items()
              if not k.startswith(('score', 'random'))}
    return by_fam, act_ms


def entropy_bits(fracs):
    p = np.array([v for v in fracs.values() if v > 0], dtype=float)
    if not len(p):
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def env_family(env):
    for fam, prefixes in (('maze', ('pointmaze', 'antmaze', 'humanoidmaze')), ('cube', ('cube',)), ('scene', ('scene',)), ('puzzle', ('puzzle',))):
        if env.startswith(prefixes):
            return fam
    return 'other'


def diag_success(env, seeds, tag):
    """{k: 3-seed mean success of score<k>_commit<k>} for one dataset (main + extra tags)."""
    env_dir = os.path.join(FLAGS.eval_root, env)
    extra = [t for t in FLAGS.extra_tags.split(',') if t]
    data = {s: load_seed(env_dir, s, tag, extra) for s in seeds}
    spec = data[seeds[0]][0]['variants']
    kmin = 1 if FLAGS.onestep_as_cell else 2
    out = {}
    for v, d in spec.items():
        if d['imagine'] == d['commit'] >= kmin and not v.startswith('mpc'):
            out[d['imagine']] = float(np.mean([data[s][0]['success'][v] for s in seeds]))
    return out


TEST_RANGE = [0, 50]
VAL_RANGE = [50, 100]


def load_validation(env_dir, seed, tag):
    """Mean success per variant of one validation run (summary only; never merged with test rows)."""
    d = os.path.join(env_dir, f'bank_sd{seed}_{tag}')
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    assert summary.get('episode_range') == VAL_RANGE, (d, summary.get('episode_range'))
    return {v: float(x) for v, x in summary['success'].items()}


def validation_selection(envs, seeds, tag):
    """(k*, c*) per env from the validation sweep: argmax of the 3-seed mean over the diagonal
    WMPP cells and, independently, over the Random intervals (ties -> smallest k / c)."""
    out = {}
    for e in envs:
        env_dir = os.path.join(FLAGS.eval_root, e)
        per = [load_validation(env_dir, s, tag) for s in seeds]
        common = set.intersection(*[set(p) for p in per])
        diag = {int(v.split('score')[1].split('_')[0]): float(np.mean([p[v] for p in per]))
                for v in common if v.startswith('score') and v == f"score{v.split('score')[1].split('_')[0]}_commit{v.split('score')[1].split('_')[0]}"}
        rand = {int(v.replace('random_commit', '')): float(np.mean([p[v] for p in per]))
                for v in common if v.startswith('random_commit')}
        assert diag and rand, (e, sorted(common))
        k = max(diag, key=lambda x: (diag[x], -x))
        c = max(rand, key=lambda x: (rand[x], -x))
        out[e] = dict(k=k, c=c, val_wmpp=diag, val_random=rand, rule='validation')
    return out


def leakage_free_k(envs, seeds, tag, rule):
    """k per dataset chosen WITHOUT that dataset's own sweep (loo_family | global)."""
    ds = {e: diag_success(e, seeds, tag) for e in envs}
    chosen = {}
    for e in envs:
        others = [o for o in envs if o != e and (rule == 'global' or env_family(o) == env_family(e))]
        if not others:
            others = [o for o in envs if o != e]
        ks = sorted(set.intersection(*[set(ds[o]) for o in others]) & set(ds[e]))
        score = {k: float(np.mean([ds[o][k] for o in others])) for k in ks}
        chosen[e] = max(ks, key=lambda k: (score[k], -k))
    return chosen


def report_env(env, seeds, tag, n_boot, envcfg, abl_fixed_ms, force_k=None, force_c=None, val_info=None,
               scorer='lavl'):
    env_dir = os.path.join(FLAGS.eval_root, env)
    rng = np.random.default_rng(0)
    extra = [t for t in FLAGS.extra_tags.split(',') if t]
    data = {s: load_seed(env_dir, s, tag, extra) for s in seeds}
    spec = data[seeds[0]][0]['variants']
    for s in seeds:
        assert set(data[s][0]['variants']) == set(spec), (env, s)
        er = data[s][0].get('episode_range')
        assert er is None or list(er) == TEST_RANGE, (env, s, er, 'reported runs must be the test episodes')

    kmin = 1 if FLAGS.onestep_as_cell else 2
    diag = sorted((v for v, d in spec.items() if d['imagine'] == d['commit'] >= kmin and not v.startswith('mpc')),
                  key=lambda v: spec[v]['imagine'])
    diag_succ = {v: float(np.mean([data[s][0]['success'][v] for s in seeds])) for v in diag}
    sel = max(diag, key=lambda v: (diag_succ[v], -spec[v]['imagine']))
    if force_k is not None:
        prefix = 'critic' if scorer == 'critic' else 'score'
        sel = f'{prefix}{force_k}_commit{force_k}'
        assert sel in spec, (env, sel, f'scorer={scorer}: merge its tag via --extra_tags')
    k = spec[sel]['imagine']
    roles = {'WMPP': sel, 'Random': f'random_commit{k}'}
    if force_c is not None:
        roles['Random'] = f'random_commit{force_c}'
    if not FLAGS.onestep_as_cell:
        roles['OneStep'] = 'score1_commit1'
    assert roles['Random'] in spec, (env, roles)
    has_os = 'OneStep' in roles
    # Sampling-MPC baseline (single fixed policy as prior; world-model search
    # without a portfolio), reported at the WMPP-selected k when available.
    mpc_vs = sorted(v for v in spec if v.startswith('mpc') and spec[v]['imagine'] == k)
    if mpc_vs:
        roles['MPC'] = mpc_vs[0]

    cfg = envcfg[env]
    official_bp = cfg['best_policy_ogbench_test']
    official_fixed = {f: float(v) for f, v in cfg['ogbench_test_family_mean'].items()}
    official_bp_test = official_fixed[official_bp]

    contrasts, per_method = {}, {}
    if FLAGS.fixed_tag:
        # Paired baselines: every bank member on the planners' episode seeds.
        fixed = {s: load_fixed(env_dir, s, FLAGS.fixed_tag) for s in seeds}
        fams = sorted(set.intersection(*[set(fixed[s][0]) for s in seeds]))
        fixed_seed_means = {f: {str(s): float(np.mean([float(r['success']) for r in fixed[s][0][f]])) for s in seeds} for f in fams}
        fixed_test = {f: float(np.mean(list(v.values()))) for f, v in fixed_seed_means.items()}
        bp = max(fams, key=lambda f: (fixed_test[f], f))
        bp_test = fixed_test[bp]
        for lab, m in roles.items():
            contrasts[f'{lab}_vs_best_policy'] = hier_boot(
                {s: paired_rows(data[s][1][m], fixed[s][0][bp]) for s in seeds}, n_boot, rng)
        fixed_ms_anchor = float(np.mean([np.mean(list(fixed[s][1].values())) for s in seeds if fixed[s][1]]))
        baseline_source = f'{FLAGS.fixed_tag} (all bank policies re-evaluated on the planners\' episode seeds; paired contrasts)'
    else:
        bp, bp_test, fixed_test, fixed_seed_means = official_bp, official_bp_test, official_fixed, None
        for lab, m in roles.items():
            contrasts[f'{lab}_vs_best_policy'] = hier_boot(
                {s: succ_arr(data[s][1], m) for s in seeds}, n_boot, rng, offset=bp_test)
        fixed_ms_anchor = abl_fixed_ms.get(env)
        baseline_source = 'eval.csv (each checkpoint\'s own OGBench evaluation; unpaired offset)'
    contrasts['WMPP_vs_Random'] = hier_boot(
        {s: paired(data[s][1], roles['WMPP'], roles['Random']) for s in seeds}, n_boot, rng)
    if 'MPC' in roles:
        contrasts['WMPP_vs_MPC'] = hier_boot(
            {s: paired(data[s][1], roles['WMPP'], roles['MPC']) for s in seeds}, n_boot, rng)
    if has_os:
        contrasts['WMPP_vs_OneStep'] = hier_boot(
            {s: paired(data[s][1], roles['WMPP'], roles['OneStep']) for s in seeds}, n_boot, rng)
        contrasts['OneStep_vs_Random'] = hier_boot(
            {s: paired(data[s][1], roles['OneStep'], roles['Random']) for s in seeds}, n_boot, rng)

    for lab, m in roles.items():
        usage = {}
        acc = dict(sw=[], plans=[], plan_ms=[], act_ms=[], dyn=[], val=[], dyn_ep=[], val_ep=[], steps=[])
        for s in seeds:
            summ, by = data[s]
            ch = summ['policy_chains'][m]
            for name, frac in ch['usage_fraction'].items():
                usage.setdefault(family(name), []).append(float(frac))
            acc['sw'].append(ch['mean_switches_per_episode'])
            acc['plans'].append(ch['mean_plans_per_episode'])
            acc['plan_ms'].append(ch['plan_ms_per_decision'])
            acc['act_ms'].append(summ['act_ms_per_step'][m])
            calls = summ['model_calls'][m]
            acc['dyn'].append(calls['dynamics_per_env_step'])
            acc['val'].append(calls['value_per_env_step'])
            acc['dyn_ep'].append(calls['dynamics_per_episode'])
            acc['val_ep'].append(calls['value_per_episode'])
            acc['steps'].append(float(np.mean([int(r['steps']) for r in by[m]])))
        usage = {f: float(np.mean(v)) for f, v in usage.items()}
        fixed_ms = fixed_ms_anchor
        act = float(np.mean(acc['act_ms']))
        per_method[lab] = dict(
            variant=m, imagine=spec[m]['imagine'], commit=spec[m]['commit'],
            success=float(np.mean([data[s][0]['success'][m] for s in seeds])),
            usage_fraction=usage, usage_entropy_bits=entropy_bits(usage),
            switches_per_episode=float(np.mean(acc['sw'])),
            decisions_per_episode=float(np.mean(acc['plans'])),
            episode_length=float(np.mean(acc['steps'])),
            dynamics_calls_per_env_step=float(np.mean(acc['dyn'])),
            value_calls_per_env_step=float(np.mean(acc['val'])),
            dynamics_calls_per_episode=float(np.mean(acc['dyn_ep'])),
            value_calls_per_episode=float(np.mean(acc['val_ep'])),
            act_ms_per_step=act,
            fixed_ms_anchor=fixed_ms,
            overhead_ms_per_step=(act - fixed_ms) if fixed_ms else None,
            plan_ms_per_decision=float(np.mean(acc['plan_ms'])),
        )

    return dict(
        env_name=env, bank_seeds=seeds, protocol='5 tasks x 50 episodes per bank seed (official OGBench)',
        selected_k=k, selected_variant=sel, select_rule=FLAGS.select_rule,
        selected_c=spec[roles['Random']]['commit'], validation=val_info,
        diagonal={v: diag_succ[v] for v in diag},
        onestep_success=float(np.mean([data[s][0]['success']['score1_commit1'] for s in seeds])),
        random_by_commit={v: float(np.mean([data[s][0]['success'][v] for s in seeds]))
                          for v in spec if spec[v]['imagine'] == 0},
        mpc_variants={v: float(np.mean([data[s][0]['success'][v] for s in seeds])) for v in spec if v.startswith('mpc')},
        mpc_config=(data[seeds[0]][0].get('mpc') if mpc_vs else None),
        best_policy=bp, best_policy_test_success=bp_test, fixed_test_success=fixed_test,
        best_policy_seeds=(fixed_seed_means[bp] if fixed_seed_means else
                           {str(s): float(cfg['ogbench_test_success'][f'{bp}-sd{s}']) for s in seeds}),
        fixed_test_success_per_seed=fixed_seed_means,
        baseline_source=baseline_source,
        official_best_policy=official_bp, official_best_policy_test_success=official_bp_test,
        official_fixed_test_success=official_fixed,
        methods=per_method, contrasts=contrasts,
    )


def fmt_ci(c):
    return f"{100 * c['delta']:+.1f} [{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}]" + ('*' if c['significant'] else '')


def markdown(reports):
    L = []
    os_ = not FLAGS.onestep_as_cell
    labs = ('WMPP', 'OneStep', 'Random') if os_ else ('WMPP', 'Random')
    L.append('## Official-protocol results (5 tasks x 50 episodes per bank seed): WMPP vs One-Step vs Random\n')
    if FLAGS.fixed_tag:
        L.append(f'Baselines: every bank policy re-evaluated under the official protocol on the planners\' episode seeds (tag {FLAGS.fixed_tag}). '
                 'Best policy = family with the highest 3-seed mean; every Δ is a paired per-episode contrast with a hierarchical bootstrap '
                 '(seeds, then episodes). * = CI excludes 0.\n')
    else:
        L.append('Baselines are each run\'s own OGBench eval.csv at the loaded checkpoint (same 250-episode protocol; no re-evaluation). '
                 'Best policy = family with the highest 3-seed mean; Δ vs it treats the eval.csv value as the reported baseline, '
                 'CI from hierarchical bootstrap of the method\'s 750 episodes. Method-vs-method contrasts are paired per episode. * = CI excludes 0.\n')
    if os_:
        L.append('| env | k=c | best policy | test | WMPP | OneStep | Random | ΔWMPP | ΔOneStep | ΔRandom | WMPP−OneStep | WMPP−Random |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    else:
        L.append('| env | k=c | best policy | test | WMPP | Random | ΔWMPP | ΔRandom | WMPP−Random |')
        L.append('|---|---|---|---|---|---|---|---|---|')
    for r in reports:
        m, c = r['methods'], r['contrasts']
        vals = dict(bp=r['best_policy_test_success'], **{lab: m[lab]['success'] for lab in labs})
        top = max(vals.values())
        f = lambda kk: (f"**{100 * vals[kk]:.1f}**" if abs(vals[kk] - top) < 1e-12 else f"{100 * vals[kk]:.1f}")
        if os_:
            L.append(f"| {r['env_name']} | {r['selected_k']} | {r['best_policy']} | {f('bp')} | {f('WMPP')} | {f('OneStep')} | {f('Random')} | "
                     f"{fmt_ci(c['WMPP_vs_best_policy'])} | {fmt_ci(c['OneStep_vs_best_policy'])} | {fmt_ci(c['Random_vs_best_policy'])} | "
                     f"{fmt_ci(c['WMPP_vs_OneStep'])} | {fmt_ci(c['WMPP_vs_Random'])} |")
        else:
            L.append(f"| {r['env_name']} | {r['selected_k']} | {r['best_policy']} | {f('bp')} | {f('WMPP')} | {f('Random')} | "
                     f"{fmt_ci(c['WMPP_vs_best_policy'])} | {fmt_ci(c['Random_vs_best_policy'])} | {fmt_ci(c['WMPP_vs_Random'])} |")
    mpc_reports = [r for r in reports if 'MPC' in r['methods']]
    if mpc_reports:
        L.append('\n### Sampling-MPC baseline (best fixed policy as action prior; N candidates = policy mean + N−1 Gaussian perturbations, '
                 'value-head scored, first action executed)\n')
        L.append('| env | MPC variant (N, σ, k, c) | prior policy | best policy | test | MPC | WMPP | ΔMPC vs best | WMPP−MPC |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for r in mpc_reports:
            m, c = r['methods'], r['contrasts']
            usage = m['MPC']['usage_fraction']
            prior = max(usage, key=usage.get) if usage else '—'
            L.append(f"| {r['env_name']} | {m['MPC']['variant']} | {prior} | {r['best_policy']} | "
                     f"{100 * r['best_policy_test_success']:.1f} | {100 * m['MPC']['success']:.1f} | {100 * m['WMPP']['success']:.1f} | "
                     f"{fmt_ci(c['MPC_vs_best_policy'])} | {fmt_ci(c['WMPP_vs_MPC'])} |")
    L.append('\n### Diagonal sweep (mean success %, cell re-selected under the official protocol)\n')
    # variants are {prefix}{k}_commit{k} with prefix 'score' (metric value) or 'critic' (direct value)
    ks = sorted({int(re.search(r'(\d+)_commit', v).group(1)) for r in reports for v in r['diagonal']})
    L.append('| env | ' + ' | '.join(f'({k},{k})' for k in ks) + ' | selected |')
    L.append('|---|' + '---|' * (len(ks) + 1))
    for r in reports:
        pref = 'critic' if (r.get('validation') or {}).get('scorer') == 'critic' else 'score'
        cells = []
        for k in ks:
            v = r['diagonal'].get(f'{pref}{k}_commit{k}')
            cells.append('—' if v is None else f'{100 * v:.1f}')
        L.append(f"| {r['env_name']} | " + ' | '.join(cells) + f" | ({r['selected_k']},{r['selected_k']}) |")
    L.append('\n### Random control at every commitment interval (mean success %)\n')
    rks = sorted({int(v[13:]) for r in reports for v in r['random_by_commit']})
    L.append('| env | ' + ' | '.join(f'c={k}' for k in rks) + ' |')
    L.append('|---|' + '---|' * len(rks))
    for r in reports:
        L.append(f"| {r['env_name']} | " + ' | '.join(f"{100 * r['random_by_commit'].get(f'random_commit{k}', float('nan')):.1f}" for k in rks) + ' |')
    L.append('\n### Switching, model calls and inference overhead\n')
    L.append('Fixed-policy ms anchor from the same-bank 20-episode runs (og50 jobs run planners only).\n')
    L.append('| env | method (k,c) | switches/ep | decisions/ep | ep len | dyn/step | val/step | dyn/ep | val/ep | ms/step | fixed anchor | overhead ms | usage entropy |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in reports:
        for lab in labs + (('MPC',) if 'MPC' in r['methods'] else ()):
            m = r['methods'][lab]
            oh = f"{m['overhead_ms_per_step']:+.2f}" if m['overhead_ms_per_step'] is not None else '—'
            fx = f"{m['fixed_ms_anchor']:.2f}" if m['fixed_ms_anchor'] else '—'
            L.append(f"| {r['env_name']} | {lab} ({m['imagine']},{m['commit']}) | {m['switches_per_episode']:.1f} | {m['decisions_per_episode']:.1f} | "
                     f"{m['episode_length']:.0f} | {m['dynamics_calls_per_env_step']:.1f} | {m['value_calls_per_env_step']:.1f} | "
                     f"{m['dynamics_calls_per_episode']:.0f} | {m['value_calls_per_episode']:.0f} | {m['act_ms_per_step']:.2f} | {fx} | {oh} | {m['usage_entropy_bits']:.2f} |")
    L.append('\n### Absolute success: fixed baselines (250 eps x 3 seeds) + methods (750 eps)\n')
    fams = sorted({f for r in reports for f in r['fixed_test_success']})
    if mpc_reports:
        labs = tuple(labs) + ('MPC',)
    L.append('| env | ' + ' | '.join(fams) + ' | ' + ' | '.join(labs) + ' |')
    L.append('|---|' + '---|' * (len(fams) + len(labs)))
    for r in reports:
        vals = {f: r['fixed_test_success'].get(f) for f in fams}
        vals.update({lab: (r['methods'][lab]['success'] if lab in r['methods'] else None) for lab in labs})
        top = max(v for v in vals.values() if v is not None)
        cells = []
        for f in fams + list(labs):
            v = vals[f]
            cells.append('—' if v is None else (f'**{100 * v:.1f}**' if abs(v - top) < 1e-12 else f'{100 * v:.1f}'))
        L.append(f"| {r['env_name']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n'


def main(_):
    seeds = [int(s) for s in FLAGS.seeds.split(',')]
    envcfg = json.load(open(FLAGS.env_config))
    abl_fixed_ms = {}
    if os.path.exists(FLAGS.abl_json):
        abl_fixed_ms = {r['env_name']: r['fixed_policy_act_ms_per_step'] for r in json.load(open(FLAGS.abl_json))}
    if FLAGS.envs == 'all':
        envs = sorted(d for d in os.listdir(FLAGS.eval_root)
                      if all(os.path.exists(os.path.join(FLAGS.eval_root, d, f'bank_sd{s}_{FLAGS.dir_tag}', 'summary.json')) for s in seeds))
    else:
        envs = FLAGS.envs.split(',')
    forced, forced_c, val = {}, {}, {}
    if FLAGS.select_rule == 'validation':
        val = validation_selection(envs, seeds, FLAGS.val_tag)
        forced = {e: v['k'] for e, v in val.items()}
        forced_c = {e: v['c'] for e, v in val.items()}
        print(f'[report] validation selection (k*, c*): {{e: (v["k"], v["c"]) for e, v in val.items()}}')
        with open(FLAGS.selected_cells_out, 'w') as f:
            json.dump(val, f, indent=1)
        print(f'[report] wrote {FLAGS.selected_cells_out}')
    elif FLAGS.select_rule == 'family':
        fk = {a: int(b) for a, b in (x.split(':') for x in FLAGS.family_k.split(','))}
        fc = {a: int(b) for a, b in (x.split(':') for x in FLAGS.family_c.split(','))}
        fs = {a: b for a, b in (x.split(':') for x in FLAGS.family_scorer.split(',') if x)}
        val = {e: dict(k=fk[env_family(e)], c=fc[env_family(e)], rule='family', family=env_family(e),
                       scorer=fs.get(env_family(e), 'lavl')) for e in envs}
        forced = {e: v['k'] for e, v in val.items()}
        forced_c = {e: v['c'] for e, v in val.items()}
        print(f'[report] family rule k={fk} c={fc} scorer={fs or "lavl everywhere"}')
        with open(FLAGS.selected_cells_out, 'w') as f:
            json.dump(val, f, indent=1)
        print(f'[report] wrote {FLAGS.selected_cells_out}')
    elif FLAGS.select_rule != 'test':
        forced = leakage_free_k(envs, seeds, FLAGS.dir_tag, FLAGS.select_rule)
        print(f'[report] {FLAGS.select_rule} selection: {forced}')
    reports = []
    for env in envs:
        print(f'[report] {env}', flush=True)
        reports.append(report_env(env, seeds, FLAGS.dir_tag, FLAGS.n_boot, envcfg, abl_fixed_ms,
                                  force_k=forced.get(env), force_c=forced_c.get(env), val_info=val.get(env),
                                  scorer=(val.get(env) or {}).get('scorer', 'lavl')))
    out = FLAGS.out or os.path.join(FLAGS.eval_root, 'og50_report')
    with open(out + '.json', 'w') as f:
        json.dump(reports, f, indent=2)
    md = markdown(reports)
    with open(out + '.md', 'w') as f:
        f.write(md)
    print(md)
    print(f'wrote {out}.json / {out}.md')


if __name__ == '__main__':
    app.run(main)
