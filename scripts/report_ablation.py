"""Report the arbitration ablation: WMPP (selected k=c) vs One-Step WMPP (k=c=1)
vs Random Arbitration (same c, uniform draws, no model).

Reads per env the paired eval dirs `bank_sd<seed>_<dir_tag>/` produced by
eval_planner.py with `--variants=score1_commit1,score<k>_commit<k>
--random_commit=<k>` (plus every fixed bank policy), and reports, per env:

  * absolute mean success of every bank policy family and of the three
    arbitration methods (mean over bank seeds x 100 paired episodes);
  * paired delta vs best-fixed with the paper's hierarchical bootstrap
    (bank seed outer cluster, episodes inner; 10k resamples, 95% CI), and the
    same paired contrast between arbitration methods;
  * policy usage (fraction of env steps per policy family, and its entropy in
    bits), switches and arbitration decisions per episode;
  * model calls: imagined transitions (dynamics) and value-head evaluations,
    per env step and per episode;
  * wall-clock inference overhead: mean ms per real env step for each method,
    the mean over the fixed policies (same node, same job), and the surplus.

Usage:
  python scripts/report_ablation.py --envs=all --dir_tag=abl --seeds=0,1,2
"""

import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('envs', 'all', 'Comma-separated env names, or "all" (every env dir with the tag).')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', 'Planner eval root.')
flags.DEFINE_string('dir_tag', 'abl', 'Suffix of per-seed eval dirs.')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds to aggregate.')
flags.DEFINE_integer('n_boot', 10000, 'Bootstrap resamples.')
flags.DEFINE_string('out', None, 'Output prefix (default: <eval_root>/ablation_<dir_tag>).')
flags.DEFINE_string(
    'env_config',
    os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'),
    'Per-env config with the best policy family by the OGBench test (5 tasks x 50 '
    'episodes at the loaded checkpoint, mean over the 3 bank seeds).',
)


def family(name):
    """'gciql-sd1' -> 'gciql'; planner names unchanged."""
    return name.rsplit('-sd', 1)[0] if '-sd' in name else name


def load_seed(env_dir, seed, tag):
    d = os.path.join(env_dir, f'bank_sd{seed}_{tag}')
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    rows = list(csv.DictReader(open(os.path.join(d, 'episodes.csv'))))
    by_method = {}
    for r in rows:
        by_method.setdefault(r['policy'], []).append(r)
    return summary, by_method


def paired(by_method, a, b):
    key = lambda r: (r['task_id'], r['episode_idx'])
    base = {key(r): float(r['success']) for r in by_method[b]}
    return np.array([float(r['success']) - base[key(r)] for r in by_method[a]])


def hier_boot(per_seed_deltas, n_boot, rng):
    """Hierarchical bootstrap over {seed: episode-delta array}."""
    seeds = list(per_seed_deltas)
    point = float(np.mean([per_seed_deltas[s].mean() for s in seeds]))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(seeds), len(seeds), replace=True)
        means = []
        for i in picked:
            d = per_seed_deltas[seeds[i]]
            means.append(d[rng.integers(0, len(d), len(d))].mean())
        boot[b] = np.mean(means)
    return dict(
        delta=point,
        ci_lo=float(np.percentile(boot, 2.5)),
        ci_hi=float(np.percentile(boot, 97.5)),
        per_seed={str(s): float(per_seed_deltas[s].mean()) for s in seeds},
        significant=bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0),
    )


def entropy_bits(fracs):
    p = np.array([v for v in fracs.values() if v > 0], dtype=float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def classify(summary):
    """Map summary variants -> {'wmpp': name, 'onestep': name, 'random': name}."""
    roles = {}
    for name, spec in summary['variants'].items():
        k, c = spec['imagine'], spec['commit']
        if k == 0:
            roles['random'] = name
        elif k == 1 and c == 1:
            roles['onestep'] = name
        elif k == c:
            roles['wmpp'] = name
    assert set(roles) == {'wmpp', 'onestep', 'random'}, (summary['env_name'], roles)
    return roles


def report_env(env, seeds, tag, n_boot):
    env_dir = os.path.join(FLAGS.eval_root, env)
    rng = np.random.default_rng(0)
    data = {s: load_seed(env_dir, s, tag) for s in seeds}
    roles = classify(data[seeds[0]][0])
    for s in seeds:
        assert classify(data[s][0]) == roles, (env, s)
    methods = [roles['wmpp'], roles['onestep'], roles['random']]
    labels = {roles['wmpp']: 'WMPP', roles['onestep']: 'OneStep', roles['random']: 'Random'}
    spec = data[seeds[0]][0]['variants']

    # --- absolute success per family / method (mean over seeds of per-seed means)
    succ_fam = {}
    for s in seeds:
        summ = data[s][0]
        for name, v in summ['success'].items():
            succ_fam.setdefault(family(name), []).append(float(v))
    success = {f: float(np.mean(v)) for f, v in succ_fam.items()}
    best_fixed = {str(s): data[s][0]['best_fixed'] for s in seeds}
    best_fixed_succ = float(np.mean([data[s][0]['success'][data[s][0]['best_fixed']] for s in seeds]))

    # --- paired contrasts
    contrasts = {}
    for m in methods:
        contrasts[f'{labels[m]}_vs_best_fixed'] = hier_boot(
            {s: paired(data[s][1], m, data[s][0]['best_fixed']) for s in seeds}, n_boot, rng
        )
    # Best policy over 3 seeds by the OGBench test protocol (250 episodes per
    # seed) — one family per env, compared on the same paired episodes via its
    # seed-s instance. Differs from the pinned per-seed best_fixed where ID
    # validation picked another algorithm.
    envcfg = json.load(open(FLAGS.env_config)).get(env, {})
    bp = envcfg.get('best_policy_ogbench_test')
    if bp is not None:
        for s in seeds:
            assert f'{bp}-sd{s}' in data[s][0]['bank'], (env, s, bp)
        for m in methods:
            contrasts[f'{labels[m]}_vs_best_policy'] = hier_boot(
                {s: paired(data[s][1], m, f'{bp}-sd{s}') for s in seeds}, n_boot, rng
            )
    contrasts['WMPP_vs_OneStep'] = hier_boot(
        {s: paired(data[s][1], roles['wmpp'], roles['onestep']) for s in seeds}, n_boot, rng
    )
    contrasts['WMPP_vs_Random'] = hier_boot(
        {s: paired(data[s][1], roles['wmpp'], roles['random']) for s in seeds}, n_boot, rng
    )
    contrasts['OneStep_vs_Random'] = hier_boot(
        {s: paired(data[s][1], roles['onestep'], roles['random']) for s in seeds}, n_boot, rng
    )

    # --- usage / switching / calls / overhead per method
    fixed_ms = float(np.mean([
        v for s in seeds for name, v in data[s][0]['act_ms_per_step'].items() if name in data[s][0]['bank']
    ]))
    per_method = {}
    for m in methods:
        usage = {}
        sw, plans, plan_ms, act_ms = [], [], [], []
        dyn_step, val_step, dyn_ep, val_ep, steps = [], [], [], [], []
        for s in seeds:
            summ, by = data[s]
            ch = summ['policy_chains'][m]
            for name, frac in ch['usage_fraction'].items():
                usage.setdefault(family(name), []).append(float(frac))
            sw.append(ch['mean_switches_per_episode'])
            plans.append(ch['mean_plans_per_episode'])
            plan_ms.append(ch['plan_ms_per_decision'])
            act_ms.append(summ['act_ms_per_step'][m])
            calls = summ.get('model_calls', {}).get(m)
            if calls is None:  # older summaries: derive from episodes.csv
                rows = by[m]
                d = sum(int(float(r.get('wm_transitions', 0) or 0)) for r in rows)
                v = sum(int(float(r.get('value_evals', 0) or 0)) for r in rows)
                n = sum(int(r['steps']) for r in rows)
                calls = dict(dynamics_per_env_step=d / n, value_per_env_step=v / n,
                             dynamics_per_episode=d / len(rows), value_per_episode=v / len(rows))
            dyn_step.append(calls['dynamics_per_env_step'])
            val_step.append(calls['value_per_env_step'])
            dyn_ep.append(calls['dynamics_per_episode'])
            val_ep.append(calls['value_per_episode'])
            steps.append(float(np.mean([int(r['steps']) for r in by[m]])))
        usage = {f: float(np.mean(v)) for f, v in usage.items()}
        per_method[labels[m]] = dict(
            variant=m,
            imagine=spec[m]['imagine'],
            commit=spec[m]['commit'],
            success=success[m],
            usage_fraction=usage,
            usage_entropy_bits=entropy_bits(usage),
            switches_per_episode=float(np.mean(sw)),
            decisions_per_episode=float(np.mean(plans)),
            episode_length=float(np.mean(steps)),
            dynamics_calls_per_env_step=float(np.mean(dyn_step)),
            value_calls_per_env_step=float(np.mean(val_step)),
            dynamics_calls_per_episode=float(np.mean(dyn_ep)),
            value_calls_per_episode=float(np.mean(val_ep)),
            act_ms_per_step=float(np.mean(act_ms)),
            overhead_ms_per_step=float(np.mean(act_ms) - fixed_ms),
            overhead_ratio=float(np.mean(act_ms) / fixed_ms) if fixed_ms > 0 else float('nan'),
            plan_ms_per_decision=float(np.mean(plan_ms)),
        )

    return dict(
        env_name=env,
        bank_seeds=seeds,
        bank_families=sorted({family(n) for s in seeds for n in data[s][0]['bank']}),
        selected_k=spec[roles['wmpp']]['imagine'],
        best_fixed=best_fixed,
        best_fixed_success=best_fixed_succ,
        best_policy=bp,
        best_policy_ogbench_test_success=(envcfg['ogbench_test_family_mean'][bp] if bp else None),
        best_policy_success=(success[bp] if bp else None),
        fixed_policy_act_ms_per_step=fixed_ms,
        success=success,
        methods=per_method,
        contrasts=contrasts,
    )


def fmt_ci(c):
    return f"{100 * c['delta']:+.1f} [{100 * c['ci_lo']:+.1f}, {100 * c['ci_hi']:+.1f}]" + ('*' if c['significant'] else '')


def markdown(reports):
    lines = []
    lines.append('## Arbitration ablation: WMPP (k=c selected) vs One-Step WMPP (k=c=1) vs Random (same c)\n')
    lines.append('Mean success (%) over 3 bank seeds x 100 paired episodes; deltas are paired vs that bank\'s best-fixed policy, hierarchical bootstrap 95% CI (* = CI excludes 0).\n')
    lines.append('| env | k=c | best-fixed | WMPP | OneStep | Random | ΔWMPP | ΔOneStep | ΔRandom | WMPP−OneStep | WMPP−Random |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in reports:
        m = r['methods']; c = r['contrasts']
        vals = dict(bf=r['best_fixed_success'], WMPP=m['WMPP']['success'], OneStep=m['OneStep']['success'], Random=m['Random']['success'])
        top = max(vals.values())
        cell = lambda k: (f"**{100 * vals[k]:.1f}**" if abs(vals[k] - top) < 1e-12 else f"{100 * vals[k]:.1f}")
        lines.append(
            f"| {r['env_name']} | {r['selected_k']} | {cell('bf')} | {cell('WMPP')} | {cell('OneStep')} | {cell('Random')} | "
            f"{fmt_ci(c['WMPP_vs_best_fixed'])} | {fmt_ci(c['OneStep_vs_best_fixed'])} | {fmt_ci(c['Random_vs_best_fixed'])} | "
            f"{fmt_ci(c['WMPP_vs_OneStep'])} | {fmt_ci(c['WMPP_vs_Random'])} |"
        )
    lines.append('\n### Versus the best policy over 3 seeds (OGBench test: 5 tasks x 50 episodes per seed, 1M checkpoint)\n')
    lines.append('Best policy = algorithm family with the highest mean OGBench-test success over the 3 bank seeds; "test" is its 250-episode score, "paired" its success on the 100 paired episodes per seed used here; deltas are paired vs its seed-s instance.\n')
    lines.append('| env | best policy | test | paired | WMPP | OneStep | Random | ΔWMPP | ΔOneStep | ΔRandom |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in reports:
        if not r.get('best_policy'):
            continue
        m = r['methods']; c = r['contrasts']
        lines.append(
            f"| {r['env_name']} | {r['best_policy']} | {100 * r['best_policy_ogbench_test_success']:.1f} | {100 * r['best_policy_success']:.1f} | "
            f"{100 * m['WMPP']['success']:.1f} | {100 * m['OneStep']['success']:.1f} | {100 * m['Random']['success']:.1f} | "
            f"{fmt_ci(c['WMPP_vs_best_policy'])} | {fmt_ci(c['OneStep_vs_best_policy'])} | {fmt_ci(c['Random_vs_best_policy'])} |"
        )
    lines.append('\n### Switching, model calls and inference overhead\n')
    lines.append('Calls are per real env step (dynamics = imagined transitions incl. ensemble members; value = value-head evaluations). Overhead = mean ms per real env step minus the mean over the fixed bank policies measured in the same job.\n')
    lines.append('| env | method | switches/ep | decisions/ep | ep len | dyn/step | val/step | dyn/ep | val/ep | ms/step | fixed ms/step | overhead ms | ×fixed | usage entropy (bits) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in reports:
        for lab in ('WMPP', 'OneStep', 'Random'):
            m = r['methods'][lab]
            lines.append(
                f"| {r['env_name']} | {lab} ({m['imagine']},{m['commit']}) | {m['switches_per_episode']:.1f} | {m['decisions_per_episode']:.1f} | "
                f"{m['episode_length']:.0f} | {m['dynamics_calls_per_env_step']:.1f} | {m['value_calls_per_env_step']:.1f} | "
                f"{m['dynamics_calls_per_episode']:.0f} | {m['value_calls_per_episode']:.0f} | {m['act_ms_per_step']:.2f} | "
                f"{r['fixed_policy_act_ms_per_step']:.2f} | {m['overhead_ms_per_step']:+.2f} | {m['overhead_ratio']:.2f} | {m['usage_entropy_bits']:.2f} |"
            )
    lines.append('\n### Absolute success table (all bank policies + methods)\n')
    fams = sorted({f for r in reports for f in r['bank_families']})
    lines.append('| env | ' + ' | '.join(fams) + ' | WMPP | OneStep | Random |')
    lines.append('|---|' + '---|' * (len(fams) + 3))
    for r in reports:
        cells = []
        vals = {f: r['success'].get(f) for f in fams}
        vals.update(WMPP=r['methods']['WMPP']['success'], OneStep=r['methods']['OneStep']['success'], Random=r['methods']['Random']['success'])
        best = max(v for v in vals.values() if v is not None)
        for f in fams + ['WMPP', 'OneStep', 'Random']:
            v = vals[f]
            if v is None:
                cells.append('—')
            else:
                t = f'{100 * v:.1f}'
                cells.append(f'**{t}**' if abs(v - best) < 1e-12 else t)
        lines.append(f"| {r['env_name']} | " + ' | '.join(cells) + ' |')
    lines.append('\n### Policy usage (fraction of env steps)\n')
    lines.append('| env | method | ' + ' | '.join(fams) + ' |')
    lines.append('|---|---|' + '---|' * len(fams))
    for r in reports:
        for lab in ('WMPP', 'OneStep', 'Random'):
            u = r['methods'][lab]['usage_fraction']
            lines.append(f"| {r['env_name']} | {lab} | " + ' | '.join(f'{100 * u[f]:.0f}' if f in u else '—' for f in fams) + ' |')
    return '\n'.join(lines) + '\n'


def main(_):
    seeds = [int(s) for s in FLAGS.seeds.split(',')]
    if FLAGS.envs == 'all':
        envs = sorted(
            d for d in os.listdir(FLAGS.eval_root)
            if all(os.path.exists(os.path.join(FLAGS.eval_root, d, f'bank_sd{s}_{FLAGS.dir_tag}', 'summary.json')) for s in seeds)
        )
    else:
        envs = FLAGS.envs.split(',')
    reports = []
    for env in envs:
        print(f'[report] {env}', flush=True)
        reports.append(report_env(env, seeds, FLAGS.dir_tag, FLAGS.n_boot))
    out = FLAGS.out or os.path.join(FLAGS.eval_root, f'ablation_{FLAGS.dir_tag}')
    with open(out + '.json', 'w') as f:
        json.dump(reports, f, indent=2)
    md = markdown(reports)
    with open(out + '.md', 'w') as f:
        f.write(md)
    print(md)
    print(f'wrote {out}.json / {out}.md')


if __name__ == '__main__':
    app.run(main)
