"""Paired evaluation: {one-step chooser, rollout ranker, each fixed policy}.

All methods see identical episode keys (same resets, same goals). Asserts
budget parity between chooser and ranker (identical world-model transitions
per env step). Reports paired success deltas vs best-fixed with episode-level
bootstrap CIs and the fraction of P3 oracle headroom captured.

Run AFTER run_wm_diagnostics.py returns GO for this env.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium
import numpy as np
import ogbench  # noqa: F401
from absl import app, flags

from evaluation.oracle import headroom
from evaluation.paired_eval import evaluate_paired
from interfaces.policy_bank import load_bank, select_best_fixed
from planners.portfolio import LeastUsedArbiter, NoisyPolicyArbiter, PolicyMPC, PortfolioMPC, RandomArbiter, RolloutRanker
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter
from world_model.success_predicates import get_progress_fn

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'OGBench dataset name.', required=True)
flags.DEFINE_string('wm_dir', None, 'World-model run dir.', required=True)
flags.DEFINE_integer('wm_epoch', 500000, 'WM checkpoint step.')
flags.DEFINE_string('policy_root', '/scratch/jwquan/wmpp/policies/p1/OGBench/p1-pilot', 'Bank root.')
flags.DEFINE_integer('policy_epoch', 1000000, 'Policy epoch.')
flags.DEFINE_string('seeds', '0', 'Bank training seeds to include (comma-separated).')
flags.DEFINE_string('best_fixed', None, 'Best-fixed policy name (from ID validation).', required=True)
flags.DEFINE_integer('horizon', None, 'Scoring horizon H (default: WM training horizon).')
flags.DEFINE_integer('commit', None, 'Commitment C (default: same as horizon).')
flags.DEFINE_string('variants', None, 'Subset of 2x2 cells to run (comma-separated names).')
flags.DEFINE_string('kc_sweep', None, 'Extra (k,k) diagonal cells, e.g. "5,25,50".')
flags.DEFINE_string('out_tag', None, 'Suffix for the output dir (avoid overwriting).')
flags.DEFINE_enum(
    'score_mode',
    'progress',
    ['progress', 'value'],
    'Rollout scoring: privileged task-dim progress, or the learned '
    'reachability value head alone (no task knowledge).',
)
flags.DEFINE_enum(
    'score_agg',
    'max',
    ['max', 'mean', 'last'],
    'Horizon aggregation of per-step scores: best state visited (max), '
    'path average (mean), or final imagined state (last).',
)
flags.DEFINE_string(
    'random_commit',
    None,
    'If set, also run the random-arbitration control: a uniform draw from the '
    'full bank every `random_commit` env steps (no world model, no value head). '
    'Comma-separated list runs one control per commitment interval.',
)
flags.DEFINE_bool(
    'skip_fixed',
    False,
    'Do not evaluate the fixed bank policies (official-protocol runs compare '
    'against eval.csv baselines instead); paired_vs_best_fixed and oracle '
    'headroom are skipped.',
)
flags.DEFINE_integer('random_seed', 0, 'Seed of the random-arbitration draw stream.')
# Sampling-MPC baseline on the best fixed policy (world-model search WITHOUT
# a portfolio): N Gaussian perturbations of the policy action, imagined k
# steps, value-head scored, replanned every c steps. Off when --mpc_n=0.
flags.DEFINE_integer('mpc_n', 0, 'Number of MPC candidates incl. the unperturbed policy mean (0 = off).')
flags.DEFINE_float('mpc_sigma', 0.2, 'Std of the action perturbation (= actor temperature).')
flags.DEFINE_string('mpc_k', '1', 'Comma-separated MPC imagination horizons k.')
flags.DEFINE_string('mpc_commit', '1', 'Comma-separated MPC commit intervals c.')
flags.DEFINE_string('mpc_policy', None, 'Policy used as the MPC prior (default: --best_fixed).')
# No-model control for MPC: the prior policy with a random perturbation
# (sigma) applied every c steps (variant noisy_s<sigma>_every<c>).
flags.DEFINE_string('noisy_sigma', None, 'Comma-separated sigmas of the NoisyPolicy control (off if unset).')
flags.DEFINE_string('noisy_every', '1', 'Comma-separated perturbation intervals c of the NoisyPolicy control.')
flags.DEFINE_integer('pmpc_n_per', 0, 'Portfolio-MPC: candidates per bank policy incl. its mean (0 = off); uses mpc_sigma and the mpc_k/mpc_commit/mpc_kc cells.')
flags.DEFINE_string('mpc_kc', None, 'Comma-separated diagonal MPC cells k=c (in addition to the mpc_k x mpc_commit product; "" to skip the product).')
# Scheduled exploration on top of WMPP: every requested WMPP variant is run
# with the least-used policy forced at the first replan >= m env steps after
# the last exploration (variant name <v>_explore<m>); the plain variants are
# NOT re-run when this flag is set (they exist in the og50 dirs).
flags.DEFINE_string('explore_every', None, 'Comma-separated exploration intervals m (env steps).')
flags.DEFINE_string('least_used_commit', None, 'Comma-separated commit intervals for the LeastUsed (round-robin) control.')
flags.DEFINE_string('least_used_order', 'name', 'Comma-separated cycle orders for LeastUsed: name, reverse, shuffle (variant suffix _<order> unless name).')
flags.DEFINE_integer('episodes_per_task', 20, 'Paired episodes per task.')
flags.DEFINE_string('oracle_dir', '/scratch/jwquan/wmpp/oracle', 'Oracle root (headroom reference).')
flags.DEFINE_string('out_dir', '/scratch/jwquan/wmpp/planner_eval', 'Output root.')


def env_id_of(dataset_name):
    tokens = dataset_name.split('-')
    return '-'.join(tokens[:-2] + tokens[-1:])


def chain_stats(rows, bank_names):
    """Aggregate policy-chain / switching / overhead stats over episodes."""
    from collections import Counter

    usage = Counter()
    chain_counter = Counter()
    switches, plans, plan_ms = [], [], []
    for r in rows:
        segments = [seg.rsplit(':', 1) for seg in r['policy_chain'].split('>') if seg]
        for name, steps in segments:
            usage[name] += int(steps)
        chain_counter['>'.join(name for name, _ in segments)] += 1
        switches.append(int(r['n_switches']))
        plans.append(int(r['n_plans']))
        plan_ms.append(float(r['plan_ms_total']))
    total_steps = sum(usage.values())
    return dict(
        usage_fraction={n: usage.get(n, 0) / total_steps for n in bank_names},
        mean_switches_per_episode=float(np.mean(switches)),
        mean_plans_per_episode=float(np.mean(plans)),
        plan_ms_per_decision=float(np.sum(plan_ms) / max(np.sum(plans), 1)),
        top_chains=[
            {'chain': c, 'episodes': n} for c, n in chain_counter.most_common(5)
        ],
    )


def bootstrap_delta(rows_a, rows_b, n_boot=10000, seed=0):
    """Paired episode-level bootstrap of mean success difference (a - b)."""
    key = lambda r: (r['task_id'], r['episode_idx'])
    b_by_key = {key(r): r['success'] for r in rows_b}
    deltas = np.array([r['success'] - b_by_key[key(r)] for r in rows_a])
    rng = np.random.default_rng(seed)
    boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)])
    return dict(
        delta=float(deltas.mean()),
        ci_lo=float(np.percentile(boot, 2.5)),
        ci_hi=float(np.percentile(boot, 97.5)),
        n_episodes=len(deltas),
    )


def main(_):
    seeds = [int(s) for s in FLAGS.seeds.split(',')]
    bank = load_bank(FLAGS.env_name, FLAGS.policy_root, FLAGS.policy_epoch, seeds=seeds)
    assert FLAGS.best_fixed in bank, (FLAGS.best_fixed, sorted(bank))
    wm = EnsembleWorldModel.load(FLAGS.wm_dir, FLAGS.wm_epoch)
    horizon = FLAGS.horizon or int(wm.config['horizon'])

    progress_fn = get_progress_fn(FLAGS.env_name)
    # 2x2 ablation cells: (scoring horizon k) x (commitment c). Budget per env
    # step is P*E*k/c — deliberately NOT equalized across cells (the ablation
    # isolates which knob carries the gain); recorded per variant below.
    commit = FLAGS.commit or horizon
    variant_specs = {
        'score1_commit1': (1, 1),
        f'score1_commit{commit}': (1, commit),
        f'score{horizon}_commit1': (horizon, 1),
        f'score{horizon}_commit{commit}': (horizon, commit),
    }
    if FLAGS.kc_sweep:
        for h in (int(x) for x in FLAGS.kc_sweep.split(',')):
            variant_specs[f'score{h}_commit{h}'] = (h, h)
    # --variants=none runs no world-model planner (e.g. a Random-only job).
    if FLAGS.variants == 'none':
        requested = []
    else:
        requested = FLAGS.variants.split(',') if FLAGS.variants else list(variant_specs)
    counters = {}
    methods = dict(bank)
    explore = [int(x) for x in FLAGS.explore_every.split(',')] if FLAGS.explore_every else [None]
    plain = list(requested)
    requested = []
    for base in plain:
        k, c = variant_specs[base]
        for m in explore:
            name = base if m is None else f'{base}_explore{m}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = RolloutRanker(
                wm,
                bank,
                counters[name],
                horizon=k,
                replan_every=c,
                progress_fn=progress_fn,
                score_mode=FLAGS.score_mode,
                score_agg=FLAGS.score_agg,
                explore_every=m,
            )
            requested.append(name)

    if FLAGS.random_commit:
        # Control: same commitment interval, uniform policy draw, zero model
        # calls. Recorded as a variant with imagine=0 so the budget table
        # shows its nominal cost (0) next to the WM planners.
        for rc in (int(x) for x in FLAGS.random_commit.split(',')):
            name = f'random_commit{rc}'
            variant_specs[name] = (0, rc)
            counters[name] = TransitionCounter()
            methods[name] = RandomArbiter(bank, rc, seed=FLAGS.random_seed)
            requested.append(name)
    if FLAGS.least_used_commit:
        for lc in (int(x) for x in FLAGS.least_used_commit.split(',')):
            for order in FLAGS.least_used_order.split(','):
                name = f'leastused_commit{lc}' + ('' if order == 'name' else f'_{order}')
                variant_specs[name] = (0, lc)
                counters[name] = TransitionCounter()
                methods[name] = LeastUsedArbiter(bank, lc, order=order, seed=FLAGS.random_seed)
                requested.append(name)
    # Nominal branch count per variant (P for bank planners, N for MPC).
    branches = {name: len(bank) for name in requested}
    mpc_cells = [(k, c) for k in (int(x) for x in FLAGS.mpc_k.split(',') if x)
                 for c in (int(x) for x in FLAGS.mpc_commit.split(',') if x)]
    if FLAGS.mpc_kc:
        mpc_cells += [(int(x), int(x)) for x in FLAGS.mpc_kc.split(',')]
    mpc_cells = list(dict.fromkeys(mpc_cells))
    if FLAGS.pmpc_n_per:
        for k, c in mpc_cells:
            name = f'pmpc{FLAGS.pmpc_n_per}_s{FLAGS.mpc_sigma:g}_score{k}_commit{c}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = PortfolioMPC(wm, bank, counters[name], n_per=FLAGS.pmpc_n_per, sigma=FLAGS.mpc_sigma,
                                         horizon=k, replan_every=c, score_agg=FLAGS.score_agg, seed=FLAGS.random_seed)
            requested.append(name)
            branches[name] = FLAGS.pmpc_n_per * len(bank)
    if FLAGS.mpc_n:
        mpc_policy = FLAGS.mpc_policy or FLAGS.best_fixed
        assert mpc_policy in bank, (mpc_policy, sorted(bank))
        for k, c in mpc_cells:
            name = f'mpc{FLAGS.mpc_n}_s{FLAGS.mpc_sigma:g}_score{k}_commit{c}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = PolicyMPC(
                wm, bank[mpc_policy], mpc_policy, counters[name],
                n_samples=FLAGS.mpc_n, sigma=FLAGS.mpc_sigma, horizon=k,
                replan_every=c, score_agg=FLAGS.score_agg, seed=FLAGS.random_seed,
            )
            requested.append(name)
            branches[name] = FLAGS.mpc_n
    if FLAGS.noisy_sigma:
        noisy_policy = FLAGS.mpc_policy or FLAGS.best_fixed
        for sg in (float(x) for x in FLAGS.noisy_sigma.split(',')):
            for c in (int(x) for x in FLAGS.noisy_every.split(',')):
                name = f'noisy_s{sg:g}_every{c}'
                variant_specs[name] = (0, c)
                counters[name] = TransitionCounter()
                methods[name] = NoisyPolicyArbiter(bank[noisy_policy], noisy_policy, sg, c, seed=FLAGS.random_seed)
                requested.append(name)
                branches[name] = 1
    if FLAGS.skip_fixed:
        for pol in bank:
            del methods[pol]

    env = gymnasium.make(env_id_of(FLAGS.env_name))
    tag = f'_{FLAGS.out_tag}' if FLAGS.out_tag else ''
    out_dir = os.path.join(FLAGS.out_dir, FLAGS.env_name, f'bank_sd{FLAGS.seeds}{tag}')
    os.makedirs(out_dir, exist_ok=True)
    rows = evaluate_paired(
        env,
        FLAGS.env_name,
        methods,
        task_ids=[1, 2, 3, 4, 5],
        episodes_per_task=FLAGS.episodes_per_task,
        out_csv=os.path.join(out_dir, 'episodes.csv'),
    )
    env.close()

    by_method = {}
    for r in rows:
        by_method.setdefault(r['policy'], []).append(r)

    E, P = wm.config['num_members'], len(bank)
    per_step, expected = {}, {}
    for name in requested:
        k, c = variant_specs[name]
        env_steps = sum(r['steps'] for r in by_method[name])
        per_step[name] = counters[name].total / env_steps
        expected[name] = E * branches[name] * k / c
        if c == 1 or k == 0:
            # Replanning every step (or never imagining) makes the budget exact.
            assert abs(per_step[name] - expected[name]) < 1e-9, (name, per_step[name])
        else:
            # A replan costs P*E*k up front; an episode ending shortly after a
            # replan amortizes it over few steps, so the average may exceed
            # the nominal rate — unboundedly so when episodes terminate within
            # one commit window (near-saturated envs). Report, never hide.
            if per_step[name] > expected[name] * 1.5:
                print(f'[budget] {name}: actual {per_step[name]:.1f}/step vs nominal '
                      f'{expected[name]:.1f} (short-episode amortization; recorded in summary)')

    summary = dict(
        env_name=FLAGS.env_name,
        bank=sorted(bank),
        bank_seeds=FLAGS.seeds,
        best_fixed=FLAGS.best_fixed,
        horizon=horizon,
        commit=commit,
        score_mode=FLAGS.score_mode,
        score_agg=FLAGS.score_agg,
        variants={n: dict(imagine=variant_specs[n][0], commit=variant_specs[n][1]) for n in requested},
        success=dict(
            sorted(
                (name, float(np.mean([r['success'] for r in rs])))
                for name, rs in by_method.items()
            )
        ),
        paired_vs_best_fixed=(
            {}
            if FLAGS.skip_fixed
            else {
                name: bootstrap_delta(by_method[name], by_method[FLAGS.best_fixed])
                for name in requested
            }
        ),
        budget_per_env_step=dict(actual=per_step, nominal=expected),
        # Model-call accounting per planner: imagined transitions (dynamics)
        # and value-head evaluations, totals over all episodes and per env
        # step / per episode. Fixed policies and the random control are 0.
        model_calls={
            name: dict(
                dynamics_total=int(sum(int(r['wm_transitions']) for r in by_method[name])),
                value_total=int(sum(int(r['value_evals']) for r in by_method[name])),
                dynamics_per_env_step=float(
                    sum(int(r['wm_transitions']) for r in by_method[name])
                    / sum(r['steps'] for r in by_method[name])
                ),
                value_per_env_step=float(
                    sum(int(r['value_evals']) for r in by_method[name])
                    / sum(r['steps'] for r in by_method[name])
                ),
                dynamics_per_episode=float(np.mean([int(r['wm_transitions']) for r in by_method[name]])),
                value_per_episode=float(np.mean([int(r['value_evals']) for r in by_method[name]])),
            )
            for name in requested
        },
        random_seed=FLAGS.random_seed if (FLAGS.random_commit or FLAGS.mpc_n) else None,
        mpc=(dict(n=FLAGS.mpc_n, sigma=FLAGS.mpc_sigma, policy=FLAGS.mpc_policy or FLAGS.best_fixed)
             if FLAGS.mpc_n else None),
        # Compute overhead: mean wall-clock per action, per method. Fixed
        # policies are the baseline; the planner surplus is the WM overhead.
        act_ms_per_step={
            name: float(np.mean([r['act_ms_per_step'] for r in rs]))
            for name, rs in sorted(by_method.items())
        },
        policy_chains={
            name: chain_stats(by_method[name], sorted(bank)) for name in requested
        },
    )

    branches_path = os.path.join(FLAGS.oracle_dir, FLAGS.env_name, 'branches.npz')
    if os.path.exists(branches_path) and not FLAGS.skip_fixed:
        branches = dict(np.load(branches_path, allow_pickle=False))
        names = [str(p) for p in branches['policies']]
        pool = [n for n in names if n in bank]
        # Oracle branches may predate this bank (e.g. collected with seed-0
        # policies only); headroom is meaningless unless it covers both the
        # comparison policy and some of the current bank.
        if FLAGS.best_fixed in names and pool:
            oracle = headroom(branches, FLAGS.best_fixed, policies=pool)
            summary['oracle_headroom'] = oracle
            if oracle['headroom'] > 0:
                summary['headroom_captured'] = {
                    name: summary['paired_vs_best_fixed'][name]['delta'] / oracle['headroom']
                    for name in requested
                }
        else:
            print(f'Skipping oracle headroom: {branches_path} does not cover this bank.')

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    app.run(main)
