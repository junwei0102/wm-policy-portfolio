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
from planners.portfolio import CriticSelectArbiter, LeastUsedArbiter, NoisyPolicyArbiter, PolicyMPC, PortfolioMPC, RandomArbiter, RolloutRanker, StallRestartArbiter
from planners.sim_rollout import SimOracleArbiter, SimRolloutRanker
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter
from world_model.success_predicates import get_progress_fn

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'OGBench dataset name.', required=True)
flags.DEFINE_string('wm_dir', None, 'World-model run dir.', required=True)
flags.DEFINE_integer('wm_epoch', 500000, 'WM checkpoint step.')
flags.DEFINE_string('policy_root', '/path/to/policies/p1-pilot', 'Bank root.')
flags.DEFINE_integer('policy_epoch', 1000000, 'Policy epoch.')
flags.DEFINE_string('seeds', '0', 'Bank training seeds to include (comma-separated).')
flags.DEFINE_string('best_fixed', None, 'Best-fixed policy name (from ID validation).', required=True)
flags.DEFINE_integer('horizon', None, 'Scoring horizon H (default: WM training horizon).')
flags.DEFINE_integer('commit', None, 'Commitment C (default: same as horizon).')
flags.DEFINE_string('variants', None, 'Subset of 2x2 cells to run (comma-separated names).')
flags.DEFINE_string('kc_sweep', None, 'Extra (k,k) diagonal cells, e.g. "5,25,50".')
flags.DEFINE_string('kxc', None, 'Extra arbitrary (k,c) cells as "k:c" pairs, e.g. "10:5,5:1" (variant score{k}_commit{c}); select them with --variants.')
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
flags.DEFINE_string(
    'critic_scorer',
    None,
    "Ablation of the LAVL head: score imagined states with this bank member's own "
    "goal-conditioned value network (e.g. 'gciql-sd0'; a bare family name such as "
    "'gciql' resolves to that family's member for the single --seeds). Adds variants "
    'critic{k}_commit{k} for every k in --critic_kc.',
)
flags.DEFINE_string('critic_kc', None, 'k=c cells for the --critic_scorer variants, e.g. "5,100".')
flags.DEFINE_string('critic_kxc', None, 'Off-diagonal (k,c) cells for the --critic_scorer variants as "k:c" pairs, e.g. "1:10,10:1" (variant critic{k}_commit{c}).')
flags.DEFINE_string('critic_select_commit', None, 'Re-selection intervals c for the model-free Q-select control qsel_commit{c}: i*=argmax_i min_j Q_j(s, pi_i(s,g), g) of the --critic_scorer member, no rollout.')
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
# Scheduled exploration on top of WMPA: every requested WMPA variant is run
# with the least-used policy forced at the first replan >= m env steps after
# the last exploration (variant name <v>_explore<m>); the plain variants are
# NOT re-run when this flag is set (they exist in the og50 dirs).
flags.DEFINE_string('explore_every', None, 'Comma-separated exploration intervals m (env steps).')
flags.DEFINE_string('least_used_commit', None, 'Comma-separated commit intervals for the LeastUsed (round-robin) control.')
flags.DEFINE_string('least_used_order', 'name', 'Comma-separated cycle orders for LeastUsed: name, reverse, shuffle (variant suffix _<order> unless name).')
# Review-driven ablations / controls (all on the same paired episodes).
flags.DEFINE_enum('ens_agg', 'mean', ['mean', 'min', 'lcb'], 'Ensemble reduction of the value scores in every WM planner (paper: mean).')
flags.DEFINE_string('abl_agg', None, 'Comma-separated horizon aggregations (last,mean) to ADD as variants <v>_agg<x> of every requested WMPA variant.')
flags.DEFINE_string('abl_ens', None, 'Comma-separated ensemble reductions (min,lcb) to ADD as variants <v>_ens<x> of every requested WMPA variant.')
flags.DEFINE_bool('abl_only', False, 'With --abl_agg/--abl_ens: do not run the plain requested variants themselves (they exist elsewhere).')
flags.DEFINE_string('stall_window', None, 'Comma-separated windows m of the no-model stall-restart controller (variant stall_w<m>).')
flags.DEFINE_float('stall_eps', 0.05, 'Stall threshold: normalised displacement over the window below which the policy is restarted.')
flags.DEFINE_string('sim_kc', None, 'Comma-separated diagonal cells k=c to run with TRUE-simulator branches instead of the WM (variant sim_score<k>_commit<k>).')
flags.DEFINE_string('bank_algos', None, 'Comma-separated algorithm families to keep in the bank (default: all).')
flags.DEFINE_string('bank_exclude', None, 'Comma-separated bank policy names to drop (e.g. the best policy).')
flags.DEFINE_string('bank_duplicate', None, 'Bank policy name to add a second time (as <name>-dup).')
flags.DEFINE_string('out_label', None, 'Override the bank_sd<seeds> part of the output dir (bank-composition runs).')
flags.DEFINE_string('episode_range', None, 'START:END — evaluate only these episode indices (seeds unchanged); '
                    'the official test set is 0:50, a validation split uses fresh indices such as 50:100.')
flags.DEFINE_integer('flush_every', 0, 'Rewrite episodes.csv every N episodes (0 = only at the end).')
flags.DEFINE_bool('dump_decisions', False, 'Record (t, obs, goal, scores, winner) at every arbitration of the plain '
                  'WMPA cells and RandomArbiter controls; written as decisions_<variant>.npz next to episodes.csv.')
flags.DEFINE_string('bank_extra', None, 'Comma list of <env_name>:<policy_root>[:<suffix>] — policies trained on another '
                    'dataset of the same env pooled into the bank as <algo>-<suffix>-sd<seed> (cross-dataset bank).')
flags.DEFINE_string('sim_oracle_commit', None, 'Comma list of c: dynamic simulator oracle sim_oracle_commit<c> '
                    '(every bank policy branched in the real simulator to episode end; privileged upper bound).')
flags.DEFINE_integer('episodes_per_task', 20, 'Paired episodes per task.')
flags.DEFINE_string('oracle_dir', '/path/to/oracle', 'Oracle root (headroom reference).')
flags.DEFINE_string('out_dir', '/path/to/planner_eval', 'Output root.')


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
    algos = FLAGS.bank_algos.split(',') if FLAGS.bank_algos else None
    bank = load_bank(FLAGS.env_name, FLAGS.policy_root, FLAGS.policy_epoch, algos=algos, seeds=seeds)
    if FLAGS.bank_extra:
        for item in FLAGS.bank_extra.split(','):
            parts = item.split(':')
            other_env, other_root = parts[0], parts[1]
            suffix = parts[2] if len(parts) > 2 else other_env.split('-')[-2]  # e.g. 'noisy'
            extra_bank = load_bank(other_env, other_root, FLAGS.policy_epoch, algos=algos, seeds=seeds, suffix=suffix)
            assert extra_bank, f'no policies for {other_env} under {other_root}'
            assert not set(extra_bank) & set(bank), sorted(set(extra_bank) & set(bank))
            bank.update(extra_bank)
            print(f'bank_extra: +{len(extra_bank)} policies from {other_env} ({suffix})')
    if FLAGS.bank_exclude:
        for n in FLAGS.bank_exclude.split(','):
            assert n in bank, (n, sorted(bank))
            del bank[n]
    if FLAGS.bank_duplicate:
        assert FLAGS.bank_duplicate in bank, (FLAGS.bank_duplicate, sorted(bank))
        bank[f'{FLAGS.bank_duplicate}-dup'] = bank[FLAGS.bank_duplicate]
    print(f'bank ({len(bank)}): {sorted(bank)}')
    assert FLAGS.best_fixed in bank or FLAGS.skip_fixed, (FLAGS.best_fixed, sorted(bank))
    wm = EnsembleWorldModel.load(FLAGS.wm_dir, FLAGS.wm_epoch)
    horizon = FLAGS.horizon or int(wm.config['horizon'])

    # The privileged progress scorer only exists for a few envs and is only
    # used by --score_mode=progress; value-mode runs must not require it.
    try:
        progress_fn = get_progress_fn(FLAGS.env_name)
    except NotImplementedError:
        assert FLAGS.score_mode != 'progress', f'no progress fn for {FLAGS.env_name}'
        progress_fn = None
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
    for spec in (FLAGS.kxc.split(',') if FLAGS.kxc else []):
        k_, c_ = (int(x) for x in spec.split(':'))
        variant_specs[f'score{k_}_commit{c_}'] = (k_, c_)
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
            if FLAGS.abl_only:
                variant_specs[name] = (k, c)
                continue
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
                ens_agg=FLAGS.ens_agg,
                log_decisions=FLAGS.dump_decisions,
            )
            requested.append(name)
        # Scoring ablations of the same cell: horizon aggregation / ensemble reduction.
        for agg in (FLAGS.abl_agg.split(',') if FLAGS.abl_agg else []):
            name = f'{base}_agg{agg}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = RolloutRanker(wm, bank, counters[name], horizon=k, replan_every=c, progress_fn=progress_fn,
                                          score_mode=FLAGS.score_mode, score_agg=agg, ens_agg=FLAGS.ens_agg)
            requested.append(name)
        for ens in (FLAGS.abl_ens.split(',') if FLAGS.abl_ens else []):
            name = f'{base}_ens{ens}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = RolloutRanker(wm, bank, counters[name], horizon=k, replan_every=c, progress_fn=progress_fn,
                                          score_mode=FLAGS.score_mode, score_agg=FLAGS.score_agg, ens_agg=ens)
            requested.append(name)

    critic_name, critic_fn = None, None
    if FLAGS.critic_scorer:
        critic_name = FLAGS.critic_scorer
        if critic_name not in bank:
            seeds_l = [int(x) for x in FLAGS.seeds.split(',')]
            assert len(seeds_l) == 1, '--critic_scorer family name needs a single --seeds'
            critic_name = f'{critic_name}-sd{seeds_l[0]}'
        assert critic_name in bank, (critic_name, sorted(bank))
        critic_fn = bank[critic_name].value
        for k in (int(x) for x in FLAGS.critic_kc.split(',')) if FLAGS.critic_kc else []:
            name = f'critic{k}_commit{k}'
            assert name not in methods, name
            variant_specs[name] = (k, k)
            counters[name] = TransitionCounter()
            methods[name] = RolloutRanker(wm, bank, counters[name], horizon=k, replan_every=k, progress_fn=progress_fn,
                                          score_mode='critic', score_agg=FLAGS.score_agg, ens_agg=FLAGS.ens_agg,
                                          critic_fn=critic_fn)
            requested.append(name)
            # the same scoring ablations as for the metric value (horizon aggregation / ensemble reduction)
            for agg in (FLAGS.abl_agg.split(',') if FLAGS.abl_agg else []):
                nm = f'{name}_agg{agg}'; variant_specs[nm] = (k, k); counters[nm] = TransitionCounter()
                methods[nm] = RolloutRanker(wm, bank, counters[nm], horizon=k, replan_every=k, progress_fn=progress_fn,
                                            score_mode='critic', score_agg=agg, ens_agg=FLAGS.ens_agg, critic_fn=critic_fn)
                requested.append(nm)
            for ens in (FLAGS.abl_ens.split(',') if FLAGS.abl_ens else []):
                nm = f'{name}_ens{ens}'; variant_specs[nm] = (k, k); counters[nm] = TransitionCounter()
                methods[nm] = RolloutRanker(wm, bank, counters[nm], horizon=k, replan_every=k, progress_fn=progress_fn,
                                            score_mode='critic', score_agg=FLAGS.score_agg, ens_agg=ens, critic_fn=critic_fn)
                requested.append(nm)
            if FLAGS.abl_only:  # keep only the ablation variants (the plain cell already exists in the sweep)
                requested.remove(name); methods.pop(name); counters.pop(name)
        # arbitrary (k, c) cells of the same critic scorer (the (k,c) sensitivity grid on the puzzles)
        for spec in (FLAGS.critic_kxc.split(',') if FLAGS.critic_kxc else []):
            k, c = (int(x) for x in spec.split(':'))
            name = f'critic{k}_commit{c}'
            assert name not in methods, name
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = RolloutRanker(wm, bank, counters[name], horizon=k, replan_every=c, progress_fn=progress_fn,
                                          score_mode='critic', score_agg=FLAGS.score_agg, ens_agg=FLAGS.ens_agg,
                                          critic_fn=critic_fn)
            requested.append(name)
        # model-free control: the same member's twin-Q critic ranks each candidate's proposed action at the current state
        for c in (int(x) for x in FLAGS.critic_select_commit.split(',')) if FLAGS.critic_select_commit else []:
            name = f'qsel_commit{c}'
            assert name not in methods, name
            variant_specs[name] = (0, c)  # k=0: no imagined transitions (budget assertion expects exactly zero)
            counters[name] = TransitionCounter()
            methods[name] = CriticSelectArbiter(bank, bank[critic_name].q_min, c, seed=FLAGS.random_seed)
            requested.append(name)

    if FLAGS.stall_window:
        # No-model restart heuristic: switch when the normalised state stalls.
        for m in (int(x) for x in FLAGS.stall_window.split(',')):
            name = f'stall_w{m}'
            variant_specs[name] = (0, 1)
            counters[name] = TransitionCounter()
            methods[name] = StallRestartArbiter(bank, wm.normalizer['obs_mean'], wm.normalizer['obs_std'], m,
                                                eps=FLAGS.stall_eps, seed=FLAGS.random_seed)
            requested.append(name)
    if FLAGS.random_commit:
        # Control: same commitment interval, uniform policy draw, zero model
        # calls. Recorded as a variant with imagine=0 so the budget table
        # shows its nominal cost (0) next to the WM planners.
        for rc in (int(x) for x in FLAGS.random_commit.split(',')):
            name = f'random_commit{rc}'
            variant_specs[name] = (0, rc)
            counters[name] = TransitionCounter()
            methods[name] = RandomArbiter(bank, rc, seed=FLAGS.random_seed, log_decisions=FLAGS.dump_decisions)
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
        prior = bank[mpc_policy]
        for k, c in mpc_cells:
            pref = 'critic' if critic_fn is not None else 'score'  # which value scores the candidates
            name = f'mpc{FLAGS.mpc_n}_s{FLAGS.mpc_sigma:g}_{pref}{k}_commit{c}'
            variant_specs[name] = (k, c)
            counters[name] = TransitionCounter()
            methods[name] = PolicyMPC(
                wm, prior, mpc_policy, counters[name],
                n_samples=FLAGS.mpc_n, sigma=FLAGS.mpc_sigma, horizon=k,
                replan_every=c, score_agg=FLAGS.score_agg, seed=FLAGS.random_seed,
                critic_fn=critic_fn,
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
    sim_names = []
    if FLAGS.sim_kc:
        # True-simulator branches (oracle dynamics) with the same value head / aggregation.
        for k in (int(x) for x in FLAGS.sim_kc.split(',')):
            name = f"sim_{'critic' if critic_fn is not None else 'score'}{k}_commit{k}"
            variant_specs[name] = (k, k)
            counters[name] = TransitionCounter()
            methods[name] = SimRolloutRanker(env, wm, bank, counters[name], horizon=k, replan_every=k,
                                             score_agg=FLAGS.score_agg, ens_agg=FLAGS.ens_agg, critic_fn=critic_fn)
            requested.append(name)
            branches[name] = len(bank)
            sim_names.append(name)
    if FLAGS.sim_oracle_commit:
        # Privileged dynamic oracle: real-simulator branches to episode end at every boundary.
        for c in (int(x) for x in FLAGS.sim_oracle_commit.split(',')):
            name = f'sim_oracle_commit{c}'
            variant_specs[name] = (0, c)
            counters[name] = TransitionCounter()
            methods[name] = SimOracleArbiter(env, bank, counters[name], commit=c,
                                             episode_len=int(env.spec.max_episode_steps), fallback=FLAGS.best_fixed)
            requested.append(name)
            branches[name] = len(bank)
            sim_names.append(name)
    episode_range = tuple(int(x) for x in FLAGS.episode_range.split(':')) if FLAGS.episode_range else None
    episode_len = int(env.spec.max_episode_steps)
    tag = f'_{FLAGS.out_tag}' if FLAGS.out_tag else ''
    out_dir = os.path.join(FLAGS.out_dir, FLAGS.env_name, f'bank_sd{FLAGS.seeds}{tag}' if not FLAGS.out_label else f'{FLAGS.out_label}{tag}')
    os.makedirs(out_dir, exist_ok=True)
    rows = evaluate_paired(
        env,
        FLAGS.env_name,
        methods,
        task_ids=[1, 2, 3, 4, 5],
        episodes_per_task=FLAGS.episodes_per_task,
        out_csv=os.path.join(out_dir, 'episodes.csv'),
        episode_range=episode_range,
        flush_every=FLAGS.flush_every or None,
        collect_decisions=FLAGS.dump_decisions,
    )
    if FLAGS.dump_decisions:
        rows, decisions = rows
        bank_names = sorted(bank)
        for name, eps in decisions.items():
            n = np.array([len(d['t']) for d in eps])
            packed = {k: np.concatenate([d[k] for d in eps]) for k in ('t', 'obs', 'goal', 'scores', 'winner', 'explore')}
            for k in ('task_id', 'episode_idx', 'reset_seed', 'ep_success', 'ep_steps'):
                packed[k] = np.repeat(np.array([d[k] for d in eps]), n)
            packed['policy_idx'] = packed.pop('winner')
            packed['policies'] = np.array(bank_names)
            np.savez_compressed(os.path.join(out_dir, f'decisions_{name}.npz'), **packed)
            print(f'decisions_{name}.npz: {int(n.sum())} decisions over {len(eps)} episodes')
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
        if name in sim_names:
            expected[name] = branches[name] * k / c  # simulator steps, no ensemble factor
            continue
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
        episode_range=list(episode_range) if episode_range else [0, FLAGS.episodes_per_task],
        episode_len=episode_len,
        bank_extra=FLAGS.bank_extra,
        best_fixed=FLAGS.best_fixed,
        horizon=horizon,
        commit=commit,
        score_mode=FLAGS.score_mode,
        score_agg=FLAGS.score_agg,
        ens_agg=FLAGS.ens_agg,
        stall_eps=FLAGS.stall_eps if FLAGS.stall_window else None,
        bank_composition=dict(algos=FLAGS.bank_algos, exclude=FLAGS.bank_exclude, duplicate=FLAGS.bank_duplicate, size=len(bank)),
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

    summary['critic_scorer'] = critic_name  # bank member whose V(s, g) scored critic* variants (None = LAVL head)
    summary['critic_select'] = {'critic': critic_name, 'commits': [int(x) for x in FLAGS.critic_select_commit.split(',')]} if FLAGS.critic_select_commit else None
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    app.run(main)
