"""WM ranking-competence diagnostics: imagined scores vs real oracle branches.

Zero simulator touches: imagines from states.npz (obs, goal) with the frozen
bank, compares against branches.npz outcomes. One max-horizon rollout per
(state, policy) provides all smaller-horizon scores via prefix maxima.

Writes <out_dir>/diagnostics.json and prints the GO/NO-GO verdict.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from absl import app, flags

from evaluation.diagnostics import (
    calibration,
    gate_verdict,
    horizon_of_validity,
    pairwise_ranking_accuracy,
    spearman,
    top1_winner_rate,
)
from interfaces.policy_bank import load_bank
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter, imagine_policy_rollout, open_loop_error
from world_model.success_predicates import get_progress_fn

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'OGBench dataset name.', required=True)
flags.DEFINE_string('wm_dir', None, 'World-model run dir.', required=True)
flags.DEFINE_integer('wm_epoch', 500000, 'World-model checkpoint step.')
flags.DEFINE_string('oracle_dir', '/scratch/jwquan/wmpp/oracle', 'Oracle output root.')
flags.DEFINE_string('policy_root', '/scratch/jwquan/wmpp/policies/p1/OGBench/p1-pilot', 'Bank root.')
flags.DEFINE_integer('policy_epoch', 1000000, 'Policy checkpoint epoch.')
flags.DEFINE_string('horizons', '1,2,5,10,20,50', 'Score horizons (max = rollout length).')
flags.DEFINE_integer('openloop_branches', 200, 'Branches sampled for open-loop error.')
flags.DEFINE_string('out', None, 'Output JSON (default: <oracle_dir>/<env>/diagnostics.json).')


def main(_):
    oracle_dir = os.path.join(FLAGS.oracle_dir, FLAGS.env_name)
    states = dict(np.load(os.path.join(oracle_dir, 'states.npz'), allow_pickle=False))
    branches = dict(np.load(os.path.join(oracle_dir, 'branches.npz'), allow_pickle=False))
    wm = EnsembleWorldModel.load(FLAGS.wm_dir, FLAGS.wm_epoch)
    bank = load_bank(FLAGS.env_name, FLAGS.policy_root, FLAGS.policy_epoch)

    policies = [str(p) for p in branches['policies']]
    assert set(policies) <= set(bank), set(policies) - set(bank)
    horizons = sorted(int(h) for h in FLAGS.horizons.split(','))
    H_max = horizons[-1]
    S, P = len(states['state_id']), len(policies)

    # Real outcomes matrix (S, P).
    outcomes = np.full((S, P), np.nan)
    outcomes[branches['state_id'], branches['policy_idx']] = branches['success']
    assert not np.isnan(outcomes).any(), 'missing (state, policy) branches'

    # Imagined scores: one H_max rollout per (state, policy); prefix maxima.
    # Two score families from the SAME rollout:
    #   progress: best task-relevant goal distance achieved within H
    #            (privileged diagnostic), and
    #   value: the learned LAVL value head under three horizon aggregations.
    # (`scores` — the former success-head family — is kept as zeros so the
    # downstream summary keys stay stable.)
    try:
        progress_fn = get_progress_fn(FLAGS.env_name)
    except NotImplementedError:
        progress_fn = None  # no privileged progress diagnostic for this env (e.g. pointmaze)
        print('[diag] no progress fn for this env; progress_* blocks will be null')
    has_value = True
    has_succ = False  # success head removed (2026-08-27); its branches below are inert
    counter = TransitionCounter()
    h_idx = np.array(horizons) - 1
    scores = np.zeros((S, P, len(horizons)), dtype=np.float64)
    prog_scores = np.zeros((S, P, len(horizons)), dtype=np.float64)
    # Value family under all three horizon aggregations (same rollouts):
    # max = best state visited, mean = running path average, last = state at h.
    value_scores = {
        agg: np.zeros((S, P, len(horizons)), dtype=np.float64) for agg in ('max', 'mean', 'last')
    }
    for s in range(S):
        obs0, goal = states['obs'][s], states['goal'][s]
        for p, name in enumerate(policies):
            roll = imagine_policy_rollout(wm, bank[name], obs0, goal, H_max, counter)
            traj = roll['obs_traj'][1:]  # (H_max, E, d)
            if progress_fn is not None:
                prog = np.stack(
                    [progress_fn(traj[:, e], np.tile(goal, (H_max, 1))) for e in range(traj.shape[1])]
                ).mean(axis=0)  # (H_max,)
                prog_scores[s, p] = np.maximum.accumulate(prog)[h_idx]
            if has_value:
                traj_e = np.moveaxis(traj, 1, 0)  # (E, H_max, d)
                goals_e = np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])
                vals = np.asarray(wm.value_score(traj_e, goals_e)).mean(axis=0)  # (H_max,)
                value_scores['max'][s, p] = np.maximum.accumulate(vals)[h_idx]
                value_scores['mean'][s, p] = (np.cumsum(vals) / np.arange(1, H_max + 1))[h_idx]
                value_scores['last'][s, p] = vals[h_idx]
        if (s + 1) % 20 == 0:
            print(f'imagined {s + 1}/{S} states ({counter.total} transitions)', flush=True)

    per_horizon_ranking = {}
    per_horizon_top1 = {}
    prog_ranking = {}
    prog_top1 = {}
    value_ranking = {agg: {} for agg in value_scores}
    value_top1 = {agg: {} for agg in value_scores}
    for hi, h in enumerate(horizons):
        if has_succ:
            per_horizon_ranking[h] = pairwise_ranking_accuracy(scores[:, :, hi], outcomes)
            per_horizon_top1[h] = top1_winner_rate(scores[:, :, hi], outcomes)
        if progress_fn is not None:
            prog_ranking[h] = pairwise_ranking_accuracy(prog_scores[:, :, hi], outcomes)
            prog_top1[h] = top1_winner_rate(prog_scores[:, :, hi], outcomes)
        if has_value:
            for agg in value_scores:
                value_ranking[agg][h] = pairwise_ranking_accuracy(value_scores[agg][:, :, hi], outcomes)
                value_top1[agg][h] = top1_winner_rate(value_scores[agg][:, :, hi], outcomes)

    H_train = int(wm.config['horizon'])
    gate_h = 2 * H_train
    gate_idx = min(range(len(horizons)), key=lambda i: abs(horizons[i] - H_train))
    calib = (
        calibration(scores[:, :, gate_idx].ravel(), outcomes.ravel()) if has_succ else None
    )

    # Open-loop error + disagreement on a sample of stored branches.
    rng = np.random.default_rng(0)
    n = min(FLAGS.openloop_branches, len(branches['state_id']))
    sel = rng.choice(len(branches['state_id']), n, replace=False)
    all_mse, all_dis = [], []
    for i in sel:
        L = int(branches['branch_len'][i])
        if L < 2:
            continue
        mse, dis = open_loop_error(
            wm,
            branches['branch_obs'][i, 0],
            branches['branch_actions'][i, :L],
            branches['branch_obs'][i, : L + 1],
        )
        all_mse.append(mse)
        all_dis.append(dis)
    flat_mse = np.concatenate(all_mse)
    flat_dis = np.concatenate(all_dis)

    # Gate condition (a) may be satisfied at ANY H <= 2*H_train; recompute top-1
    # at the best ranking horizon for reporting. Without a success head the
    # value-head verdict is the primary one.
    verdict_value = (
        gate_verdict(value_ranking['max'], value_top1['max'][horizons[gate_idx]], gate_h)
        if has_value
        else None
    )
    if has_succ:
        verdict = gate_verdict(per_horizon_ranking, per_horizon_top1[horizons[gate_idx]], gate_h)
    else:
        verdict = verdict_value
    result = dict(
        env_name=FLAGS.env_name,
        wm_dir=FLAGS.wm_dir,
        wm_epoch=FLAGS.wm_epoch,
        horizons=horizons,
        H_train=H_train,
        ranking={str(h): per_horizon_ranking[h] for h in horizons} if has_succ else None,
        top1={str(h): per_horizon_top1[h] for h in horizons} if has_succ else None,
        progress_ranking={str(h): prog_ranking[h] for h in horizons} if progress_fn is not None else None,
        progress_top1={str(h): prog_top1[h] for h in horizons} if progress_fn is not None else None,
        value_ranking={str(h): value_ranking['max'][h] for h in horizons} if has_value else None,
        value_top1={str(h): value_top1['max'][h] for h in horizons} if has_value else None,
        value_ranking_mean={str(h): value_ranking['mean'][h] for h in horizons} if has_value else None,
        value_top1_mean={str(h): value_top1['mean'][h] for h in horizons} if has_value else None,
        value_ranking_last={str(h): value_ranking['last'][h] for h in horizons} if has_value else None,
        value_top1_last={str(h): value_top1['last'][h] for h in horizons} if has_value else None,
        verdict_value=verdict_value,
        calibration_at_H_train=calib,
        disagreement_error_spearman=spearman(flat_dis, flat_mse),
        horizon_of_validity=horizon_of_validity(
            per_horizon_ranking if has_succ else (value_ranking['max'] if has_value else prog_ranking)
        ),
        imagined_transitions=counter.total,
        verdict=verdict,
    )

    out = FLAGS.out or os.path.join(oracle_dir, 'diagnostics.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    skip = (
        'ranking', 'top1', 'progress_ranking', 'progress_top1',
        'value_ranking', 'value_top1', 'value_ranking_mean', 'value_top1_mean',
        'value_ranking_last', 'value_top1_last',
    )
    print(json.dumps({k: v for k, v in result.items() if k not in skip}, indent=2, default=float))
    families = ([('success', per_horizon_ranking, per_horizon_top1)] if has_succ else []) + (
        [('progress', prog_ranking, prog_top1)] if progress_fn is not None else []
    ) + (
        [(f'value-{agg}', value_ranking[agg], value_top1[agg]) for agg in ('max', 'mean', 'last')]
        if has_value
        else []
    )
    fam = ' | '.join(name for name, _, _ in families)
    print(f'\nranking accuracy by horizon ({fam}):')
    for h in horizons:
        cells = [
            f'{rk[h]["accuracy"]:.3f} [{rk[h]["ci_lo"]:.3f}, {rk[h]["ci_hi"]:.3f}]'
            for _, rk, _ in families
        ]
        print(f'  H={h:3d}: ' + '  |  '.join(cells))
    print(f'\ntop-1 delta vs chance by horizon ({fam}):')
    for h in horizons:
        cells = [
            f'{t1[h]["delta"]:+.3f} [{t1[h]["ci_lo"]:+.3f}, {t1[h]["ci_hi"]:+.3f}]'
            for _, _, t1 in families
        ]
        print(f'  H={h:3d}: ' + '  |  '.join(cells))
    label = 'success-head' if has_succ else 'value-head'
    print(f'\nVERDICT ({label}): {"GO" if verdict["go"] else "NO-GO"} '
          f'(ranking_ok={verdict["ranking_ok"]}, top1_ok={verdict["top1_ok"]})')
    if has_value and has_succ:
        vv = result['verdict_value']
        print(f'VERDICT (value head): {"GO" if vv["go"] else "NO-GO"} '
              f'(ranking_ok={vv["ranking_ok"]}, top1_ok={vv["top1_ok"]})')


if __name__ == '__main__':
    app.run(main)
