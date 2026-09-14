"""Collect dynamic-oracle branches for one env (chunkable by task).

Usage (login node or CPU array):
  JAX_PLATFORMS=cpu MUJOCO_GL=disable python collect_oracle.py \
      --env_name=cube-double-play-v0 --base_policy=gciql-sd0 \
      --tasks=1,2,3,4,5 --eps_per_task=5 --states_per_ep=8 \
      --out_dir=/path/to/oracle

Writes <out_dir>/<env_name>/chunk_task<T>.npz per task; --merge merges chunks
into states.npz/branches.npz/branches.csv/meta.json and prints headroom.
"""

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium
import numpy as np
import ogbench  # noqa: F401
from absl import app, flags

from evaluation.oracle import collect_and_branch_episode, headroom, pack_results
from interfaces.policy_bank import find_run_dirs, load_bank
from planners.portfolio import RandomArbiter, RolloutRanker
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'cube-double-play-v0', 'OGBench dataset name.')
flags.DEFINE_string('policy_root', '/path/to/policies/p1-pilot', 'Bank root.')
flags.DEFINE_integer('epoch', 1000000, 'Checkpoint epoch.')
flags.DEFINE_string('base_policy', None, 'Bank policy generating base episodes (also the merge-time headroom reference).')
flags.DEFINE_string('base', None, 'Trajectory generator: wmpp:<k> (RolloutRanker at k=c), random:<c> (Random-Switch), or a bank '
                    'policy name; default = --base_policy. Arbiter bases need --wm_dir and use the --seeds bank.')
flags.DEFINE_string('wm_dir', None, 'World-model run dir (wmpp base only).')
flags.DEFINE_integer('wm_epoch', 1000000, 'World-model checkpoint step.')
flags.DEFINE_string('seeds', '', 'Comma list of bank training seeds for the generator bank AND the branch pool (default all).')
flags.DEFINE_integer('random_seed', 0, 'Draw-stream seed of Random-Switch (matches eval_planner --random_seed).')
flags.DEFINE_string('score_agg', 'max', 'Horizon aggregation of the wmpp base.')
flags.DEFINE_string('check_dir', '/path/to/planner_eval', 'Planner-eval root used to verify that the generator '
                    'reproduces the official-protocol episodes of the same variant (determinism check).')
flags.DEFINE_bool('assert_match', True, 'Fail if the determinism check finds a mismatch.')
flags.DEFINE_string('tasks', '1,2,3,4,5', 'Comma-separated task ids for this chunk.')
flags.DEFINE_integer('eps_per_task', 5, 'Base episodes per task.')
flags.DEFINE_integer('states_per_ep', 8, 'Decision states kept per episode.')
flags.DEFINE_integer('snapshot_every', None, 'Snapshot period (default: 50 loco / 25 manip).')
flags.DEFINE_integer('save_horizon', 100, 'Branch obs/action steps stored for diagnostics.')
flags.DEFINE_string('out_dir', '/path/to/oracle', 'Output root.')
flags.DEFINE_bool('merge', False, 'Merge chunk files instead of collecting.')


def env_id_of(dataset_name):
    tokens = dataset_name.split('-')
    return '-'.join(tokens[:-2] + tokens[-1:])


def is_manip(env_name):
    return any(env_name.startswith(p) for p in ('cube', 'scene', 'puzzle'))


def merge(out_dir):
    chunks = sorted(glob.glob(os.path.join(out_dir, 'chunk_task*.npz')))
    assert chunks, f'no chunks in {out_dir}'
    states_parts, branch_parts = [], []
    for c in chunks:
        data = np.load(c, allow_pickle=False)
        states_parts.append({k[7:]: data[k] for k in data if k.startswith('states_')})
        branch_parts.append({k[9:]: data[k] for k in data if k.startswith('branches_')})
    policies = branch_parts[0]['policies']
    for part in branch_parts:
        assert np.array_equal(part['policies'], policies)
    offset = 0
    for spart, bpart in zip(states_parts, branch_parts):
        bpart['state_id'] = bpart['state_id'] + offset
        spart['state_id'] = spart['state_id'] + offset
        offset += len(spart['state_id'])
    states = {k: np.concatenate([p[k] for p in states_parts]) for k in states_parts[0]}
    branches = {
        k: (policies if k == 'policies' else np.concatenate([p[k] for p in branch_parts]))
        for k in branch_parts[0]
    }
    np.savez_compressed(os.path.join(out_dir, 'states.npz'), **states)
    np.savez_compressed(os.path.join(out_dir, 'branches.npz'), **branches)
    with open(os.path.join(out_dir, 'branches.csv'), 'w') as f:
        f.write('state_id,policy,success,ever_success,steps\n')
        names = [str(p) for p in policies]
        for i in range(len(branches['state_id'])):
            f.write(
                f"{branches['state_id'][i]},{names[branches['policy_idx'][i]]},"
                f"{branches['success'][i]},{branches['ever_success'][i]},{branches['steps'][i]}\n"
            )
    print(f"merged {len(chunks)} chunks: {len(states['state_id'])} states, "
          f"{len(branches['state_id'])} branches")
    return states, branches


def check_against_planner_eval(check_dir, env_name, seeds, variant, base_outcomes):
    """Compare collected base-episode successes with the official-protocol rows of `variant`."""
    import csv
    seed = (seeds or [0])[0]
    ref = {}
    for path in sorted(glob.glob(os.path.join(check_dir, env_name, f'bank_sd{seed}_og50*', 'episodes.csv'))):
        for r in csv.DictReader(open(path)):
            if r['policy'] == variant:
                ref.setdefault((int(r['task_id']), int(r['episode_idx'])), float(r['success']))
    compared = [(k, v, ref[k]) for k, v in base_outcomes.items() if k in ref]
    mism = [(k, v, rv) for k, v, rv in compared if abs(v - rv) > 1e-9]
    return dict(n_compared=len(compared), n_mismatch=len(mism), mismatches=[list(map(float, (k[0], k[1], v, rv))) for k, v, rv in mism][:20])


def main(_):
    out_dir = os.path.join(FLAGS.out_dir, FLAGS.env_name)
    os.makedirs(out_dir, exist_ok=True)

    if FLAGS.merge:
        _, branches = merge(out_dir)
        names = [str(p) for p in branches['policies']]
        if FLAGS.base_policy not in names:
            print(f'merged; no headroom (base {FLAGS.base_policy} not in pool {names})')
            return
        print(f'\nheadroom vs base policy {FLAGS.base_policy} (full pool):')
        print(json.dumps(headroom(branches, FLAGS.base_policy), indent=2))
        for seed in [0, 1, 2]:
            pool = [n for n in names if n.endswith(f'-sd{seed}')]
            if len(pool) < 2:
                continue
            base = FLAGS.base_policy if FLAGS.base_policy in pool else pool[0]
            print(f'\nseed-bank sd{seed} (base {base}):')
            print(json.dumps(headroom(branches, base, policies=pool), indent=2))
        return

    seeds = [int(x) for x in FLAGS.seeds.split(',')] if FLAGS.seeds else None
    bank = load_bank(FLAGS.env_name, FLAGS.policy_root, FLAGS.epoch, seeds=seeds)
    base_spec = FLAGS.base or FLAGS.base_policy
    assert base_spec, 'need --base or --base_policy'
    if base_spec.startswith('wmpp:'):
        k = int(base_spec.split(':')[1])
        wm = EnsembleWorldModel.load(FLAGS.wm_dir, FLAGS.wm_epoch)
        base_obj = RolloutRanker(wm, bank, TransitionCounter(), horizon=k, replan_every=k,
                                 score_mode='value', score_agg=FLAGS.score_agg)
        check_variant = f'score{k}_commit{k}'
    elif base_spec.startswith('random:'):
        c = int(base_spec.split(':')[1])
        base_obj = RandomArbiter(bank, c, seed=FLAGS.random_seed)
        check_variant = f'random_commit{c}'
    else:
        assert base_spec in bank, (base_spec, sorted(bank))
        base_obj, check_variant = base_spec, base_spec
    print(f'generator: {base_spec} (bank {sorted(bank)})', flush=True)
    env = gymnasium.make(env_id_of(FLAGS.env_name))
    episode_len = env.spec.max_episode_steps
    snapshot_every = FLAGS.snapshot_every or (25 if is_manip(FLAGS.env_name) else 50)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    base_outcomes = {}
    for task_id in [int(t) for t in FLAGS.tasks.split(',')]:
        state_rows, branch_rows = [], []
        for ep in range(FLAGS.eps_per_task):
            s_rows, b_rows, base = collect_and_branch_episode(
                env,
                FLAGS.env_name,
                task_id,
                ep,
                base_obj,
                bank,
                episode_len,
                snapshot_every,
                FLAGS.states_per_ep,
                FLAGS.save_horizon,
            )
            state_rows += s_rows
            branch_rows += b_rows
            base_outcomes[(task_id, ep)] = base['success']
            print(
                f'task {task_id} ep {ep}: base success {base["success"]:.0f} '
                f'({base["steps"]} steps), {len(s_rows)} states, {len(b_rows)} branches',
                flush=True,
            )
        states, branches = pack_results(state_rows, branch_rows, FLAGS.save_horizon, obs_dim, act_dim)
        out = {f'states_{k}': v for k, v in states.items()}
        out.update({f'branches_{k}': v for k, v in branches.items()})
        np.savez_compressed(os.path.join(out_dir, f'chunk_task{task_id}.npz'), **out)
        print(f'wrote chunk_task{task_id}.npz', flush=True)

    # Determinism check: the generator must reproduce the official-protocol episodes of
    # the same variant (same reset seeds, deterministic policies / arbiter draw streams).
    check = check_against_planner_eval(FLAGS.check_dir, FLAGS.env_name, seeds, check_variant, base_outcomes)
    print(f'determinism check vs {check_variant}: {check}', flush=True)
    if FLAGS.assert_match and check['n_compared']:
        assert check['n_mismatch'] == 0, check

    meta = dict(
        env_name=FLAGS.env_name,
        base_policy=FLAGS.base_policy,
        base=base_spec,
        check_variant=check_variant,
        determinism_check=check,
        seeds=seeds,
        epoch=FLAGS.epoch,
        eps_per_task=FLAGS.eps_per_task,
        states_per_ep=FLAGS.states_per_ep,
        snapshot_every=snapshot_every,
        save_horizon=FLAGS.save_horizon,
        episode_len=int(episode_len),
        policies=find_run_dirs(FLAGS.env_name, FLAGS.policy_root, FLAGS.epoch, seeds=seeds),
        git_sha=subprocess.run(
            ['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True
        ).stdout.strip(),
    )
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    app.run(main)
