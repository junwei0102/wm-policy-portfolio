"""Onboard a new environment into the WMPP evaluation pipeline.

Given a completed policy bank (3 seeds x algorithms, OGBench impls checkpoints
with eval.csv) and a trained world model, this script

  1. reads every run's official OGBench test success from eval.csv
     (row at --policy_epoch = 5 goals x 50 episodes at the loaded checkpoint),
  2. writes/updates the env entry in manifests/wmpp_env_config.json
     (policy_root, wm_dir/epoch, best_fixed_per_seed, ogbench_test_success,
     ogbench_test_family_mean, best_policy_ogbench_test, bank_size),
  3. writes a launcher with the og50 protocol jobs per bank seed:
       * og50   : WMPP diagonal k=c in {1,5,10,25,50,100} + Random at every c
                  (planners only, --skip_fixed), and
       * og50fx : every fixed bank policy on the same episodes (static oracle),
     and submits them with --submit (job ids appended to SUBMISSIONS.log).

Usage:
  python scripts/onboard_env.py --env_name=pointmaze-medium-navigate-v0 \
      --policy_root=/scratch/jwquan/wmpp/policies/mpilot/OGBench/mpilot \
      --wm_dir=/scratch/jwquan/wmpp/wm/wmogbench/wm-value-lavl/sd000_s_<job>.<ts> \
      --wm_epoch=1000000 [--submit]
"""

import csv
import json
import os
import subprocess
import sys
import time

from absl import app, flags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from interfaces.policy_bank import find_run_dirs  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'OGBench dataset name.', required=True)
flags.DEFINE_string('policy_root', None, 'Bank root (dir holding run dirs with flags.json).', required=True)
flags.DEFINE_string('wm_dir', None, 'World-model run dir.', required=True)
flags.DEFINE_integer('wm_epoch', 1000000, 'World-model checkpoint step.')
flags.DEFINE_integer('policy_epoch', 1000000, 'Policy checkpoint step (eval.csv row used as the baseline).')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds.')
flags.DEFINE_string('env_config', os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'), 'Env registry.')
flags.DEFINE_string('launcher_dir', '/scratch/jwquan/wmpp/launchers', 'Where launch_*.sh and SUBMISSIONS.log live.')
flags.DEFINE_string('time_og50', '4:00:00', 'Walltime of the og50 (planner) jobs.')
flags.DEFINE_string('time_og50fx', '2:00:00', 'Walltime of the og50fx (fixed policies) jobs.')
flags.DEFINE_string('exclude', 'fc30554,fc30560,fc30572,fc30657', 'Bad CPU nodes.')
flags.DEFINE_bool('submit', False, 'Submit the generated sbatch lines.')
flags.DEFINE_bool('allow_incomplete', False, 'Proceed even if some (algo, seed) runs are missing.')

ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']

SBATCH = ('sbatch --account=def-gigor_cpu --cpus-per-task=8 --mem=24G --time={time} --exclude={exclude} '
          '-J {name} -o /scratch/jwquan/wmpp/logs/%x_%j.out --wrap="\nset -euo pipefail\n'
          'source /scratch/jwquan/wmpp/venv/bin/activate\n'
          'export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu\n'
          'cd /project/6067317/jwquan/wm-policy-portfolio\n{cmd}"')


def eval_rows(run_dir):
    with open(os.path.join(run_dir, 'eval.csv')) as f:
        return list(csv.DictReader(f))


def main(_):
    seeds = [int(s) for s in FLAGS.seeds.split(',')]
    runs = find_run_dirs(FLAGS.env_name, FLAGS.policy_root, FLAGS.policy_epoch, seeds=seeds)
    missing = [f'{a}-sd{s}' for s in seeds for a in ALGOS if f'{a}-sd{s}' not in runs]
    print(f'[onboard] {len(runs)} completed runs for {FLAGS.env_name}; missing: {missing or "none"}')
    if missing and not FLAGS.allow_incomplete:
        print('[onboard] refusing to proceed with an incomplete bank (use --allow_incomplete to override).')
        sys.exit(1)
    assert os.path.exists(os.path.join(FLAGS.wm_dir, f'params_{FLAGS.wm_epoch}.pkl')), 'WM checkpoint missing'

    test, last3 = {}, {}
    for name, rd in sorted(runs.items()):
        rows = eval_rows(rd)
        at_epoch = [r for r in rows if int(float(r['step'])) == FLAGS.policy_epoch]
        assert at_epoch, f'{name}: no eval.csv row at step {FLAGS.policy_epoch}'
        test[name] = float(at_epoch[-1]['evaluation/overall_success'])
        last3[name] = sum(float(r['evaluation/overall_success']) for r in rows[-3:]) / len(rows[-3:])
    best_fixed = {}
    for s in seeds:
        cands = {n: v for n, v in last3.items() if n.endswith(f'-sd{s}')}
        best_fixed[str(s)] = max(cands, key=cands.get)
    fam = {}
    for a in ALGOS:
        vals = [test[f'{a}-sd{s}'] for s in seeds if f'{a}-sd{s}' in test]
        if vals:
            fam[a] = sum(vals) / len(vals)
    best_family = max(fam, key=fam.get)
    entry = dict(
        policy_root=FLAGS.policy_root, policy_epoch=FLAGS.policy_epoch,
        wm_dir=FLAGS.wm_dir, wm_epoch=FLAGS.wm_epoch, selected_k=10,
        best_fixed_per_seed=best_fixed, ogbench_test_success=test,
        ogbench_test_family_mean=fam, best_policy_ogbench_test=best_family,
        bank_size={str(s): sum(1 for n in runs if n.endswith(f'-sd{s}')) for s in seeds},
    )
    cfg = json.load(open(FLAGS.env_config)) if os.path.exists(FLAGS.env_config) else {}
    cfg[FLAGS.env_name] = entry
    with open(FLAGS.env_config, 'w') as f:
        json.dump(cfg, f, indent=1)
    print('[onboard] official test success (eval.csv @ %d):' % FLAGS.policy_epoch)
    for a in ALGOS:
        if a in fam:
            per = ' '.join(f'sd{s}={test[f"{a}-sd{s}"]:.3f}' for s in seeds if f'{a}-sd{s}' in test)
            print(f'  {a:6s} mean {fam[a]:.3f}  ({per})')
    print(f'[onboard] best family: {best_family}; best_fixed per seed (last-3 mean): {best_fixed}')

    short = FLAGS.env_name.replace('-v0', '')
    common = (f'--env_name={FLAGS.env_name} --wm_dir={FLAGS.wm_dir} --wm_epoch={FLAGS.wm_epoch} '
              f'--policy_root={FLAGS.policy_root} --policy_epoch={FLAGS.policy_epoch} '
              f'--horizon=100 --commit=100 --score_mode=value --score_agg=max')
    lines = []
    for s in seeds:
        bf = best_fixed[str(s)]
        og50 = (f'python scripts/eval_planner.py {common} --seeds={s} --best_fixed={bf} --kc_sweep=5,10,25,50 '
                f'--variants=score1_commit1,score5_commit5,score10_commit10,score25_commit25,score50_commit50,score100_commit100 '
                f'--random_commit=1,5,10,25,50,100 --random_seed=0 --episodes_per_task=50 --skip_fixed --out_tag=og50')
        fx = (f'python scripts/eval_planner.py {common} --seeds={s} --best_fixed={bf} --variants=none '
              f'--episodes_per_task=50 --out_tag=og50fx')
        lines.append(SBATCH.format(time=FLAGS.time_og50, exclude=FLAGS.exclude, name=f'wmpp-og50-{short}-sd{s}', cmd=og50))
        lines.append(SBATCH.format(time=FLAGS.time_og50fx, exclude=FLAGS.exclude, name=f'wmpp-og50fx-{short}-sd{s}', cmd=fx))
    os.makedirs(FLAGS.launcher_dir, exist_ok=True)
    launcher = os.path.join(FLAGS.launcher_dir, f'launch_og50_{short}.sh')
    with open(launcher, 'w') as f:
        f.write('#!/bin/bash\n# generated by scripts/onboard_env.py\n' + '\n'.join(lines) + '\n')
    print(f'[onboard] wrote {launcher} ({len(lines)} jobs)')

    if FLAGS.submit:
        # Coordination guard: never double-submit a same-named job.
        queued = subprocess.run(['squeue', '-u', os.environ.get('USER', 'jwquan'), '-h', '-o', '%j'],
                                capture_output=True, text=True).stdout.split()
        ids = []
        for line in lines:
            name = line.split(' -J ')[1].split()[0]
            if name in queued:
                print(f'[onboard] SKIP {name}: already queued/running')
                continue
            out = subprocess.run(['bash', '-c', line], capture_output=True, text=True)
            print(out.stdout.strip() or out.stderr.strip())
            ids.append(out.stdout.strip().split()[-1] if out.returncode == 0 else f'FAILED:{name}')
        with open(os.path.join(FLAGS.launcher_dir, 'SUBMISSIONS.log'), 'a') as f:
            f.write(f'{time.strftime("%F %T")} {FLAGS.env_name}: og50 + og50fx via onboard_env.py ids={" ".join(ids)}\n')
    else:
        print('[onboard] dry run; re-run with --submit to launch.')


if __name__ == '__main__':
    app.run(main)
