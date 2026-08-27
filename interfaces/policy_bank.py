"""Load a frozen policy bank from impls run directories.

Scans `<policy_root>/*/flags.json`, filters by env/algo/seed, and restores each
checkpoint as a FrozenPolicy named '<algo>-sd<seed>'. Run dirs without the
requested params_<epoch>.pkl (crashed/partial runs) are skipped; when several
run dirs exist for the same (env, algo, seed), the lexicographically latest dir
(newest timestamp suffix) wins.
"""

import glob
import json
import os

from interfaces.frozen_policy import FrozenPolicy


def find_run_dirs(env_name, policy_root, epoch, algos=None, seeds=None):
    """Map '<algo>-sd<seed>' -> run_dir for completed runs of env_name."""
    run_dirs = {}
    for flags_path in sorted(glob.glob(os.path.join(policy_root, '*', 'flags.json'))):
        try:
            with open(flags_path) as f:
                flags = json.load(f)
        except json.JSONDecodeError:
            # Partially-written flags.json from a killed run (its dir would
            # fail the checkpoint gate below anyway) — skip, don't crash.
            continue
        if flags.get('env_name') != env_name:
            continue
        algo = flags['agent']['agent_name']
        seed = flags['seed']
        if algos is not None and algo not in algos:
            continue
        if seeds is not None and seed not in seeds:
            continue
        run_dir = os.path.dirname(flags_path)
        if not os.path.exists(os.path.join(run_dir, f'params_{epoch}.pkl')):
            continue
        run_dirs[f'{algo}-sd{seed}'] = run_dir
    return run_dirs


def load_bank(env_name, policy_root, epoch, algos=None, seeds=None):
    """Load all matching frozen policies. Returns {name: FrozenPolicy}."""
    run_dirs = find_run_dirs(env_name, policy_root, epoch, algos=algos, seeds=seeds)
    return {name: FrozenPolicy.load(run_dir, epoch) for name, run_dir in sorted(run_dirs.items())}


def select_best_fixed(success_by_policy):
    """Best fixed policy from {name: mean ID-validation success}; ties break by name."""
    return max(sorted(success_by_policy), key=lambda name: success_by_policy[name])
