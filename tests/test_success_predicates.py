"""Validate observation-pair success predicates against REAL branch outcomes.

For every oracle branch that terminated within the stored window, the
predicate applied to (final branch obs, episode goal) must reproduce the env's
recorded info['success'].
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from world_model.success_predicates import get_predicate

ORACLE = '/scratch/jwquan/wmpp/oracle'


def check(env_name, min_agreement=0.99):
    d = os.path.join(ORACLE, env_name)
    states = dict(np.load(os.path.join(d, 'states.npz')))
    branches = dict(np.load(os.path.join(d, 'branches.npz')))
    pred = get_predicate(env_name)

    # Branches whose final state is inside the stored window.
    m = branches['steps'] == branches['branch_len']
    final_obs = branches['branch_obs'][m, branches['branch_len'][m]]
    goals = states['goal'][branches['state_id'][m]]
    real = branches['success'][m]
    predicted = pred(final_obs, goals)
    agreement = float((predicted == real).mean())
    n_pos = int(real.sum())
    print(
        f'{env_name}: {m.sum()} checkable branches ({n_pos} successes), '
        f'agreement {agreement:.4f}'
    )
    assert agreement >= min_agreement, agreement
    assert n_pos > 0, 'no positive branches to validate against'


if __name__ == '__main__':
    check('antmaze-large-navigate-v0')
    check('cube-double-play-v0')
    print('success predicate tests PASS')
