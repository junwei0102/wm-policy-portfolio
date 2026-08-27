"""Verify batched FrozenPolicy.act matches row-wise single-obs act (temperature 0)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from interfaces.policy_bank import load_bank

POLICY_ROOT = '/scratch/jwquan/wmpp/policies/p1/OGBench/p1-pilot'
ENV = 'cube-double-play-v0'
EPOCH = 1000000

bank = load_bank(ENV, POLICY_ROOT, EPOCH, seeds=[0])
print(f'loaded {len(bank)} policies for {ENV}:', sorted(bank))
assert len(bank) == 6, f'expected 6 algos at seed 0, got {sorted(bank)}'

rng = np.random.default_rng(0)
obs = rng.normal(size=(5, 37)).astype(np.float32)
goal = rng.normal(size=(37,)).astype(np.float32)
goals = np.tile(goal, (5, 1))

for name, policy in bank.items():
    batched = policy.act(obs, goals)
    rowwise = np.stack([policy.act(obs[i], goal) for i in range(5)])
    assert batched.shape == rowwise.shape == (5, 5), (name, batched.shape)
    # Batched vs single-row XLA kernels use different reduction orders in
    # float32; ~1e-5 absolute drift is expected and harmless for [-1,1] actions.
    assert np.allclose(batched, rowwise, atol=1e-4), (
        name,
        np.max(np.abs(batched - rowwise)),
    )
    print(f'  {name}: batched == row-wise (max diff {np.max(np.abs(batched - rowwise)):.2e})')

print('policy bank batch test PASS')
