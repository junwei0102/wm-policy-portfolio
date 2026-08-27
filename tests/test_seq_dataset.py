"""Window sampler tests against a hand-built toy compact dataset."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

IMPLS = os.environ.get('OGBENCH_IMPLS', '/project/6067317/jwquan/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

from utils.datasets import Dataset  # noqa: E402

from world_model.seq_dataset import WMSequenceDataset, compute_norm_stats  # noqa: E402


def make_toy_dataset():
    # Two trajectories: 5 states (4 transitions) and 4 states (3 transitions).
    # Compact conversion (ogbench/utils.py:60-73): valids = 1 - raw_terminals;
    # terminals get an extra 1 on the last transition start.
    raw_terminals = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    valids = 1.0 - raw_terminals
    terminals = np.minimum(raw_terminals + np.concatenate([raw_terminals[1:], [1.0]]), 1.0)
    n = len(raw_terminals)
    # Observation encodes its global index so windows are directly checkable.
    observations = np.stack([np.arange(n), np.arange(n) * 10], axis=1).astype(np.float32)
    actions = np.arange(n, dtype=np.float32)[:, None]
    return Dataset.create(
        observations=observations, actions=actions, terminals=terminals, valids=valids
    )


def test_window_clamping():
    ds = WMSequenceDataset(make_toy_dataset(), horizon=3)
    # Trajectory 1 spans indices 0-4 (last valid start 3); traj 2 spans 5-8 (last valid start 7).
    batch = ds.sample(4, idxs=np.array([0, 3, 5, 7]))

    # idx 0: full window 0,1,2 valid; obs_seq = states 0..3.
    assert np.array_equal(batch['step_valid'][0], [1, 1, 1])
    assert np.array_equal(batch['obs_seq'][0, :, 0], [0, 1, 2, 3])
    assert np.array_equal(batch['action_seq'][0, :, 0], [0, 1, 2])

    # idx 3: only step 0 valid; obs clamp to final state 4; actions clamp to 3.
    assert np.array_equal(batch['step_valid'][1], [1, 0, 0])
    assert np.array_equal(batch['obs_seq'][1, :, 0], [3, 4, 4, 4])
    assert np.array_equal(batch['action_seq'][1, :, 0], [3, 3, 3])

    # idx 5 (traj 2 start): steps 5,6,7 all valid; obs 5..8. Never crosses into traj 1.
    assert np.array_equal(batch['step_valid'][2], [1, 1, 1])
    assert np.array_equal(batch['obs_seq'][2, :, 0], [5, 6, 7, 8])

    # idx 7 (last valid start of traj 2): clamp to final state 8.
    assert np.array_equal(batch['step_valid'][3], [1, 0, 0])
    assert np.array_equal(batch['obs_seq'][3, :, 0], [7, 8, 8, 8])


def test_random_windows_never_cross_boundaries():
    ds = WMSequenceDataset(make_toy_dataset(), horizon=3)
    batch = ds.sample(256)
    # Valid steps must never produce an obs jump other than +1 in the index dim.
    diffs = np.diff(batch['obs_seq'][..., 0], axis=1)
    valid = batch['step_valid'] > 0
    assert np.all(diffs[valid] == 1)
    # Start indices must all be valid transition starts (never a final state: 4 or 8).
    assert not np.any(np.isin(batch['obs_seq'][:, 0, 0], [4, 8]))


def test_reach_targets():
    gamma = 0.9
    ds = WMSequenceDataset(make_toy_dataset(), horizon=2, discount=gamma)
    batch = ds.sample(2048)
    starts = batch['obs_seq'][:, 0, 0]  # global index of s_t
    goals = batch['reach_goals'][:, 0]  # global index of the reach goal
    targets = batch['reach_targets']

    # Curgoal rows (target exactly 1): goal must be the start state itself.
    cur = targets == 1.0
    assert cur.any() and np.array_equal(goals[cur], starts[cur])

    # Trajectory-goal rows (0 < target < 1): goal strictly ahead of start in
    # the SAME trajectory, target exactly gamma^(goal - start), and offsets
    # clipped to the trajectory's final state (4 or 8).
    traj = (targets > 0) & (targets < 1)
    assert traj.any()
    assert np.all(goals[traj] > starts[traj])
    same_traj = (starts[traj] <= 4) == (goals[traj] <= 4)
    assert same_traj.all()
    assert np.allclose(targets[traj], gamma ** (goals[traj] - starts[traj]))
    assert np.all(goals[traj] <= np.where(starts[traj] <= 4, 4, 8))

    # Random-goal rows: target exactly 0; roughly the configured 0.3 mixture.
    rand = targets == 0
    assert rand.any()
    assert 0.1 < cur.mean() < 0.35
    assert 0.2 < rand.mean() < 0.4


def test_norm_stats():
    ds = make_toy_dataset()
    stats = compute_norm_stats(ds)
    obs = np.asarray(ds['observations'], dtype=np.float64)
    assert np.allclose(stats['obs_mean'], obs.mean(axis=0), atol=1e-5)
    # Deltas over valid starts only: all are +1 per index dim by construction.
    assert np.allclose(stats['delta_mean'], [1.0, 10.0], atol=1e-5)
    assert np.all(stats['delta_std'] >= 1e-6)


if __name__ == '__main__':
    test_window_clamping()
    test_random_windows_never_cross_boundaries()
    test_reach_targets()
    test_norm_stats()
    print('seq_dataset tests PASS')
