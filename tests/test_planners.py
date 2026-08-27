"""Planner tests: counter accounting, budget parity, deterministic tie-breaking."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from planners.portfolio import OneStepChooser, RandomArbiter, RolloutRanker, _lexi_argmax
from tests.test_world_model import ACT_DIM, E, OBS_DIM, make_model
from world_model.rollout import TransitionCounter


class ConstPolicy:
    def __init__(self, value):
        self.value = value

    def act(self, obs, goals, temperature=0.0):
        obs = np.atleast_2d(obs)
        return np.full((obs.shape[0], ACT_DIM), self.value, dtype=np.float32).squeeze()


def make_setup(n_policies=4):
    model, _, _ = make_model()
    bank = {f'p{i}': ConstPolicy(0.1 * i) for i in range(n_policies)}
    return model, bank


def test_lexi_argmax():
    assert _lexi_argmax(np.array([1.0, 2.0, 2.0]), np.array([0.0, 0.0, 1.0])) == 2
    assert _lexi_argmax(np.array([1.0, 2.0, 2.0]), np.array([0.0, 1.0, 1.0])) == 1  # tie -> lowest idx
    assert _lexi_argmax(np.array([3.0, 2.0, 2.0]), np.array([0.0, 9.0, 9.0])) == 0


def test_chooser_budget():
    model, bank = make_setup()
    counter = TransitionCounter()
    chooser = OneStepChooser(model, bank, counter)
    ob = np.zeros(OBS_DIM, np.float32)
    goal = np.zeros(OBS_DIM, np.float32)
    a = chooser.act(ob, goal)
    assert a.shape == (ACT_DIM,)
    assert counter.per_decision == [E * len(bank)]
    for _ in range(4):
        chooser.act(ob, goal)
    assert all(c == E * len(bank) for c in counter.per_decision)


def test_ranker_budget_and_parity():
    H = 4
    model, bank = make_setup()
    c_ranker = TransitionCounter()
    ranker = RolloutRanker(model, bank, c_ranker, horizon=H)
    ob = np.zeros(OBS_DIM, np.float32)
    goal = np.zeros(OBS_DIM, np.float32)
    ranker.reset_episode()
    n_steps = 3 * H
    for _ in range(n_steps):
        a = ranker.act(ob, goal)
        assert a.shape == (ACT_DIM,)
    assert c_ranker.per_decision == [E * len(bank) * H] * 3

    c_chooser = TransitionCounter()
    chooser = OneStepChooser(model, bank, c_chooser)
    for _ in range(n_steps):
        chooser.act(ob, goal)
    # Budget parity: identical transitions per env step.
    assert c_ranker.total == c_chooser.total == E * len(bank) * n_steps


def test_ranker_replan_resets_per_episode():
    model, bank = make_setup()
    counter = TransitionCounter()
    ranker = RolloutRanker(model, bank, counter, horizon=3)
    ob = np.zeros(OBS_DIM, np.float32)
    goal = np.zeros(OBS_DIM, np.float32)
    ranker.act(ob, goal)
    assert len(counter.per_decision) == 1
    ranker.reset_episode()
    ranker.act(ob, goal)
    assert len(counter.per_decision) == 2  # replans immediately after reset


def test_score_agg():
    model, bank = make_setup(n_policies=2)
    ranker = RolloutRanker(model, bank, TransitionCounter(), horizon=3)
    # Crafted (H, P): policy 0 peaks early then collapses; policy 1 ends high.
    scores_hp = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.6]])
    ranker.score_agg = 'max'
    assert np.argmax(ranker._agg_horizon(scores_hp)) == 0  # best visited
    ranker.score_agg = 'last'
    assert np.argmax(ranker._agg_horizon(scores_hp)) == 1  # end state
    ranker.score_agg = 'mean'
    assert np.argmax(ranker._agg_horizon(scores_hp)) == 1  # path average
    np.testing.assert_allclose(
        RolloutRanker(model, bank, TransitionCounter(), horizon=3, score_agg='mean')._agg_horizon(scores_hp),
        scores_hp.mean(axis=0),
    )

    # End-to-end: every agg runs in value mode with identical budget and a
    # deterministic winner.
    ob = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    for agg in ('max', 'mean', 'last'):
        counter = TransitionCounter()
        r = RolloutRanker(model, bank, counter, horizon=3, score_mode='value', score_agg=agg)
        r.act(ob, goal)
        assert counter.per_decision == [E * len(bank) * 3]
        assert r._current in bank


def test_deterministic_choice():
    model, bank = make_setup()
    ob = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    picks = set()
    for _ in range(3):
        ranker = RolloutRanker(model, bank, TransitionCounter(), horizon=3)
        ranker.act(ob, goal)
        picks.add(ranker._current)
    assert len(picks) == 1  # same inputs -> same winner every time


def test_one_step_cell_is_single_wm_step():
    # One-Step WMPP = RolloutRanker(k=1, c=1, value): exactly one imagined
    # transition per member per policy per env step, one value eval per
    # imagined state, re-arbitration at every step.
    model, bank = make_setup()
    counter = TransitionCounter()
    r = RolloutRanker(model, bank, counter, horizon=1, replan_every=1, score_mode='value')
    ob = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    for _ in range(5):
        r.act(ob, goal)
    assert counter.per_decision == [E * len(bank)] * 5
    info = r.episode_info()
    assert info['n_plans'] == 5
    assert info['wm_transitions'] == 5 * E * len(bank)
    assert info['value_evals'] == 5 * E * len(bank)


def test_ranker_call_accounting_matches_counter():
    H = 4
    model, bank = make_setup()
    counter = TransitionCounter()
    r = RolloutRanker(model, bank, counter, horizon=H, replan_every=H, score_mode='value')
    ob = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    for _ in range(2 * H + 1):
        r.act(ob, goal)
    info = r.episode_info()
    assert info['wm_transitions'] == counter.total == 3 * E * len(bank) * H
    assert info['value_evals'] == 3 * E * len(bank) * H


def test_random_arbiter():
    _, bank = make_setup(n_policies=4)
    ob = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    c = 3
    ra = RandomArbiter(bank, commit=c, seed=0)
    chain = []
    for _ in range(4 * c):
        ra.act(ob, goal)
        chain.append(ra._current)
    # Commitment respected: constant within each window of c steps.
    for w in range(4):
        assert len(set(chain[w * c:(w + 1) * c])) == 1
    info = ra.episode_info()
    assert info['n_plans'] == 4
    assert info['wm_transitions'] == 0 and info['value_evals'] == 0
    # Reproducible per episode: same first (ob, goal) -> same schedule.
    rb = RandomArbiter(bank, commit=c, seed=0)
    chain_b = []
    for _ in range(4 * c):
        rb.act(ob, goal)
        chain_b.append(rb._current)
    assert chain == chain_b
    # Different seed -> different stream (with overwhelming probability over many draws).
    rc = RandomArbiter(bank, commit=1, seed=1)
    rc2 = RandomArbiter(bank, commit=1, seed=0)
    draws = [(rc.act(ob, goal), rc._current)[1] for _ in range(64)]
    draws2 = [(rc2.act(ob, goal), rc2._current)[1] for _ in range(64)]
    assert draws != draws2
    # Uniform over the bank incl. the active policy: every policy appears,
    # and "stay" draws happen (no forced switch).
    rd = RandomArbiter(bank, commit=1, seed=0)
    seq = []
    for _ in range(400):
        rd.act(ob, goal)
        seq.append(rd._current)
    assert set(seq) == set(bank)
    assert any(a == b for a, b in zip(seq, seq[1:]))
    counts = np.array([seq.count(n) for n in sorted(bank)])
    assert counts.min() > 50  # roughly uniform (expected 100 each)
    # reset_episode reseeds from the next episode's first (ob, goal).
    rd.reset_episode()
    assert rd._rng is None


if __name__ == '__main__':
    test_lexi_argmax()
    test_chooser_budget()
    test_ranker_budget_and_parity()
    test_ranker_replan_resets_per_episode()
    test_score_agg()
    test_deterministic_choice()
    test_one_step_cell_is_single_wm_step()
    test_ranker_call_accounting_matches_counter()
    test_random_arbiter()
    print('planner tests PASS')
