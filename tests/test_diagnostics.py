"""Diagnostics unit tests on synthetic fixtures with known ground truth."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from evaluation.diagnostics import (
    calibration,
    gate_verdict,
    pairwise_ranking_accuracy,
    spearman,
    top1_winner_rate,
)


def test_perfect_ranker():
    rng = np.random.default_rng(0)
    outcomes = (rng.random((200, 6)) < 0.4).astype(float)
    scores = outcomes + rng.random((200, 6)) * 0.5  # scores respect outcome order
    r = pairwise_ranking_accuracy(scores, outcomes, n_boot=500)
    assert r['accuracy'] == 1.0
    assert r['ci_lo'] == 1.0
    t = top1_winner_rate(scores, outcomes, n_boot=500)
    assert t['rate'] == 1.0 and t['ci_lo'] > 0


def test_random_ranker_near_chance():
    rng = np.random.default_rng(1)
    outcomes = (rng.random((400, 6)) < 0.5).astype(float)
    scores = rng.random((400, 6))
    r = pairwise_ranking_accuracy(scores, outcomes, n_boot=500)
    assert 0.45 < r['accuracy'] < 0.55
    assert r['ci_lo'] < 0.5 < r['ci_hi']
    t = top1_winner_rate(scores, outcomes, n_boot=500)
    assert t['ci_lo'] < 0 < t['ci_hi']


def test_tie_scores_half_credit():
    outcomes = np.array([[1.0, 0.0]])
    scores = np.array([[0.5, 0.5]])
    r = pairwise_ranking_accuracy(scores, outcomes, n_boot=10)
    assert r['accuracy'] == 0.5


def test_calibration_known():
    rng = np.random.default_rng(2)
    probs = rng.random(200000)
    labels = (rng.random(200000) < probs).astype(float)  # perfectly calibrated
    c = calibration(probs, labels)
    assert c['ece'] < 0.01, c['ece']
    probs_bad = np.full(1000, 0.9)
    labels_bad = np.zeros(1000)
    c2 = calibration(probs_bad, labels_bad)
    assert c2['ece'] > 0.85


def test_spearman_known():
    x = np.arange(100.0)
    assert abs(spearman(x, x**3) - 1.0) < 1e-9  # monotone
    assert abs(spearman(x, -x) + 1.0) < 1e-9


def test_gate_verdict():
    ranking = {5: dict(ci_lo=0.62), 20: dict(ci_lo=0.45)}
    top1 = dict(ci_lo=0.05)
    v = gate_verdict(ranking, top1, max_gate_horizon=10)
    assert v['go']
    v2 = gate_verdict(ranking, dict(ci_lo=-0.02), max_gate_horizon=10)
    assert not v2['go']
    v3 = gate_verdict({20: dict(ci_lo=0.62)}, top1, max_gate_horizon=10)
    assert not v3['go']


if __name__ == '__main__':
    test_perfect_ranker()
    test_random_ranker_near_chance()
    test_tie_scores_half_credit()
    test_calibration_known()
    test_spearman_known()
    test_gate_verdict()
    print('diagnostics tests PASS')
