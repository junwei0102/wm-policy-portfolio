"""Review-driven ablation knobs: ensemble reduction, stall-restart control, oracle-dynamics ranker plumbing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from planners.portfolio import CriticSelectArbiter, RolloutRanker, StallRestartArbiter, _ens_reduce
from tests.test_world_model import ACT_DIM, E, OBS_DIM, make_model
from world_model.rollout import TransitionCounter


class LinearPolicy:
    """State- and goal-dependent deterministic policy accepting any leading batch dims."""

    def __init__(self, c):
        self.c = c

    def act(self, obs, goals, temperature=0.0):
        obs = np.asarray(obs, np.float32)
        goals = np.asarray(goals, np.float32)
        a = 0.1 * obs[..., :ACT_DIM] + 0.05 * goals[..., :ACT_DIM] + self.c
        return np.clip(a, -1, 1)


def test_ens_reduce():
    v = np.array([[1.0, 4.0], [3.0, 2.0], [2.0, 0.0]])  # (E=3, P=2)
    np.testing.assert_allclose(_ens_reduce(v, 'mean'), [2.0, 2.0])
    np.testing.assert_allclose(_ens_reduce(v, 'min'), [1.0, 0.0])
    np.testing.assert_allclose(_ens_reduce(v, 'lcb'), v.mean(0) - v.std(0))
    print('test_ens_reduce PASS')


def test_ranker_ens_agg_budget_and_effect():
    model, _, _ = make_model()
    bank = {f'p{i}': LinearPolicy(0.1 * i) for i in range(3)}
    ob = np.ones(OBS_DIM, np.float32)
    goal = -np.ones(OBS_DIM, np.float32)
    picks = {}
    for ens in ('mean', 'min', 'lcb'):
        c = TransitionCounter()
        r = RolloutRanker(model, bank, c, horizon=3, replan_every=3, score_mode='value', ens_agg=ens)
        r.reset_episode()
        a = r.act(ob, goal)
        assert a.shape == (ACT_DIM,)
        assert c.per_decision == [E * len(bank) * 3]  # same budget for every reduction
        picks[ens] = r._current
    print('test_ranker_ens_agg_budget_and_effect PASS', picks)


def test_stall_restart():
    bank = {f'p{i}': LinearPolicy(0.1 * i) for i in range(3)}
    arb = StallRestartArbiter(bank, np.zeros(OBS_DIM), np.ones(OBS_DIM), window=4, eps=0.05, seed=0)
    arb.reset_episode()
    ob = np.zeros(OBS_DIM, np.float32)
    goal = np.ones(OBS_DIM, np.float32)
    first = None
    for t in range(12):  # static state -> a restart every `window` steps after the warm-up
        arb.act(ob, goal)
        if first is None:
            first = arb._current
    info = arb.episode_info()
    assert info['n_restarts'] >= 2 and info['wm_transitions'] == 0 and info['value_evals'] == 0, info
    assert info['n_switches'] == info['n_restarts']
    # moving state -> no restart
    arb.reset_episode()
    for t in range(12):
        arb.act(ob + 0.5 * t, goal)
    assert arb.episode_info()['n_restarts'] == 0
    print('test_stall_restart PASS')


def test_critic_select():
    bank = {f'p{i}': LinearPolicy(0.1 * i) for i in range(3)}  # p2 proposes the largest first action component
    calls = []

    def q_fn(obs, goals, acts):
        assert obs.shape == (3, OBS_DIM) and goals.shape == (3, OBS_DIM) and acts.shape == (3, ACT_DIM), (obs.shape, acts.shape)
        calls.append(acts.copy())
        return acts[:, 0]

    arb = CriticSelectArbiter(bank, q_fn, commit=4, seed=0)
    arb.reset_episode()
    ob = np.zeros(OBS_DIM, np.float32)
    goal = np.zeros(OBS_DIM, np.float32)
    for t in range(12):
        a = arb.act(ob, goal)
        assert a.shape == (ACT_DIM,)
        assert arb._current == 'p2', (t, arb._current)
    info = arb.episode_info()
    assert info['n_plans'] == 3 and len(calls) == 3, info  # a selection at t = 0, 4, 8 only
    assert info['wm_transitions'] == 0 and info['value_evals'] == 3 * len(bank), info
    assert info['n_switches'] == 0 and info['policy_chain'] == 'p2:12', info
    # exact ties -> the lowest index (same rule as RolloutRanker)
    arb = CriticSelectArbiter(bank, lambda o, g, a: np.zeros(len(a)), commit=1)
    arb.reset_episode()
    arb.act(ob, goal)
    assert arb._current == 'p0', arb._current
    # a per-step selector re-selects every step
    arb = CriticSelectArbiter(bank, q_fn, commit=1)
    arb.reset_episode()
    for t in range(5):
        arb.act(ob, goal)
    assert arb.episode_info()['n_plans'] == 5
    print('test_critic_select PASS')


if __name__ == '__main__':
    test_ens_reduce()
    test_ranker_ens_agg_budget_and_effect()
    test_stall_restart()
    test_critic_select()
    print('ALL PASS')
