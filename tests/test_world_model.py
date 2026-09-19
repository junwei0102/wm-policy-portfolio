"""World-model unit tests: shapes, checkpoint round-trip, member independence."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import numpy as np

IMPLS = os.environ.get('OGBENCH_IMPLS', '/project/6067317/jwquan/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

from utils.flax_utils import save_agent  # noqa: E402

from world_model.model import EnsembleWorldModel, get_config  # noqa: E402
from world_model.rollout import TransitionCounter, imagine_policy_rollout, wm_step  # noqa: E402

OBS_DIM, ACT_DIM, E, H = 7, 3, 3, 4


def make_model(seed=0):
    config = get_config().to_dict()
    config.update(num_members=E, horizon=H, hidden_dims=(32, 32), ramp_steps=10)
    rng = np.random.default_rng(1)
    stats = dict(
        obs_mean=rng.normal(size=OBS_DIM).astype(np.float32),
        obs_std=(0.5 + rng.random(OBS_DIM)).astype(np.float32),
        delta_mean=rng.normal(size=OBS_DIM).astype(np.float32) * 0.01,
        delta_std=(0.1 + rng.random(OBS_DIM) * 0.1).astype(np.float32),
    )
    model = EnsembleWorldModel.create(
        seed,
        np.zeros((1, OBS_DIM), dtype=np.float32),
        np.zeros((1, ACT_DIM), dtype=np.float32),
        stats,
        config,
    )
    return model, config, stats


def make_batch(rng, B=16):
    return dict(
        obs_seq=rng.normal(size=(B, H + 1, OBS_DIM)).astype(np.float32),
        action_seq=rng.uniform(-1, 1, size=(B, H, ACT_DIM)).astype(np.float32),
        step_valid=np.ones((B, H), dtype=np.float32),
        step_weight=np.full(H, 1.0 / H, dtype=np.float32),
        observations=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        value_goals=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        reach_goals=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        reach_targets=rng.random(B).astype(np.float32),
        next_observations=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        rewards=(rng.random(B) < 0.2).astype(np.float32),
        masks=(rng.random(B) < 0.8).astype(np.float32),
        random_goals=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        ema_v_mean=np.float32(-100.0),
    )


def test_lavl_value_head():
    config = get_config().to_dict()
    config.update(
        num_members=E, horizon=H, hidden_dims=(32, 32), ramp_steps=10, value_head_type='lavl'
    )
    stats = dict(
        obs_mean=np.zeros(OBS_DIM, np.float32),
        obs_std=np.ones(OBS_DIM, np.float32),
        delta_mean=np.zeros(OBS_DIM, np.float32),
        delta_std=np.ones(OBS_DIM, np.float32),
    )
    model = EnsembleWorldModel.create(
        0,
        np.zeros((1, OBS_DIM), dtype=np.float32),
        np.zeros((1, ACT_DIM), dtype=np.float32),
        stats,
        config,
    )
    rng = np.random.default_rng(4)
    batch = make_batch(rng)
    obs_e = np.broadcast_to(batch['observations'], (E, *batch['observations'].shape))
    goals_e = np.broadcast_to(batch['reach_goals'], (E, *batch['reach_goals'].shape))

    # Distances nonnegative; score = -d; target starts as an exact copy.
    d = np.asarray(model.lavl_distance(obs_e, goals_e))
    assert d.shape == (E, len(batch['observations'])) and (d >= 0).all()
    assert np.allclose(np.asarray(model.value_score(obs_e, goals_e)), -d)
    d_t = np.asarray(model.lavl_distance(obs_e, goals_e, module='target_value'))
    assert np.array_equal(d, d_t)

    # Training: expectile TD loss appears, decreases, and the target LAGS the
    # online head (soft update) — it must move but stay distinct.
    _, info0 = model.total_loss(batch, grad_params=None)
    assert 'value/expectile_loss' in info0
    for _ in range(50):
        model, _ = model.update(batch)
    _, info1 = model.total_loss(batch, grad_params=None)
    assert np.isfinite(float(info1['value/expectile_loss']))
    assert float(info1['value/expectile_loss']) < float(info0['value/expectile_loss'])
    d1 = np.asarray(model.lavl_distance(obs_e, goals_e))
    d1_t = np.asarray(model.lavl_distance(obs_e, goals_e, module='target_value'))
    assert not np.array_equal(d1, d_t), 'online head did not train'
    assert not np.array_equal(d1_t, d_t), 'target head did not update'
    assert not np.array_equal(d1, d1_t), 'target must lag the online head'

    # Smoothness regularizer: enabling it adds the term and stays finite.
    config_sm = dict(config)
    config_sm['lavl_smoothness_weight'] = 10.0
    sm_model = EnsembleWorldModel.create(
        0,
        np.zeros((1, OBS_DIM), dtype=np.float32),
        np.zeros((1, ACT_DIM), dtype=np.float32),
        stats,
        config_sm,
    )
    _, sm_info = sm_model.total_loss(batch, grad_params=None)
    assert 'value/smoothness_loss' in sm_info and np.isfinite(float(sm_info['value/smoothness_loss']))
    sm_model, _ = sm_model.update(batch)

    # Checkpoint round-trip (incl. target module) is bitwise.
    tmp = tempfile.mkdtemp()
    try:
        save_agent(model, tmp, 7)
        with open(os.path.join(tmp, 'flags.json'), 'w') as f:
            import json

            json.dump(
                dict(wm=config, obs_dim=OBS_DIM, action_dim=ACT_DIM, seed=0), f
            )
        restored = EnsembleWorldModel.load(tmp, 7)
        d_r = np.asarray(restored.lavl_distance(obs_e, goals_e))
        d_rt = np.asarray(restored.lavl_distance(obs_e, goals_e, module='target_value'))
        assert np.array_equal(d1, d_r) and np.array_equal(d1_t, d_rt)
    finally:
        shutil.rmtree(tmp)


def test_removed_heads_rejected():
    """Configs asking for the removed success head / MC value head must fail loudly."""
    for bad in (dict(success_head=True), dict(value_head=False), dict(value_head_type='mc')):
        config = get_config().to_dict()
        config.update(num_members=E, hidden_dims=(32, 32), **bad)
        try:
            EnsembleWorldModel.create(
                0, np.zeros((1, OBS_DIM), np.float32), np.zeros((1, ACT_DIM), np.float32),
                dict(obs_mean=np.zeros(OBS_DIM, np.float32), obs_std=np.ones(OBS_DIM, np.float32),
                     delta_mean=np.zeros(OBS_DIM, np.float32), delta_std=np.ones(OBS_DIM, np.float32)),
                config,
            )
        except AssertionError:
            continue
        raise AssertionError(f'{bad} should have been rejected')


def test_update_and_shapes():
    model, _, _ = make_model()
    rng = np.random.default_rng(0)
    batch = make_batch(rng)
    model2, info = model.update(batch)
    assert np.isfinite(info['total_loss'])
    # Loss decreases over a few steps on a fixed batch (sanity, not strict).
    losses = [float(info['total_loss'])]
    for _ in range(50):
        model2, info = model2.update(batch)
        losses.append(float(info['total_loss']))
    assert losses[-1] < losses[0], (losses[0], losses[-1])

    obs_e = rng.normal(size=(E, 5, OBS_DIM)).astype(np.float32)
    act_e = rng.uniform(-1, 1, size=(E, 5, ACT_DIM)).astype(np.float32)
    assert model.predict_next(obs_e, act_e).shape == (E, 5, OBS_DIM)
    assert model.value_score(obs_e, obs_e).shape == (E, 5)
    assert model.imagine_openloop(
        rng.normal(size=(2, OBS_DIM)).astype(np.float32),
        rng.uniform(-1, 1, size=(2, H, ACT_DIM)).astype(np.float32),
    ).shape == (E, 2, H + 1, OBS_DIM)


def test_checkpoint_roundtrip_bitwise():
    model, config, _ = make_model(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        model, _ = model.update(make_batch(rng))

    tmp = tempfile.mkdtemp()
    try:
        save_agent(model, tmp, 3)
        import json

        with open(os.path.join(tmp, 'flags.json'), 'w') as f:
            json.dump(dict(wm=config, obs_dim=OBS_DIM, action_dim=ACT_DIM, seed=99), f)
        restored = EnsembleWorldModel.load(tmp, 3)
    finally:
        pass

    for k in model.normalizer:
        assert np.array_equal(np.asarray(model.normalizer[k]), np.asarray(restored.normalizer[k])), k
    obs_e = rng.normal(size=(E, 4, OBS_DIM)).astype(np.float32)
    act_e = rng.uniform(-1, 1, size=(E, 4, ACT_DIM)).astype(np.float32)
    assert np.array_equal(
        np.asarray(model.predict_next(obs_e, act_e)),
        np.asarray(restored.predict_next(obs_e, act_e)),
    )
    assert np.array_equal(
        np.asarray(model.value_score(obs_e, obs_e)),
        np.asarray(restored.value_score(obs_e, obs_e)),
    )
    shutil.rmtree(tmp)


def test_member_independence():
    model, _, _ = make_model()
    rng = np.random.default_rng(2)
    obs_e = rng.normal(size=(E, 4, OBS_DIM)).astype(np.float32)
    act_e = rng.uniform(-1, 1, size=(E, 4, ACT_DIM)).astype(np.float32)
    base = np.asarray(model.predict_next(obs_e, act_e))

    # Zero member 0's dynamics params; members 1..E-1 must be unchanged.
    params = jax.tree.map(lambda x: np.asarray(x).copy(), model.network.params)

    def zero_member0(leaf):
        leaf = leaf.copy()
        leaf[0] = 0
        return leaf

    params['modules_dynamics'] = jax.tree.map(zero_member0, params['modules_dynamics'])
    perturbed = model.replace(network=model.network.replace(params=params))
    out = np.asarray(perturbed.predict_next(obs_e, act_e))
    assert not np.allclose(out[0], base[0])
    assert np.array_equal(out[1:], base[1:])


def test_rollout_counter():
    model, _, _ = make_model()
    counter = TransitionCounter()
    rng = np.random.default_rng(3)
    obs_e = rng.normal(size=(E, 2, OBS_DIM)).astype(np.float32)
    act_e = rng.uniform(-1, 1, size=(E, 2, ACT_DIM)).astype(np.float32)
    wm_step(model, obs_e, act_e, counter)
    assert counter.total == E * 2

    class RandomPolicy:
        def act(self, obs, goals, temperature=0.0):
            return np.zeros((obs.shape[0], ACT_DIM), dtype=np.float32)

    counter = TransitionCounter()
    out = imagine_policy_rollout(
        model, RandomPolicy(), np.zeros(OBS_DIM, np.float32), np.zeros(OBS_DIM, np.float32), 5, counter
    )
    assert counter.total == E * 5
    assert counter.mark_decision() == E * 5
    assert out['obs_traj'].shape == (6, E, OBS_DIM)
    assert out['actions'].shape == (5, E, ACT_DIM)


if __name__ == '__main__':
    test_removed_heads_rejected()
    test_lavl_value_head()
    test_update_and_shapes()
    print('update/shapes PASS')
    test_checkpoint_roundtrip_bitwise()
    print('checkpoint round-trip PASS')
    test_member_independence()
    print('member independence PASS')
    test_rollout_counter()
    print('rollout/counter PASS')


def test_direct_value_head():
    """The optional direct (IQL) head trains jointly with the dynamics, keeps its
    own target critic, and is absent unless config.direct_head is set."""
    import numpy as np

    from world_model.model import EnsembleWorldModel, get_config

    cfg = get_config()
    cfg.direct_head = True
    cfg.horizon = 3
    cfg.num_members = E
    ex_o = np.zeros((1, OBS_DIM), np.float32)
    ex_a = np.zeros((1, ACT_DIM), np.float32)
    ns = {k: np.zeros(OBS_DIM, np.float32) for k in ('obs_mean', 'delta_mean')}
    ns.update({k: np.ones(OBS_DIM, np.float32) for k in ('obs_std', 'delta_std')})
    m = EnsembleWorldModel.create(0, ex_o, ex_a, ns, cfg)
    assert 'modules_direct_value' in m.network.params and 'modules_target_direct_critic' in m.network.params

    rng = np.random.default_rng(0)
    B = 8
    batch = dict(
        obs_seq=rng.normal(size=(B, cfg.horizon + 1, OBS_DIM)).astype(np.float32),
        action_seq=np.clip(rng.normal(size=(B, cfg.horizon, ACT_DIM)), -1, 1).astype(np.float32),
        step_valid=np.ones((B, cfg.horizon), np.float32),
        observations=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        next_observations=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        value_goals=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        random_goals=rng.normal(size=(B, OBS_DIM)).astype(np.float32),
        rewards=(rng.random(B) < 0.1).astype(np.float32),
        masks=np.ones(B, np.float32),
        step_weight=np.ones(cfg.horizon, np.float32),
        ema_v_mean=np.float32(-1.0),
    )
    v0 = np.asarray(m.direct_value_score(batch['observations'], batch['value_goals']))
    q0 = np.asarray(m.direct_q_min(batch['observations'], batch['value_goals'], batch['action_seq'][:, 0]))
    assert v0.shape == (B,) and q0.shape == (B,)
    for _ in range(20):
        m, info = m.update(batch)
    assert {'direct/value_loss', 'direct/critic_loss', 'dynamics/loss', 'value/expectile_loss'} <= set(info)
    assert not np.allclose(v0, np.asarray(m.direct_value_score(batch['observations'], batch['value_goals'])))

    plain = EnsembleWorldModel.create(0, ex_o, ex_a, ns, get_config())
    assert 'modules_direct_value' not in plain.network.params
    _, plain_info = plain.update(batch)
    assert not [k for k in plain_info if k.startswith('direct/')]
