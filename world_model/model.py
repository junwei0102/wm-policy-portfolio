"""Ensemble world model: full-observation delta dynamics + LAVL metric value head.

Mirrors the OGBench impls agent pattern (flax.struct.PyTreeNode + ModuleDict +
TrainState). Members are fully independent (disjoint params, mean of
per-member losses; gradients never mix). Normalization statistics are a
PYTREE field so they serialize/restore exactly with the checkpoint.

Predicted next observations are denormalized full observation vectors —
directly queryable by frozen policies ("policy-compatible").
"""

import json
import os
import sys

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

IMPLS = os.environ.get('OGBENCH_IMPLS', '/path/to/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent  # noqa: E402
from utils.networks import MLP  # noqa: E402


class LANValue(nn.Module):
    """LAVL's metric value head (oh-lab/LAVL, utils/networks.py: LANValue).

    d(s, g) = || phi_state(s) - phi_goal(psi(g)) ||_2 in a learned latent
    space; the value is -d. The goal first passes through a low-dimensional
    length-normalized representation psi (rep MLP -> rep_dim -> x/|x|*sqrt(dim)),
    mirroring LAVL's rep_def goal encoder. Inputs here are already-normalized
    observations. LAVL's optional smoothness regularizer has default weight 0
    upstream and is omitted.
    """

    hidden_dims: tuple
    latent_dim: int
    rep_dim: int
    layer_norm: bool = True

    def setup(self):
        self.rep = MLP((*self.hidden_dims, self.rep_dim), activate_final=False, layer_norm=self.layer_norm)
        self.phi_state = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.phi_goal = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals):
        g = self.rep(goals)
        g = g / jnp.linalg.norm(g, axis=-1, keepdims=True) * jnp.sqrt(g.shape[-1])
        phi_s = self.phi_state(observations)
        phi_g = self.phi_goal(g)
        return jnp.sqrt(jnp.sum((phi_s - phi_g) ** 2, axis=-1) + 1e-6)


def ensemblize_split(cls, num_members, **kwargs):
    """Like impls' ensemblize but maps inputs over the leading member axis.

    impls' ensemblize uses in_axes=None (inputs broadcast to all members),
    which cannot express members rolling out their OWN diverging state
    trajectories. Here inputs carry an explicit leading (E, ...) axis.
    """
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=0,
        out_axes=0,
        axis_size=num_members,
        **kwargs,
    )


class EnsembleWorldModel(flax.struct.PyTreeNode):
    rng: any
    network: any
    normalizer: any  # dict of jnp arrays — pytree field, serialized with the checkpoint.
    config: any = nonpytree_field()

    # --- normalization helpers ---

    def norm_obs(self, obs):
        return (obs - self.normalizer['obs_mean']) / self.normalizer['obs_std']

    def denorm_delta(self, delta):
        return self.normalizer['delta_mean'] + delta * self.normalizer['delta_std']

    # --- inference ---

    @jax.jit
    def predict_next(self, obs_e, actions_e, params=None):
        """(E, B, obs_dim) x (E, B, act_dim) -> (E, B, obs_dim), denormalized."""
        inputs = jnp.concatenate([self.norm_obs(obs_e), actions_e], axis=-1)
        delta = self.network.select('dynamics')(inputs, params=params)
        return obs_e + self.denorm_delta(delta)

    def lavl_distance(self, obs_e, goals_e, module='value', params=None):
        """Latent metric distance d(s, g) >= 0; the LAVL value is -d."""
        return self.network.select(module)(
            self.norm_obs(obs_e), self.norm_obs(goals_e), params=params
        )

    @jax.jit
    def value_score(self, obs_e, goals_e):
        """'Higher is better' goal-reachability score for ranking: -d(s, g) in
        the learned latent metric. Learned from data alone, no task-dimension
        knowledge."""
        return -self.lavl_distance(obs_e, goals_e)

    @jax.jit
    def imagine_openloop(self, obs0, action_seq):
        """obs0 (B, d), action_seq (B, H, da) -> (E, B, H+1, d) open-loop rollout."""
        E = self.config['num_members']
        x0 = jnp.broadcast_to(obs0, (E, *obs0.shape))

        def step(x, actions_bt):
            a = jnp.broadcast_to(actions_bt, (E, *actions_bt.shape))
            x_next = self.predict_next(x, a)
            return x_next, x_next

        _, traj = jax.lax.scan(step, x0, jnp.moveaxis(action_seq, 1, 0))
        # traj: (H, E, B, d) -> (E, B, H, d); prepend x0.
        traj = jnp.moveaxis(traj, 0, 2)
        return jnp.concatenate([x0[:, :, None, :], traj], axis=2)

    # --- losses ---

    def dynamics_loss(self, batch, grad_params):
        obs_seq = batch['obs_seq']  # (B, H+1, d)
        action_seq = batch['action_seq']  # (B, H, da)
        step_valid = batch['step_valid']  # (B, H)
        step_weight = batch['step_weight']  # (H,) curriculum weights, sum<=1

        E = self.config['num_members']
        obs_std = self.normalizer['obs_std']
        x = jnp.broadcast_to(obs_seq[:, 0], (E, *obs_seq[:, 0].shape))

        def step(x, inputs):
            actions, target, valid, weight = inputs
            a = jnp.broadcast_to(actions, (E, *actions.shape))
            inputs_e = jnp.concatenate([self.norm_obs(x), a], axis=-1)
            delta = self.network.select('dynamics')(inputs_e, params=grad_params)
            x_next = x + self.denorm_delta(delta)
            err = (x_next - target[None]) / obs_std
            mse = (err**2).mean(axis=-1)  # (E, B)
            step_loss = (mse * valid[None]).mean()
            return x_next, (step_loss, weight * step_loss)

        xs = (
            jnp.moveaxis(action_seq, 1, 0),
            jnp.moveaxis(obs_seq[:, 1:], 1, 0),
            jnp.moveaxis(step_valid, 1, 0),
            step_weight,
        )
        _, (per_step_mse, weighted) = jax.lax.scan(step, x, xs)
        loss = weighted.sum()
        return loss, {
            'loss': loss,
            'mse_step1': per_step_mse[0],
            'mse_stepH': per_step_mse[-1],
        }

    def lavl_value_loss(self, batch, grad_params):
        """LAVL expectile TD (lavl.py:value_loss): optimal stitched reachability.

        Sparse goal-conditioned reward 0/-1 with bootstrapping through the
        target metric head; expectile weighting approximates a max over
        dataset-supported actions.
        """
        E = self.config['num_members']
        gamma = self.config['lavl_discount']
        expectile = self.config['lavl_expectile']
        obs = jnp.broadcast_to(batch['observations'], (E, *batch['observations'].shape))
        next_obs = jnp.broadcast_to(batch['next_observations'], (E, *batch['next_observations'].shape))
        goals = jnp.broadcast_to(batch['value_goals'], (E, *batch['value_goals'].shape))
        # gc_negative=False rewards are success in {0,1} -> 0/-1 convention.
        r = batch['rewards'] - 1.0
        masks = batch['masks']

        next_v = -self.lavl_distance(next_obs, goals, module='target_value')  # (E, B)
        q = r + gamma * masks * next_v
        v_t = -self.lavl_distance(obs, goals, module='target_value')
        adv = q - v_t
        v = -self.lavl_distance(obs, goals, params=grad_params)
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        loss = (weight * (q - v) ** 2).mean()
        info = {
            'expectile_loss': loss,
            'v_mean': v.mean(),
            'v_min': v.min(),
            'v_max': v.max(),
        }
        if self.config['lavl_smoothness_weight'] > 0:
            # Local smoothness regularizer (lavl.py:value_loss): bound the
            # one-step value change toward RANDOM goals by a threshold set
            # from the running value mean (ema held by the trainer, passed in
            # via the batch so the checkpoint pytree stays unchanged).
            rand = jnp.broadcast_to(batch['random_goals'], obs.shape)
            threshold = 1.0 - (1.0 - gamma) * jax.lax.stop_gradient(batch['ema_v_mean'])
            v_rand = -self.lavl_distance(obs, rand, params=grad_params)
            v_next_rand = -self.lavl_distance(next_obs, rand, params=grad_params)
            sm = jnp.mean(
                0.1 * jax.nn.softplus(10 * ((v_next_rand - v_rand) ** 2 - threshold**2))
            )
            loss = loss + self.config['lavl_smoothness_weight'] * sm
            info['smoothness_loss'] = sm
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        dyn_loss, dyn_info = self.dynamics_loss(batch, grad_params)
        for k, v in dyn_info.items():
            info[f'dynamics/{k}'] = v
        val_loss, val_info = self.lavl_value_loss(batch, grad_params)
        for k, v in val_info.items():
            info[f'value/{k}'] = v
        loss = dyn_loss + self.config['value_loss_weight'] * val_loss
        info['total_loss'] = loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # Soft target update (lavl.py:target_update), tau = lavl_tau.
        tau = self.config['lavl_tau']
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_network.params['modules_value'],
            new_network.params['modules_target_value'],
        )
        new_network.params['modules_target_value'] = new_target
        return self.replace(network=new_network, rng=new_rng), info

    # --- construction / restore ---

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, norm_stats, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        # The architecture is fixed: delta dynamics + LAVL metric value head.
        # The success head (never used by any reported planner) and the MC
        # gamma^D value head were removed on 2026-08-27; their config keys are
        # still accepted so existing flags.json / command lines parse, but
        # only the retained architecture can be built or restored.
        assert not config.get('success_head', False), 'success head removed; checkpoints with one cannot be loaded'
        assert config.get('value_head', True), 'the LAVL value head is mandatory'
        assert config.get('value_head_type', 'lavl') == 'lavl', 'only the LAVL value head is supported'
        config['success_head'] = False
        config['value_head'] = True
        config['value_head_type'] = 'lavl'
        config.setdefault('value_loss_weight', 1.0)
        config.setdefault('lavl_latent_dim', 64)
        config.setdefault('lavl_rep_dim', 10)
        config.setdefault('lavl_discount', 0.999)
        config.setdefault('lavl_expectile', 0.9)
        config.setdefault('lavl_tau', 0.005)
        config.setdefault('lavl_smoothness_weight', 0.0)

        obs_dim = ex_observations.shape[-1]
        E = config['num_members']

        dynamics_def = ensemblize_split(MLP, E)(
            hidden_dims=(*config['hidden_dims'], obs_dim),
            activate_final=False,
            layer_norm=config['layer_norm'],
        )
        ex_obs_e = np.broadcast_to(ex_observations, (E, *ex_observations.shape))
        ex_act_e = np.broadcast_to(ex_actions, (E, *ex_actions.shape))
        ex_dyn = np.concatenate([ex_obs_e, ex_act_e], axis=-1)

        lan_kwargs = dict(
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['lavl_latent_dim'],
            rep_dim=config['lavl_rep_dim'],
            layer_norm=config['layer_norm'],
        )
        modules = dict(
            dynamics=dynamics_def,
            value=ensemblize_split(LANValue, E)(**lan_kwargs),
            target_value=ensemblize_split(LANValue, E)(**lan_kwargs),
        )
        ex_inputs = dict(dynamics=ex_dyn, value=(ex_obs_e, ex_obs_e), target_value=(ex_obs_e, ex_obs_e))
        network_def = ModuleDict(modules)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **ex_inputs)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        # Target starts as an exact copy of the online head (lavl.py:create).
        network.params['modules_target_value'] = network.params['modules_value']

        normalizer = {k: jnp.asarray(v) for k, v in norm_stats.items()}
        return cls(rng=rng, network=network, normalizer=normalizer, config=flax.core.FrozenDict(**config))

    @classmethod
    def load(cls, run_dir, epoch):
        """Rebuild from flags.json and restore params + normalizer exactly."""
        with open(os.path.join(run_dir, 'flags.json')) as f:
            flags = json.load(f)
        config = flags['wm']
        obs_dim, action_dim = flags['obs_dim'], flags['action_dim']
        ex_obs = np.zeros((1, obs_dim), dtype=np.float32)
        ex_act = np.zeros((1, action_dim), dtype=np.float32)
        zero_stats = {
            k: np.zeros(obs_dim, dtype=np.float32)
            for k in ['obs_mean', 'obs_std', 'delta_mean', 'delta_std']
        }
        model = cls.create(flags['seed'], ex_obs, ex_act, zero_stats, config)
        return restore_agent(model, run_dir, epoch)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            wm_name='ensemble_wm',
            lr=3e-4,
            batch_size=256,
            hidden_dims=(512, 512, 512),
            layer_norm=True,
            num_members=3,
            horizon=5,
            ramp_steps=100000,
            # Fixed architecture flags, kept so existing command lines
            # (--wm.success_head=False --wm.value_head_type=lavl) still parse.
            success_head=False,
            value_head=True,
            value_head_type='lavl',
            value_loss_weight=1.0,
            lavl_latent_dim=64,
            lavl_rep_dim=10,
            lavl_discount=0.999,
            lavl_expectile=0.9,
            lavl_tau=0.005,
            lavl_smoothness_weight=0.0,  # 10.0 for mazes/scene per LAVL's hyperparameters.sh
            discount=0.99,
            p_curgoal=0.2,
            p_trajgoal=0.5,
            p_randomgoal=0.3,
            geom_sample=True,
            frame_skip=1,
        )
    )
