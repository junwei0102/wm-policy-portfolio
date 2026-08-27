"""Train an ensemble world model on an OGBench dataset (strictly offline).

Mirrors impls/main.py wiring: absl flags + config_flags, CsvLogger + wandb
(offline by default via WMPP_WANDB_MODE), flags.json + params_<step>.pkl
checkpoints (restorable via EnsembleWorldModel.load).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
IMPLS = os.environ.get('OGBENCH_IMPLS', '/project/6067317/jwquan/ogbench/impls')
sys.path.insert(0, IMPLS)

import numpy as np
import ogbench
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.datasets import Dataset
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict

from world_model.model import EnsembleWorldModel, get_config  # noqa: E402
from world_model.seq_dataset import WMSequenceDataset, compute_norm_stats  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cube-double-play-v0', 'OGBench dataset name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_integer('train_steps', 500000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('val_interval', 5000, 'Validation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Checkpoint interval.')
flags.DEFINE_string('wandb_entity', 'distill-llms', 'W&B entity.')
flags.DEFINE_string('wandb_project', 'wmogbench', 'W&B project.')
flags.DEFINE_string(
    'wandb_dir',
    '/scratch/jwquan/wmpp/wandb',
    'Persistent W&B data dir (survives the job if a run must be re-synced).',
)

config_flags.DEFINE_config_file('wm', os.path.join(ROOT, 'world_model/model.py'), lock_config=False)


def step_weights(step, horizon, ramp_steps):
    """Curriculum: effective horizon ramps 1 -> H over ramp_steps."""
    h_eff = min(horizon, 1 + int(step / ramp_steps * (horizon - 1))) if ramp_steps > 0 else horizon
    w = np.zeros(horizon, dtype=np.float32)
    w[:h_eff] = 1.0 / h_eff
    return w


def main(_):
    config = FLAGS.wm
    exp_name = get_exp_name(FLAGS.seed)
    os.makedirs(FLAGS.wandb_dir, exist_ok=True)
    wandb.init(
        config=get_flag_dict(),
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        group=FLAGS.run_group,
        tags=[FLAGS.run_group],
        name=exp_name,
        dir=FLAGS.wandb_dir,
        # Compute nodes have outbound internet (verified: job 53629725 synced
        # live); netrc auth is picked up automatically.
        mode=os.environ.get('WMPP_WANDB_MODE', 'online'),
        save_code=True,
    )
    save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_raw, val_raw = ogbench.make_env_and_datasets(
        FLAGS.env_name, compact_dataset=True, dataset_only=True
    )
    train_dataset = Dataset.create(**train_raw)
    val_dataset = Dataset.create(**val_raw)

    norm_stats = compute_norm_stats(train_dataset)
    seq_kwargs = dict(
        horizon=config['horizon'],
        discount=config['discount'],
        p_curgoal=config['p_curgoal'],
        p_trajgoal=config['p_trajgoal'],
        p_randomgoal=config['p_randomgoal'],
        geom_sample=config['geom_sample'],
        frame_skip=config['frame_skip'],
    )
    train_seq = WMSequenceDataset(train_dataset, **seq_kwargs)
    val_seq = WMSequenceDataset(val_dataset, **seq_kwargs)

    obs_dim = train_dataset['observations'].shape[-1]
    action_dim = train_dataset['actions'].shape[-1]

    flag_dict = get_flag_dict()
    flag_dict['obs_dim'] = obs_dim
    flag_dict['action_dim'] = action_dim
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    np.random.seed(FLAGS.seed)
    model = EnsembleWorldModel.create(
        FLAGS.seed,
        np.zeros((1, obs_dim), dtype=np.float32),
        np.zeros((1, action_dim), dtype=np.float32),
        norm_stats,
        config.to_dict(),
    )

    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    full_weights = step_weights(FLAGS.train_steps, config['horizon'], 0)

    # LAVL smoothness threshold uses a running mean of V; the ema lives here
    # (host-side) and enters the loss through the batch (lavl.py keeps it on
    # the agent; we keep the checkpoint pytree unchanged instead).
    ema_v_mean = -1.0 / (1.0 - config['lavl_discount'])

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_seq.sample(config['batch_size'])
        batch['step_weight'] = step_weights(i, config['horizon'], config['ramp_steps'])
        batch['ema_v_mean'] = np.float32(ema_v_mean)
        model, update_info = model.update(batch)
        if 'value/v_mean' in update_info:
            tau = config['lavl_tau']
            ema_v_mean = (1 - tau) * ema_v_mean + tau * float(update_info['value/v_mean'])

        if i % FLAGS.log_interval == 0:
            metrics = {f'training/{k}': float(v) for k, v in update_info.items()}
            if i % FLAGS.val_interval == 0:
                val_batch = val_seq.sample(config['batch_size'])
                val_batch['step_weight'] = full_weights
                val_batch['ema_v_mean'] = np.float32(ema_v_mean)
                _, val_info = model.total_loss(val_batch, grad_params=None)
                metrics.update({f'validation/{k}': float(v) for k, v in val_info.items()})
                # Rank correlation of the value score vs gamma^D targets (diagnostic).
                tg, lg = [], []
                for _ in range(4):
                    vb = val_seq.sample(config['batch_size'])
                    E = config['num_members']
                    obs = np.broadcast_to(vb['observations'], (E, *vb['observations'].shape))
                    goals = np.broadcast_to(vb['reach_goals'], (E, *vb['reach_goals'].shape))
                    lg.append(np.asarray(model.value_score(obs, goals)).mean(axis=0))
                    tg.append(vb['reach_targets'])
                tg, lg = np.concatenate(tg), np.concatenate(lg)
                r_t = np.argsort(np.argsort(tg)).astype(np.float64)
                r_l = np.argsort(np.argsort(lg)).astype(np.float64)
                metrics['validation/value_rank_corr'] = float(np.corrcoef(r_t, r_l)[0, 1])
            wandb.log(metrics, step=i)
            train_logger.log(metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(model, save_dir, i)

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
