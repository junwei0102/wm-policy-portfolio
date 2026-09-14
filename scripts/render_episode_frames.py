"""Replay one og50 WMPP episode deterministically and render MuJoCo frames at every
arbitration step (and the final step). Needs a working GL backend (MUJOCO_GL=egl on a
GPU node); with --no_render it only replays and checks the policy chain against the log.

Frames -> <out_dir>/<env>/task<T>_ep<E>/frames.npz  (t, frame (N,H,W,3), policy_idx, obs,
goal, chain) plus t<t>.png per frame and goal.png (the task's goal state rendered by OGBench).

  MUJOCO_GL=egl JAX_PLATFORMS=cpu OGBENCH_IMPLS=... python scripts/render_episode_frames.py \
      --env cube-double-play-v0 --task 5 --episode 31 --k 10
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gymnasium
import numpy as np
import ogbench  # noqa: F401  (registers envs)

from evaluation.paired_eval import _ensure_seedable_action_space, episode_seed
from interfaces.policy_bank import load_bank
from planners.portfolio import RolloutRanker
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='cube-double-play-v0')
ap.add_argument('--task', type=int, default=5)
ap.add_argument('--episode', type=int, default=31)
ap.add_argument('--seed', type=int, default=0, help='bank seed')
ap.add_argument('--k', type=int, default=10)
ap.add_argument('--size', type=int, default=480)
ap.add_argument('--camera', default=None, help="OGBench camera name (default: env's 'front')")
ap.add_argument('--eval_root', default='/scratch/jwquan/wmpp/planner_eval')
ap.add_argument('--tag', default='og50dec')
ap.add_argument('--out_dir', default='/scratch/jwquan/wmpp/planner_eval/episode_frames')
ap.add_argument('--no_render', action='store_true')
args = ap.parse_args()

cfg = json.load(open(os.path.join(ROOT, 'manifests', 'wmpp_env_config.json')))[args.env]
z = np.load(os.path.join(args.eval_root, args.env, f'bank_sd{args.seed}_{args.tag}', f'decisions_score{args.k}_commit{args.k}.npz'))
m = (z['task_id'] == args.task) & (z['episode_idx'] == args.episode)
assert m.any(), 'episode not in the decision log'
logged_t, logged_pick, logged_steps = z['t'][m], z['policy_idx'][m], int(z['ep_steps'][m][0])
pol = [str(p) for p in z['policies']]

wm = EnsembleWorldModel.load(cfg['wm_dir'], cfg['wm_epoch'])
bank = load_bank(args.env, cfg['policy_root'], cfg['policy_epoch'], seeds=[args.seed])
assert sorted(bank) == pol
planner = RolloutRanker(wm, bank, TransitionCounter(), horizon=args.k, replan_every=args.k,
                        score_mode='value', score_agg='max', log_decisions=True)

tokens = args.env.split('-'); env_id = '-'.join(tokens[:-2] + tokens[-1:])
env = gymnasium.make(env_id, width=args.size, height=args.size)
seed = episode_seed(args.env, f'task{args.task}', args.episode)
assert seed == int(z['reset_seed'][m][0])
# identical seeding to evaluation.paired_eval.run_episode
_ensure_seedable_action_space()
np.random.seed(seed)
env.unwrapped.np_random = np.random.default_rng(seed)
env.action_space.seed(int(seed))
planner.reset_episode()
ob, info = env.reset(options=dict(task_id=args.task, render_goal=not args.no_render))
goal = info['goal']
out = os.path.join(args.out_dir, args.env, f'task{args.task}_ep{args.episode}')
os.makedirs(out, exist_ok=True)


def render():
    return None if args.no_render else np.asarray(env.render(camera=args.camera) if args.camera else env.render()).copy()


frames, ts, picks, obs_at = [], [], [], []
if not args.no_render and info.get('goal_rendered') is not None:
    import imageio
    imageio.imwrite(os.path.join(out, 'goal.png'), np.asarray(info['goal_rendered']))
t, done, success = 0, False, 0.0
while not done:
    if t % args.k == 0:  # arbitration happens inside planner.act at this step
        frames.append(render()); ts.append(t); obs_at.append(np.asarray(ob).copy())
    action = np.clip(planner.act(ob, goal, temperature=0.0), -1, 1)
    if t % args.k == 0:
        picks.append(pol.index(planner._current))
    ob, _, terminated, truncated, info = env.step(action)
    t += 1; success = float(info['success']); done = terminated or truncated
frames.append(render()); ts.append(t); obs_at.append(np.asarray(ob).copy()); picks.append(-1)
chain_ok = (t == logged_steps) and np.array_equal(np.array(ts[:-1]), logged_t) and np.array_equal(np.array(picks[:-1]), logged_pick)
print(f'replayed {t} steps, success {success}; chain matches og50dec log: {chain_ok}')
assert chain_ok, 'replay diverged from the logged episode'
if not args.no_render:
    import imageio
    for tt, fr in zip(ts, frames):
        imageio.imwrite(os.path.join(out, f't{tt:03d}.png'), fr)
    np.savez_compressed(os.path.join(out, 'frames.npz'), t=np.array(ts), frame=np.stack(frames), policy_idx=np.array(picks),
                        obs=np.stack(obs_at), goal=goal, policies=np.array(pol), success=success, steps=t)
    print('wrote', out)
