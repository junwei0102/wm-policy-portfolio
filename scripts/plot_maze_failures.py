"""Failure anatomy on a point/ant maze: xy paths of failed episodes, coloured by the policy the
arbiter executes, plus the per-step scores of every bank member for one failed episode.

Reads the decision dumps written by `eval_planner.py --dump_decisions` (decisions_<variant>.npz;
at c=1 every environment step is a decision, so the dump is the full trajectory).

  python scripts/plot_maze_failures.py --env_name=pointmaze-medium-navigate-v0 --tag=og50dump \
      --variant=score1_commit1 --tasks=3,4 --out_dir=/scratch/jwquan/wmpp/analysis/pointmaze_failures
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.patches import Rectangle

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'pointmaze-medium-navigate-v0', '')
flags.DEFINE_string('eval_root', '/scratch/jwquan/wmpp/planner_eval', '')
flags.DEFINE_string('tag', 'og50dump', 'dir tag of the --dump_decisions runs.')
flags.DEFINE_string('variant', 'score1_commit1', 'planner variant to plot.')
flags.DEFINE_string('reference', 'random_commit5', 'variant drawn in the comparison row ("" = none).')
flags.DEFINE_string('tasks', '3,4', 'task ids to plot.')
flags.DEFINE_string('seeds', '0,1,2', 'bank seeds.')
flags.DEFINE_integer('max_paths', 40, 'paths drawn per panel.')
flags.DEFINE_float('stall_eps', 0.05, 'per-step displacement below which a step counts as stalled.')
flags.DEFINE_string('out_dir', '/scratch/jwquan/wmpp/analysis/pointmaze_failures', '')

COLORS = {'crl': '#9467bd', 'gcbc': '#8c564b', 'gciql': '#d62728', 'gcivl': '#ff7f0e', 'hiql': '#2ca02c', 'qrl': '#1f77b4'}


def maze_geometry(env_name):
    import ogbench
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    u = env.unwrapped
    geo = dict(maze=np.array(u.maze_map), unit=float(u._maze_unit), ox=float(u._offset_x), oy=float(u._offset_y),
               tasks={i + 1: dict(init=t['init_xy'], goal=t['goal_xy']) for i, t in enumerate(u.task_infos)})
    env.close()
    return geo


def draw_maze(ax, geo):
    m, u = geo['maze'], geo['unit']
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == 1:
                ax.add_patch(Rectangle((j * u - geo['ox'] - u / 2, i * u - geo['oy'] - u / 2), u, u, fc='#444', ec='none'))
    ax.set_xlim(-geo['ox'] - u / 2, (m.shape[1] - 1) * u - geo['ox'] + u / 2)
    ax.set_ylim(-geo['oy'] - u / 2, (m.shape[0] - 1) * u - geo['oy'] + u / 2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def episodes(npz):
    """Split a decisions_*.npz into per-episode dicts."""
    key = np.stack([npz['task_id'], npz['episode_idx']], axis=1)
    cuts = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1) | (np.diff(npz['t']) < 0)) + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(key)]):
        out.append(dict(task=int(npz['task_id'][a]), ep=int(npz['episode_idx'][a]), success=bool(npz['ep_success'][a]),
                        steps=int(npz['ep_steps'][a]), t=npz['t'][a:b], xy=npz['obs'][a:b, :2], goal=npz['goal'][a, :2],
                        scores=npz['scores'][a:b], pol=npz['policy_idx'][a:b]))
    return out


def main(_):
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    geo = maze_geometry(FLAGS.env_name)
    tasks = [int(x) for x in FLAGS.tasks.split(',')]
    seeds = [int(x) for x in FLAGS.seeds.split(',')]
    data, names = {}, None
    for v in [FLAGS.variant] + ([FLAGS.reference] if FLAGS.reference else []):
        for s in seeds:
            p = os.path.join(FLAGS.eval_root, FLAGS.env_name, f'bank_sd{s}_{FLAGS.tag}', f'decisions_{v}.npz')
            z = np.load(p)
            names = [str(n).rsplit('-sd', 1)[0] for n in z['policies']]
            data[(v, s)] = episodes(z)

    # ---- figure 1: xy paths, rows = (failed, successful, reference), cols = task x seed
    rows = [('failed', FLAGS.variant, False), ('successful', FLAGS.variant, True)]
    if FLAGS.reference:
        rows.append((f'{FLAGS.reference} (all)', FLAGS.reference, None))
    fig, axes = plt.subplots(len(rows), len(tasks) * len(seeds), figsize=(3.0 * len(tasks) * len(seeds), 3.1 * len(rows)), squeeze=False)
    stats = {}
    for c, (task, s) in enumerate((t, s) for t in tasks for s in seeds):
        for r, (lab, v, want) in enumerate(rows):
            ax = axes[r, c]; draw_maze(ax, geo)
            eps = [e for e in data[(v, s)] if e['task'] == task and (want is None or e['success'] == want)]
            for e in eps[:FLAGS.max_paths]:
                xy, pol = e['xy'], e['pol']
                for i in range(len(names)):  # one scatter per policy keeps the colour legend exact
                    msk = pol == i
                    if msk.any():
                        ax.scatter(xy[msk, 0], xy[msk, 1], s=2.0, color=COLORS.get(names[i], '#000'), alpha=0.55, lw=0, rasterized=True)
                if want is False:
                    ax.plot(*xy[-1], marker='x', ms=5, mew=1.2, color='k')
            ti = geo['tasks'][task]
            ax.plot(*ti['init'], marker='o', ms=7, mfc='w', mec='k', mew=1.2, zorder=5)
            ax.plot(*ti['goal'], marker='*', ms=12, mfc='gold', mec='k', mew=0.8, zorder=5)
            ax.set_title(f'task {task}, bank seed {s}\n{v}: {lab} ({len(eps)})', fontsize=8)
            if want is False:
                end = np.array([e['xy'][-1] for e in eps]) if eps else np.zeros((0, 2))
                last = [names[np.bincount(e['pol'][-200:], minlength=len(names)).argmax()] for e in eps]
                stall = [float((np.linalg.norm(np.diff(e['xy'], axis=0), axis=1) < FLAGS.stall_eps).mean()) for e in eps]
                nosw = [bool((e['pol'] == e['pol'][0]).all()) for e in eps]
                stats[f'task{task}_sd{s}'] = dict(n_failed=len(eps), never_switched=int(np.sum(nosw)),
                                                  stalled_step_fraction=float(np.mean(stall)) if stall else None,
                                                  policy_in_last_200_steps={n: last.count(n) for n in set(last)},
                                                  end_xy_mean=end.mean(0).tolist() if len(end) else None,
                                                  end_xy_std=end.std(0).tolist() if len(end) else None)
    handles = [plt.Line2D([], [], marker='o', ls='', color=COLORS[n], label=n.upper()) for n in names]
    fig.legend(handles=handles, loc='lower center', ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p1 = os.path.join(FLAGS.out_dir, f'paths_{FLAGS.variant}.png'); fig.savefig(p1, dpi=170); plt.close(fig)

    # ---- figure 2: per-step scores in one failed and one successful episode of each task (first seed that has both)
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.0 * len(tasks), 6.0), squeeze=False)
    for c, task in enumerate(tasks):
        for r, want in enumerate((False, True)):
            ax = axes[r, c]
            pick = next(((s, e) for s in seeds for e in data[(FLAGS.variant, s)] if e['task'] == task and e['success'] == want), None)
            if pick is None:
                ax.set_axis_off(); continue
            s, e = pick
            T = min(len(e['t']), 300)
            for i, n in enumerate(names):
                ax.plot(e['t'][:T], e['scores'][:T, i], color=COLORS.get(n, '#000'), lw=1.2, label=n.upper())
            ymin = ax.get_ylim()[0]
            ax.scatter(e['t'][:T], np.full(T, ymin), c=[COLORS.get(names[i], '#000') for i in e['pol'][:T]], s=6, marker='s', lw=0)
            ax.set_title(f"task {task}, bank seed {s}, episode {e['ep']}: {'success' if want else 'FAILED'} ({e['steps']} steps)\n"
                         'lines = score of each member, bottom band = executed policy', fontsize=9)
            ax.set_xlabel('environment step'); ax.set_ylabel('score  (-latent distance to goal)')
            if r == 0 and c == 0:
                ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    p2 = os.path.join(FLAGS.out_dir, f'scores_{FLAGS.variant}.png'); fig.savefig(p2, dpi=150); plt.close(fig)
    json.dump(stats, open(os.path.join(FLAGS.out_dir, f'stats_{FLAGS.variant}.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1)); print('wrote', p1, p2)


if __name__ == '__main__':
    app.run(main)
