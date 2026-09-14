"""One real WMPP episode as a flow figure: timeline of executed policies + task progress,
and the arbitration at two or three highlighted decision points (imagined value of every
bank policy under the world model, arg max = the policy that was executed).

Data: decisions_<variant>.npz from `eval_planner.py --dump_decisions` (state, goal and
per-policy score at every arbitration of the real episode). The highlighted decisions are
re-imagined offline with the same world model and bank so the full k-step value traces can
be drawn; the recomputed scores are asserted equal to the logged ones.

  source /scratch/jwquan/wmpp/venv/bin/activate
  export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu
  python scripts/plot_episode_flow.py --env cube-double-play-v0 --task 5 --episode 31 \
      --decisions 20,80,120 --out WMPP_ICLR2027/figures/episode_flow_cube_double
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='cube-double-play-v0')
ap.add_argument('--eval_root', default='/scratch/jwquan/wmpp/planner_eval')
ap.add_argument('--tag', default='og50dec')
ap.add_argument('--variant', default='score10_commit10')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--task', type=int, default=5)
ap.add_argument('--episode', type=int, default=31)
ap.add_argument('--decisions', default='20,80,120', help='env steps t of the highlighted decisions')
ap.add_argument('--k', type=int, default=10)
ap.add_argument('--out', default=None)
args = ap.parse_args()

NCUBE = {'single': 1, 'double': 2, 'triple': 3, 'quadruple': 4}[args.env.split('-')[1]]
CM, TOL_CM = 10.0, 4.0
ORDER = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
LABEL = dict(gcbc='GCBC', gcivl='GCIVL', gciql='GCIQL', qrl='QRL', crl='CRL', hiql='HIQL')
COL = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']))
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#a8a7a1', '#e6e5e0'
CIRCLED = '①②③④⑤'
CUBE_NAMES = ['red', 'blue', 'orange', 'green']  # OGBench cube colours by index (cube_env.py)

# ----------------------------------------------------------------------------- episode data
z = np.load(os.path.join(args.eval_root, args.env, f'bank_sd{args.seed}_{args.tag}', f'decisions_{args.variant}.npz'))
pol = [str(p) for p in z['policies']]; fam = [p.split('-')[0] for p in pol]
m = (z['task_id'] == args.task) & (z['episode_idx'] == args.episode)
assert m.any(), 'episode not found'
t_dec, obs, goal, scores, pick = z['t'][m], z['obs'][m], z['goal'][m], z['scores'][m], z['policy_idx'][m]
ep_steps, ep_success = int(z['ep_steps'][m][0]), float(z['ep_success'][m][0])
print(f'{args.env} task {args.task} ep {args.episode}: {ep_steps} steps, success {ep_success}, '
      f'{len(t_dec)} decisions, chain ' + '>'.join(f'{fam[p]}@{t}' for t, p in zip(t_dec, pick)))
cube_d = np.stack([CM * np.linalg.norm(obs[:, 19 + 9 * i:22 + 9 * i] - goal[:, 19 + 9 * i:22 + 9 * i], axis=1)
                   for i in range(NCUBE)], 1)  # (D, C) cm
hl = [int(x) for x in args.decisions.split(',')]
hl_idx = [int(np.flatnonzero(t_dec == t)[0]) for t in hl]

# ----------------------------------------------------------------------------- re-imagine the highlighted decisions
from interfaces.policy_bank import load_bank
from world_model.model import EnsembleWorldModel
from world_model.rollout import TransitionCounter, imagine_policy_rollout

cfg = json.load(open(os.path.join(ROOT, 'manifests', 'wmpp_env_config.json')))[args.env]
wm = EnsembleWorldModel.load(cfg['wm_dir'], cfg['wm_epoch'])
bank = load_bank(args.env, cfg['policy_root'], cfg['policy_epoch'], seeds=[args.seed])
assert sorted(bank) == pol, (sorted(bank), pol)
E = wm.config['num_members']; K = args.k
traces = {}  # decision idx -> (P, E, K) value traces
for j in hl_idx:
    v = np.zeros((len(pol), E, K), np.float32)
    for p, name in enumerate(pol):
        traj = imagine_policy_rollout(wm, bank[name], obs[j], goal[j], K, TransitionCounter())['obs_traj'][1:]
        traj_e = np.moveaxis(traj, 1, 0)
        v[p] = np.asarray(wm.value_score(traj_e, np.broadcast_to(goal[j], traj_e.shape[:-1] + goal[j].shape[-1:])))
    rec = v.mean(1).max(1)
    assert np.allclose(rec, scores[j], atol=1e-3), (rec, scores[j])
    assert int(rec.argmax()) == int(pick[j])
    traces[j] = v
print('recomputed scores match the logged ones at', hl)

# ----------------------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
order = np.argsort([ORDER.index(f) for f in fam])
out = args.out or os.path.join(ROOT, 'WMPP_ICLR2027', 'figures', f'episode_flow_{args.env.replace("-v0", "").replace("-", "_")}')

fig = plt.figure(figsize=(5.5, 3.15))
gs = fig.add_gridspec(2, len(hl), height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.42, left=0.09, right=0.99, top=0.93, bottom=0.12)
ax = fig.add_subplot(gs[0, :])
# progress curves (one per cube), sampled at the decision times
styles = ['-', (0, (3, 1.5)), (0, (1, 1)), (0, (5, 1, 1, 1))]
for i in range(NCUBE):
    ax.plot(t_dec, cube_d[:, i], color=INK, lw=1.2, ls=styles[i], marker='o', ms=2.2, mew=0, label=f'{CUBE_NAMES[i]} cube → goal')
ax.axhline(TOL_CM, color=MUTED, lw=0.7, ls=':'); ax.text(1, TOL_CM + 0.5, 'success tol. (4 cm)', fontsize=5.5, color=INK2, va='bottom')
# executed-policy band under the curves
ymax = float(np.nanmax(cube_d)) * 1.08; band_h = 0.16 * ymax; y0 = -band_h - 0.04 * ymax
for j, (t, p) in enumerate(zip(t_dec, pick)):
    t1 = t_dec[j + 1] if j + 1 < len(t_dec) else ep_steps
    ax.add_patch(Rectangle((t, y0), t1 - t, band_h, facecolor=COL[fam[p]], edgecolor='white', lw=0.6, clip_on=False))
    if (j == 0 or fam[pick[j - 1]] != fam[p]) and t1 - t >= 7:
        ax.text(t + 0.6, y0 + band_h / 2, LABEL[fam[p]], fontsize=5.2, color='white', va='center', ha='left', clip_on=False)
ax.text(-1.5, y0 + band_h / 2, 'executes', fontsize=6, color=INK2, va='center', ha='right', clip_on=False)
# highlighted decisions
for n, j in enumerate(hl_idx):
    ax.axvline(t_dec[j], color=INK2, lw=0.6, ls='--', zorder=0)
    ax.text(t_dec[j] + 1.2, ymax * 0.97, CIRCLED[n], fontsize=9, ha='left', va='top', color=INK)
if ep_success:
    ax.plot([ep_steps], [TOL_CM], marker='*', ms=7, color=INK, mec='white', mew=0.5, clip_on=False, zorder=5)
    ax.text(ep_steps - 3, TOL_CM + 0.05 * ymax, f'success $t={ep_steps}$', fontsize=5.8, ha='right', va='bottom', color=INK)
ax.set_xlim(0, ep_steps + 2); ax.set_ylim(0, ymax)
ax.set_xlabel('real environment step  (arbitration every $k=c=%d$ steps)' % K, labelpad=1)
ax.set_ylabel('distance to goal (cm)')
ax.set_title(f'(a) WMPP on one {args.env.replace("-v0", "")} episode (task {args.task}; every fixed policy fails on it)', loc='left', pad=4)
ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper center', frameon=False, ncol=NCUBE, handlelength=2.2, borderaxespad=0.1, bbox_to_anchor=(0.62, 1.0))

# bottom: arbitration at each highlighted decision
h = np.arange(1, K + 1)
for n, j in enumerate(hl_idx):
    axb = fig.add_subplot(gs[1, n])
    v = traces[j]; vm = v.mean(1); sc = vm.max(1); w = int(pick[j])
    for p in order:
        win = p == w
        axb.fill_between(h, v[p].min(0), v[p].max(0), color=COL[fam[p]], alpha=0.10, lw=0)
        axb.plot(h, vm[p], color=COL[fam[p]], lw=1.7 if win else 0.9, alpha=1 if win else 0.8, zorder=4 if win else 2)
        axb.plot([int(vm[p].argmax()) + 1], [sc[p]], marker='o', ms=4.5 if win else 2.5, color=COL[fam[p]], mec='white', mew=0.6, zorder=5)
    # end labels, de-overlapped
    ys = vm[:, -1].copy(); idx = np.argsort(ys); gap = 0.075 * np.ptp(vm)
    for a, b in zip(idx[:-1], idx[1:]):
        ys[b] = max(ys[b], ys[a] + gap)
    for p in order:
        axb.text(K + 0.4, ys[p], LABEL[fam[p]], color=COL[fam[p]], fontsize=5.8, va='center', fontweight='bold' if p == w else 'normal')
    axb.set_xlim(0.5, K + 3.2); axb.set_xticks([1, 5, 10])
    axb.set_xlabel('imagined step', labelpad=1)
    if n == 0:
        axb.set_ylabel(r'imagined value $V(\hat s, g)$')
    phase = ('holding' if obs[j, 18] > 0.5 else 'free gripper')
    axb.set_title(f'{CIRCLED[n]} $t={t_dec[j]}$ ({phase}): pick {LABEL[fam[w]]}', loc='left', fontsize=7)
    axb.grid(True, axis='y', color=GRID, lw=0.5); axb.set_axisbelow(True)
fig.savefig(out + '.png', dpi=250, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
print('figure ->', out + '.{pdf,png}')
