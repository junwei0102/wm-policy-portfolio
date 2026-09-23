"""WMPP flow figure from rendered frames: key states of one real episode as MuJoCo images,
with the policy the arbiter selected at that state written under each frame, plus the
executed-policy band over the whole episode.

Input: frames.npz from scripts/render_episode_frames.py (frames at every arbitration step
and at the final step) and the decision log (scores) from eval_planner --dump_decisions.

  python scripts/plot_episode_frames.py --env cube-double-play-v0 --task 5 --episode 31 \
      --frames 0,20,70,120,133 --out WMPP_ICLR2027/figures/episode_frames_cube_double
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--env', default='cube-double-play-v0')
ap.add_argument('--task', type=int, default=5)
ap.add_argument('--episode', type=int, default=31)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--k', type=int, default=10)
ap.add_argument('--frames', default='0,20,70,120,133', help='env steps to show (must be arbitration steps or the final step)')
ap.add_argument('--notes', default=None, help='optional "|"-separated caption per frame, same order as --frames')
ap.add_argument('--crop', default='0.12,0.05,0.88,0.85', help='fractional crop x0,y0,x1,y1 of each frame')
ap.add_argument('--frames_root', default='/scratch/jwquan/wmpp/planner_eval/episode_frames')
ap.add_argument('--eval_root', default='/scratch/jwquan/wmpp/planner_eval')
ap.add_argument('--tag', default='og50dec')
ap.add_argument('--out', default=None)
args = ap.parse_args()

ORDER = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
LABEL = dict(gcbc='GCBC', gcivl='GCIVL', gciql='GCIQL', qrl='QRL', crl='CRL', hiql='HIQL')
COL = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']))
# pastel fills (draw.io style) so the badges match the teaser figure; text stays dark on them
COL.update(hiql='#E1D5E7', gciql='#FFE6CC', gcivl='#D5E8D4')
EDGE = dict(hiql='#9673A6', gciql='#D79B00', gcivl='#82B366')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#a8a7a1', '#e6e5e0'
CM, TOL_CM, NCUBE = 10.0, 4.0, {'single': 1, 'double': 2, 'triple': 3, 'quadruple': 4}[args.env.split('-')[1]]

fr = np.load(os.path.join(args.frames_root, args.env, f'task{args.task}_ep{args.episode}', 'frames.npz'))
pol = [str(p) for p in fr['policies']]; fam = [p.split('-')[0] for p in pol]
t_all, pick_all, steps = fr['t'], fr['policy_idx'], int(fr['steps'])
z = np.load(os.path.join(args.eval_root, args.env, f'bank_sd{args.seed}_{args.tag}', f'decisions_score{args.k}_commit{args.k}.npz'))
m = (z['task_id'] == args.task) & (z['episode_idx'] == args.episode)
sc_by_t = dict(zip(z['t'][m].tolist(), z['scores'][m]))
assert [str(p) for p in z['policies']] == pol
show = [int(x) for x in args.frames.split(',')]
notes = args.notes.split('|') if args.notes else [None] * len(show)
x0, y0, x1, y1 = (float(v) for v in args.crop.split(','))
obs_all, goal = fr['obs'], fr['goal']
cube_d = np.stack([CM * np.linalg.norm(obs_all[:, 19 + 9 * i:22 + 9 * i] - goal[19 + 9 * i:22 + 9 * i], axis=1) for i in range(NCUBE)], 1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
# ---- font sizes and layout (points); tweak here ----
FS_PICK, FS_NOTE, FS_BAND, FS_XLABEL = 7.8, 7.5, 7.0, 0   # pick badge, note under frame, policy name in band, band caption
FIGSIZE, BAND_RATIO = (5.5, 2.3), 0.17                     # figure size (in); band height relative to the frame row
PICK_WORD = 'WMPA picks'                                    # badge text under each arbitration frame
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'pdf.fonttype': 42, 'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral'],
                     'mathtext.fontset': 'stix'})
out = args.out or os.path.join(ROOT, 'WMPP_ICLR2027', 'figures', f'episode_frames_{args.env.replace("-v0", "").replace("-", "_")}')

n = len(show)
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(2, n, height_ratios=[1.0, BAND_RATIO], hspace=0.60, wspace=0.06, left=0.005, right=0.995, top=0.995, bottom=0.02)
axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
for i, (ax, t) in enumerate(zip(axes, show)):
    j = int(np.flatnonzero(t_all == t)[0])
    img = fr['frame'][j]; H, W = img.shape[:2]
    ax.imshow(img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)], interpolation='lanczos')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID); s.set_linewidth(0.6)
    placed = int((cube_d[j] <= TOL_CM).sum())
    p = int(pick_all[j])
    if p >= 0:
        s_ = sc_by_t[t]; srt = np.sort(s_); lead = srt[-1] - srt[-2]
        ax.text(0.5, -0.05, f'{PICK_WORD} {LABEL[fam[p]]}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_PICK, color=INK,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor=COL[fam[p]], edgecolor=EDGE.get(fam[p], INK2), lw=0.6))
    else:
        ax.text(0.5, -0.05, 'success', transform=ax.transAxes, ha='center', va='top', fontsize=FS_PICK, color=INK, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=INK2, lw=0.6))
    note = notes[i] or (f'{placed}/{NCUBE} cube{"s" if NCUBE > 1 else ""} placed' + (', holding' if obs_all[j, 18] > 0.5 else ''))
    ax.text(0.5, -0.27, note.replace('\\n', '\n'), transform=ax.transAxes, ha='center', va='top', fontsize=FS_NOTE, color=INK2, linespacing=1.1)
    if i < n - 1:
        fig.patches.append(FancyArrowPatch((0.985, 0.5), (1.06, 0.5), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=7,
                                           color=INK2, lw=0.7, clip_on=False))
# executed-policy band over the whole episode
axb = fig.add_subplot(gs[1, :])
for j, (t, p) in enumerate(zip(t_all[:-1], pick_all[:-1])):
    t1 = t_all[j + 1]
    axb.add_patch(Rectangle((t, 0), t1 - t, 1, facecolor=COL[fam[p]], edgecolor=EDGE.get(fam[p], INK2), lw=0.5))
    if t1 - t >= 5:  # every block carries its policy name, centered (blocks narrower than 5 steps stay unlabeled)
        axb.text((t + t1) / 2, 0.5, LABEL[fam[p]], fontsize=FS_BAND, color=INK, va='center', ha='center')
axb.set_xlim(0, steps); axb.set_ylim(0, 1); axb.set_yticks([])
axb.set_xticks([])
BAND_CAPTION = ''  # caption under the band; '' = none (was 'policy executed over the episode (one block per arbitration)')
if BAND_CAPTION:
    axb.set_xlabel(BAND_CAPTION, fontsize=FS_XLABEL, labelpad=2)
for s in axb.spines.values():
    s.set_visible(False)
fig.savefig(out + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02); fig.savefig(out + '.pdf', bbox_inches='tight', dpi=300, pad_inches=0.02)
print('figure ->', out + '.{pdf,png}')
