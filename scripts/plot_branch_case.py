"""Qualitative 'one real state, six branches' figure (prototype, 2026-09-09).

Stage 1 (--compute): from every stored WMPP decision state in oracle_wmpp/<env>
(states.npz / branches.npz, true-simulator branches of the seed-0 bank), imagine
each bank policy closed-loop through the world model for --horizon steps and
record the ensemble value trace. Cached to <cache_dir>/<env>.npz. Zero simulator
touches. Runs on CPU in ~5 min for cube-double-play (142 states x 6 policies).

Stage 2 (default): plot one state --state as three panels: (a) top-down real
state, (b) true-simulator cube-goal distance of each policy's branch, (c) imagined
value of each policy with the max-over-first-k pick. Candidate states (WM pick
succeeds, best fixed policy fails) are listed with --list.

  source /scratch/jwquan/wmpp/venv/bin/activate
  export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu
  python scripts/plot_branch_case.py --env cube-double-play-v0 --compute
  python scripts/plot_branch_case.py --env cube-double-play-v0 --list
  python scripts/plot_branch_case.py --env cube-double-play-v0 --state 62 --k 10 \
      --out WMPP_ICLR2027/figures/branch_case_cube_double
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='cube-double-play-v0')
ap.add_argument('--oracle_root', default='/scratch/jwquan/wmpp/oracle_wmpp')
ap.add_argument('--cache_dir', default='/scratch/jwquan/wmpp/planner_eval/branch_case')
ap.add_argument('--horizon', type=int, default=50)
ap.add_argument('--compute', action='store_true')
ap.add_argument('--list', action='store_true')
ap.add_argument('--state', type=int, default=62)
ap.add_argument('--k', type=int, default=10)
ap.add_argument('--out', default=None)
args = ap.parse_args()
cache = os.path.join(args.cache_dir, f'{args.env}.npz')


def compute():
    from interfaces.policy_bank import load_bank
    from world_model.model import EnsembleWorldModel
    from world_model.rollout import TransitionCounter, imagine_policy_rollout
    cfg = json.load(open(f'{ROOT}/manifests/wmpp_env_config.json'))[args.env]
    O = os.path.join(args.oracle_root, args.env)
    states = dict(np.load(f'{O}/states.npz')); br = dict(np.load(f'{O}/branches.npz'))
    policies = [str(p) for p in br['policies']]
    wm = EnsembleWorldModel.load(cfg['wm_dir'], cfg['wm_epoch'])
    bank = load_bank(args.env, cfg['policy_root'], cfg['policy_epoch'], seeds=[0])
    S, P, E, H = len(states['state_id']), len(policies), wm.config['num_members'], args.horizon
    outcomes = np.full((S, P), np.nan); outcomes[br['state_id'], br['policy_idx']] = br['success']
    bobs = np.full((S, P) + br['branch_obs'].shape[1:], np.nan, dtype=np.float32)
    bobs[br['state_id'], br['policy_idx']] = br['branch_obs']
    blen = np.zeros((S, P), int); blen[br['state_id'], br['policy_idx']] = br['branch_len']
    vals = np.zeros((S, P, E, H), np.float32); counter = TransitionCounter()
    for s in range(S):
        obs0, goal = states['obs'][s], states['goal'][s]
        for p, name in enumerate(policies):
            traj = imagine_policy_rollout(wm, bank[name], obs0, goal, H, counter)['obs_traj'][1:]  # (H, E, d)
            traj_e = np.moveaxis(traj, 1, 0)
            vals[s, p] = np.asarray(wm.value_score(traj_e, np.broadcast_to(goal, traj_e.shape[:-1] + goal.shape[-1:])))
        if (s + 1) % 20 == 0:
            print(f'{s + 1}/{S}', flush=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    np.savez_compressed(cache, policies=np.array(policies), obs=states['obs'], goal=states['goal'],
                        task_id=states['task_id'], episode_idx=states['episode_idx'], t=states['t'],
                        reset_seed=states['reset_seed'], base_success=states['base_success'],
                        outcomes=outcomes, bobs=bobs, blen=blen, vals=vals)
    print('wrote', cache, 'transitions', counter.total)


if args.compute:
    compute(); sys.exit()

d = np.load(cache); pol = [str(p) for p in d['policies']]
fam = [p.split('-')[0] for p in pol]
vals_all = d['vals']; vm_all = vals_all.mean(2); out_all = d['outcomes']
K = args.k
score_all = vm_all[:, :, :K].max(2); pick_all = score_all.argmax(1)
if args.list:
    S = len(pol) and vm_all.shape[0]
    print('state task ep t base | outcomes', pol, '| pick ok margin')
    for s in range(S):
        o = out_all[s]
        if o.max() > o.min() and o[pick_all[s]] == 1:
            srt = np.sort(score_all[s])
            print(s, int(d['task_id'][s]), int(d['episode_idx'][s]), int(d['t'][s]), int(d['base_success'][s]), '|',
                  o.astype(int), '|', pol[pick_all[s]], f'{srt[-1] - srt[-2]:.2f}')
    sys.exit()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#a8a7a1', '#e6e5e0'
ORDER = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
LABEL = dict(gcbc='GCBC', gcivl='GCIVL', gciql='GCIQL', qrl='QRL', crl='CRL', hiql='HIQL')
COL = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']))
CM, TOL_CM, NCUBE = 10.0, 4.0, 2  # 1 obs unit = 0.1 m; cube blocks at 19 + 9 i
s = args.state; out = args.out or f'branch_case_{args.env}_s{s}'
obs, goal = d['obs'][s], d['goal'][s]
vals = vals_all[s]; vm = vm_all[s]; score = score_all[s]; pick = int(pick_all[s]); outc = out_all[s]
order = np.argsort([ORDER.index(f) for f in fam])


def cube_dist_cm(o, g):
    return CM * np.stack([np.linalg.norm(o[..., 19 + 9 * i:22 + 9 * i] - g[..., 19 + 9 * i:22 + 9 * i], axis=-1)
                          for i in range(NCUBE)], -1)


def spread_labels(ys, min_gap):
    idx = np.argsort(ys); y = np.array(ys, float)
    for a, b in zip(idx[:-1], idx[1:]):
        if y[b] - y[a] < min_gap:
            y[b] = y[a] + min_gap
    return y


fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8), gridspec_kw=dict(width_ratios=[0.78, 1.1, 1.1], wspace=0.5))
# (a) real state, top-down
ax = axes[0]; ee = obs[12:14] * CM; pts = []
for i, mk in enumerate(['s', 'D']):
    c = obs[19 + 9 * i:21 + 9 * i] * CM; g = goal[19 + 9 * i:21 + 9 * i] * CM; pts += [c, g]
    at_goal = np.linalg.norm(c - g) <= TOL_CM
    ax.scatter(*g, marker=mk, s=95 if at_goal else 60, facecolor='none', edgecolor=INK2, linestyle='--', linewidth=0.9, zorder=2)
    ax.scatter(*c, marker=mk, s=45, color=INK2, zorder=3)
    if not at_goal:
        ax.annotate('', xy=g, xytext=c, arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8, shrinkA=4, shrinkB=5))
        ax.text(g[0] + 1.5, g[1], f'goal {i + 1}', ha='left', va='center', fontsize=6, color=INK2)
        ax.text(c[0], c[1] - 2.4, f'cube {i + 1}', ha='center', va='top', fontsize=6, color=INK)
    else:
        ax.text(c[0] + 2.0, c[1], f'cube {i + 1}\n(placed)', ha='left', va='center', fontsize=6, color=INK)
ax.scatter(*ee, marker='+', s=45, color=INK, linewidth=1.1, zorder=4)
ax.text(ee[0] + 1.8, ee[1] + 1.8, 'gripper', ha='left', va='bottom', fontsize=6, color=INK)
pts = np.array(pts + [ee]); lo, hi = pts.min(0) - 7, pts.max(0) + 9
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.set_title(f'(a) Real state ($t{{=}}{int(d["t"][s])}$)', loc='left')
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
# (b) true simulator branches
ax = axes[1]; ends = []
for p in order:
    L = int(d['blen'][s, p]); dist = cube_dist_cm(d['bobs'][s, p, :L + 1], goal).max(-1); ok = outc[p] == 1
    ax.plot(np.arange(L + 1), dist, color=COL[fam[p]], lw=1.5 if ok else 1.1, ls='-' if ok else (0, (3, 2)))
    ends.append((p, dist[-1], ok))
for (p, y0, ok), y in zip(ends, spread_labels([e[1] for e in ends], 3.2)):
    ax.text(103, y, LABEL[fam[p]] + (' ✓' if ok else ''), color=COL[fam[p]], fontsize=6, va='center')
ax.axhline(TOL_CM, color=MUTED, lw=0.8, ls=':'); ax.text(1, TOL_CM + 1, 'success (4 cm)', fontsize=5.5, color=INK2)
ax.set_xlabel('real steps after the decision'); ax.set_ylabel('cube–goal distance (cm)')
ax.set_title('(b) True simulator', loc='left')
ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(bottom=0)
ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
# (c) imagined value
ax = axes[2]; Hh = vals.shape[-1]; h = np.arange(1, Hh + 1); ends = []
ax.axvspan(0.5, K + 0.5, color=GRID, alpha=0.7, lw=0)
for p in order:
    ok = outc[p] == 1
    ax.fill_between(h, vals[p].min(0), vals[p].max(0), color=COL[fam[p]], alpha=0.10, lw=0)
    ax.plot(h, vm[p], color=COL[fam[p]], lw=1.5 if ok else 1.1, ls='-' if ok else (0, (3, 2)))
    ends.append((p, vm[p, -1]))
    ax.plot([int(vm[p, :K].argmax()) + 1], [score[p]], marker='o', ms=4.5 if p == pick else 3, color=COL[fam[p]],
            mec='white', mew=0.7, zorder=5)
yr = np.ptp(vm)
for (p, y0), y in zip(ends, spread_labels([e[1] for e in ends], 0.07 * yr)):
    ax.text(Hh + 1.5, y, LABEL[fam[p]], color=COL[fam[p]], fontsize=6, va='center')
hk = int(vm[pick, :K].argmax()) + 1
ax.annotate(f'arg max: {LABEL[fam[pick]]}', xy=(hk, score[pick]), xytext=(K + 5, score[pick] + 0.12 * yr), fontsize=6.5,
            color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkB=3))
ax.text(K + 1.2, 0.97, f'$k{{=}}{K}$', ha='left', va='top', fontsize=6, color=INK2, transform=ax.get_xaxis_transform())
ax.set_xlabel('imagined steps'); ax.set_ylabel(r'imagined value $V(\hat{s},g)$')
ax.set_title('(c) World model', loc='left')
ax.set_xlim(0, Hh + 12); ax.set_xticks([0, 10, 25, 50])
ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.legend(handles=[Line2D([], [], color=INK2, lw=1.5, label='policy reaches the goal in the true simulator (b)'),
                    Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 2)), label='policy fails (b)')],
           loc='lower center', ncol=2, frameon=False, handlelength=2.2, bbox_to_anchor=(0.5, -0.2), columnspacing=1.5)
fig.savefig(out + '.png', dpi=250, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
print('state', s, 'task', int(d['task_id'][s]), 'ep', int(d['episode_idx'][s]), 't', int(d['t'][s]),
      '| pick', pol[pick], '| outcomes', {LABEL[fam[p]]: int(outc[p]) for p in order},
      '| scores', {LABEL[fam[p]]: round(float(score[p]), 2) for p in order})
