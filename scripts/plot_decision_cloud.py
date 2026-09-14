"""Decision point cloud: which policy the arbiter picks, as a function of the task state.

Reads decisions_<variant>.npz written by `eval_planner.py --dump_decisions` (one row per
arbitration: t, obs, goal, scores, policy_idx, episode key) for WMPP and Random-Switch on
one env, derives task-phase features from the cube observation layout, and

  1. prints / saves GATE numbers: how predictable the chosen policy is from the state
     (k-NN and softmax regression, 5-fold, chance = majority class), mutual information
     between choice and phase with a permutation null, for WMPP and for Random
     (Random must sit at chance);
  2. draws the figure: (a) WMPP decisions on task axes coloured by the chosen policy,
     (b) Random on the same axes, (c) per-phase choice histograms WMPP vs Random.

Usage (venv):
  python scripts/plot_decision_cloud.py --env cube-double-play-v0 --tag og50dec \
      --wmpp score10_commit10 --random random_commit10 --seeds 0,1,2 \
      --out WMPP_ICLR2027/figures/decision_cloud_cube_double
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
ap.add_argument('--wmpp', default='score10_commit10')
ap.add_argument('--random', default='random_commit10')
ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--out', default=None, help='figure stem (.pdf/.png); default figures/decision_cloud_<env>')
ap.add_argument('--report', default=None, help='JSON report path (default <eval_root>/decision_cloud_<env>.json)')
ap.add_argument('--margin_filter', action='store_true', help='panel (a): only WMPP decisions with score margin above the median')
ap.add_argument('--n_perm', type=int, default=1000)
args = ap.parse_args()

NCUBE = {'single': 1, 'double': 2, 'triple': 3, 'quadruple': 4}[args.env.split('-')[1]]
CM, TOL = 10.0, 0.4  # 1 obs unit = 0.1 m; success tolerance 4 cm
ORDER = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
LABEL = dict(gcbc='GCBC', gcivl='GCIVL', gciql='GCIQL', qrl='QRL', crl='CRL', hiql='HIQL')
COL = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']))
PHASES = [f'reach {i + 1}' if j == 0 else f'carry {i + 1}' for i in range(NCUBE) for j in range(2)] + ['done']


# ----------------------------------------------------------------------------- loading
def load(variant):
    parts = []
    for sd in args.seeds.split(','):
        p = os.path.join(args.eval_root, args.env, f'bank_sd{sd}_{args.tag}', f'decisions_{variant}.npz')
        if not os.path.exists(p):
            print('missing', p); continue
        z = dict(np.load(p, allow_pickle=False))
        z['seed'] = np.full(len(z['t']), int(sd)); z['fam'] = np.array([str(n).split('-')[0] for n in z['policies']])
        parts.append(z)
    assert parts, variant
    keys = [k for k in parts[0] if k not in ('policies', 'fam')]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d['fam'] = parts[0]['fam']
    assert all((p['fam'] == d['fam']).all() for p in parts), 'bank family order differs across seeds'
    return d


def features(d):
    """Task-phase features of every decision (cube envs)."""
    obs, goal = d['obs'], d['goal']
    grip = obs[:, 12:15]
    cube = np.stack([obs[:, 19 + 9 * i:22 + 9 * i] for i in range(NCUBE)], 1)  # (N, C, 3)
    cgoal = np.stack([goal[:, 19 + 9 * i:22 + 9 * i] for i in range(NCUBE)], 1)
    d_goal_all = np.linalg.norm(cube - cgoal, axis=-1)  # (N, C)
    placed = d_goal_all <= TOL
    n_placed = placed.sum(1)
    # target = first unplaced cube (cube index order = task order in cube-*-play goals)
    unplaced = np.where(placed, np.inf, 1.0)
    tgt = np.argmax(~placed, axis=1)  # first False; if all placed -> 0 (ignored via done)
    d_grip = CM * np.linalg.norm(grip - cube[np.arange(len(obs)), tgt], axis=-1)
    d_goal = CM * d_goal_all[np.arange(len(obs)), tgt]
    d_gg = CM * np.linalg.norm(grip - cgoal[np.arange(len(obs)), tgt], axis=-1)  # gripper -> target's goal
    holding = obs[:, 18] > 0.5
    done = n_placed == NCUBE
    phase = np.where(done, 2 * NCUBE, 2 * tgt + holding.astype(int))
    frac = d['t'] / np.maximum(d['ep_steps'], 1)
    sc = d['scores']
    margin = np.full(len(obs), np.nan)
    if not np.isnan(sc).all():
        srt = np.sort(sc, axis=1); margin = srt[:, -1] - srt[:, -2]
    X = np.stack([d_grip, d_goal, holding.astype(float), n_placed.astype(float), frac, obs[:, 17]], 1)
    return dict(d_grip=d_grip, d_goal=d_goal, d_gg=d_gg, holding=holding, n_placed=n_placed, phase=phase, frac=frac,
                margin=margin, X=X, done=done)


# ----------------------------------------------------------------------------- predictability
def _cv_folds(n, k, rng):
    idx = rng.permutation(n); return np.array_split(idx, k)


def knn_cv(X, y, k=15, folds=5, seed=0):
    rng = np.random.default_rng(seed); Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    correct = 0
    for te in _cv_folds(len(y), folds, rng):
        tr = np.setdiff1d(np.arange(len(y)), te)
        for ch in np.array_split(te, max(1, len(te) // 1000)):  # chunked: (1000, n_tr) distance blocks
            D = ((ch_x := Xs[ch])[:, None, :] - Xs[None, tr, :]) ** 2
            nn = np.argpartition(D.sum(-1), k, axis=1)[:, :k]
            votes = np.apply_along_axis(lambda r: np.bincount(r, minlength=y.max() + 1).argmax(), 1, y[tr][nn])
            correct += (votes == y[ch]).sum()
    return correct / len(y)


def softmax_cv(X, y, folds=5, seed=0, epochs=300, lr=0.5, l2=1e-3):
    rng = np.random.default_rng(seed); Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], 1); C = y.max() + 1
    correct = 0
    for te in _cv_folds(len(y), folds, rng):
        tr = np.setdiff1d(np.arange(len(y)), te); W = np.zeros((Xs.shape[1], C))
        Y = np.eye(C)[y[tr]]
        for _ in range(epochs):
            Z = Xs[tr] @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            W -= lr * (Xs[tr].T @ (P - Y) / len(tr) + l2 * W)
        correct += ((Xs[te] @ W).argmax(1) == y[te]).sum()
    return correct / len(y)


def mutual_info(y, z):
    """I(y; z) in bits for two integer arrays."""
    joint = np.zeros((y.max() + 1, z.max() + 1)); np.add.at(joint, (y, z), 1); joint /= joint.sum()
    py, pz = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (py @ pz)[nz])).sum())


def gate(name, d, f, mask=None):
    y = d['policy_idx'].astype(int); X = f['X']; ph = f['phase']; Xraw = d['obs']
    if mask is not None:
        y, X, ph, Xraw = y[mask], X[mask], ph[mask], Xraw[mask]
    chance = np.bincount(y).max() / len(y)
    rng = np.random.default_rng(0)
    mi = mutual_info(y, ph)
    null = np.array([mutual_info(rng.permutation(y), ph) for _ in range(args.n_perm)])
    res = dict(n=int(len(y)), chance=float(chance),
               knn_task=float(knn_cv(X, y)), softmax_task=float(softmax_cv(X, y)),
               knn_raw=float(knn_cv(Xraw, y)),
               mi_phase_bits=mi, mi_null_mean=float(null.mean()), mi_null_p=float((null >= mi).mean()))
    print(f'{name:>28s}  n={res["n"]:6d}  chance={chance:.3f}  kNN(task)={res["knn_task"]:.3f}  '
          f'softmax(task)={res["softmax_task"]:.3f}  kNN(raw)={res["knn_raw"]:.3f}  '
          f'I(choice;phase)={mi:.4f} bits (null {null.mean():.4f}, p={res["mi_null_p"]:.3f})')
    return res


def phase_hist(d, f):
    """(seeds, phases, policies) choice fractions."""
    seeds = np.unique(d['seed']); P = len(d['fam']); out = np.full((len(seeds), len(PHASES), P), np.nan)
    for a, sd in enumerate(seeds):
        for ph in range(len(PHASES)):
            m = (d['seed'] == sd) & (f['phase'] == ph)
            if m.sum():
                out[a, ph] = np.bincount(d['policy_idx'][m].astype(int), minlength=P) / m.sum()
    return out


# ----------------------------------------------------------------------------- main
W = load(args.wmpp); R = load(args.random)
fw, fr = features(W), features(R)
print(f'{args.env}: WMPP {len(W["t"])} decisions, Random {len(R["t"])} decisions; policies {list(W["fam"])}')
print('phase counts WMPP  ', dict(zip(PHASES, np.bincount(fw['phase'], minlength=len(PHASES)))))
print('phase counts Random', dict(zip(PHASES, np.bincount(fr['phase'], minlength=len(PHASES)))))
report = dict(env=args.env, tag=args.tag, wmpp=args.wmpp, random=args.random, phases=PHASES, policies=list(W['fam']))
report['gate'] = {'wmpp': gate('WMPP', W, fw), 'random': gate('Random', R, fr)}
conf = fw['margin'] >= np.nanmedian(fw['margin'])
report['gate']['wmpp_confident'] = gate('WMPP (margin >= median)', W, fw, conf)
hw, hr = phase_hist(W, fw), phase_hist(R, fr)
report['phase_hist'] = {'wmpp_mean': np.nanmean(hw, 0).tolist(), 'random_mean': np.nanmean(hr, 0).tolist()}
rep_path = args.report or os.path.join(args.eval_root, f'decision_cloud_{args.env}.json')
with open(rep_path, 'w') as fh:
    json.dump(report, fh, indent=1)
print('report ->', rep_path)

# ----------------------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
fam = list(W['fam']); order = np.argsort([ORDER.index(x) for x in fam])
out = args.out or os.path.join(ROOT, 'WMPP_ICLR2027', 'figures', f'decision_cloud_{args.env.replace("-v0", "").replace("-", "_")}')

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.45))
rng = np.random.default_rng(0)


def cloud(ax, d, f, title, mask=None):
    m = ~f['done'] if mask is None else (~f['done'] & mask)
    idx = np.flatnonzero(m); rng.shuffle(idx)  # random draw order so no colour sits on top
    sub = idx[::max(1, len(idx) // 4000)]
    cols = np.array([COL[fam[p]] for p in d['policy_idx'][sub]])
    ax.scatter(f['d_grip'][sub], f['d_gg'][sub], s=3, lw=0, c=cols, alpha=0.6, rasterized=True)
    ax.set_xlabel('gripper → cube (cm)'); ax.set_ylabel('gripper → goal (cm)')
    ax.set_title(title, loc='left'); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    xm = np.nanmax(np.r_[fw['d_grip'], fr['d_grip']]); ym = np.nanmax(np.r_[fw['d_gg'], fr['d_gg']])
    ax.set_xlim(-1, xm * 1.04); ax.set_ylim(-1, ym * 1.04)


cloud(axes[0], W, fw, '(a) WMPP decisions', conf if args.margin_filter else None)
cloud(axes[1], R, fr, '(b) Random-Switch')
ax = axes[2]
mw, mr = np.nanmean(hw, 0), np.nanmean(hr, 0)
keep = [i for i in range(len(PHASES)) if not np.isnan(mw[i]).all() and PHASES[i] != 'done']
x = np.arange(len(keep)); wdt = 0.38
for j, (mat, off, hatch) in enumerate([(mw, -wdt / 2, None), (mr, wdt / 2, '////')]):
    bottom = np.zeros(len(keep))
    for p in order:
        vals = mat[keep, p]
        ax.bar(x + off, vals, wdt, bottom=bottom, color=COL[fam[p]], hatch=hatch, edgecolor='white', lw=0.4)
        bottom += vals
ax.set_xticks(x); ax.set_xticklabels([PHASES[i].replace(' ', '\n') for i in keep])
ax.set_xlabel('solid: WMPP, hatched: Random', fontsize=6)
ax.set_ylabel('fraction of decisions'); ax.set_ylim(0, 1); ax.set_title('(c) Choice by phase', loc='left')
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=COL[fam[p]], ms=4, label=LABEL[fam[p]]) for p in order],
           loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.27), columnspacing=1.2, handletextpad=0.3)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + '.png', dpi=250, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight', dpi=250)
print('figure ->', out + '.{pdf,png}')


# ----------------------------------------------------------------------------- fallback: where is each policy over-chosen?
# Density ratio P(policy | cell) / P(policy) on the task axes, one small panel per policy
# (WMPP only; Random is flat at 1 by construction). Cells with < min_n decisions are blank.
fig2, axs = plt.subplots(1, len(order), figsize=(5.5, 1.35), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.12))
m = ~fw['done']; xg, yg = fw['d_grip'][m], fw['d_gg'][m]; pid = W['policy_idx'][m].astype(int)
xe = np.linspace(0, np.nanpercentile(xg, 99), 9); ye = np.linspace(0, np.nanpercentile(yg, 99), 9)
Hall, _, _ = np.histogram2d(xg, yg, [xe, ye]); min_n = 60
im = None
for ax, p in zip(axs, order):
    Hp, _, _ = np.histogram2d(xg[pid == p], yg[pid == p], [xe, ye])
    ratio = np.where(Hall >= min_n, (Hp / np.maximum(Hall, 1)) / (pid == p).mean(), np.nan)
    im = ax.pcolormesh(xe, ye, np.log2(ratio).T, cmap='RdBu_r', vmin=-1, vmax=1, rasterized=True)
    ax.set_title(LABEL[fam[p]], fontsize=7, pad=2); ax.set_aspect('auto')
    ax.tick_params(length=2)
axs[0].set_ylabel('gripper → goal (cm)'); fig2.supxlabel('gripper → cube (cm)', fontsize=7, y=-0.06)
cb = fig2.colorbar(im, ax=axs, fraction=0.025, pad=0.02, ticks=[-1, 0, 1]); cb.ax.set_yticklabels(['½×', '1×', '2×']); cb.ax.tick_params(labelsize=6)
cb.set_label('how often WMPP picks it\nvs. its overall share', fontsize=6)
fig2.savefig(out + '_ratio.png', dpi=250, bbox_inches='tight'); fig2.savefig(out + '_ratio.pdf', bbox_inches='tight', dpi=250)
print('figure ->', out + '_ratio.{pdf,png}')
