"""Constants and statistics helpers shared by the paper generator scripts.

No absl flags are defined here, so every generator (report_og50, make_paper_assets,
report_ablations, report_diagnostics) can import it without duplicate-flag errors.
"""
import csv
import os

import numpy as np

NAME = {'hiql': 'HIQL', 'gciql': 'GCIQL', 'gcivl': 'GCIVL', 'crl': 'CRL', 'gcbc': 'GCBC', 'qrl': 'QRL'}
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
FAMILIES = [
    ('Maze', ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0']),
    ('Cube', ['cube-single-play-v0', 'cube-single-noisy-v0', 'cube-double-play-v0', 'cube-double-noisy-v0',
              'cube-triple-play-v0', 'cube-triple-noisy-v0', 'cube-quadruple-play-v0', 'cube-quadruple-noisy-v0']),
    ('Scene', ['scene-play-v0', 'scene-noisy-v0']),
    ('Puzzle', ['puzzle-3x3-play-v0', 'puzzle-3x3-noisy-v0', 'puzzle-4x4-play-v0', 'puzzle-4x4-noisy-v0',
                'puzzle-4x5-play-v0', 'puzzle-4x5-noisy-v0', 'puzzle-4x6-play-v0', 'puzzle-4x6-noisy-v0']),
]
ENV_FAMILY = {e: fam for fam, fenvs in FAMILIES for e in fenvs}
KS = [1, 5, 10, 25, 50, 100]


def short(env):
    return env.replace('-v0', '')


def tex_env(env):
    return '\\texttt{' + short(env) + '}'


def macro_key(env):
    """Env name -> LaTeX-safe macro suffix; identical to make_paper_assets historically."""
    key = ''.join(w.capitalize() for w in short(env).replace('x', 'by').split('-'))
    for dgt, word in (('3', 'Three'), ('4', 'Four'), ('5', 'Five'), ('6', 'Six')):
        key = key.replace(dgt, word)
    return key


def family(name):
    return name.rsplit('-sd', 1)[0] if '-sd' in name else name


def fmt(x, bold=False, nd=0):
    """Success rates are reported as integers (percent)."""
    s = f'{x:.{nd}f}'
    return f'\\textbf{{{s}}}' if bold else s


def fmt_ci(c, star=True):
    s = f"{100 * c['delta']:+.0f} [{100 * c['ci_lo']:+.0f}, {100 * c['ci_hi']:+.0f}]"
    return s + ('*' if star and c['significant'] else '')


def fmt_pm_ci(mean, half, bold=False):
    """'mean +- half-width of the 95% interval', integers."""
    s = f'{mean:.0f} \\pm {half:.0f}'
    return f'$\\mathbf{{{s}}}$' if bold else f'${s}$'


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rows_by_policy(env_dir, seed, tag, label=None):
    """{policy: [episode rows]} of <env_dir>/bank_sd<seed>_<tag>/episodes.csv
    (or <label>_<tag>/ when label is given); {} when the run does not exist yet."""
    d = os.path.join(env_dir, f'{label}_{tag}' if label else f'bank_sd{seed}_{tag}')
    p = os.path.join(d, 'episodes.csv')
    if not os.path.exists(p):
        return {}
    out = {}
    for r in read_rows(p):
        out.setdefault(r['policy'], []).append(r)
    return out


def paired_rows(rows_a, rows_b):
    """Per-episode success difference a-b for two row lists keyed by (task, episode).
    Asserts that both lists cover the same episodes with the same reset seed."""
    key = lambda r: (r['task_id'], r['episode_idx'])
    base = {key(r): r for r in rows_b}
    assert set(base) == {key(r) for r in rows_a}, 'episode sets differ'
    out = []
    for r in rows_a:
        b = base[key(r)]
        assert r['reset_seed'] == b['reset_seed'], f'reset seed mismatch at {key(r)}'
        out.append(float(r['success']) - float(b['success']))
    return np.array(out)


def hier_boot(per_seed, n_boot, rng, offset=0.0):
    """Hierarchical bootstrap of the mean of {seed: array}; `offset` shifts
    the point estimate and CI (used for delta-vs-reported-baseline)."""
    seeds = list(per_seed)
    point = float(np.mean([per_seed[s].mean() for s in seeds])) - offset
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(seeds), len(seeds), replace=True)
        means = [per_seed[seeds[i]][rng.integers(0, len(per_seed[seeds[i]]), len(per_seed[seeds[i]]))].mean() for i in picked]
        boot[b] = np.mean(means) - offset
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(delta=point, ci_lo=lo, ci_hi=hi,
                per_seed={str(s): float(per_seed[s].mean() - offset) for s in seeds},
                significant=bool(lo > 0 or hi < 0), p_value=boot_p_value(boot))


def near_top(v, top, frac=0.95):
    """OGBench convention: bold every entry at or above `frac` of the row maximum."""
    return top > 0 and v >= frac * top - 1e-9


def half_width(per_seed, n_boot=2000):
    """Half-width (in points) of the 95% hierarchical-bootstrap interval of a success mean,
    {seed: array of per-episode successes}; a fresh generator per call so that the same
    data gives the same +- in every table."""
    c = hier_boot(per_seed, n_boot, np.random.default_rng(1))
    return 100 * (c['ci_hi'] - c['ci_lo']) / 2


def rows_hw(rows_by_seed, n_boot=2000):
    """half_width of {seed: list of episode rows}."""
    return half_width({s: np.array([float(r['success']) for r in v]) for s, v in rows_by_seed.items()}, n_boot)


def boot_p_value(boot):
    """Two-sided bootstrap p-value of 'delta = 0' (floored at 1/n_boot), consistent with
    the percentile-CI significance rule used throughout."""
    boot = np.asarray(boot)
    return float(max(2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0)), 1.0 / len(boot)))


def holm(pvals):
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * p[i] / (rank + 1))
        adj[i] = min(1.0, running)
    return adj


def hier_boot3(per_cell, n_boot, rng):
    """Crossed three-level bootstrap of the mean of {(wm_seed, bank_seed): array}:
    resample WM seeds and bank seeds independently (with replacement), then episodes
    within every selected cell. Point estimate = mean of the cell means."""
    ws = sorted({w for w, _ in per_cell})
    ss = sorted({s for _, s in per_cell})
    cells = {c: np.asarray(v, dtype=float) for c, v in per_cell.items()}
    point = float(np.mean([cells[c].mean() for c in cells]))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pw = rng.choice(len(ws), len(ws), replace=True)
        ps = rng.choice(len(ss), len(ss), replace=True)
        means = []
        for i in pw:
            for j in ps:
                arr = cells[(ws[i], ss[j])]
                means.append(arr[rng.integers(0, len(arr), len(arr))].mean())
        boot[b] = np.mean(means)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(delta=point, ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0),
                p_value=boot_p_value(boot), n_wm_seeds=len(ws), n_bank_seeds=len(ss),
                wm_means={str(w): float(np.mean([cells[(w, s)].mean() for s in ss])) for w in ws},
                bank_means={str(s): float(np.mean([cells[(w, s)].mean() for w in ws])) for s in ss})


def seed_mean(rows_by_seed):
    return float(np.mean([np.mean([float(r['success']) for r in v]) for v in rows_by_seed.values()]))


def contrast(rows_a_by_seed, rows_b_by_seed, n_boot=10000, rng=None):
    """Paired per-episode contrast a-b, hierarchical over the shared seed keys."""
    rng = np.random.default_rng(0) if rng is None else rng
    per_seed = {s: paired_rows(rows_a_by_seed[s], rows_b_by_seed[s]) for s in rows_a_by_seed}
    return hier_boot(per_seed, n_boot, rng)


def wm_val_rank_corr(wm_dir, step=None):
    """validation/value_rank_corr from <wm_dir>/train.csv at the checkpoint step
    (last row at or before `step`; final row when step is None). None if unavailable."""
    p = os.path.join(wm_dir, 'train.csv')
    if not os.path.exists(p):
        return None
    rows = read_rows(p)
    if not rows:
        return None
    col = next((c for c in rows[0] if 'rank_corr' in c), None)
    if col is None:
        return None
    if step is not None and 'step' in rows[0]:
        cand = [r for r in rows if r.get('step') and float(r['step']) <= step + 1e-9]
        rows = cand or rows
    for r in reversed(rows):
        v = r.get(col, '')
        if v not in ('', 'nan', None):
            return float(v)
    return None
