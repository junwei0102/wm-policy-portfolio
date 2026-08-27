"""Ranking-competence diagnostics: WM-imagined scores vs real branch outcomes.

All functions are pure numpy over (S states x P policies) matrices, so they are
unit-testable on synthetic fixtures with known ground truth.
"""

import numpy as np


def _cluster_bootstrap(stat_fn, n_states, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_states, n_states)
        vals.append(stat_fn(idx))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pairwise_ranking_accuracy(scores, outcomes, n_boot=10000, seed=0):
    """scores, outcomes: (S, P). Accuracy over discordant pairs; ties = 0.5.

    Bootstrap resamples STATES (pairs within a state are correlated).
    """
    S, P = scores.shape
    iu, ju = np.triu_indices(P, k=1)
    y_diff = outcomes[:, iu] - outcomes[:, ju]  # (S, n_pairs)
    m_diff = scores[:, iu] - scores[:, ju]
    discordant = y_diff != 0
    correct = np.where(m_diff * y_diff > 0, 1.0, np.where(m_diff == 0, 0.5, 0.0))

    def acc(state_idx):
        d = discordant[state_idx]
        if d.sum() == 0:
            return np.nan
        return correct[state_idx][d].mean()

    accuracy = acc(np.arange(S))
    ci_lo, ci_hi = _cluster_bootstrap(acc, S, n_boot, seed)
    return dict(
        accuracy=float(accuracy),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_discordant_pairs=int(discordant.sum()),
        n_states=int(S),
    )


def top1_winner_rate(scores, outcomes, n_boot=10000, seed=0):
    """States where some but not all policies succeed: does argmax score succeed?

    Chance baseline: expected hit rate of a uniformly random pick, per state.
    CI is on (rate - chance).
    """
    S, P = scores.shape
    informative = (outcomes.sum(axis=1) > 0) & (outcomes.sum(axis=1) < P)
    hits = outcomes[np.arange(S), scores.argmax(axis=1)]
    chance_s = outcomes.mean(axis=1)

    def delta(state_idx):
        m = informative[state_idx]
        if m.sum() == 0:
            return np.nan
        return hits[state_idx][m].mean() - chance_s[state_idx][m].mean()

    m = informative
    rate = float(hits[m].mean()) if m.sum() else np.nan
    chance = float(chance_s[m].mean()) if m.sum() else np.nan
    ci_lo, ci_hi = _cluster_bootstrap(delta, S, n_boot, seed)
    return dict(
        rate=rate,
        chance=chance,
        delta=rate - chance if m.sum() else np.nan,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_informative_states=int(m.sum()),
    )


def calibration(probs, labels, n_bins=10):
    """Reliability over equal-width bins; ECE, Brier."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    bin_conf, bin_acc, bin_frac = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_frac.append(0.0)
            continue
        conf, acc, frac = probs[m].mean(), labels[m].mean(), m.mean()
        bin_conf.append(float(conf))
        bin_acc.append(float(acc))
        bin_frac.append(float(frac))
        ece += frac * abs(conf - acc)
    return dict(
        ece=float(ece),
        brier=float(((probs - labels) ** 2).mean()),
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_fraction=bin_frac,
    )


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan

    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return sums[inv] / counts[inv]

    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def horizon_of_validity(per_horizon_ranking, threshold=0.5):
    """Largest horizon whose ranking-accuracy CI lower bound clears threshold."""
    valid = [int(h) for h, r in per_horizon_ranking.items() if r['ci_lo'] > threshold]
    return max(valid) if valid else None


def gate_verdict(per_horizon_ranking, top1, max_gate_horizon):
    """GO iff ranking CI-lower > 0.5 at some H <= max_gate_horizon AND top-1
    beats chance with CI clear of 0."""
    ranking_ok = any(
        r['ci_lo'] > 0.5 for h, r in per_horizon_ranking.items() if int(h) <= max_gate_horizon
    )
    top1_ok = np.isfinite(top1['ci_lo']) and top1['ci_lo'] > 0
    return dict(
        go=bool(ranking_ok and top1_ok),
        ranking_ok=bool(ranking_ok),
        top1_ok=bool(top1_ok),
        max_gate_horizon=int(max_gate_horizon),
    )
