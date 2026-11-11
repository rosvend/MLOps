"""Noise tools: bootstrap intervals and the rule for calling a paired difference real."""

import numpy as np


def _resample(y, rng):
    idx = rng.integers(0, len(y), len(y))
    return idx if 0 < y[idx].sum() < len(y) else _resample(y, rng)


def bootstrap_ci(y, score, metric, n_boot=1000, seed=0, level=0.95):
    """Point estimate and percentile interval, resampling loans."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = _resample(y, rng)
        vals.append(metric(y[i], score[i]))
    a = (1 - level) / 2
    return float(metric(y, score)), *np.quantile(vals, [a, 1 - a]).astype(float)


def paired_bootstrap_diff(y, s_new, s_old, metric, n_boot=1000, seed=0, level=0.95):
    """Metric(new) - metric(old) on the same resampled loans, so shared noise cancels."""
    y, s_new, s_old = map(np.asarray, (y, s_new, s_old))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = _resample(y, rng)
        d.append(metric(y[i], s_new[i]) - metric(y[i], s_old[i]))
    a = (1 - level) / 2
    lo, hi = np.quantile(d, [a, 1 - a])
    return float(metric(y, s_new) - metric(y, s_old)), float(lo), float(hi)


def paired_fold_verdict(diffs, k=2.0):
    """Real only if the mean gain beats k standard errors of the folds and no fold is negative."""
    d = np.asarray(diffs, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "se": float(se),
        "wins": int((d > 0).sum()),
        "folds": len(d),
        "real": bool(d.mean() > k * se and (d > 0).all()),
    }
