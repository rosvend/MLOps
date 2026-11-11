import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.signal.stats import bootstrap_ci, paired_bootstrap_diff, paired_fold_verdict


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    return y, y * 0.8 + rng.normal(size=n)


def test_ci_contains_point_and_is_ordered():
    y, s = _data()
    point, lo, hi = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=1)
    assert lo < point < hi


def test_paired_diff_of_identical_scores_is_zero():
    y, s = _data()
    d, lo, hi = paired_bootstrap_diff(y, s, s, roc_auc_score, n_boot=100, seed=1)
    assert d == lo == hi == 0


def test_verdict_needs_gain_above_spread_in_every_fold():
    assert paired_fold_verdict([0.02, 0.03, 0.025, 0.03])["real"]
    assert not paired_fold_verdict([0.02, -0.01, 0.03, 0.0])["real"]
    assert not paired_fold_verdict([0.001, 0.05, -0.04, 0.03])["real"]
