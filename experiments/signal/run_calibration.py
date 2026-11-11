"""Experiment 5: reliability of the champion's probabilities, Platt and isotonic fitted in-fold."""

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.signal.common import OUT, folds_for, get_dev, logistic
from experiments.signal.folds import pooled, score_model
from experiments.signal.stats import bootstrap_ci

SHARE = 0.157  # flagged share from config/training/default.yaml
X, y, v = get_dev()
F = folds_for(y, v)
variants = {"champion (raw)": logistic(),
            "Platt (sigmoid)": CalibratedClassifierCV(logistic(), method="sigmoid", cv=5),
            "isotonic": CalibratedClassifierCV(logistic(), method="isotonic", cv=5)}


def ece(yt, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"y": yt, "p": p, "q": q}).groupby("q", observed=True).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum())


rows, pool = [], {}
for proto in ("time", "strat"):
    n1 = 4 if proto == "time" else 5
    for name, m in variants.items():
        tab, oof = score_model(m, X, y, F[proto])
        yy, pp = pooled({k: oof[k] for k in list(oof)[:n1]}, y)
        pool[(proto, name)] = (yy, pp)
        per_fold_brier = [brier_score_loss(y.iloc[va], s) for va, s in oof.values()]
        cuts = [np.quantile(s, 1 - SHARE) for _, s in oof.values()]
        flagged_rate = np.mean([y.iloc[va][s >= np.quantile(s, 1 - SHARE)].mean() for va, s in oof.values()])
        b, blo, bhi = bootstrap_ci(yy, pp, brier_score_loss, 500, 0)
        rows.append({"protocol": proto, "model": name, "brier": b, "brier_lo": blo, "brier_hi": bhi,
                     "brier_fold_sd": np.std(per_fold_brier, ddof=1), "logloss": log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6)),
                     "ece": ece(yy, pp), "mean_pred": pp.mean(), "observed": yy.mean(),
                     "roc_auc": roc_auc_score(yy, pp), "pr_auc": average_precision_score(yy, pp),
                     "prob_cut_at_15.7%": float(np.mean(cuts)), "flagged_default_rate": flagged_rate})
prev = y.mean()
rows.append({"protocol": "-", "model": "constant prevalence", "brier": brier_score_loss(y, np.full(len(y), prev))})
res = pd.DataFrame(rows); res.to_csv(OUT / "results" / "calibration.csv", index=False)
print(res.round(4).to_string())

# flagged-set overlap between raw and calibrated scores (time protocol, pooled)
yy, raw = pool[("time", "champion (raw)")]
ov = []
for name in ("Platt (sigmoid)", "isotonic"):
    _, c = pool[("time", name)]
    a = raw >= np.quantile(raw, 1 - SHARE); b = c >= np.quantile(c, 1 - SHARE)
    ov.append({"model": name, "flagged_overlap_with_raw": float((a & b).sum() / a.sum()), "ties_share_at_cut": float((c == np.quantile(c, 1 - SHARE)).mean())})
pd.DataFrame(ov).to_csv(OUT / "results" / "calibration_overlap.csv", index=False); print(pd.DataFrame(ov))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, col in zip(variants, ("#d95f02", "#1f77b4", "#2ca02c")):
    yy, pp = pool[("time", name)]
    fr, mp = calibration_curve(yy, pp, n_bins=8, strategy="quantile")
    ax[0].plot(mp, fr, "o-", color=col, label=name)
ax[0].plot([0, .2], [0, .2], "k:", label="perfect"); ax[0].set_xlabel("mean predicted probability"); ax[0].set_ylabel("observed default rate")
ax[0].set_title("Reliability, time-ordered validation (8 quantile bins)"); ax[0].legend(); ax[0].grid(alpha=.3)
yy, pp = pool[("time", "champion (raw)")]
ax[1].hist(pp, bins=40, color="#d95f02"); ax[1].axvline(np.quantile(pp, 1 - SHARE), color="k", ls="--", label="15.7% review cut")
ax[1].set_xlabel("predicted default probability"); ax[1].set_title("Champion score distribution"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "plots" / "calibration.png", dpi=130)
