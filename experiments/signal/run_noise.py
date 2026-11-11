"""Experiment 1: how much of the champion's score is sampling noise, and does more data help."""

import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.signal.common import OUT, SEED, folds_for, get_dev, logistic
from experiments.signal.folds import pooled, score_model
from experiments.signal.stats import bootstrap_ci, paired_bootstrap_diff

X, y, v = get_dev()
F = folds_for(y, v)
res = {}
rows = []
for name, folds in F.items():
    tab, oof = score_model(logistic(), X, y, folds)
    n_first = 4 if name == "time" else 5  # pool one repeat only so rows are not counted twice
    yy, ss = pooled({k: oof[k] for k in list(oof)[:n_first]}, y)
    for mname, m in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)):
        p, lo, hi = bootstrap_ci(yy, ss, m, 1000, SEED)
        rows.append({"protocol": name, "metric": mname, "pooled": p, "ci_lo": lo, "ci_hi": hi,
                     "fold_mean": tab[mname].mean(), "fold_sd": tab[mname].std(ddof=1),
                     "fold_min": tab[mname].min(), "fold_max": tab[mname].max(), "n_folds": len(tab),
                     "n_pos_pooled": int(yy.sum()), "prevalence": float(yy.mean())})
    tab.to_csv(OUT / "results" / f"noise_folds_{name}.csv", index=False)
    if name == "time":
        # validation positives per month, to show how thin each fold is
        res["time_fold_positives"] = [int(y.iloc[va].sum()) for _, va in folds]
        res["time_fold_sizes"] = [len(va) for _, va in folds]
pd.DataFrame(rows).to_csv(OUT / "results" / "noise_ci.csv", index=False)

# near-identical models: the paired difference between them is pure noise, which sets the detectable gain
_, o_a = score_model(logistic(), X, y, F["time"])
_, o_b = score_model(logistic(C=0.03), X, y, F["time"])
_, o_c = score_model(logistic(C=0.3), X, y, F["time"])
yy, sa = pooled(o_a, y)
_, sb = pooled(o_b, y)
_, sc = pooled(o_c, y)
nd = []
for lab, s2 in (("C=0.03 vs 0.087", sb), ("C=0.3 vs 0.087", sc)):
    for mname, m in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)):
        d, lo, hi = paired_bootstrap_diff(yy, s2, sa, m, 1000, SEED)
        nd.append({"comparison": lab, "metric": mname, "diff": d, "ci_lo": lo, "ci_hi": hi, "halfwidth": (hi - lo) / 2})
pd.DataFrame(nd).to_csv(OUT / "results" / "noise_nearidentical_diff.csv", index=False)

# learning curve: subsample the training part of each fold, score on the untouched validation part
fracs = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
lc = []
for proto in ("time", "strat"):
    for k, (tr, va) in enumerate(F[proto]):
        for f in fracs:
            for rep in range(3 if f < 1 else 1):
                rng = np.random.default_rng(SEED + 100 * rep + k)
                sub = np.sort(rng.choice(tr, int(len(tr) * f), replace=False))
                if y.iloc[sub].sum() < 5:
                    continue
                m = clone(logistic()).fit(X.iloc[sub], y.iloc[sub])
                s = m.predict_proba(X.iloc[va])[:, 1]
                lc.append({"protocol": proto, "fold": k, "frac": f, "n_train": len(sub), "n_pos": int(y.iloc[sub].sum()),
                           "pr_auc": average_precision_score(y.iloc[va], s), "roc_auc": roc_auc_score(y.iloc[va], s)})
lc = pd.DataFrame(lc)
lc.to_csv(OUT / "results" / "learning_curve.csv", index=False)
json.dump(res, open(OUT / "results" / "noise_meta.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, mname in zip(ax, ("roc_auc", "pr_auc")):
    for proto, col in (("strat", "#1f77b4"), ("time", "#d95f02")):
        g = lc[lc.protocol == proto].groupby("frac")[mname].agg(["mean", "std"])
        n = lc[lc.protocol == proto].groupby("frac")["n_train"].mean()
        a.errorbar(n, g["mean"], yerr=g["std"], marker="o", capsize=3, color=col,
                   label={"strat": "stratified 5-fold", "time": "time-ordered"}[proto])
    a.set_xlabel("training loans"); a.set_ylabel(mname.replace("_", "-").upper()); a.grid(alpha=.3)
ax[0].legend(); fig.suptitle("Learning curve of the champion (bars = spread across folds/subsamples)")
fig.tight_layout(); fig.savefig(OUT / "plots" / "learning_curve.png", dpi=130)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
tab = pd.read_csv(OUT / "results" / "noise_folds_time.csv")
ax.bar(range(1, 5), tab.roc_auc, color="#d95f02"); ax.axhline(0.5, color="k", ls=":")
ax.set_xticks(range(1, 5)); ax.set_xticklabels(["2025-02", "2025-03", "2025-04", "2025-05"])
ax.set_ylim(0.4, 0.8); ax.set_ylabel("ROC-AUC"); ax.set_title("Time-ordered folds: one month each")
for i, (n, p) in enumerate(zip(res["time_fold_sizes"], res["time_fold_positives"])):
    ax.text(i + 1, 0.42, f"{p} defaults\n of {n}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "plots" / "time_folds_roc.png", dpi=130)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(nd).round(4).to_string())
print(lc.groupby(["protocol", "frac"])[["n_train", "n_pos", "pr_auc", "roc_auc"]].agg(["mean", "std"]).round(3).to_string())
