"""Experiments 2 and 3: label maturity by vintage and term, then a univariate signal audit."""

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.signal.common import OUT, SEED, get_dev
from experiments.signal.stats import bootstrap_ci
from src.features.engineering import FeatureEngineer

X, y, v = get_dev()
R = OUT / "results"
d = X.assign(y=y, mes=v)

# ---- maturity (dev months only)
d["plazo_g"] = pd.cut(d.plazo_meses, [0, 6, 10, 12, 24, 100], labels=["<=6", "7-10", "11-12", "13-24", ">24"])
by_m = d.groupby("mes").agg(loans=("y", "size"), defaults=("y", "sum"), rate=("y", "mean"), plazo_medio=("plazo_meses", "mean"),
                            share_largo=("plazo_meses", lambda s: (s > 12).mean()))
by_m.to_csv(R / "maturity_by_month.csv")
pm = d.pivot_table(index="mes", columns="plazo_g", values="y", aggfunc="mean", observed=True)
pn = d.pivot_table(index="mes", columns="plazo_g", values="y", aggfunc="size", observed=True)
pm.to_csv(R / "maturity_rate_month_term.csv"); pn.to_csv(R / "maturity_n_month_term.csv")
by_p = d.groupby("plazo_g", observed=True).y.agg(["size", "sum", "mean"])
by_p.to_csv(R / "maturity_by_term.csv")
# calendar arithmetic only: months between origination and the end of the book (April 2026, per docs)
end = pd.Period("2026-04", "M")
age = d.mes.map(lambda m: (end - pd.Period(m, "M")).n)
d["age"] = age
frac = (d.groupby("mes").apply(lambda g: (g.age >= g.plazo_meses).mean(), include_groups=False)).rename("share_term_elapsed")
frac.to_csv(R / "maturity_term_elapsed.csv")
# month trend in the default rate, short mature terms only vs everything, via logistic slope with bootstrap CI
from sklearn.linear_model import LogisticRegression


def slope(g):
    t = pd.PeriodIndex(g.mes, freq="M").astype(int) if False else g.mes.map(lambda m: (pd.Period(m, "M") - pd.Period("2024-11", "M")).n)
    return LogisticRegression(C=1e6, max_iter=500).fit(t.to_numpy().reshape(-1, 1), g.y).coef_[0, 0]


rng = np.random.default_rng(SEED)
tr = []
for lab, g in (("all terms", d), ("term <= 12", d[d.plazo_meses <= 12]), ("term <= 6", d[d.plazo_meses <= 6])):
    b = [slope(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(300)]
    tr.append({"subset": lab, "n": len(g), "slope_per_month": slope(g), "ci_lo": np.quantile(b, .025), "ci_hi": np.quantile(b, .975)})
pd.DataFrame(tr).to_csv(R / "maturity_trend.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
se = np.sqrt(by_m.rate * (1 - by_m.rate) / by_m.loans)
ax[0].errorbar(range(len(by_m)), by_m.rate * 100, yerr=1.96 * se * 100, marker="o", capsize=3, color="#d95f02")
ax[0].set_xticks(range(len(by_m))); ax[0].set_xticklabels(by_m.index, rotation=45); ax[0].set_ylabel("default rate %")
ax[0].set_title("Default rate by origination month (95% CI)"); ax[0].grid(alpha=.3)
bp = by_p.copy(); se = np.sqrt(bp["mean"] * (1 - bp["mean"]) / bp["size"])
ax[1].bar(bp.index.astype(str), bp["mean"] * 100, yerr=1.96 * se * 100, capsize=3, color="#1f77b4")
for i, n in enumerate(bp["size"]): ax[1].text(i, 1, f"n={n}", ha="center", fontsize=8, color="w")
ax[1].set_title("Default rate by loan term (months)"); ax[1].set_ylabel("default rate %")
fig.tight_layout(); fig.savefig(OUT / "plots" / "maturity_month_term.png", dpi=130)
print(by_m.round(3)); print(pm.round(3)); print(by_p); print(frac.round(2)); print(pd.DataFrame(tr).round(4))

# ---- signal audit
Xa = X.copy()
rows = []
for c in Xa.columns:
    s = Xa[c]
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        for lvl, g in y.groupby(s.astype(str)):
            rows.append({"feature": f"{c}={lvl}", "kind": "cat", "auc": np.nan, "n": len(g), "rate": g.mean(), "missing": 0.0})
        continue
    num = pd.to_numeric(s.astype("float64"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = num.fillna(num.median())
    if filled.nunique() < 2:
        continue
    p, lo, hi = bootstrap_ci(y.to_numpy(), filled.to_numpy(), roc_auc_score, 300, SEED)
    per_month = [roc_auc_score(y[v == m], filled[v == m]) for m in sorted(v.unique()) if y[v == m].nunique() > 1]
    rows.append({"feature": c, "kind": "num", "auc": p, "ci_lo": lo, "ci_hi": hi, "dir": "+" if p > .5 else "-",
                 "months_same_sign": int(sum((a > .5) == (p > .5) for a in per_month)), "months": len(per_month),
                 "missing": float(num.isna().mean())})
aud = pd.DataFrame(rows)
aud["strength"] = (aud.auc - .5).abs()
aud.sort_values("strength", ascending=False).to_csv(R / "signal_univariate.csv", index=False)
print(aud.sort_values("strength", ascending=False).head(25).round(3).to_string())

# missingness: lift of default rate when absent, with a bootstrap interval
ml = []
for c in [c for c in X.columns if c.startswith("falta_")]:
    m = X[c].astype(bool).to_numpy()
    if m.sum() < 5: continue
    lift = y[m].mean() / y[~m].mean()
    b = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y)); mm, yy = m[i], y.to_numpy()[i]
        if mm.sum() > 0: b.append(yy[mm].mean() / yy[~mm].mean())
    ml.append({"flag": c, "missing_share": m.mean(), "n_missing": int(m.sum()), "rate_missing": y[m].mean(), "rate_present": y[~m].mean(),
               "lift": lift, "ci_lo": np.quantile(b, .025), "ci_hi": np.quantile(b, .975)})
ml = pd.DataFrame(ml); ml.to_csv(R / "signal_missingness.csv", index=False); print(ml.round(3).to_string())

# does winsor/log/impute change a feature's univariate AUC? (descriptive; fitted on all development rows)
fe = FeatureEngineer().fit(X, y); Z = fe.transform(X)
cmp = []
for c in Z.columns:
    if c in aud.feature.values and Z[c].nunique() > 1:
        raw = aud.set_index("feature").loc[c, "auc"]
        cmp.append({"feature": c, "auc_raw": raw, "auc_engineered": roc_auc_score(y, Z[c]), "log_or_cap": c in fe.log_scale_ or c in fe.heavy_tailed_})
cmp = pd.DataFrame(cmp); cmp["delta"] = cmp.auc_engineered - cmp.auc_raw
cmp.sort_values("delta", key=abs, ascending=False).to_csv(R / "signal_engineering_effect.csv", index=False)
print(cmp.sort_values("delta", key=abs, ascending=False).head(8).round(4).to_string())

top = aud[aud.kind == "num"].sort_values("strength", ascending=False).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(top.auc, range(len(top)), xerr=[top.auc - top.ci_lo, top.ci_hi - top.auc], fmt="o", capsize=3, color="#1f77b4")
ax.axvline(.5, color="k", ls=":"); ax.set_yticks(range(len(top))); ax.set_yticklabels(top.feature)
ax.set_xlabel("univariate ROC-AUC (95% bootstrap CI)"); ax.set_title("Strongest single features, development months")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "plots" / "signal_univariate.png", dpi=130)
