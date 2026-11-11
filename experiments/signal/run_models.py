"""Experiments 3-6: feature additions, imbalance handling, calibration and model families.

Every candidate is scored on the SAME folds as the champion (time-ordered expanding origin
and 3x repeated stratified 5-fold) and compared with it through paired differences.
"""

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.signal.common import OUT, SEED, folds_for, get_dev, logistic
from experiments.signal.folds import pooled, score_model
from experiments.signal.stats import paired_bootstrap_diff, paired_fold_verdict
from experiments.signal.transformers import (
    AddInteractions, AddTargetEncoding, ColumnDropper, Resampled, WoeBinner,
)
from src.features.engineering import FeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, v = get_dev()
F = folds_for(y, v)
DERIVED = ["dti", "pti", "monto_sobre_ingreso", "ratio_ingreso_declarado_bureau", "creditos_por_anio_adulto",
           "cuota_supera_salario", "tiene_mora_bureau", "rango_edad"]
FALTA = [c for c in X.columns if c.startswith("falta_")]
CONT = ["plazo_meses", "edad_cliente", "puntaje_datacredito", "pti", "dti", "monto_sobre_ingreso", "capital_prestado",
        "cuota_pactada", "salario_cliente", "saldo_total", "cant_creditosvigentes", "huella_consulta"]
NAMES = list(FeatureEngineer().fit(X, y).feature_names_out_)  # column names only
MONO = {"puntaje_datacredito": -1, "saldo_mora": 1, "dti": 1, "pti": 1}
INTER = [("plazo_meses", "pti"), ("plazo_meses", "puntaje_datacredito"), ("puntaje_datacredito", "dti"),
         ("plazo_meses", "capital_prestado")]


def tree_pipe(model):
    return Pipeline([("features", FeatureEngineer()), ("model", model)])


def lgbm(depth, mono=False, leaves=None, n=200):
    kw = dict(n_estimators=n, learning_rate=0.03, max_depth=depth, num_leaves=leaves or 2 ** depth, min_child_samples=80,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, n_jobs=4, verbose=-1, random_state=SEED)
    if mono:
        kw["monotone_constraints"] = [MONO.get(c, 0) for c in NAMES]
    return tree_pipe(lgb.LGBMClassifier(**kw))


CANDS = {
    # ablations of the existing engineering
    "abl: raw columns only": logistic(pre=[ColumnDropper(DERIVED + FALTA)]),
    "abl: no falta_* flags": logistic(pre=[ColumnDropper(FALTA)]),
    "abl: no ratios/bands": logistic(pre=[ColumnDropper(DERIVED)]),
    "abl: no plazo": logistic(pre=[ColumnDropper(["plazo_meses"])]),
    # principled additions
    "add: WOE plazo": logistic(post=[WoeBinner(["plazo_meses"], 5)]),
    "add: WOE 12 continuous": logistic(post=[WoeBinner(CONT, 5)]),
    "add: WOE only (replace)": logistic(post=[WoeBinner(CONT, 5, keep_original=False)]),
    "add: 4 interactions": logistic(post=[AddInteractions(INTER)]),
    "add: target enc tipo_credito/laboral": logistic(pre=[AddTargetEncoding(["tipo_credito", "tipo_laboral"])]),
    "add: WOE plazo + target enc": logistic(pre=[AddTargetEncoding(["tipo_credito", "tipo_laboral"])], post=[WoeBinner(["plazo_meses"], 5)]),
    # imbalance
    "imb: class_weight balanced": logistic(class_weight="balanced"),
    **{f"imb: {m} ratio {r}": logistic(model=Resampled(LogisticRegression(C=0.0869, max_iter=2000), m, r, seed=SEED))
       for m in ("over", "under", "smote") for r in (0.25, 1.0)},
    # families
    "fam: logistic C=0.01": logistic(C=0.01),
    "fam: logistic C=0.03": logistic(C=0.03),
    "fam: logistic C=0.3": logistic(C=0.3),
    "fam: logistic C=1": logistic(C=1.0),
    "fam: L1 logistic C=0.05": logistic(model=LogisticRegression(penalty="l1", solver="liblinear", C=0.05)),
    "fam: L1 logistic C=0.1": logistic(model=LogisticRegression(penalty="l1", solver="liblinear", C=0.1)),
    "fam: LGBM stumps (additive)": lgbm(1, n=300),
    "fam: LGBM depth 2": lgbm(2),
    "fam: LGBM depth 3": lgbm(3),
    "fam: LGBM depth 2 monotone": lgbm(2, mono=True),
    "fam: XGB depth 2": tree_pipe(xgb.XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.03, min_child_weight=10,
                                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=10, n_jobs=4, random_state=SEED)),
    "fam: random forest (leaf 50)": tree_pipe(RandomForestClassifier(300, min_samples_leaf=50, max_features="sqrt", n_jobs=4, random_state=SEED)),
}


def evaluate(models, tag):
    base = {k: score_model(logistic(), X, y, f) for k, f in F.items()}
    rows = []
    for name, model in models.items():
        t0 = time.time()
        r = {"model": name}
        for k, f in F.items():
            tab, oof = score_model(model, X, y, f)
            bt, bo = base[k]
            n1 = 4 if k == "time" else 5
            yy, s_new = pooled({i: oof[i] for i in list(oof)[:n1]}, y)
            _, s_old = pooled({i: bo[i] for i in list(bo)[:n1]}, y)
            for m, fn in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score)):
                ver = paired_fold_verdict((tab[m] - bt[m]).to_numpy())
                d, lo, hi = paired_bootstrap_diff(yy, s_new, s_old, fn, 500, SEED)
                r |= {f"{k}_{m}": tab[m].mean(), f"{k}_{m}_sd": tab[m].std(ddof=1), f"{k}_{m}_d": ver["mean"],
                      f"{k}_{m}_se": ver["se"], f"{k}_{m}_wins": f"{ver['wins']}/{ver['folds']}", f"{k}_{m}_real": ver["real"],
                      f"{k}_{m}_boot_lo": lo, f"{k}_{m}_boot_hi": hi}
        r["confirmed_pr"] = bool(r["time_pr_auc_real"] and r["strat_pr_auc_real"] and r["time_pr_auc_boot_lo"] > 0)
        r["confirmed_roc"] = bool(r["time_roc_auc_real"] and r["strat_roc_auc_real"] and r["time_roc_auc_boot_lo"] > 0)
        rows.append(r)
        print(f"{name:42s} time ROC {r['time_roc_auc']:.4f} ({r['time_roc_auc_d']:+.4f}, {r['time_roc_auc_wins']}) PR {r['time_pr_auc']:.4f} "
              f"({r['time_pr_auc_d']:+.4f}) | strat ROC {r['strat_roc_auc']:.4f} ({r['strat_roc_auc_d']:+.4f}) PR {r['strat_pr_auc']:.4f} "
              f"({r['strat_pr_auc_d']:+.4f}) [{time.time()-t0:.0f}s]", flush=True)
    out = pd.DataFrame(rows)
    base_row = {"model": "CHAMPION (logistic C=0.0869)"}
    for k in F:
        for m in ("pr_auc", "roc_auc"):
            base_row |= {f"{k}_{m}": base[k][0][m].mean(), f"{k}_{m}_sd": base[k][0][m].std(ddof=1)}
    out = pd.concat([pd.DataFrame([base_row]), out], ignore_index=True)
    out.to_csv(OUT / "results" / f"models_{tag}.csv", index=False)
    return out


if __name__ == "__main__":
    evaluate(CANDS, "all")
    print("comparisons made:", len(CANDS))
