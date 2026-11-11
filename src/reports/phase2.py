"""Phase 2 results: the same TEST window scored by every model, then compared.

    make phase2-report

Pure functions turn a table of predictions (one column per model) into metrics, bootstrap
intervals and markdown. `main` is the only part that touches disk, GCS or the models.
Nothing here selects a model: the TEST window is only reported on.
"""

import argparse
import glob
import json
import logging
import re
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "config" / "reports" / "phase2.yaml"
CI_METRICS = ("pr_auc", "roc_auc", "gini", "f1")


# ---------- metrics ----------

def flagged(scores, share: float) -> np.ndarray:
    """The top `share` of loans by score: the review queue."""
    scores = np.asarray(scores, dtype=float)
    return scores >= np.quantile(scores, 1 - share)


def accuracy(y, scores, share: float) -> float:
    return float((flagged(scores, share) == np.asarray(y).astype(bool)).mean())


def trivial_baseline(y) -> dict[str, float]:
    """Predict that nobody defaults."""
    y = np.asarray(y)
    return {"accuracy": float(1 - y.mean()), "precision": 0.0, "recall": 0.0, "f1": 0.0}


def point_metrics(y, scores, share: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype=float)
    pred = flagged(scores, share)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    roc = float(roc_auc_score(y, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": roc,
        "gini": 2 * roc - 1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / len(y),
        "flagged": tp + fp,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _resamples(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, n)


def bootstrap(y, scores, share: float, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95 % percentile intervals; each resample refits the flag threshold like the real run."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    draws = {m: [] for m in CI_METRICS}
    for idx in _resamples(len(y), n_boot, seed):
        if y[idx].all() or not y[idx].any():
            continue
        point = point_metrics(y[idx], scores[idx], share)
        for m in CI_METRICS:
            draws[m].append(point[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in draws.items()}


def paired_diff(y, a, b, metric: str, n_boot: int, seed: int, share: float = 0.157) -> tuple[float, float]:
    """95 % interval of metric(a) - metric(b) on the same resampled loans."""
    y, a, b = np.asarray(y), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = []
    for idx in _resamples(len(y), n_boot, seed):
        if y[idx].all() or not y[idx].any():
            continue
        diffs.append(point_metrics(y[idx], a[idx], share)[metric] - point_metrics(y[idx], b[idx], share)[metric])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def metrics_table(preds: pd.DataFrame, entries: list[dict], share: float, n_boot: int, seed: int, reference: str):
    """One row per entry; an optional entry with no predictions yet is marked pending."""
    from sklearn.metrics import brier_score_loss

    rows = []
    for e in entries:
        base = {"id": e["id"], "label": e["label"], "platform": e["platform"]}
        if e["id"] not in preds:
            if not e.get("optional"):
                raise KeyError(f"faltan predicciones para {e['id']}")
            rows.append(base | {"status": "pending"})
            continue
        s = preds[e["id"]].to_numpy(dtype=float)
        row = base | {"status": "ok"} | point_metrics(preds["y"], s, share)
        row["brier"] = float(brier_score_loss(preds["y"], s)) if e.get("probabilistic") else None
        for m, ci in bootstrap(preds["y"], s, share, n_boot, seed).items():
            row[f"{m}_ci"] = list(ci)
        if e["id"] != reference:
            lo, hi = paired_diff(preds["y"], s, preds[reference], "pr_auc", n_boot, seed, share)
            row["pr_auc_diff_vs_reference"] = [lo, hi]
            row["distinguishable_from_reference"] = not (lo <= 0 <= hi)
        rows.append(row)
    return rows


def reproducibility(preds: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> dict[str, float]:
    """Largest absolute score difference between two columns that should be the same model."""
    return {n: float(np.abs(preds[a] - preds[b]).max()) for n, a, b in pairs}


def cv_vs_oot(champion: dict) -> list[dict]:
    """Cross-validated PR-AUC against the held-out PR-AUC; a positive gap is optimism."""
    return [
        {**{k: r[k] for k in ("model", "cv_mean", "cv_std", "pr_auc")}, "gap": r["cv_mean"] - r["pr_auc"]}
        for r in champion["comparison"]
        if r.get("cv_mean") == r.get("cv_mean")
    ]


def time_fold_summary(folds: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        m: {"mean": float(folds[m].mean()), "sd": float(folds[m].std()),
            "min": float(folds[m].min()), "max": float(folds[m].max())}
        for m in ("pr_auc", "roc_auc")
    }


# ---------- predictions ----------

def parse_automl(lines, key: str, n_rows: int) -> pd.Series:
    """P(default) per TEST row from the batch jsonl; rows AutoML refused to score stay NaN."""
    out = pd.Series(np.nan, index=range(n_rows), dtype=float)
    for line in lines:
        rec = json.loads(line)
        by_class = dict(zip(rec["prediction"]["classes"], rec["prediction"]["scores"]))
        out[int(rec["instance"][key])] = float(by_class["1"])
    return out


def fill_unscored(scores: pd.Series) -> pd.Series:
    """Unscored rows are never flagged: score 0, as in reports/gcp/automl_metrics.json."""
    return scores.fillna(0.0)


def collect_predictions(frame, label, split, test_value, scorers, extra) -> pd.DataFrame:
    """Label plus one score column per model, over the TEST rows only."""
    test = frame[frame[split] == test_value].reset_index(drop=True)
    X = test.drop(columns=[label, split])
    out = pd.DataFrame({"y": test[label].astype(int)})
    for name, fn in scorers.items():
        out[name] = np.asarray(fn(X), dtype=float)
    for name, series in extra.items():
        out[name] = series.to_numpy(dtype=float)
    return out


# ---------- rendering ----------

PERCENT = {"accuracy"}


def _cell(row: dict, key: str) -> str:
    v = row.get(key)
    if v is None:
        return "n/a"
    text = f"{v * 100:.1f} %" if key in PERCENT else f"{v:.3f}"
    ci = row.get(f"{key}_ci")
    return f"{text} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else text


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| Model | Platform | " + " | ".join(label for _, label in columns) + " |"
    lines = [head, "|" + "---|" * (len(columns) + 2)]
    for r in rows:
        if r["status"] == "pending":
            cells = ["pending (still training)"] + [""] * (len(columns) - 1)
        else:
            cells = [_cell(r, k) for k, _ in columns]
        lines.append(f"| {r['label']} | {r['platform']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(template: str, values: dict[str, str]) -> str:
    def sub(m):
        return values[m.group(1)]

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", sub, template)


# ---------- I/O (not unit tested: needs models, disk and GCS) ----------

def _load_cfg(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def _fetch(cfg: dict) -> Path:
    dest = RAIZ / cfg["fetch_dir"]
    for uri, sub in ((cfg["cloud_models_uri"], "models"), (cfg["automl_output_uri"], "automl")):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "rsync", "-r", uri, str(dest / sub)], check=True)
    return dest


def _model_scorer(path: Path):
    from src.gcp.data import bools_to_float

    model = joblib.load(path)
    return lambda X: model.predict_proba(bools_to_float(X))[:, 1]


def _automl_scores(fetched: Path, pattern: str, key: str, n_rows: int) -> pd.Series | None:
    dirs = sorted(glob.glob(str(fetched / "automl" / pattern)))
    files = sorted(glob.glob(str(Path(dirs[-1]) / "prediction.results-*"))) if dirs else []
    if not files:
        return None
    lines = [ln for f in files for ln in Path(f).read_text().splitlines() if ln.strip()]
    return parse_automl(lines, key, n_rows)


def build_predictions(cfg: dict, fetched: Path) -> pd.DataFrame:
    from src.models.heuristic import score_frame

    frame = pd.read_csv(RAIZ / cfg["split_csv"])
    scorers = {f"local_{n}": _model_scorer(RAIZ / p) for n, p in cfg["local_models"].items()}
    scorers |= {f"cloud_{n}": _model_scorer(fetched / "models" / n / "champion.joblib") for n in cfg["local_models"]}
    scorers["local_heuristic"] = lambda X: score_frame(X).to_numpy(dtype=float)
    n_test = int((frame[cfg["split_column"]] == cfg["test_value"]).sum())
    extra = {}
    for col, pattern in cfg["automl_batch_glob"].items():
        scores = _automl_scores(fetched, pattern, cfg["automl_key_column"], n_test)
        if scores is not None:
            extra[col] = scores
    return collect_predictions(frame, cfg["label_column"], cfg["split_column"], cfg["test_value"], scorers, extra)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    from src.reports import phase2_plots as plots

    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="reuse the predictions cache, no GCS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = _load_cfg()
    out = cfg["outputs"]

    cache = RAIZ / out["predictions_cache"]
    if args.offline and cache.exists():
        preds = pd.read_csv(cache)
    else:
        preds = build_predictions(cfg, _fetch(cfg))
        cache.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(cache, index=False)
    for e in cfg["entries"]:
        if e["platform"] == "automl" and e["id"] in preds:
            preds[e["id"]] = fill_unscored(preds[e["id"]])

    share, entries, platforms = cfg["flagged_share"], cfg["entries"], cfg["platforms"]
    rows = metrics_table(preds, entries, share, cfg["n_boot"], cfg["seed"], cfg["reference_entry"])
    baseline = trivial_baseline(preds["y"])
    repro = reproducibility(preds, [(n, f"local_{n}", f"cloud_{n}") for n in cfg["local_models"]])
    src = cfg["sources"]
    champion = _read_json(RAIZ / src["champion_json"])
    cv = cv_vs_oot(champion)
    folds = pd.read_csv(RAIZ / src["time_folds_csv"])
    fold_summary = time_fold_summary(folds)
    noise = pd.read_csv(RAIZ / src["noise_ci_csv"])
    v1 = _read_json(RAIZ / src["automl_v1_json"])
    v2_json = _read_json(RAIZ / src["automl_v2_json"])
    v2_ready = any(r["id"] == "automl_v2" and r["status"] == "ok" for r in rows)

    tables = {
        "n_test": len(preds), "n_defaults": int(preds["y"].sum()), "flagged_share": share,
        "trivial_baseline": baseline, "rows": rows, "reproducibility_max_abs_diff": repro,
        "cv_vs_oot": cv, "time_folds": fold_summary,
        "time_folds_pooled_ci": noise[noise.protocol == "time"].to_dict("records"),
        "automl_v1_notes": {k: v1[k] for k in ("scored_rows", "coverage", "scored_subset_only")} if v1 else None,
        "automl_v2": {"batch_scored": v2_ready, "metrics_json_present": v2_json is not None},
    }
    tj = RAIZ / out["tables_json"]
    tj.write_text(json.dumps(tables, indent=2))

    pdir = RAIZ / out["plots_dir"]
    pdir.mkdir(parents=True, exist_ok=True)
    plots.metric_bars(rows, platforms, pdir / "metric_bars.png")
    plots.pr_curves(preds, entries, platforms, pdir / "pr_curves.png")
    plots.accuracy_vs_baseline(rows, baseline["accuracy"], platforms, pdir / "accuracy_vs_baseline.png")
    boot = {r["id"]: r["pr_auc_ci"] for r in rows if r["status"] == "ok" and r["platform"] == "local"}
    plots.cv_vs_oot(cv, {m: boot[f"local_{m}"] for m in (c["model"] for c in cv)}, pdir / "cv_vs_oot.png")

    for r in rows:
        r["platform"] = platforms[r["platform"]]
    template = (RAIZ / out["results_template"]).read_text()
    text = render(template, _template_values(tables, rows, cfg))
    (RAIZ / out["results_md"]).write_text(text)
    _log.info("escrito %s, %s y %d figuras", out["results_md"], out["tables_json"], 4)


def _template_values(t: dict, rows: list[dict], cfg: dict) -> dict[str, str]:
    cols = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("gini", "Gini"), ("f1", "F1")]
    ok = [r for r in rows if r["status"] == "ok"]
    acc_cols = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("brier", "Brier")]
    paired = ["| Model | Platform | PR-AUC change vs local logistic (95 % interval) | Distinguishable? |", "|---|---|---|---|"]
    for r in ok:
        if "pr_auc_diff_vs_reference" in r:
            lo, hi = r["pr_auc_diff_vs_reference"]
            paired.append(f"| {r['label']} | {r['platform']} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if r['distinguishable_from_reference'] else 'no'} |")
    cvt = ["| Model | CV PR-AUC (mean +/- sd) | TEST PR-AUC | Gap |", "|---|---|---|---|"]
    cvt += [f"| {c['model']} | {c['cv_mean']:.3f} +/- {c['cv_std']:.3f} | {c['pr_auc']:.3f} | {c['gap']:+.3f} |" for c in t["cv_vs_oot"]]
    tf = t["time_folds"]
    tft = ["| Metric | Mean | Std. dev. | Min | Max |", "|---|---|---|---|---|"]
    tft += [f"| {m.upper().replace('_', '-')} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.3f} | {s['max']:.3f} |" for m, s in tf.items()]
    rep = ["| Model | Largest score difference, local vs cloud |", "|---|---|"]
    rep += [f"| {n} | {d:.1e} |" for n, d in t["reproducibility_max_abs_diff"].items()]
    base = t["trivial_baseline"]
    v2 = "included below" if t["automl_v2"]["batch_scored"] else "**pending**: AutoML v2 is still training, its row is blank until `make phase2-report` is re-run"
    return {
        "n_test": f"{t['n_test']:,}", "n_defaults": str(t["n_defaults"]),
        "baseline_accuracy": f"{base['accuracy'] * 100:.1f} %",
        "headline_table": markdown_table(rows, cols),
        "accuracy_table": markdown_table(rows, acc_cols),
        "paired_table": "\n".join(paired), "cv_table": "\n".join(cvt),
        "timefold_table": "\n".join(tft), "reproducibility_table": "\n".join(rep),
        "v2_status": v2,
    }


if __name__ == "__main__":
    main()
