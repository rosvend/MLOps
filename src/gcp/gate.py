"""Promotion gate: score a candidate and the champion on the same rows and decide promote or keep.

    uv run python -m src.gcp.gate RUN_DIR NAME [--evaluation-uri URI] [--out decision.json]

Exits non-zero only on errors; "keep" is a normal answer. Read docs/gcp/training-workflow.md
for the limitation of the default evaluation rows.
"""

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.gcp.data import bools_to_float
from src.gcp.gate_spec import Bootstrap, GateSpec, default_gate_spec
from src.gcp.spec import default_dataset_spec
from src.gcp.storage import read_bytes, read_csv
from src.models.training_spec import GcpDatasetSpec

MODEL_FILE, META_FILE = "champion.joblib", "champion.meta.json"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "reports" / "gcp" / "gate_decision.json"


@dataclass(frozen=True)
class Bundle:
    model: object
    meta: dict


def load_bundle(model_dir: str, fetch) -> Bundle:
    """fetch(uri) -> bytes, so tests need no network."""
    base = model_dir.rstrip("/")
    return Bundle(joblib.load(io.BytesIO(fetch(f"{base}/{MODEL_FILE}"))), json.loads(fetch(f"{base}/{META_FILE}")))


def evaluation_rows(frame: pd.DataFrame, ds: GcpDatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    """TEST rows when the file has the split column, otherwise every row."""
    if ds.split_column in frame:
        frame = frame[frame[ds.split_column] == ds.test_value]
    X = bools_to_float(frame.drop(columns=[ds.label_column, ds.split_column], errors="ignore"))
    return X.reset_index(drop=True), frame[ds.label_column].astype(int).reset_index(drop=True)


def bootstrap_diff_ci(y, scores_a, scores_b, spec: Bootstrap) -> tuple[float, float]:
    """Paired bootstrap interval of PR-AUC(a) - PR-AUC(b): both models see the same resampled rows."""
    y, a, b = np.asarray(y), np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(spec.seed)
    diffs = []
    for _ in range(spec.resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    tail = (1 - spec.confidence) / 2 * 100
    lo, hi = np.percentile(diffs, [tail, 100 - tail])
    return float(lo), float(hi)


def _check(passed: bool, detail: str) -> dict:
    return {"passed": bool(passed), "detail": detail}


def _sanity_checks(cand: Bundle, champ: Bundle, spec: GateSpec) -> dict:
    cm = cand.meta
    return {
        "meta_has_threshold": _check("threshold" in cm, "the candidate meta carries its frozen threshold"),
        "same_features": _check(
            cm.get("features") == champ.meta.get("features"), "candidate and champion use the same feature list"
        ),
    }


def _score(bundle: Bundle, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(bundle.model.predict_proba(X[bundle.meta["features"]])[:, 1], dtype=float)


def decide(cand: Bundle, champ: Bundle, X: pd.DataFrame, y, spec: GateSpec) -> dict:
    checks = _sanity_checks(cand, champ, spec)
    metrics = None
    if all(c["passed"] for c in checks.values()):
        cs, hs = _score(cand, X), _score(champ, X)
        checks["no_missing_scores"] = _check(
            not (np.isnan(cs).any() or np.isnan(hs).any()), "no missing scores from either model"
        )
        lo_r, hi_r = spec.sanity.flagged_share_ratio
        fit = cand.meta.get("flagged_share_at_fit")
        share = float((cs >= cand.meta["threshold"]).mean())
        in_band = fit is not None and lo_r * fit <= share <= hi_r * fit
        checks["flagged_share_in_band"] = _check(
            in_band, f"share flagged at the frozen threshold {share:.3f}, expected {lo_r}-{hi_r} x {fit}"
        )
        if checks["no_missing_scores"]["passed"]:
            pr_c, pr_h = average_precision_score(y, cs), average_precision_score(y, hs)
            lo, hi = bootstrap_diff_ci(y, cs, hs, spec.promotion.bootstrap)
            gain = float(pr_c - pr_h)
            metrics = {"pr_auc_candidate": float(pr_c), "pr_auc_champion": float(pr_h), "pr_auc_gain": gain,
                       "diff_ci": [lo, hi], "flagged_share": share}
            checks["pr_auc_gain"] = _check(
                gain >= spec.promotion.min_pr_auc_gain,
                f"PR-AUC gain {gain:.4f}, needs at least {spec.promotion.min_pr_auc_gain}",
            )
            conf = spec.promotion.bootstrap.confidence
            checks["bootstrap_excludes_zero"] = _check(
                lo > 0, f"{conf:.0%} paired bootstrap interval of the PR-AUC difference [{lo:.4f}, {hi:.4f}], needs lower bound above 0"
            )
    reasons = [f"{name}: {c['detail']}" for name, c in checks.items() if not c["passed"]]
    promote = not reasons
    return {
        "decision": "promote" if promote else "keep",
        "reasons": reasons or ["every check passed"],
        "checks": checks,
        "metrics": metrics,
    }


def run_gate(
    run_dir: str, name: str, champion_uri: str, evaluation_uri: str,
    spec: GateSpec, ds: GcpDatasetSpec, fetch, read_frame,
) -> dict:
    cand = load_bundle(f"{run_dir.rstrip('/')}/models/{name}", fetch)
    champ = load_bundle(champion_uri, fetch)
    X, y = evaluation_rows(read_frame(evaluation_uri), ds)
    decision = decide(cand, champ, X, y, spec)
    return {
        **decision,
        "candidate": {"run_dir": run_dir, "name": name},
        "champion_uri": champion_uri,
        "evaluation": {"uri": evaluation_uri, "rows": len(X), "positives": int(y.sum())},
        "limitation": " ".join(spec.limitation.split()),
    }


def _read_frame(uri: str) -> pd.DataFrame:
    return read_csv(uri) if uri.startswith("gs://") else pd.read_csv(uri)


def main(argv: list[str] | None = None) -> None:
    spec = default_gate_spec()
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("name")
    parser.add_argument("--champion-uri", default=spec.champion_uri)
    parser.add_argument("--evaluation-uri", default=spec.evaluation_uri, help="labelled loans file (gs:// or local)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    decision = run_gate(
        args.run_dir, args.name, args.champion_uri, args.evaluation_uri,
        spec, default_dataset_spec(), read_bytes, _read_frame,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
