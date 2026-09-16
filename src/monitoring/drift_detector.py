"""Statistical drift between a reference distribution and a current one.

Built on the verified Evidently 0.7 API: `Report(metrics=[...], include_tests=True)`,
`.run(current_data=, reference_data=)` returns a `Snapshot`. Metrics are matched back to
columns by their `config`, not by list position - the dict export does not guarantee
order survives a future version, and matching by config is what the library itself gives
us to do it safely.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.legacy.tests.base_test import TestStatus
from evidently.metrics import DriftedColumnsCount, ValueDrift

from src.monitoring.spec import MonitoringSpec

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftSummary:
    dataset_drift_detected: bool
    drift_share: float
    drifted_features: list[str]
    methods_used: dict[str, str] = field(default_factory=dict)
    snapshot: Any = field(repr=False, default=None)

    def to_dict(self) -> dict:
        return {
            "dataset_drift_detected": self.dataset_drift_detected,
            "drift_share": self.drift_share,
            "drifted_features": self.drifted_features,
        }

    def save(self, html_path: str | Path, json_path: str | Path) -> None:
        html_path, json_path = Path(html_path), Path(json_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.snapshot is not None:
            self.snapshot.save_html(str(html_path))
        json_path.write_text(json.dumps(self.to_dict(), indent=2))


def _numeric_method(spec: MonitoringSpec, n: int) -> str:
    if n < spec.numeric_sample_size_cutoff:
        return spec.numeric_small_sample_stattest
    return spec.numeric_large_sample_stattest


def _is_drifted(status) -> bool:
    return status == TestStatus.FAIL


def run_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, spec: MonitoringSpec
) -> DriftSummary:
    """Compares only the columns both frames share - a column unique to either side (a
    label, an id) is not a feature to compare and must not crash the report."""
    compartidas = [c for c in reference.columns if c in current.columns]
    ref, cur = reference[compartidas], current[compartidas]

    numericas = [c for c in compartidas if pd.api.types.is_numeric_dtype(ref[c])]
    categoricas = [c for c in compartidas if c not in numericas]
    metodos = {c: _numeric_method(spec, len(ref)) for c in numericas}
    metodos |= {c: spec.categorical_stattest for c in categoricas}

    metricas = [
        DriftedColumnsCount(
            num_stattest=_numeric_method(spec, len(ref)),
            cat_stattest=spec.categorical_stattest,
            drift_share=spec.drift_share_threshold,
        ),
    ]
    for columna in numericas:
        metricas.append(ValueDrift(column=columna, method=metodos[columna], threshold=spec.alpha))
    for columna in categoricas:
        metricas.append(ValueDrift(column=columna, method=metodos[columna], threshold=spec.alpha))

    report = Report(metrics=metricas, include_tests=True)
    snapshot = report.run(current_data=cur, reference_data=ref)
    data = snapshot.dict()

    id_a_columna = {m["id"]: m["config"].get("column") for m in data["metrics"]}
    id_a_valor = {m["id"]: m["value"] for m in data["metrics"]}
    drifted: list[str] = []
    dataset_drift_detected = False
    drift_share = 0.0

    for prueba in data["tests"]:
        metric_id = prueba["metric_config"]["metric_id"]
        columna = id_a_columna.get(metric_id)
        if columna is None:
            # The DriftedColumnsCount test: no column, it is the dataset-level verdict.
            dataset_drift_detected = _is_drifted(prueba["status"])
            drift_share = float(id_a_valor[metric_id]["share"])
        elif _is_drifted(prueba["status"]):
            drifted.append(columna)

    return DriftSummary(
        dataset_drift_detected=dataset_drift_detected,
        drift_share=drift_share,
        drifted_features=sorted(drifted),
        methods_used=metodos,
        snapshot=snapshot,
    )
