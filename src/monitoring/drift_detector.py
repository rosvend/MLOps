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
from evidently import DataDefinition, Dataset, Report
from evidently.legacy.tests.base_test import TestStatus
from evidently.metrics import ValueDrift

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


# Extension dtypes pandas needs for nullable data (Int64, Float64, boolean, the string[]
# Feast returns for categoricals) are not what Evidently's column-type inference expects -
# it raised "Cannot calculate drift metric ... type ColumnType.Unknown" on edad_cliente
# (plain Int64) until this normalisation was added. Plain numpy dtypes only from here on;
# NaN survives the cast (Evidently ignores it, unlike a mis-detected column type).
_A_NUMERICO = {"Int64", "Float64", "boolean", "bool"}
_A_TEXTO = {"string[python]", "string"}


def _normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for columna in df.columns:
        dtype = str(df[columna].dtype)
        if dtype in _A_NUMERICO:
            df[columna] = df[columna].astype("float64")
        elif dtype in _A_TEXTO or dtype == "category":
            df[columna] = df[columna].astype("object")
    return df


def run_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, spec: MonitoringSpec
) -> DriftSummary:
    """Compares only the columns both frames share - a column unique to either side (a
    label, an id) is not a feature to compare and must not crash the report."""
    compartidas = [c for c in reference.columns if c in current.columns]
    ref, cur = _normalizar_tipos(reference[compartidas]), _normalizar_tipos(current[compartidas])

    # Classified from the reference alone: current is live-logged data (SQLite -> JSON ->
    # DataFrame), and a column that is entirely null in the logged sample comes back
    # object dtype regardless of what it is in the reference - is_numeric_dtype on
    # current would then disagree with the reference's own classification.
    numericas = [c for c in compartidas if pd.api.types.is_numeric_dtype(ref[c])]
    categoricas = [c for c in compartidas if c not in numericas]
    metodos = {c: _numeric_method(spec, len(ref)) for c in numericas}
    metodos |= {c: spec.categorical_stattest for c in categoricas}

    # Both frames coerced to the type the reference's classification promised, not
    # merely to "a plain numpy dtype" - current's actual dtype can still disagree with
    # what it was classified as, and Evidently has no tolerance for that mismatch.
    for columna in numericas:
        ref[columna] = pd.to_numeric(ref[columna], errors="coerce")
        cur[columna] = pd.to_numeric(cur[columna], errors="coerce")
    for columna in categoricas:
        ref[columna] = ref[columna].astype(object)
        cur[columna] = cur[columna].astype(object)

    # A column with zero non-null values on either side has nothing to test drift on -
    # Evidently refuses outright ("An empty column ... was provided"), correctly. Skipped
    # rather than crashing every other column's result along with it; logged so the gap
    # is visible rather than silently absent from drifted_features.
    vacias = [c for c in [*numericas, *categoricas] if ref[c].isna().all() or cur[c].isna().all()]
    if vacias:
        _log.warning("columnas sin datos para probar drift, omitidas: %s", vacias)
    numericas = [c for c in numericas if c not in vacias]
    categoricas = [c for c in categoricas if c not in vacias]

    # drift_share and dataset_drift_detected are derived from these same per-column
    # tests below, not from a separately-configured DriftedColumnsCount metric: that
    # metric picks its own per-column significance internally, which does not
    # necessarily agree with `alpha` here, and the two silently disagreeing on which
    # columns counted as drifted is worse than not having the second metric at all.
    metricas = [
        ValueDrift(column=columna, method=metodos[columna], threshold=spec.alpha)
        for columna in [*numericas, *categoricas]
    ]

    # An explicit DataDefinition, not Evidently's own inference: a low-cardinality
    # numeric column (a count like huella_consulta) would otherwise be auto-classified
    # categorical by Evidently's cardinality heuristic, disagreeing with the dtype-based
    # split above and raising "stattest wasserstein isn't applicable to feature of type
    # cat" the moment a numeric column crossed the sample-size cutoff into wasserstein.
    definicion = DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    ref_ds = Dataset.from_pandas(ref, data_definition=definicion)
    cur_ds = Dataset.from_pandas(cur, data_definition=definicion)

    report = Report(metrics=metricas, include_tests=True)
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    data = snapshot.dict()

    id_a_columna = {m["id"]: m["config"]["column"] for m in data["metrics"]}
    drifted = sorted(
        id_a_columna[prueba["metric_config"]["metric_id"]]
        for prueba in data["tests"]
        if _is_drifted(prueba["status"])
    )
    drift_share = len(drifted) / len(compartidas) if compartidas else 0.0

    return DriftSummary(
        dataset_drift_detected=drift_share > spec.drift_share_threshold,
        drift_share=drift_share,
        drifted_features=drifted,
        methods_used=metodos,
        snapshot=snapshot,
    )
