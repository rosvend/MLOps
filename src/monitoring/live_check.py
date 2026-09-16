"""Live drift check: the reference training window against whatever the API has scored.

No labels exist for live traffic - a credit application's outcome is unknown at scoring
time - so this reports feature and prediction drift plus the operational flagged-share
check, and stops there. Model-quality drift is src/pipelines/monitor.py's job, where both
sides of the comparison are labelled.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.monitoring.alerts import send_alert
from src.monitoring.drift_detector import DriftSummary, run_drift_report
from src.monitoring.predictions_log import PredictionsLog
from src.monitoring.spec import MonitoringSpec

_log = logging.getLogger(__name__)


class InsufficientCurrentData(Exception):
    """Fewer logged predictions than the configured minimum - a drift check on a handful
    of rows is noise, not a report."""

    def __init__(self, have: int, need: int):
        self.have, self.need = have, need
        super().__init__(f"{have} predicciones registradas, se necesitan al menos {need}")


@dataclass(frozen=True)
class LiveDriftResult:
    drift: DriftSummary
    flagged_share_current: float
    flagged_share_at_fit: float
    current_rows: int


def run_live_drift_check(
    reference: pd.DataFrame,
    predictions_log: PredictionsLog,
    flagged_share_at_fit: float,
    spec: MonitoringSpec,
) -> LiveDriftResult:
    current = predictions_log.read_all()
    if len(current) < spec.min_current_rows:
        raise InsufficientCurrentData(len(current), spec.min_current_rows)

    # Operational drift is a plain proportion, not an Evidently metric - it is a business
    # threshold, not a distributional test - so review_flag is excluded from the
    # statistical comparison and checked separately, right below.
    columnas_estadisticas = [c for c in reference.columns if c != "review_flag"]
    resumen = run_drift_report(reference[columnas_estadisticas], current, spec)

    flagged_actual = float(current["review_flag"].astype(bool).mean())

    if resumen.dataset_drift_detected:
        send_alert(
            f"dataset drift detected: {resumen.drift_share:.1%} of features drifted "
            f"against the training reference ({len(current)} logged predictions)",
            resumen.to_dict(),
            spec,
        )
    if flagged_actual > spec.flagged_share_threshold:
        send_alert(
            f"flagged share {flagged_actual:.1%} exceeds the {spec.flagged_share_threshold:.0%} "
            f"operational threshold on live traffic",
            {"flagged_share": flagged_actual, "n": len(current)},
            spec,
        )

    return LiveDriftResult(
        drift=resumen,
        flagged_share_current=flagged_actual,
        flagged_share_at_fit=flagged_share_at_fit,
        current_rows=len(current),
    )


RAIZ = Path(__file__).resolve().parents[2]


def load_reference(path: str | Path) -> pd.DataFrame:
    ruta = Path(path)
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    return pd.read_parquet(ruta)
