"""Programmatic champion selection.

Three criteria, in the order a risk committee would weigh them: how well the model
separates defaults out of time, how stable it was across folds, and what it costs to
train and to score. The primary metric leads; the rest are reported and break ties.

The incumbent heuristic is a candidate like any other. A selector that cannot return the
model already in place is not making a decision, it is rubber-stamping one.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CandidateResult:
    """Everything known about one candidate after tuning and the held-out evaluation."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    cv_mean: float = float("nan")
    cv_std: float = float("nan")
    metrics: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = float("nan")
    predict_ms_per_1k: float = float("nan")
    run_id: str | None = None
    tuned: bool = True

    def score(self, metric: str) -> float:
        """NaN means the evaluation did not produce this metric, so it must not compete."""
        valor = self.metrics.get(metric)
        if valor is None:
            return float("-inf")
        valor = float(valor)
        return float("-inf") if math.isnan(valor) else valor


def comparison_table(resultados: list[CandidateResult], metricas: list[str]) -> pd.DataFrame:
    filas = []
    for r in resultados:
        fila = {"model": r.name, "cv_mean": r.cv_mean, "cv_std": r.cv_std}
        fila |= {m: r.metrics.get(m) for m in metricas}
        fila |= {"fit_s": r.fit_seconds, "predict_ms_1k": r.predict_ms_per_1k}
        filas.append(fila)
    return pd.DataFrame(filas).set_index("model")


def select_champion(resultados: list[CandidateResult], metric: str) -> CandidateResult:
    """Highest out-of-time primary metric wins; fold stability breaks a tie.

    A candidate with no usable score cannot win. If none has one the metric is
    misspelled or every evaluation failed, and either way returning whichever came
    first in the list would be a decision nobody made.
    """
    if not resultados:
        raise ValueError("no hay candidatos que comparar")
    comparables = [r for r in resultados if r.score(metric) > float("-inf")]
    if not comparables:
        raise ValueError(
            f"ningún candidato reporta {metric!r}: "
            f"{sorted({m for r in resultados for m in r.metrics})}"
        )

    def _estabilidad(r: CandidateResult) -> float:
        # An unknown spread must not beat a measured one when the scores tie.
        return -1.0 if math.isnan(r.cv_std) else -r.cv_std

    return max(comparables, key=lambda r: (round(r.score(metric), 4), _estabilidad(r)))


def justification(champion: CandidateResult, resultados: list[CandidateResult], metric: str) -> str:
    otros = [r for r in resultados if r.name != champion.name]
    mejor_otro = max(otros, key=lambda r: r.score(metric)) if otros else None
    lineas = [
        f"Champion: {champion.name}",
        f"  {metric:<10} {champion.score(metric):.4f}"
        + (f"  (next best {mejor_otro.name} at {mejor_otro.score(metric):.4f})" if mejor_otro else ""),
        f"  gini       {champion.metrics.get('gini', float('nan')):.4f} out of time",
        f"  stability  cv {champion.cv_mean:.4f} +/- {champion.cv_std:.4f} across folds",
        f"  cost       {champion.fit_seconds:.1f}s to fit, "
        f"{champion.predict_ms_per_1k:.1f} ms per 1k rows to score",
    ]
    return "\n".join(lineas)


def as_record(champion: CandidateResult, tabla: pd.DataFrame, metric: str) -> dict[str, Any]:
    return {
        "primary_metric": metric,
        "champion": asdict(champion),
        "comparison": tabla.reset_index().to_dict(orient="records"),
    }
