"""Programmatic champion selection.

Three criteria, in the order a risk committee would weigh them: how well the model
separates defaults out of time, how stable it was across folds, and what it costs to
train and to score. The primary metric leads; the rest are reported and break ties.

The incumbent heuristic is a candidate like any other. A selector that cannot return the
model already in place is not making a decision, it is rubber-stamping one.
"""

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
        valor = self.metrics.get(metric)
        return float("-inf") if valor is None else float(valor)


def comparison_table(resultados: list[CandidateResult], metricas: list[str]) -> pd.DataFrame:
    filas = []
    for r in resultados:
        fila = {"model": r.name, "cv_mean": r.cv_mean, "cv_std": r.cv_std}
        fila |= {m: r.metrics.get(m) for m in metricas}
        fila |= {"fit_s": r.fit_seconds, "predict_ms_1k": r.predict_ms_per_1k}
        filas.append(fila)
    return pd.DataFrame(filas).set_index("model")


def select_champion(resultados: list[CandidateResult], metric: str) -> CandidateResult:
    """Highest out-of-time primary metric wins; fold stability breaks a tie."""
    if not resultados:
        raise ValueError("no hay candidatos que comparar")
    return max(
        resultados,
        key=lambda r: (round(r.score(metric), 4), -(r.cv_std if r.cv_std == r.cv_std else 1.0)),
    )


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
