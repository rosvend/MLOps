"""One rule per EDA finding, each scoring a single application in isolation.

Weights and band cut-points come from the scorecard (config/model/heuristic.yaml),
injected so an experiment can vary them without touching this file. The rate behind
every number is tabulated in docs/heuristic-model.md.
"""

import bisect
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.features.spec import default_spec
from src.models.scorecard import (
    HUELLA_BANDAS,
    PUNTAJE_BANDAS,
    Scorecard,
    default_scorecard,
)

Record = Mapping[str, Any] | pd.Series

# A missing column is not a missing value: two rules pay points for absence, so a
# dropped or renamed column would silently push loans over the flag threshold.
REQUIRED_COLUMNS = frozenset(
    {
        "puntaje_datacredito",
        "huella_consulta",
        "rango_edad",
        "edad_cliente",
        "tipo_laboral",
        "ratio_ingreso_declarado_bureau",
        "tendencia_ingresos",
        "capital_prestado",
        "plazo_meses",
        "promedio_ingresos_datacredito",
    }
)




def _falta(valor: Any) -> bool:
    return valor is None or pd.isna(valor)


def _puntos(tabla: dict[str, int], etiqueta: str, campo: str) -> int:
    """A label the table does not know is a drift bug, not a zero-point band."""
    if etiqueta not in tabla:
        raise KeyError(f"{campo}: la banda {etiqueta!r} no tiene puntaje asignado")
    return tabla[etiqueta]


def bureau_score_band(record: Record, scorecard: Scorecard | None = None) -> int:
    """The strongest single signal: worst decile 9.8% against best 3.0%."""
    s = scorecard or default_scorecard()
    puntaje = record.get("puntaje_datacredito")
    if _falta(puntaje):
        return s.points.puntaje_bureau_ausente
    bajo, alto = s.cut_points.puntaje_bureau_terciles
    if puntaje < bajo:
        etiqueta = PUNTAJE_BANDAS[0]
    else:
        etiqueta = PUNTAJE_BANDAS[1] if puntaje < alto else PUNTAJE_BANDAS[2]
    return _puntos(s.points.puntaje_bureau, etiqueta, "puntaje_datacredito")


def inquiry_band(record: Record, scorecard: Scorecard | None = None) -> int:
    """Bureau inquiries add information the score misses, inside every score band."""
    s = scorecard or default_scorecard()
    huella = record.get("huella_consulta")
    if _falta(huella):
        return 0
    bajo, medio = s.cut_points.huella_bandas
    if huella <= bajo:
        etiqueta = HUELLA_BANDAS[0]
    else:
        etiqueta = HUELLA_BANDAS[1] if huella <= medio else HUELLA_BANDAS[2]
    return _puntos(s.points.huella, etiqueta, "huella_consulta")


def age_band(record: Record, scorecard: Scorecard | None = None) -> int:
    """The only applicant-form variable that competes with the bureau: 8.8% to 3.2%."""
    s = scorecard or default_scorecard()
    rango = record.get("rango_edad")
    if _falta(rango):
        return s.points.rango_edad_ausente
    return _puntos(s.points.rango_edad, str(rango), "rango_edad")


def young_independent(record: Record, scorecard: Scorecard | None = None) -> int:
    """Self-employment only carries risk under 36; averaged over all ages it vanishes."""
    s = scorecard or default_scorecard()
    edad = record.get("edad_cliente")
    independiente = default_spec().vocabularies.tipo_laboral_independiente
    if _falta(edad) or record.get("tipo_laboral") != independiente:
        return 0
    return s.points.independiente_joven if edad < s.cut_points.edad_joven else 0


def income_gap_quartile(record: Record, scorecard: Scorecard | None = None) -> int:
    """What discriminates is the gap between declared and bureau-observed income."""
    s = scorecard or default_scorecard()
    brecha = record.get("ratio_ingreso_declarado_bureau")
    if _falta(brecha):
        return 0
    return s.points.brecha_ingreso[bisect.bisect_right(s.cut_points.brecha_ingreso_cuartiles, brecha)]


def decreasing_income_trend(record: Record, scorecard: Scorecard | None = None) -> int:
    """Stacks with over-declaration: both together reach 9.1% against 2.8%."""
    s = scorecard or default_scorecard()
    tendencia = record.get("tendencia_ingresos")
    if _falta(tendencia):
        return 0
    return _puntos(s.points.tendencia, str(tendencia), "tendencia_ingresos")


def high_amount_long_term(record: Record, scorecard: Scorecard | None = None) -> int:
    """The one lever the bank sets itself: 11.0% default over 626 loans."""
    s = scorecard or default_scorecard()
    capital = record.get("capital_prestado")
    plazo = record.get("plazo_meses")
    if _falta(capital) or _falta(plazo):
        return 0
    riesgoso = (
        capital >= s.cut_points.capital_cuartil_alto and plazo >= s.cut_points.plazo_largo_meses
    )
    return s.points.monto_alto_plazo_largo if riesgoso else 0


def missing_bureau_income(record: Record, scorecard: Scorecard | None = None) -> int:
    """Absence is itself a signal: the bureau income block is missing for riskier clients."""
    s = scorecard or default_scorecard()
    if _falta(record.get("promedio_ingresos_datacredito")):
        return s.points.ingreso_bureau_ausente
    return 0
