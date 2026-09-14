"""One rule per EDA finding, each scoring a single application in isolation.

Points are round((lift - 1) x 4) over the default rates measured in the EDA, so a
band that doubles the portfolio rate costs +4 and a band that halves it pays -2.
The rate behind every number is tabulated in docs/heuristic-model.md.
"""

import bisect
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.models.thresholds import (
    BRECHA_INGRESO_CUARTILES,
    CAPITAL_CUARTIL_ALTO,
    EDAD_JOVEN,
    HUELLA_BANDAS,
    PLAZO_LARGO_MESES,
    PUNTAJE_BUREAU_TERCILES,
)

Record = Mapping[str, Any]

PUNTOS_PUNTAJE_BUREAU = {"bajo": 2, "medio": -1, "alto": -2}
PUNTOS_PUNTAJE_BUREAU_AUSENTE = 2
PUNTOS_HUELLA = {"0-3": -1, "4-6": 0, "7+": 2}
PUNTOS_RANGO_EDAD = {"18-25": 3, "26-35": 1, "36-45": -1, "46-55": -1, "56-65": 0, "66+": -1}
PUNTOS_RANGO_EDAD_AUSENTE = 2
PUNTOS_INDEPENDIENTE_JOVEN = 3
PUNTOS_BRECHA_INGRESO = (-1, -1, 0, 1)
PUNTOS_TENDENCIA = {"Decreciente": 1, "Estable": 0, "Creciente": -1}
PUNTOS_MONTO_ALTO_PLAZO_LARGO = 5
PUNTOS_INGRESO_BUREAU_AUSENTE = 1


def _falta(valor: Any) -> bool:
    return valor is None or pd.isna(valor)


def bureau_score_band(record: Record) -> int:
    """The strongest single signal: worst decile 9.8% against best 3.0%."""
    puntaje = record.get("puntaje_datacredito")
    if _falta(puntaje):
        return PUNTOS_PUNTAJE_BUREAU_AUSENTE
    bajo, alto = PUNTAJE_BUREAU_TERCILES
    if puntaje < bajo:
        return PUNTOS_PUNTAJE_BUREAU["bajo"]
    return PUNTOS_PUNTAJE_BUREAU["medio" if puntaje < alto else "alto"]


def inquiry_band(record: Record) -> int:
    """Bureau inquiries add information the score misses, inside every score band."""
    huella = record.get("huella_consulta")
    if _falta(huella):
        return 0
    bajo, medio = HUELLA_BANDAS
    if huella <= bajo:
        return PUNTOS_HUELLA["0-3"]
    return PUNTOS_HUELLA["4-6" if huella <= medio else "7+"]


def age_band(record: Record) -> int:
    """The only applicant-form variable that competes with the bureau: 8.8% to 3.2%."""
    rango = record.get("rango_edad")
    if _falta(rango):
        return PUNTOS_RANGO_EDAD_AUSENTE
    return PUNTOS_RANGO_EDAD.get(str(rango), 0)


def young_independent(record: Record) -> int:
    """Self-employment only carries risk under 36; averaged over all ages it vanishes."""
    edad = record.get("edad_cliente")
    if _falta(edad) or record.get("tipo_laboral") != "Independiente":
        return 0
    return PUNTOS_INDEPENDIENTE_JOVEN if edad < EDAD_JOVEN else 0


def income_gap_quartile(record: Record) -> int:
    """What discriminates is the gap between declared and bureau-observed income."""
    brecha = record.get("ratio_ingreso_declarado_bureau")
    if _falta(brecha):
        return 0
    return PUNTOS_BRECHA_INGRESO[bisect.bisect_right(BRECHA_INGRESO_CUARTILES, brecha)]


def decreasing_income_trend(record: Record) -> int:
    """Stacks with over-declaration: both together reach 9.1% against 2.8%."""
    tendencia = record.get("tendencia_ingresos")
    if _falta(tendencia):
        return 0
    return PUNTOS_TENDENCIA.get(str(tendencia), 0)


def high_amount_long_term(record: Record) -> int:
    """The one lever the bank sets itself: 11.0% default over 626 loans."""
    capital = record.get("capital_prestado")
    plazo = record.get("plazo_meses")
    if _falta(capital) or _falta(plazo):
        return 0
    riesgoso = capital >= CAPITAL_CUARTIL_ALTO and plazo >= PLAZO_LARGO_MESES
    return PUNTOS_MONTO_ALTO_PLAZO_LARGO if riesgoso else 0


def missing_bureau_income(record: Record) -> int:
    """Absence is itself a signal: the bureau income block is missing for riskier clients."""
    return PUNTOS_INGRESO_BUREAU_AUSENTE if _falta(record.get("promedio_ingresos_datacredito")) else 0
