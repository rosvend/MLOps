"""The scorecard specification: band cut-points and the points each band charges.

The numbers live in config/model/heuristic.yaml, not here, so a run can record
exactly which weights produced it and an experiment can override them without a
code change. Loading is plain YAML so rules work without Hydra initialised.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from src.features.cleaning import TENDENCIAS
from src.features.derive import RANGO_EDAD_LABELS

SCORECARD_PATH = Path(__file__).resolve().parents[2] / "config" / "model" / "heuristic.yaml"

# Band labels the points tables must key on; the validators below enforce the match.
PUNTAJE_BANDAS = ("bajo", "medio", "alto")
HUELLA_BANDAS = ("0-3", "4-6", "7+")


class CutPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puntaje_bureau_terciles: tuple[int, int]
    huella_bandas: tuple[int, int]
    brecha_ingreso_cuartiles: tuple[float, ...]
    capital_cuartil_alto: int
    plazo_largo_meses: int
    edad_joven: int


class Points(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puntaje_bureau: dict[str, int]
    puntaje_bureau_ausente: int
    huella: dict[str, int]
    rango_edad: dict[str, int]
    rango_edad_ausente: int
    independiente_joven: int
    brecha_ingreso: tuple[int, ...]
    tendencia: dict[str, int]
    monto_alto_plazo_largo: int
    ingreso_bureau_ausente: int


class Scorecard(BaseModel):
    """Higher score = higher risk."""

    model_config = ConfigDict(extra="forbid")

    threshold: int
    deciles: int = 10
    decile_minimo: int = 200
    base_rate: float
    lift_multiplier: int
    cut_points: CutPoints
    points: Points

    @model_validator(mode="after")
    def _vocabularies_match(self):
        # Rename a band label in derive.py and every applicant in it would otherwise
        # score 0 silently: no exception, no schema violation, no failing test.
        if set(self.points.rango_edad) != set(RANGO_EDAD_LABELS):
            raise ValueError(
                f"points.rango_edad {sorted(self.points.rango_edad)} no coincide con "
                f"RANGO_EDAD_LABELS {sorted(RANGO_EDAD_LABELS)}"
            )
        if set(self.points.tendencia) != set(TENDENCIAS):
            raise ValueError(
                f"points.tendencia {sorted(self.points.tendencia)} no coincide con "
                f"TENDENCIAS {sorted(TENDENCIAS)}"
            )
        if set(self.points.puntaje_bureau) != set(PUNTAJE_BANDAS):
            raise ValueError(f"points.puntaje_bureau debe traer las bandas {list(PUNTAJE_BANDAS)}")
        if set(self.points.huella) != set(HUELLA_BANDAS):
            raise ValueError(f"points.huella debe traer las bandas {list(HUELLA_BANDAS)}")
        return self

    @model_validator(mode="after")
    def _cut_points_ascend(self):
        # An inverted override would silently mis-band every applicant instead of failing.
        for nombre in ("puntaje_bureau_terciles", "huella_bandas", "brecha_ingreso_cuartiles"):
            cortes = getattr(self.cut_points, nombre)
            if list(cortes) != sorted(cortes):
                raise ValueError(f"cut_points.{nombre} debe ir de menor a mayor: {list(cortes)}")
        return self

    @model_validator(mode="after")
    def _band_counts_agree(self):
        esperado = len(self.cut_points.brecha_ingreso_cuartiles) + 1
        if len(self.points.brecha_ingreso) != esperado:
            raise ValueError(
                f"points.brecha_ingreso necesita {esperado} pesos para "
                f"{len(self.cut_points.brecha_ingreso_cuartiles)} cortes"
            )
        if len(self.points.huella) != len(self.cut_points.huella_bandas) + 1:
            raise ValueError("points.huella no cubre las bandas de huella_bandas")
        return self


@lru_cache(maxsize=1)
def default_scorecard() -> Scorecard:
    return Scorecard(**yaml.safe_load(SCORECARD_PATH.read_text(encoding="utf-8")))
