"""How the feature pipeline behaves, loaded from config/features/default.yaml.

Nothing here duplicates the YAML: this module only gives it a validated shape. Loading
is plain YAML so every module works without Hydra initialised, exactly as
src/models/scorecard.py does for the scorecard.

Named `spec` rather than `config` to keep it distinct from src/config.py, which composes
the application config. This module imports nothing from the project, so it cannot take
part in an import cycle.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "features" / "default.yaml"


class Bounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edad: tuple[int, int]
    puntaje_bureau: tuple[int, int]
    salario_maximo: int
    capital_minimo: int
    plazo_maximo_meses: int

    @model_validator(mode="after")
    def _ascend(self):
        for nombre in ("edad", "puntaje_bureau"):
            bajo, alto = getattr(self, nombre)
            if bajo >= alto:
                raise ValueError(f"bounds.{nombre} debe ir de menor a mayor: {(bajo, alto)}")
        return self


class Vocabularies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tendencias: list[str]
    tipos_credito_frecuentes: list[str]
    tipo_credito_residual: str
    tipos_laboral: list[str]
    tipo_laboral_independiente: str

    @property
    def tipos_credito(self) -> list[str]:
        return [*self.tipos_credito_frecuentes, self.tipo_credito_residual]

    @model_validator(mode="after")
    def _independiente_es_un_tipo_laboral(self):
        if self.tipo_laboral_independiente not in self.tipos_laboral:
            raise ValueError(
                f"vocabularies.tipo_laboral_independiente "
                f"{self.tipo_laboral_independiente!r} no está en tipos_laboral"
            )
        return self


class AgeBands(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: list[int]
    labels: list[str]

    @model_validator(mode="after")
    def _one_more_edge_than_band(self):
        if len(self.bins) != len(self.labels) + 1:
            raise ValueError(
                f"age_bands necesita {len(self.labels) + 1} cortes para "
                f"{len(self.labels)} etiquetas, trae {len(self.bins)}"
            )
        if self.bins != sorted(self.bins):
            raise ValueError(f"age_bands.bins debe ir de menor a mayor: {self.bins}")
        return self


class Units(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: int
    thousands: list[str]


class Columns(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sin_varianza: list[str]
    enteros: list[str]
    vigiladas: list[str]


class Engineering(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winsor_quantile: float
    heavy_tailed: list[str]
    log_scale: list[str]

    @model_validator(mode="after")
    def _log_columns_are_clipped_first(self):
        sueltas = set(self.log_scale) - set(self.heavy_tailed)
        if sueltas:
            raise ValueError(
                f"engineering.log_scale sin recorte previo en heavy_tailed: {sorted(sueltas)}"
            )
        if not 0.5 < self.winsor_quantile < 1.0:
            raise ValueError(
                f"engineering.winsor_quantile fuera de rango: {self.winsor_quantile}"
            )
        return self


class FeatureSpec(BaseModel):
    """Column roles, cleaning rules and fitted-transform settings for the whole pipeline."""

    model_config = ConfigDict(extra="forbid")

    entity_key: str
    event_timestamp: str
    target: str
    withheld: list[str]
    bounds: Bounds
    vocabularies: Vocabularies
    age_bands: AgeBands
    units: Units
    columns: Columns
    engineering: Engineering
    output_path: str
    event_timestamp_timezone: str = "UTC"
    feature_views: dict[str, list[str]]

    @property
    def no_son_features(self) -> frozenset[str]:
        """Every column with a role other than "feature"."""
        return frozenset({self.entity_key, self.event_timestamp, self.target, *self.withheld})

    @model_validator(mode="after")
    def _roles_are_disjoint(self):
        reservados = [self.entity_key, self.event_timestamp, self.target, *self.withheld]
        repetidos = {c for c in reservados if reservados.count(c) > 1}
        if repetidos:
            raise ValueError(f"una columna no puede tener dos roles: {sorted(repetidos)}")
        return self

    @property
    def feature_view_columns(self) -> list[str]:
        return [c for columnas in self.feature_views.values() for c in columnas]

    @model_validator(mode="after")
    def _views_partition_the_features(self):
        """Every feature belongs to exactly one view: none duplicated, none forgotten."""
        todas = self.feature_view_columns
        repetidas = {c for c in todas if todas.count(c) > 1}
        if repetidas:
            raise ValueError(f"columnas en más de una feature view: {sorted(repetidas)}")
        reservadas = self.no_son_features & set(todas)
        if reservadas:
            raise ValueError(f"una feature view no puede exponer columnas reservadas: {sorted(reservadas)}")
        return self

    @model_validator(mode="after")
    def _no_feature_is_also_reserved(self):
        chocan = self.no_son_features & set(self.engineering.heavy_tailed)
        if chocan:
            raise ValueError(f"engineering no puede tocar columnas reservadas: {sorted(chocan)}")
        return self

    @model_validator(mode="after")
    def _age_bands_cover_the_accepted_ages(self):
        """An accepted age with no band gets rango_edad = NA, and the scorecard then pays
        it the "age unknown" points although the age is known - silently, with no error."""
        joven, mayor = self.bounds.edad
        # pd.cut leaves the first edge open, so it has to sit below the youngest age.
        if self.age_bands.bins[0] >= joven or self.age_bands.bins[-1] < mayor:
            raise ValueError(
                f"age_bands.bins {self.age_bands.bins} no cubre bounds.edad {[joven, mayor]}"
            )
        return self


@lru_cache(maxsize=1)
def default_spec() -> FeatureSpec:
    return FeatureSpec(**yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8")))
