"""Cleaning: sentinels, types, units and inconsistency flags.

Every rule applied here comes from config/features/default.yaml, injected as a
FeatureSpec so an experiment can vary it without touching this module.
"""

import numpy as np
import pandas as pd

from src.features.spec import FeatureSpec, default_spec

_PAGO_ATIEMPO = {"1": True, "0": False, "true": True, "false": False}


def nullify_sentinels(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Values the source system writes to mean "unknown" but that read as valid numbers."""
    spec = spec or default_spec()
    edad_min, edad_max = spec.bounds.edad
    puntaje_min, puntaje_max = spec.bounds.puntaje_bureau
    df = df.copy()
    df["edad_cliente"] = df["edad_cliente"].mask(~df["edad_cliente"].between(edad_min, edad_max))
    df["puntaje_datacredito"] = df["puntaje_datacredito"].mask(
        ~df["puntaje_datacredito"].between(puntaje_min, puntaje_max)
    )
    df["salario_cliente"] = df["salario_cliente"].mask(
        (df["salario_cliente"] <= 0) | (df["salario_cliente"] > spec.bounds.salario_maximo)
    )
    df["promedio_ingresos_datacredito"] = df["promedio_ingresos_datacredito"].mask(
        df["promedio_ingresos_datacredito"] <= 0
    )
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(spec.vocabularies.tendencias)
    )
    return df


def drop_unusable_columns(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or default_spec()
    # errors=ignore: a scoring payload has no reason to carry a column we only delete.
    return df.drop(columns=spec.columns.sin_varianza, errors="ignore")


def _codigo(valor: object) -> str:
    """4, 4.0 and "4" are the same product code, whatever the source stores."""
    if pd.isna(valor):
        return ""
    if isinstance(valor, (int, float, np.integer, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor).strip()


def _clave_pago(valor: object) -> str | None:
    if pd.isna(valor):
        return None
    if isinstance(valor, (bool, np.bool_)):
        return "true" if valor else "false"
    if isinstance(valor, (int, float, np.integer, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor).strip().lower()


def normalize_target(serie: pd.Series) -> pd.Series:
    """An unreadable outcome must never default to "paid on time": it would hide a default."""
    normalizado = serie.map(lambda valor: _PAGO_ATIEMPO.get(_clave_pago(valor)))
    if normalizado.isna().any():
        rechazados = sorted({repr(v) for v in serie[normalizado.isna()].unique()})
        raise ValueError(f"Pago_atiempo trae valores que no son booleanos: {', '.join(rechazados)}")
    return normalizado.astype(bool)


def _parse_puntaje(serie: pd.Series) -> pd.Series:
    """The decimal comma only needs undoing when the source hands us text."""
    if serie.dtype == object:
        serie = serie.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v)
    return serie.astype("Float64")


def coerce_types(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or default_spec()
    df = df.copy()
    df[spec.event_timestamp] = pd.to_datetime(df[spec.event_timestamp], dayfirst=True)
    df["puntaje"] = _parse_puntaje(df["puntaje"])
    if spec.target in df:
        df[spec.target] = normalize_target(df[spec.target])
    df[spec.columns.enteros] = df[spec.columns.enteros].round().astype("Int64")
    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = pd.Categorical(
        df["tipo_laboral"], categories=spec.vocabularies.tipos_laboral
    )
    df["tendencia_ingresos"] = pd.Categorical(
        df["tendencia_ingresos"], categories=spec.vocabularies.tendencias, ordered=True
    )
    return df


def scale_bureau_balances(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Stored in thousands by the bureau; a median of 16 178 COP against a 3 000 000 salary
    is not a real balance. Confirmed with the business 2026-09-16."""
    spec = spec or default_spec()
    df = df.copy()
    for columna in spec.units.thousands:
        if columna in df:
            df[columna] = df[columna] * spec.units.factor
    return df


def group_rare_credit_types(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Residual product codes carry no sample; kept apart so they cannot fake a finding."""
    spec = spec or default_spec()
    df = df.copy()
    codigos = df["tipo_credito"].map(_codigo)
    frecuente = codigos.isin(spec.vocabularies.tipos_credito_frecuentes)
    # Fixed categories: otherwise the encoding would depend on which rows are in the batch.
    df["tipo_credito"] = pd.Categorical(
        np.where(frecuente, codigos, spec.vocabularies.tipo_credito_residual),
        categories=spec.vocabularies.tipos_credito,
    )
    return df


def flag_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Marked rather than removed: dropping them would bias the very rate we measure."""
    df = df.copy()
    # Nullable on purpose: an unverifiable salary is not an affordable instalment.
    df["cuota_supera_salario"] = (df["cuota_pactada"] > df["salario_cliente"]).astype("boolean")
    return df


def clean(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or default_spec()
    df = nullify_sentinels(df, spec)
    df = drop_unusable_columns(df, spec)
    df = coerce_types(df, spec)
    df = scale_bureau_balances(df, spec)
    df = group_rare_credit_types(df, spec)
    return flag_inconsistencies(df)
