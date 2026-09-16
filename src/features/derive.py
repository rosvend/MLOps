"""Derived features: ratios, bureau contrasts, bands and missingness indicators.

Every one is a pure function of a single row, so a loan derives the same values alone
as it does inside a portfolio. Nothing here is fitted; that is src/features/engineering.py.
"""

import pandas as pd

from src.features.spec import FeatureSpec, default_spec


def add_affordability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """What decides credit risk is the proportion of income committed, not the absolute amount."""
    df = df.copy()
    salario = df["salario_cliente"]
    df["dti"] = (df["total_otros_prestamos"] / salario).astype("Float64")
    df["pti"] = (df["cuota_pactada"] / salario).astype("Float64")
    df["monto_sobre_ingreso"] = (df["capital_prestado"] / salario).astype("Float64")
    return df


def add_bureau_contrast(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Contrasts what the client declares against what the bureau observes."""
    spec = spec or default_spec()
    edad_adulta = spec.bounds.edad[0]
    df = df.copy()
    anios_adulto = (df["edad_cliente"] - edad_adulta).replace(0, pd.NA)
    df["ratio_ingreso_declarado_bureau"] = (
        df["salario_cliente"] / df["promedio_ingresos_datacredito"]
    ).astype("Float64")
    df["creditos_por_anio_adulto"] = (df["cant_creditosvigentes"] / anios_adulto).astype("Float64")
    df["tiene_mora_bureau"] = (df["saldo_mora"] > 0).astype("boolean")
    return df


def add_bands(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or default_spec()
    df = df.copy()
    df["rango_edad"] = pd.cut(
        df["edad_cliente"], bins=spec.age_bands.bins, labels=spec.age_bands.labels
    )
    df["mes_prestamo"] = df[spec.event_timestamp].dt.to_period("M").astype(str)
    return df


def add_missingness_flags(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Absence is information, so record it rather than letting an imputer bury it."""
    spec = spec or default_spec()
    df = df.copy()
    for columna in spec.columns.vigiladas:
        df[f"falta_{columna}"] = df[columna].isna().to_numpy(dtype=bool)
    return df


def add_derived_features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or default_spec()
    df = add_affordability_ratios(df)
    df = add_bureau_contrast(df, spec)
    df = add_bands(df, spec)
    return add_missingness_flags(df, spec)
