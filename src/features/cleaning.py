import numpy as np
import pandas as pd

TENDENCIAS = ["Decreciente", "Estable", "Creciente"]
TIPOS_CREDITO_FRECUENTES = [4, 9, 10]
EDAD_MAXIMA = 100
PUNTAJE_BUREAU_MIN, PUNTAJE_BUREAU_MAX = 150, 950
SALARIO_MAXIMO = 1_000_000_000

_COLUMNAS_SIN_VARIANZA = ["saldo_mora_codeudor"]
_ENTEROS = [
    "edad_cliente",
    "salario_cliente",
    "puntaje_datacredito",
    "promedio_ingresos_datacredito",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
]


def nullify_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Values the source system writes to mean "unknown" but that read as valid numbers."""
    df = df.copy()
    df["edad_cliente"] = df["edad_cliente"].mask(df["edad_cliente"] > EDAD_MAXIMA)
    df["puntaje_datacredito"] = df["puntaje_datacredito"].mask(
        ~df["puntaje_datacredito"].between(PUNTAJE_BUREAU_MIN, PUNTAJE_BUREAU_MAX)
    )
    df["salario_cliente"] = df["salario_cliente"].mask(
        (df["salario_cliente"] <= 0) | (df["salario_cliente"] > SALARIO_MAXIMO)
    )
    df["promedio_ingresos_datacredito"] = df["promedio_ingresos_datacredito"].mask(
        df["promedio_ingresos_datacredito"] <= 0
    )
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(df["tendencia_ingresos"].isin(TENDENCIAS))
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=_COLUMNAS_SIN_VARIANZA)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], dayfirst=True)
    df["puntaje"] = df["puntaje"].str.replace(",", ".").astype("Float64")
    df["Pago_atiempo"] = df["Pago_atiempo"].astype(bool)
    df[_ENTEROS] = df[_ENTEROS].round().astype("Int64")
    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = df["tipo_laboral"].astype("category")
    df["tendencia_ingresos"] = pd.Categorical(df["tendencia_ingresos"], categories=TENDENCIAS, ordered=True)
    return df


def group_rare_credit_types(df: pd.DataFrame) -> pd.DataFrame:
    """Residual product codes carry no sample; kept apart so they cannot fake a finding."""
    df = df.copy()
    frecuente = df["tipo_credito"].isin(TIPOS_CREDITO_FRECUENTES)
    df["tipo_credito"] = pd.Categorical(np.where(frecuente, df["tipo_credito"].astype(str), "Otro"))
    return df


def flag_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Marked rather than removed: dropping them would bias the very rate we measure."""
    df = df.copy()
    df["cuota_supera_salario"] = (df["cuota_pactada"] > df["salario_cliente"]).fillna(False).astype(bool)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = nullify_sentinels(df)
    df = drop_unusable_columns(df)
    df = coerce_types(df)
    df = group_rare_credit_types(df)
    return flag_inconsistencies(df)
