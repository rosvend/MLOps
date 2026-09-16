import numpy as np
import pandas as pd

TENDENCIAS = ["Decreciente", "Estable", "Creciente"]
TIPOS_CREDITO_FRECUENTES = ["4", "9", "10"]
TIPOS_CREDITO = [*TIPOS_CREDITO_FRECUENTES, "Otro"]
TIPOS_LABORAL = ["Empleado", "Independiente"]
EDAD_ADULTA = 18
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
_PAGO_ATIEMPO = {"1": True, "0": False, "true": True, "false": False}


def nullify_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Values the source system writes to mean "unknown" but that read as valid numbers."""
    df = df.copy()
    df["edad_cliente"] = df["edad_cliente"].mask(
        ~df["edad_cliente"].between(EDAD_ADULTA, EDAD_MAXIMA)
    )
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
    # errors=ignore: a scoring payload has no reason to carry a column we only delete.
    return df.drop(columns=_COLUMNAS_SIN_VARIANZA, errors="ignore")


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


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], dayfirst=True)
    df["puntaje"] = _parse_puntaje(df["puntaje"])
    if "Pago_atiempo" in df:
        df["Pago_atiempo"] = normalize_target(df["Pago_atiempo"])
    df[_ENTEROS] = df[_ENTEROS].round().astype("Int64")
    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = pd.Categorical(df["tipo_laboral"], categories=TIPOS_LABORAL)
    df["tendencia_ingresos"] = pd.Categorical(df["tendencia_ingresos"], categories=TENDENCIAS, ordered=True)
    return df


def group_rare_credit_types(df: pd.DataFrame) -> pd.DataFrame:
    """Residual product codes carry no sample; kept apart so they cannot fake a finding."""
    df = df.copy()
    codigos = df["tipo_credito"].map(_codigo)
    # Fixed categories: otherwise the encoding would depend on which rows are in the batch.
    df["tipo_credito"] = pd.Categorical(
        np.where(codigos.isin(TIPOS_CREDITO_FRECUENTES), codigos, "Otro"), categories=TIPOS_CREDITO
    )
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
