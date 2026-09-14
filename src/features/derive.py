import pandas as pd

EDAD_ADULTA = 18
RANGO_EDAD_BINS = [17, 25, 35, 45, 55, 65, 100]
RANGO_EDAD_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def add_affordability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """What decides credit risk is the proportion of income committed, not the absolute amount."""
    df = df.copy()
    salario = df["salario_cliente"]
    df["dti"] = (df["total_otros_prestamos"] / salario).astype("Float64")
    df["pti"] = (df["cuota_pactada"] / salario).astype("Float64")
    df["monto_sobre_ingreso"] = (df["capital_prestado"] / salario).astype("Float64")
    return df


def add_bureau_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Contrasts what the client declares against what the bureau observes."""
    df = df.copy()
    anios_adulto = (df["edad_cliente"] - EDAD_ADULTA).replace(0, pd.NA)
    df["ratio_ingreso_declarado_bureau"] = (
        df["salario_cliente"] / df["promedio_ingresos_datacredito"]
    ).astype("Float64")
    df["creditos_por_anio_adulto"] = (df["cant_creditosvigentes"] / anios_adulto).astype("Float64")
    df["tiene_mora_bureau"] = (df["saldo_mora"] > 0).fillna(False).astype(bool)
    return df


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rango_edad"] = pd.cut(df["edad_cliente"], bins=RANGO_EDAD_BINS, labels=RANGO_EDAD_LABELS)
    df["mes_prestamo"] = df["fecha_prestamo"].dt.to_period("M").astype(str)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_affordability_ratios(df)
    df = add_bureau_contrast(df)
    return add_bands(df)
