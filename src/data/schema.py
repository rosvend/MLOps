import pandas as pd
import pandera.pandas as pa

from src.features.derive import COLUMNAS_VIGILADAS, RANGO_EDAD_LABELS
from src.features.cleaning import (
    EDAD_ADULTA,
    EDAD_MAXIMA,
    PUNTAJE_BUREAU_MAX,
    PUNTAJE_BUREAU_MIN,
    SALARIO_MAXIMO,
    TENDENCIAS,
    TIPOS_CREDITO,
    TIPOS_LABORAL,
)

CAPITAL_MINIMO = 360_000
PLAZO_MAXIMO_MESES = 90


class CreditoFeaturesSchema(pa.DataFrameModel):
    """Contract every downstream stage can rely on; a violation fails here, not silently later.

    Excludes the target on purpose, so scoring never requires knowing the outcome.
    """

    tipo_credito: pd.CategoricalDtype = pa.Field(isin=TIPOS_CREDITO)
    fecha_prestamo: pd.Timestamp = pa.Field()
    capital_prestado: int = pa.Field(ge=CAPITAL_MINIMO)
    plazo_meses: int = pa.Field(ge=1, le=PLAZO_MAXIMO_MESES)
    edad_cliente: pd.Int64Dtype = pa.Field(ge=EDAD_ADULTA, le=EDAD_MAXIMA, nullable=True)
    tipo_laboral: pd.CategoricalDtype = pa.Field(isin=TIPOS_LABORAL)
    salario_cliente: pd.Int64Dtype = pa.Field(gt=0, le=SALARIO_MAXIMO, nullable=True)
    total_otros_prestamos: int = pa.Field(ge=0)
    cuota_pactada: int = pa.Field(gt=0)
    puntaje: pd.Float64Dtype = pa.Field(nullable=True)
    puntaje_datacredito: pd.Int64Dtype = pa.Field(
        ge=PUNTAJE_BUREAU_MIN, le=PUNTAJE_BUREAU_MAX, nullable=True
    )
    cant_creditosvigentes: int = pa.Field(ge=0)
    huella_consulta: int = pa.Field(ge=0)
    saldo_mora: pd.Int64Dtype = pa.Field(ge=0, nullable=True)
    saldo_total: pd.Int64Dtype = pa.Field(ge=0, nullable=True)
    saldo_principal: pd.Int64Dtype = pa.Field(ge=0, nullable=True)
    creditos_sectorFinanciero: int = pa.Field(ge=0)
    creditos_sectorCooperativo: int = pa.Field(ge=0)
    creditos_sectorReal: int = pa.Field(ge=0)
    promedio_ingresos_datacredito: pd.Int64Dtype = pa.Field(gt=0, nullable=True)
    tendencia_ingresos: pd.CategoricalDtype = pa.Field(isin=TENDENCIAS, nullable=True)

    cuota_supera_salario: pd.BooleanDtype = pa.Field(nullable=True)
    dti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    pti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    monto_sobre_ingreso: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    ratio_ingreso_declarado_bureau: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    creditos_por_anio_adulto: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    tiene_mora_bureau: pd.BooleanDtype = pa.Field(nullable=True)
    rango_edad: pd.CategoricalDtype = pa.Field(isin=RANGO_EDAD_LABELS, nullable=True)
    mes_prestamo: str = pa.Field(str_matches=r"^\d{4}-\d{2}$")

    falta_edad_cliente: bool = pa.Field()
    falta_salario_cliente: bool = pa.Field()
    falta_puntaje_datacredito: bool = pa.Field()
    falta_promedio_ingresos_datacredito: bool = pa.Field()
    falta_tendencia_ingresos: bool = pa.Field()
    falta_saldo_total: bool = pa.Field()
    falta_saldo_principal: bool = pa.Field()
    falta_saldo_mora: bool = pa.Field()

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def indicadores_cubren_las_columnas_vigiladas(cls, df: pd.DataFrame) -> bool:
        """The indicators and COLUMNAS_VIGILADAS must not drift apart."""
        declarados = {c for c in df.columns if c.startswith("falta_")}
        return declarados == {f"falta_{c}" for c in COLUMNAS_VIGILADAS}

    @pa.dataframe_check
    def saldo_principal_no_supera_total(cls, df: pd.DataFrame) -> pd.Series:
        return (df["saldo_principal"] <= df["saldo_total"]).fillna(True)

    @pa.dataframe_check
    def fecha_prestamo_no_es_futura(cls, df: pd.DataFrame) -> pd.Series:
        # Compare dates, not instants: a loan disbursed at 14:40 today is not in the future.
        return df["fecha_prestamo"].dt.normalize() <= pd.Timestamp.today().normalize()


class CreditoLabelledSchema(CreditoFeaturesSchema):
    """The features plus the outcome; required to train or evaluate, never to score."""

    Pago_atiempo: bool = pa.Field()
