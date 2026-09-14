import pandas as pd
import pandera.pandas as pa

from src.features.cleaning import (
    PUNTAJE_BUREAU_MAX,
    PUNTAJE_BUREAU_MIN,
    SALARIO_MAXIMO,
    TENDENCIAS,
)
from src.features.derive import EDAD_ADULTA

CAPITAL_MINIMO = 360_000
PLAZO_MAXIMO_MESES = 90
EDAD_MAXIMA_VALIDA = 100


class CreditoSchema(pa.DataFrameModel):
    """Contract every downstream stage can rely on; a violation fails here, not silently later."""

    tipo_credito: pd.CategoricalDtype = pa.Field(isin=["4", "9", "10", "Otro"])
    fecha_prestamo: pd.Timestamp = pa.Field()
    capital_prestado: int = pa.Field(ge=CAPITAL_MINIMO)
    plazo_meses: int = pa.Field(ge=1, le=PLAZO_MAXIMO_MESES)
    edad_cliente: pd.Int64Dtype = pa.Field(ge=EDAD_ADULTA, le=EDAD_MAXIMA_VALIDA, nullable=True)
    tipo_laboral: pd.CategoricalDtype = pa.Field(isin=["Empleado", "Independiente"])
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
    Pago_atiempo: bool = pa.Field()

    cuota_supera_salario: bool = pa.Field()
    dti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    pti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    monto_sobre_ingreso: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    ratio_ingreso_declarado_bureau: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    creditos_por_anio_adulto: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    tiene_mora_bureau: bool = pa.Field()
    rango_edad: pd.CategoricalDtype = pa.Field(nullable=True)
    mes_prestamo: str = pa.Field(str_matches=r"^\d{4}-\d{2}$")

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def saldo_principal_no_supera_total(cls, df: pd.DataFrame) -> pd.Series:
        return (df["saldo_principal"] <= df["saldo_total"]).fillna(True)

    @pa.dataframe_check
    def fecha_prestamo_no_es_futura(cls, df: pd.DataFrame) -> pd.Series:
        return df["fecha_prestamo"] <= pd.Timestamp.today()
