import pandas as pd
import pandera.pandas as pa

from src.features.spec import default_spec

# Read once at import: pandera declares field constraints as class attributes.
_SPEC = default_spec()
_EDAD_MIN, _EDAD_MAX = _SPEC.bounds.edad
_PUNTAJE_MIN, _PUNTAJE_MAX = _SPEC.bounds.puntaje_bureau


class CreditoFeaturesSchema(pa.DataFrameModel):
    """Contract every downstream stage can rely on; a violation fails here, not silently later.

    Excludes the target on purpose, so scoring never requires knowing the outcome.
    """

    cliente_id: str = pa.Field(str_matches=r"^CLI-\d{7}$", unique=True)
    tipo_credito: pd.CategoricalDtype = pa.Field(isin=_SPEC.vocabularies.tipos_credito)
    fecha_prestamo: pd.Timestamp = pa.Field()
    capital_prestado: int = pa.Field(ge=_SPEC.bounds.capital_minimo)
    plazo_meses: int = pa.Field(ge=1, le=_SPEC.bounds.plazo_maximo_meses)
    edad_cliente: pd.Int64Dtype = pa.Field(ge=_EDAD_MIN, le=_EDAD_MAX, nullable=True)
    tipo_laboral: pd.CategoricalDtype = pa.Field(isin=_SPEC.vocabularies.tipos_laboral)
    salario_cliente: pd.Int64Dtype = pa.Field(gt=0, le=_SPEC.bounds.salario_maximo, nullable=True)
    total_otros_prestamos: int = pa.Field(ge=0)
    cuota_pactada: int = pa.Field(gt=0)
    puntaje: pd.Float64Dtype = pa.Field(nullable=True)
    puntaje_datacredito: pd.Int64Dtype = pa.Field(
        ge=_PUNTAJE_MIN, le=_PUNTAJE_MAX, nullable=True
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
    tendencia_ingresos: pd.CategoricalDtype = pa.Field(isin=_SPEC.vocabularies.tendencias, nullable=True)

    cuota_supera_salario: pd.BooleanDtype = pa.Field(nullable=True)
    dti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    pti: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    monto_sobre_ingreso: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    ratio_ingreso_declarado_bureau: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    creditos_por_anio_adulto: pd.Float64Dtype = pa.Field(ge=0, nullable=True)
    tiene_mora_bureau: pd.BooleanDtype = pa.Field(nullable=True)
    rango_edad: pd.CategoricalDtype = pa.Field(isin=_SPEC.age_bands.labels, nullable=True)
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
        """The indicators and spec.columns.vigiladas must not drift apart."""
        declarados = {c for c in df.columns if c.startswith("falta_")}
        return declarados == {f"falta_{c}" for c in _SPEC.columns.vigiladas}

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
