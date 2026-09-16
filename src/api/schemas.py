"""Request and response contracts for the scoring API.

The caller sends an application, not features. Everything the model needs is derived
server-side by the same code that trained it, so a client cannot compute `dti`
differently from training - it does not compute it at all.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Bounds mirror the sentinels src/features/cleaning.py applies, so a value the pipeline
# would silently null is refused at the edge instead, where the caller can see it.
Dinero = Annotated[int, Field(ge=0, le=1_000_000_000_000)]
Conteo = Annotated[int, Field(ge=0, le=10_000)]


class ApplicationRecord(BaseModel):
    """One credit application, as the originating system holds it."""

    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(min_length=1, max_length=64)
    fecha_prestamo: datetime | None = Field(
        default=None, description="Decision moment; defaults to now when omitted."
    )

    tipo_credito: str = Field(min_length=1, max_length=16)
    capital_prestado: Dinero
    plazo_meses: Annotated[int, Field(ge=1, le=600)]
    edad_cliente: Annotated[int, Field(ge=18, le=100)] | None = None
    tipo_laboral: Literal["Empleado", "Independiente"]
    salario_cliente: Dinero | None = None
    total_otros_prestamos: Dinero = 0
    cuota_pactada: Dinero

    puntaje_datacredito: Annotated[int, Field(ge=0, le=1000)] | None = None
    cant_creditosvigentes: Conteo = 0
    huella_consulta: Conteo = 0
    saldo_mora: Dinero | None = None
    saldo_total: Dinero | None = None
    saldo_principal: Dinero | None = None
    creditos_sectorFinanciero: Conteo = 0
    creditos_sectorCooperativo: Conteo = 0
    creditos_sectorReal: Conteo = 0
    promedio_ingresos_datacredito: Dinero | None = None
    tendencia_ingresos: Literal["Decreciente", "Estable", "Creciente"] | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ApplicationRecord] = Field(min_length=1)


class Prediction(BaseModel):
    application_id: str
    probability_default: float = Field(ge=0.0, le=1.0)
    review_flag: bool
    threshold: float


class BatchResponse(BaseModel):
    model: str
    threshold: float
    flagged: int
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None
    n_features: int | None = None
    git_sha: str | None = None
    feature_store: str
    online_store: None = None
