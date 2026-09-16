"""Loading the champion and scoring applications with it.

Feature computation goes through the same `clean` -> `add_derived_features` ->
`features()` code the training pipeline used. That is the whole reason the API takes
raw application fields: shared code cannot drift from itself.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features.cleaning import clean
from src.features.contract import features
from src.features.derive import add_derived_features
from src.features.spec import FeatureSpec, default_spec
from src.models.training_spec import ServingSpec, default_serving_spec
from src.monitoring.predictions_log import PredictionsLog
from src.monitoring.spec import default_monitoring_spec

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]

# Columns clean() reads that an application does not carry. The source system drops the
# codebtor balance entirely and the label does not exist yet at decision time.
_AUSENTES_EN_SERVICIO = {"saldo_mora_codeudor": 0, "puntaje": None}


@dataclass(frozen=True)
class Champion:
    pipeline: object
    meta: dict
    spec: FeatureSpec
    serving: ServingSpec

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def name(self) -> str:
        return str(self.meta["model"])


@lru_cache(maxsize=1)
def load_champion(model_path: str, meta_path: str) -> Champion:
    """Read once per process; the artifact is immutable for the life of the container."""
    modelo = joblib.load(RAIZ / model_path)
    meta = json.loads((RAIZ / meta_path).read_text())
    _log.info("champion %s cargado, umbral %.6f", meta["model"], meta["threshold"])
    return Champion(
        pipeline=modelo, meta=meta, spec=default_spec(), serving=default_serving_spec()
    )


@lru_cache(maxsize=1)
def load_predictions_log() -> PredictionsLog:
    """One connection for the life of the process, same lifecycle as the champion."""
    return PredictionsLog(RAIZ / default_monitoring_spec().predictions_db_path)


def _frame(records: list[dict], champion: Champion) -> pd.DataFrame:
    """Build the frame clean() expects from what the caller actually sent."""
    spec = champion.spec
    ahora = pd.Timestamp.now().floor("s")
    filas = []
    for registro in records:
        fila = dict(registro)
        fila.pop("application_id", None)
        fila[spec.entity_key] = champion.serving.default_entity_id
        fila[spec.event_timestamp] = fila.get(spec.event_timestamp) or ahora
        filas.append({**_AUSENTES_EN_SERVICIO, **fila})
    return pd.DataFrame(filas)


def score(
    records: list[dict],
    champion: Champion,
    decision_threshold: float | None = None,
    predictions_log: PredictionsLog | None = None,
) -> list[dict]:
    """Probability of default and the decision the frozen threshold makes from it.

    decision_threshold is a purely additive what-if: it never changes review_flag, which
    stays decided by the frozen threshold regardless of what is supplied here.
    predictions_log, when given, records each prediction as the "current data" future
    drift checks compare against the training reference.
    """
    crudo = _frame(records, champion)
    preparado = features(add_derived_features(clean(crudo, champion.spec), champion.spec), champion.spec)
    faltan = [c for c in champion.meta["features"] if c not in preparado.columns]
    if faltan:
        raise ValueError(f"la vista de features no trae {faltan}")

    columnas_modelo = preparado[champion.meta["features"]]
    proba = champion.pipeline.predict_proba(columnas_modelo)[:, 1]
    umbral = champion.threshold
    predicciones = [
        {
            "application_id": registro["application_id"],
            "probability_default": float(p),
            # Frozen cut-point: the same applicant gets the same answer alone or in a
            # batch of a thousand. A per-batch quantile would not.
            "review_flag": bool(p >= umbral),
            "threshold": umbral,
            "review_flag_at_custom_threshold": (
                bool(p >= decision_threshold) if decision_threshold is not None else None
            ),
        }
        for registro, p in zip(records, proba)
    ]

    if predictions_log is not None:
        registros_log = []
        for (_, fila), pred in zip(columnas_modelo.iterrows(), predicciones):
            registros_log.append(
                {
                    **fila.to_dict(),
                    "application_id": pred["application_id"],
                    "probability_default": pred["probability_default"],
                    "review_flag": pred["review_flag"],
                }
            )
        predictions_log.append(registros_log)

    return predicciones


def feature_store_status() -> str:
    """Offline registry only. Stage 6 has no online store, and saying otherwise would lie."""
    registro = RAIZ / "feature_repo" / "data" / "registry.db"
    if not registro.exists():
        return "registry not found (run `make feast-apply`)"
    try:
        from feast import FeatureStore

        tienda = FeatureStore(repo_path=str(RAIZ / "feature_repo"))
        return f"offline registry ok ({len(tienda.list_feature_views())} feature views)"
    except ImportError:
        return "feast not bundled in this image"
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return f"registry unreadable: {type(exc).__name__}"
