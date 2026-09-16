"""Batch scoring API for the champion credit model.

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Scoring only. Drift detection, performance monitoring and alerting are stage 9 and are
deliberately absent.
"""

import logging

from fastapi import FastAPI, HTTPException, status

from src.api.schemas import BatchRequest, BatchResponse, Health
from src.api.service import feature_store_status, load_champion, score
from src.models.training_spec import default_serving_spec

_log = logging.getLogger(__name__)
app = FastAPI(
    title="Credit scoring - batch",
    description="Scores credit applications with the champion model selected in stage 7.",
    version="1.0.0",
)


def _champion():
    serving = default_serving_spec()
    return load_champion(serving.model_path, serving.meta_path)


@app.get("/health", response_model=Health)
def health() -> Health:
    """Reports what is actually true, including that there is no online store."""
    try:
        champion = _champion()
    except Exception as exc:  # noqa: BLE001 - an unloadable model is a health answer
        _log.warning("champion no disponible: %s", exc)
        return Health(
            status="degraded", model_loaded=False, feature_store=feature_store_status()
        )
    return Health(
        status="ok",
        model_loaded=True,
        model_name=champion.name,
        threshold=champion.threshold,
        n_features=len(champion.meta["features"]),
        git_sha=champion.meta.get("git_sha"),
        feature_store=feature_store_status(),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    # The batch size is bounded by BatchRequest itself, so an oversized request is
    # refused during validation rather than after every record has been parsed.
    champion = _champion()
    registros = [r.model_dump() for r in request.records]
    try:
        predicciones = score(registros, champion)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    return BatchResponse(
        model=champion.name,
        threshold=champion.threshold,
        flagged=sum(p["review_flag"] for p in predicciones),
        predictions=predicciones,
    )
