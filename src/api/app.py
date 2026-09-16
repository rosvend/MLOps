"""Batch scoring API for the champion credit model, plus live drift monitoring.

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Scoring, request logging and drift detection against the training reference. No
retraining trigger, no dashboard beyond Evidently's own HTML export - this stage detects
and alerts, it does not close the loop.
"""

import logging
from functools import partial

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from starlette.concurrency import run_in_threadpool

from src.api.schemas import BatchRequest, BatchResponse, DriftCheckResponse, Health
from src.api.service import feature_store_status, load_champion, load_predictions_log, score
from src.models.training_spec import default_serving_spec
from src.monitoring.live_check import InsufficientCurrentData, load_reference, run_live_drift_check
from src.monitoring.spec import default_monitoring_spec

_log = logging.getLogger(__name__)
app = FastAPI(
    title="Credit scoring - batch",
    description="Scores credit applications with the champion model selected in stage 7, "
    "and checks logged traffic for drift against the training reference.",
    version="1.1.0",
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
        predicciones = score(
            registros,
            champion,
            decision_threshold=request.decision_threshold,
            predictions_log=load_predictions_log(),
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    return BatchResponse(
        model=champion.name,
        threshold=champion.threshold,
        flagged=sum(p["review_flag"] for p in predicciones),
        predictions=predicciones,
    )


@app.post("/monitoring/run-drift-check", response_model=DriftCheckResponse)
async def run_drift_check(background_tasks: BackgroundTasks) -> DriftCheckResponse:
    """Reference vs logged production traffic. No labels exist for live traffic, so
    this reports feature/prediction/operational drift only - never a quality metric it
    has no ground truth to support.

    Genuinely asynchronous: the (CPU-bound) drift computation runs in a thread pool so
    the event loop keeps serving other requests, and the JSON summary is computed and
    returned in this same response. Only the slower HTML dashboard write happens after
    the response, via a background task.
    """
    champion = _champion()
    spec = default_monitoring_spec()

    def _cargar_y_comparar():
        # load_reference (pd.read_parquet) and the drift computation are both blocking
        # I/O/CPU work; both run in the thread pool, not just the heavier comparison -
        # reading the parquet on the event loop thread would still delay other requests.
        referencia = load_reference(spec.reference_path)
        return run_live_drift_check(
            referencia, load_predictions_log(), champion.meta["flagged_share_at_fit"], spec
        )

    try:
        resultado = await run_in_threadpool(_cargar_y_comparar)
    except InsufficientCurrentData as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    background_tasks.add_task(
        partial(resultado.drift.save, spec.live_report_html, spec.live_report_json)
    )
    return DriftCheckResponse(
        dataset_drift_detected=resultado.drift.dataset_drift_detected,
        drift_share=resultado.drift.drift_share,
        drifted_features=resultado.drift.drifted_features,
        flagged_share_current=resultado.flagged_share_current,
        flagged_share_at_fit=resultado.flagged_share_at_fit,
        current_rows=resultado.current_rows,
    )
