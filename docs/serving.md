# Batch scoring (stage 8)

The champion selected in stage 7 behind a batch prediction API, containerised.

```
src/pipelines/export_champion.py  make export-champion
src/api/schemas.py                request and response contracts
src/api/service.py                artifact loading, feature computation, scoring
src/api/app.py                    FastAPI app
config/serving/default.yaml       artifact paths, batch limit
docker/Dockerfile                 multi-stage, non-root
docker-compose.yml
```

## The artifact carries its own decision rule

`make export-champion` reads `reports/champion.json`, pulls that MLflow run's pipeline and
writes `models/champion.joblib` (10.6 KB) beside `models/champion.meta.json`:

```json
{ "model": "logistic", "threshold": 0.082282,
  "flagged_share_at_fit": 0.157,
  "calibrated_on": "training window before 2025-06",
  "mlflow_run_id": "...", "git_sha": "592772f", "features": [...] }
```

A probability means nothing without the cut-point that turns it into a decision, and an
artifact nobody can trace to a run has no business in an image. Both are gitignored — they
are build outputs, rebuilt from the recorded run.

## The threshold is frozen, and the flagged share drifts

The cut-point is the 15.7 % quantile **of the training window**, computed once at export.
Applied to the newest vintages it flags **20.66 %**, not 15.7 %.

That is the correct behaviour, not a defect. A per-batch quantile would make one
applicant's decision depend on who else was scored alongside them — the batch-dependence
this pipeline has refused at every stage — and it is undefined for a single record. The
share moving is information about the population, and hiding it by re-quantiling would
destroy the signal along with the guarantee.

## The API takes applications, not features

`POST /predict/batch` accepts raw application fields and runs the **same**
`clean → add_derived_features → features()` code the training pipeline used. A caller
cannot compute `dti` differently from training because the caller does not compute it.

A test pins this: for 50 real loans the API's probabilities equal
`champion.predict_proba` through the training path to within 1e-9. If the serving feature
computation ever drifts, that test fails.

Pydantic validates the request; the pandera `CreditoFeaturesSchema` remains the
batch-data contract. Loosening its entity-id format to admit serving ids would weaken a
check that earns its place elsewhere, so the two layers stay separate — transformation
code shared, validation appropriate to each caller.

```bash
curl -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' -d '{
  "records": [{"application_id":"APP-00002","tipo_credito":"9","capital_prestado":4200000,
    "plazo_meses":24,"edad_cliente":23,"tipo_laboral":"Independiente","salario_cliente":1200000,
    "total_otros_prestamos":3500000,"cuota_pactada":390000,"puntaje_datacredito":680,
    "cant_creditosvigentes":7,"huella_consulta":11,"tendencia_ingresos":"Decreciente"}]}'
```

```json
{"model":"logistic","threshold":0.08228209689721733,"flagged":1,
 "predictions":[{"application_id":"APP-00002","probability_default":0.7258075096667608,
                 "review_flag":true,"threshold":0.08228209689721733}]}
```

Bureau fields are optional: missing bureau data is information the model handles through
the `falta_*` indicators, not a reason to refuse the request.

## `/health` says what is true

```json
{"status":"ok","model_loaded":true,"model_name":"logistic","threshold":0.082282,
 "n_features":35,"git_sha":"592772f","feature_store":"...","online_store":null}
```

`online_store` is `null` because stage 6 has none, by design. A health check that implied
otherwise would be lying about the system it is meant to describe.

`feature_store` reports what the process can actually see. Run locally it reads the
registry (`offline registry ok (4 feature views)`); inside the scoring image it says the
registry is absent, because the image deliberately does not carry Feast — scoring computes
features from the request payload and never retrieves from the store.

## Container

Multi-stage, installed from `uv.lock` with `--frozen`. The builder shares the runtime's
base image and builds the venv at `/app/.venv`, the path it will run from: a virtualenv
records its interpreter's absolute path, so building elsewhere produces one the runtime
cannot execute.

A `serving` dependency group keeps MLflow, XGBoost, LightGBM, Feast and Jupyter out — the
champion is a logistic regression and needs none of them to answer a request. The image is
681 MB, essentially all pandas, numpy, scipy and scikit-learn.

Runs as `scoring` (uid 1001), read-only root filesystem, `/tmp` on tmpfs.

```bash
make export-champion && docker compose up --build -d
docker compose exec scoring-api id     # uid=1001(scoring)
```

## Out of scope

Stage 9: no drift detection, no performance monitoring, no alerting. This stage ends at a
container that scores a batch.
