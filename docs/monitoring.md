# Monitoring, drift and alerting (stage 9)

Drift detection against the training baseline, request logging, an alerting hook, and
tooling to exercise the API from outside. No retraining trigger and no dashboard beyond
Evidently's own HTML export - this stage detects and alerts, it does not close the loop.

```
src/monitoring/spec.py            MonitoringSpec, config/monitoring/default.yaml
src/monitoring/drift_detector.py  the Evidently wrapper
src/monitoring/predictions_log.py SQLite (WAL) log of every scored request
src/monitoring/alerts.py          always logs, optionally POSTs a webhook
src/monitoring/live_check.py      reference vs logged production traffic
src/pipelines/export_reference.py make export-reference
src/pipelines/monitor.py          make monitor (offline, labelled)
tests/postman/                    Postman v2.1 collection
scripts/setup_tunnel.sh           make tunnel
```

## Two report modes, on purpose

A credit application's outcome is not known at scoring time - that is the entire premise
of this project. So "model quality drift" can only be reported where labels exist:

**`make monitor` (offline)** compares the stage-7 training window against the held-out
test window - both labelled. It is the only place in the project that can honestly report
classification-quality drift (Gini, PR-AUC) alongside feature and prediction drift.
Writes `reports/evidently_drift_report.html` and `reports/evidently_metrics.json`.

**`POST /monitoring/run-drift-check` (live)** compares the same training reference
against whatever `/predict/batch` has logged so far. No labels, so no quality metric -
feature, prediction and operational drift only, and the response never claims otherwise.

## Reference data

`make export-reference` (after `make export-champion`) writes
`models/reference_data.parquet`: the training window's engineered features, the
champion's own predictions on them, and the label. Reference is the training
distribution - the textbook definition of a drift baseline, and the one that cannot
silently include the future.

## Statistical methods, and why two dtype bugs are worth knowing about

Numeric columns use KS below `numeric_sample_size_cutoff` reference rows (an exact
p-value test, suited to small samples) and Wasserstein at or above it (a distance test
that scales better on large samples, where KS becomes oversensitive to trivial
differences). Categorical columns use chi-square. `alpha` (0.05) applies to the p-value
methods only - Wasserstein has its own scale-relative threshold, not a p-value.

`drift_share` and `drifted_features` are derived from the same per-column tests, not
from a separately-configured `DriftedColumnsCount` metric - that metric picks its own
per-column significance internally, and it disagreed with the per-column tests
configured with `alpha` during development (6/35 drifted columns by one accounting,
20/35 by the other, on the identical two frames). One source of truth now.

Pandas' nullable extension dtypes (`Int64`, `Float64`, `boolean`, Feast's
`string[python]`) are not what Evidently's column-type inference expects, and a column
entirely null in a live-logged sample comes back `object` dtype regardless of what it is
in the reference. Both frames are coerced to the type the reference's own classification
promised before Evidently ever sees them; a column with genuinely no data on either side
is skipped and logged rather than raising for every other column along with it. Both bugs
were found by running the containerised endpoint against real traffic, not by synthetic
tests - reported honestly here because a synthetic-only test suite would not have caught
either.

## Request logging

Every `/predict/batch` call logs each prediction's engineered features, probability and
decision to `data/monitoring/predictions.db` - SQLite in WAL mode, safe under concurrent
requests without a database server. Values are stored as JSON per row rather than fixed
columns, because a request can legitimately omit optional bureau fields and a fixed
schema would force a null for whatever any single request happened to leave out.

## Alerting

Always logs at `WARNING`. If `monitoring.webhook_url` is set, also POSTs a body carrying
both `{"text": ...}` (Slack) and `{"content": ...}` (Discord) - unknown keys are ignored
by both, so one body works for either without the config needing to name which service is
listening. A webhook failure is caught and logged, never raised: an alert about a broken
system must not itself break the request that triggered it. No credentials exist in this
environment, so it is unset by default - point it at a real incoming webhook URL to wire
it up.

Fires on two conditions, independently: `dataset_drift_detected` from the drift check, or
the logged `review_flag` share exceeding `flagged_share_threshold` - a plain proportion
against the champion's `flagged_share_at_fit`, not routed through Evidently at all, since
it is a business threshold rather than a distributional test.

## The optional `decision_threshold` override

`POST /predict/batch` accepts an optional top-level `decision_threshold`. It is purely
additive: the frozen `review_flag` - the same applicant gets the same answer alone or in
any batch, the guarantee stage 8 established - never moves. An extra
`review_flag_at_custom_threshold` field reports what the decision would be at the
supplied cut-point, useful for capacity-planning "what if we moved the bar" queries.
Pinned by a test that scores one applicant with and without an override and asserts the
frozen fields stay bit-identical.

## Running it

```bash
make export-champion export-reference
make monitor                              # offline: train vs test, with model quality
docker compose up --build                 # live: logs /predict/batch, serves drift check
```

`data/monitoring/` is bind-mounted, read-write, into the otherwise read-only, non-root
container - the one write path the live endpoint needs. **On a fresh clone it must be
group/world-writable** (`chmod 777 data/monitoring`) before `docker compose up`: the
container runs as `scoring` (uid 1001), a different user from whoever owns the directory
on the host, and git does not track directory permissions.

`make tunnel` (`ngrok http 8000`, guarded on the API already being up) exposes the local
port publicly for testing from Postman or a webhook - loopback-only by default in
`docker-compose.yml`, since the endpoint has no authentication.

## Postman collection

`tests/postman/credit_scoring_api.postman_collection.json` - v2.1, four requests
(`GET /health`, `POST /predict/batch` default and with a `decision_threshold` override,
`POST /monitoring/run-drift-check`), each with a test script. Run against the live
container:

```bash
npx newman run tests/postman/credit_scoring_api.postman_collection.json
```

## Known limitations

- One held-out window backs `make monitor`; rolling-origin evaluation over several
  cut-points would be a stronger basis, as noted in `docs/model-training.md`.
- The drift-check endpoint reports feature and prediction drift honestly, but with only a
  handful of logged requests the statistics are noisy - `min_current_rows` (30 by
  default) is a floor, not a guarantee of a stable estimate.
- No retraining trigger. Detecting and alerting on drift is this stage's job; deciding
  what to do about it is a human's, for now.
