# Prediction log in BigQuery

Every `/predict/batch` call records what it scored; the drift check later reads those
rows back. Locally that is a SQLite file. On Cloud Run it is a BigQuery table.

## Why not SQLite or Cloud SQL

- **SQLite on Cloud Run fails.** The container user cannot create `/app/data`
  (`PermissionError`), and even with a writable path the disk is ephemeral and private to
  each instance: rows vanish on restart and instances never see each other's rows.
- **Cloud SQL** works but is a server billed by the hour, with networking and a password
  to manage, for a log that is only appended to and read in bulk.
- **BigQuery** is serverless, durable, shared by all instances, needs no secret (the
  runtime service account is enough) and suits append-then-read-everything.

## How the implementation is chosen

`config/monitoring/default.yaml` has a `predictions_log` section:

```yaml
predictions_log:
  type: sqlite            # sqlite | bigquery
  sqlite_path: data/monitoring/predictions.db
  bigquery_dataset: mlops_monitoring
  bigquery_table: predictions
```

`type` is `sqlite` by default, so `make serve` and `docker compose` are unchanged. Cloud
Run sets the environment variable `PREDICTIONS_LOG_TYPE=bigquery`, which overrides `type`
without editing code or the image. Both implementations satisfy the `PredictionsStore`
protocol (`append`, `read_all`) in `src/monitoring/predictions_log.py`. The BigQuery
client uses the project of the runtime credentials.

Deploy flag (done by the deploy step, not here):
`gcloud run services update scoring-api --region us-central1 --update-env-vars PREDICTIONS_LOG_TYPE=bigquery`

## Schema

Table `mlops-roysandoval.mlops_monitoring.predictions` (dataset in `us-central1`):

| column | type | meaning |
|---|---|---|
| `scored_at` | TIMESTAMP, required | when the request was scored (UTC) |
| `payload` | JSON, required | the record: features, probability, decision |

The payload is JSON for the same reason as in SQLite: a request may omit optional fields.
`read_all()` returns the same DataFrame as the SQLite log (`scored_at` as an ISO string
plus one column per payload key; an empty frame when there are no rows).

## Infrastructure

`infra/monitoring.tf` creates the API, dataset and table, and grants the runtime account
`roles/bigquery.dataEditor` on that dataset only plus `roles/bigquery.jobUser` on the
project (needed to run the read query; it grants no data access by itself). The table has
`deletion_protection`.

Commands run:

```bash
export GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval
cd infra && tofu plan -out=plan.out && tofu apply plan.out   # 5 to add, 0 changed, 0 destroyed
```

A live smoke test created a throwaway table `smoke_tmp`, appended one row, read it back
through `BigQueryPredictionsLog`, and deleted the table; `predictions` was never written.

## Cost

An estimate, not a bill: a few thousand small rows per month is far inside the BigQuery
free tier (10 GB storage and 1 TB of queries per month). Streaming inserts are billed per
volume (about 0.01 USD per 200 MB), which at this size rounds to cents or less.
