# Load test with Locust

Simulates users calling the scoring API on Cloud Run: 8 parts single applications, 2 parts batches of 20, 1 part
health checks, with 0.5 to 2 s of think time between calls.

```bash
export API_URL=https://scoring-api-o4j26oozvq-uc.a.run.app      # the Cloud Run URL (gcloud run services describe scoring-api)
export GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval

# With the web UI at http://localhost:8089
uv run --group loadtest locust -f loadtest/locustfile.py --host $API_URL

# Headless: 20 users, 5 new users per second, 60 seconds, results in CSV files
uv run --group loadtest locust -f loadtest/locustfile.py --host $API_URL \
    -u 20 -r 5 -t 60s --headless --csv outputs/loadtest/run1
```

The service is private, so every request carries an ID token. The file gets one from `gcloud auth print-identity-token`
and refreshes it every 45 minutes; set `ID_TOKEN` yourself to use a fixed one (it expires after about an hour).

What to expect and what to watch:

- **Cold start.** The service scales to zero, so the first request after a quiet period takes about 30 s.
- **Autoscaling.** Cloud Run adds instances as load grows, up to `--max-instances 2` in our deploy.
- **Side effect.** Every prediction is logged to BigQuery, so a load test adds rows to the "current data" that the drift check
  compares with the training reference. Synthetic traffic will normally look different from the training loans and
  trigger a drift alert. To avoid it, run the API locally (`make serve`, which logs to SQLite) and point `--host` at it.
- **Canary.** Start a load test, then run the Deploy workflow; failures during the traffic shift would show up as
  errors in the Locust charts.
