# Testing the deployed API

The scoring API on Cloud Run is private, so both test tools need an ID token. Postman checks behaviour and Locust checks load.

| Tool | File | What it checks |
| --- | --- | --- |
| Postman | `tests/postman/scoring_api_cloud.postman_collection.json` | 10 requests, 26 assertions: basic predictions, input validation (422 for bad input), refusal of calls without a token, the drift check |
| Locust | `loadtest/locustfile.py` | Latency, errors and autoscaling under simulated users |

## 1. Postman

The collection passes against the live service. To run it:

1. Get a token, valid for about an hour: `gcloud auth print-identity-token`.
2. Import the collection. The environment file is optional because the collection has its own `base_url`. Paste the token into the `id_token` variable.
3. Check that `base_url` matches `gcloud run services describe scoring-api --region us-central1 --format='value(status.url)'`.
4. Run from the top. The first request is the warm-up, because the first call after a quiet period can take about 30 s.

With **No environment** selected, Postman cannot read environment variables and shows `getaddrinfo EAI_AGAIN {{base_url}}`. Select "Scoring API (Cloud Run)", or put the values in the collection's Variables tab. Your Google account needs the Cloud Run Invoker role on the service. From a terminal:

```bash
npx newman run tests/postman/scoring_api_cloud.postman_collection.json \
  -e tests/postman/scoring_api_cloud.postman_environment.json \
  --env-var "id_token=$(gcloud auth print-identity-token)" --timeout-request 90000
```

## 2. Locust

The simulated users send single applications (8 parts), batches of 20 (2 parts) and health checks (1 part), with 0.5 to 2 s between calls. The file gets an ID token from `gcloud` and refreshes it every 45 minutes.

```bash
export API_URL=$(gcloud run services describe scoring-api --region us-central1 --format='value(status.url)')
export GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval
uv run --group loadtest locust -f loadtest/locustfile.py --host $API_URL \
    -u 20 -r 5 -t 60s --headless --csv outputs/loadtest/run1
```

| Run | Result |
| --- | --- |
| First, before the fix | Errors on 28 % of requests (HTTP 500) |
| Second, after the fix | 0 errors in 480 requests. Median 0.7 s, p95 1.6 s, p99 2.8 s, about 8 requests per second, Cloud Run scaled to both allowed instances (`--max-instances 2`) |

Sequential calls take about 300 ms, so under load latency more than doubles. Every request is logged to BigQuery, so a load test adds synthetic rows that the next drift check will read.

## 3. What the tests found

| Finding | Detail | Action |
| --- | --- | --- |
| Cold start | The first call after idle took about 34 s, because the service scales to zero | Keeping one instance warm would remove it at a standing cost |
| Production bug | An application without an optional field was logged with a `NaN`. BigQuery's JSON type rejected the whole insert and the API returned a 500 | Fixed by writing `null`; a logging failure can no longer fail a prediction. Shipped through the Deploy workflow |
| Logging adds latency | Each prediction writes to BigQuery inside the request | How much of the load-test latency this explains was not measured |
| Test traffic pollutes the log | Synthetic requests are logged like real ones, so the next drift check reports drift | Run the API locally (`make serve`, which logs to SQLite) for load tests that should not touch the log |
