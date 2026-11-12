# Postman collections

| File | Target |
| --- | --- |
| `credit_scoring_api.postman_collection.json` | The API running locally (`make serve`, `http://localhost:8000`), no login |
| `scoring_api_cloud.postman_collection.json` + `scoring_api_cloud.postman_environment.json` | The private API on Cloud Run: 10 requests, 26 assertions (basics, validation, security, drift check) |

## Run the cloud collection

1. Get a token (it expires after about an hour): `gcloud auth print-identity-token`
2. In Postman, import the collection (the environment file is optional: the collection has its own `base_url`). If you use the environment, select it in the top-right dropdown, because with **No environment** selected Postman cannot read environment variables and shows `getaddrinfo EAI_AGAIN {{base_url}}`. Select the environment "Scoring API (Cloud Run)" and paste the token into the
   `id_token` variable (or, without an environment, into the collection's Variables tab). Check that `base_url` matches `gcloud run services describe scoring-api --region us-central1 --format='value(status.url)'`.
3. Run the collection from the top. The first request is the warm-up: the service scales to zero, so the first call after
   a quiet period can take about 30 s.

From a terminal, without Postman:

```bash
npx newman run tests/postman/scoring_api_cloud.postman_collection.json \
  -e tests/postman/scoring_api_cloud.postman_environment.json \
  --env-var "id_token=$(gcloud auth print-identity-token)" --timeout-request 90000
```

Your Google account needs the Cloud Run Invoker role on the service (the project owner has it). Every prediction request adds
a row to the BigQuery log that the drift check reads.
