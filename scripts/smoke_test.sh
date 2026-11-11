#!/usr/bin/env bash
# Usage: ID_TOKEN=<jwt> scripts/smoke_test.sh <base-url>; the token is optional for a local API.
set -euo pipefail

url="${1:?usage: smoke_test.sh <base-url>}"
request="$(dirname "$0")/../config/ci/smoke_request.json"
auth=()
[ -n "${ID_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${ID_TOKEN}")

health="$(curl -fsS --max-time 30 "${auth[@]}" "${url}/health")"
echo "health: ${health}"
echo "${health}" | jq -e '.status == "ok" and .model_loaded == true' > /dev/null

scored="$(curl -fsS --max-time 60 "${auth[@]}" -H 'Content-Type: application/json' \
  -d @"${request}" "${url}/predict/batch")"
echo "predict: ${scored}"
echo "${scored}" | jq -e '.predictions | length == 1 and (.[0].probability_default | . >= 0 and . <= 1)' > /dev/null

echo "smoke test passed: ${url}"
