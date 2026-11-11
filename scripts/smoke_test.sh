#!/usr/bin/env bash
# Usage: ID_TOKEN=<jwt> scripts/smoke_test.sh <base-url>; the token is optional for a local API.
set -euo pipefail

url="${1:?usage: smoke_test.sh <base-url>}"
request="$(dirname "$0")/../config/ci/smoke_request.json"
auth=()
[ -n "${ID_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${ID_TOKEN}")

# A revision that has been idle must cold-start: allow time and retry before calling it a failure.
health="$(curl -fsS --max-time 90 --retry 3 --retry-delay 5 --retry-all-errors "${auth[@]}" "${url}/health")"
echo "health: ${health}"
echo "${health}" | jq -e '.status == "ok" and .model_loaded == true' > /dev/null

scored="$(curl -fsS --max-time 90 --retry 2 --retry-delay 5 --retry-all-errors "${auth[@]}" -H 'Content-Type: application/json' \
  -d @"${request}" "${url}/predict/batch")"
echo "predict: ${scored}"
echo "${scored}" | jq -e '.predictions | length == 1 and (.[0].probability_default | . >= 0 and . <= 1)' > /dev/null

echo "smoke test passed: ${url}"
