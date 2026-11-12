#!/usr/bin/env bash
# Usage: scripts/aws_smoke_test.sh <function-url>; signs with the credentials in the environment (SigV4).
set -euo pipefail

url="${1:?usage: aws_smoke_test.sh <function-url>}"
url="${url%/}"
request="$(dirname "$0")/../config/ci/smoke_request.json"
sign=(--aws-sigv4 "aws:amz:${AWS_REGION:-us-east-1}:lambda" --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}")
[ -n "${AWS_SESSION_TOKEN:-}" ] && sign+=(-H "x-amz-security-token: ${AWS_SESSION_TOKEN}")

# A version that has been idle cold-starts: allow time and retry before calling it a failure.
health="$(curl -fsS --max-time 90 --retry 3 --retry-delay 5 --retry-all-errors "${sign[@]}" "${url}/health")"
echo "health: ${health}"
echo "${health}" | jq -e '.status == "ok" and .model_loaded == true' > /dev/null

scored="$(curl -fsS --max-time 90 --retry 2 --retry-delay 5 --retry-all-errors "${sign[@]}" -H 'Content-Type: application/json' \
  -d @"${request}" "${url}/predict/batch")"
echo "predict: ${scored}"
echo "${scored}" | jq -e '.predictions | length == 1 and (.[0].probability_default | . >= 0 and . <= 1)' > /dev/null

echo "smoke test passed"
