#!/usr/bin/env bash
# Public URL for the local scoring API, so Postman (or a reviewer) can reach it.
set -euo pipefail

if ! command -v ngrok > /dev/null 2>&1; then
    echo "ngrok is not on PATH - install it from https://ngrok.com/download, then run 'ngrok config add-authtoken <token>'" >&2
    exit 1
fi

# Tunnelling to a dead port publishes a 502, not an API.
if ! curl -sf --connect-timeout 5 --max-time 10 http://localhost:8000/health > /dev/null; then
    echo "API is not running on :8000 - start it first with 'make serve' or 'docker compose up'" >&2
    exit 1
fi

exec ngrok http 8000
