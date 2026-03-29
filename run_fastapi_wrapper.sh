#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"

python -m pip install -r requirements-api.txt
python -m uvicorn fastapi_wrapper:app --host "${HOST}" --port "${PORT}" --app-dir .
