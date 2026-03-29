#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-nvidia/llama-nemotron-embed-vl-1b-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-dev-key}"
DTYPE="${DTYPE:-bfloat16}"

echo "Starting vLLM embedding server for ${MODEL} on ${HOST}:${PORT}"

vllm serve "${MODEL}" \
  --task embed \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}"
