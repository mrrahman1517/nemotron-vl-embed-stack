#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "Set NGC_API_KEY before running this script."
  exit 1
fi

IMAGE="${IMAGE:-nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0}"
PORT="${PORT:-8000}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"

mkdir -p "${LOCAL_NIM_CACHE}"

echo "Starting NVIDIA NIM ${IMAGE} on port ${PORT}"

docker run -it --rm \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p "${PORT}:8000" \
  "${IMAGE}"
