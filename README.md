# NemotronModels

## Run the notebook

From this folder, run:

```powershell
.\run_notebook.ps1 -Launch
```

This script will:

- install the notebook dependencies into a local `.packages` folder
- install `torch`
- launch `nemotron_testing.ipynb`

## Gemma 4 MAX vs vLLM Colab notebook

This repo now also includes a Colab-oriented benchmark notebook for comparing MAX and vLLM on Gemma 4:

- `gemma4_max_vs_vllm_colab.ipynb`
- `build_gemma4_max_vs_vllm_colab_notebook.py`
- `gemma4_colab_benchmark_helper.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrrahman1517/nemotron-vl-embed-stack/blob/codex-gemma4-max-vs-vllm-colab/gemma4_max_vs_vllm_colab.ipynb)

Use this notebook when you want to test the April 2, 2026 Modular blog claim that MAX achieves higher throughput than vLLM on Gemma 4.

Important limitation:

- the published claim was measured on NVIDIA B200
- typical Colab GPUs are L4 or A100
- so a standard Colab run is a directional comparison unless you have access to a B200-class runtime

Recommended starting points:

- L4 24 GiB: `google/gemma-4-E4B-it`, `NUM_PROMPTS = 64`, `MAX_CONCURRENCY = 8`
- A100 40 GiB: `google/gemma-4-E4B-it`, `NUM_PROMPTS = 96` to `128`, `MAX_CONCURRENCY = 8` to `16`
- 80 GiB or B200-class GPU: `google/gemma-4-26B-A4B-it`, `NUM_PROMPTS = 128`, `MAX_CONCURRENCY = 16`

Token setup:

- accept the Gemma 4 model license on Hugging Face first
- add a Colab secret named `HF_TOKEN` if you want a clean notebook run
- if you skip the secret, the notebook will prompt for the token with hidden input

MAX startup troubleshooting:

- current MAX docs list NVIDIA driver `580+` for supported GPU acceleration
- many Colab runtimes use an older NVIDIA driver, which can cause `max serve` to stall or fail
- current MAX docs also note that older NVIDIA drivers can sometimes be bypassed by setting `MODULAR_NVPTX_COMPILER_PATH` to a system `ptxas` binary
- the notebook now does a MAX driver preflight and a `max warm-cache` step before `max serve`
- if you still want to try an older driver anyway, set `FAIL_ON_UNSUPPORTED_MAX_DRIVER = False` in the setup cell

## Notes

- The notebook now falls back to CPU automatically when CUDA is not available.
- The ONNX export is written as an `onnx_export/` bundle in Colab because the model is too large for a single ONNX protobuf file.
- If model download access requires authentication, set `HF_TOKEN` in your environment before launching Jupyter.

## Efficient inference options

### vLLM

For a practical open-source serving path, use vLLM on Linux with NVIDIA GPUs.

The current vLLM docs include a dedicated `LlamaNemotronVLForEmbedding` implementation for this model family, so vLLM is the best self-managed path if you want a fast embedding service.

Files in this repo:

- `serve_vllm_linux.sh`: starts an OpenAI-compatible vLLM embedding server
- `vllm_embed_client.py`: sends text embedding requests to that server
- `docker-compose.yaml`: starts either a vLLM service or the NVIDIA NIM service
- `.env.example`: environment variables for Docker Compose
- `benchmark_embeddings.py`: benchmarks `/v1/embeddings` latency and throughput
- `fastapi_wrapper.py`: adds stable app-facing endpoints on top of either backend
- `run_fastapi_wrapper.sh` and `run_fastapi_wrapper.ps1`: launch the wrapper locally

Typical flow:

```bash
pip install -U vllm
export VLLM_API_KEY=local-dev-key
./serve_vllm_linux.sh
python vllm_embed_client.py
python benchmark_embeddings.py --api-key "$VLLM_API_KEY"
```

Notes:

- This route is best for text embeddings first.
- For image or image+text inputs, keep using the same model but send multimodal requests through vLLM's multimodal embedding APIs.
- vLLM deployment is primarily a Linux/NVIDIA GPU workflow, not a native Windows one.

### TensorRT / NVIDIA deployment

For this exact model, the safest production-oriented TensorRT path is to use NVIDIA's NIM / Triton stack rather than trying to hand-convert the Hugging Face checkpoint yourself.

Why:

- The Hugging Face model card lists TensorRT, Triton, and NeMo Retriever Embedding NIM as the software integration path for this model family.
- NVIDIA's NIM model page for `llama-nemotron-embed-vl-1b-v2` is available now.
- TensorRT-LLM AutoDeploy currently documents support for `AutoModelForCausalLM` and `AutoModelForImageTextToText`, but not pooling-style embedding models like this one.

Recommendation:

- Use vLLM if you want an open-source embedding server you can run directly today.
- Use NVIDIA NIM / Triton if your goal is the most supported TensorRT-backed production deployment path.

Files in this repo:

- `run_nim_embed_vl_linux.sh`: launches the official NVIDIA NIM container for this model
- `vllm_embed_client.py`: can also call the local `/v1/embeddings` endpoint exposed by NIM if you point `VLLM_BASE_URL` at it
- `docker-compose.yaml`: can also launch the NIM container using the `nim` profile
- `benchmark_embeddings.py`: can benchmark the NIM endpoint too

Typical flow:

```bash
export NGC_API_KEY=...
./run_nim_embed_vl_linux.sh
export VLLM_BASE_URL=http://127.0.0.1:8000
unset VLLM_API_KEY
python vllm_embed_client.py
python benchmark_embeddings.py --input-type query
```

## Docker Compose

This repo now includes a Compose file with two profiles:

- `vllm`: launches the official `vllm/vllm-openai` image
- `nim`: launches the official NVIDIA NIM container for this model
- `wrapper`: launches a FastAPI service that fronts either backend

Example:

```bash
cp .env.example .env

# vLLM
docker compose --profile vllm up

# NVIDIA NIM
docker compose --profile nim up

# FastAPI wrapper in front of whichever backend is on port 8000
docker compose --profile wrapper up
```

For vLLM, set `HF_TOKEN` if the model download needs authentication.

For NIM, set `NGC_API_KEY`.

## Benchmarking

You can benchmark either server with:

```bash
python benchmark_embeddings.py
```

Useful options:

- `--base-url http://127.0.0.1:8000`
- `--api-key local-dev-key`
- `--input-type query`
- `--batch-size 8`
- `--requests 30`
- `--warmup 3`

## FastAPI wrapper

The wrapper gives you cleaner application-facing routes while still using the OpenAI-compatible backend under the hood.

Endpoints:

- `GET /healthz`
- `GET /config`
- `POST /embed/query`
- `POST /embed/document`
- `POST /embed/text`
- `POST /v1/embeddings`

Local run:

```bash
python -m pip install -r requirements-api.txt
python -m uvicorn fastapi_wrapper:app --host 127.0.0.1 --port 8010
```

Example request:

```bash
curl -X POST http://127.0.0.1:8010/embed/query \
  -H "Content-Type: application/json" \
  -d '{"texts":["How is AI improving robotics?"]}'
```

If a backend rejects `input_type`, the wrapper can automatically retry without it when `ALLOW_INPUT_TYPE_FALLBACK=true`.
