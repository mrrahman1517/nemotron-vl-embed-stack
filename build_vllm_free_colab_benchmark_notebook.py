from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HELPER_PATH = REPO_ROOT / "vllm_free_colab_benchmark_helper.py"
NOTEBOOK_PATH = REPO_ROOT / "vllm_free_colab_benchmark.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def main() -> None:
    helper_source = HELPER_PATH.read_text(encoding="utf-8").rstrip()
    helper_write_cell = "%%writefile vllm_free_colab_benchmark_helper.py\n" + helper_source + "\n"

    cells = [
        markdown_cell(
            """# Free Colab vLLM GPU Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrrahman1517/nemotron-vl-embed-stack/blob/codex-gemma4-max-vs-vllm-colab/vllm_free_colab_benchmark.ipynb)

This notebook is for a **free Colab GPU** workflow when you want a quick, usable **vLLM-only** throughput measurement on an open model.

What it does:

- detects the current NVIDIA GPU
- chooses a small open model that should fit typical free Colab runtimes
- launches a local `vllm serve` endpoint
- runs a small concurrent benchmark against `/v1/chat/completions`
- reports request throughput, token throughput, and latency percentiles

What it does **not** do:

- it does **not** compare against MAX
- it does **not** verify the April 2, 2026 Modular Gemma 4 claim
- it does **not** use TPU
"""
        ),
        markdown_cell(
            """## Source Notes

Primary sources:

- Colab overview: [Google Colab](https://colab.google/)
- Colab GPU note that the assigned NVIDIA GPU may vary across sessions: [RAPIDS cuDF on Colab](https://colab.google/articles/cudf)
- vLLM TPU quickstart showing that TPU support is a separate `vllm-tpu` flow on a **Google Cloud TPU VM**, not this Colab GPU notebook: [vLLM TPU Quickstart](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/)

Practical interpretation:

- free Colab can give you a GPU, but the exact GPU varies by session
- this notebook is designed around the common free-tier GPU case, especially **T4-class** sessions
- if you land on **TPU v5e-1**, this is the wrong notebook because vLLM TPU uses a different stack and MAX is not part of that flow
"""
        ),
        markdown_cell(
            """## Recommended Use

- On a **T4 / 16 GiB-class** runtime, keep the default model and benchmark settings
- On a **larger free GPU** if you happen to get one, the notebook will automatically choose a slightly larger open model
- If Colab gives you **no GPU**, reconnect or change the runtime type to GPU if that option is available

This notebook favors **reliability on free Colab** over squeezing out the absolute highest possible throughput.
"""
        ),
        code_cell(
            """import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "uv", "requests", "pandas", "matplotlib"],
    check=True,
)
"""
        ),
        code_cell(helper_write_cell),
        code_cell(
            """import os
from pathlib import Path

from vllm_free_colab_benchmark_helper import choose_free_colab_model, detect_gpu

os.environ.setdefault("HF_HOME", "/content/hf-cache")

gpu = detect_gpu()
print("Detected GPU:", gpu)

if gpu is None:
    raise RuntimeError(
        "No NVIDIA GPU was detected. Free Colab resource availability varies by session. "
        "Reconnect or switch the runtime type to GPU before using this notebook."
    )

FORCE_MODEL = None
MODEL, MODEL_NOTE = choose_free_colab_model(gpu, FORCE_MODEL)

GPU_MEMORY_UTILIZATION = 0.85 if gpu.memory_gb < 20 else 0.90
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 4 if gpu.memory_gb < 20 else 8
NUM_REQUESTS = 24 if gpu.memory_gb < 20 else 48
CONCURRENCY = 2 if gpu.memory_gb < 20 else 4
MAX_TOKENS = 96
STARTUP_TIMEOUT_S = 1800
VLLM_PORT = 8000
LOG_DIR = Path("/content/vllm-free-logs")
RESULT_DIR = Path("/content/vllm-free-results")
ENV_ROOT = Path("/content/vllm-free-envs")

print("Selected model:", MODEL)
print("Selection note:", MODEL_NOTE)
print("GPU memory utilization:", GPU_MEMORY_UTILIZATION)
print("Benchmark requests:", NUM_REQUESTS)
print("Benchmark concurrency:", CONCURRENCY)
"""
        ),
        code_cell(
            """import os
import subprocess
from pathlib import Path


def ensure_uv_env(env_path: Path) -> Path:
    if not env_path.exists():
        subprocess.run(["uv", "venv", str(env_path)], check=True)
    return env_path / "bin" / "python"


ENV_ROOT.mkdir(parents=True, exist_ok=True)
VLLM_ENV = ENV_ROOT / "vllm"
vllm_python = ensure_uv_env(VLLM_ENV)

subprocess.run(
    ["uv", "pip", "install", "--python", str(vllm_python), "vllm", "requests", "pandas", "matplotlib"],
    check=True,
)

VLLM_BIN = str(VLLM_ENV / "bin" / "vllm")
VLLM_SITE_PACKAGES = subprocess.run(
    [str(vllm_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
vllm_version = subprocess.run([VLLM_BIN, "--version"], capture_output=True, text=True).stdout.strip()

print("vLLM version:", vllm_version)
print("vLLM site-packages:", VLLM_SITE_PACKAGES)
"""
        ),
        markdown_cell(
            """## Launch vLLM

This starts an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`.
"""
        ),
        code_cell(
            """from vllm_free_colab_benchmark_helper import (
    ensure_server_ready,
    start_logged_process,
    tail_log,
    warmup_chat,
)

server_env = os.environ.copy()
server_env["HF_HOME"] = os.environ["HF_HOME"]
server_env["VIRTUAL_ENV"] = str(VLLM_ENV)
server_env["PATH"] = f"{VLLM_ENV / 'bin'}:{server_env['PATH']}"
server_env["PYTHONNOUSERSITE"] = "1"
server_env["PYTHONPATH"] = VLLM_SITE_PACKAGES
server_env["MPLBACKEND"] = "Agg"

vllm_handle = start_logged_process(
    "vllm_server",
    [
        VLLM_BIN,
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--dtype",
        "half",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--disable-log-requests",
    ],
    log_dir=LOG_DIR,
    env=server_env,
)

ensure_server_ready(
    f"http://127.0.0.1:{VLLM_PORT}",
    vllm_handle,
    timeout_s=STARTUP_TIMEOUT_S,
)
warmup_chat(f"http://127.0.0.1:{VLLM_PORT}", MODEL)
print(tail_log(vllm_handle.log_path, lines=40))
"""
        ),
        markdown_cell(
            """## Run the Benchmark

The benchmark sends concurrent chat-completions requests and computes throughput and latency from the returned usage stats.
"""
        ),
        code_cell(
            """from vllm_free_colab_benchmark_helper import benchmark_chat_completions, save_json

result = benchmark_chat_completions(
    base_url=f"http://127.0.0.1:{VLLM_PORT}",
    model=MODEL,
    num_requests=NUM_REQUESTS,
    concurrency=CONCURRENCY,
    max_tokens=MAX_TOKENS,
)

result_path = RESULT_DIR / "vllm_free_colab_benchmark.json"
save_json(result_path, result)
print("Saved results to:", result_path)
result
"""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import pandas as pd

summary = pd.DataFrame([result])
display(summary)

plot_data = pd.DataFrame(
    {
        "metric": [
            "request_throughput_rps",
            "completion_token_throughput_tps",
            "mean_latency_ms",
            "p95_latency_ms",
        ],
        "value": [
            result["request_throughput_rps"],
            result["completion_token_throughput_tps"],
            result["mean_latency_ms"],
            result["p95_latency_ms"],
        ],
    }
)

ax = plot_data.plot.bar(x="metric", y="value", legend=False, figsize=(10, 4), color="#2b6cb0")
ax.set_title("vLLM Free Colab Benchmark Summary")
ax.set_ylabel("Value")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """from vllm_free_colab_benchmark_helper import stop_process

stop_process(vllm_handle)
"""
        ),
        markdown_cell(
            """## Reading the Result

- `request_throughput_rps`: completed requests per second
- `completion_token_throughput_tps`: generated output tokens per second
- `mean_latency_ms`: average end-to-end request latency
- `p95_latency_ms`: tail latency for the slower requests

This is a good notebook for getting a **real free-Colab vLLM number**. It is not the right notebook for MAX comparison, TPU benchmarking, or reproducing the B200 Gemma 4 blog claim.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
