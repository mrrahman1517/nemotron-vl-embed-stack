from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HELPER_PATH = REPO_ROOT / "gemma4_colab_benchmark_helper.py"
NOTEBOOK_PATH = REPO_ROOT / "gemma4_max_vs_vllm_colab.ipynb"


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
    helper_write_cell = "%%writefile gemma4_colab_benchmark_helper.py\n" + helper_source + "\n"

    cells = [
        markdown_cell(
            """# Gemma 4 MAX vs vLLM Colab Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrrahman1517/nemotron-vl-embed-stack/blob/codex-gemma4-max-vs-vllm-colab/gemma4_max_vs_vllm_colab.ipynb)

This notebook is designed to test the April 2, 2026 Modular blog claim that **MAX delivers 15% higher throughput than vLLM for Gemma 4 on NVIDIA B200**.

What this notebook does:

- runs **MAX** and **vLLM** side by side against the same Gemma 4 model
- benchmarks both with the **same harness**: `max benchmark`
- reports request throughput, output token throughput, TTFT, TPOT, ITL, and optional GPU stats

Important caveat:

- the blog claim is about **NVIDIA B200** hardware
- most Colab runtimes provide **L4** or **A100**, not B200
- if you run this on L4 or A100, the result is still useful, but it is a **directional comparison**, not an exact reproduction of the published number
"""
        ),
        markdown_cell(
            """## Source Notes

Primary sources used to shape this notebook:

- Modular blog claim and benchmark footnote: [Day Zero Launch: Fastest Performance for Gemma 4 on NVIDIA and AMD](https://www.modular.com/blog/day-zero-launch-fastest-performance-for-gemma-4-on-nvidia-and-amd)
- MAX package installation and GPU compatibility: [Modular packages](https://docs.modular.com/stable/max/packages/)
- MAX supported-model guidance and Gemma 4 architecture support: [Supported models](https://docs.modular.com/max/models/)
- MAX benchmark CLI, including `modular-chat` and `vllm` backends: [max benchmark](https://docs.modular.com/max/cli/benchmark/)
- MAX CLI serve example: [max CLI](https://docs.modular.com/stable/max/max-cli)
- vLLM Gemma 4 deployment guidance and minimum GPU sizes: [Gemma 4 Usage Guide - vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)

Practical interpretation:

- if the runtime has about **24 GiB** VRAM, this notebook defaults to `google/gemma-4-E4B-it`
- if the runtime has about **80 GiB** VRAM, it defaults to `google/gemma-4-26B-A4B-it`
- only a **B200-class** run can directly test the blog's exact hardware claim
"""
        ),
        markdown_cell(
            """## Recommended Runtime Settings

Use these as starting points before you run the benchmark:

- **Colab L4 (24 GiB)**: keep the default `google/gemma-4-E4B-it`, use `NUM_PROMPTS = 64`, `MAX_CONCURRENCY = 8`, and leave `GPU_MEMORY_UTILIZATION = 0.90`
- **Colab A100 (40 GiB)**: still use `google/gemma-4-E4B-it`, and you can usually try `NUM_PROMPTS = 96` to `128` and `MAX_CONCURRENCY = 8` to `16`
- **80 GiB GPU or larger**: switch to `google/gemma-4-26B-A4B-it`, use `NUM_PROMPTS = 128`, `MAX_CONCURRENCY = 16`, and keep `GPU_MEMORY_UTILIZATION` between `0.85` and `0.90`
- **B200**: this is the only class of hardware that can directly test the published claim instead of just giving a directional comparison

If either engine OOMs during load, lower `GPU_MEMORY_UTILIZATION` first, then reduce `MAX_CONCURRENCY`, and finally reduce `NUM_PROMPTS`.
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

from gemma4_colab_benchmark_helper import choose_gemma4_model, detect_gpu

os.environ.setdefault("HF_HOME", "/content/hf-cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    from google.colab import userdata

    HF_TOKEN = HF_TOKEN or userdata.get("HF_TOKEN")
except Exception:
    pass

if not HF_TOKEN:
    raise ValueError(
        "Set HF_TOKEN in a Colab secret named HF_TOKEN or in os.environ before continuing."
    )

os.environ["HF_TOKEN"] = HF_TOKEN

gpu = detect_gpu()
print("Detected GPU:", gpu)

FORCE_MODEL = None
MODULAR_CHANNEL = "stable"  # change to "nightly" if you want the newest MAX build
MODULAR_VERSION = "26.2"
VLLM_VERSION = "0.18.2rc1.dev7"
GPU_MEMORY_UTILIZATION = 0.90
MAX_BATCH_SIZE = 8
BENCHMARK_DATASET = "random"
NUM_PROMPTS = 64 if gpu and gpu.memory_gb < 75 else 128
MAX_CONCURRENCY = 8 if gpu and gpu.memory_gb < 75 else 16
RANDOM_INPUT_LEN = 550
RANDOM_OUTPUT_LEN = 256
RANDOM_COV = "0.0,0.0"
REQUEST_RATE = "inf"
MAX_PORT = 8000
VLLM_PORT = 8001
LOG_DIR = Path("/content/benchmark-logs")
RESULT_DIR = Path("/content/benchmark-results")
ENV_ROOT = Path("/content/benchmark-envs")

MODEL, MODEL_NOTE = choose_gemma4_model(gpu, FORCE_MODEL)
MAX_MODEL_LEN = 8192 if "E4B" in MODEL else 32768

print("Selected model:", MODEL)
print("Selection note:", MODEL_NOTE)
if gpu and "B200" not in gpu.name.upper():
    print("This is not a B200 runtime, so the result should be interpreted as directional rather than exact.")
if gpu:
    try:
        driver_major = int(gpu.driver_version.split(".")[0])
        if driver_major < 580:
            print("Warning: current MAX docs list NVIDIA driver 580+ for supported GPU acceleration.")
    except Exception:
        pass
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
MAX_ENV = ENV_ROOT / "max"
VLLM_ENV = ENV_ROOT / "vllm"

max_python = ensure_uv_env(MAX_ENV)
vllm_python = ensure_uv_env(VLLM_ENV)

if MODULAR_CHANNEL == "nightly":
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(max_python),
            "modular",
            "--index",
            "https://whl.modular.com/nightly/simple/",
            "--prerelease",
            "allow",
        ],
        check=True,
    )
    ACTUAL_MODULAR_SPEC = "modular (nightly)"
else:
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(max_python),
            f"modular=={MODULAR_VERSION}",
            "--extra-index-url",
            "https://modular.gateway.scarf.sh/simple/",
        ],
        check=True,
    )
    ACTUAL_MODULAR_SPEC = f"modular=={MODULAR_VERSION}"

subprocess.run(["uv", "pip", "install", "--python", str(max_python), "requests"], check=True)

ACTUAL_VLLM_SPEC = f"vllm=={VLLM_VERSION}"
try:
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(vllm_python),
            f"vllm=={VLLM_VERSION}",
            "--torch-backend=auto",
            "--extra-index-url",
            "https://wheels.vllm.ai/nightly",
            "--prerelease",
            "allow",
        ],
        check=True,
    )
except subprocess.CalledProcessError:
    print("Exact vLLM prerelease install failed; falling back to the latest available vLLM build.")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(vllm_python),
            "vllm",
            "--torch-backend=auto",
        ],
        check=True,
    )
    ACTUAL_VLLM_SPEC = "vllm (latest available)"

subprocess.run(
    ["uv", "pip", "install", "--python", str(vllm_python), "transformers==5.5.0"],
    check=True,
)

MAX_BIN = str(MAX_ENV / "bin" / "max")
VLLM_BIN = str(VLLM_ENV / "bin" / "vllm")

max_version = subprocess.run([MAX_BIN, "--version"], capture_output=True, text=True).stdout.strip()
vllm_version_output = subprocess.run([VLLM_BIN, "--version"], capture_output=True, text=True).stdout.strip()

print("MAX install:", ACTUAL_MODULAR_SPEC)
print("MAX version:", max_version)
print("vLLM install:", ACTUAL_VLLM_SPEC)
print("vLLM version:", vllm_version_output)
"""
        ),
        markdown_cell(
            """## Launch and Benchmark MAX

This uses the OpenAI-compatible `max serve` endpoint and then benchmarks it with `max benchmark --backend modular-chat`.
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import (
    ensure_server_ready,
    start_logged_process,
    tail_log,
    warmup_chat,
)

server_env = os.environ.copy()
server_env["HF_TOKEN"] = HF_TOKEN
server_env["HF_HOME"] = os.environ["HF_HOME"]

max_handle = start_logged_process(
    "max_server",
    [
        MAX_BIN,
        "serve",
        "--model",
        MODEL,
        "--devices",
        "gpu:0",
        "--port",
        str(MAX_PORT),
        "--max-length",
        str(MAX_MODEL_LEN),
        "--max-batch-size",
        str(MAX_BATCH_SIZE),
        "--device-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
    ],
    log_dir=LOG_DIR,
    env=server_env,
)

ensure_server_ready(f"http://127.0.0.1:{MAX_PORT}", timeout_s=1800)
warmup_chat(f"http://127.0.0.1:{MAX_PORT}", MODEL)
print(tail_log(max_handle.log_path, lines=40))
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import load_benchmark_result, run_benchmark, summarize_result

max_result_path = run_benchmark(
    max_bin=MAX_BIN,
    backend="modular-chat",
    model=MODEL,
    port=MAX_PORT,
    result_dir=RESULT_DIR,
    result_filename="max_random.json",
    dataset_name=BENCHMARK_DATASET,
    num_prompts=NUM_PROMPTS,
    request_rate=REQUEST_RATE,
    max_concurrency=MAX_CONCURRENCY,
    random_input_len=RANDOM_INPUT_LEN,
    random_output_len=RANDOM_OUTPUT_LEN,
    random_coefficient_of_variation=RANDOM_COV,
    tokenizer=MODEL,
)

max_result = load_benchmark_result(max_result_path)
max_summary = summarize_result(max_result, "MAX")
max_summary
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import stop_process

stop_process(max_handle)
"""
        ),
        markdown_cell(
            """## Launch and Benchmark vLLM

This uses `vllm serve` with the same model and then benchmarks it with `max benchmark --backend vllm`.
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import (
    ensure_server_ready,
    start_logged_process,
    tail_log,
    warmup_chat,
)

server_env = os.environ.copy()
server_env["HF_TOKEN"] = HF_TOKEN
server_env["HF_HOME"] = os.environ["HF_HOME"]

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
        "--limit-mm-per-prompt",
        "image=0,audio=0",
    ],
    log_dir=LOG_DIR,
    env=server_env,
)

ensure_server_ready(f"http://127.0.0.1:{VLLM_PORT}", timeout_s=1800)
warmup_chat(f"http://127.0.0.1:{VLLM_PORT}", MODEL)
print(tail_log(vllm_handle.log_path, lines=40))
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import load_benchmark_result, run_benchmark, summarize_result

vllm_result_path = run_benchmark(
    max_bin=MAX_BIN,
    backend="vllm",
    model=MODEL,
    port=VLLM_PORT,
    result_dir=RESULT_DIR,
    result_filename="vllm_random.json",
    dataset_name=BENCHMARK_DATASET,
    num_prompts=NUM_PROMPTS,
    request_rate=REQUEST_RATE,
    max_concurrency=MAX_CONCURRENCY,
    random_input_len=RANDOM_INPUT_LEN,
    random_output_len=RANDOM_OUTPUT_LEN,
    random_coefficient_of_variation=RANDOM_COV,
    tokenizer=MODEL,
)

vllm_result = load_benchmark_result(vllm_result_path)
vllm_summary = summarize_result(vllm_result, "vLLM")
vllm_summary
"""
        ),
        code_cell(
            """from gemma4_colab_benchmark_helper import stop_process

stop_process(vllm_handle)
"""
        ),
        markdown_cell(
            """## Compare Results

The most important number for the blog claim is **output token throughput**. This cell also compares request throughput and latency metrics.
"""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import pandas as pd

from gemma4_colab_benchmark_helper import percent_delta

comparison = pd.DataFrame([max_summary, vllm_summary]).set_index("label")
display(comparison)

output_delta_pct = percent_delta(
    comparison.loc["MAX", "output_throughput"],
    comparison.loc["vLLM", "output_throughput"],
)
request_delta_pct = percent_delta(
    comparison.loc["MAX", "request_throughput"],
    comparison.loc["vLLM", "request_throughput"],
)

if output_delta_pct is not None:
    print(f"MAX output throughput delta vs vLLM: {output_delta_pct:.2f}%")
if request_delta_pct is not None:
    print(f"MAX request throughput delta vs vLLM: {request_delta_pct:.2f}%")

plot_columns = [
    column
    for column in [
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
    ]
    if column in comparison.columns
]

comparison[plot_columns].plot(kind="bar", figsize=(10, 5), rot=0)
plt.title(f"{MODEL} on {gpu.name if gpu else 'unknown GPU'}")
plt.tight_layout()
plt.show()

RESULT_DIR.mkdir(parents=True, exist_ok=True)
summary_path = RESULT_DIR / "max_vs_vllm_summary.csv"
comparison.to_csv(summary_path)
print("Saved summary to", summary_path)
print("Raw benchmark JSON files are in", RESULT_DIR)
"""
        ),
        markdown_cell(
            """## How To Interpret What You See

- If your output throughput delta is positive, MAX was faster on **your** runtime and settings.
- If your runtime was not a **B200**, do not treat the result as a direct reproduction of the Modular blog number.
- If the notebook had to fall back from `vllm==0.18.2rc1.dev7`, note that in your conclusions because the blog explicitly names that version.
- To get closer to the blog setup, rerun this notebook on a **B200** or at least an **80 GiB** GPU and use `google/gemma-4-26B-A4B-it`.
- This notebook intentionally uses a **single benchmark harness** across both engines to reduce client-side measurement bias.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": NOTEBOOK_PATH.name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
