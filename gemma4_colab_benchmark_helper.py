from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - installed inside Colab
    requests = None


@dataclass
class GPUInfo:
    name: str
    memory_gb: float
    driver_version: str
    compute_capability: str | None = None


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def capture(cmd: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None) -> str:
    print("$", shlex.join(cmd))
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
    )
    return completed.stdout.strip()


def detect_gpu() -> GPUInfo | None:
    try:
        output = capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        return None

    first_line = output.splitlines()[0].strip()
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return None

    name = parts[0]
    memory_gb = round(float(parts[1]) / 1024.0, 2)
    driver_version = parts[2]
    compute_capability = parts[3] if len(parts) >= 4 else None
    return GPUInfo(
        name=name,
        memory_gb=memory_gb,
        driver_version=driver_version,
        compute_capability=compute_capability,
    )


def choose_gemma4_model(gpu: GPUInfo | None, force_model: str | None = None) -> tuple[str, str]:
    if force_model:
        return force_model, "Using a user-forced model override."

    if gpu is None:
        return (
            "google/gemma-4-E4B-it",
            "GPU detection failed; defaulting to Gemma 4 E4B for a smaller-footprint run.",
        )

    if gpu.memory_gb >= 75:
        if "B200" in gpu.name.upper():
            return (
                "google/gemma-4-26B-A4B-it",
                "This runtime is closest to the blog's hardware class, though exact parity still depends on the full software stack.",
            )
        return (
            "google/gemma-4-26B-A4B-it",
            "This runtime has enough memory for the larger Gemma 4 MoE model, but it is still not the exact B200 setup from the blog.",
        )

    if gpu.memory_gb >= 22:
        return (
            "google/gemma-4-E4B-it",
            "This is a directional Colab-friendly run on a smaller Gemma 4 variant because typical Colab GPUs do not expose B200-class memory.",
        )

    raise RuntimeError(
        f"Detected only {gpu.memory_gb:.1f} GiB of VRAM on {gpu.name}. "
        "Use at least a 24 GiB GPU for Gemma 4 E4B, or switch to an 80 GiB-class runtime for Gemma 4 26B-A4B."
    )


def start_logged_process(
    name: str,
    command: list[str],
    *,
    log_dir: str | Path,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> ManagedProcess:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")

    print("$", shlex.join(command))
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
    )
    return ManagedProcess(name=name, command=command, process=process, log_path=log_path)


def tail_log(log_path: str | Path, lines: int = 80) -> str:
    log_path = Path(log_path)
    if not log_path.exists():
        return f"{log_path} does not exist."
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def ensure_server_ready(base_url: str, *, timeout_s: int = 1800) -> None:
    if requests is None:
        raise RuntimeError("requests is required for readiness checks.")

    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=10)
            if response.ok:
                print(f"Server ready at {base_url}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # pragma: no cover - runtime specific
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_s}s. Last error: {last_error}")


def ensure_server_ready_with_logs(
    base_url: str,
    handle: ManagedProcess,
    *,
    timeout_s: int = 1800,
    log_interval_s: int = 30,
) -> None:
    if requests is None:
        raise RuntimeError("requests is required for readiness checks.")

    deadline = time.time() + timeout_s
    last_error = None
    next_log_time = time.time() + log_interval_s

    while time.time() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(
                f"{handle.name} exited before becoming ready. "
                f"Last log lines:\n{tail_log(handle.log_path, lines=120)}"
            )

        try:
            response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=10)
            if response.ok:
                print(f"Server ready at {base_url}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # pragma: no cover - runtime specific
            last_error = str(exc)

        if time.time() >= next_log_time:
            print(f"Still waiting for {handle.name} at {base_url}")
            print(tail_log(handle.log_path, lines=80))
            next_log_time = time.time() + log_interval_s

        time.sleep(5)

    raise TimeoutError(
        f"Server at {base_url} did not become ready within {timeout_s}s. "
        f"Last error: {last_error}\nLast log lines:\n{tail_log(handle.log_path, lines=120)}"
    )


def warmup_chat(base_url: str, model: str) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for warmup requests.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with the single word ready.",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    print(json.dumps(body.get("usage", {}), indent=2))
    return body


def stop_process(handle: ManagedProcess | None, *, grace_seconds: int = 20) -> None:
    if handle is None:
        return
    process = handle.process
    if process.poll() is not None:
        return

    print(f"Stopping {handle.name} (pid={process.pid})")
    process.send_signal(signal.SIGINT if sys.platform != "win32" else signal.CTRL_BREAK_EVENT)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_benchmark(
    *,
    max_bin: str,
    backend: str,
    model: str,
    port: int,
    result_dir: str | Path,
    result_filename: str,
    dataset_name: str = "random",
    num_prompts: int = 128,
    request_rate: str = "inf",
    max_concurrency: int = 16,
    random_input_len: int = 550,
    random_output_len: int = 256,
    random_coefficient_of_variation: str = "0.0,0.0",
    collect_gpu_stats: bool = True,
    tokenizer: str | None = None,
) -> Path:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / result_filename
    if result_path.exists():
        result_path.unlink()

    command = [
        max_bin,
        "benchmark",
        "--model",
        model,
        "--backend",
        backend,
        "--endpoint",
        "/v1/chat/completions",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--max-concurrency",
        str(max_concurrency),
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]

    if tokenizer:
        command.extend(["--tokenizer", tokenizer])

    if collect_gpu_stats:
        command.append("--collect-gpu-stats")

    if dataset_name == "random":
        command.extend(
            [
                "--random-input-len",
                str(random_input_len),
                "--random-output-len",
                str(random_output_len),
                "--random-coefficient-of-variation",
                random_coefficient_of_variation,
            ]
        )

    save_flags = ("--save-results", "--save-result")
    last_error: Exception | None = None
    for save_flag in save_flags:
        cmd = [*command, save_flag]
        try:
            run(cmd)
            return result_path
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if result_path.exists():
                return result_path
            print(f"{save_flag} failed, retrying with the alternate save flag.")

    if result_path.exists():
        return result_path
    raise RuntimeError(f"Benchmark did not produce {result_path}") from last_error


def load_benchmark_result(result_path: str | Path) -> dict[str, Any]:
    result_path = Path(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def summarize_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    summary = {
        "label": label,
        "backend": result.get("backend"),
        "model_id": result.get("model_id"),
        "completed": result.get("completed"),
        "duration_s": result.get("duration"),
        "request_throughput": result.get("request_throughput"),
        "output_throughput": result.get("output_throughput"),
        "total_token_throughput": result.get("total_token_throughput"),
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "p99_ttft_ms": result.get("p99_ttft_ms"),
        "mean_tpot_ms": result.get("mean_tpot_ms"),
        "mean_itl_ms": result.get("mean_itl_ms"),
        "max_concurrency": result.get("max_concurrency"),
        "gpu_utilization": result.get("gpu_utilization"),
        "peak_gpu_memory_used": result.get("peak_gpu_memory_used"),
    }
    return summary


def percent_delta(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return ((float(numerator) - float(denominator)) / float(denominator)) * 100.0
