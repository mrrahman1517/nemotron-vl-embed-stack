from __future__ import annotations

import json
import shlex
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - installed in Colab
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


def capture(cmd: list[str]) -> str:
    print("$", shlex.join(cmd))
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
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

    return GPUInfo(
        name=parts[0],
        memory_gb=round(float(parts[1]) / 1024.0, 2),
        driver_version=parts[2],
        compute_capability=parts[3] if len(parts) >= 4 else None,
    )


def choose_free_colab_model(
    gpu: GPUInfo | None,
    force_model: str | None = None,
) -> tuple[str, str]:
    if force_model:
        return force_model, "Using a user-forced model override."

    if gpu is None:
        return (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "GPU detection failed, so the notebook is using a very small open model to keep the vLLM benchmark runnable.",
        )

    if gpu.memory_gb >= 20:
        return (
            "Qwen/Qwen2.5-3B-Instruct",
            "This runtime has enough memory for a slightly larger open model while still being practical for Colab benchmarking.",
        )

    return (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "This model is small enough for typical free Colab T4 runtimes while still giving a useful vLLM throughput signal.",
    )


def start_logged_process(
    name: str,
    command: list[str],
    *,
    log_dir: str | Path,
    env: dict[str, str] | None = None,
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
    )
    return ManagedProcess(name=name, command=command, process=process, log_path=log_path)


def tail_log(log_path: str | Path, *, lines: int = 80) -> str:
    log_path = Path(log_path)
    if not log_path.exists():
        return f"{log_path} does not exist."
    return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def ensure_server_ready(
    base_url: str,
    handle: ManagedProcess,
    *,
    timeout_s: int = 1800,
    log_interval_s: int = 30,
) -> None:
    if requests is None:
        raise RuntimeError("requests is required for readiness checks.")

    deadline = time.time() + timeout_s
    next_log_time = time.time() + log_interval_s
    last_error = None

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
        except Exception as exc:  # pragma: no cover - Colab specific
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
        "messages": [{"role": "user", "content": "Reply with the single word ready."}],
        "temperature": 0,
        "max_tokens": 8,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def benchmark_chat_completions(
    *,
    base_url: str,
    model: str,
    num_requests: int = 24,
    concurrency: int = 2,
    max_tokens: int = 96,
    timeout_s: int = 300,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for benchmarking.")

    prompt = (
        "Explain in one short paragraph how tensor cores help transformer inference. "
        "Keep the answer factual and concise."
    )

    def one_request(request_id: int) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt} Request {request_id}."}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
        )
        latency_s = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return {
            "latency_s": latency_s,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    started_all = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(one_request, request_id) for request_id in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = time.perf_counter() - started_all

    latencies_ms = [item["latency_s"] * 1000.0 for item in results]
    prompt_tokens = sum(item["prompt_tokens"] for item in results)
    completion_tokens = sum(item["completion_tokens"] for item in results)
    total_tokens = sum(item["total_tokens"] for item in results)

    return {
        "model": model,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "completed_requests": len(results),
        "wall_time_s": wall_time_s,
        "request_throughput_rps": len(results) / wall_time_s if wall_time_s else None,
        "prompt_token_throughput_tps": prompt_tokens / wall_time_s if wall_time_s else None,
        "completion_token_throughput_tps": completion_tokens / wall_time_s if wall_time_s else None,
        "total_token_throughput_tps": total_tokens / wall_time_s if wall_time_s else None,
        "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "max_latency_ms": max(latencies_ms) if latencies_ms else None,
    }


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


def save_json(path: str | Path, body: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
