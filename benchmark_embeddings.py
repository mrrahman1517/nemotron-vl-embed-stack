import argparse
import json
import os
import statistics
import time
import urllib.request


DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"


def post_embeddings(base_url: str, model: str, texts: list[str], api_key: str, input_type: str | None, timeout: float) -> dict:
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    if input_type:
        payload["input_type"] = input_type

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible embeddings endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--input-type", default=os.environ.get("EMBED_INPUT_TYPE"))
    parser.add_argument("--prompt", default="How is AI improving robotics?")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    texts = [args.prompt] * args.batch_size

    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Measured requests: {args.requests}")
    if args.input_type:
        print(f"Input type: {args.input_type}")

    for _ in range(args.warmup):
        post_embeddings(args.base_url, args.model, texts, args.api_key, args.input_type, args.timeout)

    latencies = []
    first_dims = None
    start = time.perf_counter()
    for _ in range(args.requests):
        request_start = time.perf_counter()
        result = post_embeddings(args.base_url, args.model, texts, args.api_key, args.input_type, args.timeout)
        latencies.append(time.perf_counter() - request_start)
        if first_dims is None and result.get("data"):
            first_dims = len(result["data"][0]["embedding"])
    total = time.perf_counter() - start

    embeddings_returned = args.requests * args.batch_size
    print(f"Embedding dims: {first_dims}")
    print(f"Mean latency (s): {statistics.mean(latencies):.3f}")
    print(f"P50 latency (s): {percentile(latencies, 50):.3f}")
    print(f"P95 latency (s): {percentile(latencies, 95):.3f}")
    print(f"Total wall time (s): {total:.3f}")
    print(f"Embeddings/sec: {embeddings_returned / total:.2f}")


if __name__ == "__main__":
    main()
