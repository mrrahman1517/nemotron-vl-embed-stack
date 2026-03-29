import json
import os
import urllib.request


BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
INPUT_TYPE = os.environ.get("EMBED_INPUT_TYPE", "")


def create_embeddings(texts: list[str]) -> dict:
    payload = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    if INPUT_TYPE:
        payload["input_type"] = INPUT_TYPE
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        f"{BASE_URL}/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    texts = [
        "How is AI improving robotics?",
        "AI helps robots perceive, plan, and act more autonomously.",
    ]
    result = create_embeddings(texts)
    for item in result.get("data", []):
        vector = item["embedding"]
        print(f"index={item['index']} dims={len(vector)} first5={vector[:5]}")
