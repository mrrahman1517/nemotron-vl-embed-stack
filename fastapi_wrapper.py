from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BACKEND_MODEL = os.environ.get("BACKEND_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")
REQUEST_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT_SECONDS", "300"))
ALLOW_INPUT_TYPE_FALLBACK = os.environ.get("ALLOW_INPUT_TYPE_FALLBACK", "true").lower() in {"1", "true", "yes"}

app = FastAPI(title="Nemotron VL Embed Wrapper", version="0.1.0")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    model: str | None = None
    input_type: str | None = None
    encoding_format: str = "float"


class HealthResponse(BaseModel):
    ok: bool
    backend_base_url: str
    backend_model: str
    input_type_fallback: bool


async def _post_embeddings(
    client: httpx.AsyncClient,
    texts: list[str],
    model: str,
    input_type: str | None,
    encoding_format: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": encoding_format,
    }
    if input_type:
        payload["input_type"] = input_type

    headers = {"Content-Type": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"

    response = await client.post(
        f"{BACKEND_BASE_URL}/v1/embeddings",
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code in (400, 404, 422) and input_type and ALLOW_INPUT_TYPE_FALLBACK:
        fallback_payload = {
            "model": model,
            "input": texts,
            "encoding_format": encoding_format,
        }
        response = await client.post(
            f"{BACKEND_BASE_URL}/v1/embeddings",
            json=fallback_payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if response.is_success:
            data = response.json()
            data["wrapper_note"] = (
                "Backend did not accept input_type; request was retried without it."
            )
            return data

    if not response.is_success:
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    return response.json()


@app.on_event("startup")
async def startup() -> None:
    app.state.client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.client.aclose()


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        ok=True,
        backend_base_url=BACKEND_BASE_URL,
        backend_model=BACKEND_MODEL,
        input_type_fallback=ALLOW_INPUT_TYPE_FALLBACK,
    )


@app.get("/config")
async def config() -> dict[str, Any]:
    return {
        "backend_base_url": BACKEND_BASE_URL,
        "backend_model": BACKEND_MODEL,
        "has_backend_api_key": bool(BACKEND_API_KEY),
        "allow_input_type_fallback": ALLOW_INPUT_TYPE_FALLBACK,
        "timeout_seconds": REQUEST_TIMEOUT,
    }


@app.post("/embed/query")
async def embed_query(request: EmbedRequest) -> dict[str, Any]:
    model = request.model or BACKEND_MODEL
    return await _post_embeddings(
        app.state.client,
        request.texts,
        model,
        "query",
        request.encoding_format,
    )


@app.post("/embed/document")
async def embed_document(request: EmbedRequest) -> dict[str, Any]:
    model = request.model or BACKEND_MODEL
    return await _post_embeddings(
        app.state.client,
        request.texts,
        model,
        "document",
        request.encoding_format,
    )


@app.post("/embed/text")
async def embed_text(request: EmbedRequest) -> dict[str, Any]:
    model = request.model or BACKEND_MODEL
    return await _post_embeddings(
        app.state.client,
        request.texts,
        model,
        request.input_type,
        request.encoding_format,
    )


@app.post("/v1/embeddings")
async def embeddings_passthrough(request: EmbedRequest) -> dict[str, Any]:
    model = request.model or BACKEND_MODEL
    return await _post_embeddings(
        app.state.client,
        request.texts,
        model,
        request.input_type,
        request.encoding_format,
    )
