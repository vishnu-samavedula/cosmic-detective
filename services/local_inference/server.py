"""Small OpenAI-compatible local server for Cosmic Detective.

Loads one pinned LFM2.5-VL-450M base plus the project's PEFT adapter. The base
route temporarily disables the adapter; the trained route uses it. This is a
development server, not a production inference runtime.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(
    os.environ.get(
        "COSMIC_LOCAL_BASE_PATH",
        ROOT / "models/LFM2.5-VL-450M-fc6221c",
    )
).resolve()
ADAPTER_PATH = Path(
    os.environ.get(
        "COSMIC_LOCAL_ADAPTER_PATH",
        ROOT / "models/gz2-450m-lora",
    )
).resolve()
BASE_NAME = os.environ.get("COSMIC_LOCAL_BASE_MODEL", "local-base-450m")
TRAINED_NAME = os.environ.get(
    "COSMIC_LOCAL_TRAINED_MODEL", "local-gz2-450m"
)

app = FastAPI(title="Cosmic Detective local inference")
_state: dict[str, Any] = {}
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 64
    temperature: float = 0
    stream: bool = False
    stream_options: dict[str, Any] | None = None


def _device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _load() -> tuple[Any, Any, str]:
    if _state:
        return _state["model"], _state["processor"], _state["device"]
    with _load_lock:
        if _state:
            return _state["model"], _state["processor"], _state["device"]
        if not (BASE_PATH / "model.safetensors").is_file():
            raise RuntimeError(f"Base model is missing at {BASE_PATH}")
        if not (ADAPTER_PATH / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"LoRA adapter is missing at {ADAPTER_PATH}")

        device, dtype = _device()
        processor = AutoProcessor.from_pretrained(
            ADAPTER_PATH, local_files_only=True
        )
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_PATH,
            dtype=dtype,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base,
            ADAPTER_PATH,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        _state.update(model=model, processor=processor, device=device)
        return model, processor, device


def _decode_image(url: str) -> Image.Image:
    prefix, separator, encoded = url.partition(",")
    if not separator or ";base64" not in prefix or not prefix.startswith("data:image/"):
        raise ValueError("Only base64 image data URLs are supported")
    return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")


def _messages_for_processor(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            prepared.append({"role": message.get("role", "user"), "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("Each message needs text or multimodal content")
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(url, str):
                    raise ValueError("image_url.url is required")
                parts.append({"type": "image", "image": _decode_image(url)})
        prepared.append({"role": message.get("role", "user"), "content": parts})
    return prepared


def _generate(request: CompletionRequest) -> dict[str, Any]:
    if request.model not in {BASE_NAME, TRAINED_NAME}:
        raise ValueError(f"Unknown local model: {request.model}")
    model, processor, device = _load()
    messages = _messages_for_processor(request.messages)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    generation: dict[str, Any] = {
        "max_new_tokens": max(1, min(request.max_tokens, 512)),
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        generation["temperature"] = request.temperature

    adapter_context = (
        model.disable_adapter()
        if request.model == BASE_NAME
        else nullcontext()
    )
    with _generation_lock, adapter_context, torch.inference_mode():
        output = model.generate(**inputs, **generation)
    generated = output[:, input_tokens:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": int(generated.shape[-1]),
    }


def _event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if _state else "not_loaded",
        "device": _state.get("device"),
        "models": [BASE_NAME, TRAINED_NAME],
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": BASE_NAME, "object": "model"},
            {"id": TRAINED_NAME, "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest):
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(_generate, request)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    completion_id = f"chatcmpl-local-{uuid.uuid4().hex}"
    usage = {
        "prompt_tokens": result["input_tokens"],
        "completion_tokens": result["output_tokens"],
        "total_tokens": result["input_tokens"] + result["output_tokens"],
    }
    if not request.stream:
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "local_elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        )

    async def stream():
        yield _event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"content": result["text"]}, "finish_reason": None}
                ],
            }
        )
        yield _event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
        )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
