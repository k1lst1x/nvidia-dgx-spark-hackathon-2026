import asyncio
import base64
import binascii
import json
import logging
import os
import random
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "1500000"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.5"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("guidelens")
app = FastAPI(title="GuideLens Vision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8081").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Session-ID"],
)

_slots = asyncio.Semaphore(MAX_CONCURRENCY)
_requests: dict[str, deque[float]] = defaultdict(deque)


class DetectedObject(BaseModel):
    label: str = Field(max_length=40)
    confidence: float = Field(ge=0, le=1)
    position: str = Field(max_length=30)


class VisionResult(BaseModel):
    summary: str = Field(max_length=100)
    speech: str = Field(max_length=240)
    scene_key: str = Field(max_length=80)
    urgency: Literal["low", "medium", "high"]
    changed: bool
    objects: list[DetectedObject] = Field(max_length=8)
    task: Literal["scene", "sign"]


class SignSequenceRequest(BaseModel):
    frames: list[str] = Field(min_length=3, max_length=3)


VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "speech": {"type": "string"},
        "scene_key": {"type": "string"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "changed": {"type": "boolean"},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "position": {"type": "string"},
                },
                "required": ["label", "confidence", "position"],
                "additionalProperties": False,
            },
        },
        "task": {"type": "string", "enum": ["scene", "sign"]},
    },
    "required": ["summary", "speech", "scene_key", "urgency", "changed", "objects", "task"],
    "additionalProperties": False,
}

SCENE_PROMPT = """You are GuideLens, a visual accessibility assistant for a blind user.
Describe only useful facts visible in the current image. First look for a clearly foregrounded or
hand-held object and identify it if visually supported; do not let the camera holder's face or body
dominate the answer. Glasses are eyewear, never a frisbee. Do not force a COCO-style label: if an
item is unclear, say it is an unrecognized object instead of guessing. Then prioritize immediate
hazards, people, doors, stairs, curbs, vehicles, and clear paths. Never claim exact distance; use
short approximate US customary distances only when useful. Do not identify people or infer
sensitive traits. Keep speech under 25 words. Compare with the previous scene key and set
changed=false when there is no meaningful navigation change. scene_key must be a short, stable,
lowercase semantic fingerprint. task must be scene. Output only the requested JSON."""

SIGN_PROMPT = """You are assisting communication with a signer using three chronological frames
sampled from a short signing sequence. Interpret only when the complete sign is visually clear.
Three frames cannot reliably represent every motion-based sign; when uncertain say 'Sign not clear—
please repeat it slowly.' Never invent a translation. Keep speech under 18 words, set objects to an
empty array, use a stable scene_key, and set task to sign. Output only the requested JSON."""


def _rate_limit(client_id: str) -> None:
    now = time.monotonic()
    bucket = _requests[client_id]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if bucket and now - bucket[-1] < MIN_REQUEST_INTERVAL:
        raise HTTPException(429, "Frames are arriving too quickly", headers={"Retry-After": "2"})
    if len(bucket) >= 20:
        raise HTTPException(429, "Session rate limit exceeded", headers={"Retry-After": "5"})
    bucket.append(now)


async def _call_groq(images: list[tuple[bytes, str]], previous_scene_key: str, task: str) -> VisionResult:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "Vision service is not configured")
    content = [{"type": "text", "text": f"Previous scene key: {previous_scene_key or 'none'}. Analyze the current {'frame sequence' if task == 'sign' else 'frame'}."}]
    for image, mime in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SCENE_PROMPT if task == "scene" else SIGN_PROMPT},
            {
                "role": "user",
                "content": content,
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "vision_result", "strict": True, "schema": VISION_SCHEMA},
        },
        "temperature": 0.1,
        "max_completion_tokens": 220,
        "stream": False,
    }
    timeout = httpx.Timeout(12.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        retry = float(response.headers.get("retry-after", 0))
                        await asyncio.sleep(max(retry, min(0.5 * (2**attempt) + random.random(), 4)))
                        continue
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
                return VisionResult.model_validate(json.loads(raw))
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise HTTPException(504, "Vision provider timed out")
                await asyncio.sleep(0.5 * (2**attempt) + random.random())
            except (KeyError, json.JSONDecodeError, ValidationError):
                raise HTTPException(502, "Vision provider returned an invalid response")
            except httpx.HTTPStatusError as exc:
                logger.warning("groq_error status=%s", exc.response.status_code)
                raise HTTPException(502, "Vision provider request failed")
    raise HTTPException(502, "Vision provider request failed")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "groq",
        "configured": bool(os.getenv("GROQ_API_KEY")),
        "model": MODEL,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/describe", response_model=VisionResult)
async def describe(
    request: Request,
    image: UploadFile = File(...),
    task: str = Form("scene"),
    previous_scene_key: str = Form(""),
):
    if task not in {"scene", "sign"}:
        raise HTTPException(422, "task must be 'scene' or 'sign'")
    request_id = uuid.uuid4().hex[:12]
    client_id = request.headers.get("X-Session-ID") or (request.client.host if request.client else "unknown")
    _rate_limit(client_id)
    if image.content_type not in {"image/jpeg", "image/webp", "image/png"}:
        raise HTTPException(415, "JPEG, WebP, or PNG required")
    body = await image.read(MAX_IMAGE_BYTES + 1)
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is empty or too large")
    started = time.monotonic()
    async with _slots:
        result = await _call_groq([(body, image.content_type)], previous_scene_key[:80], task)
    logger.info(
        "vision_complete request_id=%s latency_ms=%d bytes=%d changed=%s urgency=%s",
        request_id,
        int((time.monotonic() - started) * 1000),
        len(body),
        result.changed,
        result.urgency,
    )
    return result


@app.post("/v1/interpret-sign", response_model=VisionResult)
async def interpret_sign(request: Request, sequence: SignSequenceRequest):
    client_id = request.headers.get("X-Session-ID") or (request.client.host if request.client else "unknown")
    _rate_limit(client_id)
    decoded: list[tuple[bytes, str]] = []
    for frame in sequence.frames:
        if not frame.startswith("data:image/jpeg;base64,"):
            raise HTTPException(415, "Each frame must be a JPEG data URL")
        try:
            body = base64.b64decode(frame.split(",", 1)[1], validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(422, "A frame contains invalid base64 data")
        if not body or len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "A sign frame is empty or too large")
        decoded.append((body, "image/jpeg"))
    started = time.monotonic()
    async with _slots:
        result = await _call_groq(decoded, "", "sign")
    logger.info(
        "sign_complete latency_ms=%d frame_count=%d total_bytes=%d",
        int((time.monotonic() - started) * 1000),
        len(decoded),
        sum(len(body) for body, _ in decoded),
    )
    return result
