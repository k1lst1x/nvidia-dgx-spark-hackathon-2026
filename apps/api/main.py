import asyncio
import base64
import binascii
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from depth import ApproachingObjectTracker, DepthAnythingMetric, readings_for_boxes

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "1500000"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.5"))
YOLO_ENABLED = os.getenv("ULTRALYTICS_YOLO_ENABLED", "false").lower() in {"1", "true", "yes"}
YOLO_MODEL = os.getenv("ULTRALYTICS_YOLO_MODEL", "yolo11n.pt")
YOLO_DEVICE = os.getenv("ULTRALYTICS_YOLO_DEVICE", "cpu")
YOLO_CONFIDENCE = float(os.getenv("ULTRALYTICS_YOLO_CONFIDENCE", "0.40"))
# One Ultralytics model instance is shared by the API process. Keep the ByteTrack
# state scoped to one active camera feed so object IDs never bleed between users.
YOLO_TRACK_IDLE_SECONDS = float(os.getenv("YOLO_TRACK_IDLE_SECONDS", "4"))

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
_depth_model = DepthAnythingMetric()
_approach_tracker = ApproachingObjectTracker()
_yolo_tracking_lock = threading.Lock()
_yolo_tracker_session: str | None = None
_yolo_tracker_seen_at = 0.0


class DetectedObject(BaseModel):
    label: str = Field(max_length=40)
    confidence: float = Field(ge=0, le=1)
    position: str = Field(max_length=30)
    # Normalized coordinates are filled by the verified local YOLO detector.
    # They are optional so Groq-only deployments retain the original API shape.
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    width: float | None = Field(default=None, ge=0, le=1)
    height: float | None = Field(default=None, ge=0, le=1)
    distance_m: float | None = Field(default=None, gt=0, le=80)
    # Short-lived ByteTrack ID for associating one camera feed's boxes over time.
    # It is intentionally not a persistent identity or person identifier.
    track_id: int | None = Field(default=None, ge=1)


class ApproachingWarning(BaseModel):
    label: str = Field(max_length=40)
    distance_m: float = Field(gt=0, le=80)


class VisionResult(BaseModel):
    summary: str = Field(max_length=100)
    speech: str = Field(max_length=240)
    scene_key: str = Field(max_length=80)
    urgency: Literal["low", "medium", "high"]
    changed: bool
    objects: list[DetectedObject] = Field(max_length=8)
    approaching: list[ApproachingWarning] = Field(default_factory=list, max_length=3)
    task: Literal["scene", "sign"]


class SignSequenceRequest(BaseModel):
    frames: list[str] = Field(min_length=3, max_length=3)
    previous_transcript: str = Field(default="", max_length=600)


class SignTranslation(BaseModel):
    transcript: str = Field(max_length=600)
    segment: str = Field(max_length=180)
    confidence: float = Field(ge=0, le=1)
    is_final: bool
    should_speak: bool


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

SIGN_TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "segment": {"type": "string"},
        "confidence": {"type": "number"},
        "is_final": {"type": "boolean"},
        "should_speak": {"type": "boolean"},
    },
    "required": ["transcript", "segment", "confidence", "is_final", "should_speak"],
    "additionalProperties": False,
}

SCENE_PROMPT = """You are GuideLens, a visual accessibility assistant for a blind user.
Describe only useful facts visible in the current image. First look for a clearly foregrounded or
hand-held object and identify it if visually supported; do not let the camera holder's face or body
dominate the answer. Glasses are eyewear, never a frisbee. Do not force a COCO-style label: if an
item is unclear, say it is an unrecognized object instead of guessing. Then prioritize immediate
hazards, people, doors, stairs, curbs, vehicles, and clear paths. Never claim exact distance; use
short approximate US customary distances only when useful. Do not identify people or infer
 sensitive traits. If verified detector labels are supplied, use them as evidence but do not invent
 details beyond them. Keep speech under 25 words. Compare with the previous scene key and set
changed=false when there is no meaningful navigation change. scene_key must be a short, stable,
lowercase semantic fingerprint. task must be scene. Output only the requested JSON."""

SIGN_PROMPT = """You are the conservative temporal decoder in an experimental American Sign
Language (ASL) communication aid. You receive three chronological frames sampled across one
overlapping video window and the transcript accepted from earlier windows. Consider both hands,
motion between frames, body pose, facial grammar, and fingerspelling. Translate only visually
supported ASL; do not treat arbitrary gestures as signs and never invent missing words.

Return transcript as the complete previous transcript plus only a genuinely new continuation.
Return segment as only that new continuation, or an empty string when the window is unclear,
duplicated, transitional, or contains no signing. Preserve an already accepted transcript verbatim.
Set confidence conservatively. is_final means a phrase boundary is visible (pause or hands lowered),
not merely that one window ended. should_speak is true only for a non-empty, confident, final new
segment. Output only the requested JSON."""


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


def _position_for_box(x: float, y: float, width: float, height: float) -> str:
    horizontal = "left" if x + width / 2 < 0.33 else "right" if x + width / 2 > 0.67 else "center"
    vertical = "upper" if y + height / 2 < 0.33 else "lower" if y + height / 2 > 0.67 else "middle"
    return horizontal if vertical == "middle" else f"{vertical} {horizontal}"


@lru_cache(maxsize=1)
def _get_yolo_model():
    """Load the official Ultralytics checkpoint only when the opt-in detector is used."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - deployment configuration guard
        raise RuntimeError("Ultralytics is not installed; rebuild the API image with requirements.txt") from exc
    return YOLO(YOLO_MODEL)


def _reset_yolo_tracker(model) -> None:
    """Clear ByteTrack state attached to an already-initialized Ultralytics predictor."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()):
        reset = getattr(tracker, "reset", None)
        if reset:
            reset()


def _detect_with_yolo(image_bytes: bytes, session_id: str = "") -> list[DetectedObject]:
    """Run YOLO11 + ByteTrack and return display-ready normalized, tracked boxes."""
    global _yolo_tracker_session, _yolo_tracker_seen_at
    if not YOLO_ENABLED:
        return []
    try:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # `persist=True` carries IDs between captures in one feed. Ultralytics'
        # tracker lives on its shared predictor, so protect it with a lock and
        # reset it for a new/idle session before accepting another camera frame.
        with _yolo_tracking_lock:
            model = _get_yolo_model()
            now = time.monotonic()
            if (
                _yolo_tracker_session is not None
                and (
                    _yolo_tracker_session != session_id
                    or now - _yolo_tracker_seen_at > YOLO_TRACK_IDLE_SECONDS
                )
            ):
                _reset_yolo_tracker(model)
            _yolo_tracker_session = session_id
            _yolo_tracker_seen_at = now
            result = model.track(
                source=image,
                persist=True,
                tracker="bytetrack.yaml",
                conf=YOLO_CONFIDENCE,
                imgsz=640,
                device=YOLO_DEVICE,
                verbose=False,
                max_det=12,
            )[0]
        image_width, image_height = image.size
        names = result.names
        detections: list[DetectedObject] = []
        for box in result.boxes:
            left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
            x = max(0.0, min(1.0, left / image_width))
            y = max(0.0, min(1.0, top / image_height))
            width = max(0.0, min(1.0 - x, (right - left) / image_width))
            height = max(0.0, min(1.0 - y, (bottom - top) / image_height))
            if width < 0.02 or height < 0.02:
                continue
            class_id = int(box.cls[0].item())
            label = str(names[class_id])
            track_id = (
                int(box.id[0].item())
                if getattr(result.boxes, "is_track", False) and box.id is not None
                else None
            )
            detections.append(DetectedObject(
                label=label,
                confidence=float(box.conf[0].item()),
                position=_position_for_box(x, y, width, height),
                x=x,
                y=y,
                width=width,
                height=height,
                track_id=track_id,
            ))
        return detections
    except Exception:
        logger.exception("yolo_detection_failed model=%s", YOLO_MODEL)
        return []


async def _call_groq(
    images: list[tuple[bytes, str]],
    previous_scene_key: str,
    task: str,
    verified_objects: list[DetectedObject] | None = None,
) -> VisionResult:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "Vision service is not configured")
    detector_context = ""
    if task == "scene" and verified_objects:
        detector_context = " Verified local YOLO detections: " + ", ".join(
            f"{item.label} ({item.confidence:.0%}, {item.position})" for item in verified_objects
        ) + "."
    content = [{"type": "text", "text": (
        f"Previous scene key: {previous_scene_key or 'none'}. "
        f"Analyze the current {'frame sequence' if task == 'sign' else 'frame'}.{detector_context}"
    )}]
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


async def _call_groq_sign(
    images: list[tuple[bytes, str]], previous_transcript: str
) -> SignTranslation:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "Vision service is not configured")
    content = [{
        "type": "text",
        "text": (
            "Previously accepted transcript (preserve exactly): "
            f"{json.dumps(previous_transcript or '[empty]')}. Decode the next chronological window."
        ),
    }]
    for image, mime in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        })
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SIGN_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sign_translation",
                "strict": True,
                "schema": SIGN_TRANSLATION_SCHEMA,
            },
        },
        "reasoning_effort": "none",
        "temperature": 0.1,
        "max_completion_tokens": 260,
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
                result = SignTranslation.model_validate(json.loads(raw))
                # The model may only append to an accepted transcript. Enforce that contract
                # server-side so a noisy window cannot rewrite a user's conversation.
                if previous_transcript and not result.transcript.startswith(previous_transcript):
                    return SignTranslation(
                        transcript=previous_transcript,
                        segment="",
                        confidence=0,
                        is_final=False,
                        should_speak=False,
                    )
                continuation = result.transcript[len(previous_transcript):].strip()
                result.segment = continuation
                if result.confidence < 0.68 or not result.segment.strip():
                    return SignTranslation(
                        transcript=previous_transcript,
                        segment="",
                        confidence=result.confidence,
                        is_final=False,
                        should_speak=False,
                    )
                return result
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise HTTPException(504, "Vision provider timed out")
                await asyncio.sleep(0.5 * (2**attempt) + random.random())
            except (KeyError, json.JSONDecodeError, ValidationError):
                raise HTTPException(502, "Vision provider returned an invalid response")
            except httpx.HTTPStatusError as exc:
                logger.warning("groq_sign_error status=%s", exc.response.status_code)
                raise HTTPException(502, "Vision provider request failed")
    raise HTTPException(502, "Vision provider request failed")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "groq",
        "configured": bool(os.getenv("GROQ_API_KEY")),
        "model": MODEL,
        "yolo": {"enabled": YOLO_ENABLED, "model": YOLO_MODEL if YOLO_ENABLED else None},
        "depth": {"enabled": _depth_model.enabled, "ready": _depth_model.ready},
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
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "Vision service is not configured")
    started = time.monotonic()
    async with _slots:
        verified_objects = (
            await asyncio.to_thread(_detect_with_yolo, body, client_id)
            if task == "scene" else []
        )
        warnings = []
        if verified_objects and _depth_model.ready:
            depth_map = await asyncio.to_thread(_depth_model.infer, body)
            readings = readings_for_boxes(depth_map, verified_objects)
            distances = {
                (reading.label, round(reading.x, 3), round(reading.y, 3)): reading.depth_m
                for reading in readings
            }
            verified_objects = [
                item.model_copy(update={
                    "distance_m": distances.get((item.label, round(item.x or 0, 3), round(item.y or 0, 3))),
                })
                for item in verified_objects
            ]
            warnings = _approach_tracker.update(client_id, readings)
        result = await _call_groq(
            [(body, image.content_type)], previous_scene_key[:80], task, verified_objects,
        )
    # Geometry comes from the detector, never the language model. This lets the
    # mobile client draw accurate fallback boxes when its local model is absent.
    if verified_objects:
        result.objects = verified_objects[:8]
    if warnings:
        result.approaching = [
            ApproachingWarning(label=warning.label, distance_m=round(warning.distance_m, 1))
            for warning in warnings[:3]
        ]
        lead = result.approaching[0]
        result.summary = f"Caution: {lead.label} approaching"
        result.speech = f"Caution. {lead.label} is getting closer, about {lead.distance_m:.1f} meters away."
        result.urgency = "high"
        result.changed = True
    logger.info(
        "vision_complete request_id=%s latency_ms=%d bytes=%d changed=%s urgency=%s",
        request_id,
        int((time.monotonic() - started) * 1000),
        len(body),
        result.changed,
        result.urgency,
    )
    return result


@app.post("/v1/interpret-sign", response_model=SignTranslation)
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
        result = await _call_groq_sign(decoded, sequence.previous_transcript.strip())
    logger.info(
        "sign_complete latency_ms=%d frame_count=%d total_bytes=%d",
        int((time.monotonic() - started) * 1000),
        len(decoded),
        sum(len(body) for body, _ in decoded),
    )
    return result
