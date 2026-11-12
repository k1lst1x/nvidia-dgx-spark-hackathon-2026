import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import threading
import random
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

import campus_memory
import observation_store
from camera_coach import assess_frame
from depth import (
    DepthAnythingMetric,
    relative_proximity_for_boxes,
    surface_readings_for_boxes,
)
from gpu_status import gpu_status
from motion import MotionHistory
from obstacle_points import OBSTACLE_MAX_RANGE_M, OBSTACLE_POINTS_ENABLED, obstacle_points
from point_cloud import cloud_for_surface, open3d_status
from rate_limit import LimitExceeded, LimitRule, TieredSlidingWindowLimiter
from segmentation import YoloSegmenter
from slam import OrbSlamClient
from visualization import RerunView
from schemas import (
    BridgeConversation,
    BridgeMessage,
    BridgeMessageRequest,
    CameraCoach,
    CampusAlert,
    CampusAlertCreateRequest,
    DetectedObject,
    SignSequenceRequest,
    SignTranslation,
    Observation,
    ObservationChanges,
    ObservationCreateRequest,
    SharedSightDashboard,
    TrackingResult,
    UniSignRawRequest,
    VisionResult,
    WhatChangedRequest,
    WhatChangedResponse,
)
from unisign_backend import UniSignBackend, UniSignBackendError
from sign_english import english_text, yolo_sign_phrase
from yolo_sign import YoloSignDetector
from device_manager import report as device_report
from yolo_tracker import YoloTracker

VISION_API_URL = os.getenv("VISION_API_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.getenv("VISION_MODEL") or os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
VISION_PROVIDER = "groq" if "api.groq.com" in VISION_API_URL else "local"
VISION_DISABLE_THINKING = os.getenv("VISION_DISABLE_THINKING", "false").lower() in {"1", "true", "yes"}


def _vision_api_key() -> str | None:
    return os.getenv("VISION_API_KEY") or os.getenv("GROQ_API_KEY")


def _vision_configured() -> bool:
    return VISION_PROVIDER == "local" or bool(_vision_api_key())


def _vision_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _vision_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _apply_vision_options(payload: dict) -> dict:
    if VISION_DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
# With a local model on the DGX Spark, Groq's cloud is only a backup.
VISION_FALLBACK_TO_GROQ = os.getenv("VISION_FALLBACK_TO_GROQ", "true").lower() in {"1", "true", "yes"}
SPARK_VISION_TIMEOUT_S = float(os.getenv("SPARK_VISION_TIMEOUT_S", "10"))
_last_vision_answer: dict[str, object] = {"provider": None, "at": None, "fell_back": False}


def _vision_providers(interactive: bool = False) -> list[dict[str, object]]:
    """Ordered vision endpoints: the Spark first, Groq only as backup."""
    providers: list[dict[str, object]] = []
    if VISION_PROVIDER == "local":
        providers.append({
            "name": "spark", "url": VISION_API_URL, "model": MODEL, "key": os.getenv("VISION_API_KEY"),
            "timeout": min(SPARK_VISION_TIMEOUT_S, 8.0) if interactive else SPARK_VISION_TIMEOUT_S, "attempts": 1,
        })
        groq_key = os.getenv("GROQ_API_KEY")
        if VISION_FALLBACK_TO_GROQ and groq_key:
            providers.append({"name": "groq", "url": GROQ_API_URL, "model": GROQ_MODEL, "key": groq_key,
                              "timeout": 8.0 if interactive else 12.0, "attempts": 2 if interactive else 3})
    elif _vision_configured():
        providers.append({"name": "groq", "url": VISION_API_URL, "model": MODEL, "key": _vision_api_key(),
                          "timeout": 8.0 if interactive else 12.0, "attempts": 2 if interactive else 3})
    return providers


async def _post_vision(payload: dict, parse, *, interactive: bool = False):
    """Send a chat request to each provider in order and return the first valid parse.

    A provider that times out, is unreachable, rate-limits, errors, or returns
    output that does not parse is skipped for the next one, so a Spark outage
    falls back to Groq instead of failing the request.
    """
    providers = _vision_providers(interactive)
    if not providers:
        raise HTTPException(503, "Vision service is not configured")
    failure = HTTPException(502, "Vision provider request failed")
    for index, provider in enumerate(providers):
        headers = {"Content-Type": "application/json"}
        if provider["key"]:
            headers["Authorization"] = f"Bearer {provider['key']}"
        body = _apply_vision_options({**payload, "model": provider["model"]})
        timeout = httpx.Timeout(float(provider["timeout"]), connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(int(provider["attempts"])):
                try:
                    response = await client.post(str(provider["url"]), headers=headers, json=body)
                    if response.status_code == 429 or response.status_code >= 500:
                        failure = HTTPException(
                            429 if response.status_code == 429 else 502,
                            "Vision provider is rate-limited; try again shortly" if response.status_code == 429
                            else "Vision provider request failed",
                        )
                        if attempt + 1 < int(provider["attempts"]):
                            retry = float(response.headers.get("retry-after", 0) or 0)
                            await asyncio.sleep(max(retry, min(0.5 * (2**attempt) + random.random(), 4)))
                            continue
                        break
                    response.raise_for_status()
                    parsed = parse(response.json()["choices"][0]["message"]["content"])
                    _last_vision_answer.update(provider=provider["name"], at=time.time(), fell_back=index > 0)
                    if index:
                        logger.warning("vision_fallback_used provider=%s", provider["name"])
                    return parsed
                except (httpx.TimeoutException, httpx.NetworkError):
                    failure = HTTPException(504, "Vision provider timed out")
                    if attempt + 1 < int(provider["attempts"]):
                        await asyncio.sleep(0.5 * (2**attempt) + random.random())
                        continue
                    break
                except (KeyError, IndexError, json.JSONDecodeError, ValidationError):
                    failure = HTTPException(502, "Vision provider returned an invalid response")
                    break
                except httpx.HTTPStatusError as exc:
                    logger.warning("vision_error provider=%s status=%s", provider["name"], exc.response.status_code)
                    failure = HTTPException(502, "Vision provider request failed")
                    break
        logger.warning("vision_provider_failed provider=%s detail=%s", provider["name"], failure.detail)
    raise failure


MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "1500000"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "1.5"))
TRACK_MIN_REQUEST_INTERVAL = float(os.getenv("TRACK_MIN_REQUEST_INTERVAL", "0.7"))
SIGN_MIN_CONFIDENCE = float(os.getenv("SIGN_MIN_CONFIDENCE", "0.60"))
SIGN_FRAME_WIDTH = int(os.getenv("SIGN_FRAME_WIDTH", "320"))
YOLO_ENABLED = os.getenv("ULTRALYTICS_YOLO_ENABLED", "false").lower() in {"1", "true", "yes"}
POINT_CLOUD_ENABLED = os.getenv("POINT_CLOUD_ENABLED", "false").lower() in {"1", "true", "yes"}
POINT_CLOUD_INTERVAL_S = float(os.getenv("POINT_CLOUD_INTERVAL_S", "0.5"))
YOLO_MODEL = os.getenv("ULTRALYTICS_YOLO_MODEL", "yolo11n.pt")
# A larger model is cheap on a GPU and much slower on CPU, so CPU can use its own.
YOLO_CPU_MODEL = os.getenv("ULTRALYTICS_YOLO_CPU_MODEL") or None
YOLO_CONFIDENCE = float(os.getenv("ULTRALYTICS_YOLO_CONFIDENCE", "0.40"))
# One Ultralytics model instance is shared by the API process. Keep the BoT-SORT
# state scoped to one active camera feed so object IDs never bleed between users.
YOLO_TRACK_IDLE_SECONDS = float(os.getenv("YOLO_TRACK_IDLE_SECONDS", "4"))
YOLO_TRACKER_CONFIG = os.getenv("YOLO_TRACKER_CONFIG", str(Path(__file__).with_name("botsort.yaml")))
# Never enable collision/mobility alerts based on experimental monocular geometry.
MOBILITY_COLLISION_WARNINGS_ENABLED = os.getenv("MOBILITY_COLLISION_WARNINGS_ENABLED", "false").lower() in {"1", "true", "yes"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("guidelens")
app = FastAPI(title="GuideLens Vision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    # Expo/Web and Electron serve the local app via dynamic or standard loopback ports.
    # Support loopback origins on any port by default; production still supplies its own
    # comma-separated CORS_ORIGINS value.
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:8081,http://127.0.0.1:8081,http://localhost:8082,http://127.0.0.1:8082"
    ).split(","),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:[0-9]+)?$",
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Session-ID",
        "X-Device-ID",
        "X-Device-Secret",
        "X-Shared-Sight-Key",
        "X-Moderator-Key",
        "Authorization",
    ],
)

_slots = asyncio.Semaphore(MAX_CONCURRENCY)
_general_limits = TieredSlidingWindowLimiter(
    session=LimitRule(20, 60, MIN_REQUEST_INTERVAL, detail="Session rate limit exceeded"),
    peer=LimitRule(60, 60, detail="Source rate limit exceeded"),
    global_budget=LimitRule(240, 60, max_keys=1, detail="Inference capacity limit exceeded"),
)
_track_limits = TieredSlidingWindowLimiter(
    session=LimitRule(16, 10, TRACK_MIN_REQUEST_INTERVAL, detail="Tracking session rate limit exceeded"),
    peer=LimitRule(64, 10, detail="Tracking source rate limit exceeded"),
    global_budget=LimitRule(256, 10, max_keys=1, detail="Tracking capacity limit exceeded"),
)
_bridge_limits = TieredSlidingWindowLimiter(
    device=LimitRule(8, 300, detail="Bridge join limit exceeded"),
    peer=LimitRule(10, 300, detail="Bridge join limit exceeded"),
    global_budget=LimitRule(100, 60, max_keys=1, detail="Bridge join capacity exceeded"),
)
_memory_write_limits = TieredSlidingWindowLimiter(
    session=LimitRule(60, 60, detail="Memory rate limit exceeded"),
    peer=LimitRule(120, 60, detail="Memory source rate limit exceeded"),
    global_budget=LimitRule(600, 60, max_keys=1, detail="Memory capacity limit exceeded"),
)
_depth_model = DepthAnythingMetric()
_segmenter = YoloSegmenter()
_slam = OrbSlamClient()
_rerun = RerunView()
_cloud_at = 0.0
_cloud_cache: dict[tuple, dict[str, object]] = {}
_unisign_backend = UniSignBackend()
_yolo_sign_detector = YoloSignDetector()
_motion_history = MotionHistory()
_yolo_tracker = YoloTracker(
    enabled=YOLO_ENABLED, model_path=YOLO_MODEL, cpu_model_path=YOLO_CPU_MODEL,
    confidence=YOLO_CONFIDENCE, tracker_config=YOLO_TRACKER_CONFIG,
    idle_seconds=YOLO_TRACK_IDLE_SECONDS, on_reset=_motion_history.reset,
)


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

Write transcript and segment only in natural, grammatical written English: reorder ASL gloss into
English word order, add the small function words English needs (is, the, a, to), use sentence case
with ending punctuation, and turn WH-questions into English questions (BATHROOM WHERE -> "Where is
the bathroom?"). Never output ASL gloss in capitals, another language, or a description of the
gestures. Fingerspelled names stay as the spelled English word.

Return transcript as the complete previous transcript plus only a genuinely new continuation.
Return segment as only that new continuation, or an empty string when the window is unclear,
duplicated, transitional, or contains no signing. Preserve an already accepted transcript verbatim.
Set confidence conservatively. is_final means a phrase boundary is visible (pause or hands lowered),
not merely that one window ended. should_speak is true only for a non-empty, confident, final new
segment. Output only the requested JSON."""


def _peer_key(request: Request) -> str:
    # Deliberately ignore forwarding headers: they are attacker-controlled unless
    # a deployment has an explicit trusted-proxy policy.
    return request.client.host if request.client else "unknown"


def _session_id(request: Request) -> str:
    value = request.headers.get("X-Session-ID") or _peer_key(request)
    if len(value) > 100:
        raise HTTPException(422, "X-Session-ID must contain at most 100 characters")
    return value


def _limited_key(peer: str, session_id: str) -> str:
    return peer + ":" + hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _apply_limits(limiter: TieredSlidingWindowLimiter, *checks: tuple[str, str]) -> None:
    try:
        limiter.check(*checks)
    except LimitExceeded as error:
        raise HTTPException(
            429, error.detail, headers={"Retry-After": str(error.retry_after)},
        ) from error


def _rate_limit(request: Request, session_id: str) -> None:
    peer = _peer_key(request)
    _apply_limits(
        _general_limits,
        ("session", _limited_key(peer, session_id)),
        ("peer", peer),
        ("global_budget", "global"),
    )


def _track_rate_limit(request: Request, session_id: str) -> None:
    """Allow fast tracking while enforcing independent peer/global ceilings."""
    peer = _peer_key(request)
    _apply_limits(
        _track_limits,
        ("session", _limited_key(peer, session_id)),
        ("peer", peer),
        ("global_budget", "global"),
    )


def _bridge_join_rate_limit(request: Request, device_id: str) -> None:
    _apply_limits(
        _bridge_limits,
        ("device", device_id),
        ("peer", _peer_key(request)),
        ("global_budget", "global"),
    )


def _memory_rate_limit(request: Request, session_id: str) -> None:
    peer = _peer_key(request)
    _apply_limits(
        _memory_write_limits,
        ("session", _limited_key(peer, session_id)),
        ("peer", peer),
        ("global_budget", "global"),
    )


def _clear_rate_limits_for_tests() -> None:
    for limiter in (_general_limits, _track_limits, _bridge_limits, _memory_write_limits):
        limiter.clear()


def _local_scene_result(objects: list[DetectedObject]) -> VisionResult:
    """Keep verified local vision usable when the optional cloud narrator fails."""
    labels = list(dict.fromkeys(item.label for item in objects))[:3]
    description = ", ".join(labels) if labels else "no recognized objects"
    return VisionResult(
        summary=f"Detected {description}",
        speech=f"Detected {description}.",
        scene_key=f"local:{'-'.join(labels) or 'empty'}",
        urgency="low",
        changed=True,
        objects=objects[:8],
        task="scene",
    )


def _with_mask_stats(objects: list[DetectedObject], masks: list) -> list[DetectedObject]:
    updated = []
    for item, mask in zip(objects, masks):
        if mask is None:
            updated.append(item)
            continue
        coverage = float(mask.mean()) if getattr(mask, "size", 0) else 0.0
        spatial = dict(item.spatial or {})
        spatial.update({"mask_coverage": round(coverage, 4), "estimate": True})
        updated.append(item.model_copy(update={"spatial": spatial}))
    return updated


def _object_key(item: DetectedObject) -> tuple:
    return (item.track_id, item.label, round(item.x or 0, 3), round(item.y or 0, 3))


def _with_point_clouds(objects: list[DetectedObject], depth_map, masks: list) -> list[DetectedObject]:
    global _cloud_at
    if depth_map is None:
        return objects
    now = time.monotonic()
    if now - _cloud_at < POINT_CLOUD_INTERVAL_S and _cloud_cache:
        updated = []
        for item in objects:
            cached = _cloud_cache.get(_object_key(item))
            if cached is None:
                updated.append(item)
                continue
            spatial = dict(item.spatial or {})
            spatial.update(cached)
            updated.append(item.model_copy(update={"spatial": spatial}))
        return updated
    height, width = depth_map.shape[:2]
    updated = []
    fresh: dict[tuple, dict[str, object]] = {}
    for index, item in enumerate(objects):
        segmented = bool(masks and index < len(masks) and masks[index] is not None)
        mask = masks[index] if segmented else None
        if mask is None and None not in (item.x, item.y, item.width, item.height):
            import numpy as np

            mask = np.zeros(depth_map.shape[:2], dtype=bool)
            left = max(0, int((item.x or 0) * width))
            right = min(width, int(((item.x or 0) + (item.width or 0)) * width))
            top = max(0, int((item.y or 0) * height))
            bottom = min(height, int(((item.y or 0) + (item.height or 0)) * height))
            mask[top:bottom, left:right] = True
        summary = cloud_for_surface(depth_map, mask, width, height)
        if summary is None:
            updated.append(item)
            continue
        summary["depth_source"] = "mask" if segmented else "box"
        fresh[_object_key(item)] = summary
        spatial = dict(item.spatial or {})
        spatial.update(summary)
        updated.append(item.model_copy(update={"spatial": spatial}))
    _cloud_cache.clear()
    _cloud_cache.update(fresh)
    _cloud_at = now
    return updated


_INFER_DEPTH = object()


def _add_depth_and_motion(
    objects: list[DetectedObject], image_bytes: bytes, session_id: str, depth_map=_INFER_DEPTH, masks: list | None = None,
) -> list[DetectedObject]:
    """Attach conservative depth/motion debug data without calling Groq.

    Relative-depth checkpoints can only expose within-frame ordering, so they
    intentionally never produce distance, velocity, TTC, or warnings.
    Segmentation and point clouds are optional and never replace the tracker.
    """
    if not objects:
        return objects
    if masks is None or len(masks) != len(objects):
        try:
            masks = _segmenter.masks_for(image_bytes, objects)
        except Exception:
            logger.exception("segmentation_skipped")
            masks = [None] * len(objects)
    objects = _with_mask_stats(objects, masks)
    pose_snapshot = _slam.latest(session_id, image_bytes)
    if not _depth_model.ready:
        _rerun.log(objects)
        return objects
    if depth_map is _INFER_DEPTH:
        depth_map = _depth_model.infer(image_bytes)
    if depth_map is None:
        _rerun.log(objects)
        return objects
    if _depth_model.is_metric:
        surface_readings = surface_readings_for_boxes(depth_map, objects, masks)
        by_track_reading = {
            reading.track_id: reading for reading in surface_readings if reading.track_id is not None
        }
        by_track = {
            track_id: (reading.depth_m, reading.confidence, reading.spread_m)
            for track_id, reading in by_track_reading.items()
        }
        by_box = {
            (reading.label, round(reading.x, 3), round(reading.y, 3)): reading
            for reading in surface_readings
        }
        pose = None
        if pose_snapshot and pose_snapshot.get("metric_scale"):
            pose = _slam.pose_for_motion(depth_map.shape[1], depth_map.shape[0], pose_snapshot)
        estimates = _motion_history.update(session_id, objects, by_track, pose=pose)
        updated = []
        for item in objects:
            reading = by_track_reading.get(item.track_id) if item.track_id is not None else None
            if reading is None:
                reading = by_box.get((item.label, round(item.x or 0, 3), round(item.y or 0, 3)))
            spatial = dict(item.spatial or {})
            if reading is not None:
                spatial.update({
                    "depth_source": reading.source,
                    "depth_confidence": round(reading.confidence, 3),
                    "estimate": True,
                })
            updated.append(item.model_copy(update={
                "distance_m": reading.depth_m if reading else None,
                "motion": estimates[item.track_id].as_dict() if item.track_id in estimates else None,
                "spatial": spatial or None,
            }))
        objects = updated
        if POINT_CLOUD_ENABLED:
            try:
                objects = _with_point_clouds(objects, depth_map, masks)
            except Exception:
                logger.exception("point_cloud_skipped")
        _rerun.log(objects)
        return objects
    proximity = relative_proximity_for_boxes(depth_map, objects, masks)
    _rerun.log(objects)
    return [
        item.model_copy(update={
            "relative_proximity": proximity.get((item.label, round(item.x or 0, 3), round(item.y or 0, 3))),
        })
        for item in objects
    ]


async def _call_groq(
    images: list[tuple[bytes, str]],
    previous_scene_key: str,
    task: str,
    verified_objects: list[DetectedObject] | None = None,
) -> VisionResult:
    if not _vision_configured():
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
    return await _post_vision(payload, lambda raw: VisionResult.model_validate(json.loads(raw)))
    raise HTTPException(502, "Vision provider request failed")


async def _call_groq_sign(
    images: list[tuple[bytes, str]], previous_transcript: str
) -> SignTranslation:
    if not _vision_configured():
        raise HTTPException(503, "Vision service is not configured")
    content = [{
        "type": "text",
        "text": (
            "Previously accepted transcript (preserve exactly): "
            f"{json.dumps(previous_transcript or '[empty]')}. The image is a chronological contact sheet: "
            "read the signing sequence from left to right and decode the next window."
        ),
    }]
    # Vision providers commonly charge a minimum token budget for *each*
    # image. Combine the three chronological samples into one compact strip;
    # it preserves motion order while avoiding triple image-token usage.
    try:
        from PIL import Image

        frames = [Image.open(BytesIO(image)).convert("RGB") for image, _ in images]
        # Handshapes are unreadable below roughly 250 px across the signer.
        thumb_width = min(SIGN_FRAME_WIDTH, max(frame.width for frame in frames))
        thumbs = [frame.resize((thumb_width, max(32, round(frame.height * thumb_width / frame.width)))) for frame in frames]
        height = max(frame.height for frame in thumbs)
        strip = Image.new("RGB", (thumb_width * len(thumbs), height), "black")
        for index, frame in enumerate(thumbs):
            strip.paste(frame, (index * thumb_width, 0))
        buffer = BytesIO()
        strip.save(buffer, format="JPEG", quality=72, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    except Exception as error:
        raise HTTPException(422, "Sign frames could not be prepared") from error
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SIGN_PROMPT},
            {"role": "user", "content": content},
        ],
        # JSON mode is substantially lower latency than schema-constrained
        # decoding for multimodal Qwen requests. Pydantic still validates the
        # response before it reaches the client.
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_completion_tokens": 120,
        "stream": False,
    }
    # Sign Context is interactive. Do not leave the camera UI waiting through
    # three long cloud retries; the next overlapping window can try again.
    # Sign Context is interactive: short timeouts, and the next overlapping
    # window can try again.
    result = await _post_vision(
        payload, lambda raw: SignTranslation.model_validate(json.loads(raw)), interactive=True,
    )
    combined = f"{result.transcript} {result.segment}".strip()
    if combined and not english_text(combined):
        return SignTranslation(
            transcript=previous_transcript,
            segment="",
            confidence=0,
            is_final=False,
            should_speak=False,
        )
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
    # Some vision providers fill ``segment`` correctly but omit the
    # accumulated transcript. Normalize that valid first-window
    # response instead of silently throwing it away in the UI.
    if not result.transcript.strip() and result.segment.strip():
        result.transcript = f"{previous_transcript} {result.segment}".strip()
    continuation = english_text(result.transcript[len(previous_transcript):])
    result.segment = continuation
    result.transcript = f"{previous_transcript} {continuation}".strip()
    if result.confidence < SIGN_MIN_CONFIDENCE or not result.segment.strip():
        return SignTranslation(
            transcript=previous_transcript,
            segment="",
            confidence=result.confidence,
            is_final=False,
            should_speak=False,
        )
    return result
    raise HTTPException(502, "Vision provider request failed")


def _warm_up_models() -> None:
    """Load detector and depth weights before the first camera frame arrives.

    A cold first frame takes over ten seconds on a GPU, and the app discards
    tracking replies older than three seconds.
    """
    try:
        import cv2
        import numpy as np

        ok, encoded = cv2.imencode(".jpg", np.full((480, 640, 3), 127, dtype=np.uint8))
        if not ok:
            return
        frame = encoded.tobytes()
        if YOLO_ENABLED:
            _yolo_tracker.load()
            _yolo_tracker.detect_with_masks(frame, "__warmup__")
        if _depth_model.ready:
            _depth_model.infer(frame)
        logger.info("model_devices %s", {name: value.get("active") for name, value in device_report()["components"].items()})
    except Exception:
        logger.exception("model_warmup_failed")


@app.on_event("startup")
async def _start_warm_up() -> None:
    if os.getenv("MODEL_WARMUP", "true").lower() in {"1", "true", "yes"}:
        threading.Thread(target=_warm_up_models, name="model-warmup", daemon=True).start()


@app.get("/health")
@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "provider": VISION_PROVIDER,
        "gpu": gpu_status(),
        "configured": _vision_configured(),
        "model": MODEL,
        "vision_providers": [str(provider["name"]) for provider in _vision_providers()],
        "last_vision_answer": dict(_last_vision_answer),
        "devices": device_report(),
        "yolo": {
            "enabled": YOLO_ENABLED,
            "model": _yolo_tracker.model_path if YOLO_ENABLED else None,
            "tracker": "botsort",
            "camera_motion_compensation": "sparseOptFlow",
            "segmentation_enabled": _segmenter.enabled,
            "segmentation_model": _segmenter.model_path if _segmenter.enabled else None,
            "segmentation_last_ms": _segmenter.last_ms,
            **_yolo_tracker.status(),
        },
        "depth": {
            **_depth_model.status(),
            "point_cloud_enabled": POINT_CLOUD_ENABLED,
            "obstacle_points_enabled": OBSTACLE_POINTS_ENABLED,
            "obstacle_range_m": OBSTACLE_MAX_RANGE_M,
            "open3d": open3d_status()["installed"],
        },
        "slam": _slam.status(),
        "rerun": _rerun.status(),
        # Asset readiness only; this endpoint must not allocate the 1.19 GB
        # Uni-Sign checkpoint or mT5 decoder just to answer a health probe.
        "unisign": _unisign_backend.status(),
        "motion": {
            "experimental": True,
            "requires_metric_depth_for_motion": True,
            "collision_warnings_enabled": MOBILITY_COLLISION_WARNINGS_ENABLED,
        },
        "shared_sight": {
            "enabled": True,
            "device_allowlist_enabled": bool(os.getenv("SHARED_SIGHT_DEVICE_IDS")),
            "shared_auth_configured": bool(os.getenv("SHARED_SIGHT_TOKEN")),
            "minimum_confidence": observation_store.MIN_CONFIDENCE,
            "verified_shared_minimum_confidence": observation_store.VERIFIED_SHARED_MIN_CONFIDENCE,
        },
        "time": datetime.now(timezone.utc).isoformat(),
    }


async def _read_tracking_image(image: UploadFile) -> bytes:
    if image.content_type not in {"image/jpeg", "image/webp", "image/png"}:
        raise HTTPException(415, "JPEG, WebP, or PNG required")
    body = await image.read(MAX_IMAGE_BYTES + 1)
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is empty or too large")
    return body


def _obstacle_points_active() -> bool:
    return OBSTACLE_POINTS_ENABLED and _depth_model.ready and _depth_model.is_metric


def _track_geometry(
    objects: list[DetectedObject], image_bytes: bytes, session_id: str, masks: list | None = None,
) -> tuple[list[DetectedObject], list[list[float]]]:
    """Run depth once per tracking frame for both object distances and obstacle dots.

    Obstacle dots cover surfaces the detector cannot name (walls, trees, poles),
    so they are produced even when no object was detected.
    """
    if not _obstacle_points_active():
        return _add_depth_and_motion(objects, image_bytes, session_id, masks=masks), []
    depth_map = _depth_model.infer(image_bytes)
    objects = _add_depth_and_motion(objects, image_bytes, session_id, depth_map=depth_map, masks=masks)
    try:
        # The response schema allows at most 512, whatever OBSTACLE_MAX_POINTS says.
        obstacles = obstacle_points(depth_map, objects)[:512]
    except Exception:
        logger.exception("obstacle_points_skipped")
        obstacles = []
    return objects, obstacles


@app.post("/v1/track", response_model=TrackingResult)
async def track_scene(request: Request, image: UploadFile = File(...)):
    """Fast detector/depth/motion path, intentionally separate from Groq."""
    client_id = _session_id(request)
    body = await _read_tracking_image(image)
    _track_rate_limit(request, client_id)
    started = time.monotonic()
    async with _slots:
        objects, masks = await asyncio.to_thread(_yolo_tracker.detect_with_masks, body, client_id)
        objects, obstacles = await asyncio.to_thread(_track_geometry, objects, body, client_id, masks)
        try:
            camera_coach = await asyncio.to_thread(assess_frame, body, objects)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    latency_ms = int((time.monotonic() - started) * 1000)
    observation_store.record_tracking_latency(latency_ms)
    obstacles_enabled = _obstacle_points_active()
    return TrackingResult(
        objects=objects,
        tracking_latency_ms=latency_ms,
        detector="yolo11-botsort" if YOLO_ENABLED else "disabled",
        metric_depth=_depth_model.ready and _depth_model.is_metric,
        camera_coach=camera_coach,
        collision_warnings_enabled=False,
        segmentation=_segmenter.enabled,
        point_cloud=POINT_CLOUD_ENABLED and _depth_model.ready and _depth_model.is_metric,
        camera_pose=_slam.snapshot_for(client_id),
        obstacle_points=obstacles,
        obstacle_range_m=OBSTACLE_MAX_RANGE_M if obstacles_enabled else None,
    )


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
    client_id = _session_id(request)
    body = await _read_tracking_image(image)
    if not _vision_configured():
        raise HTTPException(503, "Vision service is not configured")
    _rate_limit(request, client_id)
    started = time.monotonic()
    async with _slots:
        verified_objects, verified_masks = (
            await asyncio.to_thread(_yolo_tracker.detect_with_masks, body, client_id)
            if task == "scene" else ([], None)
        )
        if verified_objects and _depth_model.ready:
            verified_objects = await asyncio.to_thread(
                _add_depth_and_motion, verified_objects, body, client_id, _INFER_DEPTH, verified_masks,
            )
        try:
            result = await _call_groq(
                [(body, image.content_type)], previous_scene_key[:80], task, verified_objects,
            )
        except HTTPException as error:
            if task != "scene" or not verified_objects:
                raise
            logger.warning("cloud_scene_unavailable_using_local_detection detail=%s", error.detail)
            result = _local_scene_result(verified_objects)
    # Geometry comes from the detector, never the language model. This lets the
    # mobile client draw accurate fallback boxes when its local model is absent.
    if verified_objects:
        result.objects = verified_objects[:8]
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
    client_id = _session_id(request)
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
    _rate_limit(request, client_id)
    started = time.monotonic()
    try:
        async with _slots:
            result = await _call_groq_sign(decoded, sequence.previous_transcript.strip())
    except Exception as exc:
        # Fallback to local YOLO11 Sign Language Detector on the last frame
        last_frame_bytes = decoded[-1][0] if decoded else b""
        detected = _yolo_sign_detector.detect(last_frame_bytes) if last_frame_bytes else []
        words = [(sign, yolo_sign_phrase(sign.label)) for sign in detected]
        words = [
            (sign, phrase)
            for sign, phrase in words
            if phrase and sign.confidence >= SIGN_MIN_CONFIDENCE
        ]
        if words:
            top, sign_word = words[0]
            prev = sequence.previous_transcript.strip()
            norm_prev = prev.rstrip(".!?, ").lower()
            norm_word = sign_word.rstrip(".!?, ").lower()
            if norm_prev.endswith(norm_word):
                # The gesture is being held across consecutive sample windows; do not duplicate
                new_transcript = prev
                segment = ""
                should_speak = False
            else:
                new_transcript = (prev + " " + sign_word).strip() if prev else sign_word
                segment = sign_word
                should_speak = True
            result = SignTranslation(
                transcript=new_transcript,
                segment=segment,
                confidence=top.confidence,
                is_final=True,
                should_speak=should_speak,
            )
        else:
            raise exc

    logger.info(
        "sign_complete latency_ms=%d frame_count=%d total_bytes=%d",
        int((time.monotonic() - started) * 1000),
        len(decoded),
        sum(len(body) for body, _ in decoded),
    )
    return result


@app.post("/v1/detect-sign")
async def detect_sign_yolo(request: Request, file: UploadFile = File(...)):
    """Runs YOLO11 Sign Language detector on an uploaded frame."""
    session_id = _session_id(request)
    content = await _read_tracking_image(file)
    _rate_limit(request, session_id)
    async with _slots:
        signs = await asyncio.to_thread(_yolo_sign_detector.detect, content)
    primary = signs[0].label if signs else ""
    conf = signs[0].confidence if signs else 0.0
    return {
        "available": _yolo_sign_detector.available,
        "detected_signs": signs,
        "primary_sign": primary,
        "confidence": conf,
    }


@app.get("/v1/unisign/status")
async def unisign_status():
    """Expose only opt-in backend readiness; never load a large model on health checks."""
    return _unisign_backend.status()


@app.post("/v1/unisign/interpret")
async def interpret_unisign_experimental(request: Request, sequence: UniSignRawRequest):
    """Research endpoint for explicitly opted-in MediaPipe compatibility runs."""
    if not _unisign_backend.allow_presence_proxy or not sequence.experimental:
        raise HTTPException(
            503,
            "MediaPipe-to-Uni-Sign is experimental: set experimental=true and "
            "UNISIGN_ALLOW_MEDIAPIPE_PRESENCE_PROXY=true after evaluation.",
        )
    if not _unisign_backend.enabled:
        raise HTTPException(503, "Uni-Sign is disabled")
    device_id, session_id = _authorize_sight(request)
    _rate_limit(request, session_id)
    async with _slots:
        try:
            raw_dicts = [f.model_dump(exclude_none=True) for f in sequence.raw_holistic]
            return await asyncio.to_thread(
                _unisign_backend.translate_raw_mediapipe, raw_dicts, experimental=sequence.experimental
            )
        except UniSignBackendError as error:
            raise HTTPException(503, str(error)) from error


@app.get("/v1/memory/alerts", response_model=list[CampusAlert])
async def get_campus_alerts(request: Request, location: str | None = None):
    """Shared, verified campus reports for a room (one store for every phone)."""
    device_id, session_id = _authorize_sight(request)
    # A camera polling for alerts is an active phone on the dashboard.
    observation_store.touch_device(device_id, session_id)
    return [CampusAlert.model_validate(alert) for alert in observation_store.shared_alerts(device_id, location)]


def _authorize_sight(request: Request) -> tuple[str, str]:
    try:
        shared_key = request.headers.get("X-Shared-Sight-Key") or request.headers.get("X-Moderator-Key")
        return observation_store.authorize_device(
            request.headers.get("X-Device-ID"),
            request.headers.get("X-Session-ID"),
            request.headers.get("X-Device-Secret"),
            shared_key,
        )
    except observation_store.ObservationAccessError as error:
        raise HTTPException(401, str(error)) from error


def _is_trusted_shared_mutator(request: Request, device_id: str) -> bool:
    """Check whether caller is enrolled/authorized to mutate shared safety state."""
    token = os.getenv("SHARED_SIGHT_TOKEN", "").strip()
    allowed_devices = {v.strip() for v in os.getenv("SHARED_SIGHT_DEVICE_IDS", "").split(",") if v.strip()}
    if not token and not allowed_devices:
        return False
    if allowed_devices and device_id in allowed_devices:
        return True
    if token:
        supplied = request.headers.get("X-Shared-Sight-Key") or request.headers.get("X-Moderator-Key") or ""
        if supplied and hmac.compare_digest(supplied, token):
            return True
    return False


def _is_moderator(request: Request, device_id: str) -> bool:
    """Check whether caller possesses moderator/administrator privileges."""
    token = os.getenv("SHARED_SIGHT_TOKEN", "").strip()
    moderator_token = os.getenv("SHARED_SIGHT_MODERATOR_TOKEN", token).strip()
    allowed_devices = {v.strip() for v in os.getenv("SHARED_SIGHT_DEVICE_IDS", "").split(",") if v.strip()}
    if allowed_devices and device_id in allowed_devices:
        return True
    mod_key = request.headers.get("X-Moderator-Key", "").strip()
    if moderator_token and mod_key and hmac.compare_digest(mod_key, moderator_token):
        return True
    return False


@app.post("/v1/observations", response_model=Observation)
async def submit_observation(request: Request, body: ObservationCreateRequest):
    device_id, session_id = _authorize_sight(request)
    if body.privacy_scope == "shared":
        if not _is_trusted_shared_mutator(request, device_id):
            raise HTTPException(403, "Shared safety mutations require trusted authorization")
    try:
        result = observation_store.create_observation(
            zone_id=body.zone_id,
            category=body.category,
            label=body.label,
            description=body.description,
            confidence=body.confidence,
            privacy_scope=body.privacy_scope,
            verified=body.verified,
            room_x=body.room_x,
            room_y=body.room_y,
            expires_in_seconds=body.expires_in_seconds,
            origin=body.origin,
            source_device=device_id,
            source_session=session_id,
        )
    except observation_store.DuplicateObservationError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Observation.model_validate(result)


@app.get("/v1/observations/changes", response_model=ObservationChanges)
async def get_observation_changes(request: Request, zone_id: str, compared_seconds: int = 3600):
    _authorize_sight(request)
    if not zone_id.strip() or len(zone_id) > 80:
        raise HTTPException(422, "zone_id must contain 1 to 80 characters")
    return ObservationChanges.model_validate(observation_store.changes(zone_id, compared_seconds))


@app.get("/v1/personal/observations", response_model=list[Observation])
async def search_personal_observations(request: Request, query: str = "", limit: int = 50):
    device_id, _ = _authorize_sight(request)
    return [Observation.model_validate(item) for item in observation_store.search_personal(device_id, query, limit)]


@app.delete("/v1/observations/{observation_id}")
async def remove_owned_observation(request: Request, observation_id: int):
    device_id, _ = _authorize_sight(request)
    if not observation_store.delete_owned(observation_id, device_id):
        raise HTTPException(404, "Observation not found or not owned by this device")
    return {"status": "removed", "id": observation_id}


@app.get("/v1/shared-sight/dashboard", response_model=SharedSightDashboard)
async def shared_sight_dashboard(request: Request, zone_id: str, limit: int = 20):
    device_id, session_id = _authorize_sight(request)
    if not zone_id.strip() or len(zone_id) > 80:
        raise HTTPException(422, "zone_id must contain 1 to 80 characters")
    observation_store.touch_device(device_id, session_id)
    return SharedSightDashboard.model_validate({**observation_store.dashboard(zone_id, limit), "gpu": gpu_status()})


@app.post("/v1/memory/observe", response_model=CampusAlert)
async def observe_campus_hazard(request: Request, body: CampusAlertCreateRequest):
    """Record a hazard, obstacle, sign, or room status a person confirmed."""
    device_id, session_id = _authorize_sight(request)
    if not _is_trusted_shared_mutator(request, device_id):
        raise HTTPException(403, "Shared safety mutations require trusted authorization")
    _memory_rate_limit(request, session_id)
    try:
        alert = observation_store.report_alert(
            location=body.location, category=body.category, label=body.label,
            description=body.description, source_device=device_id, source_session=session_id,
        )
    except observation_store.DuplicateObservationError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return CampusAlert.model_validate(alert)


@app.post("/v1/memory/resolve/{alert_id}")
async def resolve_campus_alert(request: Request, alert_id: int):
    """Mark a campus hazard as cleared or resolved."""
    device_id, _session_id = _authorize_sight(request)
    is_mod = _is_moderator(request, device_id)
    try:
        resolved = observation_store.resolve_shared(alert_id, actor_device=device_id, is_moderator=is_mod)
    except observation_store.ObservationPermissionError as error:
        raise HTTPException(403, str(error)) from error
    if not resolved:
        raise HTTPException(404, "Alert not found")
    return {"status": "resolved", "id": alert_id}


@app.post("/v1/memory/what-changed", response_model=WhatChangedResponse)
async def what_changed(request: Request, body: WhatChangedRequest):
    """Compare current objects against baseline memory for the location."""
    device_id, session_id = _authorize_sight(request)
    _memory_rate_limit(request, session_id)
    is_mod = _is_moderator(request, device_id)
    baseline_exists = campus_memory.has_baseline(body.location)
    if body.update_baseline and baseline_exists and not is_mod:
        raise HTTPException(403, "Only an authorized moderator can replace an established room baseline")
    if not baseline_exists and not _is_trusted_shared_mutator(request, device_id):
        raise HTTPException(403, "Establishing a shared room baseline requires trusted authorization")
    dict_objects = [obj.model_dump() for obj in body.current_objects]
    result = campus_memory.detect_what_changed(
        location=body.location,
        current_objects=dict_objects,
        update_baseline=body.update_baseline if (is_mod or not baseline_exists) else False,
    )
    return WhatChangedResponse.model_validate(result)


def _bridge_call(action, *args, **kwargs):
    try:
        return action(*args, **kwargs)
    except observation_store.BridgeNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except observation_store.BridgeFullError as error:
        raise HTTPException(409, str(error)) from error
    except observation_store.BridgeAccessError as error:
        raise HTTPException(403, str(error)) from error


def _bridge_code(code: str) -> str:
    if len(code) != 6 or not code.isdigit():
        raise HTTPException(422, "A bridge code is six digits")
    return code


@app.post("/v1/bridge/conversations", response_model=BridgeConversation)
async def create_bridge_conversation(request: Request):
    """Start a Deaf-Blind bridge conversation and get a 6-digit code to share."""
    device_id, _session_id = _authorize_sight(request)
    return BridgeConversation.model_validate(observation_store.create_bridge(device_id))


@app.post("/v1/bridge/conversations/{code}/join", response_model=BridgeConversation)
async def join_bridge_conversation(request: Request, code: str):
    """Join a conversation as its second device (one pair per conversation)."""
    device_id, _session_id = _authorize_sight(request)
    _bridge_join_rate_limit(request, device_id)
    return BridgeConversation.model_validate(
        _bridge_call(observation_store.join_bridge, _bridge_code(code), device_id)
    )


@app.get("/v1/bridge/conversations/{code}", response_model=BridgeConversation)
async def get_bridge_conversation(request: Request, code: str):
    device_id, _session_id = _authorize_sight(request)
    return BridgeConversation.model_validate(
        _bridge_call(observation_store.bridge_status, _bridge_code(code), device_id)
    )


@app.post("/v1/bridge/message", response_model=BridgeMessage)
async def post_bridge_message(request: Request, body: BridgeMessageRequest):
    """Post a Deaf-Blind bridge message (sign translation or spoken reply)."""
    device_id, _session_id = _authorize_sight(request)
    msg = _bridge_call(
        observation_store.add_bridge_message,
        body.session_id, device_id, body.sender_role, body.text, body.client_id,
    )
    return BridgeMessage.model_validate(msg)


@app.get("/v1/bridge/messages", response_model=list[BridgeMessage])
async def get_bridge_messages(
    request: Request, session_id: str, after_id: int | None = None, limit: int = 50,
):
    """The newest messages in time order, or only those after ``after_id`` for live refresh."""
    device_id, _session_id = _authorize_sight(request)
    messages = _bridge_call(
        observation_store.list_bridge_messages, _bridge_code(session_id), device_id, after_id, limit,
    )
    return [BridgeMessage.model_validate(m) for m in messages]

