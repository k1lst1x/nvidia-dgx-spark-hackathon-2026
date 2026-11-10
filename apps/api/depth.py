"""Depth Anything V2 preprocessing and conservative approach helpers for Scene Assist.

The model is optional.  It uses the official Depth Anything V2 metric-depth
implementation vendored into the API image at build time; the checkpoint is
mounted/provisioned separately and is never downloaded by the application.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("guidelens.depth")


class Box(Protocol):
    label: str
    x: float | None
    y: float | None
    width: float | None
    height: float | None


@dataclass(frozen=True)
class DepthReading:
    label: str
    x: float
    y: float
    width: float
    height: float
    depth_m: float
    observed_at: float


@dataclass(frozen=True)
class SurfaceDepthReading(DepthReading):
    """Robust visible-surface depth with a confidence derived from map spread."""

    track_id: int | None
    confidence: float
    spread_m: float
    source: str = "box"


@dataclass(frozen=True)
class ApproachingObject:
    label: str
    distance_m: float


def _iou(left: DepthReading, right: DepthReading) -> float:
    x1, y1 = max(left.x, right.x), max(left.y, right.y)
    x2, y2 = min(left.x + left.width, right.x + right.width), min(left.y + left.height, right.y + right.height)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - overlap
    return overlap / union if union else 0.0


class ApproachingObjectTracker:
    """Only warn after a matching object gets materially nearer and larger.

    This rejects common monocular-depth noise and camera pans.  It is not a
    collision detector and deliberately makes no time-to-impact claim.
    """

    def __init__(self) -> None:
        self._history: dict[str, deque[DepthReading]] = defaultdict(lambda: deque(maxlen=48))
        self._last_warning: dict[tuple[str, str], float] = {}

    def update(self, session_id: str, readings: list[DepthReading]) -> list[ApproachingObject]:
        now = time.monotonic()
        history = self._history[session_id]
        while history and now - history[0].observed_at > 6:
            history.popleft()
        warnings: list[ApproachingObject] = []
        for current in readings:
            prior = next(
                (
                    reading
                    for reading in reversed(history)
                    if reading.label == current.label
                    and 0.5 <= current.observed_at - reading.observed_at <= 5
                    and _iou(reading, current) >= 0.2
                ),
                None,
            )
            if prior:
                depth_change = prior.depth_m - current.depth_m
                area_growth = (current.width * current.height) / max(prior.width * prior.height, 1e-6)
                key = (session_id, current.label)
                if (
                    depth_change >= 0.45
                    and current.depth_m <= prior.depth_m * 0.78
                    and area_growth >= 1.15
                    and now - self._last_warning.get(key, float("-inf")) >= 4
                ):
                    warnings.append(ApproachingObject(current.label, current.depth_m))
                    self._last_warning[key] = now
            history.append(current)
        return warnings


class DepthAnythingMetric:
    """Lazy Depth Anything V2 wrapper for metric or relative-depth checkpoints.

    ``metric`` checkpoints produce depth in meters. Pretrained ViT checkpoints
    such as ``depth_anything_v2_vitl.pth`` produce *relative* depth only: a
    larger value means visually nearer in that image, not a physical distance.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("DEPTH_ANYTHING_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.mode = os.getenv("DEPTH_ANYTHING_MODE", "metric").lower()
        if self.mode not in {"metric", "relative"}:
            raise ValueError("DEPTH_ANYTHING_MODE must be 'metric' or 'relative'")
        default_repo = "/opt/depth-anything-v2/metric_depth" if self.is_metric else "/opt/depth-anything-v2"
        default_checkpoint = (
            "/models/depth_anything_v2_metric_hypersim_vits.pth"
            if self.is_metric else "/models/depth_anything_v2_vitl.pth"
        )
        self.repo_path = Path(os.getenv("DEPTH_ANYTHING_REPO", default_repo))
        self.checkpoint = Path(os.getenv("DEPTH_ANYTHING_CHECKPOINT", default_checkpoint))
        self.device = os.getenv("DEPTH_ANYTHING_DEVICE", "cuda")
        self.max_depth = float(os.getenv("DEPTH_ANYTHING_MAX_DEPTH_M", "20"))
        # "auto": the model's native 518 on a GPU; 364 on CPU, where 518 takes
        # seconds per frame and the app drops replies older than 3 s.
        # ViT patches are 14 px, so sizes should be multiples of 14.
        self._input_size_setting = os.getenv("DEPTH_ANYTHING_INPUT_SIZE", "auto")
        self.input_size = 518
        self.last_ms: int | None = None
        self._device_used: str | None = None
        self._model = None
        self._load_error: str | None = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "ready": self.ready and self._load_error is None,
            "mode": self.mode,
            "metric": self.is_metric,
            "weights_present": self.checkpoint.is_file(),
            "code_present": self.repo_path.is_dir(),
            "load_failed": self._load_error is not None,
            "fallback": "box-interior",
            "device": self._device_used,
            "input_size": self.input_size if self._device_used else None,
            "last_ms": self.last_ms,
        }

    @property
    def is_metric(self) -> bool:
        return self.mode == "metric"

    @property
    def ready(self) -> bool:
        return self.enabled and self.repo_path.is_dir() and self.checkpoint.is_file()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.ready or self._load_error:
            return None
        try:
            import torch

            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
            from depth_anything_v2.dpt import DepthAnythingV2

            config = (
                {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": self.max_depth}
                if self.is_metric
                else {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
            )
            wants_gpu = self.device in {"auto", "cuda"} or self.device.startswith("cuda:")
            device = ("cuda" if self.device == "auto" else self.device) if wants_gpu and torch.cuda.is_available() else (
                "cpu" if wants_gpu else self.device
            )
            self._device_used = device
            if self._input_size_setting == "auto":
                self.input_size = 518 if device.startswith("cuda") else 364
            else:
                self.input_size = int(self._input_size_setting)
            model = DepthAnythingV2(**config)
            try:
                state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            self._model = model.to(device).eval()
            return self._model
        except Exception as exc:
            self._load_error = type(exc).__name__
            logger.exception("depth_model_unavailable")
            return None

    def infer(self, image_bytes: bytes):
        """Return an HxW metric-meter or unitless relative-depth map."""
        try:
            return self._infer_map(image_bytes)
        except Exception:
            logger.exception("depth_inference_failed")
            return None

    def _infer_map(self, image_bytes: bytes):
        model = self._load()
        if model is None:
            return None
        import cv2
        import numpy as np

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            return None
        # The upstream helper chooses a device independently of the configured
        # model device. Run the shared preprocessing path explicitly so CPU,
        # CUDA, and MPS deployments cannot create an image/model device mismatch.
        import torch
        import torch.nn.functional as functional

        started = time.perf_counter()
        tensor, (height, width) = model.image2tensor(image, self.input_size)
        device = next(model.parameters()).device
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(tensor.to(device))
        prediction = functional.interpolate(
            prediction.float()[:, None], (height, width), mode="bilinear", align_corners=True
        )[0, 0]
        result = prediction.detach().cpu().numpy()
        self.last_ms = int((time.perf_counter() - started) * 1000)
        return result


def readings_for_boxes(depth_map, objects: list[Box]) -> list[DepthReading]:
    """Use each box's central area to avoid background depth at object edges."""
    if depth_map is None:
        return []
    import numpy as np

    height, width = depth_map.shape[:2]
    readings: list[DepthReading] = []
    now = time.monotonic()
    for item in objects:
        if None in (item.x, item.y, item.width, item.height):
            continue
        x, y, box_width, box_height = item.x, item.y, item.width, item.height
        left = max(0, min(width - 1, int((x + box_width * 0.18) * width)))
        right = max(left + 1, min(width, int((x + box_width * 0.82) * width)))
        top = max(0, min(height - 1, int((y + box_height * 0.18) * height)))
        bottom = max(top + 1, min(height, int((y + box_height * 0.82) * height)))
        values = depth_map[top:bottom, left:right]
        distance_m = float(np.median(values[np.isfinite(values)])) if values.size else float("nan")
        if not np.isfinite(distance_m) or distance_m <= 0:
            continue
        readings.append(DepthReading(item.label, x, y, box_width, box_height, distance_m, now))
    return readings


def reject_depth_outliers(values):
    """Drop distant spikes without letting a few bad pixels set the distance."""
    import numpy as np

    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size < 12:
        return finite
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    limit = max(3 * 1.4826 * mad, center * 0.35, 0.2)
    kept = finite[np.abs(finite - center) <= limit]
    return kept if kept.size >= 9 else finite


def _box_bounds(depth_map, item: Box, margin: float):
    height, width = depth_map.shape[:2]
    if None in (item.x, item.y, item.width, item.height):
        return None
    x, y, box_width, box_height = item.x, item.y, item.width, item.height
    left = max(0, min(width - 1, int((x + box_width * margin) * width)))
    right = max(left + 1, min(width, int((x + box_width * (1 - margin)) * width)))
    top = max(0, min(height - 1, int((y + box_height * margin) * height)))
    bottom = max(top + 1, min(height, int((y + box_height * (1 - margin)) * height)))
    return top, bottom, left, right


def _finite_values(depth_map, item: Box, mask):
    """Sample the visible surface, then the box interior when the mask is unusable."""
    import numpy as np

    height, width = depth_map.shape[:2]
    if mask is not None:
        region = np.asarray(mask)
        if region.shape[:2] != (height, width):
            import cv2

            region = cv2.resize(region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        else:
            region = region.astype(bool)
        bounds = _box_bounds(depth_map, item, 0.0)
        if bounds is not None:
            top, bottom, left, right = bounds
            window = np.zeros(region.shape, dtype=bool)
            window[top:bottom, left:right] = True
            overlap = region & window
            box_pixels = max(1, int(window.sum()))
            if int(overlap.sum()) >= 9 and int(overlap.sum()) / box_pixels <= 0.92:
                return reject_depth_outliers(depth_map[overlap]), "mask"
    bounds = _box_bounds(depth_map, item, 0.25)
    if bounds is None:
        return np.empty(0, dtype=np.float32), "box"
    top, bottom, left, right = bounds
    return reject_depth_outliers(depth_map[top:bottom, left:right]), "box"


def surface_readings_for_boxes(depth_map, objects: list[Box], masks: list | None = None) -> list[SurfaceDepthReading]:
    """Measure the object interior rather than a broad box/background mixture.

    The central 50% is deliberately conservative around detector boundaries.
    Median/MAD provide a measurement-quality signal; they do not calibrate an
    otherwise relative depth checkpoint.
    """
    if depth_map is None:
        return []
    import numpy as np

    readings: list[SurfaceDepthReading] = []
    now = time.monotonic()
    for index, item in enumerate(objects):
        if None in (item.x, item.y, item.width, item.height):
            continue
        x, y, box_width, box_height = item.x, item.y, item.width, item.height
        mask = masks[index] if masks and index < len(masks) else None
        values, source = _finite_values(depth_map, item, mask)
        if values.size < 9:
            continue
        distance_m = float(np.median(values))
        if not np.isfinite(distance_m) or distance_m <= 0:
            continue
        mad = float(np.median(np.abs(values - distance_m)))
        # The scale is intentionally strict: a depth discontinuity inside the
        # box reduces confidence instead of becoming an apparently precise range.
        spread = 1.4826 * mad
        expected = int(np.asarray(mask).astype(bool).sum()) if mask is not None and source == "mask" else max(9, values.size)
        valid_fraction = min(1.0, values.size / max(9, expected))
        confidence = float(np.clip(valid_fraction * (1.0 - spread / max(distance_m * 0.35, 0.25)), 0, 1))
        readings.append(SurfaceDepthReading(
            item.label, x, y, box_width, box_height, distance_m, now,
            getattr(item, "track_id", None), confidence, spread, source,
        ))
    return readings


def relative_proximity_for_boxes(depth_map, objects: list[Box], masks: list | None = None) -> dict[tuple[str, float, float], float]:
    """Map each detected box to a normalized *within-frame* near-depth score.

    The output is in [0, 1] and is intentionally not converted to meters. It
    only tells the UI which detected objects are comparatively closer in the
    current camera frame.
    """
    if depth_map is None:
        return {}
    import numpy as np

    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return {}
    low, high = np.percentile(finite, [5, 95])
    scale = max(float(high - low), 1e-6)
    scores: dict[tuple[str, float, float], float] = {}
    for index, item in enumerate(objects):
        if None in (item.x, item.y, item.width, item.height):
            continue
        x, y, box_width, box_height = item.x, item.y, item.width, item.height
        mask = masks[index] if masks and index < len(masks) else None
        values, _source = _finite_values(depth_map, item, mask)
        if values.size == 0:
            continue
        score = float(np.clip((float(np.median(values)) - low) / scale, 0, 1))
        scores[(item.label, round(x, 3), round(y, 3))] = score
    return scores
