"""Depth Anything V2 preprocessing and conservative approach helpers for Scene Assist.

The model is optional.  It uses the official Depth Anything V2 metric-depth
implementation vendored into the API image at build time; the checkpoint is
mounted/provisioned separately and is never downloaded by the application.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
        self.input_size = int(os.getenv("DEPTH_ANYTHING_INPUT_SIZE", "518"))
        self._model = None

    @property
    def is_metric(self) -> bool:
        return self.mode == "metric"

    @property
    def ready(self) -> bool:
        return self.enabled and self.repo_path.is_dir() and self.checkpoint.is_file()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.ready:
            return None
        import torch

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        from depth_anything_v2.dpt import DepthAnythingV2

        config = (
            {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": self.max_depth}
            if self.is_metric
            else {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
        )
        device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
        model = DepthAnythingV2(**config)
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
        self._model = model.to(device).eval()
        return self._model

    def infer(self, image_bytes: bytes):
        """Return an HxW metric-meter or unitless relative-depth map."""
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

        tensor, (height, width) = model.image2tensor(image, self.input_size)
        device = next(model.parameters()).device
        with torch.no_grad():
            prediction = model(tensor.to(device))
        prediction = functional.interpolate(
            prediction[:, None], (height, width), mode="bilinear", align_corners=True
        )[0, 0]
        return prediction.detach().cpu().numpy()


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


def relative_proximity_for_boxes(depth_map, objects: list[Box]) -> dict[tuple[str, float, float], float]:
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
    height, width = depth_map.shape[:2]
    scores: dict[tuple[str, float, float], float] = {}
    for item in objects:
        if None in (item.x, item.y, item.width, item.height):
            continue
        x, y, box_width, box_height = item.x, item.y, item.width, item.height
        left = max(0, min(width - 1, int((x + box_width * 0.18) * width)))
        right = max(left + 1, min(width, int((x + box_width * 0.82) * width)))
        top = max(0, min(height - 1, int((y + box_height * 0.18) * height)))
        bottom = max(top + 1, min(height, int((y + box_height * 0.82) * height)))
        values = depth_map[top:bottom, left:right]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        score = float(np.clip((float(np.median(values)) - low) / scale, 0, 1))
        scores[(item.label, round(x, 3), round(y, 3))] = score
    return scores
