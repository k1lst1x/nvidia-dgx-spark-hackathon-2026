"""Optional YOLO11 instance segmentation beside the existing BoT-SORT detector.

The box tracker remains the source of object identity. Segmentation only adds a
visible-surface mask, matched by class and overlap. Any failure returns no masks
so detection continues.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import lru_cache
from io import BytesIO

logger = logging.getLogger("guidelens.segmentation")


def _iou(box, other) -> float:
    if None in (box.x, box.y, box.width, box.height, other[0], other[1], other[2], other[3]):
        return 0.0
    left = max(box.x, other[0])
    top = max(box.y, other[1])
    right = min(box.x + box.width, other[0] + other[2])
    bottom = min(box.y + box.height, other[1] + other[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = box.width * box.height + other[2] * other[3] - overlap
    return overlap / union if union else 0.0


class YoloSegmenter:
    def __init__(self) -> None:
        self.enabled = os.getenv("ULTRALYTICS_YOLO_SEG_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.model_path = os.getenv("ULTRALYTICS_YOLO_SEG_MODEL", "yolo11n-seg.pt")
        from yolo_tracker import resolve_device

        self.device = resolve_device(os.getenv("ULTRALYTICS_YOLO_DEVICE", "cpu"))
        self.confidence = float(os.getenv("ULTRALYTICS_YOLO_CONFIDENCE", "0.40"))
        self.last_ms: int | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "model": self.model_path if self.enabled else None,
            "last_ms": self.last_ms,
        }

    @lru_cache(maxsize=1)
    def _model(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def masks_for(self, image_bytes: bytes, objects: list) -> list:
        """Return one full-frame boolean mask per object, or None when unmatched."""
        empty = [None] * len(objects)
        if not self.enabled or not objects:
            return empty
        started = time.perf_counter()
        try:
            import numpy as np
            from PIL import Image

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            with self._lock:
                result = self._model().predict(
                    source=image, conf=self.confidence, imgsz=640, device=self.device, verbose=False, max_det=12,
                    half=self.device.startswith("cuda"),
                    # Masks at the original frame size. Resizing the letterboxed
                    # inference mask would include the padding and shift it.
                    retina_masks=True,
                )[0]
            if result.masks is None or result.boxes is None:
                return empty
            candidates = []
            mask_data = result.masks.data.cpu().numpy()
            for index, box in enumerate(result.boxes):
                left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                label = str(result.names[int(box.cls[0].item())])
                normalized = (
                    max(0.0, left / width), max(0.0, top / height),
                    max(0.0, (right - left) / width), max(0.0, (bottom - top) / height),
                )
                mask = mask_data[index]
                if mask.shape != (height, width):
                    mask_image = Image.fromarray((mask * 255).astype("uint8")).resize((width, height))
                    mask = np.asarray(mask_image) > 127
                else:
                    mask = mask > 0.5
                candidates.append((label, normalized, mask))
            matched: list = []
            used: set[int] = set()
            for item in objects:
                best_score = 0.15
                best_index = -1
                for index, (label, box, _mask) in enumerate(candidates):
                    if index in used or label != item.label:
                        continue
                    score = _iou(item, box)
                    if score > best_score:
                        best_score = score
                        best_index = index
                if best_index < 0:
                    matched.append(None)
                    continue
                used.add(best_index)
                matched.append(candidates[best_index][2])
            return matched
        except Exception:
            logger.exception("segmentation_failed model=%s", self.model_path)
            return empty
        finally:
            self.last_ms = int((time.perf_counter() - started) * 1000)
