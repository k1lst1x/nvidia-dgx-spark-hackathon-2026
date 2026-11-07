"""Session-scoped YOLO11/BoT-SORT detection service."""

import logging
import threading
import time
from functools import lru_cache
from io import BytesIO

from schemas import DetectedObject


class YoloTracker:
    def __init__(self, *, enabled: bool, model_path: str, device: str, confidence: float, tracker_config: str, idle_seconds: float):
        self.enabled = enabled
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._session: str | None = None
        self._seen_at = 0.0
        self._logger = logging.getLogger("guidelens")

    @lru_cache(maxsize=1)
    def _model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; rebuild the API image with requirements.txt") from exc
        return YOLO(self.model_path)

    @staticmethod
    def _position(x: float, y: float, width: float, height: float) -> str:
        horizontal = "left" if x + width / 2 < .33 else "right" if x + width / 2 > .67 else "center"
        vertical = "upper" if y + height / 2 < .33 else "lower" if y + height / 2 > .67 else "middle"
        return horizontal if vertical == "middle" else f"{vertical} {horizontal}"

    @staticmethod
    def _reset(model) -> None:
        for tracker in getattr(getattr(model, "predictor", None), "trackers", ()):
            reset = getattr(tracker, "reset", None)
            if reset:
                reset()

    def detect(self, image_bytes: bytes, session_id: str) -> list[DetectedObject]:
        if not self.enabled:
            return []
        try:
            from PIL import Image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            with self._lock:
                model = self._model()
                now = time.monotonic()
                if self._session and (self._session != session_id or now - self._seen_at > self.idle_seconds):
                    self._reset(model)
                self._session, self._seen_at = session_id, now
                result = model.track(source=image, persist=True, tracker=self.tracker_config, conf=self.confidence,
                                     imgsz=640, device=self.device, verbose=False, max_det=12)[0]
            image_width, image_height = image.size
            objects: list[DetectedObject] = []
            for box in result.boxes:
                left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                x, y = max(0., min(1., left / image_width)), max(0., min(1., top / image_height))
                width, height = max(0., min(1. - x, (right - left) / image_width)), max(0., min(1. - y, (bottom - top) / image_height))
                if width < .02 or height < .02:
                    continue
                track_id = int(box.id[0].item()) if getattr(result.boxes, "is_track", False) and box.id is not None else None
                objects.append(DetectedObject(label=str(result.names[int(box.cls[0].item())]), confidence=float(box.conf[0].item()),
                                              position=self._position(x, y, width, height), x=x, y=y, width=width, height=height,
                                              track_id=track_id))
            return objects
        except Exception:
            self._logger.exception("yolo_detection_failed model=%s", self.model_path)
            return []
