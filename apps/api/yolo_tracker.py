"""Session-scoped YOLO11/BoT-SORT detection service."""

import logging
import threading
import time
from functools import lru_cache
from io import BytesIO
from typing import Callable

import device_manager
from schemas import DetectedObject

# BoT-SORT keeps a track alive through frames where the detector is unsure
# (track_low_thresh in botsort.yaml). Those weak detections must reach the
# tracker, so the model runs at this floor and weak boxes are reported only
# for tracks that were already confirmed over several frames.
TRACKER_INPUT_CONFIDENCE = 0.12
CONFIRMED_TRACK_HITS = 3


def resolve_device(device: str) -> str:
    """'auto' picks CUDA, then MPS, then CPU, as far as this PyTorch build can use them."""
    return device_manager.select_device("yolo", device).device


@lru_cache(maxsize=1)
def _has_quantize() -> bool:
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError:
        return False
    return "quantize" in DEFAULT_CFG_DICT


def precision(half: bool) -> dict[str, object]:
    """FP16/FP32 argument for this Ultralytics version (8.4 replaced `half` with `quantize`)."""
    if _has_quantize():
        return {"quantize": 16 if half else None}
    return {"half": half}


def warm_up_yolo(model, device: str, half: bool = False) -> None:
    """One inference on a blank frame: proves the device runs this model."""
    import numpy as np

    model.predict(source=np.full((480, 640, 3), 127, dtype=np.uint8), imgsz=640, device=device, verbose=False,
                  **precision(half))


class YoloTracker:
    """One shared YOLO model; BoT-SORT state belongs to one camera session at a time.

    The device is fixed when the model loads. If the accelerator fails later,
    the model is rebuilt on CPU and the session starts over with fresh track
    IDs; a session is never continued across devices.
    """

    def __init__(self, *, enabled: bool, model_path: str, device: str | None = None, confidence: float,
                 tracker_config: str, idle_seconds: float, cpu_model_path: str | None = None,
                 on_reset: Callable[[str], None] | None = None):
        self.enabled = enabled
        self._gpu_model_path = model_path
        self._cpu_model_path = cpu_model_path or model_path
        self._choice = device_manager.select_device("yolo", device)
        self.device = self._choice.device
        self.half = device_manager.half_enabled("yolo", self.device)
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.idle_seconds = idle_seconds
        self.last_ms: int | None = None
        self._lock = threading.Lock()
        self._session: str | None = None
        self._seen_at = 0.0
        self._hits: dict[int, int] = {}
        self._loaded = None
        self._on_reset = on_reset
        self.restarts = 0
        self._logger = logging.getLogger("guidelens")

    @property
    def model_path(self) -> str:
        """A larger model is cheap on a GPU and much slower on CPU, so CPU can use its own."""
        return self._cpu_model_path if self.device == "cpu" else self._gpu_model_path

    @property
    def segments(self) -> bool:
        """True when the tracking model also returns instance masks."""
        return self.model_path.removesuffix(".pt").endswith("-seg")

    def status(self) -> dict[str, object]:
        return {"device": self.device, "half": self.half, "segments": self.segments, "last_ms": self.last_ms,
                "restarts": self.restarts}

    def _load(self, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; rebuild the API image with requirements.txt") from exc
        path = self._cpu_model_path if device == "cpu" else self._gpu_model_path
        return YOLO(path)

    def _model(self):
        """Load once; the caller holds the lock."""
        if self._loaded is None:
            def warm(model, device):
                warm_up_yolo(model, device, device_manager.half_enabled("yolo", device))

            self._loaded, self.device = device_manager.load_with_fallback(
                self._choice, self._load, warm, model=self._gpu_model_path,
            )
            self.half = device_manager.half_enabled("yolo", self.device)
        return self._loaded

    def load(self) -> None:
        """Load and verify the model now instead of on the first camera frame."""
        if self.enabled:
            with self._lock:
                self._model()

    def _forget_session(self, session_id: str | None) -> None:
        self._hits.clear()
        if session_id and self._on_reset:
            self._on_reset(session_id)

    def _restart_on_cpu(self, error: BaseException) -> None:
        """The accelerator failed mid-session: new CPU model, new tracker, new session."""
        failed = self.device
        self._logger.error("yolo_accelerator_failed device=%s; restarting on cpu: %s", failed, error)
        self._loaded = None
        device_manager.release(failed)
        self._forget_session(self._session)
        self._session = None
        self._loaded = self._load("cpu")
        warm_up_yolo(self._loaded, "cpu")
        self.device, self.half = "cpu", False
        self.restarts += 1
        device_manager.record("yolo", self._choice, "cpu", fallback_from=failed,
                              error=f"{type(error).__name__}: {error}"[:300], model=self._cpu_model_path)

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

    def _track(self, model, image):
        return model.track(
            source=image, persist=True, tracker=self.tracker_config, conf=TRACKER_INPUT_CONFIDENCE,
            imgsz=640, device=self.device, verbose=False, max_det=20, retina_masks=True, **precision(self.half),
        )[0]

    def detect(self, image_bytes: bytes, session_id: str) -> list[DetectedObject]:
        return self.detect_with_masks(image_bytes, session_id)[0]

    def detect_with_masks(self, image_bytes: bytes, session_id: str) -> tuple[list[DetectedObject], list | None]:
        """Tracked objects, plus one full-frame mask per object when the model segments.

        Masks come from the same forward pass as the boxes, so they always
        belong to the object they are paired with.
        """
        if not self.enabled:
            return [], None
        started = time.perf_counter()
        try:
            from PIL import Image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            with self._lock:
                model = self._model()
                now = time.monotonic()
                if self._session and (self._session != session_id or now - self._seen_at > self.idle_seconds):
                    self._reset(model)
                    self._forget_session(self._session)
                self._session, self._seen_at = session_id, now
                try:
                    result = self._track(model, image)
                except Exception as error:
                    if self.device == "cpu" or not device_manager.is_accelerator_error(error):
                        raise
                    self._restart_on_cpu(error)
                    self._session, self._seen_at = session_id, time.monotonic()
                    result = self._track(self._loaded, image)
                tracked = getattr(result.boxes, "is_track", False)
                live_ids = set()
                if tracked and result.boxes.id is not None:
                    for value in result.boxes.id.tolist():
                        track_id = int(value)
                        live_ids.add(track_id)
                        self._hits[track_id] = self._hits.get(track_id, 0) + 1
                for stale in [key for key in self._hits if key not in live_ids and len(self._hits) > 200]:
                    self._hits.pop(stale, None)
                hits = dict(self._hits)
            mask_data = None
            if self.segments and result.masks is not None:
                mask_data = result.masks.data.cpu().numpy() > 0.5
            image_width, image_height = image.size
            objects: list[DetectedObject] = []
            masks: list = []
            for index, box in enumerate(result.boxes):
                left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                x, y = max(0., min(1., left / image_width)), max(0., min(1., top / image_height))
                width, height = max(0., min(1. - x, (right - left) / image_width)), max(0., min(1. - y, (bottom - top) / image_height))
                if width < .02 or height < .02:
                    continue
                confidence = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if tracked and box.id is not None else None
                # A weak box is kept only as the continuation of a confirmed track.
                if confidence < self.confidence and (track_id is None or hits.get(track_id, 0) < CONFIRMED_TRACK_HITS):
                    continue
                objects.append(DetectedObject(label=str(result.names[int(box.cls[0].item())]), confidence=confidence,
                                              position=self._position(x, y, width, height), x=x, y=y, width=width, height=height,
                                              track_id=track_id))
                mask = None
                if mask_data is not None and index < len(mask_data) and mask_data[index].shape == (image_height, image_width):
                    mask = mask_data[index]
                masks.append(mask)
            return objects, (masks if self.segments else None)
        except Exception:
            self._logger.exception("yolo_detection_failed model=%s", self.model_path)
            return [], None
        finally:
            self.last_ms = int((time.perf_counter() - started) * 1000)
