"""Session-scoped YOLO11 sign language detection service."""

import logging
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import device_manager
from schemas import DetectedObject

SIGN_CLASSES = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
    9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'ExcuseMe', 15: 'F',
    16: 'Food', 17: 'G', 18: 'H', 19: 'Hello', 20: 'Help', 21: 'House',
    22: 'I', 23: 'I Love You', 24: 'Internet', 25: 'J', 26: 'K', 27: 'L',
    28: 'M', 29: 'N', 30: 'No', 31: 'O', 32: 'P', 33: 'Please', 34: 'Q',
    35: 'R', 36: 'S', 37: 'T', 38: 'ThankYou', 39: 'U', 40: 'V', 41: 'W',
    42: 'X', 43: 'Y', 44: 'Yes', 45: 'Z',
}


class YoloSignDetector:
    def __init__(self, *, model_path: str | None = None, confidence: float = 0.55, device: str | None = None):
        self.confidence = confidence
        self.device = device_manager.select_device("sign", device).device
        self._lock = threading.Lock()
        self._logger = logging.getLogger("guidelens.yolo_sign")

        if model_path:
            self.model_path = model_path
        else:
            default_pt = Path(__file__).parent / "models" / "yolo11_sign_language.pt"
            default_onnx = Path(__file__).parent / "models" / "yolo11_sign_language.onnx"
            if default_pt.exists():
                self.model_path = str(default_pt)
            elif default_onnx.exists():
                self.model_path = str(default_onnx)
            else:
                self.model_path = ""

    @property
    def available(self) -> bool:
        return self._model() is not None

    @lru_cache(maxsize=1)
    def _model(self):
        if not self.model_path or not Path(self.model_path).exists():
            return None
        try:
            from ultralytics import YOLO
            return YOLO(self.model_path)
        except Exception as exc:
            self._logger.warning("Could not load YOLO sign model: %s", exc)
            return None

    def detect(self, image_bytes: bytes) -> list[DetectedObject]:
        model = self._model()
        if not model:
            return []
        try:
            from PIL import Image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            with self._lock:
                results = model.predict(source=image, conf=self.confidence, device=self.device, verbose=False, imgsz=640)
            if not results:
                return []
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []

            image_width, image_height = image.size
            detections: list[DetectedObject] = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                norm_x = max(0.0, min(1.0, x1 / image_width))
                norm_y = max(0.0, min(1.0, y1 / image_height))
                norm_w = max(0.0, min(1.0, (x2 - x1) / image_width))
                norm_h = max(0.0, min(1.0, (y2 - y1) / image_height))

                label = SIGN_CLASSES.get(cls_id, str(cls_id))
                detections.append(
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        position="center",
                        x=round(norm_x, 3),
                        y=round(norm_y, 3),
                        width=round(norm_w, 3),
                        height=round(norm_h, 3),
                    )
                )
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections
        except Exception as exc:
            self._logger.warning("Error running YOLO sign detection: %s", exc)
            return []
