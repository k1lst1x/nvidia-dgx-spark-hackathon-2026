"""Small image-quality checks for framing guidance; no images are persisted."""

from __future__ import annotations

import cv2
import numpy as np

from schemas import CameraCoach, DetectedObject


def assess_frame(image_bytes: bytes, objects: list[DetectedObject]) -> CameraCoach:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError("Camera frame could not be decoded")
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    brightness = float(np.mean(image))
    cropped = []
    for item in objects:
        if item.x is None or item.y is None or item.width is None or item.height is None:
            continue
        if item.x <= 0.015 or item.y <= 0.015 or item.x + item.width >= 0.985 or item.y + item.height >= 0.985:
            cropped.append(item.label)
    return CameraCoach(
        blur=sharpness < 55,
        poor_lighting=brightness < 42,
        overexposed=brightness > 232,
        cropped_labels=list(dict.fromkeys(cropped))[:8],
        sharpness_score=round(sharpness, 1),
        brightness_score=round(brightness, 1),
    )
