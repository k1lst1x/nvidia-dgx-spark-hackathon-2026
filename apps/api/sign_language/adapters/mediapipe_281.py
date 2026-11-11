"""Extract "visionmate-281-v1" feature frames from video files with MediaPipe.

The layout is defined in apps/mobile/src/signFeatures.ts and specified in
docs/sign/FEATURE_SCHEMA_281.md. `build_sign_feature_frame` reproduces it and
must keep matching the test vector in apps/mobile/tests/signFeatures.test.cjs
(see tests/test_sign_features_281.py).

Extraction follows the web tracker's primary path (HandTracker.web.tsx):
Holistic landmarker in VIDEO mode with the same model file and thresholds.
MediaPipe's Python API has no separate "presence" thresholds; the web
tracker's 0.4 presence thresholds are applied as landmark confidence.

Runs in its own environment with `mediapipe` installed, never in the API's.

Usage:
  python mediapipe_281.py --dataset-dir datasets/how2sign \
      --model apps/mobile/public/mediapipe/models/holistic_landmarker.task
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

SIGN_FEATURE_SCHEMA = "visionmate-281-v1"
SIGN_FEATURE_COUNT = 281
OFFSETS = {"left_hand": 0, "right_hand": 63, "pose": 126, "blendshapes": 225, "presence": 277}

BLENDSHAPE_ORDER = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
_BLENDSHAPE_INDEX = {name: index for index, name in enumerate(BLENDSHAPE_ORDER)}


def _coordinate(point, axis: str) -> float:
    value = point.get(axis) if isinstance(point, dict) else getattr(point, axis, None)
    return 0.0 if value is None else float(value)


def _write_points(frame: list[float], offset: int, points, count: int) -> None:
    if not points:
        return
    for index, point in enumerate(list(points)[:count]):
        if point is None:
            continue
        frame[offset + index * 3] = _coordinate(point, "x")
        frame[offset + index * 3 + 1] = _coordinate(point, "y")
        frame[offset + index * 3 + 2] = _coordinate(point, "z")


def build_sign_feature_frame(left_hand=None, right_hand=None, pose=None, face_present=False,
                             blendshapes=None) -> list[float]:
    """Same values, in the same positions, as buildSignFeatureFrame in signFeatures.ts.

    Points are objects or dicts with x, y and optional z. Blendshapes are
    (name, score) pairs or objects with category_name/categoryName and score.
    """
    frame = [0.0] * SIGN_FEATURE_COUNT
    _write_points(frame, OFFSETS["left_hand"], left_hand, 21)
    _write_points(frame, OFFSETS["right_hand"], right_hand, 21)
    _write_points(frame, OFFSETS["pose"], pose, 33)
    for shape in blendshapes or []:
        if isinstance(shape, tuple):
            name, score = shape
        else:
            name = getattr(shape, "category_name", None) or getattr(shape, "categoryName", None)
            score = shape.score
        index = _BLENDSHAPE_INDEX.get(name)
        if index is not None and score is not None and math.isfinite(float(score)):
            frame[OFFSETS["blendshapes"] + index] = float(score)
    presence = OFFSETS["presence"]
    frame[presence] = 1.0 if left_hand else 0.0
    frame[presence + 1] = 1.0 if right_hand else 0.0
    frame[presence + 2] = 1.0 if face_present else 0.0
    frame[presence + 3] = 1.0 if pose else 0.0
    return frame


def _first_person(landmarks):
    """Holistic returns one person; accept both flat and per-person nested lists."""
    if not landmarks:
        return None
    first = landmarks[0]
    return first if isinstance(first, (list, tuple)) else landmarks


def extract_video(video: Path, model: Path, fps: float = 20.0) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.4, min_face_landmarks_confidence=0.4,
        min_pose_detection_confidence=0.4, min_pose_landmarks_confidence=0.4,
        min_hand_landmarks_confidence=0.4,
    )
    capture = cv2.VideoCapture(str(video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames, timestamps = [], []
    counts = {"left_hand": 0, "right_hand": 0, "face": 0, "pose": 0}
    next_keep_ms = 0.0
    started = time.perf_counter()
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp_ms = index * 1000.0 / source_fps
            index += 1
            # Same rule as SignRecorder: keep a frame once 1000/fps ms have passed.
            if timestamp_ms + 1e-6 < next_keep_ms:
                continue
            next_keep_ms = timestamp_ms + 1000.0 / fps
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(round(timestamp_ms)))
            left = _first_person(result.left_hand_landmarks)
            right = _first_person(result.right_hand_landmarks)
            face = _first_person(result.face_landmarks)
            pose = _first_person(result.pose_landmarks)
            shapes = _first_person(result.face_blendshapes)
            for key, value in (("left_hand", left), ("right_hand", right), ("face", face), ("pose", pose)):
                counts[key] += 1 if value else 0
            frames.append(build_sign_feature_frame(left, right, pose, bool(face), shapes))
            timestamps.append(round(timestamp_ms))
    capture.release()
    elapsed = time.perf_counter() - started
    total = max(1, len(frames))
    return {
        "features": np.asarray(frames, dtype=np.float32).reshape(-1, SIGN_FEATURE_COUNT),
        "timestamps_ms": np.asarray(timestamps, dtype=np.int64),
        "stats": {
            "frames": len(frames), "source_fps": round(source_fps, 3), "sampled_fps": fps,
            "seconds": round(elapsed, 2), "ms_per_frame": round(1000 * elapsed / total, 1),
            **{f"{key}_present_ratio": round(value / total, 3) for key, value in counts.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="holistic_landmarker.task")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    import numpy as np

    samples_dir = args.dataset_dir / "samples"
    out_dir = samples_dir / "keypoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (samples_dir / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    report = []
    for row in manifest:
        extracted = extract_video(args.dataset_dir / row["video_path"], args.model, args.fps)
        path = out_dir / f"{row['sample_id']}_visionmate281.npz"
        np.savez_compressed(path, features=extracted["features"], timestamps_ms=extracted["timestamps_ms"],
                            feature_schema=SIGN_FEATURE_SCHEMA, sentence_name=row["sentence_name"],
                            text=row["english_text"])
        report.append({"sample_id": row["sample_id"], "path": str(path), **extracted["stats"]})
        print(json.dumps(report[-1]))

    evaluation_dir = args.dataset_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "mediapipe_281_extraction.json").write_text(
        json.dumps({"feature_schema": SIGN_FEATURE_SCHEMA, "model": args.model.name, "samples": report}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
