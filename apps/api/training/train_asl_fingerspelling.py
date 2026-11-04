#!/usr/bin/env python3
"""Train a static ASL fingerspelling classifier from the supplied zip archive.

The dataset contains isolated images for 24 static letters (A-I and K-Y).  J and
Z are motion signs and are deliberately excluded.  This script streams images
from Dataset_Original inside the archive, extracts MediaPipe hand landmarks, and
trains a softmax classifier on normalized landmark vectors.  The emitted JSON
can be consumed by a browser or backend without a Python runtime.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
IMAGE_PATH = re.compile(r"^Dataset_Original/(Train|Test)/([A-Z])/.*\.(?:jpg|jpeg|png)$", re.IGNORECASE)
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the inner dataset zip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/api/models/asl_fingerspelling_landmarks.json"),
        help="Path for the portable softmax model JSON.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Limit each split/class for a quick smoke run; 0 uses every available image.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path("apps/api/models/hand_landmarker.task"),
        help="MediaPipe Hand Landmarker task file. It is downloaded if absent.",
    )
    return parser.parse_args()


def normalized_landmarks(hand_landmarks, handedness: str) -> np.ndarray:
    points = np.array([[point.x, point.y, point.z] for point in hand_landmarks], dtype=np.float32)
    if handedness.lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]
    wrist = points[0].copy()
    scale = np.linalg.norm(points[9, :2] - wrist[:2])
    if scale < 1e-5:
        raise ValueError("hand scale is too small")
    return ((points - wrist) / scale).reshape(-1)


def image_records(archive: zipfile.ZipFile, max_per_class: int, seed: int):
    records: dict[tuple[str, str], list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        match = IMAGE_PATH.match(info.filename)
        if not match:
            continue
        split, label = match.groups()
        label = label.upper()
        if label in LABELS:
            records.setdefault((split.lower(), label), []).append(info)

    rng = random.Random(seed)
    selected: list[tuple[str, str, zipfile.ZipInfo]] = []
    for (split, label), files in sorted(records.items()):
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        selected.extend((split, label, info) for info in files)
    rng.shuffle(selected)
    return selected


def ensure_hand_model(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand model to {path}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    return path


def extract_features(dataset: Path, max_per_class: int, seed: int, hand_model: Path):
    features: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    targets: dict[str, list[str]] = {"train": [], "test": []}
    skipped: Counter[str] = Counter()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    with zipfile.ZipFile(dataset) as archive, vision.HandLandmarker.create_from_options(options) as hands:
        records = image_records(archive, max_per_class, seed)
        print(f"Processing {len(records):,} images from {dataset.name}")
        for index, (split, label, info) in enumerate(records, start=1):
            try:
                with Image.open(io.BytesIO(archive.read(info))) as image:
                    rgb = np.asarray(image.convert("RGB"))
                result = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.hand_landmarks or not result.handedness:
                    skipped["no_hand"] += 1
                    continue
                handedness = result.handedness[0][0].category_name
                features[split].append(normalized_landmarks(result.hand_landmarks[0], handedness))
                targets[split].append(label)
            except Exception as error:  # Keep long dataset runs resilient to individual corrupt files.
                skipped[type(error).__name__] += 1
            if index % 1000 == 0 or index == len(records):
                print(f"  {index:,}/{len(records):,} images processed; usable={sum(map(len, features.values())):,}")

    print(f"Skipped images: {dict(skipped)}")
    return (
        np.asarray(features["train"], dtype=np.float32),
        np.asarray(targets["train"]),
        np.asarray(features["test"], dtype=np.float32),
        np.asarray(targets["test"]),
    )


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset archive not found: {args.dataset}")

    hand_model = ensure_hand_model(args.hand_model)
    x_train, y_train, x_test, y_test = extract_features(args.dataset, args.max_per_class, args.seed, hand_model)
    if not len(x_train) or not len(x_test):
        raise SystemExit("No usable training/test landmarks were extracted.")

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    classifier = LogisticRegression(
        C=8.0,
        max_iter=600,
        solver="lbfgs",
        random_state=args.seed,
    ).fit(x_train_scaled, y_train)

    predictions = classifier.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4%}")
    print(classification_report(y_test, predictions, labels=list(classifier.classes_), zero_division=0))

    model = {
        "schema_version": 1,
        "task": "static_asl_fingerspelling",
        "labels": classifier.classes_.tolist(),
        "unsupported_motion_letters": ["J", "Z"],
        "feature_spec": {
            "landmarks_per_hand": 21,
            "coordinates_per_landmark": 3,
            "normalization": "wrist-centered; scaled by wrist-to-middle-mcp distance; left hands mirrored",
        },
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": {"coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist()},
        "metrics": {
            "test_accuracy": float(accuracy),
            "train_examples": int(len(y_train)),
            "test_examples": int(len(y_test)),
            "class_counts": dict(sorted(Counter(y_train).items())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote model: {args.output}")


if __name__ == "__main__":
    main()
