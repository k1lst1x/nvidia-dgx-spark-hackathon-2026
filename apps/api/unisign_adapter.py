"""Explicit MediaPipe-to-Uni-Sign pose compatibility adapter.

Uni-Sign's released How2Sign pose-only checkpoint was trained with RTM
WholeBody's 133 2-D points plus per-point confidence.  It is not compatible
with VisionMate's 281-value vector, and this module intentionally never tries
to reverse that vector into a skeleton.

The adapter maps *raw* MediaPipe Holistic landmarks to the four streams that
the official implementation consumes:

* ``body``: 9 points (nose, ears, shoulders, elbows, wrists)
* ``left`` / ``right``: 21 wrist-centred hand points
* ``face_all``: 18 approximate face points (jaw, mouth, nose)

Coordinate correspondence for the face is only a semantic approximation:
MediaPipe Face Mesh and RTM WholeBody have different landmark definitions.
Likewise, MediaPipe Tasks does not expose RTM-style keypoint confidence.  The
default policy therefore rejects raw MediaPipe data for official-checkpoint
inference. ``presence_proxy`` is available only for adapter experiments and
is annotated in metadata so it cannot be mistaken for a verified evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

ConfidencePolicy = Literal["reject", "presence_proxy"]
UNISIGN_PART_SIZES = {"body": 9, "left": 21, "right": 21, "face_all": 18}
UNISIGN_SCORE_THRESHOLD = 0.3

# Uni-Sign selects RTM WholeBody [0, 3..10] for the body stream.  This is the
# semantic MediaPipe Pose equivalent: nose, ears, shoulders, elbows, wrists.
MEDIAPIPE_BODY_INDICES = (0, 7, 8, 11, 12, 13, 14, 15, 16)

# Uni-Sign's face stream selects RTM jaw, mouth, and nose positions. These
# FaceMesh indices are semantic approximations—not an asserted topology match.
MEDIAPIPE_FACE_JAW = (234, 132, 58, 172, 150, 152, 377, 397, 454)
MEDIAPIPE_FACE_MOUTH = (61, 40, 37, 0, 267, 270, 291, 17)
MEDIAPIPE_FACE_NOSE = 1


class UniSignCompatibilityError(ValueError):
    """Raised before incompatible data can reach an official checkpoint."""


@dataclass(frozen=True)
class UniSignPoseSequence:
    body: np.ndarray
    left: np.ndarray
    right: np.ndarray
    face_all: np.ndarray
    timestamps_ms: np.ndarray
    confidence_policy: str
    topology_verified: bool
    normalized: bool

    def as_model_input(self) -> dict[str, np.ndarray]:
        """Return official Uni-Sign tensor layout: ``(batch, time, joints, 3)``."""
        return {
            key: value[None, ...]
            for key, value in {
                "body": self.body,
                "left": self.left,
                "right": self.right,
                "face_all": self.face_all,
            }.items()
        }


def _point_array(points: Sequence[dict[str, Any]] | None, expected: int, name: str) -> np.ndarray:
    if points is None:
        return np.zeros((expected, 2), dtype=np.float32)
    if len(points) < expected:
        raise UniSignCompatibilityError(f"{name} needs {expected} landmarks; got {len(points)}")
    try:
        return np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise UniSignCompatibilityError(f"{name} contains invalid landmark coordinates") from error


def _scores(frame: dict[str, Any], count: int, policy: ConfidencePolicy, part: str, present: bool) -> np.ndarray:
    supplied = frame.get(f"{part}_confidence")
    if supplied is not None:
        values = np.asarray(supplied, dtype=np.float32)
        if values.shape != (count,):
            raise UniSignCompatibilityError(f"{part}_confidence must contain {count} values")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise UniSignCompatibilityError(f"{part}_confidence must be finite scores in [0, 1]")
        return values
    if policy == "reject":
        raise UniSignCompatibilityError(
            "Official Uni-Sign inference requires RTM WholeBody per-landmark confidence scores. "
            "MediaPipe Tasks does not provide them; use the RTM extractor or explicitly opt into "
            "the unevaluated presence_proxy experiment."
        )
    # This is not detector confidence. It denotes only that a MediaPipe point
    # was present, and metadata preserves the approximation for evaluation.
    return np.ones((count,), dtype=np.float32) if present else np.zeros((count,), dtype=np.float32)


def _crop_scale(body: np.ndarray, body_score: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    # Match Uni-Sign datasets.load_part_kp: crop scale is computed over every
    # valid body point in the full clip, rather than independently per frame.
    valid = body[body_score > UNISIGN_SCORE_THRESHOLD]
    if len(valid) < 4:
        return np.zeros_like(body), 0.0, np.zeros(2, dtype=np.float32)
    minimum = valid.min(axis=0)
    maximum = valid.max(axis=0)
    scale = float(np.max(maximum - minimum))
    if scale <= 0:
        return np.zeros_like(body), 0.0, np.zeros(2, dtype=np.float32)
    origin = (minimum + maximum - scale) / 2
    normalized = np.clip(((body - origin) / scale - 0.5) * 2, -1, 1)
    normalized[body_score <= UNISIGN_SCORE_THRESHOLD] = 0
    return normalized, scale, origin


def _normalize_part(points: np.ndarray, scores: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(points)
    output = np.clip(points / scale, -1, 1)
    # Hands and face are wrist/nose centred before this function. Uni-Sign
    # applies the global body crop scale, not a second per-part crop.
    output[scores <= UNISIGN_SCORE_THRESHOLD] = 0
    return output


class UniSignAdapter:
    """Map exported raw Holistic frames into the public Uni-Sign pose schema."""

    def __init__(self, confidence_policy: ConfidencePolicy = "reject") -> None:
        self.confidence_policy = confidence_policy

    def adapt(self, frames: Sequence[dict[str, Any]]) -> UniSignPoseSequence:
        if not frames:
            raise UniSignCompatibilityError("At least one raw Holistic frame is required")
        body_frames: list[np.ndarray] = []
        left_frames: list[np.ndarray] = []
        right_frames: list[np.ndarray] = []
        face_frames: list[np.ndarray] = []
        body_scores: list[np.ndarray] = []
        left_scores: list[np.ndarray] = []
        right_scores: list[np.ndarray] = []
        face_scores: list[np.ndarray] = []
        timestamps: list[float] = []

        for frame in frames:
            if frame.get("confidence_representation") == "not_available_from_mediapipe_tasks" and self.confidence_policy == "reject":
                raise UniSignCompatibilityError(
                    "This recording explicitly states that MediaPipe confidence is unavailable. "
                    "It cannot be passed to the official Uni-Sign checkpoint unchanged."
                )
            pose_source = frame.get("pose")
            left_source = frame.get("left_hand")
            right_source = frame.get("right_hand")
            face_source = frame.get("face")
            pose = _point_array(pose_source, 33, "pose")
            left = _point_array(left_source, 21, "left_hand")
            right = _point_array(right_source, 21, "right_hand")
            face = _point_array(face_source, 455, "face")
            body = pose[list(MEDIAPIPE_BODY_INDICES)]
            # Uni-Sign's official preprocessing makes hand coordinates
            # wrist-relative and face points nose-relative before crop scaling.
            left = left - left[0]
            right = right - right[0]
            face = np.concatenate([
                face[list(MEDIAPIPE_FACE_JAW)],
                face[list(MEDIAPIPE_FACE_MOUTH)],
                face[[MEDIAPIPE_FACE_NOSE]],
            ])
            face = face - face[-1]

            body_frames.append(body)
            left_frames.append(left)
            right_frames.append(right)
            face_frames.append(face)
            body_scores.append(_scores(frame, 9, self.confidence_policy, "body", pose_source is not None))
            left_scores.append(_scores(frame, 21, self.confidence_policy, "left", left_source is not None))
            right_scores.append(_scores(frame, 21, self.confidence_policy, "right", right_source is not None))
            face_scores.append(_scores(frame, 18, self.confidence_policy, "face_all", face_source is not None))
            timestamps.append(float(frame.get("timestamp_ms", len(timestamps))))

        body_values = np.asarray(body_frames, dtype=np.float32)
        left_values = np.asarray(left_frames, dtype=np.float32)
        right_values = np.asarray(right_frames, dtype=np.float32)
        face_values = np.asarray(face_frames, dtype=np.float32)
        body_score_values = np.asarray(body_scores, dtype=np.float32)
        left_score_values = np.asarray(left_scores, dtype=np.float32)
        right_score_values = np.asarray(right_scores, dtype=np.float32)
        face_score_values = np.asarray(face_scores, dtype=np.float32)
        body_normalized, scale, _ = _crop_scale(body_values, body_score_values)

        return UniSignPoseSequence(
            body=np.concatenate([body_normalized, body_score_values[..., None]], axis=-1),
            left=np.concatenate([_normalize_part(left_values, left_score_values, np.zeros(2), scale), left_score_values[..., None]], axis=-1),
            right=np.concatenate([_normalize_part(right_values, right_score_values, np.zeros(2), scale), right_score_values[..., None]], axis=-1),
            face_all=np.concatenate([_normalize_part(face_values, face_score_values, np.zeros(2), scale), face_score_values[..., None]], axis=-1),
            timestamps_ms=np.asarray(timestamps, dtype=np.float64),
            confidence_policy=self.confidence_policy,
            topology_verified=False,
            normalized=True,
        )
