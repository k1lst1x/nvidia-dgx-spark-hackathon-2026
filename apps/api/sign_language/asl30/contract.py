"""Strict contract for the separate, fixed-vocabulary ASL-30 experiment.

The live 250-class recognizer consumes 390-value PopSign frames.  This module
does *not* adapt, replace, or load that recognizer.  It validates isolated
word recordings produced by the existing 281-value SignRecorder before they
can enter an ASL-30 training manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


FEATURE_SCHEMA = "visionmate-281-v1"
FEATURE_COUNT = 281
UNKNOWN_GLOSS = "UNKNOWN"

# Canonical training labels.  Keep display/English aliases outside this
# contract: a training label must describe the sign that a fluent reviewer
# approved, not merely a phrase shown in the UI.
TARGET_GLOSSES = (
    "HELLO", "GOODBYE", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "STOP", "WAIT",
    "WATER", "FOOD", "EAT", "DRINK", "BATHROOM", "HOME", "SCHOOL", "FRIEND", "FAMILY", "SLEEP",
    "GO", "COME", "WANT", "NEED", "LIKE", "UNDERSTAND", "KNOW", "WHERE", "WHAT", "WHO",
)
TRAINING_GLOSSES = frozenset((*TARGET_GLOSSES, UNKNOWN_GLOSS))


class CaptureValidationError(ValueError):
    """A recording is not safe to include in an ASL-30 train/eval manifest."""


@dataclass(frozen=True)
class Asl30Sample:
    """Validated isolated-sign sequence ready for JSONL serialization."""

    sample_id: str
    signer_id: str
    gloss: str
    fps: float
    frames: list[list[float]]
    timestamps_ms: list[int]

    def as_manifest_row(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task": "isolated_asl30",
            "feature_schema": FEATURE_SCHEMA,
            "id": self.sample_id,
            "signer_id": self.signer_id,
            "gloss": self.gloss,
            "fps": self.fps,
            "frames": self.frames,
            "timestamps_ms": self.timestamps_ms,
        }


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CaptureValidationError(f"{field} must be a non-empty string")
    return value.strip()


def canonical_gloss(value: str) -> str:
    """Normalize only spelling separators; never silently create an alias."""
    return "_".join(value.strip().upper().replace("-", " ").split())


def validate_capture(payload: Mapping[str, Any]) -> Asl30Sample:
    """Validate one isolated ASL-30 recording exported by ``SignRecorder``.

    A target example must have exactly one approved gloss. ``UNKNOWN`` is the
    required negative class for non-sign movement and unsupported signs.  The
    validation deliberately rejects multi-gloss sentence recordings, arbitrary
    feature dimensions, non-finite values, and non-monotonic timestamps.
    """
    if payload.get("language") != "ASL":
        raise CaptureValidationError("language must be 'ASL'")
    if payload.get("feature_schema") != FEATURE_SCHEMA:
        raise CaptureValidationError(
            f"feature_schema must be {FEATURE_SCHEMA!r}; do not mix 390-feature recordings"
        )

    sample_id = _required_text(payload, "id")
    signer_id = _required_text(payload, "signerId")
    raw_glosses = payload.get("glosses")
    if not isinstance(raw_glosses, list) or len(raw_glosses) != 1 or not isinstance(raw_glosses[0], str):
        raise CaptureValidationError("ASL-30 requires exactly one verified gloss per recording")
    gloss = canonical_gloss(raw_glosses[0])
    if gloss not in TRAINING_GLOSSES:
        allowed = ", ".join((*TARGET_GLOSSES, UNKNOWN_GLOSS))
        raise CaptureValidationError(f"unsupported ASL-30 gloss {gloss!r}; allowed: {allowed}")

    fps = payload.get("fps")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise CaptureValidationError("fps must be a positive finite number")
    frames = payload.get("frames")
    timestamps = payload.get("timestamps_ms")
    if not isinstance(frames, list) or not frames:
        raise CaptureValidationError("frames must be a non-empty list")
    if not isinstance(timestamps, list) or len(timestamps) != len(frames):
        raise CaptureValidationError("timestamps_ms must contain one timestamp per frame")

    checked_frames: list[list[float]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != FEATURE_COUNT:
            raise CaptureValidationError(f"frame {index} must contain exactly {FEATURE_COUNT} features")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in frame):
            raise CaptureValidationError(f"frame {index} contains a non-finite feature")
        checked_frames.append([float(value) for value in frame])

    checked_timestamps: list[int] = []
    previous = -1
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise CaptureValidationError(f"timestamp {index} must be finite")
        rounded = int(round(timestamp))
        if rounded < 0 or rounded <= previous:
            raise CaptureValidationError("timestamps_ms must be strictly increasing from zero")
        checked_timestamps.append(rounded)
        previous = rounded

    return Asl30Sample(
        sample_id=sample_id,
        signer_id=signer_id,
        gloss=gloss,
        fps=float(fps),
        frames=checked_frames,
        timestamps_ms=checked_timestamps,
    )
