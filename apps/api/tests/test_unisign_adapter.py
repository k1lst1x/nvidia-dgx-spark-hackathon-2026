import numpy as np
import pytest

from unisign_adapter import UniSignAdapter, UniSignCompatibilityError


def raw_frame(timestamp: float, *, include_scores: bool = False):
    frame = {
        "timestamp_ms": timestamp,
        "left_hand": [{"x": 0.3 + i / 1000, "y": 0.4, "z": 0} for i in range(21)],
        "right_hand": [{"x": 0.7 + i / 1000, "y": 0.4, "z": 0} for i in range(21)],
        "pose": [{"x": 0.2 + i / 100, "y": 0.2 + i / 200, "z": 0} for i in range(33)],
        "face": [{"x": 0.25 + i / 2000, "y": 0.15 + i / 3000, "z": 0} for i in range(478)],
        "confidence_representation": "not_available_from_mediapipe_tasks",
    }
    if include_scores:
        frame.update({
            "body_confidence": [0.9] * 9,
            "left_confidence": [0.9] * 21,
            "right_confidence": [0.9] * 21,
            "face_all_confidence": [0.9] * 18,
        })
    return frame


def test_rejects_unavailable_mediapipe_confidence_by_default():
    with pytest.raises(UniSignCompatibilityError, match="confidence"):
        UniSignAdapter().adapt([raw_frame(1)])


def test_presence_proxy_emits_official_part_shapes_with_explicit_warning_metadata():
    result = UniSignAdapter("presence_proxy").adapt([raw_frame(1), raw_frame(51)])
    inputs = result.as_model_input()
    assert inputs["body"].shape == (1, 2, 9, 3)
    assert inputs["left"].shape == (1, 2, 21, 3)
    assert inputs["right"].shape == (1, 2, 21, 3)
    assert inputs["face_all"].shape == (1, 2, 18, 3)
    assert result.confidence_policy == "presence_proxy"
    assert result.topology_verified is False
    assert np.all(np.isfinite(inputs["face_all"]))


def test_supplied_scores_are_preserved_without_proxy():
    frame = raw_frame(1, include_scores=True)
    frame.pop("confidence_representation")
    result = UniSignAdapter().adapt([frame])
    assert np.allclose(result.body[..., 2], 0.9)
