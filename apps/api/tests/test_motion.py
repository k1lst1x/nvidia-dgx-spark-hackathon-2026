from dataclasses import dataclass

from motion import MotionHistory


@dataclass
class Box:
    track_id: int | None = 7
    label: str = "person"
    confidence: float = 0.9
    x: float = 0.4
    y: float = 0.2
    width: float = 0.2
    height: float = 0.5


def _update(history: MotionHistory, depth: float, timestamp: float):
    return history.update("session", [Box()], {7: (depth, 0.95, 0.03)}, observed_at=timestamp)[7]


def test_camera_relative_approach_requires_consistent_metric_history():
    history = MotionHistory()
    result = None
    for timestamp, depth in [(0.0, 4.0), (0.2, 3.8), (0.4, 3.6), (0.6, 3.4), (0.8, 3.2), (1.0, 3.0)]:
        result = _update(history, depth, timestamp)
    assert result is not None
    assert result.trend == "approaching"
    assert result.relative_depth_velocity_mps is not None
    assert result.relative_depth_velocity_mps < 0
    assert result.time_to_closest_approach_s is not None
    assert result.camera_relative is True
    assert result.world_motion_status == "unavailable"


def test_single_or_noisy_measurements_do_not_claim_motion():
    history = MotionHistory()
    first = _update(history, 4.0, 0.0)
    assert first.trend == "unknown"
    # A 20m one-frame jump is rejected by the outlier gate.
    assert history.update("session", [Box()], {7: (24.0, 0.95, 0.03)}, observed_at=0.1) == {}


def test_stale_tracks_expire():
    history = MotionHistory(stale_seconds=0.5)
    _update(history, 4.0, 0.0)
    history.update("session", [], {}, observed_at=1.0)
    assert history.active_track_count("session") == 0
