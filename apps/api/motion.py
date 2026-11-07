"""Conservative, experimental geometry for a moving-camera accessibility demo.

BoT-SORT's global-motion compensation makes 2-D association more resilient to
camera motion.  It does *not* estimate the glasses' 3-D pose.  This module
therefore reports camera-relative depth trends by default and exposes a
separate, opt-in pose interface for experiments that have calibrated depth and
camera extrinsics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol


class TrackedBox(Protocol):
    label: str
    confidence: float
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    track_id: int | None


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pixel camera intrinsics required for metric 3-D backprojection."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class CameraPose:
    """World-from-camera rigid transform, supplied by a calibrated device."""

    timestamp_s: float
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation_m: tuple[float, float, float]
    intrinsics: CameraIntrinsics
    confidence: float = 1.0


@dataclass(frozen=True)
class IMUSample:
    timestamp_s: float
    linear_acceleration_mps2: tuple[float, float, float]
    angular_velocity_rps: tuple[float, float, float]
    confidence: float = 0.0


class EgoMotionProvider(Protocol):
    """Optional bridge to a glasses AR/SLAM or IMU subsystem.

    Implementations must provide pose in a documented world reference frame.
    IMU alone is deliberately insufficient: integrating it without a stable
    pose source drifts rapidly and is never used to manufacture world motion.
    """

    def latest_pose(self, timestamp_s: float) -> CameraPose | None: ...

    def latest_imu(self, timestamp_s: float) -> IMUSample | None: ...


@dataclass(frozen=True)
class SurfaceMeasurement:
    track_id: int
    label: str
    timestamp_s: float
    center_x: float
    center_y: float
    depth_m: float
    confidence: float
    depth_spread_m: float
    world_point_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class MotionEstimate:
    track_id: int
    trend: str
    direction: str
    distance_m: float | None
    relative_depth_velocity_mps: float | None
    time_to_closest_approach_s: float | None
    minimum_separation_m: float | None
    confidence: float
    sample_count: int
    camera_relative: bool = True
    world_motion_status: str = "unavailable"
    world_speed_mps: float | None = None
    experimental: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "trend": self.trend,
            "direction": self.direction,
            "distance_m": self.distance_m,
            "relative_depth_velocity_mps": self.relative_depth_velocity_mps,
            "time_to_closest_approach_s": self.time_to_closest_approach_s,
            "minimum_separation_m": self.minimum_separation_m,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "camera_relative": self.camera_relative,
            "world_motion_status": self.world_motion_status,
            "world_speed_mps": self.world_speed_mps,
            "experimental": True,
        }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _robust_slope(samples: list[SurfaceMeasurement]) -> tuple[float | None, float]:
    """Theil-Sen slope and a conservative temporal/depth consistency score."""
    pairs: list[float] = []
    for left_index, left in enumerate(samples):
        for right in samples[left_index + 1:]:
            elapsed = right.timestamp_s - left.timestamp_s
            if elapsed >= 0.08:
                pairs.append((right.depth_m - left.depth_m) / elapsed)
    if len(pairs) < 3:
        return None, 0.0
    slope = _median(pairs)
    residual = _median([abs(value - slope) for value in pairs])
    # 0.25 m/s spread is already a weak result for a monocular system.
    stability = max(0.0, min(1.0, 1.0 - residual / 0.25))
    return slope, stability


def _world_point(
    box: TrackedBox, depth_m: float, pose: CameraPose | None
) -> tuple[float, float, float] | None:
    if pose is None or None in (box.x, box.y, box.width, box.height):
        return None
    intrinsics = pose.intrinsics
    if min(intrinsics.fx, intrinsics.fy, intrinsics.image_width, intrinsics.image_height) <= 0:
        return None
    u = (box.x + box.width / 2) * intrinsics.image_width
    v = (box.y + box.height / 2) * intrinsics.image_height
    point = ((u - intrinsics.cx) * depth_m / intrinsics.fx, (v - intrinsics.cy) * depth_m / intrinsics.fy, depth_m)
    rotated = tuple(sum(pose.rotation[row][column] * point[column] for column in range(3)) for row in range(3))
    return tuple(rotated[index] + pose.translation_m[index] for index in range(3))


class MotionHistory:
    """Short, session-scoped history with stale-track expiry and outlier gates."""

    def __init__(self, history_seconds: float = 4.0, stale_seconds: float = 2.0) -> None:
        self.history_seconds = history_seconds
        self.stale_seconds = stale_seconds
        self._tracks: dict[str, dict[int, deque[SurfaceMeasurement]]] = defaultdict(dict)

    def _expire(self, session_id: str, now: float) -> None:
        tracks = self._tracks[session_id]
        for track_id in list(tracks):
            values = tracks[track_id]
            while values and now - values[0].timestamp_s > self.history_seconds:
                values.popleft()
            if not values or now - values[-1].timestamp_s > self.stale_seconds:
                del tracks[track_id]

    def update(
        self,
        session_id: str,
        boxes: list[TrackedBox],
        readings: dict[int, tuple[float, float, float]],
        *,
        observed_at: float | None = None,
        pose: CameraPose | None = None,
    ) -> dict[int, MotionEstimate]:
        now = observed_at if observed_at is not None else time.monotonic()
        self._expire(session_id, now)
        tracks = self._tracks[session_id]
        results: dict[int, MotionEstimate] = {}
        for box in boxes:
            if box.track_id is None or box.track_id not in readings:
                continue
            depth_m, depth_confidence, depth_spread_m = readings[box.track_id]
            if not math.isfinite(depth_m) or depth_m <= 0:
                continue
            history = tracks.setdefault(box.track_id, deque(maxlen=120))
            # Reject a one-frame spike rather than claiming movement.  A large
            # change is accepted only after it persists in a later observation.
            if history:
                previous = history[-1]
                elapsed = max(now - previous.timestamp_s, 1e-3)
                max_plausible_jump = max(1.5, 3.0 * elapsed + previous.depth_m * 0.5)
                if abs(depth_m - previous.depth_m) > max_plausible_jump:
                    continue
            measurement_confidence = max(0.0, min(1.0, box.confidence * depth_confidence))
            sample = SurfaceMeasurement(
                track_id=box.track_id,
                label=box.label,
                timestamp_s=now,
                center_x=(box.x or 0) + (box.width or 0) / 2,
                center_y=(box.y or 0) + (box.height or 0) / 2,
                depth_m=depth_m,
                confidence=measurement_confidence,
                depth_spread_m=depth_spread_m,
                world_point_m=_world_point(box, depth_m, pose),
            )
            history.append(sample)
            results[box.track_id] = self._estimate(history, sample)
        return results

    def _estimate(self, history: deque[SurfaceMeasurement], current: SurfaceMeasurement) -> MotionEstimate:
        samples = list(history)
        slope, stability = _robust_slope(samples)
        recent_confidence = _median([sample.confidence for sample in samples[-5:]])
        confidence = max(0.0, min(1.0, recent_confidence * stability * min(1.0, len(samples) / 6)))
        trend = "unknown"
        if slope is not None and confidence >= 0.45:
            trend = "approaching" if slope <= -0.15 else "receding" if slope >= 0.15 else "stable"
        direction = "left" if current.center_x < 0.33 else "right" if current.center_x > 0.67 else "ahead"
        tca: float | None = None
        minimum: float | None = None
        if trend == "approaching" and slope is not None:
            # Camera-relative 1-D extrapolation only, not collision prediction.
            tca = min(8.0, current.depth_m / max(-slope, 1e-6))
            minimum = 0.0
        world_samples = [sample for sample in samples if sample.world_point_m is not None]
        world_speed = None
        world_status = "unavailable"
        if len(world_samples) >= 2:
            before, after = world_samples[-2], world_samples[-1]
            elapsed = after.timestamp_s - before.timestamp_s
            if elapsed >= 0.08 and before.world_point_m and after.world_point_m:
                world_speed = math.dist(before.world_point_m, after.world_point_m) / elapsed
                world_status = "experimental"
        return MotionEstimate(
            track_id=current.track_id,
            trend=trend,
            direction=direction,
            distance_m=current.depth_m,
            relative_depth_velocity_mps=round(slope, 3) if slope is not None and confidence >= 0.45 else None,
            time_to_closest_approach_s=round(tca, 2) if tca is not None and confidence >= 0.55 else None,
            minimum_separation_m=minimum if tca is not None and confidence >= 0.55 else None,
            confidence=round(confidence, 3),
            sample_count=len(samples),
            world_motion_status=world_status,
            world_speed_mps=round(world_speed, 3) if world_speed is not None else None,
        )

    def active_track_count(self, session_id: str) -> int:
        self._expire(session_id, time.monotonic())
        return len(self._tracks.get(session_id, {}))
