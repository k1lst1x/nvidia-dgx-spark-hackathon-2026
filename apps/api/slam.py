"""Optional pose client for a separate ORB-SLAM3 process.

The detector, BoT-SORT tracker, and depth path never wait on this client
unless ORB_SLAM3_ENABLED is true. Monocular ORB-SLAM3 has no absolute scale,
so a pose is passed into world-motion math only when the external process
explicitly reports metric scale and this machine has camera intrinsics.
"""

from __future__ import annotations

import logging
import os
import threading

from motion import CameraIntrinsics, CameraPose

logger = logging.getLogger("guidelens.slam")


def _flag(name: str) -> bool:
    return os.getenv(name, "false").lower() in {"1", "true", "yes"}


class OrbSlamClient:
    def __init__(self) -> None:
        self.enabled = _flag("ORB_SLAM3_ENABLED")
        self.endpoint = os.getenv("ORB_SLAM3_ENDPOINT", "http://127.0.0.1:8099/pose")
        self._lock = threading.Lock()
        self._snapshots: dict[str, dict[str, object]] = {}
        self._warned = False
        # The pose process is slower than detection. Frames go to it from one
        # background worker, newest first; a request never waits for a pose.
        self._pending: tuple[str, bytes] | None = None
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "source": "orb-slam3" if self.enabled else "disabled",
            "metric_scale": False,
            "fallback": "camera-relative-motion",
        }

    def latest(self, session_id: str, image_bytes: bytes | None = None) -> dict[str, object] | None:
        """The newest pose for this session (from an earlier frame); queues this frame."""
        if not self.enabled:
            return None
        if image_bytes:
            with self._lock:
                # Only the newest frame matters; an unsent older one is dropped.
                self._pending = (session_id, image_bytes)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._run, name="orb-pose", daemon=True)
                    self._worker.start()
            self._wake.set()
        with self._lock:
            return self._snapshots.get(session_id) or self._unavailable()

    def _run(self) -> None:
        while True:
            if not self._wake.wait(timeout=30):
                with self._lock:
                    if self._pending is None:
                        self._worker = None
                        return
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None:
                continue
            session_id, image_bytes = pending
            self._push_frame(image_bytes)
            snapshot = self._fetch()
            with self._lock:
                self._snapshots[session_id] = snapshot

    def snapshot_for(self, session_id: str) -> dict[str, object] | None:
        with self._lock:
            return self._snapshots.get(session_id)

    def pose_for_motion(self, width: int, height: int, snapshot: dict[str, object] | None) -> CameraPose | None:
        """Return a pose only when scale and calibration are both present."""
        if not snapshot or not snapshot.get("available") or not snapshot.get("metric_scale"):
            return None
        if not all(os.getenv(name) for name in ("CAMERA_FX", "CAMERA_FY", "CAMERA_CX", "CAMERA_CY")):
            return None
        rotation = snapshot.get("rotation")
        translation = snapshot.get("translation_m")
        if not _matrix(rotation) or not _vector(translation):
            return None
        return CameraPose(
            timestamp_s=float(snapshot.get("timestamp_s") or 0),
            rotation=tuple(tuple(float(value) for value in row) for row in rotation),
            translation_m=tuple(float(value) for value in translation),
            intrinsics=CameraIntrinsics(
                fx=float(os.environ["CAMERA_FX"]),
                fy=float(os.environ["CAMERA_FY"]),
                cx=float(os.environ["CAMERA_CX"]),
                cy=float(os.environ["CAMERA_CY"]),
                image_width=width,
                image_height=height,
            ),
            confidence=float(snapshot.get("confidence") or 0),
        )

    @staticmethod
    def _unavailable() -> dict[str, object]:
        return {
            "available": False,
            "tracking": "unavailable",
            "metric_scale": False,
            "source": "orb-slam3",
            "estimate": True,
        }

    def _fetch(self) -> dict[str, object]:
        unavailable = self._unavailable()
        try:
            import httpx

            response = httpx.get(self.endpoint, timeout=0.15)
            response.raise_for_status()
            payload = response.json()
            self._warned = False
        except Exception:
            if not self._warned:
                logger.warning("orb_slam3_unavailable endpoint=%s", self.endpoint)
                self._warned = True
            return unavailable
        tracking = str(payload.get("tracking") or "lost")
        metric_scale = bool(payload.get("metric_scale"))
        return {
            "available": tracking == "ok",
            "tracking": tracking,
            "metric_scale": metric_scale,
            "source": str(payload.get("source") or "orb-slam3"),
            "estimate": True,
            "timestamp_s": payload.get("timestamp_s"),
            "translation_m": payload.get("translation_m"),
            "rotation": payload.get("rotation"),
            "confidence": payload.get("confidence"),
        }

    def _push_frame(self, image_bytes: bytes) -> None:
        frame_url = self.endpoint.rsplit("/", 1)[0] + "/frame"
        try:
            import httpx

            httpx.post(frame_url, content=image_bytes, headers={"Content-Type": "application/octet-stream"}, timeout=0.8)
        except Exception:
            return


def _matrix(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _vector(value) -> bool:
    return isinstance(value, list) and len(value) == 3
