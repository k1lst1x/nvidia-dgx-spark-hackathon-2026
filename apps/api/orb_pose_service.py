"""Local pose process for the ORB-SLAM3 adapter.

This machine has no C++ toolchain, so the official ORB-SLAM3 binary cannot be
built here. The process speaks the same /pose and /frame contract. It tracks
the camera with ORB features and an essential-matrix pose. The step size is
not metric, and the response says so.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 8099


class Tracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orb = cv2.ORB_create(800)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._previous = None
        self._rotation = np.eye(3)
        self._translation = np.zeros((3, 1))
        self._pose = {
            "tracking": "lost",
            "metric_scale": False,
            "source": "orb-pose",
            "estimate": True,
            "timestamp_s": time.time(),
            "translation_m": [0.0, 0.0, 0.0],
            "rotation": np.eye(3).tolist(),
            "confidence": 0.0,
        }

    def pose(self) -> dict:
        with self._lock:
            return dict(self._pose)

    def update(self, image_bytes: bytes) -> None:
        try:
            self._update(image_bytes)
        except Exception:
            with self._lock:
                self._pose = self._pack("lost", 0.0)

    def _update(self, image_bytes: bytes) -> None:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return
        if image.shape[1] > 640:
            scale = 640 / image.shape[1]
            image = cv2.resize(image, (640, int(image.shape[0] * scale)))
        points, descriptors = self._orb.detectAndCompute(image, None)
        with self._lock:
            previous = self._previous
            self._previous = (points, descriptors)
            if previous is None or descriptors is None or previous[1] is None or len(points) < 20:
                self._pose = self._pack("lost", 0.0)
                return
            pairs = self._matcher.knnMatch(previous[1], descriptors, k=2)
            matches = [first for first, *rest in pairs if rest and first.distance < 0.75 * rest[0].distance]
            if len(matches) < 30:
                self._pose = self._pack("lost", 0.2)
                return
            first_points = np.float32([previous[0][match.queryIdx].pt for match in matches])
            second_points = np.float32([points[match.trainIdx].pt for match in matches])
            shift = float(np.median(np.linalg.norm(second_points - first_points, axis=1)))
            if shift < 1.5:
                self._pose = self._pack("ok", 0.8)
                return
            height, width = image.shape[:2]
            focal = width / (2 * np.tan(np.deg2rad(30)))
            intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
            essential, _mask = cv2.findEssentialMat(first_points, second_points, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if essential is None:
                self._pose = self._pack("ok" if len(matches) >= 80 else "lost", 0.5 if len(matches) >= 80 else 0.2)
                return
            inliers, rotation, step, _pose_mask = cv2.recoverPose(essential, first_points, second_points, intrinsics)
            inliers = int(inliers or 0)
            if inliers < 20:
                self._pose = self._pack("ok" if len(matches) >= 80 else "lost", 0.5 if len(matches) >= 80 else 0.2)
                return
            # recoverPose returns a unit step. Keep it small and labeled non-metric.
            self._translation = self._translation + self._rotation @ (step * 0.05)
            self._rotation = rotation @ self._rotation
            confidence = max(0.0, min(1.0, inliers / max(len(matches), 1)))
            self._pose = self._pack("ok", confidence)

    def _pack(self, tracking: str, confidence: float) -> dict:
        return {
            "tracking": tracking,
            "metric_scale": False,
            "source": "orb-pose",
            "estimate": True,
            "timestamp_s": time.time(),
            "translation_m": [round(float(value), 4) for value in self._translation.flatten()],
            "rotation": [[round(float(value), 5) for value in row] for row in self._rotation.tolist()],
            "confidence": round(confidence, 3),
        }


TRACKER = Tracker()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] != "/pose":
            self.send_error(404)
            return
        body = json.dumps(TRACKER.pose()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/frame":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        TRACKER.update(self.rfile.read(length))
        self.send_response(202)
        self.end_headers()

    def log_message(self, _format, *_args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"pose service listening on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
