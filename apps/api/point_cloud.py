"""Visible-surface point clouds from a depth map and an optional object mask.

Open3D is used when installed. The coordinates themselves come from a pinhole
projection, so a missing Open3D install still returns a center and an extent.
These are estimates from the depth map, not measured object dimensions.
"""

from __future__ import annotations

import os

import numpy as np


def open3d_status() -> dict[str, object]:
    try:
        import open3d as o3d

        return {"installed": True, "version": str(getattr(o3d, "__version__", "unknown"))}
    except Exception:
        return {"installed": False, "version": None}


def camera_intrinsics(width: int, height: int) -> tuple[float, float, float, float, bool]:
    """Return fx, fy, cx, cy and whether the caller supplied a calibration."""
    supplied = all(os.getenv(name) for name in ("CAMERA_FX", "CAMERA_FY", "CAMERA_CX", "CAMERA_CY"))
    if supplied:
        return (
            float(os.environ["CAMERA_FX"]), float(os.environ["CAMERA_FY"]),
            float(os.environ["CAMERA_CX"]), float(os.environ["CAMERA_CY"]), True,
        )
    # A 60-degree horizontal field is only a local-test stand-in.
    fx = width / (2 * np.tan(np.deg2rad(60) / 2))
    return fx, fx, width / 2, height / 2, False


def cloud_for_surface(depth_map, mask, width: int, height: int) -> dict[str, object] | None:
    if depth_map is None:
        return None
    region = np.ones(depth_map.shape[:2], dtype=bool) if mask is None else mask.astype(bool)
    if region.shape[:2] != depth_map.shape[:2]:
        return None
    rows, cols = np.nonzero(region)
    if rows.size < 9:
        return None
    depth = depth_map[rows, cols].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    rows, cols, depth = rows[valid], cols[valid], depth[valid]
    if depth.size >= 12:
        center = float(np.median(depth))
        spread = float(np.median(np.abs(depth - center)))
        limit = max(3 * 1.4826 * spread, center * 0.35, 0.2)
        keep = np.abs(depth - center) <= limit
        if int(keep.sum()) >= 9:
            rows, cols, depth = rows[keep], cols[keep], depth[keep]
    if depth.size < 9:
        return None
    if depth.size > 800:
        choice = np.linspace(0, depth.size - 1, 800).astype(int)
        rows, cols, depth = rows[choice], cols[choice], depth[choice]
    fx, fy, cx, cy, calibrated = camera_intrinsics(width, height)
    x = (cols - cx) * depth / fx
    y = (rows - cy) * depth / fy
    points = np.stack((x, y, depth), axis=1)
    low = np.percentile(points, 5, axis=0)
    high = np.percentile(points, 95, axis=0)
    summary: dict[str, object] = {
        "estimate": True,
        "calibrated_camera": calibrated,
        "point_count": int(points.shape[0]),
        "center_m": [round(float(value), 3) for value in np.median(points, axis=0)],
        "extent_m": [round(float(value), 3) for value in (high - low)],
        "open3d": False,
    }
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        summary["open3d"] = True
        summary["point_count"] = len(cloud.points)
    except Exception:
        summary["open3d"] = False
    return summary
