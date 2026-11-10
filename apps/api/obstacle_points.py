"""Sparse obstacle dots from a metric depth map.

The detector only names things it knows (COCO classes). Walls, trees, poles,
hedges and other unlabeled surfaces still block the path, so this samples the
metric depth map on a coarse grid and returns a few camera-frame points for
the nearest surface in each cell.

Rules that keep the output small and honest:

* Only metric depth is used. Relative depth has no meters to clip against.
* Only surfaces within ``OBSTACLE_MAX_RANGE_M`` (default 4.5 m) are returned:
  monocular depth beyond a few meters is too noisy to draw as an obstacle.
* Cells mostly covered by a detected object's box are skipped, because that
  object already has its own labeled marker on the map.
* Horizontal surfaces (floor, ceiling) are skipped; vertical ones are kept.
* The output is capped at ``OBSTACLE_MAX_POINTS`` points per frame.

Coordinates follow ``point_cloud.cloud_for_surface``: x to the right, y down,
z forward from the camera, in meters. They are estimates, not a SLAM map.
"""

from __future__ import annotations

import os

import numpy as np

from point_cloud import camera_intrinsics

OBSTACLE_POINTS_ENABLED = os.getenv("OBSTACLE_POINTS_ENABLED", "true").lower() in {"1", "true", "yes"}
OBSTACLE_MIN_RANGE_M = float(os.getenv("OBSTACLE_MIN_RANGE_M", "0.25"))
OBSTACLE_MAX_RANGE_M = float(os.getenv("OBSTACLE_MAX_RANGE_M", "4.5"))
OBSTACLE_GRID_COLS = int(os.getenv("OBSTACLE_GRID_COLS", "24"))
OBSTACLE_GRID_ROWS = int(os.getenv("OBSTACLE_GRID_ROWS", "16"))
OBSTACLE_MAX_POINTS = int(os.getenv("OBSTACLE_MAX_POINTS", "192"))
# Floors and ceilings are surfaces too, but not obstacles. Without this the
# lower half of every frame would fill the map with floor dots.
OBSTACLE_SKIP_HORIZONTAL = os.getenv("OBSTACLE_SKIP_HORIZONTAL", "true").lower() in {"1", "true", "yes"}
# A cell this covered by a detected object's box belongs to that object.
_OBJECT_COVERAGE_SKIP = 0.5
# Fraction of valid pixels a cell needs before it is trusted.
_MIN_VALID_FRACTION = 0.35


def _object_mask(shape: tuple[int, int], objects) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for item in objects or []:
        x, y, w, h = (getattr(item, name, None) for name in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            continue
        left = max(0, int(x * width))
        right = min(width, int(np.ceil((x + w) * width)))
        top = max(0, int(y * height))
        bottom = min(height, int(np.ceil((y + h) * height)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _is_horizontal(grid: np.ndarray, r: int, c: int) -> bool:
    """True when the surface runs away from the camera rather than facing it.

    Between vertically adjacent cells, a floor changes mostly in depth (z) while
    a wall, trunk or pole changes mostly in height (y). The 1.5 ratio tolerates
    roughly 30 degrees of camera pitch, since the phone's tilt is unknown.
    """
    here = grid[r, c]
    votes = []
    for neighbor in (r - 1, r + 1):
        if not 0 <= neighbor < grid.shape[0]:
            continue
        other = grid[neighbor, c]
        if not np.isfinite(other[2]):
            continue
        dy = abs(float(other[1] - here[1]))
        dz = abs(float(other[2] - here[2]))
        votes.append(dz > 1.5 * dy)
    return bool(votes) and all(votes)


def obstacle_points(
    depth_map,
    objects=None,
    *,
    max_range_m: float | None = None,
    min_range_m: float | None = None,
    cols: int | None = None,
    rows: int | None = None,
    max_points: int | None = None,
    skip_horizontal: bool | None = None,
) -> list[list[float]]:
    """Return up to ``max_points`` [x, y, z] camera-frame points within range."""
    if depth_map is None or getattr(depth_map, "ndim", 0) < 2:
        return []
    max_range = OBSTACLE_MAX_RANGE_M if max_range_m is None else max_range_m
    min_range = OBSTACLE_MIN_RANGE_M if min_range_m is None else min_range_m
    cols = max(1, OBSTACLE_GRID_COLS if cols is None else cols)
    rows = max(1, OBSTACLE_GRID_ROWS if rows is None else rows)
    limit = max(0, OBSTACLE_MAX_POINTS if max_points is None else max_points)
    skip_horizontal = OBSTACLE_SKIP_HORIZONTAL if skip_horizontal is None else skip_horizontal
    if limit == 0 or max_range <= min_range:
        return []

    depth = np.asarray(depth_map, dtype=np.float32)[:, :, 0] if depth_map.ndim > 2 else np.asarray(depth_map, dtype=np.float32)
    height, width = depth.shape
    if height < rows or width < cols:
        return []
    covered = _object_mask((height, width), objects)
    fx, fy, cx, cy, _calibrated = camera_intrinsics(width, height)
    # Split the map into rows x cols equal blocks (dropping < 1 cell of edge
    # pixels) so every cell is reduced in one vectorized call.
    cell_h, cell_w = height // rows, width // cols
    step = 2 if cell_h * cell_w > 400 else 1  # plenty of pixels for a percentile
    blocks = depth[: rows * cell_h, : cols * cell_w].reshape(rows, cell_h, cols, cell_w)
    blocks = blocks[:, ::step, :, ::step].transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    valid = np.isfinite(blocks) & (blocks > 0)
    valid_count = valid.sum(axis=2)
    valid_fraction = valid_count / blocks.shape[2]
    coverage = (
        covered[: rows * cell_h, : cols * cell_w]
        .reshape(rows, cell_h, cols, cell_w).mean(axis=(1, 3))
    )
    usable = (valid_fraction >= _MIN_VALID_FRACTION) & (coverage < _OBJECT_COVERAGE_SKIP)
    z = np.full((rows, cols), np.nan, dtype=np.float32)
    if usable.any():
        # The near edge of the cell is what the user would walk into. A low
        # percentile ignores single-pixel spikes better than the minimum.
        # Invalid pixels sort last as +inf, so index into the valid prefix.
        ordered = np.sort(np.where(valid[usable], blocks[usable], np.inf), axis=1)
        rank = np.floor(0.2 * (valid_count[usable] - 1)).astype(int)
        z[usable] = np.take_along_axis(ordered, rank[:, None], axis=1)[:, 0]
    u = (np.arange(cols, dtype=np.float32) + 0.5) * cell_w
    v = (np.arange(rows, dtype=np.float32) + 0.5) * cell_h
    grid = np.stack((
        (u[None, :] - cx) * z / fx,
        (v[:, None] - cy) * z / fy,
        z,
    ), axis=2)

    found: list[tuple[float, list[float]]] = []
    for r in range(rows):
        for c in range(cols):
            x, y, z = (float(value) for value in grid[r, c])
            if not np.isfinite(z) or not (min_range <= z <= max_range):
                continue
            distance = float(np.sqrt(x * x + y * y + z * z))
            if distance > max_range:
                continue
            if skip_horizontal and _is_horizontal(grid, r, c):
                continue
            found.append((distance, [round(x, 3), round(y, 3), round(z, 3)]))

    # When over budget, keep the nearest surfaces: they matter most.
    found.sort(key=lambda entry: entry[0])
    return [point for _distance, point in found[:limit]]
