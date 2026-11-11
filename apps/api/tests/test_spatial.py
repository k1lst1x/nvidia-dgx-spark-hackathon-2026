import numpy as np

from depth import DepthAnythingMetric, reject_depth_outliers, surface_readings_for_boxes
from point_cloud import cloud_for_surface
from segmentation import YoloSegmenter
from slam import OrbSlamClient
from visualization import RerunView


class Box:
    def __init__(self):
        self.track_id = 3
        self.label = "chair"
        self.x = 0.25
        self.y = 0.25
        self.width = 0.5
        self.height = 0.5


def test_mask_depth_ignores_the_background_inside_the_box():
    depth = np.full((40, 40), 6.0, dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    mask[16:24, 16:24] = True
    depth[mask] = 1.5
    readings = surface_readings_for_boxes(depth, [Box()], [mask])
    assert len(readings) == 1
    assert abs(readings[0].depth_m - 1.5) < 0.05


def test_box_depth_still_works_without_a_mask():
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    readings = surface_readings_for_boxes(depth, [Box()], [None])
    assert len(readings) == 1
    assert abs(readings[0].depth_m - 2.0) < 0.05


def test_point_cloud_center_uses_only_the_masked_surface():
    depth = np.full((40, 40), 4.0, dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    depth[mask] = 2.0
    summary = cloud_for_surface(depth, mask, 40, 40)
    assert summary is not None
    assert summary["estimate"] is True
    assert summary["calibrated_camera"] is False
    assert abs(summary["center_m"][2] - 2.0) < 0.05
    assert all(value >= 0 for value in summary["extent_m"])


def test_segmentation_stays_off_unless_requested(monkeypatch):
    monkeypatch.delenv("ULTRALYTICS_YOLO_SEG_ENABLED", raising=False)
    segmenter = YoloSegmenter()
    assert segmenter.enabled is False
    assert segmenter.masks_for(b"not-an-image", [Box()]) == [None]


def test_enabled_segmentation_failure_returns_no_masks(monkeypatch):
    monkeypatch.setenv("ULTRALYTICS_YOLO_SEG_ENABLED", "true")
    segmenter = YoloSegmenter()
    assert segmenter.masks_for(b"not-an-image", [Box()]) == [None]
    assert segmenter.last_ms is not None


def test_empty_mask_falls_back_to_the_box_interior():
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    readings = surface_readings_for_boxes(depth, [Box()], [mask])
    assert len(readings) == 1
    assert readings[0].source == "box"
    assert abs(readings[0].depth_m - 2.0) < 0.05


def test_depth_spikes_are_rejected():
    values = np.concatenate([np.full(80, 2.0), np.full(8, 40.0)])
    kept = reject_depth_outliers(values)
    assert abs(float(np.median(kept)) - 2.0) < 0.05
    assert float(np.max(kept)) < 10


def test_metric_depth_stays_off_without_weights(monkeypatch):
    monkeypatch.setenv("DEPTH_ANYTHING_ENABLED", "true")
    monkeypatch.setenv("DEPTH_ANYTHING_MODE", "metric")
    monkeypatch.setenv("DEPTH_ANYTHING_CHECKPOINT", r"C:\missing\weights.pth")
    monkeypatch.setenv("DEPTH_ANYTHING_REPO", r"C:\missing\repo")
    model = DepthAnythingMetric()
    assert model.is_metric is True
    assert model.ready is False
    assert model.status()["weights_present"] is False
    assert model.infer(b"not-an-image") is None


def test_slam_disabled_does_not_connect(monkeypatch):
    monkeypatch.delenv("ORB_SLAM3_ENABLED", raising=False)
    client = OrbSlamClient()
    assert client.latest("session") is None
    assert client.pose_for_motion(640, 480, {"available": True, "metric_scale": False}) is None


def test_slam_outage_keeps_detection_independent(monkeypatch):
    monkeypatch.setenv("ORB_SLAM3_ENABLED", "true")
    monkeypatch.setenv("ORB_SLAM3_ENDPOINT", "http://127.0.0.1:9/pose")
    client = OrbSlamClient()
    snapshot = client.latest("session")
    assert snapshot is not None
    assert snapshot["available"] is False
    assert snapshot["metric_scale"] is False
    assert client.pose_for_motion(640, 480, snapshot) is None


def test_rerun_stays_off(monkeypatch):
    monkeypatch.delenv("RERUN_ENABLED", raising=False)
    view = RerunView()
    view.log([])
    assert view.enabled is False
    assert view.status()["active"] is False


def test_open3d_records_the_cloud_when_installed(monkeypatch):
    import sys
    import types

    geometry = types.ModuleType("open3d.geometry")
    utility = types.ModuleType("open3d.utility")

    class PointCloud:
        def __init__(self):
            self.points = []

    class Vector3dVector:
        def __init__(self, points):
            self.points = points

        def __len__(self):
            return len(self.points)

    geometry.PointCloud = PointCloud
    utility.Vector3dVector = Vector3dVector
    open3d = types.ModuleType("open3d")
    open3d.geometry = geometry
    open3d.utility = utility
    monkeypatch.setitem(sys.modules, "open3d", open3d)
    monkeypatch.setitem(sys.modules, "open3d.geometry", geometry)
    monkeypatch.setitem(sys.modules, "open3d.utility", utility)
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    summary = cloud_for_surface(depth, mask, 40, 40)
    assert summary is not None
    assert summary["open3d"] is True


def test_obstacle_points_keep_only_surfaces_within_range():
    from obstacle_points import obstacle_points

    depth = np.full((64, 96), 8.0, dtype=np.float32)
    depth[:, :32] = 2.0  # a wall on the left, 2 m away
    points = obstacle_points(depth, [], max_range_m=4.5, cols=12, rows=8, max_points=500)
    assert points, "the near wall should produce dots"
    for x, y, z in points:
        assert abs(z - 2.0) < 0.05
        assert x < 0  # left of the optical center
        assert (x * x + y * y + z * z) ** 0.5 <= 4.5


def test_obstacle_points_skip_detected_objects_and_respect_the_cap():
    from obstacle_points import obstacle_points

    depth = np.full((64, 96), 3.0, dtype=np.float32)
    everything = obstacle_points(depth, [], max_range_m=4.5, cols=12, rows=8, max_points=500)
    with_box = obstacle_points(depth, [Box()], max_range_m=4.5, cols=12, rows=8, max_points=500)
    assert len(with_box) < len(everything)
    capped = obstacle_points(depth, [], max_range_m=4.5, cols=12, rows=8, max_points=10)
    assert len(capped) == 10


def test_obstacle_points_ignore_the_floor_but_keep_a_wall():
    from obstacle_points import obstacle_points
    from point_cloud import camera_intrinsics

    height, width = 96, 128
    _fx, fy, _cx, cy, _ = camera_intrinsics(width, height)
    rows = np.arange(height, dtype=np.float32)[:, None]
    # A level camera 1.45 m above a flat floor: depth = h * fy / (v - cy).
    floor = np.where(rows > cy + 1, 1.45 * fy / np.maximum(rows - cy, 1e-3), 30.0)
    depth = np.broadcast_to(floor, (height, width)).astype(np.float32).copy()
    assert obstacle_points(depth, [], max_range_m=4.5, cols=16, rows=12, max_points=500) == []
    depth[:, 96:] = 1.8  # a pole/wall on the right, 1.8 m away
    points = obstacle_points(depth, [], max_range_m=4.5, cols=16, rows=12, max_points=500)
    assert points
    assert all(x > 0 and abs(z - 1.8) < 0.05 for x, _y, z in points)


def test_obstacle_points_need_a_depth_map():
    from obstacle_points import obstacle_points

    assert obstacle_points(None, []) == []
    nan_map = np.full((64, 96), np.nan, dtype=np.float32)
    assert obstacle_points(nan_map, [], cols=12, rows=8) == []


def test_track_geometry_returns_obstacles_without_any_detection(monkeypatch):
    import main

    depth = np.full((64, 96), 2.5, dtype=np.float32)
    monkeypatch.setattr(main, "OBSTACLE_POINTS_ENABLED", True)
    monkeypatch.setattr(type(main._depth_model), "ready", property(lambda self: True))
    monkeypatch.setattr(main._depth_model, "mode", "metric")
    calls = []
    monkeypatch.setattr(main._depth_model, "infer", lambda body: calls.append(body) or depth)
    objects, obstacles = main._track_geometry([], b"frame", "session")
    assert objects == []
    assert obstacles
    assert len(calls) == 1  # depth runs once per frame


def test_gpu_status_parses_nvidia_smi_and_handles_no_gpu(monkeypatch):
    import gpu_status

    parsed = gpu_status.parse("NVIDIA GB10, 37, 20480, [N/A], 61\n")
    assert parsed["available"] is True
    gpu = parsed["gpus"][0]
    assert gpu["name"] == "NVIDIA GB10" and gpu["utilization_pct"] == 37 and gpu["memory_total_mb"] is None
    monkeypatch.setattr(gpu_status.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu_status, "_cache", None)
    assert gpu_status.gpu_status() == {"available": False, "gpus": []}
