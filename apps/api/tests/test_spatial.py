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
