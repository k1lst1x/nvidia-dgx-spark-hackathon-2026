"""Real YOLO11n + BoT-SORT on every device this machine has.

Skipped where the device, Ultralytics or the weights are missing, so the same
file runs on a Mac (cpu, mps), an NVIDIA machine (cpu, cuda) and a CPU-only
machine (cpu). Frames are a real photo shifted a few pixels at a time, like a
slowly panning camera.
"""

from io import BytesIO
from pathlib import Path

import pytest

import device_manager
from yolo_tracker import YoloTracker

API = Path(__file__).resolve().parents[1]
WEIGHTS = API / "models" / "yolo11n.pt"


def _available(device: str) -> bool:
    try:
        import torch
    except ImportError:
        return device == "cpu"
    if device.startswith("cuda"):
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def _frames(count: int = 8) -> list[bytes]:
    from PIL import Image
    from ultralytics.utils import ASSETS

    photo = Image.open(ASSETS / "bus.jpg").convert("RGB")
    frames = []
    for step in range(count):
        crop = photo.crop((step * 6, step * 4, step * 6 + 640, step * 4 + 640))
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
        frames.append(buffer.getvalue())
    return frames


@pytest.fixture(autouse=True)
def fresh_hardware():
    device_manager.hardware.cache_clear()
    yield
    device_manager.hardware.cache_clear()


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
def test_yolo_tracks_real_frames_with_stable_ids(device):
    pytest.importorskip("ultralytics")
    if not WEIGHTS.is_file():
        pytest.skip(f"{WEIGHTS.name} is not present")
    if not _available(device):
        pytest.skip(f"{device} is not available to this PyTorch build")
    resets = []
    tracker = YoloTracker(enabled=True, model_path=str(WEIGHTS), device=device, confidence=0.4,
                          tracker_config=str(API / "botsort.yaml"), idle_seconds=60, on_reset=resets.append)
    tracker.load()
    assert tracker.device == device, device_manager.report()["components"].get("yolo")

    ids_per_frame = []
    for frame in _frames():
        objects, _ = tracker.detect_with_masks(frame, "hardware-test")
        assert any(item.label == "person" for item in objects)
        ids_per_frame.append({item.track_id for item in objects if item.track_id is not None})
    persistent = set.intersection(*ids_per_frame[1:])
    assert persistent, f"no track ID survived consecutive frames on {device}: {ids_per_frame}"
    assert tracker.device == device and tracker.restarts == 0 and resets == []


def test_missing_gpu_request_runs_on_cpu():
    pytest.importorskip("ultralytics")
    if not WEIGHTS.is_file():
        pytest.skip(f"{WEIGHTS.name} is not present")
    tracker = YoloTracker(enabled=True, model_path=str(WEIGHTS), device="cuda:7", confidence=0.4,
                          tracker_config=str(API / "botsort.yaml"), idle_seconds=60)
    tracker.load()
    assert tracker.device == "cpu"
    objects, _ = tracker.detect_with_masks(_frames(1)[0], "fallback-test")
    assert objects
