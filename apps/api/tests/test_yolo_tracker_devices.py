"""YoloTracker device handling with a fake Ultralytics model."""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import device_manager
from yolo_tracker import YoloTracker


class Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class FakeModel:
    """Tracks one box whose ID is kept by its own tracker, like BoT-SORT with persist=True."""

    def __init__(self, path, fail_track_on=None):
        self.path = path
        self.devices = []
        self.fail_track_on = fail_track_on
        self.next_id = 1
        self.predictor = SimpleNamespace(trackers=[SimpleNamespace(reset=self.reset)])

    def reset(self):
        self.next_id += 100

    def predict(self, *, device, **_):
        self.devices.append(("predict", device))

    def track(self, *, device, **_):
        self.devices.append(("track", device))
        if device == self.fail_track_on:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        box = SimpleNamespace(xyxy=[Value([10.0, 10.0, 60.0, 60.0])], conf=[Value(0.9)], cls=[Value(0)],
                              id=[Value(self.next_id)])
        return [SimpleNamespace(boxes=Boxes([box], self.next_id), names={0: "person"}, masks=None)]


class Boxes:
    is_track = True

    def __init__(self, boxes, track_id):
        self._boxes = boxes
        self.id = Value([track_id])

    def __iter__(self):
        return iter(self._boxes)


def frame():
    buffer = BytesIO()
    Image.new("RGB", (100, 100), (90, 90, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def gpu_machine(monkeypatch):
    properties = SimpleNamespace(name="Test GPU", total_memory=8 * 1024**3, major=8, minor=9)
    torch = SimpleNamespace(
        __version__="2.14.1", version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_properties=lambda index: properties, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(device_manager, "_torch", lambda: torch)
    monkeypatch.setattr(device_manager, "_nvidia_smi", lambda: None)
    for name in ("YOLO_DEVICE", "ULTRALYTICS_YOLO_DEVICE", "YOLO_HALF"):
        monkeypatch.delenv(name, raising=False)
    device_manager.hardware.cache_clear()
    yield
    device_manager.hardware.cache_clear()


def tracker(monkeypatch, models, resets=None):
    instance = YoloTracker(enabled=True, model_path="gpu-model.pt", cpu_model_path="cpu-model.pt", confidence=0.4,
                           tracker_config="botsort.yaml", idle_seconds=60,
                           on_reset=resets.append if resets is not None else None)

    def load(device):
        return models.pop(0)

    monkeypatch.setattr(instance, "_load", load)
    return instance


def test_gpu_failure_mid_session_restarts_on_cpu_with_a_new_session(gpu_machine, monkeypatch):
    gpu = FakeModel("gpu-model.pt", fail_track_on="cuda:0")
    cpu = FakeModel("cpu-model.pt")
    resets = []
    yolo = tracker(monkeypatch, [gpu, cpu], resets)
    yolo.load()
    assert yolo.device == "cuda:0" and yolo.model_path == "gpu-model.pt"

    objects, _ = yolo.detect_with_masks(frame(), "camera-1")
    assert [item.track_id for item in objects] == [1]
    assert yolo.device == "cpu" and yolo.model_path == "cpu-model.pt" and yolo.restarts == 1
    assert resets == ["camera-1"]
    assert ("predict", "cpu") in cpu.devices

    yolo.detect_with_masks(frame(), "camera-1")
    assert all(device == "cpu" for _, device in cpu.devices)
    assert yolo.restarts == 1
    status = device_manager.report()["components"]["yolo"]
    assert status["active"] == "cpu" and status["fallback_from"] == "cuda:0"


def test_gpu_that_fails_its_warm_up_never_tracks_on_gpu(gpu_machine, monkeypatch):
    class BrokenGpu(FakeModel):
        def predict(self, *, device, **_):
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    cpu = FakeModel("cpu-model.pt")
    yolo = tracker(monkeypatch, [BrokenGpu("gpu-model.pt"), cpu])
    yolo.detect_with_masks(frame(), "camera-1")
    assert yolo.device == "cpu" and yolo.restarts == 0
    assert [device for kind, device in cpu.devices if kind == "track"] == ["cpu"]


def test_a_new_camera_session_does_not_inherit_tracks(gpu_machine, monkeypatch):
    model = FakeModel("gpu-model.pt")
    resets = []
    yolo = tracker(monkeypatch, [model], resets)
    first, _ = yolo.detect_with_masks(frame(), "camera-1")
    again, _ = yolo.detect_with_masks(frame(), "camera-1")
    other, _ = yolo.detect_with_masks(frame(), "camera-2")
    assert first[0].track_id == again[0].track_id
    assert other[0].track_id != first[0].track_id
    assert resets == ["camera-1"]


def test_non_accelerator_errors_do_not_switch_devices(gpu_machine, monkeypatch):
    class Broken(FakeModel):
        def track(self, **_):
            raise RuntimeError("shape mismatch")

    yolo = tracker(monkeypatch, [Broken("gpu-model.pt")])
    assert yolo.detect_with_masks(frame(), "camera-1") == ([], None)
    assert yolo.device == "cuda:0" and yolo.restarts == 0
