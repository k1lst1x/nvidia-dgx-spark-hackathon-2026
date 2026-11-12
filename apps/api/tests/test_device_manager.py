"""Device selection and CPU fallback, with PyTorch replaced by a fake."""

from types import SimpleNamespace

import pytest

import device_manager


def fake_torch(*, cuda_build="13.0", cuda=False, devices=1, mps=False):
    properties = SimpleNamespace(name="Test GPU", total_memory=8 * 1024**3, major=8, minor=9)
    return SimpleNamespace(
        __version__="2.14.1",
        version=SimpleNamespace(cuda=cuda_build),
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: devices if cuda else 0,
                             get_device_properties=lambda index: properties, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.fixture
def machine(monkeypatch):
    for name in ("YOLO_DEVICE", "ULTRALYTICS_YOLO_DEVICE", "SEG_DEVICE", "DEPTH_DEVICE", "DEPTH_ANYTHING_DEVICE",
                 "SIGN_DEVICE", "UNISIGN_DEVICE", "YOLO_HALF"):
        monkeypatch.delenv(name, raising=False)

    def use(torch=None, gpus=None):
        monkeypatch.setattr(device_manager, "_torch", lambda: torch)
        monkeypatch.setattr(device_manager, "_nvidia_smi", lambda: {"gpus": gpus} if gpus is not None else None)
        device_manager.hardware.cache_clear()

    yield use
    device_manager.hardware.cache_clear()


def test_auto_prefers_cuda_then_mps_then_cpu(machine):
    machine(fake_torch(cuda=True, mps=True))
    assert device_manager.select_device("yolo").device == "cuda:0"
    machine(fake_torch(cuda_build=None, mps=True))
    assert device_manager.select_device("yolo").device == "mps"
    machine(fake_torch(cuda_build=None))
    assert device_manager.select_device("yolo").device == "cpu"


def test_nvidia_card_with_cpu_only_pytorch_selects_cpu_and_says_why(machine):
    machine(fake_torch(cuda_build=None), gpus=[{"name": "RTX 4050"}])
    choice = device_manager.select_device("yolo")
    assert choice.device == "cpu"
    assert "CPU-only build" in choice.reason
    assert device_manager.hardware()["problems"]


def test_cuda_build_that_cannot_reach_the_gpu_reports_driver_problem(machine):
    machine(fake_torch(cuda=False), gpus=[{"name": "RTX 4050"}])
    assert "driver" in device_manager.select_device("yolo").reason


def test_requested_devices_that_do_not_exist_fall_back_to_cpu(machine):
    machine(fake_torch(cuda=True, devices=1))
    assert device_manager.select_device("yolo", "cuda:1").device == "cpu"
    assert device_manager.select_device("yolo", "cuda").device == "cuda:0"
    assert device_manager.select_device("yolo", "0").device == "cuda:0"
    assert device_manager.select_device("yolo", "mps").device == "cpu"
    assert device_manager.select_device("yolo", "cpu").device == "cpu"


def test_no_pytorch_means_cpu(machine):
    machine(None)
    assert device_manager.select_device("depth").device == "cpu"


def test_each_component_reads_its_own_variable_and_legacy_names_still_work(machine, monkeypatch):
    machine(fake_torch(cuda=True, devices=2))
    monkeypatch.setenv("ULTRALYTICS_YOLO_DEVICE", "cpu")
    monkeypatch.setenv("DEPTH_DEVICE", "cuda:1")
    assert device_manager.select_device("yolo").device == "cpu"
    assert device_manager.select_device("depth").device == "cuda:1"
    assert device_manager.select_device("sign").device == "cuda:0"
    monkeypatch.setenv("YOLO_DEVICE", "auto")
    assert device_manager.select_device("yolo").device == "cuda:0"


def test_fp16_is_opt_in_and_cuda_only(monkeypatch):
    monkeypatch.delenv("YOLO_HALF", raising=False)
    assert not device_manager.half_enabled("yolo", "cuda:0")
    monkeypatch.setenv("YOLO_HALF", "true")
    assert device_manager.half_enabled("yolo", "cuda:0")
    assert not device_manager.half_enabled("yolo", "mps")
    assert not device_manager.half_enabled("yolo", "cpu")


def test_accelerator_failure_reloads_a_fresh_model_on_cpu(machine):
    machine(fake_torch(cuda=True))
    loads = []

    def load(device):
        loads.append(device)
        return {"device": device}

    def warm(model, device):
        if device != "cpu":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    model, device = device_manager.load_with_fallback(device_manager.select_device("yolo"), load, warm)
    assert (model, device, loads) == ({"device": "cpu"}, "cpu", ["cuda:0", "cpu"])
    status = device_manager.report()["components"]["yolo"]
    assert status["active"] == "cpu" and status["fallback_from"] == "cuda:0"


def test_cpu_failure_is_raised(machine):
    machine(fake_torch(cuda_build=None))

    def warm(model, device):
        raise ValueError("broken weights")

    with pytest.raises(ValueError):
        device_manager.load_with_fallback(device_manager.select_device("yolo"), lambda device: object(), warm)


def test_accelerator_errors_are_recognized():
    assert device_manager.is_accelerator_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert device_manager.is_accelerator_error(RuntimeError("MPS backend out of memory"))
    assert not device_manager.is_accelerator_error(RuntimeError("shape mismatch"))
    assert not device_manager.is_accelerator_error(ValueError("CUDA"))
