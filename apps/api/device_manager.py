"""Choose the compute device for each model: CUDA first, then Apple MPS, then CPU.

Each component reads its own variable, so models that would not fit on one GPU
together can be split (for example YOLO on CUDA, depth on CPU):

  YOLO_DEVICE   detector + BoT-SORT    (legacy name: ULTRALYTICS_YOLO_DEVICE)
  SEG_DEVICE    separate segmenter     (legacy name: ULTRALYTICS_YOLO_DEVICE)
  DEPTH_DEVICE  Depth Anything V2      (legacy name: DEPTH_ANYTHING_DEVICE)
  SIGN_DEVICE   server sign models     (legacy name: UNISIGN_DEVICE)

Values: auto, cpu, mps, cuda (= cuda:0), cuda:N. Availability is decided by
PyTorch, never by the presence of a graphics card: a CPU-only PyTorch build on
a machine with an NVIDIA GPU still selects CPU, and `hardware()` says why.

FP16 is off by default. YOLO_HALF / SEG_HALF / DEPTH_HALF / SIGN_HALF=true
enable it, on CUDA only.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, TypeVar

logger = logging.getLogger("guidelens.devices")

T = TypeVar("T")

COMPONENTS: dict[str, tuple[str, str | None]] = {
    "yolo": ("YOLO_DEVICE", "ULTRALYTICS_YOLO_DEVICE"),
    "segmentation": ("SEG_DEVICE", "ULTRALYTICS_YOLO_DEVICE"),
    "depth": ("DEPTH_DEVICE", "DEPTH_ANYTHING_DEVICE"),
    "sign": ("SIGN_DEVICE", "UNISIGN_DEVICE"),
}
_HALF_ENV = {"yolo": "YOLO_HALF", "segmentation": "SEG_HALF", "depth": "DEPTH_HALF", "sign": "SIGN_HALF"}
_ACCELERATOR_ERROR = re.compile(r"cuda|cudnn|cublas|nvrtc|mps|metal|device-side|out of memory|no kernel image", re.I)

_report_lock = threading.Lock()
_report: dict[str, dict[str, object]] = {}


@dataclass(frozen=True)
class DeviceChoice:
    component: str
    requested: str
    device: str
    reason: str


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _mps_available(torch) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend and backend.is_available())


def _nvidia_smi() -> dict[str, object] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return {"error": "nvidia-smi failed"}
    gpus = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory_mb": parts[2]})
    return {"gpus": gpus}


@lru_cache(maxsize=1)
def hardware() -> dict[str, object]:
    """What PyTorch can actually use on this machine, and what stands in the way."""
    torch = _torch()
    smi = _nvidia_smi()
    nvidia_seen = bool(smi and smi.get("gpus"))
    info: dict[str, object] = {"torch": None, "torch_cuda_build": None, "cuda_available": False,
                               "cuda_devices": [], "mps_available": False, "nvidia_smi": smi, "problems": []}
    problems: list[str] = info["problems"]  # type: ignore[assignment]
    if torch is None:
        problems.append("PyTorch is not installed; every model runs without an accelerator.")
        return info
    info["torch"] = torch.__version__
    info["torch_cuda_build"] = torch.version.cuda
    info["mps_available"] = _mps_available(torch)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # a broken driver can raise here instead of returning False
        problems.append(f"torch.cuda.is_available() failed: {exc}")
    if info["cuda_available"]:
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": properties.name,
                            "memory_mb": properties.total_memory // (1024 * 1024),
                            "capability": f"{properties.major}.{properties.minor}"})
        info["cuda_devices"] = devices
    elif nvidia_seen and torch.version.cuda is None:
        problems.append(
            f"nvidia-smi sees an NVIDIA GPU but PyTorch {torch.__version__} is a CPU-only build. "
            "Install a CUDA build from https://pytorch.org/get-started/locally/ (docs/DEVICES.md)."
        )
    elif nvidia_seen:
        problems.append(
            f"PyTorch is built for CUDA {torch.version.cuda} but cannot use the GPU. The NVIDIA driver may be "
            "too old for that CUDA version, or CUDA_VISIBLE_DEVICES hides the GPU."
        )
    return info


def _normalize(value: str) -> str:
    value = (value or "auto").strip().lower()
    if value in {"", "gpu"}:
        return "auto"
    if value == "cuda" or value.isdigit():
        return f"cuda:{value if value.isdigit() else 0}"
    return value


def _auto(info: dict[str, object]) -> tuple[str, str]:
    if info["cuda_available"]:
        return "cuda:0", "CUDA is available"
    if info["mps_available"]:
        return "mps", "Apple MPS is available"
    problems = info["problems"]
    return "cpu", str(problems[0]) if problems else "no CUDA or MPS device"


def requested_device(component: str) -> str:
    name, legacy = COMPONENTS[component]
    return os.getenv(name) or (os.getenv(legacy) if legacy else None) or "auto"


def select_device(component: str, requested: str | None = None) -> DeviceChoice:
    """Resolve a component's setting to a device PyTorch can use now."""
    wanted = _normalize(requested if requested is not None else requested_device(component))
    info = hardware()
    if wanted == "auto":
        device, reason = _auto(info)
    elif wanted == "cpu":
        device, reason = "cpu", "requested"
    elif wanted == "mps":
        device, reason = ("mps", "requested") if info["mps_available"] else ("cpu", "MPS requested but unavailable")
    elif re.fullmatch(r"cuda:\d+", wanted):
        index = int(wanted.split(":")[1])
        count = len(info["cuda_devices"])  # type: ignore[arg-type]
        if index < count:
            device, reason = wanted, "requested"
        else:
            detail = info["problems"][0] if info["problems"] else f"{count} CUDA device(s) visible"  # type: ignore[index]
            device, reason = "cpu", f"{wanted} requested but unavailable: {detail}"
    else:
        logger.warning("unknown_device_setting component=%s value=%s using=auto", component, wanted)
        device, reason = _auto(info)
        reason = f"unknown setting {wanted!r}; {reason}"
    if device == "cpu" and wanted not in {"cpu", "auto"}:
        logger.warning("device_unavailable component=%s requested=%s reason=%s", component, wanted, reason)
    return DeviceChoice(component, wanted, device, reason)


def half_enabled(component: str, device: str) -> bool:
    return device.startswith("cuda") and os.getenv(_HALF_ENV[component], "false").lower() in {"1", "true", "yes"}


def is_accelerator_error(exc: BaseException) -> bool:
    torch = _torch()
    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch else None
    accelerator = getattr(torch, "AcceleratorError", None) if torch else None
    if any(kind and isinstance(exc, kind) for kind in (oom, accelerator)):
        return True
    return isinstance(exc, RuntimeError) and bool(_ACCELERATOR_ERROR.search(str(exc)))


def release(device: str) -> None:
    """Free cached accelerator memory after a model on `device` was dropped."""
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and _mps_available(torch):
            torch.mps.empty_cache()
    except Exception:
        logger.debug("cache_release_failed device=%s", device, exc_info=True)


def record(component: str, choice: DeviceChoice, device: str, **details: object) -> None:
    """Remember what a component runs on, for /health."""
    with _report_lock:
        _report[component] = {"requested": choice.requested, "selected": choice.device, "active": device,
                              "reason": choice.reason, **details}


def report() -> dict[str, object]:
    info = hardware()
    with _report_lock:
        components = {name: dict(value) for name, value in _report.items()}
    return {
        "torch": info["torch"], "torch_cuda_build": info["torch_cuda_build"],
        "cuda_available": info["cuda_available"], "cuda_devices": info["cuda_devices"],
        "mps_available": info["mps_available"], "problems": info["problems"], "components": components,
    }


def load_with_fallback(
    choice: DeviceChoice, load: Callable[[str], T], warm_up: Callable[[T, str], None], **details: object,
) -> tuple[T, str]:
    """Load and test-run a model on the chosen device; on failure, start over on CPU.

    The CPU attempt gets a fresh model, so no half-initialised accelerator
    state is carried over. A CPU failure is raised: there is nothing left.
    """
    device = choice.device
    try:
        model = load(device)
        warm_up(model, device)
    except Exception as exc:
        if device == "cpu":
            raise
        logger.exception("accelerator_init_failed component=%s device=%s; retrying on cpu", choice.component, device)
        model = None
        release(device)
        model = load("cpu")
        warm_up(model, "cpu")
        record(choice.component, choice, "cpu", fallback_from=device, error=f"{type(exc).__name__}: {exc}"[:300],
               **details)
        return model, "cpu"
    record(choice.component, choice, device, **details)
    return model, device


if __name__ == "__main__":
    import json

    print(json.dumps({"hardware": hardware(),
                      "selected": {name: select_device(name).__dict__ for name in COMPONENTS}}, indent=2))
