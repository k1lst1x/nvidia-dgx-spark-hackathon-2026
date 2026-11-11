"""GPU load for the dashboard, read from nvidia-smi on the DGX Spark.

Returns {"available": False} on machines without an NVIDIA GPU (a laptop,
CI), so callers never have to guess. Results are cached for a few seconds:
the dashboard polls, and nvidia-smi is not free.
"""

from __future__ import annotations

import shutil
import subprocess
import time

_CACHE_SECONDS = 3.0
_cache: tuple[float, dict[str, object]] | None = None
_FIELDS = ("name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu")


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # e.g. "[N/A]" on unified-memory systems


def parse(output: str) -> dict[str, object]:
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(_FIELDS):
            continue
        name, utilization, used, total, temperature = parts
        gpus.append({
            "name": name,
            "utilization_pct": _number(utilization),
            "memory_used_mb": _number(used),
            "memory_total_mb": _number(total),
            "temperature_c": _number(temperature),
        })
    return {"available": bool(gpus), "gpus": gpus}


def gpu_status(now: float | None = None) -> dict[str, object]:
    global _cache
    current = now or time.monotonic()
    if _cache and current - _cache[0] < _CACHE_SECONDS:
        return _cache[1]
    result: dict[str, object] = {"available": False, "gpus": []}
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=True,
            )
            result = parse(completed.stdout)
        except (subprocess.SubprocessError, OSError):
            result = {"available": False, "gpus": [], "error": "nvidia-smi failed"}
    _cache = (current, result)
    return result
