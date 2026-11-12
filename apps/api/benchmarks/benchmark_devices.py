"""CPU vs CUDA vs MPS: YOLO + BoT-SORT and Depth Anything V2 latency, memory and FP16 agreement.

Runs the API's own YoloTracker and DepthAnythingMetric, so each number is what
one /v1/track frame costs in that component (JPEG decode included). Frames are
a real photo panned a few pixels per frame.

  python benchmarks/benchmark_devices.py --models models/yolo11n.pt models/yolo11n-seg.pt models/yolo11m-seg.pt
  python benchmarks/benchmark_devices.py --depth --output benchmarks/results/devices.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))

import device_manager  # noqa: E402


def frames(count: int) -> list[bytes]:
    from PIL import Image
    from ultralytics.utils import ASSETS

    photo = Image.open(ASSETS / "bus.jpg").convert("RGB")
    out = []
    for step in range(count):
        shift = step % 25
        crop = photo.crop((shift * 6, shift * 4, shift * 6 + 640, shift * 4 + 640))
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
        out.append(buffer.getvalue())
    return out


def available_devices() -> list[str]:
    info = device_manager.hardware()
    devices = ["cpu"]
    devices += [f"cuda:{item['index']}" for item in info["cuda_devices"]]
    if info["mps_available"]:
        devices.append("mps")
    return devices


def memory(device: str) -> dict[str, float | None]:
    import torch

    rss = None
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        pass
    vram = None
    if device.startswith("cuda"):
        vram = torch.cuda.max_memory_allocated(device) / 1024**2
    elif device == "mps":
        vram = torch.mps.current_allocated_memory() / 1024**2
    return {"ram_mb": round(rss, 0) if rss else None, "accelerator_mb": round(vram, 0) if vram is not None else None}


def reset_peak(device: str) -> None:
    import torch

    device_manager.release(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    median = statistics.median(ordered)
    return {"p50_ms": round(median, 1), "p95_ms": round(p95, 1), "fps": round(1000 / median, 1)}


def iou(a, b) -> float:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.width * a.height + b.width * b.height - overlap
    return overlap / union if union else 0.0


def agreement(reference: list[list], other: list[list]) -> dict[str, float]:
    """Share of FP32 boxes that FP16 also finds (same class, IoU >= 0.7), and the largest confidence change."""
    matched = total = 0
    confidence_delta = 0.0
    for ref_frame, other_frame in zip(reference, other):
        for box in ref_frame:
            total += 1
            best = max((o for o in other_frame if o.label == box.label), key=lambda o: iou(box, o), default=None)
            if best is not None and iou(box, best) >= 0.7:
                matched += 1
                confidence_delta = max(confidence_delta, abs(best.confidence - box.confidence))
    return {"boxes_matched": round(matched / total, 3) if total else 1.0, "max_confidence_delta": round(confidence_delta, 3),
            "fp32_boxes": total, "fp16_boxes": sum(len(f) for f in other)}


def bench_yolo(model: str, device: str, half: bool, images: list[bytes], warmup: int) -> dict[str, object]:
    from yolo_tracker import YoloTracker

    os.environ["YOLO_HALF"] = "true" if half else "false"
    reset_peak(device)
    tracker = YoloTracker(enabled=True, model_path=model, device=device, confidence=0.4,
                          tracker_config=str(API / "botsort.yaml"), idle_seconds=600)
    started = time.perf_counter()
    tracker.load()
    load_s = time.perf_counter() - started
    if tracker.device != device:
        return {"error": f"fell back to {tracker.device}", "devices": device_manager.report()["components"].get("yolo")}
    for image in images[:warmup]:
        tracker.detect_with_masks(image, "warmup")
    latencies, detections, ids = [], [], []
    for image in images:
        sync(device)
        t0 = time.perf_counter()
        objects, _ = tracker.detect_with_masks(image, "benchmark")
        sync(device)
        latencies.append((time.perf_counter() - t0) * 1000)
        detections.append(objects)
        ids.append({item.track_id for item in objects if item.track_id is not None})
    stable = len(set.intersection(*ids[1:])) if len(ids) > 1 else 0
    result = {"model": Path(model).name, "device": device, "precision": "fp16" if half else "fp32",
              "load_s": round(load_s, 2), **summary(latencies), **memory(device),
              "objects_per_frame": round(statistics.mean(len(d) for d in detections), 1),
              "ids_kept_all_frames": stable, "_detections": detections}
    del tracker
    device_manager.release(device)
    return result


def bench_depth(device: str, half: bool, images: list[bytes], warmup: int) -> dict[str, object]:
    import numpy as np
    from depth import DepthAnythingMetric

    os.environ.update({"DEPTH_ANYTHING_ENABLED": "true", "DEPTH_DEVICE": device, "DEPTH_HALF": "true" if half else "false"})
    reset_peak(device)
    model = DepthAnythingMetric()
    if not model.ready:
        return {"error": "Depth Anything weights or code missing (DEPTH_ANYTHING_REPO / DEPTH_ANYTHING_CHECKPOINT)"}
    model.infer(images[0])
    if model.status()["device"] != device:
        return {"error": f"fell back to {model.status()['device']}"}
    for image in images[:warmup]:
        model.infer(image)
    latencies, maps = [], []
    for image in images:
        sync(device)
        t0 = time.perf_counter()
        maps.append(model.infer(image))
        sync(device)
        latencies.append((time.perf_counter() - t0) * 1000)
    result = {"model": model.checkpoint.name, "device": device, "precision": "fp16" if half else "fp32",
              "input_size": model.input_size, **summary(latencies), **memory(device),
              "_maps": [np.asarray(m) for m in maps]}
    del model
    device_manager.release(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[str(API / "models" / "yolo11n.pt")])
    parser.add_argument("--devices", nargs="+", help="default: every device PyTorch can use here")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true", help="skip the FP16 comparison on CUDA")
    parser.add_argument("--depth", action="store_true", help="also benchmark Depth Anything V2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np

    hardware = device_manager.hardware()
    devices = args.devices or available_devices()
    images = frames(args.frames)
    print(f"torch {hardware['torch']} (CUDA build {hardware['torch_cuda_build']}), devices: {', '.join(devices)}")
    for problem in hardware["problems"]:
        print("note:", problem)

    rows: list[dict[str, object]] = []
    for model in args.models:
        for device in devices:
            precisions = [False] + ([True] if device.startswith("cuda") and not args.no_fp16 else [])
            fp32 = None
            for half in precisions:
                row = bench_yolo(model, device, half, images, args.warmup)
                detections = row.pop("_detections", None)
                if half and fp32 is not None and detections is not None:
                    row["fp16_vs_fp32"] = agreement(fp32, detections)
                elif not half:
                    fp32 = detections
                rows.append({"component": "yolo+botsort", **row})
                print(json.dumps(rows[-1]))
    if args.depth:
        for device in devices:
            reference = None
            for half in [False] + ([True] if device.startswith("cuda") and not args.no_fp16 else []):
                row = bench_depth(device, half, images[: max(10, args.frames // 3)], args.warmup)
                maps = row.pop("_maps", None)
                if half and reference is not None and maps is not None:
                    relative = [np.median(np.abs(a - b) / np.maximum(a, 1e-3)) for a, b in zip(reference, maps)]
                    row["fp16_vs_fp32_median_relative_diff"] = round(float(np.median(relative)), 4)
                elif not half:
                    reference = maps
                rows.append({"component": "depth", **row})
                print(json.dumps(rows[-1]))

    print("\n| component | model | device | precision | p50 ms | p95 ms | FPS | RAM MB | GPU MB | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        if "error" in row:
            print(f"| {row['component']} | | | | | | | | | {row['error']} |")
            continue
        notes = row.get("fp16_vs_fp32") or row.get("fp16_vs_fp32_median_relative_diff") or ""
        if "ids_kept_all_frames" in row:
            notes = f"IDs kept: {row['ids_kept_all_frames']}" + (f"; FP16 vs FP32: {notes}" if notes else "")
        print(f"| {row['component']} | {row['model']} | {row['device']} | {row['precision']} | {row['p50_ms']} | "
              f"{row['p95_ms']} | {row['fps']} | {row['ram_mb']} | {row['accelerator_mb']} | {notes} |")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"machine": platform.platform(), "python": platform.python_version(),
                                           "hardware": hardware, "results": rows}, indent=2, default=str),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
