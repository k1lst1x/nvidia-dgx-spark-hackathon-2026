"""Evaluate VisionMate tracking on annotated moving-camera recordings.

Usage:
  PYTHONPATH=apps/api python apps/api/benchmarks/benchmark_motion.py --manifest clips.jsonl

Each JSONL row represents one annotated frame with a scenario, timestamp,
``ground_truth`` entries (object_id, depth_m, trend) and optional predictions
from a capture run.  The script refuses to invent metrics when a field or a
required scenario is absent; use it after exporting real recordings.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_SCENARIOS = {
    "stationary_objects",
    "approaching_pedestrians",
    "crossing_objects",
    "camera_turns",
    "camera_shake",
}


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percent / 100)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    scenarios = {row.get("scenario") for row in rows}
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    frame_times = [float(row["frame_time_ms"]) for row in rows if isinstance(row.get("frame_time_ms"), (int, float))]
    depth_errors: list[float] = []
    trend_hits = 0
    trend_total = 0
    ids_by_truth: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        predictions = {str(item.get("ground_truth_id")): item for item in row.get("predictions", [])}
        for truth in row.get("ground_truth", []):
            predicted = predictions.get(str(truth.get("id")))
            if not predicted:
                continue
            if isinstance(truth.get("depth_m"), (int, float)) and isinstance(predicted.get("distance_m"), (int, float)):
                depth_errors.append(abs(float(truth["depth_m"]) - float(predicted["distance_m"])))
            if truth.get("trend") in {"approaching", "receding", "stable"} and predicted.get("trend"):
                trend_total += 1
                trend_hits += int(truth["trend"] == predicted["trend"])
            if isinstance(predicted.get("track_id"), int):
                ids_by_truth[str(truth.get("id"))].append(predicted["track_id"])
    id_switches = sum(sum(left != right for left, right in zip(ids, ids[1:])) for ids in ids_by_truth.values())
    result = {
        "frames": len(rows),
        "scenarios_present": sorted(scenarios - {None}),
        "missing_required_scenarios": sorted(REQUIRED_SCENARIOS - scenarios),
        "tracking_latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "pipeline_fps": round(1000 / statistics.mean(frame_times), 2) if frame_times else None,
        "depth_mae_m": round(statistics.mean(depth_errors), 3) if depth_errors else None,
        "trend_accuracy": round(trend_hits / trend_total, 3) if trend_total else None,
        "id_switches": id_switches if ids_by_truth else None,
        "notes": "null means the recording manifest did not contain ground truth or timing data; no metric was estimated.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
