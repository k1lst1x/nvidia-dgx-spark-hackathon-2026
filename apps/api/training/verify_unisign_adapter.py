#!/usr/bin/env python3
"""Validate Uni-Sign pose inputs without loading an unverified checkpoint.

This verifies topology, normalized ranges, timestamps and confidence provenance
for an exported VisionMate recording. It deliberately refuses to claim that a
MediaPipe mapping is checkpoint-compatible until it has been benchmarked
against the official RTM WholeBody sample pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unisign_adapter import UniSignAdapter, UniSignCompatibilityError, UNISIGN_PART_SIZES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--allow-presence-proxy", action="store_true")
    args = parser.parse_args()
    record = json.loads(args.recording.read_text(encoding="utf-8"))
    frames = record.get("raw_holistic")
    if not isinstance(frames, list):
        raise SystemExit("Recording has no raw_holistic frames. Capture a new sample with current VisionMate.")
    policy = "presence_proxy" if args.allow_presence_proxy else "reject"
    try:
        pose = UniSignAdapter(policy).adapt(frames)
    except UniSignCompatibilityError as error:
        raise SystemExit(f"Not checkpoint-compatible: {error}") from error
    for name, points in pose.as_model_input().items():
        assert points.shape[-2:] == (UNISIGN_PART_SIZES[name], 3)
        assert points.shape[0] == 1
    print(json.dumps({
        "frames": len(frames),
        "shapes": {name: list(value.shape) for name, value in pose.as_model_input().items()},
        "confidence_policy": pose.confidence_policy,
        "topology_verified": pose.topology_verified,
        "timestamp_monotonic": bool((pose.timestamps_ms[1:] >= pose.timestamps_ms[:-1]).all()),
        "warning": "presence_proxy is not RTM detector confidence and must not be used for accuracy claims",
    }, indent=2))


if __name__ == "__main__":
    main()
