#!/usr/bin/env python3
"""Create a validated JSONL ASL-30 manifest from SignRecorder JSON exports.

This command does not download data, train a model, or change the production
recognizer. It is a reproducible gate that prevents unsupported labels,
multi-word recordings, 390-feature frames, and malformed timing from leaking
into the separate ASL-30 experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import CaptureValidationError, validate_capture


def capture_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*.json") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="SignRecorder JSON file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Validated JSONL output path")
    args = parser.parse_args()

    paths = capture_paths(args.input)
    if not paths:
        raise SystemExit(f"No JSON recordings found at {args.input}")
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise CaptureValidationError("recording must be a JSON object")
            sample = validate_capture(payload)
            if sample.sample_id in seen_ids:
                raise CaptureValidationError(f"duplicate id {sample.sample_id!r}")
            seen_ids.add(sample.sample_id)
            rows.append(sample.as_manifest_row())
        except (OSError, json.JSONDecodeError, CaptureValidationError) as error:
            errors.append(f"{path}: {error}")

    if errors:
        raise SystemExit("Manifest not written; fix every invalid capture:\n" + "\n".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} validated ASL-30 recordings to {args.output}")


if __name__ == "__main__":
    main()
