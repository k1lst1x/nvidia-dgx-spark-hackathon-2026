"""Pick a small, verified set of How2Sign validation clips for model evaluation.

A clip is accepted only if:
- its file name equals SENTENCE_NAME in the original validation CSV (matched by
  identifier, never by position in a listing);
- the video opens and its duration is within 0.5 s of END - START;
- it is 1.3-8 s long with at most 5 words per second (the range the
  pretrained models were trained on);
- neither the clip nor its sentence appears in any Sign Language Bridge
  dataset file, nor in Uni-Sign's How2Sign training labels, so neither model
  has seen it or its reference text.

Usage:
  python select_how2sign_samples.py --csv how2sign_val.csv --clips raw_videos \
      --out datasets/how2sign --count 8
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import pickle
import random
import shutil
import urllib.request
from pathlib import Path

import cv2

BRIDGE_DATA = "https://raw.githubusercontent.com/mamounyosef/sign-language-bridge/main/data/"
BRIDGE_FILES = [f"{v}_dataset_{split}.tsv" for v in range(2, 8) for split in ("train", "val", "test")]
UNISIGN_LABELS = "https://raw.githubusercontent.com/ZechengLi19/Uni-Sign/main/data/How2Sign/labels.train"


class _PlainUnpickler(pickle.Unpickler):
    """Labels are plain dicts and strings; refuse anything that would run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _fetch(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "visionmate-eval"}), timeout=60).read()


def _normalize(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def seen_by_models() -> tuple[set[str], set[str]]:
    """Clip names and normalized sentences in either model's data."""
    names: set[str] = set()
    sentences: set[str] = set()
    for file in BRIDGE_FILES:
        for row in csv.reader(io.StringIO(_fetch(BRIDGE_DATA + file).decode("utf-8")), delimiter="\t"):
            if len(row) >= 2:
                names.add(row[0])
                sentences.add(_normalize(row[1]))
    labels = _PlainUnpickler(io.BytesIO(gzip.decompress(_fetch(UNISIGN_LABELS)))).load()
    for key, value in labels.items():
        names.add(str(key))
        if isinstance(value, dict):
            sentences.add(_normalize(str(value.get("text", ""))))
    return names, sentences


def video_info(path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return {
            "fps": round(fps, 3),
            "frames": int(frames),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": round(frames / fps, 3) if fps else 0.0,
        }
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    seen_names, seen_sentences = seen_by_models()

    candidates = []
    rejected = {"missing_clip": 0, "seen_by_a_model": 0, "length_or_rate": 0}
    for row in rows:
        name = row["SENTENCE_NAME"]
        path = args.clips / f"{name}.mp4"
        duration = float(row["END"]) - float(row["START"])
        words = len(row["SENTENCE"].split())
        if not path.exists():
            rejected["missing_clip"] += 1
        elif name in seen_names or _normalize(row["SENTENCE"]) in seen_sentences:
            rejected["seen_by_a_model"] += 1
        elif not (1.3 <= duration <= 8.0) or words / duration > 5.0:
            rejected["length_or_rate"] += 1
        else:
            candidates.append((row, path, duration))

    # One clip per source video, so the set spans signers and topics.
    random.Random(args.seed).shuffle(candidates)
    picked, used_videos, mismatched = [], set(), []
    for row, path, duration in candidates:
        if row["VIDEO_ID"] in used_videos:
            continue
        info = video_info(path)
        if not info or abs(info["duration_s"] - duration) > 0.5:
            mismatched.append(row["SENTENCE_NAME"])
            continue
        used_videos.add(row["VIDEO_ID"])
        picked.append((row, path, duration, info))
        if len(picked) == args.count:
            break

    rgb_dir = args.out / "samples" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "samples" / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "sentence_name", "video_path", "keypoints_path", "english_text",
                         "start_s", "end_s", "video_duration_s", "fps", "width", "height"])
        for index, (row, path, duration, info) in enumerate(picked, start=1):
            sample_id = f"sample_{index:03d}"
            shutil.copy2(path, rgb_dir / f"{sample_id}.mp4")
            writer.writerow([sample_id, row["SENTENCE_NAME"], f"samples/rgb/{sample_id}.mp4", "",
                             row["SENTENCE"], row["START"], row["END"], info["duration_s"],
                             info["fps"], info["width"], info["height"]])

    report = {
        "csv_rows": len(rows), "candidates": len(candidates), "rejected": rejected,
        "duration_mismatch_skipped": mismatched, "picked": len(picked),
        "seen_clip_names": len(seen_names), "seen_sentences": len(seen_sentences),
    }
    (args.out / "samples" / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
