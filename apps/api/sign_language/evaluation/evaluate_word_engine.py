"""Score the on-device 250-word sign model on samples saved by the /record screen.

Each sample is one JSON file from SignRecorder with `raw_holistic` frames and
verified glosses. A sample whose gloss is NONE is a clip of movement without
signing; any word confirmed on it is a false word.

The 390-value frames are rebuilt from the raw landmarks exactly as
createMultimodalFrame in HandTracker.web.tsx does, and the ASLWordEngine loop
in aslWordClassifier.web.ts is replayed: motion gate, 64-frame buffer, a
window every 3rd frame from 12 frames, the vote rule in signConfirmation.ts,
and the commit cooldowns. The rule-based gestures are not replayed, so this
scores the model and its confirmation only.

Recordings are sampled at 20 fps; the live engine sees every camera frame.

Usage:
  python evaluate_word_engine.py --samples path/to/recordings \
      --model apps/mobile/assets/models/asl_word_transformer.onnx \
      --labels apps/mobile/assets/models/asl_word_labels.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

FACE_INDICES = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    1, 2, 98, 327, 4, 5, 195, 197, 33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 46, 53, 52, 65, 55, 276, 283, 282, 295, 285,
)
POSE_INDICES = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27)
HAND_INDICES = tuple(range(21))
NO_SIGN = "NONE"

# aslWordClassifier.web.ts and signConfirmation.ts
SEQ_LEN = 64
MIN_FRAMES = 12
EVERY_N = 3
MOTION_GATE = 0.003
WORD_CONFIDENCE_FLOOR = 0.3
WORD_CONFIDENT = 0.45
COMMIT_COOLDOWN_MS = 1200
REPEAT_COOLDOWN_MS = 2500


def _flatten(points, indices) -> list[float]:
    out = [0.0] * (len(indices) * 3)
    if not points:
        return out
    for slot, source in enumerate(indices):
        if source < len(points) and points[source]:
            point = points[source]
            out[slot * 3: slot * 3 + 3] = [point["x"], point["y"], point.get("z", 0.0)]
    return out


def frame_390(raw: dict) -> list[float]:
    return (_flatten(raw.get("left_hand"), HAND_INDICES) + _flatten(raw.get("right_hand"), HAND_INDICES)
            + _flatten(raw.get("face"), FACE_INDICES) + _flatten(raw.get("pose"), POSE_INDICES))


def model_label(gloss: str) -> str:
    return re.sub(r"[^a-z0-9]", "", gloss.lower())


def vote(top: list[tuple[str, float]], margin: float, allowed: set[str] | None = None) -> str:
    best = top[0]
    if best[1] < WORD_CONFIDENCE_FLOOR:
        return ""
    if len(top) > 1 and best[1] - top[1][1] < margin:
        return ""
    if allowed is not None and best[0] not in allowed:
        return ""
    return best[0]


def replay(windows: list[tuple[float, list[tuple[str, float]]]], margin: float,
           allowed: set[str] | None = None) -> list[str]:
    """Words the engine confirms from a clip's inference windows (time ms, top 3)."""
    committed: list[str] = []
    candidate, stable, last, last_time = "", 0, "", -math.inf
    for now, top in windows:
        if vote(top, margin, allowed):
            label, confidence = top[0]
            stable = stable + 1 if label == candidate else 1
            candidate = label
            if stable >= (2 if confidence >= WORD_CONFIDENT else 3) and (
                    label != last or now - last_time > COMMIT_COOLDOWN_MS):
                if not (label == last and now - last_time <= REPEAT_COOLDOWN_MS):
                    committed.append(label)
                last, last_time, stable = label, now, 0
        else:
            stable = max(0, stable - 1)
            if stable == 0:
                candidate = ""
    return committed


def windows_for(sample: dict, session, classes: list[str]):
    import numpy as np

    raw_frames = sample.get("raw_holistic") or []
    buffer: list[list[float]] = []
    counter, last_wrist, windows = 0, None, []
    for raw in raw_frames:
        hand = raw.get("left_hand") or raw.get("right_hand")
        if not hand:
            continue
        wrist = hand[0]
        motion = math.dist((wrist["x"], wrist["y"]), last_wrist) if last_wrist else 0.05
        last_wrist = (wrist["x"], wrist["y"])
        if motion < MOTION_GATE and len(buffer) >= 16:
            continue
        buffer.append(frame_390(raw))
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)
        counter += 1
        if len(buffer) < MIN_FRAMES or counter % EVERY_N:
            continue
        data = np.zeros((1, SEQ_LEN, 390), dtype=np.float32)
        data[0, :len(buffer)] = buffer
        logits = session.run(None, {"landmarks": data})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities)[:3]
        windows.append((float(raw["timestamp_ms"]), [(classes[i], float(probabilities[i])) for i in order]))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True, help="folder of /record JSON files")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--output", type=Path, help="write per-sample results as JSON")
    parser.add_argument("--core-vocabulary", type=Path,
                        help="asl_core_vocabulary.json: confirm only these signs, as the app does by default")
    args = parser.parse_args()

    import onnxruntime as ort

    classes = json.loads(args.labels.read_text(encoding="utf-8"))["classes"]
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    allowed = (set(json.loads(args.core_vocabulary.read_text(encoding="utf-8"))["signs"])
               if args.core_vocabulary else None)

    results, skipped = [], []
    for path in sorted(args.samples.glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        glosses = sample.get("glosses") or []
        if len(glosses) != 1 or not sample.get("raw_holistic"):
            skipped.append(path.name)
            continue
        expected = NO_SIGN if glosses[0] == NO_SIGN else model_label(glosses[0])
        if expected != NO_SIGN and expected not in classes:
            skipped.append(f"{path.name} ({glosses[0]} is not in the model's vocabulary)")
            continue
        windows = windows_for(sample, session, classes)
        best_window = max(windows, key=lambda item: item[1][0][1])[1][0][0] if windows else ""
        results.append({
            "file": path.name, "signer": sample.get("signerId", ""), "expected": expected,
            "windows": len(windows), "best_window_top1": best_window,
            "committed": {str(margin): replay(windows, margin, allowed) for margin in args.margins},
        })

    signs = [r for r in results if r["expected"] != NO_SIGN]
    none = [r for r in results if r["expected"] == NO_SIGN]
    print(f"{len(signs)} sign clips, {len(none)} no-sign clips, "
          f"{len({r['signer'] for r in results})} signers, {len(skipped)} skipped")
    if signs:
        top1 = sum(r["best_window_top1"] == r["expected"] for r in signs) / len(signs)
        print(f"model top-1 on the clip's most confident window: {top1:.1%}")
    print("margin | correct word confirmed | wrong word confirmed | nothing confirmed | false words on no-sign clips")
    for margin in args.margins:
        key = str(margin)
        right = sum(r["expected"] in r["committed"][key] for r in signs)
        wrong = sum(any(w != r["expected"] for w in r["committed"][key]) for r in signs)
        silent = sum(not r["committed"][key] for r in signs)
        false_words = sum(len(r["committed"][key]) for r in none)
        n = max(1, len(signs))
        print(f"{margin:6.2f} | {right / n:22.1%} | {wrong / n:20.1%} | {silent / n:17.1%} | {false_words}")
    key = "0.1" if 0.1 in args.margins else str(args.margins[-1])
    confusions = Counter((r["expected"], w) for r in signs + none for w in r["committed"][key] if w != r["expected"])
    if confusions:
        print(f"most common wrong words at margin {key}:", confusions.most_common(10))
    for name in skipped:
        print("skipped:", name)
    if args.output:
        args.output.write_text(json.dumps({"results": results, "skipped": skipped}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
