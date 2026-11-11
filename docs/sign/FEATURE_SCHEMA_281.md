# Feature schema `visionmate-281-v1`

The 281-value frame built by `buildSignFeatureFrame` in
`apps/mobile/src/signFeatures.ts` (called from `HandTracker.web.tsx`), sampled by `SignRecorder`, and
required by `VisionMateTransformer` (`FEATURE_DIM = 281`).

**Status:** this is what the code produces. No model has been trained on it
yet, so nothing has validated it against recognition accuracy.

## Layout

| Index | Count | Content | Order |
|---|---|---|---|
| 0–62 | 63 | Left hand | 21 MediaPipe hand landmarks × (x, y, z) |
| 63–125 | 63 | Right hand | 21 landmarks × (x, y, z) |
| 126–224 | 99 | Pose | 33 MediaPipe pose landmarks × (x, y, z), indices 0–32 |
| 225–276 | 52 | Face blendshapes | scores 0–1, **alphabetical by category name** |
| 277 | 1 | Left hand present | 1 or 0 |
| 278 | 1 | Right hand present | 1 or 0 |
| 279 | 1 | Face present | 1 or 0 |
| 280 | 1 | Pose present | 1 or 0 |

Value at hand landmark `i`, axis `a` (0 = x, 1 = y, 2 = z):
left `i*3 + a`, right `63 + i*3 + a`. Pose landmark `j`: `126 + j*3 + a`.

## Conventions

- **Coordinates:** MediaPipe image-normalized. x and y are 0–1 across the
  video frame (x to the right, y down). z is MediaPipe's relative depth:
  wrist-relative for hands, hip-relative for pose. They are **not** world
  coordinates and **not** normalized to the body. Hand, pose and face z
  values are on different scales.
- **Missing parts:** all coordinates are 0 and the presence flag is 0. A
  present hand at exactly (0, 0, 0) cannot occur in practice, but models
  should use the flags, not the zeros.
- **Left / right:** from MediaPipe's handedness label. If the label is
  missing, the first hand goes to the right slot, the second to the left.
  The label describes the hand as MediaPipe sees the image, so a mirrored
  preview can swap it. **Not verified against a real signer.**
- **Blendshapes:** placed by name using the fixed list `BLENDSHAPE_ORDER`
  (the 52 MediaPipe names, alphabetical; `_neutral` first). This is not
  MediaPipe's native order. Unknown names are ignored. A unit test pins the
  list and the positions.
- **Blendshapes are not geometry.** They carry expression strength, not
  where the mouth or eyebrows are. Models that need face landmarks must take
  them from `raw_holistic.face`, never from these 52 values.
- **Presence flags** mean "MediaPipe returned this part", not a confidence.
- **Timing:** `SignRecorder` keeps a frame only if at least 1000/fps ms
  passed since the last kept frame (default 20 FPS, so ≥ 50 ms). Each
  recording stores `timestamps_ms`, one per frame, from the first frame, so
  dropped or late frames are visible.

## Recording format (`SignRecorder.stop`)

```json
{
  "id": "sample-001", "signerId": "signer-a", "language": "ASL",
  "glosses": ["ME", "NEED", "WATER"], "text": "I need water.",
  "fps": 20, "feature_schema": "visionmate-281-v1",
  "frames": [[281 numbers], ...], "timestamps_ms": [0, 50, 101],
  "raw_holistic": [{"timestamp_ms": 0, "left_hand": [...], "right_hand": [...],
                    "face": [...], "pose": [...],
                    "confidence_representation": "not_available_from_mediapipe_tasks"}]
}
```

`train_sign_transformer.py` reads `frames` (or `features`) and `glosses`.

## Training / inference consistency

Only one producer of this vector exists (the web tracker), so training and
inference cannot disagree yet. They will if a second producer is written
(for example a Python MediaPipe script for datasets). Before that happens:

1. Reproduce the test vector in `apps/mobile/tests/signFeatures.test.cjs`
   exactly (the blendshape list, timestamps and the vector are now pinned).

## Open questions to settle with data

- Whether to make coordinates body-relative (shoulder-centred, scaled by
  shoulder width). This removes camera distance and position, but must not
  remove movement.
- Whether to store handedness as signer's dominant / non-dominant hand
  instead of left / right.
- Whether pose landmarks below the hips (25–32) add anything but noise.

These are Phase 2.2 experiments and need a labelled evaluation set.
