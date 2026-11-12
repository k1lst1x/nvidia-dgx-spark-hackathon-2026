# ASL 30-word and full-alphabet capability audit

**Status:** source audit completed 2026-11-11. This is an implementation
inventory, not an accuracy claim. A class is only *validated* after testing
with held-out, signer-independent recordings.

## What is already running

- `apps/mobile/components/HandTracker.web.tsx` extracts MediaPipe hand, face,
  and upper-body pose landmarks. Its `createMultimodalFrame` function produces
  the exact **390-value** layout consumed by the shipped word model: left hand,
  right hand, 76 face landmarks, and 12 pose landmarks.
- `apps/mobile/src/aslWordClassifier.web.ts` loads
  `asl_word_transformer.onnx` once with ONNX Runtime Web. The companion labels
  file declares a 64-frame, 390-value-per-frame, 250-class Transformer.
  `ASLWordEngine` uses bounded buffering, a motion gate, consecutive-window
  voting, and a duplicate cooldown.
- `apps/mobile/src/aslClassifier.ts` runs the separate 63-value static
  hand-landmark alphabet classifier. It supports A–I and K–Y; the model itself
  declares J and Z as unsupported motion letters. `ASLGestureEngine` provides
  hold, space, deletion, and timed repeated-letter behaviour.
- `apps/mobile/src/aslWordClassifier.web.ts` also has a hand-written temporal
  kinematic recognizer. It can emit `sorry`, `help`, and `stop`, but its
  confidence values are constants rather than calibrated probabilities.
- Text-to-speech and the Bridge remain separate output/communication features.
  They must only receive finalized recognition results.

## Important compatibility boundary

The repository has **two different feature schemas**:

- The deployed PopSign isolated-word ONNX model: **64 x 390** input.
- `apps/api/training/sign_transformer.py`: **281**-value CTC frames for an
  untrained next-stage model; no matching checkpoint is in the repository.

They are not interchangeable. The exported ONNX model does not include a
trainable checkpoint or an exposed reusable encoder interface in this project.
Do not train 281-value examples and attempt to attach their head to the
390-value ONNX model. A 30-word model must use one schema end-to-end, with its
own exported inference artifact and label file.

## Target vocabulary audit

| Target meaning | Current path | Status |
|---|---|---|
| HELLO | ONNX label `hello` | available, unvalidated |
| GOODBYE | ONNX label `bye`; the app renders it as “Goodbye” | display alias only; class performance unvalidated |
| THANK YOU | ONNX label `thankyou` | available, unvalidated |
| PLEASE | ONNX label `please` | available, unvalidated |
| SORRY | kinematic rule | available as an uncalibrated rule |
| YES | ONNX label `yes` and kinematic rule | available, unvalidated |
| NO | ONNX label `no` and kinematic rule | available, unvalidated |
| HELP | kinematic rule | available as an uncalibrated rule |
| STOP | kinematic rule | available as an uncalibrated rule |
| WAIT | ONNX label `wait` | available, unvalidated |
| WATER | ONNX label `water` and kinematic rule | available, unvalidated |
| FOOD | ONNX label `food` and kinematic rule | available, unvalidated |
| EAT | no label or recognition path | unsupported |
| DRINK | ONNX label `drink` and kinematic rule | available, unvalidated |
| BATHROOM | nearby ONNX label `potty` | not a verified training-label alias |
| HOME | ONNX label `home` | available, unvalidated |
| SCHOOL | no label or recognition path | unsupported |
| FRIEND | no label or recognition path | unsupported |
| FAMILY | no label or recognition path | unsupported |
| SLEEP | ONNX label `sleep` | available, unvalidated |
| GO | ONNX label `go` | available, unvalidated |
| COME | no label or recognition path | unsupported |
| WANT | no label or recognition path | unsupported |
| NEED | no label or recognition path | unsupported |
| LIKE | ONNX label `like` | available, unvalidated |
| UNDERSTAND | no label or recognition path | unsupported |
| KNOW | no label or recognition path | unsupported |
| WHERE | ONNX label `where` | available, unvalidated |
| WHAT | no label or recognition path | unsupported |
| WHO | ONNX label `who` | available, unvalidated |

The shipped labels contain no direct classes for **EAT, SCHOOL, FRIEND,
FAMILY, COME, WANT, NEED, UNDERSTAND, KNOW,** or **WHAT**. The repository
contains no dataset manifest, checkpoint, or evaluation evidence to prove that
`potty` is the intended ASL class for BATHROOM. It must remain an unverified
candidate until a fluent ASL reviewer verifies the source-label meaning.

## Existing safety limitations

- The default UI is a 45-label allowlist and can be switched to all 250 labels.
  It is not yet the requested exact-30 mode.
- The word model is evaluated in the browser only. Native currently uses a
  server image-sequence decoder, not the same landmark sequence and ONNX path.
- Current confidence floors and margins are heuristic. Existing documentation
  records false positives on unrelated signing, so they are not evidence of
  calibrated UNKNOWN rejection.
- The supplied 24-letter score (88.1%) is an internal image-split result; it
  is not signer-independent validation.

## Safe implementation order

1. Keep the current models untouched and add a target-vocabulary manifest with
   explicit `available`, `rule`, `candidate alias`, and `unsupported` states.
2. Add a J/Z classifier only after collecting labelled J, Z, and `UNKNOWN`
   motion sequences from multiple signers. It should be a small temporal model
   with a held-out signer split.
3. Collect licensed, reviewed landmark sequences for all 30 target meanings
   plus non-sign/unsupported movement. Train and export a single fixed-30
   temporal model with an UNKNOWN/no-sign class; do not merge independently
   calibrated probabilities from separate models without validation.
4. Add a feature-flagged inference adapter for web and native that consumes the
   same timestamped landmark sequences, model, labels, decoder, and thresholds.
5. Only after evaluation should the normal UI expose the exact 30-word mode;
   keep the 250-class model behind a developer setting for comparison.

## Data and licensing gate

The current PopSign/Google ISLR-derived artifact is an **isolated-sign**
model. It should not be described or used as unrestricted ASL-to-English
translation. Before collecting a missing class from an external dataset:

1. Record the exact source URL, licence, applicable terms, gloss definition,
   and whether the clip remains available.
2. Have a fluent ASL reviewer verify that the source gloss represents the
   target sign and document accepted regional variants.
3. Retain only the sequences needed for the approved target class and include
   signer-disjoint train, validation, and test splits.
4. Do not download WLASL or another external corpus into this repository until
   its terms and the video availability are explicitly recorded.

## Evidence locations

- Word model loading, buffering, and decoder:
  `apps/mobile/src/aslWordClassifier.web.ts`
- 390-value feature layout:
  `apps/mobile/components/HandTracker.web.tsx`
- Static alphabet classifier and decoder: `apps/mobile/src/aslClassifier.ts`
- Shipped word labels/configuration:
  `apps/mobile/assets/models/asl_word_labels.json`
- Current fixed-vocabulary measurements and limits:
  `docs/sign/FIXED_VOCABULARY_PLAN.md`
- Model/data license open items: `docs/sign/MODEL_LICENSE_MATRIX.md`
