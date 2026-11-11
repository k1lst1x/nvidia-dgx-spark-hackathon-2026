# Sign pipeline audit

Audited from the code on `main` at `438cf1b` plus commit `70d8c98`, on 2026-11-11.
Everything below was read from the repository or measured in a CPU-only
container. Nothing here was tested with a real camera, a real signer or the
DGX Spark.

## Summary

| Component | State | Evidence |
|---|---|---|
| MediaPipe landmark extraction (web) | **Runs** | `apps/mobile/components/HandTracker.web.tsx` |
| 250-word isolated-sign Transformer (ONNX, in the browser) | **Runs; accuracy not measured by us** | `apps/mobile/assets/models/asl_word_transformer.onnx` |
| Rule-based gestures (19 signs) | **Runs; not validated** | `KinematicGestureRecognizer` in `aslWordClassifier.web.ts` |
| Fingerspelling classifier (24 static letters) | **Runs**; 88.1% on its own 679-image test split | `asl_fingerspelling_landmarks.json` |
| Cloud sign decoder (vision LLM over 3 frames) | **Runs when a vision provider is configured** | `/v1/interpret-sign` |
| 281-feature causal CTC Transformer | **Code only. No checkpoint, no training data, not loaded anywhere** | `apps/api/training/sign_transformer.py` |
| `SignRecorder` (281-feature capture) | **Code only. No screen uses it** | `apps/mobile/src/landmarkRecorder.web.ts` |
| YOLO11 sign detector fallback | **Not operational: weights are not in the repo** | `apps/api/yolo_sign.py` |
| Uni-Sign backend | **Not operational: off by default, assets not present, mapping unverified** | `apps/api/unisign_backend.py` |
| OpenHands | **Not present in the code** | no references |
| Sign Language Bridge (Qwen3-VL LoRA) | **Not present in the code** | no references |
| Sign Language Translator | **Not present in the code** | no references |
| Pinto keypoint / point-history ONNX models | **Bundled but never loaded** | no references in `src/`, `components/`, `app/` |
| Native apps (iOS/Android) | **No sign tracking** | `HandTracker.native.tsx` renders nothing |

The system that runs today is an **isolated-sign recognizer with a word-by-word
English gloss**, plus an optional cloud decoder. It is not continuous sign
recognition. The "Temporal Transformer with CTC head" in the specification
exists as an architecture and a training script only.

## 1. Landmark extraction

- **File:** `apps/mobile/components/HandTracker.web.tsx` (web only).
- **Runtime:** `@mediapipe/tasks-vision` 1.2.0, loaded from the app's own
  `/mediapipe` folder, with the public CDN as fallback.
- **Models:** `holistic_landmarker` (float16) first. If it fails, separate
  `hand_landmarker`, `face_landmarker` and `pose_landmarker_lite`.
  GPU delegate first, CPU on failure.
- **Hands:** up to 2. Detection, presence and tracking thresholds are 0.45
  (0.4 for holistic).
- **Rate:** one detection per video frame the browser delivers. The "about 20
  FPS" figure in the specification is `SignRecorder`'s sampling rate, not a
  measured extraction rate. **Extraction FPS has not been measured.**
- **Per-landmark confidence:** not available from MediaPipe Tasks. The code
  records this explicitly (`confidence_representation`).

## 2. Feature vectors

Two different vectors are built from the same landmarks:

| Vector | Size | Built by | Consumed by |
|---|---|---|---|
| PopSign layout | 390 | `createMultimodalFrame` | the 250-word ONNX model (live) |
| VisionMate CTC layout | 281 | `createCtcFeatureFrame` | `SignRecorder` only (no model) |

The 281 layout is specified in `FEATURE_SCHEMA_281.md`.

390 layout: left hand 21×3, right hand 21×3, 76 selected face landmarks ×3,
12 pose landmarks ×3 (`POSE_INDICES = 0, 11–16, 23–27`). Missing parts are
zeros. Coordinates are raw MediaPipe image-normalized values.
**Not verified:** that this matches the upstream model's training
preprocessing (landmark selection, order, NaN handling, any normalization).
The upstream repository was not readable from this environment. Until that is
checked, the model's accuracy in this app is unknown.

## 3. Models that run

### 3.1 Isolated-sign Transformer (primary recognizer)

- **File:** `apps/mobile/assets/models/asl_word_transformer.onnx` (9.8 MB),
  SHA-256 in `THIRD_PARTY_NOTICES.md`.
- **Source:** `github.com/aymenwnia/ASL-translator`,
  `models/sign_transformer_best.pth`, converted by
  `apps/api/training/export_asl_word_transformer.py`.
- **Architecture:** 4 layers, 8 heads, width 256, feed-forward 512; mean + max
  pooling; 250 classes.
- **Input:** `landmarks`, float32, shape `[1, 64, 390]`. All-zero frames are
  treated as padding.
- **Output:** `logits`, shape `[1, 250]`. Softmax gives a per-window class
  probability. It is not calibrated.
- **Vocabulary:** 250 words in `asl_word_labels.json` (PopSign / Google ISLR).
- **Runtime:** ONNX Runtime Web, WASM, in the browser.
- **Measured here:** 2.2 ms per inference on 4 CPU cores with native ONNX
  Runtime. Browser WASM on a phone will be slower; not measured.
- **Accuracy:** not measured by this project.

How it is driven (`ASLWordEngine`):
1. A rolling buffer of up to 64 frames; inference every 3rd frame once 12
   frames exist. Shorter buffers are zero-padded at the end.
2. A motion gate skips inference when the wrist moves less than 0.003 per frame.
3. A sign is confirmed when it is the top class in 2 consecutive windows at
   ≥ 0.45, or 3 at ≥ 0.30.
4. The same sign is not confirmed again within 2.5 s.
5. The buffer resets after 8 frames with no hands. The phrase ends after 30
   frames with no hands. A phrase holds at most 12 words.

### 3.2 Rule-based gestures

`KinematicGestureRecognizer` checks hand shapes and motion with hand-written
rules for 19 labels (`hello`, `yes`, `no`, `please`, `thankyou`, `sorry`,
`help`, `stop`, `water`, `food`, `drink`, `more`, `done`, `pain`, `safe`,
`ily`, `call`, `okay`, `peace`). A match at ≥ 0.80 **overrides the
Transformer** and is confirmed immediately. The confidence values are
constants written in the code, not measured. Some labels (`peace`, `okay`,
`call`) are gestures rather than ASL signs.

### 3.3 Fingerspelling

- **File:** `asl_fingerspelling_landmarks.json`; logistic regression over 63
  wrist-centred hand coordinates, in `apps/mobile/src/aslClassifier.ts`.
- **Vocabulary:** 24 static letters. **J and Z are not supported** (they need
  motion).
- **Self-reported accuracy:** 88.1% on 679 held-out images (84,497 training
  examples). Signer independence of that split is unknown.

### 3.4 Cloud sign decoder

- **Endpoint:** `POST /v1/interpret-sign` with exactly 3 JPEG frames and the
  previous transcript.
- **Model:** the configured vision model (Spark first, Groq as backup after
  `70d8c98`). Frames are joined into one strip, 320 px per frame.
- **Guards in code:** the model may only append to the transcript; non-English
  output is dropped; results below `SIGN_MIN_CONFIDENCE` (0.45) are dropped.
- **Limits:** three stills cannot carry the movement of a sign. A general
  vision LLM was not trained on ASL. Its `confidence` is self-reported text,
  not a probability. **This is the component most likely to produce a fluent
  sentence that the signer did not sign.**
- **Fallback:** on failure it calls the YOLO sign detector, which has no
  weights (see below), so the request fails.

## 4. Models that exist in code but do not run

### 4.1 VisionMate causal CTC Transformer

- **Files:** `apps/api/training/sign_transformer.py`,
  `train_sign_transformer.py`.
- **Architecture:** 281 → 256, 4 layers, 8 heads, feed-forward 1024, causal
  mask, CTC head of `num_glosses + 1`. Blank index 0, consistently in the
  model, loss and decoder.
- **Checkpoint:** none. The trainer's default output
  `apps/api/models/visionmate_ctc_transformer.pt` does not exist.
- **Training data:** none in the repository.
- **Loading code:** none. No server module or app screen imports it.
- **Decoder:** greedy `decode_ctc` only, whole-sequence. No streaming state,
  no beam search.
- **Dependencies:** `torch` is in `requirements-train.txt` only.

### 4.2 `SignRecorder`

Complete and consistent with the trainer's JSONL format (`frames`, `glosses`,
`signerId`, optional `raw_holistic`). `HandTracker` accepts a
`trainingRecorder` prop, but **no screen passes one**, so there is no way to
record a sample from the app today.

### 4.3 YOLO11 sign detector

`yolo_sign.py` expects `apps/api/models/yolo11_sign_language.pt` or `.onnx`.
Neither is in the repository, and the source and licence of those weights are
not documented. It classifies single frames into 46 classes (digits, letters
and 10 words), so it cannot recognize signs that depend on movement.

### 4.4 Uni-Sign

- **Files:** `unisign_backend.py`, `unisign_adapter.py`,
  `training/verify_unisign_adapter.py`, `tests/test_unisign_adapter.py`.
- **Endpoints:** `GET /v1/unisign/status`, `POST /v1/unisign/interpret`.
- **State:** off (`UNISIGN_ENABLED=false`). Needs the official repository at
  `/opt/uni-sign`, the How2Sign pose-only checkpoint and `mt5-base`; none are
  present. No app code calls the endpoint.
- **Known incompatibility, stated in the code:** the checkpoint was trained on
  RTM WholeBody keypoints with per-point confidence. MediaPipe has neither the
  same face topology nor confidences. The adapter refuses by default and marks
  any forced run `topology_verified: false`.

## 5. Live flow to the screen and to speech

`HandTracker` → `ASLWordEngine` / `ASLGestureEngine` → `onContextChange` →
`apps/mobile/app/camera.tsx` (Sign to voice) and `app/bridge.tsx`.

- **Text:** `apps/mobile/src/signEnglish.ts` turns confirmed labels into
  English with fixed rules (`glossToEnglish`). No language model.
- **Speech:** `expo-speech`. In Sign to voice, **each confirmed word is spoken
  as soon as it is confirmed**, and the whole sentence is spoken again when
  the hands have been down for 30 frames. The cloud decoder's segment is
  spoken when it returns one.
- **Bridge:** a word is sent once per gesture at ≥ 0.70; spelled words are
  sent only when the user presses Send.

## 6. Known failure modes

1. **Unverified input layout** for the 250-word model (section 2).
2. **No "no sign" class.** Any movement is scored against 250 words; the
   threshold of 0.30 is low for 250 classes. False words are likely on
   non-signing movement.
3. **Rules override the model** with fixed confidence (3.2).
4. **Handedness** comes from MediaPipe's label. With a mirrored front camera
   it can be swapped, which moves the hand into the wrong slot of both vectors.
5. **Speech before the phrase is final** (section 5).
6. **Pause = end of sentence.** There is no "finish sentence" control in Sign
   to voice.
7. **Cloud decoder can invent text** (3.4), and its fallback is missing.
8. **Blendshape order** in the 281 vector is alphabetical, not MediaPipe's
   order (`FEATURE_SCHEMA_281.md`). Harmless while consistent, but nothing
   enforces it.
9. **Native apps** have no sign tracking.

## 7. API endpoints

| Endpoint | Purpose | State |
|---|---|---|
| `POST /v1/interpret-sign` | 3 frames → English (vision LLM) | works with a provider |
| `POST /v1/detect-sign` | 1 frame → YOLO sign class | returns empty (no weights) |
| `GET /v1/unisign/status` | Uni-Sign readiness | works, reports not ready |
| `POST /v1/unisign/interpret` | raw Holistic → English | returns 503 |

There is no endpoint for landmark sequences, and none is needed today because
the only landmark model runs in the browser.

## 8. What this means for the specification

- **Phase 4 (CTC decoding, streaming state)** has nothing to decode until the
  CTC model has data and a checkpoint. The utterance state machine can still
  be built on the isolated-sign recognizer.
- **Phases 3 and 5** need verified signing data, which the project does not
  have. `SignRecorder` must be wired to a screen first.
- **Definition of Done, current status:** correct checkpoint loads (yes, for
  the 250-word model); input mapping confirmed (**no**); runs on real camera
  data (yes, untested here); vocabulary respected (yes); duplicate control
  (yes); sentence finalization (pause only); no unstable speech (**no**);
  performance on target hardware (**no**); tested against verified ASL data
  (**no**); limits documented (this file).
