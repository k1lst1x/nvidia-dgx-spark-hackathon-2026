# GuideLens — 1st Place Winner: SFBU NVIDIA DGX Spark Hackathon 2026

[![1st Place Winner](https://img.shields.io/badge/1st_Place_Winner-SFBU_NVIDIA_DGX_Spark_Hackathon_2026-gold?style=for-the-badge)](https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026)
[![Release v1.0.0](https://img.shields.io/badge/Release-v1.0.0-emerald?style=for-the-badge)](https://github.com/k1lst1x/nvidia-dgx-spark-hackathon-2026/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**GuideLens** is the **1st Place winning solution** of the **SFBU NVIDIA DGX Spark Hackathon 2026**. It is a real-time, privacy-first accessible AI vision and sign language ecosystem engineered for blind, visually impaired, deaf, and deaf-blind individuals.

Powered by edge AI and the **NVIDIA DGX Spark**, GuideLens translates visual scenes into natural spoken briefs, bridges spoken speech and American Sign Language (ASL) via 3D skeletal avatars and on-device MediaPipe gesture recognition, and maps hazards through a cooperative, privacy-preserving campus memory.

> **Safety:** Assistive companion technology. Not a substitute for a white cane, guide dog, orientation training, mobility aids, or medical devices.

## Award-Winning Highlights

- **Spatial Perception & Scene Audio Briefs:** Instant on-device YOLOX Nano + BoT-SORT object tracking and depth awareness, producing calm spoken scene summaries via DGX Spark (Qwen-VL) with cloud fallback.
- **Deaf–Blind Bridge:** Direct two-way cross-disability communication. ASL gestures captured via camera are voiced aloud to blind users, while spoken responses render in giant high-contrast typography and animated signs for deaf participants.
- **English ⇄ ASL Sign Studio:** Bidirectional English-to-ASL grammar engine featuring a hardware-accelerated Three.js WebGL 3D PBR skeletal hand avatar with 360° orbit drag, pacing controls, and automatic fingerspelling.
- **Shared Sight & Campus Memory:** Collaborative spatial hazard detection (wet floors, moved obstacles, closed doors) pinned to shared 3D room sketches without ever storing or uploading raw camera imagery.
- **100% Offline Autonomy:** Zero-network fallback with embedded WASM/native neural inference for object detection and sign language translation anywhere on campus.

## Stack

- Expo SDK 58 + React Native 0.88 + TypeScript for iOS and Android.
- Electron for signed-or-local desktop installers on Windows, macOS and Linux.
- `expo-camera` + `expo-speech` for camera preview and native text-to-speech.
- YOLOX Nano + ONNX Runtime for private on-device object detection on iOS, Android and desktop.
- FastAPI on Python 3.12 in Docker as a secure Groq Vision gateway.
- NVIDIA DGX Spark as the intended inference host using NIM/Triton or TensorRT.
- GitHub Actions + Pages for automatic web delivery.

Expo/React Native has no LTS line. We pin a stable Expo SDK and upgrade one SDK at a time. Node 22 and Python 3.12 are stable LTS-supported foundations.

## Run the Expo MVP

Install Node.js 22.13+. Because GuideLens includes ONNX Runtime native code, use an Expo development build rather than Expo Go.

```bash
npm install
npx eas-cli build --platform android --profile development
npx eas-cli build --platform ios --profile development
npm run dev
```

Install the development build from its EAS link, then scan the development-server QR. Run the browser-only fallback with `npm run web`; web uses Groq without the native YOLO detector.

```bash
cp .env.example .env
# Add a newly rotated GROQ_API_KEY to .env. Never commit it.
docker compose up --build
curl http://localhost:8000/health
```

Set `EXPO_PUBLIC_API_URL` to the backend address reachable from the phone, for example `http://192.168.1.25:8000`, before starting Expo. The camera loop sends one compressed frame at a time, adapts its interval to scene urgency/change, suppresses repeated speech, and exponentially backs off on rate limits or network failures.

## Native iPhone client: VisionMate Phase 1

`VisionMate/` is a native SwiftUI iOS 17 project for three everyday visual tasks:

- describe the scene in front of the user;
- read visible text aloud; and
- accept a spoken question about what is visible.

It uses only Apple frameworks: `AVFoundation` for the rear camera and spoken answers, `Speech` for question transcription, `URLSession` for the vision request, and Keychain for an optional API key. The UI has large high-contrast controls, VoiceOver labels and hints, automatic spoken answers, and a replay control.

Open `VisionMate/VisionMate.xcodeproj` in Xcode 16 or later, select a physical iPhone running iOS 17 or later, choose your Apple development team under **Signing & Capabilities**, and run it. A physical phone is required for the camera and speech flow.

In VisionMate, open **Server settings** and enter the OpenAI-compatible DGX Spark base URL. The default is:

```
http://dgx-spark.local:8000/v1
```

VisionMate calls `POST /v1/chat/completions` with a JPEG data URL and an optional spoken question. Use a multimodal model such as the default `Qwen/Qwen3-VL-32B-Instruct`: text-only Qwen3-32B cannot accept the image input required by the camera features. Qwen documents Qwen3-VL image chat through its [OpenAI-compatible API example](https://github.com/QwenLM/Qwen3-VL).

For a local HTTP address, add an App Transport Security exception appropriate to the deployment network before distribution. The development plist permits local networking only; it does not permit arbitrary insecure internet traffic.

## Distribution

| Target | Hackathon path | Public release |
|---|---|---|
| Web | GitHub Pages URL | Same URL in a browser |
| iPhone/iPad | Signed EAS development build or VisionMate | TestFlight / App Store via EAS Build or Xcode |
| Android | Installable APK from GitHub Releases | Google Play via EAS Build |
| Windows | Portable `.exe` or installer | Code-sign and distribute the installer |
| macOS | `.dmg` or `.zip` application | Sign and notarize with an Apple Developer account |
| Linux | AppImage or `.deb` | GitHub Releases or a package repository |

GitHub Pages hosts the web app and download links, but cannot distribute a generally installable unsigned iOS app. iOS builds must be signed and normally delivered through TestFlight/App Store. Android APKs can be attached to GitHub Releases. HarmonyOS and KaiOS exist, but are not sensible MVP targets; the PWA is the practical fallback.

## Recommended first commits

1. `chore: scaffold universal Expo app and repository tooling`
2. `feat: add accessible camera and speech demo`
3. `feat: add mocked DGX inference API`
4. `ci: deploy web MVP to GitHub Pages`
5. `docs: document device testing, safety and distribution`

Before replacing mocks, validate camera → API → speech on SFBU Wi-Fi. A phone cannot reach `localhost` on the DGX: set `EXPO_PUBLIC_API_URL` to the DGX Spark LAN address and allow TCP port 8000. Keep a local fallback for demo resilience.

## Groq Vision architecture

The app never receives the Groq key. It sends a compressed JPEG to `POST /v1/describe`; the backend validates size and media type, rate-limits sessions, bounds concurrency, and calls `qwen/qwen3.8-27b` with strict structured output. Only meaningful scene changes are spoken. Frames and model responses are not persisted or logged.

Groq bills each image as input tokens, so cloud analysis is intentionally near-real-time rather than video streaming: one request is allowed in flight and the on-device detector decides when Groq is needed.

## Shared Sight, What Changed, and Personal Recall

Shared Sight uses SQLite at `/data/visionmate_memory.sqlite3` in Docker Compose, persisted under `apps/api/data/`. It stores observation labels and metadata only; it does not store camera images or BoT-SORT track IDs. Reports include room/zone, timestamp, source device/session, confidence, verification status, privacy scope, and expiry. Unverified reports below `OBSERVATION_MIN_CONFIDENCE` are rejected. A user-confirmed shared detection uses the separate `OBSERVATION_VERIFIED_SHARED_MIN_CONFIDENCE` floor (0.40 by default); the user must explicitly confirm it, and its original model confidence is retained. Duplicate labels in the same zone/scope are rejected for 30 seconds. Expired reports are removed as the store is read or written.

Choose a room from the existing room selector. In Scene Assist, review the displayed object detections and tap **I confirm these detections · Share** to send high-confidence object labels to that room. This action sends labels and confidence only, not the image. The Hub dashboard polls recent verified reports, shows active devices and recent tracking latency, and offers Personal Recall search. Personal Recall is private to the stable device ID and persists across app sessions. The What Changed summary compares verified shared observations from the selected zone across two adjacent time windows. An absence from a window is not treated as proof that an object was removed.

The Hub also includes a coarse shared room sketch. Users place a pin on the sketch when submitting a verified shared report; pin coordinates are normalized within that sketch and shown to other authorized devices for the same zone. This is a user-placed, not-to-scale diagram: camera detections are not automatically positioned, and the sketch does not represent metric depth, SLAM, or a measured floor plan.

Every client generates a 256-bit device secret. Its stable public ID contains only the first 128 bits; the API verifies the complete secret without allocating enrollment state, so a different client cannot impersonate that ID, read its Personal Recall, or reuse its bridge membership. The API also supports an optional deployment-wide `SHARED_SIGHT_TOKEN` and comma-separated `SHARED_SIGHT_DEVICE_IDS` allowlist. Before upgrading an existing SQLite database, configure `SHARED_SIGHT_TOKEN`; a legacy ID that already owns records may enroll its new device secret only while presenting that migration key, after which the API stores only its SHA-256 digest. Fresh installations require no migration. SQLite remains intended for a single API instance/demo deployment, not multiple replicas.

Inference limits combine per-session cadence with peer and process-wide ceilings, so changing `X-Session-ID` cannot create unlimited paid/GPU work. Bridge joins have separate peer, device, and global attempt budgets. These in-process controls protect the single-process demo; a multi-worker or internet-facing deployment should enforce matching distributed limits at a trusted gateway.

The tracking response also includes Camera Coach heuristics for blur, lighting, overexposure, and detector boxes touching image edges. These checks provide framing guidance and are not calibrated quality guarantees. Object detection, tracking, depth, motion, and speech paths remain separate from SQLite writes. Shared Sight is a recent-report feature; it does not provide reliable autonomous navigation or collision avoidance.

## ASL fingerspelling training

The supplied ASL dataset trains a **static fingerspelling** model for A–I and K–Y. It does not contain full sentence meanings, and J and Z are excluded because they require motion. The trainer streams the archive directly, extracts the same MediaPipe hand-landmark representation used by the web app, and exports a portable softmax classifier.

```bash
uv venv --python 3.12 /private/tmp/asl-fingerspelling-venv
uv pip install --python /private/tmp/asl-fingerspelling-venv/bin/python -r apps/api/requirements-train.txt
/private/tmp/asl-fingerspelling-venv/bin/python apps/api/training/train_asl_fingerspelling.py \
  --dataset "/path/to/American Sign Language Image Dataset (figerspelling original + keypoints + evaluation).zip" \
  --output apps/api/models/asl_fingerspelling_landmarks.json
```

The live sign feature samples overlapping temporal windows, preserves an accepted transcript across windows, suppresses uncertain/repeated segments, and speaks only confident phrase boundaries. Desktop/web tracks both hands, 76 facial landmarks, and 12 upper-body pose landmarks locally. A bundled ONNX Transformer recognizes isolated signs from the 250-word PopSign vocabulary, while the static fingerspelling model handles letters and explicit emergency phrase shortcuts. Cloud phrase decoding is added automatically when a backend is configured. Android samples camera windows for the backend decoder; native offline hand landmarks remain future work. These are isolated-word and experimental phrase features, not a validated continuous-ASL interpreter.

### Causal temporal CTC recognizer

`apps/api/training/sign_transformer.py` defines the next-stage continuous-sign recognizer: a four-layer, eight-head causal Transformer that projects each 281-value fused hand/pose/face frame into 256 dimensions and emits gloss logits through a CTC head. Its causal attention mask only exposes past and present frames, making it suitable for live decoding. The recognition model and English translation model remain separate so a decoder cannot silently invent missing signing information.

Train it with a JSONL manifest of **gloss-labelled temporal sequences** (not one sentence label repeated per frame):

```bash
uv pip install --python /private/tmp/asl-fingerspelling-venv/bin/python -r apps/api/requirements-train.txt
/private/tmp/asl-fingerspelling-venv/bin/python apps/api/training/train_sign_transformer.py \
  --train /path/to/train_sequences.jsonl \
  --validation /path/to/validation_sequences.jsonl
```

Each JSONL row has `features` shaped `(frames, 281)` and a `glosses` list, for example `{"features": [[...]], "glosses": ["ME", "NEED", "WATER"]}`. The provided fingerspelling archive cannot train this model because it has static letter images rather than synchronized movement, face, pose, and gloss-sequence labels.

For small, privacy-preserving capture sessions, `apps/mobile/src/landmarkRecorder.web.ts` provides `SignRecorder`. It samples the live 281-feature stream at 20 FPS and downloads JSON records containing landmark frames, signer ID, verified glosses, and optional English text—never the camera video. The trainer accepts its `frames` field directly. Add controls only for an annotation workflow where a fluent signer can verify the gloss labels.

### Optional Uni-Sign How2Sign research backend

VisionMate retains raw MediaPipe Holistic hand, face, pose, and timestamp data in recorder exports as well as its existing 281-value CTC stream. `apps/api/unisign_adapter.py` maps raw landmarks to the four pose streams used by Uni-Sign's released pose-only architecture: body `(T,9,3)`, left hand `(T,21,3)`, right hand `(T,21,3)`, and face `(T,18,3)`. This is not a direct checkpoint-compatible replacement: the official How2Sign checkpoint expects RTM WholeBody 133-point topology and per-landmark confidence scores, while MediaPipe Tasks does not expose those scores and Face Mesh does not have the same 18-point face topology.

The adapter refuses official-checkpoint inference by default in that case. A clearly labelled `presence_proxy` experiment exists only to test data plumbing and must not be used for translation-accuracy claims. Validate an export with:

```bash
PYTHONPATH=apps/api python apps/api/training/verify_unisign_adapter.py recording.json
PYTHONPATH=apps/api python apps/api/training/verify_unisign_adapter.py recording.json --allow-presence-proxy
```

Uni-Sign's official repository and checkpoint require a separate local installation, include an mT5 decoder, and are released under a non-commercial checkpoint license. The released online script processes a finite clip rather than causal live windows, so a real-time deployment needs benchmarked causal/windowed decoding and selective fine-tuning with paired MediaPipe/RTM pose data.

## On-device detection

The native and desktop apps bundle the official Apache-2.0 YOLOX Nano ONNX model (3.5 MB). ONNX Runtime performs local detection; no object-detection frames leave the device. A compact COCO object signature triggers Groq only when objects change or when the eight-second semantic-description refresh expires. Desktop uses ONNX Runtime WebAssembly inside an Electron sandbox, so it continues detecting objects even without a configured backend.

## Desktop and Android downloads

The `Native release builds` GitHub Actions workflow produces real installable artifacts:

- Windows: NSIS `.exe` installer;
- macOS: `.dmg` and `.zip` application;
- Linux: AppImage and `.deb`;
- Android: installable debug APK for hackathon testing.

Run the workflow manually from **Actions**, or create a tag such as `v1.0.0` to build and attach every artifact to a GitHub Release. Set the repository Actions variable `EXPO_PUBLIC_API_URL` to the public HTTPS backend URL before building. YOLOX works offline; Groq scene descriptions require that backend.

Desktop artifacts are intentionally unsigned hackathon builds. Windows SmartScreen and macOS Gatekeeper may warn users. Public distribution should use Windows code signing and Apple signing/notarization. The Android debug APK is appropriate for direct testing, not a Play Store production release.

The model is intentionally YOLOX Nano rather than an Ultralytics checkpoint: YOLOX is Apache-2.0 compatible with this MIT project, while current Ultralytics models require AGPL-3.0 compliance or a commercial license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The bundled detector recognizes the 80 COCO classes and is a prototype aid, not a safety-certified navigation system. It cannot detect every curb, stair, hole, glass surface, or unusual obstacle.

### Optional Ultralytics YOLO scene detector

Scene Assist can additionally run an official Ultralytics YOLO checkpoint in the FastAPI service. This is opt-in: it augments the Groq scene prompt with verified labels and coordinates, and provides overlay boxes when the local detector is unavailable. It does not replace the fast bundled YOLOX detector used for offline updates.

```bash
# .env
ULTRALYTICS_YOLO_ENABLED=true
ULTRALYTICS_YOLO_MODEL=yolo11n.pt
# auto = CUDA, then Apple MPS, then CPU. See docs/DEVICES.md.
YOLO_DEVICE=auto
docker compose up --build
# With an NVIDIA GPU: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Each model picks its device on startup (`YOLO_DEVICE`, `SEG_DEVICE`, `DEPTH_DEVICE`, `SIGN_DEVICE`), runs one test inference there, and falls back to CPU if that fails. `/health` → `devices` shows what each model runs on and why. [docs/DEVICES.md](docs/DEVICES.md) covers installing a CUDA or MPS build of PyTorch on Windows, Linux, DGX Spark and macOS, and lists measured CPU vs GPU speeds.

The checkpoint downloads into the service's model cache the first time Scene Assist runs; it is not committed to this repository. Ultralytics distributes its implementation under AGPL-3.0 and offers an Enterprise license for proprietary distribution. Review the [Ultralytics license terms](https://docs.ultralytics.com/help/license/) before enabling it outside an AGPL-compatible project.

When this detector is enabled, Scene Assist uses BoT-SORT with sparse optical-flow camera-motion compensation to keep a short-lived numeric ID on an object while the glasses camera moves. The ID resets after four idle seconds (configurable with `YOLO_TRACK_IDLE_SECONDS`) or when another session uses the shared API worker. Camera-motion compensation improves 2-D association only: it is not metric 3-D ego-motion, a persistent person identifier, or a class-accuracy improvement.

### Experimental distance and relative-motion debug data

Scene Assist can use the official Apache-2.0 [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) metric-depth implementation to estimate the central visible surface of each tracked box. `POST /v1/track` is the fast detector → depth → timestamped-history path and does not call Groq; `POST /v1/describe` remains the slower verbal-description path. With a metric checkpoint and several stable readings, the API can expose **experimental camera-relative** depth trends, relative depth velocity, and a 1-D closest-approach projection in object debug data.

Download the official **metric Hypersim Small** checkpoint described in the repository's [metric-depth instructions](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) and save it as:

```text
apps/api/models/depth_anything_v2_metric_hypersim_vits.pth
```

Then enable it locally:

```bash
DEPTH_ANYTHING_ENABLED=true
DEPTH_DEVICE=auto # CUDA, then MPS, then CPU
docker compose up --build
```

Meter values are approximate. The app intentionally keeps collision and mobility alerts disabled (`MOBILITY_COLLISION_WARNINGS_ENABLED=false`): BoT-SORT's compensation does not tell whether an apparent change came from the user walking or an object moving. World-frame motion requires calibrated metric depth plus an external camera-pose source and camera intrinsics; the included `EgoMotionProvider` interface is experimental and has no production provider yet. Do not use this for street crossing, obstacle avoidance, or any other safety-critical decision.

To benchmark real recordings, annotate the five scenarios—stationary objects, approaching pedestrians, crossing objects, camera turns, and camera shake—then run:

```bash
PYTHONPATH=apps/api python apps/api/benchmarks/benchmark_motion.py --manifest clips.jsonl
```

The tool reports null rather than fabricating FPS, depth error, trend accuracy, or ID stability when the supplied recording manifest lacks timing or ground truth.

### Installable builds

Source code and build configuration are ready, but GitHub Pages does not produce native binaries. Create an Android APK with:

```bash
npx eas-cli login
npx eas-cli build --platform android --profile preview
```

EAS returns an installable APK URL. For iPhone, an Apple Developer account and signing are required:

```bash
npx eas-cli build --platform ios --profile production
npx eas-cli submit --platform ios
```

The submitted iOS build is distributed to testers through TestFlight.

If a Groq key is ever pasted into chat, an issue, a commit, or a client bundle, revoke it immediately and create a new one. Never reuse an exposed key.

## GitHub Pages

Set **Settings → Pages → Source: GitHub Actions**, then push `main`. If the repository is renamed, update `experiments.baseUrl` in `apps/mobile/app.json`.

MIT licensed.
