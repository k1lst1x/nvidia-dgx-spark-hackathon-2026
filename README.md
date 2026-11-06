# GuideLens

Accessible AI vision companion for the **SFBU NVIDIA DGX Spark Hackathon 2026**. GuideLens uses a phone camera to identify nearby objects and speaks a concise scene description. This repo includes a universal Expo app, a web landing/demo, and a FastAPI gateway using Groq Vision.

It also includes **VisionMate**, a native SwiftUI iPhone client for the first phase of the assistive vision MVP.

> **Safety:** experimental prototypes. They must not replace a cane, guide dog, orientation training, medication instructions, or other mobility and safety aids.

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

## On-device detection

The native and desktop apps bundle the official Apache-2.0 YOLOX Nano ONNX model (3.5 MB). ONNX Runtime performs local detection; no object-detection frames leave the device. A compact COCO object signature triggers Groq only when objects change or when the eight-second semantic-description refresh expires. Desktop uses ONNX Runtime WebAssembly inside an Electron sandbox, so it continues detecting objects even without a configured backend.

## Desktop and Android downloads

The `Native release builds` GitHub Actions workflow produces real installable artifacts:

- Windows: NSIS `.exe` installer;
- macOS: `.dmg` and `.zip` application;
- Linux: AppImage and `.deb`;
- Android: installable debug APK for hackathon testing.

Run the workflow manually from **Actions**, or create a tag such as `v0.1.0` to build and attach every artifact to a GitHub Release. Set the repository Actions variable `EXPO_PUBLIC_API_URL` to the public HTTPS backend URL before building. YOLOX works offline; Groq scene descriptions require that backend.

Desktop artifacts are intentionally unsigned hackathon builds. Windows SmartScreen and macOS Gatekeeper may warn users. Public distribution should use Windows code signing and Apple signing/notarization. The Android debug APK is appropriate for direct testing, not a Play Store production release.

The model is intentionally YOLOX Nano rather than an Ultralytics checkpoint: YOLOX is Apache-2.0 compatible with this MIT project, while current Ultralytics models require AGPL-3.0 compliance or a commercial license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The bundled detector recognizes the 80 COCO classes and is a prototype aid, not a safety-certified navigation system. It cannot detect every curb, stair, hole, glass surface, or unusual obstacle.

### Optional Ultralytics YOLO scene detector

Scene Assist can additionally run an official Ultralytics YOLO checkpoint in the FastAPI service. This is opt-in: it augments the Groq scene prompt with verified labels and coordinates, and provides overlay boxes when the local detector is unavailable. It does not replace the fast bundled YOLOX detector used for offline updates.

```bash
# .env
ULTRALYTICS_YOLO_ENABLED=true
ULTRALYTICS_YOLO_MODEL=yolo11n.pt
# Use 0 on a CUDA-equipped DGX host, otherwise leave cpu.
ULTRALYTICS_YOLO_DEVICE=cpu
docker compose up --build
```

The checkpoint downloads into the service's model cache the first time Scene Assist runs; it is not committed to this repository. Ultralytics distributes its implementation under AGPL-3.0 and offers an Enterprise license for proprietary distribution. Review the [Ultralytics license terms](https://docs.ultralytics.com/help/license/) before enabling it outside an AGPL-compatible project.

When this detector is enabled, Scene Assist uses ByteTrack to keep a short-lived numeric ID on a detected object while the same camera feed is active. The ID resets after four idle seconds (configurable with `YOLO_TRACK_IDLE_SECONDS`) or when another session uses the shared API worker. It improves overlay continuity and depth matching; it is not a person identifier and does not improve YOLO's class accuracy.

### Approximate distance and approaching-object warnings

Scene Assist can use the official Apache-2.0 [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) metric-depth implementation to estimate object depth and issue a conservative warning only when a matching detected box becomes both closer and larger across several frames. This is not a collision detector, distance measurement tool, or safe navigation feature.

Download the official **metric Hypersim Small** checkpoint described in the repository's [metric-depth instructions](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) and save it as:

```text
apps/api/models/depth_anything_v2_metric_hypersim_vits.pth
```

Then enable it locally:

```bash
DEPTH_ANYTHING_ENABLED=true
DEPTH_ANYTHING_DEVICE=cuda # use cpu when CUDA is unavailable
docker compose up --build
```

Meter values are approximate. Alerts require a sustained depth decrease plus bounding-box growth and must not be relied on for street crossing, obstacle avoidance, or any other safety-critical decision.

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
