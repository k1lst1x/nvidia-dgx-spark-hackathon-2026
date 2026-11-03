# GuideLens

Accessible AI vision companion for the **SFBU NVIDIA DGX Spark Hackathon 2026**. GuideLens uses a phone camera to identify nearby objects and speaks a concise scene description. This repo includes a universal Expo app, a web landing/demo, and a FastAPI gateway using Groq Vision.

It also includes **VisionMate**, a native SwiftUI iPhone client for the first phase of the assistive vision MVP.

> **Safety:** experimental prototypes. They must not replace a cane, guide dog, orientation training, medication instructions, or other mobility and safety aids.

## Stack

- Expo SDK 58 + React Native 0.88 + TypeScript for iOS, Android and web/PWA.
- `expo-camera` + `expo-speech` for camera preview and native text-to-speech.
- YOLOX Nano + ONNX Runtime for private on-device object detection on iOS and Android.
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
| Web/PWA | GitHub Pages URL | Same URL; install from browser where supported |
| iPhone/iPad | Signed EAS development build or VisionMate | TestFlight / App Store via EAS Build or Xcode |
| Android | EAS preview APK | Google Play via EAS Build |

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

## On-device detection

The native app bundles the official Apache-2.0 YOLOX Nano ONNX model (3.5 MB). ONNX Runtime performs local detection; no object-detection frames leave the phone. A compact COCO object signature triggers Groq only when objects change or when the eight-second semantic-description refresh expires. The browser build gracefully falls back to Groq-only analysis.

The model is intentionally YOLOX Nano rather than an Ultralytics checkpoint: YOLOX is Apache-2.0 compatible with this MIT project, while current Ultralytics models require AGPL-3.0 compliance or a commercial license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The bundled detector recognizes the 80 COCO classes and is a prototype aid, not a safety-certified navigation system. It cannot detect every curb, stair, hole, glass surface, or unusual obstacle.

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
