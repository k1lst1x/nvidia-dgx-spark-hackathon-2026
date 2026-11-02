# GuideLens

Accessible AI vision companion for the **SFBU NVIDIA DGX Spark Hackathon 2026**. GuideLens uses a phone camera to identify nearby objects and speaks a concise scene description. This repo includes a working mocked MVP: one universal Expo app, a web landing/demo, and a FastAPI inference contract ready for DGX Spark.

It also includes **VisionMate**, a native SwiftUI iPhone client for the first phase of the assistive vision MVP.

> **Safety:** experimental prototypes. They must not replace a cane, guide dog, orientation training, medication instructions, or other mobility and safety aids.

## Stack

- Expo SDK 58 + React Native 0.88 + TypeScript for iOS, Android and web/PWA.
- `expo-camera` + `expo-speech` for camera preview and native text-to-speech.
- FastAPI on Python 3.12 in Docker as the backend boundary.
- NVIDIA DGX Spark as the intended inference host using NIM/Triton or TensorRT.
- GitHub Actions + Pages for automatic web delivery.

Expo/React Native has no LTS line. We pin a stable Expo SDK and upgrade one SDK at a time. Node 22 and Python 3.12 are stable LTS-supported foundations.

## Run the Expo MVP

Install Node.js 22.13+ and Expo Go on the phone.

```bash
npm install
npm run dev
```

Scan the QR with Expo Go (Android) or the iPhone Camera app. Run web with `npm run web`.

```bash
docker compose up --build
curl http://localhost:8000/health
```

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
| iPhone/iPad | Expo Go QR or VisionMate development build | TestFlight / App Store via EAS Build or Xcode |
| Android | Expo Go QR or preview APK | Google Play via EAS Build |

GitHub Pages hosts the web app and download links, but cannot distribute a generally installable unsigned iOS app. iOS builds must be signed and normally delivered through TestFlight/App Store. Android APKs can be attached to GitHub Releases. HarmonyOS and KaiOS exist, but are not sensible MVP targets; the PWA is the practical fallback.

## Recommended first commits

1. `chore: scaffold universal Expo app and repository tooling`
2. `feat: add accessible camera and speech demo`
3. `feat: add mocked DGX inference API`
4. `ci: deploy web MVP to GitHub Pages`
5. `docs: document device testing, safety and distribution`

Before replacing mocks, validate camera → API → speech on SFBU Wi-Fi. A phone cannot reach `localhost` on the DGX: set `EXPO_PUBLIC_API_URL` to the DGX Spark LAN address and allow TCP port 8000. Keep a local fallback for demo resilience.

## Real inference

Send a throttled JPEG to `POST /v1/describe`. Replace the mock body in `apps/api/main.py` with the local Triton/NIM call. Return short `speech`, a one-line `summary`, and object boxes. Do not retain video by default.

## GitHub Pages

Set **Settings → Pages → Source: GitHub Actions**, then push `main`. If the repository is renamed, update `experiments.baseUrl` in `apps/mobile/app.json`.

MIT licensed.
