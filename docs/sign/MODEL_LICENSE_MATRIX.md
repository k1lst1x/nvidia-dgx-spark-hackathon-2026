# Model and licence matrix

**How to read this:** "In repo" means the licence text or notice is in this
repository (`THIRD_PARTY_NOTICES.md`, a file header or `package.json`).
"Not verified" means it was **not** checked against the upstream source for
this audit; external repositories and model pages were not readable from the
audit environment. Treat every "not verified" as an open task before any
public or commercial release. This is an engineering inventory, not legal
advice.

## Shipped in the app today

| Component | Used for | Code licence | Weights licence | Training data | Status |
|---|---|---|---|---|---|
| MediaPipe Tasks Vision 1.2.0 (`@mediapipe/tasks-vision`) | landmark runtime | Apache-2.0 (npm package) | hand / face / pose / holistic `.task` models: **not verified** (Google model cards) | Google internal | Models are downloaded at build time from Google storage and served by the app. Redistribution terms of the `.task` files: not verified. |
| ASL isolated-word Transformer | 250-word recognizer | MIT (in repo notice) | MIT (in repo notice) | PopSign ASL v1.0 / Google ISLR: CC BY 4.0 **and competition rules** (in repo notice) | Attribution is in `THIRD_PARTY_NOTICES.md`. Whether the competition rules allow commercial use of a model trained on the data: **not verified**. Dataset authors say it is not for general ASL translation. |
| Fingerspelling classifier | 24 letters | this project | this project | "American Sign Language Image Dataset" zip named in the README: **source and licence not recorded** | Must be identified before any redistribution. |
| ONNX Runtime Web | inference runtime | MIT (in repo notice) | n/a | n/a | OK |
| YOLOX Nano | object detection (native) | Apache-2.0 (in repo notice) | Apache-2.0 | COCO | OK. Not a sign model. |
| YOLO11n (`yolo11n.onnx`, `yolo11n.pt`) | object detection (web, server) | **AGPL-3.0** or Ultralytics Enterprise | same | COCO | Not a sign model, but it is bundled: AGPL obligations apply to distribution. Recorded in `THIRD_PARTY_NOTICES.md`. |
| Vision LLM for `/v1/interpret-sign` | cloud sign decoding | provider terms | model licence of the configured model (for example Qwen): **not verified** | n/a | Frames leave the device when Groq is used. |

## In code but not operational

| Component | Code licence | Weights licence | Data | Notes |
|---|---|---|---|---|
| VisionMate CTC Transformer | this project | none exist | none | Licence will follow the data used to train it. |
| YOLO11 sign detector (`yolo11_sign_language.pt`) | Ultralytics AGPL-3.0 | **unknown: weights and their source are not in the repo** | unknown | Do not ship until the source is identified. |
| Uni-Sign | **not verified** | CC BY-NC 4.0 per the project specification: **non-commercial** | How2Sign and others: research terms, **not verified** | Not vendored. Research use only unless cleared. Needs `mt5-base` (Apache-2.0 per its model card: not verified). |

## Named in the specification, not in the code

| Component | Stated in the specification | Verified here |
|---|---|---|
| OpenHands | Apache-2.0 framework; checkpoint and dataset terms vary | No. Checkpoints are per-dataset; ASL ones are trained on WLASL-family data, whose terms need checking. |
| Sign Language Bridge | Apache-2.0 code and adapter; base model and data terms upstream | No. Base model (Qwen3-VL) licence and training-data terms not checked. |
| Sign Language Translator | Apache-2.0 framework | No. |

## Actions

1. Done: YOLO11 and MediaPipe are in `THIRD_PARTY_NOTICES.md`; the unused Pinto models were removed.
2. The 250-word model's MIT licence was confirmed in its upstream repository.
3. Record the source and licence of the fingerspelling dataset.
4. Read the Google ISLR competition rules for commercial use of derived models.
5. Keep Uni-Sign behind its flag and out of any commercial build.
6. Verify each "not verified" cell against the upstream page before release.
