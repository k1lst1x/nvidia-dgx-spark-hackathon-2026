# Third-party notices

## YOLOX Nano

GuideLens bundles the official `yolox_nano.onnx` model from the YOLOX 0.1.1rc0 release:

- Source: https://github.com/Megvii-BaseDetection/YOLOX
- Model: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
- SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- License: Apache License 2.0

Copyright (c) 2021 Megvii, Inc. All rights reserved.

The YOLOX source and model are provided under the Apache License 2.0. A copy of that license is available at https://www.apache.org/licenses/LICENSE-2.0.

## ONNX Runtime

GuideLens uses Microsoft ONNX Runtime for on-device inference on iOS and Android.

- Source: https://github.com/microsoft/onnxruntime
- License: MIT

## ASL isolated-word Transformer

GuideLens includes an ONNX conversion of the 250-class landmark Transformer
checkpoint published by AWAINIA Aymen:

- Source: https://github.com/aymenwnia/ASL-translator
- Original checkpoint: `models/sign_transformer_best.pth`
- ONNX SHA-256: `0b174222850924623cbe6c93dc4232dcdd36155181c4638031f4d41ee1248a0f`
- Code and checkpoint repository license: MIT
- Training data: PopSign ASL v1.0 / Google Isolated Sign Language Recognition,
  CC BY 4.0 and subject to the competition rules

Copyright (c) 2026 AWAINIA Aymen.

This model recognizes isolated signs from a fixed 250-word vocabulary. The
dataset authors state that the dataset is not intended as a general ASL-to-English
translation or natural-language interface. GuideLens therefore labels this
feature experimental and does not claim continuous-ASL translation accuracy.

## Ultralytics YOLO11

GuideLens bundles `yolo11n.onnx` (web object detection) and `yolo11n.pt`
(optional server tracking), and uses the Ultralytics package on the server.

- Source: https://github.com/ultralytics/ultralytics
- License: AGPL-3.0, or an Ultralytics Enterprise licence

AGPL-3.0 applies to distributing these weights and to offering the server
that uses them over a network. Review it, or obtain an Enterprise licence,
before any distribution beyond the hackathon.

## MediaPipe

GuideLens uses `@mediapipe/tasks-vision` 1.2.0 and serves the hand, face,
pose and holistic landmarker models downloaded at build time from Google.

- Source: https://github.com/google-ai-edge/mediapipe
- Package license: Apache License 2.0
- Model terms: see each model card; not independently verified by this project

The 250-word sign model's input layout was checked against its upstream
repository; see `docs/sign/SIGN_PIPELINE_AUDIT.md`.
