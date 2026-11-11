# How2Sign evaluation of pretrained sign translators

**Status:** evaluation only. Nothing here is used by the app, and the live
sign pipeline (`asl_word_transformer.onnx`, the 281-feature path) is unchanged.
No model was trained or fine-tuned.

**Licence:** How2Sign and the Uni-Sign checkpoint are CC BY-NC 4.0
(non-commercial). Clips, keypoints and model outputs stay in `datasets/`,
which git ignores. Model weights and environments live outside the repo in
`C:\Users\zafar\sign-models`.

## Recommendation

Do not integrate Uni-Sign or Sign Language Bridge into VisionMate. Neither
translates How2Sign sentences usefully:

- **Uni-Sign** (pose only, How2Sign checkpoint): BLEU-4 2.0, chrF 21.4 on 8
  clips. Fluent English that usually gets the topic, not the sentence.
- **Sign Language Bridge** (Qwen3-VL-2B + LoRA): BLEU-4 0.95, chrF 17.0 on
  the same 8 clips, in line with its authors' 1.6 / 17.4 on 944 test clips.
  Half its outputs are one template ("So, you want to make sure that…"),
  and on this laptop it needs about 15 s per clip and nearly all 6 GB of GPU
  memory.

Keep the current word-level pipeline. Open-vocabulary ASL-to-English is not
reliable enough to speak to a user. The practical path is a fixed vocabulary
of words or phrases, recorded with the 281-feature recorder and trained on
the existing `VisionMateTransformer`, with an LLM forming sentences from
recognised words and low-confidence results rejected. If open translation is
wanted later, Uni-Sign is the better of the two, and its output must be shown
as an unverified guess.

## Results

8 How2Sign validation clips, none seen by either model (below). Scores are
sacreBLEU corpus BLEU-4 and chrF. On so few sentences they are noisy; the
gap between them and a usable system is not.

| Run | Clips | BLEU-4 | chrF | Translate (median) | Pose (per frame, CPU) |
|---|---|---|---|---|---|
| Uni-Sign, lightweight pose, Uni-Sign decoding | 8 | 1.43 | 20.90 | 1.18 s | 163 ms |
| Uni-Sign, lightweight pose, repeat guard | 8 | 2.01 | 21.41 | 0.72 s | (reused) |
| Uni-Sign, lightweight pose, repeat guard (same 3 clips) | 3 | 3.60 | 23.20 | | |
| Uni-Sign, performance pose, repeat guard | 3 | 3.44 | 23.57 | 1.57 s | ≈ 390 ms |
| Sign Language Bridge, crop + CLAHE + landmarks (as trained) | 8 | 0.95 | 17.02 | 15.1 s | ≈ 80 ms |
| Sign Language Bridge, crop only (as in the authors' test) | 8 | 0.94 | 16.79 | 14.9 s | ≈ 80 ms |
| Sign Language Bridge (authors' report) | 944 | 1.64 | 17.42 | | |

Uni-Sign: model load 12–15 s, peak GPU memory 2.6 GB (RTX 4050 Laptop, bf16),
fits next to the running API. The repeat guard is `no_repeat_ngram_size=3`,
`repetition_penalty=1.2`; it removed one looping output ("you're talking to
me" ×10) and changed little else. The more accurate pose model did not
improve the 3 clips it ran on, so the poor output is the model, not the
keypoints.

Example (sample_001): reference "Avoid wearing height on top or volume on
the sides."; Uni-Sign "It's going to have a smaller amount of hair."

Sign Language Bridge:

- **Preprocessing makes no difference.** The authors' test script
  (`validate()` in `qwen3vl_training.py`) applies only the signer crop, not
  the CLAHE and landmark overlay used in training, so their published score
  has a train/test mismatch. Running it both ways gives the same quality
  (0.95 vs 0.94 BLEU-4).
- **Generic output.** 8 of 16 outputs start "So, you want to make sure
  that…" and none matches its reference. sample_007 ("Is it 2008?") gave
  "This is a 720 turn.", "I'm going to show you how to do that." and "It's
  going to look like this." under three setups that differ only in float
  rounding (see below): the answer depends on noise, not on the signing.
- **Cost.** Model load 12 s, peak GPU memory 5.35 GB. Per clip: about 1.6 s
  signer box, 5 s landmarks (CPU), 1–7 s video preparation, 6–19 s
  generation. It cannot run next to other GPU work on a 6 GB card.
- **Beams.** The authors decode with 5 beams. On 6 GB that fits only the two
  shortest clips; 3 clips used 3 beams and 3 used 1 (recorded per sample as
  `beams_used`). The short clips were no better than the long ones.

Example (sample_001): Bridge "The first thing that you want to do is make
sure that your hair is completely dry."

Per-sample predictions: `datasets/how2sign/evaluation/predictions_*.json`;
model-input frames: `datasets/how2sign/evaluation/previews/`; summary:
`datasets/how2sign/evaluation/results.csv`.

## Samples

`datasets/how2sign/samples/manifest.csv`: 8 clips from the official
validation split (`how2sign_val.csv`, `val_rgb_front_clips.zip`), matched by
`SENTENCE_NAME` = clip file name, never by position.

Selection (`apps/api/sign_language/evaluation/select_how2sign_samples.py`):

- Excluded any clip or sentence in Sign Language Bridge's data files
  (v2–v7 train/val/test; its own split puts 709 official validation clips in
  training) or in Uni-Sign's How2Sign training labels: 1033 clips removed.
- Kept 1.3–8 s and at most 5 words per second (Bridge's training range): 502
  removed. 2 rows have no clip in the zip.
- One clip per source video; video length within 0.5 s of `END − START`.
- 204 candidates remained; 8 were copied as `samples/rgb/sample_NNN.mp4`.

All are 1280×720 at 24 fps except sample_004 (60 fps).

## Keypoint formats

| | How2Sign OpenPose (file A) | Uni-Sign input | VisionMate `visionmate-281-v1` |
|---|---|---|---|
| Source | OpenPose JSON per frame | RTMPose COCO-WholeBody, 133 points | MediaPipe Holistic |
| Points | body 25, hands 2×21, face 70 | body, 2×21 hands, face | 2×21 hands, 33 pose, 52 blendshapes, 4 presence flags |
| Coordinates | pixels, with confidence | x/W, y/H, with confidence | normalized x, y, z; no confidence |
| Available here | no (not downloaded) | extracted with rtmlib | extracted with MediaPipe |

The formats are not interchangeable. Uni-Sign needs per-point confidence,
which MediaPipe does not give; `apps/api/unisign_adapter.py` already refuses
MediaPipe recordings for that reason, so no new adapter was written.

## How each pipeline was run

**Uni-Sign** (`apps/api/sign_language/inference/unisign_how2sign.py`), in
its own Python 3.10 environment (torch 2.1.1 CUDA 12.1, transformers 4.40):

1. rtmlib `Wholebody` keypoints per frame, divided by width and height, first
   person only, saved as `samples/keypoints/sample_NNN_rtmw_<mode>.npz`. This
   matches Uni-Sign's `demo/pose_extraction.py`.
2. Uni-Sign's own dataset code and collate, unchanged, with `--dataset How2Sign`
   (English prompt). Checkpoint loads with no missing or unexpected keys.
3. mT5 beam search, 4 beams, up to 100 new tokens.

DeepSpeed and decord are imported by Uni-Sign's training utilities but unused
for inference; the script inserts small stand-ins after importing
transformers.

**Sign Language Bridge** (`apps/api/sign_language/inference/bridge_how2sign.py`),
in its own Python 3.12 environment with the authors' versions (torch 2.10
CUDA 12.8, transformers 4.57.3, peft 0.18.1, qwen-vl-utils 0.0.14; PyAV
below 14 for torchvision's video reader). Preprocessing is the authors' own
code, imported from their repository rather than rewritten:

1. Signer box: `data_code/signer_cropper.py` with the settings of
   `11_extract_signer_bboxes.py` (MediaPipe pose "full", every 4th frame,
   25 % padding).
2. Landmarks: the functions of `21_extract_landmarks.py` (RTMPose
   "performance" on the crop, at most 20 fps, their thresholds, spike
   rejection, gap filling and smoothing). OpenCV replaces decord.
3. Frames, prompts, crop, CLAHE, overlay: their `Qwen3VLCollator` (20 fps,
   their pixel limits, system prompt "You are a sign language translator.").
4. Their test decoding: 5 beams, length penalty 0.6, no repeated 4-grams,
   repetition penalty 1.1, 32 new tokens.

Changes needed to fit 6 GB, all recorded in the prediction files:

- The adapter is not merged. Qwen3-VL-2B ties the output head to the input
  embeddings, so merging the head's LoRA would also change the embeddings
  (PEFT warns about exactly this).
- The tied embedding / output-head weight (593 MB) is kept in CPU memory.
  This reproduced the unmodified run's output exactly.
- The video is encoded once and its features repeated for each beam, instead
  of once per beam. Same computation, but different GPU kernels, so float
  rounding differs and beam search can pick another sentence.
- A GPU memory cap (90 %) so Windows does not silently spill into system RAM
  (that made one clip take about 20 minutes). A clip that does not fit with
  5 beams is retried with 3, then 1.
- Casting the float32 adapter to bfloat16 would save more memory but changed
  outputs, so it was not used.

**MediaPipe 281 features** (`apps/api/sign_language/adapters/mediapipe_281.py`),
in its own Python 3.12 environment with mediapipe:

- Holistic landmarker, VIDEO mode, the app's `holistic_landmarker.task` and
  thresholds (0.4). Python has no "presence" thresholds; landmark confidence
  0.4 is used instead.
- Same frame-keeping rule as `SignRecorder` (keep a frame once 50 ms have
  passed), so 24 fps video gives 12 frames per second, as the app would.
- `build_sign_feature_frame` reproduces `signFeatures.ts`;
  `apps/api/tests/test_sign_features_281.py` checks it against the pinned test
  vector and the TypeScript blendshape list.
- Output: `samples/keypoints/sample_NNN_visionmate281.npz` (features,
  timestamps, schema, text). About 50 ms per frame on CPU. Pose and face found
  in every frame; left hand in 81–100 %, right hand in 93–100 % of frames.

There is no trained model for these features yet, so they could not be
scored. How2Sign has English sentences, not glosses, so it cannot provide
gloss targets for the CTC model.

## Limitations and errors

- 8 clips (3 for the pose comparison). Enough to rule the models out for
  VisionMate, not to rank them precisely.
- Bridge ran with 5 beams on only 2 of 8 clips (see Results), on an RTX 4050
  instead of the authors' A100, without FlashAttention or Liger kernels.
  Float differences change individual outputs, but every setup tried gave
  the same kind of generic sentence.
- The performance-pose run stalled on sample_004 (the 60 fps clip) for over an
  hour with low free memory and was stopped. The cause was not found.
- RTMPose ran on CPU onnxruntime. GPU onnxruntime would be needed for
  anything close to real time.
- Official OpenPose keypoints (file A) were not used, so rtmlib keypoints were
  not compared with them.
- MediaPipe handedness is the label as seen in the image and is not verified
  against a real signer.

## Reproducing

```powershell
$m = "C:\Users\zafar\sign-models"
# Samples (any Python with opencv); --clips is the folder of extracted val clips
python apps\api\sign_language\evaluation\select_how2sign_samples.py `
  --csv datasets\how2sign\raw\original_annotations\how2sign_val.csv `
  --clips <val_rgb_front_clips folder> --out datasets\how2sign --count 8
# Uni-Sign
& "$m\envs\unisign\Scripts\python.exe" apps\api\sign_language\inference\unisign_how2sign.py `
  --unisign-repo "$m\repos\Uni-Sign" --checkpoint "$m\weights\how2sign_pose_only_slt.pth" `
  --mt5 "$m\weights\mt5-base" --dataset-dir datasets\how2sign --pose-mode lightweight `
  --no-repeat-ngram 3 --repetition-penalty 1.2
# Sign Language Bridge (weights: Qwen/Qwen3-VL-2B-Instruct, mamounyosef/sign-language-bridge adapter/)
& "$m\envs\bridge\Scripts\python.exe" apps\api\sign_language\inference\bridge_how2sign.py `
  --bridge-repo "$m\repos\sign-language-bridge" --base-model "$m\weights\Qwen3-VL-2B-Instruct" `
  --adapter "$m\weights\sign-language-bridge\adapter" --dataset-dir datasets\how2sign --save-previews
# MediaPipe 281 features
& "$m\envs\mediapipe\Scripts\python.exe" apps\api\sign_language\adapters\mediapipe_281.py `
  --dataset-dir datasets\how2sign --model apps\mobile\public\mediapipe\models\holistic_landmarker.task
```
