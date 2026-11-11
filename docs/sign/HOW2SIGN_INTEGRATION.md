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
- **Sign Language Bridge** (Qwen3-VL-2B + LoRA): its authors report BLEU-4
  1.6, chrF 17.4 on their 944-clip How2Sign test set. It was not run here
  (see Limitations).

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
| Sign Language Bridge (authors' report) | 944 | 1.64 | 17.42 | | |

Uni-Sign: model load 12–15 s, peak GPU memory 2.6 GB (RTX 4050 Laptop, bf16),
fits next to the running API. The repeat guard is `no_repeat_ngram_size=3`,
`repetition_penalty=1.2`; it removed one looping output ("you're talking to
me" ×10) and changed little else. The more accurate pose model did not
improve the 3 clips it ran on, so the poor output is the model, not the
keypoints.

Example (sample_001): reference "Avoid wearing height on top or volume on
the sides."; Uni-Sign "It's going to have a smaller amount of hair."

Per-sample predictions: `datasets/how2sign/evaluation/predictions_unisign_*.json`;
summary: `datasets/how2sign/evaluation/results.csv`.

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
- Bridge was not run locally. It needs a newer transformers in another
  environment, about 5 GB of weights, and most of the 6 GB GPU (the API would
  have to stop), with 1.5 GB of system RAM free. Given its published scores
  it was not worth it.
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
# MediaPipe 281 features
& "$m\envs\mediapipe\Scripts\python.exe" apps\api\sign_language\adapters\mediapipe_281.py `
  --dataset-dir datasets\how2sign --model apps\mobile\public\mediapipe\models\holistic_landmarker.task
```
