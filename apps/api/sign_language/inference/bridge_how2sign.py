"""Run Sign Language Bridge (Qwen3-VL-2B + LoRA) on How2Sign manifest samples.

Runs in its own environment (see docs/sign/HOW2SIGN_INTEGRATION.md), never in
the API's. Preprocessing comes from the authors' own code, imported from a
clone of github.com/mamounyosef/sign-language-bridge, not rewritten:

1. Signer box: data_code/signer_cropper.py (MediaPipe pose "full", every 4th
   frame, union of upper-body points, 25 % padding), as in
   11_extract_signer_bboxes.py.
2. Landmarks: RTMPose whole-body on the cropped frames, sampled at most
   20 fps, with the authors' thresholds and temporal clean-up, using the
   functions in data_code/21_extract_landmarks.py. OpenCV replaces decord
   for decoding.
3. Frames, prompts, crop, CLAHE and landmark overlay: Qwen3VLCollator from
   model_training_scripts/qwen3vl_training.py.

Two preprocessing modes:
  trained    crop + CLAHE + landmark overlay, as during training.
  published  crop only. The authors' test script (validate() in
             qwen3vl_training.py) skips CLAHE and the overlay, so their
             published test score was produced this way.

The adapter is kept separate from the base weights. Qwen3-VL-2B ties the
output head to the input embeddings, so merging the head's LoRA would also
change the embeddings, which training never did.

Decoding defaults are the authors' test settings (5 beams, length penalty
0.6, no-repeat 4-grams, repetition penalty 1.1, 32 new tokens). If 5 beams do
not fit in GPU memory the sample is retried with fewer beams, and the beam
count used is recorded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.machinery
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path
from unittest import mock


def _stub_modal() -> None:
    """The authors' modules import Modal (their cloud runner) at import time."""
    modal = mock.MagicMock(name="modal")
    modal.__spec__ = importlib.machinery.ModuleSpec("modal", None)
    sys.modules["modal"] = modal


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def offload_tied_embeddings(model) -> int:
    """Keep the tied 151,936 x 2,048 embedding / output-head weight in CPU memory.

    Frees about 600 MB of GPU memory. Only token lookups and the output head
    (applied to the last position during generation) run on the CPU.
    """
    import torch
    import torch.nn.functional as F

    class CpuEmbedding(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def forward(self, ids):
            return F.embedding(ids.to("cpu"), self.weight).to(ids.device)

    class CpuLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
            self.bias = None
            self.in_features, self.out_features = weight.shape[1], weight.shape[0]

        def forward(self, x):
            return F.linear(x.to("cpu", self.weight.dtype), self.weight).to(x.device, x.dtype)

    inner = model.base_model.model
    head = inner.lm_head
    if head.base_layer.weight.data_ptr() != inner.get_input_embeddings().weight.data_ptr():
        raise RuntimeError("Expected the output head to share the embedding weight.")
    weight = torch.nn.Parameter(head.base_layer.weight.detach().to("cpu"), requires_grad=False)
    inner.model.language_model.embed_tokens = CpuEmbedding(weight)
    head.base_layer = CpuLinear(weight)
    torch.cuda.empty_cache()
    return weight.numel() * weight.element_size()


def share_vision_across_beams(model) -> dict:
    """Encode the video once per generate() call instead of once per beam.

    For beam search transformers copies the video for every beam and runs the
    vision encoder on each copy. The copies are identical, so this encodes one
    and repeats the features, which gives the same result with a fraction of
    the memory. Falls back to the normal path if the layout is not an exact
    repeat.
    """
    import torch

    inner = model.base_model.model
    vision_language = inner.model
    original_expand = inner._expand_inputs_for_generation
    original_features = vision_language.get_image_features
    state = {"copies": 1, "shared": 0}

    def expand(expand_size=1, is_encoder_decoder=False, input_ids=None, **kwargs):
        pixels = kwargs.pop("pixel_values_videos", None)
        input_ids, kwargs = original_expand(expand_size=expand_size, is_encoder_decoder=is_encoder_decoder,
                                            input_ids=input_ids, **kwargs)
        if pixels is not None:
            kwargs["pixel_values_videos"] = pixels
        state["copies"] = expand_size
        return input_ids, kwargs

    def features(pixel_values, image_grid_thw=None):
        copies, rows = state["copies"], image_grid_thw.shape[0]
        if copies > 1 and rows % copies == 0:
            one = image_grid_thw[: rows // copies]
            if (int(one.prod(-1).sum()) == pixel_values.shape[0]
                    and torch.equal(image_grid_thw, one.repeat(copies, 1))):
                embeds, deepstack = original_features(pixel_values, one)
                embeds = torch.cat(embeds, dim=0).repeat(copies, 1)
                deepstack = [layer.repeat(copies, 1) for layer in deepstack]
                sizes = (image_grid_thw.prod(-1) // vision_language.visual.spatial_merge_size ** 2).tolist()
                state["shared"] += 1
                return torch.split(embeds, sizes), deepstack
        return original_features(pixel_values, image_grid_thw)

    inner._expand_inputs_for_generation = expand
    vision_language.get_image_features = features
    return state


def signer_bbox(cropper, video: Path) -> tuple[tuple[int, ...] | None, dict]:
    started = time.perf_counter()
    result = cropper.compute_bbox(str(video))
    stats = {"bbox_seconds": round(time.perf_counter() - started, 2),
             "bbox_detection_rate": round(result.detection_rate, 3), "bbox_failed": result.failed}
    if result.failed:
        return None, stats
    return (result.x1, result.y1, result.x2, result.y2, result.frame_width, result.frame_height), stats


def extract_landmarks(landmarks_module, video: Path, bbox) -> tuple[dict, dict]:
    """21_extract_landmarks._process_one, with OpenCV decoding instead of decord."""
    import cv2
    import numpy as np

    lm = landmarks_module
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    count = len(frames)
    step = max(1, math.ceil(fps / lm.TARGET_SAMPLE_FPS))
    indices = list(range(0, count, step))
    if not indices or indices[-1] != count - 1:
        indices.append(count - 1)
    rgb = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in indices])
    cropped, *_ = lm._compute_crop(rgb, bbox)
    n, crop_h, crop_w = cropped.shape[:3]
    bgr = np.ascontiguousarray(cropped[:, :, :, ::-1])
    pose = np.full((n, 6, 2), np.nan, dtype=np.float32)
    left = np.full((n, 21, 2), np.nan, dtype=np.float32)
    right = np.full((n, 21, 2), np.nan, dtype=np.float32)
    for t in range(n):
        pose[t], left[t], right[t] = lm._run_rtmpose_on_frame_bgr(bgr[t], crop_h, crop_w)
    pose, left, right = lm._postprocess(pose, left, right)
    stats = {
        "landmark_frames": n, "landmark_seconds": round(time.perf_counter() - started, 2),
        "crop_w": int(crop_w), "crop_h": int(crop_h),
        "left_hand_ratio": round(float(np.mean(~np.all(np.isnan(left), axis=(1, 2)))), 3),
        "right_hand_ratio": round(float(np.mean(~np.all(np.isnan(right), axis=(1, 2)))), 3),
    }
    return {"frame_indices": np.asarray(indices, dtype=np.uint16), "pose": pose,
            "left_hand": left, "right_hand": right}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bridge-repo", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True, help="Qwen3-VL-2B-Instruct folder")
    parser.add_argument("--adapter", type=Path, required=True, help="sign-language-bridge adapter/ folder")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["trained", "published"], choices=["trained", "published"])
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--pose-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--debug-oom", action="store_true")
    parser.add_argument("--no-share-vision", dest="share_vision", action="store_false",
                        help="Run the vision encoder once per beam, as transformers does by default.")
    parser.add_argument("--bf16-adapter", action="store_true",
                        help="Cast the float32 adapter to bfloat16 to save activation memory.")
    parser.add_argument("--no-offload-embeddings", dest="offload_embeddings", action="store_false",
                        help="Keep the tied embedding / output-head weight on the GPU.")
    parser.add_argument("--save-previews", action="store_true",
                        help="Save the middle model-input frame of each clip to evaluation/previews/.")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torchvision  # noqa: F401
    import transformers  # noqa: F401
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _stub_modal()
    sys.path.insert(0, str(args.bridge_repo / "data_code"))
    sys.path.insert(0, str(args.bridge_repo / "model_training_scripts"))
    import qwen3vl_training as training
    from signer_cropper import SignerCropper

    landmarks = _load_module("bridge_extract_landmarks", args.bridge_repo / "data_code" / "21_extract_landmarks.py")
    from rtmlib import RTMPose, Wholebody

    mode_cfg = Wholebody.MODE[landmarks.RTMPOSE_MODE]
    landmarks._WORKER_MODEL = RTMPose(mode_cfg["pose"], model_input_size=mode_cfg["pose_input_size"],
                                      to_openpose=False, backend="onnxruntime", device=args.pose_device)
    bbox_script = _load_module("bridge_extract_bboxes", args.bridge_repo / "data_code" / "11_extract_signer_bboxes.py")
    cropper = SignerCropper(sample_every_n=bbox_script.SAMPLE_EVERY_N, padding_ratio=bbox_script.PADDING_RATIO,
                            min_detection_confidence=bbox_script.MIN_DETECTION_CONFIDENCE,
                            model_variant=bbox_script.MODEL_VARIANT)

    config = dict(training.CONFIG)
    samples_dir = args.dataset_dir / "samples"
    with (samples_dir / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = [row for row in csv.DictReader(handle) if not args.samples or row["sample_id"] in args.samples]

    # Preprocessing runs before the model loads so pose models and Qwen3-VL never share memory.
    prepared = []
    for row in manifest:
        video = (args.dataset_dir / row["video_path"]).resolve()
        bbox, bbox_stats = signer_bbox(cropper, video)
        lms, lm_stats = extract_landmarks(landmarks, video, bbox)
        prepared.append((row, video, bbox, lms, {**bbox_stats, **lm_stats}))
        print(f"{row['sample_id']}: bbox {bbox} {lm_stats}", flush=True)
    del cropper
    landmarks._WORKER_MODEL = None

    # On Windows the driver silently spills into system RAM instead of failing, which
    # stalls generation; a hard cap turns that into an OutOfMemoryError we can handle.
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.base_model))
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(str(args.base_model), dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", device_map="cuda")
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    # The adapter is stored in float32 and PEFT casts each layer input to it, which
    # doubles activation memory. Under bfloat16 autocast (as in training) the LoRA
    # matmuls run in bfloat16 either way, so bfloat16 weights give the same math.
    if args.bf16_adapter:
        model.to(torch.bfloat16)
    model.eval()
    offloaded = offload_tied_embeddings(model) if args.offload_embeddings else 0
    beam_sharing = share_vision_across_beams(model) if args.share_vision else None
    print(f"  GPU memory after load: {torch.cuda.memory_allocated() // 2**20} MB "
          f"({offloaded // 2**20} MB of embeddings on CPU)", flush=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    collator = training.Qwen3VLCollator(processor, config, is_training=False)

    out_dir = args.dataset_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        predictions = []
        for row, video, bbox, lms, prep_stats in prepared:
            sample = {"vid": row["sentence_name"], "text": row["english_text"], "file_path": str(video),
                      "bbox": bbox, "landmarks": lms}
            messages = collator._build_messages(sample, include_assistant=False)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            prep_started = time.perf_counter()
            _, videos, video_kwargs = process_vision_info(messages, image_patch_size=16,
                                                          return_video_kwargs=True, return_video_metadata=True)
            clips, metadata = (list(part) for part in zip(*videos))
            clips = collator._apply_signer_crop(clips, [bbox] * len(clips))
            if mode == "trained":
                clips = collator._apply_clahe(clips)
                clips = collator._apply_landmark_overlay(clips, [lms] * len(clips))
            if args.save_previews:
                from PIL import Image

                preview_dir = out_dir / "previews"
                preview_dir.mkdir(exist_ok=True)
                middle = clips[0][clips[0].shape[0] // 2].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
                Image.fromarray(middle).save(preview_dir / f"{row['sample_id']}_{mode}.png")
            inputs = processor(text=[text], videos=clips, video_metadata=metadata, return_tensors="pt",
                               padding=True, do_resize=False, **video_kwargs)
            frames = int(clips[0].shape[0])
            prep_seconds = time.perf_counter() - prep_started
            print(f"  {row['sample_id']}: {frames} frames of {tuple(clips[0].shape[-2:])}, "
                  f"{inputs['input_ids'].shape[1]} prompt tokens", flush=True)

            beams, output, error = args.beams, None, None
            started = time.perf_counter()
            while output is None and beams >= 1:
                try:
                    batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    if beams > 1 and batch.get("video_grid_thw") is not None:
                        # Same per-frame split as the authors' validate() for beam search.
                        grid = torch.repeat_interleave(batch["video_grid_thw"], batch["video_grid_thw"][:, 0], dim=0)
                        grid[:, 0] = 1
                        batch["video_grid_thw"] = grid.clone()
                    if beam_sharing is not None:
                        beam_sharing.update(copies=1, shared=0)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model.generate(**batch, max_new_tokens=32, num_beams=beams, length_penalty=0.6,
                                                no_repeat_ngram_size=4, repetition_penalty=1.1, do_sample=False,
                                                temperature=None, top_p=None, top_k=None, use_cache=True)
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError as oom:
                    error = f"out of GPU memory with {beams} beams"
                    output = None
                    print(f"  {row['sample_id']}: {error}: {str(oom).splitlines()[0][:160]}", flush=True)
                    if args.debug_oom:
                        import traceback

                        print("".join(traceback.format_exc().splitlines(keepends=True)[-14:]), flush=True)
                    del oom
                    beams = beams - 2 if beams > 2 else beams - 1
                finally:
                    batch = None
                    # The failed attempt's tensors stay alive in traceback cycles until collected.
                    gc.collect()
                    torch.cuda.empty_cache()
            seconds = time.perf_counter() - started
            if output is None:
                hypothesis = ""
            else:
                trimmed = output[0][inputs["input_ids"].shape[1]:]
                hypothesis = processor.decode(trimmed, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False).strip()
            predictions.append({
                "sample_id": row["sample_id"], "sentence_name": row["sentence_name"],
                "model": "sign-language-bridge step 4610", "preprocessing": mode,
                "reference": row["english_text"], "prediction": hypothesis,
                "beams_used": beams if output is not None else 0, "error": error if output is None else None,
                "beams_note": error if output is not None and error else None,
                "vision_encoded_once": bool(beam_sharing and beam_sharing["shared"]),
                "decoded_frames": frames, "visual_tokens": int(inputs["video_grid_thw"].prod(dim=1).sum() // 4),
                "preprocess_seconds": round(prep_seconds, 2), "generate_seconds": round(seconds, 2),
                **prep_stats,
            })
            print(f"[{mode}] {row['sample_id']} ({beams} beams): {hypothesis!r}", flush=True)

        summary = {
            "model": "Qwen3-VL-2B-Instruct + sign-language-bridge adapter (unmerged)",
            "preprocessing": mode, "model_load_seconds": round(load_seconds, 2),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "gpu": torch.cuda.get_device_name(0), "requested_beams": args.beams,
            "gpu_memory_fraction": args.gpu_memory_fraction, "embeddings_on_cpu": args.offload_embeddings,
            "vision_shared_across_beams": args.share_vision, "adapter_dtype": "bfloat16" if args.bf16_adapter else "float32",
            "decoding": {"length_penalty": 0.6, "no_repeat_ngram_size": 4, "repetition_penalty": 1.1,
                         "max_new_tokens": 32},
            "video_fps": config["video_fps"], "video_max_pixels": config["video_max_pixels"],
            "predictions": predictions,
        }
        (out_dir / f"predictions_bridge_{mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
