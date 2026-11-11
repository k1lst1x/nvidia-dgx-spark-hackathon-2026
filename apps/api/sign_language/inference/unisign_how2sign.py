"""Run Uni-Sign's pretrained How2Sign pose-only translator on manifest samples.

Runs in its own environment (see docs/sign/HOW2SIGN_INTEGRATION.md), never in
the API's. Steps per sample, following Uni-Sign's demo/online_inference.py:

1. RTMPose whole-body keypoints with rtmlib (COCO-WholeBody, 133 points),
   divided by frame width and height. Saved to samples/keypoints/ so the
   video is not reprocessed.
2. Uni-Sign's own load_part_kp and collate (unchanged), the official
   checkpoint, bfloat16, beam search with 4 beams, up to 100 new tokens.

DeepSpeed and decord are imported by Uni-Sign's training utilities but not
used on this path; small stand-ins avoid installing them on Windows.

Usage (from the Uni-Sign env):
  python unisign_how2sign.py --unisign-repo C:/.../Uni-Sign \
      --checkpoint C:/.../how2sign_pose_only_slt.pth --mt5 C:/.../mt5-base \
      --dataset-dir datasets/how2sign
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path


def _stub_training_only_modules() -> None:
    deepspeed = types.ModuleType("deepspeed")
    comm = types.ModuleType("deepspeed.comm")
    accelerator = types.ModuleType("deepspeed.accelerator")
    accelerator.get_accelerator = lambda: None
    deepspeed.comm = comm
    deepspeed.accelerator = accelerator
    decord = types.ModuleType("decord")
    decord.VideoReader = None
    decord.cpu = None
    sys.modules.update({"deepspeed": deepspeed, "deepspeed.comm": comm,
                        "deepspeed.accelerator": accelerator, "decord": decord})


def extract_pose(video: Path, wholebody) -> tuple[dict, dict]:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    keypoints, scores, people = [], [], []
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        points, confidence = wholebody(frame)
        people.append(len(points))
        if len(points) == 0:
            # No person found: zero points and zero confidence for this frame.
            points, confidence = np.zeros((1, 133, 2)), np.zeros((1, 133))
        keypoints.append(points[:1] / np.array([width, height])[None, None])
        scores.append(confidence[:1])
    capture.release()
    elapsed = time.perf_counter() - started
    stats = {
        "frames": len(keypoints),
        "pose_seconds": round(elapsed, 3),
        "pose_ms_per_frame": round(1000 * elapsed / max(1, len(keypoints)), 1),
        "frames_without_person": sum(1 for count in people if count == 0),
        "frames_with_several_people": sum(1 for count in people if count > 1),
    }
    return {"keypoints": keypoints, "scores": scores}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unisign-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mt5", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pose-mode", default="lightweight", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--pose-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-repeat-ngram", type=int, default=0,
                        help="Block repeated n-grams while decoding (0 keeps Uni-Sign's default decoding).")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--reuse-keypoints", action="store_true",
                        help="Load keypoints saved by an earlier run instead of running pose estimation again.")
    parser.add_argument("--samples", nargs="*", default=None, help="Only run these sample ids, e.g. sample_001.")
    args = parser.parse_args()
    decoding = "default" if args.no_repeat_ngram == 0 and args.repetition_penalty == 1.0 else (
        f"norepeat{args.no_repeat_ngram}_rep{args.repetition_penalty:g}")

    # transformers probes for optional packages on first import; import it
    # before the stand-ins exist so it does not mistake them for real installs.
    import torchvision  # noqa: F401
    import transformers  # noqa: F401

    _stub_training_only_modules()
    sys.path.insert(0, str(args.unisign_repo))
    import config  # Uni-Sign's config module

    config.mt5_path = str(args.mt5)
    import numpy as np
    import torch
    from rtmlib import Wholebody

    import utils as unisign_utils
    from datasets import S2T_Dataset_online
    from models import Uni_Sign

    unisign_args = unisign_utils.get_args_parser().parse_args([
        "--dataset", "How2Sign", "--task", "SLT", "--finetune", str(args.checkpoint), "--seed", str(args.seed),
    ])
    unisign_utils.set_seed(args.seed)

    samples_dir = args.dataset_dir / "samples"
    with (samples_dir / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = [row for row in csv.DictReader(handle) if not args.samples or row["sample_id"] in args.samples]

    wholebody = Wholebody(to_openpose=False, mode=args.pose_mode, backend="onnxruntime", device=args.pose_device)

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Uni_Sign(args=unisign_args)
    state = torch.load(args.checkpoint, map_location="cpu")["model"]
    result = model.load_state_dict(state, strict=True)
    model.cuda().eval().to(torch.bfloat16)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    keypoint_dir = samples_dir / "keypoints"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for row in manifest:
        video = args.dataset_dir / row["video_path"]
        keypoint_path = keypoint_dir / f"{row['sample_id']}_rtmw_{args.pose_mode}.npz"
        if args.reuse_keypoints and keypoint_path.exists():
            saved = np.load(keypoint_path)
            pose = {"keypoints": list(saved["keypoints"]), "scores": list(saved["scores"])}
            pose_stats = {"frames": len(pose["keypoints"]), "pose_reused": True}
        else:
            pose, pose_stats = extract_pose(video, wholebody)
            np.savez_compressed(keypoint_path, keypoints=np.stack(pose["keypoints"]).astype(np.float32),
                                scores=np.stack(pose["scores"]).astype(np.float32))

        dataset = S2T_Dataset_online(args=unisign_args)
        dataset.rgb_data = str(video)
        dataset.pose_data = pose
        src_input, tgt_input = dataset.collate_fn([dataset[0]])
        for key, value in src_input.items():
            if isinstance(value, torch.Tensor):
                src_input[key] = value.to(torch.bfloat16).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            encoded = model(src_input, tgt_input)
            output = model.mt5_model.generate(
                inputs_embeds=encoded["inputs_embeds"], attention_mask=encoded["attention_mask"],
                max_new_tokens=100, num_beams=4,
                no_repeat_ngram_size=args.no_repeat_ngram, repetition_penalty=args.repetition_penalty,
            )
        torch.cuda.synchronize()
        text = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        predictions.append({
            "sample_id": row["sample_id"], "sentence_name": row["sentence_name"], "model": "uni-sign-how2sign-pose-only",
            "reference": row["english_text"], "prediction": text,
            "inference_seconds": round(time.perf_counter() - started, 3), "keypoints_path": str(keypoint_path),
            **pose_stats,
        })
        print(f"{row['sample_id']}: {text!r}")

    out_dir = args.dataset_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "Uni-Sign how2sign_pose_only_slt.pth", "pose": f"rtmlib Wholebody {args.pose_mode} on {args.pose_device}",
        "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        "model_load_seconds": round(load_seconds, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0), "seed": args.seed,
        "no_repeat_ngram_size": args.no_repeat_ngram, "repetition_penalty": args.repetition_penalty,
        "predictions": predictions,
    }
    suffix = args.pose_mode if decoding == "default" else f"{args.pose_mode}_{decoding}"
    (out_dir / f"predictions_unisign_{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
