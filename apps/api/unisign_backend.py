"""Optional loader for the official Uni-Sign How2Sign pose-only checkpoint.

No Uni-Sign code or weights are vendored.  The adapter expects the official
repository plus independently obtained, license-compliant model assets. Model
construction and state-dict loading intentionally use the repository's
unmodified ``Uni_Sign`` architecture with ``strict=True``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from unisign_adapter import UniSignAdapter, UniSignCompatibilityError


class UniSignBackendError(RuntimeError):
    pass


class UniSignBackend:
    def __init__(self) -> None:
        self.enabled = os.getenv("UNISIGN_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.repo = Path(os.getenv("UNISIGN_REPO", "/opt/uni-sign"))
        self.checkpoint = Path(os.getenv("UNISIGN_CHECKPOINT", "/models/how2sign_pose_only_slt.pth"))
        self.mt5_path = Path(os.getenv("UNISIGN_MT5_PATH", "/models/mt5-base"))
        self.device = os.getenv("UNISIGN_DEVICE", "cuda")
        self.allow_presence_proxy = os.getenv("UNISIGN_ALLOW_MEDIAPIPE_PRESENCE_PROXY", "false").lower() in {"1", "true", "yes"}
        self._model: Any | None = None
        self._torch: Any | None = None
        self.load_error: str | None = None

    def status(self) -> dict[str, object]:
        assets_ready = self.repo.is_dir() and self.checkpoint.is_file() and self.mt5_path.is_dir()
        return {
            "enabled": self.enabled,
            "assets_ready": assets_ready,
            "loaded": self._model is not None,
            "experimental_mediapipe_proxy": self.allow_presence_proxy,
            "error": self.load_error,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.enabled:
            raise UniSignBackendError("Uni-Sign is disabled")
        missing = [str(path) for path in (self.repo, self.checkpoint, self.mt5_path) if not path.exists()]
        if missing:
            raise UniSignBackendError("Missing Uni-Sign assets: " + ", ".join(missing))
        try:
            import torch
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise UniSignBackendError("UNISIGN_DEVICE requests CUDA but CUDA is unavailable")
            # Uni-Sign imports config.mt5_path by value during models import.
            # Override only the local configuration value, leaving architecture
            # and checkpoint parameter names untouched.
            sys.path.insert(0, str(self.repo))
            config = importlib.import_module("config")
            config.mt5_path = str(self.mt5_path)
            models = importlib.import_module("models")
            args = SimpleNamespace(hidden_dim=256, dataset="How2Sign", rgb_support=False, label_smoothing=0.0)
            model = models.Uni_Sign(args)
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(self.device)
            self._model, self._torch = model, torch
        except Exception as error:  # Preserve the original error for health/UI diagnostics.
            self.load_error = str(error)
            raise UniSignBackendError(f"Uni-Sign could not load: {error}") from error

    def translate_raw_mediapipe(self, raw_frames: list[dict[str, object]], *, experimental: bool) -> dict[str, object]:
        if not experimental or not self.allow_presence_proxy:
            raise UniSignBackendError(
                "MediaPipe-to-Uni-Sign is experimental: set experimental=true and "
                "UNISIGN_ALLOW_MEDIAPIPE_PRESENCE_PROXY=true after evaluation."
            )
        try:
            sequence = UniSignAdapter("presence_proxy").adapt(raw_frames)
        except UniSignCompatibilityError as error:
            raise UniSignBackendError(str(error)) from error
        self._load()
        assert self._model is not None and self._torch is not None
        torch = self._torch
        if sequence.body.shape[0] > 256:
            # The official online dataset sampler uses up to 256 frames. Use
            # deterministic uniform sampling—not random training-time sampling.
            indices = torch.linspace(0, sequence.body.shape[0] - 1, 256).long().cpu().numpy()
            sequence = type(sequence)(
                **{name: (getattr(sequence, name)[indices] if name in {"body", "left", "right", "face_all", "timestamps_ms"} else getattr(sequence, name))
                   for name in sequence.__dataclass_fields__}
            )
        src = {
            # ``as_model_input`` already supplies the batch dimension.
            name: torch.from_numpy(value).to(self.device)
            for name, value in sequence.as_model_input().items()
        }
        src["attention_mask"] = torch.ones((1, sequence.body.shape[0]), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            encoded = self._model(src, {"gt_sentence": [""]})
            output = self._model.generate(encoded, max_new_tokens=100, num_beams=4)
        text = self._model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return {
            "transcript": text,
            "frames": int(sequence.body.shape[0]),
            "topology_verified": False,
            "confidence_policy": "presence_proxy",
            "warning": "Experimental MediaPipe mapping; output is not a validated live translation.",
        }
