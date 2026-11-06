"""Causal multimodal Transformer for continuous sign recognition.

Each frame is a 281-value fused feature vector:
  126 hand-landmark values (two hands), 99 pose values (33 points),
  52 facial blendshape values, and 4 presence masks.

The model predicts a sequence of gloss IDs with CTC. It is deliberately a
recognizer, not an English language model: gloss-to-English translation should
remain a separate, auditable component.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


FEATURE_DIM = 281
CTC_BLANK_ID = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 1024) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {values.size(1)} exceeds positional limit {self.pe.size(1)}")
        return values + self.pe[:, : values.size(1)]


class VisionMateTransformer(nn.Module):
    """Causal encoder whose logits are trained with ``torch.nn.CTCLoss``."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_glosses: int = 100,
        dropout: float = 0.1,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        if input_size != FEATURE_DIM:
            raise ValueError(f"Expected the fused {FEATURE_DIM}-feature representation, got {input_size}")
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        # ID zero is reserved by CTC for the blank symbol.
        self.ctc_head = nn.Linear(d_model, num_glosses + 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.size(-1) != FEATURE_DIM:
            raise ValueError(f"Expected (batch, frames, {FEATURE_DIM}) features, got {tuple(features.shape)}")
        time = features.size(1)
        encoded = self.position(self.embedding(features))
        # True values are masked. This prevents live decoding from accessing
        # movement that has not occurred yet.
        causal_mask = torch.triu(
            torch.ones((time, time), device=features.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = None
        if lengths is not None:
            positions = torch.arange(time, device=features.device).unsqueeze(0)
            padding_mask = positions >= lengths.to(features.device).unsqueeze(1)
        return self.ctc_head(self.encoder(encoded, mask=causal_mask, src_key_padding_mask=padding_mask))


def decode_ctc(logits: torch.Tensor, id_to_gloss: Sequence[str]) -> list[list[str]]:
    """Greedily decode ``(batch, frames, glosses + blank)`` logits."""
    predicted = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded: list[list[str]] = []
    for sequence in predicted:
        glosses: list[str] = []
        previous = CTC_BLANK_ID
        for token_id in sequence:
            if token_id != CTC_BLANK_ID and token_id != previous:
                glosses.append(id_to_gloss[token_id - 1])
            previous = token_id
        decoded.append(glosses)
    return decoded


def ctc_minimum_frames(target_ids: Sequence[int]) -> int:
    """Minimum input length needed by CTC, accounting for repeated glosses."""
    repeats = sum(left == right for left, right in zip(target_ids, target_ids[1:]))
    return len(target_ids) + repeats


if __name__ == "__main__":
    model = VisionMateTransformer(num_glosses=100)
    sample = torch.randn(2, 90, FEATURE_DIM)
    print(model(sample).shape)  # torch.Size([2, 90, 101])
