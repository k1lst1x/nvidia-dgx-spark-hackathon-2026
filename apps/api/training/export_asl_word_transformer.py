"""Export the MIT-licensed 250-class ASL landmark Transformer to ONNX.

The source checkpoint and label map come from:
https://github.com/aymenwnia/ASL-translator
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pe[:, : values.shape[1], :]


class SignTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = config["d_model"]
        self.input_proj = nn.Sequential(
            nn.Linear(config["input_dim"], width),
            nn.LayerNorm(width),
            nn.Dropout(0.3),
        )
        self.pos_encoder = PositionalEncoding(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config["n_heads"],
            dim_feedforward=config["dim_ff"],
            dropout=0.3,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config["n_layers"])
        self.classifier = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(width, config["n_classes"]),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding_mask = values.abs().sum(dim=-1) == 0
        encoded = self.input_proj(values)
        encoded = self.pos_encoder(encoded)
        encoded = self.transformer(encoded, src_key_padding_mask=padding_mask)
        present = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        masked = encoded * present
        mean = masked.sum(dim=1) / present.sum(dim=1).clamp(min=1)
        maximum = encoded.masked_fill(padding_mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.classifier(torch.cat([mean, maximum], dim=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.labels.read_text())
    model = SignTransformer(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    sample = torch.randn(1, config["seq_len"], config["input_dim"])
    sample[:, -8:, :] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["landmarks"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
