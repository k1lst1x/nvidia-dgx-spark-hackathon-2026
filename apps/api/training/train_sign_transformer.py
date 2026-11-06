#!/usr/bin/env python3
"""Train VisionMate's causal CTC recognizer from gloss-labelled sequences.

Input is JSONL. Each record must look like:
{"features": [[281 floats], ...], "glosses": ["ME", "NEED", "WATER"]}

This intentionally does not accept the supplied fingerspelling image archive:
that archive contains static letter images, not labelled movement sequences.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sign_transformer import FEATURE_DIM, VisionMateTransformer, ctc_minimum_frames


@dataclass(frozen=True)
class Example:
    features: torch.Tensor
    glosses: tuple[str, ...]


def load_manifest(path: Path) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            # Web landmark capture exports `frames`; accept `features` as an
            # alias for pre-existing manifests.
            features = torch.tensor(record.get("frames", record.get("features")), dtype=torch.float32)
            glosses = tuple(str(gloss).upper() for gloss in record["glosses"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid manifest row {line_number} in {path}: {error}") from error
        if features.ndim != 2 or features.size(1) != FEATURE_DIM:
            raise ValueError(f"Row {line_number} must contain (frames, {FEATURE_DIM}) features")
        if not len(features) or not glosses:
            raise ValueError(f"Row {line_number} needs at least one frame and one gloss")
        examples.append(Example(features=features, glosses=glosses))
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


class SequenceDataset(Dataset[Example]):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate(vocabulary: dict[str, int]):
    def build_batch(examples: list[Example]):
        frames = pad_sequence([example.features for example in examples], batch_first=True)
        frame_lengths = torch.tensor([len(example.features) for example in examples], dtype=torch.long)
        target_rows = [[vocabulary[gloss] for gloss in example.glosses] for example in examples]
        for index, target in enumerate(target_rows):
            required = ctc_minimum_frames(target)
            if frame_lengths[index] < required:
                raise ValueError(
                    f"Example {index} has {frame_lengths[index]} frames but CTC needs at least {required} for repeated glosses"
                )
        targets = torch.tensor([token for row in target_rows for token in row], dtype=torch.long)
        target_lengths = torch.tensor([len(row) for row in target_rows], dtype=torch.long)
        return frames, targets, frame_lengths, target_lengths

    return build_batch


def run_epoch(
    model: VisionMateTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for frames, targets, frame_lengths, target_lengths in loader:
        frames, targets = frames.to(device), targets.to(device)
        frame_lengths, target_lengths = frame_lengths.to(device), target_lengths.to(device)
        with torch.set_grad_enabled(training):
            logits = model(frames, frame_lengths)
            loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, frame_lengths, target_lengths)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Gloss-labelled JSONL training manifest")
    parser.add_argument("--validation", type=Path, help="Optional JSONL validation manifest")
    parser.add_argument("--output", type=Path, default=Path("apps/api/models/visionmate_ctc_transformer.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples = load_manifest(args.train)
    validation_examples = load_manifest(args.validation) if args.validation else []
    vocabulary = {gloss: index + 1 for index, gloss in enumerate(sorted({g for e in train_examples for g in e.glosses}))}
    unknown = sorted({g for e in validation_examples for g in e.glosses if g not in vocabulary})
    if unknown:
        raise ValueError(f"Validation glosses missing from train vocabulary: {', '.join(unknown)}")

    collate_fn = collate(vocabulary)
    train_loader = DataLoader(SequenceDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(SequenceDataset(validation_examples), batch_size=args.batch_size, collate_fn=collate_fn) if validation_examples else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionMateTransformer(num_glosses=len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, None, device) if validation_loader else None
        report = f"epoch={epoch:03d} train_ctc={train_loss:.4f}"
        if validation_loss is not None:
            report += f" validation_ctc={validation_loss:.4f}"
        print(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "architecture": "visionmate_causal_transformer_ctc",
            "feature_dim": FEATURE_DIM,
            "vocabulary": vocabulary,
            "model_state_dict": model.cpu().state_dict(),
        },
        args.output,
    )
    print(f"Wrote checkpoint: {args.output}")


if __name__ == "__main__":
    main()
