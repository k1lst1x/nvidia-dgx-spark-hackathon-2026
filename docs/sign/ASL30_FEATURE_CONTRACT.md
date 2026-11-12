# Separate ASL-30 experiment: feature and data contract

This is a new, isolated training path. It does **not** replace the deployed
390-feature PopSign ONNX model, alter Bridge behaviour, or enable a new user
facing recognizer.

## Chosen candidate schema

The experiment uses the existing `visionmate-281-v1` capture format:

- 126 hand-landmark coordinates (two hands)
- 99 pose-landmark coordinates
- 52 facial-expression blendshape scores
- 4 landmark-presence flags
- timestamps sampled by `SignRecorder` at a maximum 20 FPS cadence

This is a sensible candidate for isolated ASL-30 classification because it
contains hand movement, two-hand relationships, body location, facial
expression signals, and timing. It is **not yet evidence** that it can
recognize every target sign. That requires a signer-disjoint evaluation.

The authoritative layout remains [FEATURE_SCHEMA_281.md](FEATURE_SCHEMA_281.md).
The ASL-30 path will reject any other representation, including the live
PopSign 390-feature frames.

## Recording contract

`apps/api/sign_language/asl30/contract.py` accepts only an isolated recording
with:

- exactly one approved target gloss or `UNKNOWN`;
- `feature_schema: "visionmate-281-v1"`;
- one finite 281-value frame per strictly increasing timestamp;
- an anonymous, stable signer ID;
- no automatically inferred labels.

`UNKNOWN` is for normal movement, incomplete signing, and signs outside the
30-word vocabulary. It is a required training class, not a fallback word.
Each target must be reviewed by a fluent ASL signer before the capture enters
the manifest.

Prepare recordings without touching production inference:

```bash
cd apps/api
python -m sign_language.asl30.prepare_manifest \
  --input ../../datasets/asl30/recordings \
  --output ../../datasets/asl30/manifests/all.jsonl
```

The command fails rather than silently dropping malformed or unsupported
captures. Split the resulting manifest by **signer**, never by random frame or
recording, before training. Do not train until every target class and UNKNOWN
has sufficient examples in each required split.

## Integration gate

Only integrate this experiment after all of these are true:

1. A fixed ASL-30 model has been trained and exported from this schema.
2. Per-class precision, recall, macro F1, false-positive rate on `UNKNOWN`,
   and latency are measured on held-out signers.
3. Web and native consume the same timestamped 281-frame sequence and decoder.
4. The old recognizer remains selectable behind a feature flag until the new
   path passes the agreed thresholds.
