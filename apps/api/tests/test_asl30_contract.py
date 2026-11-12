from __future__ import annotations

import unittest

from sign_language.asl30.contract import CaptureValidationError, FEATURE_COUNT, validate_capture


def recording(**overrides):
    value = {
        "id": "hello-001",
        "signerId": "signer-a",
        "language": "ASL",
        "glosses": ["HELLO"],
        "fps": 20,
        "feature_schema": "visionmate-281-v1",
        "frames": [[0.0] * FEATURE_COUNT, [0.1] * FEATURE_COUNT],
        "timestamps_ms": [0, 50],
    }
    value.update(overrides)
    return value


class Asl30ContractTests(unittest.TestCase):
    def test_accepts_one_target_gloss_and_the_281_schema(self):
        sample = validate_capture(recording(glosses=["thank you"]))
        self.assertEqual(sample.gloss, "THANK_YOU")
        self.assertEqual(sample.as_manifest_row()["task"], "isolated_asl30")

    def test_accepts_unknown_negative_examples(self):
        self.assertEqual(validate_capture(recording(glosses=["unknown"])).gloss, "UNKNOWN")

    def test_rejects_sentence_and_wrong_schema(self):
        with self.assertRaisesRegex(CaptureValidationError, "exactly one"):
            validate_capture(recording(glosses=["HELLO", "WATER"]))
        with self.assertRaisesRegex(CaptureValidationError, "feature_schema"):
            validate_capture(recording(feature_schema="popsign-390"))

    def test_rejects_malformed_frames_and_timestamps(self):
        with self.assertRaisesRegex(CaptureValidationError, "exactly 281"):
            validate_capture(recording(frames=[[0.0] * 280, [0.0] * 280]))
        with self.assertRaisesRegex(CaptureValidationError, "strictly increasing"):
            validate_capture(recording(timestamps_ms=[0, 0]))

