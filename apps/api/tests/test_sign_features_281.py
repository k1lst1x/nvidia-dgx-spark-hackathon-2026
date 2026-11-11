"""The Python 281 producer must match apps/mobile/tests/signFeatures.test.cjs."""

import re
from pathlib import Path

from sign_language.adapters.mediapipe_281 import BLENDSHAPE_ORDER, build_sign_feature_frame

TS_SOURCE = Path(__file__).resolve().parents[2] / "mobile" / "src" / "signFeatures.ts"


def _points(count, base):
    return [{"x": base + i, "y": base + i + 0.25, "z": base + i + 0.5} for i in range(count)]


def test_blendshape_order_matches_the_typescript_list():
    block = re.search(r"BLENDSHAPE_ORDER = \[(.*?)\] as const", TS_SOURCE.read_text(encoding="utf-8"), re.S).group(1)
    assert tuple(re.findall(r"'([^']+)'", block)) == BLENDSHAPE_ORDER
    assert list(BLENDSHAPE_ORDER) == sorted(BLENDSHAPE_ORDER)


def test_test_vector_every_block_lands_at_its_documented_index():
    frame = build_sign_feature_frame(
        _points(21, 100), _points(21, 200), _points(33, 300), True,
        [("noseSneerRight", 0.9), ("jawOpen", 0.5), ("_neutral", 0.1), ("notABlendshape", 1)],
    )
    assert len(frame) == 281
    assert frame[0:3] == [100, 100.25, 100.5]
    assert frame[60:63] == [120, 120.25, 120.5]
    assert frame[63:66] == [200, 200.25, 200.5]
    assert frame[126:129] == [300, 300.25, 300.5]
    assert frame[222:225] == [332, 332.25, 332.5]
    assert frame[225] == 0.1
    assert frame[225 + 25] == 0.5
    assert frame[276] == 0.9
    assert len([v for v in frame[225:277] if v != 0]) == 3
    assert frame[277:] == [1, 1, 1, 1]


def test_missing_parts_are_zeros_with_a_zero_presence_flag():
    frame = build_sign_feature_frame(None, _points(21, 1), None, False)
    assert all(v == 0 for v in frame[0:63])
    assert all(v == 0 for v in frame[126:277])
    assert frame[277:] == [0, 1, 0, 0]
