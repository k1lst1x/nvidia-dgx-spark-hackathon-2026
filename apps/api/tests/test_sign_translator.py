"""Unit tests for English-to-ASL Sign Language translation."""

from fastapi.testclient import TestClient
import main
from sign_translator import translate_english_to_asl

client = TestClient(main.app)


def test_translate_english_to_asl_idioms():
    res = translate_english_to_asl("Thank you very much, you are welcome!")
    assert "THANK-YOU" in res.asl_gloss
    assert "WELCOME" in res.asl_gloss
    assert res.tokens[0].gloss == "THANK-YOU"
    assert res.tokens[1].gloss == "WELCOME"


def test_translate_english_wh_question_reordering():
    # In ASL, WH question words (WHERE, WHAT, WHO) go to the end of the clause
    res = translate_english_to_asl("Where is Room 204?")
    assert res.asl_gloss.endswith("WHERE")
    assert "ROOM" in res.asl_gloss


def test_translate_english_fingerspelling_for_unrecognized_words():
    res = translate_english_to_asl("Alex")
    # "Alex" should be fingerspelled A-L-E-X
    assert [t.gloss for t in res.tokens] == ["A", "L", "E", "X"]
    assert all(t.fingerspell for t in res.tokens)


def test_translate_to_sign_api_endpoint():
    main._clear_rate_limits_for_tests()
    response = client.post(
        "/v1/translate-to-sign",
        headers={"X-Session-ID": "test-sign-translate"},
        json={"text": "Please help me, where is the bathroom?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "PLEASE" in data["asl_gloss"]
    assert "HELP" in data["asl_gloss"]
    assert "WHERE" in data["asl_gloss"]
    assert len(data["tokens"]) > 0
