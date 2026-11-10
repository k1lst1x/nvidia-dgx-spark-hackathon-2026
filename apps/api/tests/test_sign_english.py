import asyncio
import json
from io import BytesIO

import httpx
from PIL import Image

import main
from sign_english import english_text, yolo_sign_phrase


def _frames():
    buffer = BytesIO()
    Image.new("RGB", (640, 360), "gray").save(buffer, format="JPEG")
    return [(buffer.getvalue(), "image/jpeg")] * 3


def _decode_with_reply(monkeypatch, reply: dict, previous: str = ""):
    sent = {}

    async def fake_post(self, url, headers=None, json=None):
        sent["payload"] = json
        content = {"choices": [{"message": {"content": __import__("json").dumps(reply)}}]}
        return httpx.Response(200, json=content, request=httpx.Request("POST", url))

    monkeypatch.setattr(main, "_vision_configured", lambda: True)
    monkeypatch.setattr(main, "_vision_headers", lambda: {})
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return asyncio.run(main._call_groq_sign(_frames(), previous)), sent["payload"]


def test_sign_decoder_sends_readable_frames_and_returns_english(monkeypatch):
    reply = {"transcript": "Hello. WHERE BATHROOM", "segment": "WHERE BATHROOM", "confidence": 0.9, "is_final": True, "should_speak": True}
    result, payload = _decode_with_reply(monkeypatch, reply, previous="Hello.")
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    strip = Image.open(BytesIO(__import__("base64").b64decode(image_url.split(",", 1)[1])))
    assert strip.width == 3 * main.SIGN_FRAME_WIDTH
    assert result.transcript == "Hello. Where bathroom"
    assert result.segment == "Where bathroom"
    assert result.should_speak


def test_sign_decoder_drops_non_english_output(monkeypatch):
    reply = {"transcript": "Спасибо", "segment": "Спасибо", "confidence": 0.95, "is_final": True, "should_speak": True}
    result, _ = _decode_with_reply(monkeypatch, reply)
    assert result.segment == ""
    assert not result.should_speak


def test_english_text_keeps_natural_english():
    assert english_text("  Where is the   bathroom? ") == "Where is the bathroom?"


def test_english_text_rewrites_all_caps_gloss_in_sentence_case():
    assert english_text("I WANT WATER") == "I want water"


def test_english_text_drops_other_scripts():
    assert english_text("Спасибо") == ""
    assert english_text("谢谢 thank you") == ""
    assert english_text("Merci, café") == ""


def test_yolo_fallback_speaks_words_but_not_letters():
    assert yolo_sign_phrase("ThankYou") == "Thank you."
    assert yolo_sign_phrase("ExcuseMe") == "Excuse me."
    assert yolo_sign_phrase("A") == ""
    assert yolo_sign_phrase("7") == ""
