"""The DGX Spark answers first; Groq's cloud is used only when the Spark fails."""

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

import main

SCENE = {"summary": "Chair ahead", "speech": "A chair is ahead.", "scene_key": "chair", "urgency": "low",
         "changed": True, "objects": [], "task": "scene"}


class FakeClient:
    """Stands in for httpx.AsyncClient; behaviour is set per URL."""
    behaviour: dict[str, object] = {}
    calls: list[tuple[str, str]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        FakeClient.calls.append((url, json["model"]))
        action = FakeClient.behaviour[url]
        if action == "timeout":
            raise httpx.ConnectTimeout("spark is unplugged")
        content = action if isinstance(action, str) else globals()["json"].dumps(action)
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def spark_with_backup(monkeypatch):
    monkeypatch.setattr(main, "VISION_PROVIDER", "local")
    monkeypatch.setattr(main, "VISION_API_URL", "http://spark.local:8000/v1/chat/completions")
    monkeypatch.setattr(main, "MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeClient)
    FakeClient.calls = []
    return "http://spark.local:8000/v1/chat/completions"


def describe():
    return asyncio.run(main._post_vision({"messages": []}, lambda raw: main.VisionResult.model_validate(json.loads(raw))))


def test_spark_answers_first_and_groq_is_not_called(spark_with_backup):
    FakeClient.behaviour = {spark_with_backup: SCENE, main.GROQ_API_URL: SCENE}
    assert describe().summary == "Chair ahead"
    assert [url for url, _ in FakeClient.calls] == [spark_with_backup]
    assert main._last_vision_answer["provider"] == "spark"


def test_spark_down_falls_back_to_groq(spark_with_backup):
    FakeClient.behaviour = {spark_with_backup: "timeout", main.GROQ_API_URL: SCENE}
    assert describe().summary == "Chair ahead"
    assert FakeClient.calls[-1] == (main.GROQ_API_URL, main.GROQ_MODEL)
    assert main._last_vision_answer == {**main._last_vision_answer, "provider": "groq", "fell_back": True}


def test_spark_invalid_output_falls_back_to_groq(spark_with_backup):
    FakeClient.behaviour = {spark_with_backup: "not json", main.GROQ_API_URL: SCENE}
    assert describe().summary == "Chair ahead"


def test_spark_down_without_groq_key_reports_timeout(spark_with_backup, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    FakeClient.behaviour = {spark_with_backup: "timeout"}
    with pytest.raises(HTTPException) as error:
        describe()
    assert error.value.status_code == 504
    assert main._vision_providers() and [p["name"] for p in main._vision_providers()] == ["spark"]
