"""Unit tests for Campus Memory & Shared Sight & Deaf-Blind Bridge."""

import os
import uuid
from fastapi.testclient import TestClient

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("VISION_API_KEY", None)
os.environ.pop("VISION_API_URL", None)

from main import app  # noqa: E402

client = TestClient(app, headers={"X-Device-ID": "test-device", "X-Session-ID": "test-session"})
anonymous = TestClient(app)


def test_campus_endpoints_require_device_identity():
    assert anonymous.get("/v1/memory/alerts").status_code == 401
    assert anonymous.post("/v1/memory/resolve/1").status_code == 401
    assert anonymous.get("/v1/bridge/messages?session_id=x").status_code == 401


def test_get_campus_alerts_returns_only_real_reports():
    response = client.get("/v1/memory/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    for item in alerts:
        assert "id" in item
        assert "location" in item
        assert "category" in item
        assert "description" in item
        assert "relative_time" in item


def test_filter_alerts_by_location():
    response = client.get("/v1/memory/alerts?location=Library")
    assert response.status_code == 200
    alerts = response.json()
    for a in alerts:
        assert "Library" in a["location"] or a["location"] == "All Campus"


def test_post_observation_and_resolve():
    new_obs = {
        "location": "Room 204",
        "category": "hazard",
        "label": "spilled coffee",
        "description": "Coffee spilled near desk 3.",
        "severity": "medium",
    }
    response = client.post(
        "/v1/memory/observe",
        headers={"X-Session-ID": "student-test-device"},
        json=new_obs,
    )
    assert response.status_code == 200
    alert = response.json()
    alert_id = alert["id"]
    assert alert["label"] == "spilled coffee"
    assert alert["reported_by"] == "student-test-device"

    # Resolve it
    res = client.post(f"/v1/memory/resolve/{alert_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"


def test_what_changed_detects_delta():
    location = "integration-test-delta-room"
    baseline = [
        {"label": "table", "x": 0.2, "y": 0.5, "width": 0.6, "height": 0.4},
        {"label": "chair", "x": 0.25, "y": 0.55, "width": 0.15, "height": 0.25},
    ]
    first = client.post(
        "/v1/memory/what-changed",
        json={"location": location, "current_objects": baseline, "update_baseline": True},
    )
    assert first.status_code == 200
    current = [
        {"label": "table", "x": 0.2, "y": 0.5, "width": 0.6, "height": 0.4},
        {"label": "chair", "x": 0.25, "y": 0.55, "width": 0.15, "height": 0.25},
        {"label": "box", "x": 0.45, "y": 0.60, "width": 0.20, "height": 0.20},
    ]
    res = client.post(
        "/v1/memory/what-changed",
        json={"location": location, "current_objects": current, "update_baseline": False},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["has_changed"] is True
    assert "box" in body["added_objects"]
    assert "box" in body["speech"].lower()


def _bridge_device(name: str) -> TestClient:
    device = f"{name}-{uuid.uuid4().hex[:8]}"
    return TestClient(app, headers={"X-Device-ID": device, "X-Session-ID": f"session-{device}"})


def test_bridge_messages_lifecycle():
    deaf_phone = _bridge_device("deaf")
    blind_phone = _bridge_device("blind")

    created = deaf_phone.post("/v1/bridge/conversations")
    assert created.status_code == 200
    code = created.json()["code"]
    assert len(code) == 6 and code.isdigit()
    assert created.json()["partner_joined"] is False

    joined = blind_phone.post(f"/v1/bridge/conversations/{code}/join")
    assert joined.status_code == 200
    assert joined.json()["partner_joined"] is True
    assert deaf_phone.get(f"/v1/bridge/conversations/{code}").json()["devices"] == 2

    r1 = deaf_phone.post("/v1/bridge/message", json={"session_id": code, "sender_role": "deaf", "text": "THANK YOU"})
    assert r1.status_code == 200
    assert r1.json()["text"] == "THANK YOU"
    assert r1.json()["from_this_device"] is True

    r2 = blind_phone.post("/v1/bridge/message", json={
        "session_id": code, "sender_role": "blind", "text": "You're welcome, let me know if you need anything else.",
    })
    assert r2.status_code == 200

    # The partner sees both messages, in order, and knows which ones it sent.
    msgs = blind_phone.get(f"/v1/bridge/messages?session_id={code}").json()
    assert [m["text"] for m in msgs][0] == "THANK YOU"
    assert [m["from_this_device"] for m in msgs] == [False, True]

    # Live refresh asks only for what is new.
    newer = deaf_phone.get(f"/v1/bridge/messages?session_id={code}&after_id={r1.json()['id']}").json()
    assert [m["sender_role"] for m in newer] == ["blind"]


def test_bridge_is_one_pair_and_private():
    first, second, outsider = _bridge_device("a"), _bridge_device("b"), _bridge_device("c")
    code = first.post("/v1/bridge/conversations").json()["code"]
    assert second.post(f"/v1/bridge/conversations/{code}/join").status_code == 200
    # Rejoining is harmless; a third device is turned away.
    assert second.post(f"/v1/bridge/conversations/{code}/join").status_code == 200
    assert outsider.post(f"/v1/bridge/conversations/{code}/join").status_code == 409
    assert outsider.get(f"/v1/bridge/messages?session_id={code}").status_code == 403
    assert outsider.post("/v1/bridge/message", json={"session_id": code, "sender_role": "deaf", "text": "hi"}).status_code == 403
    assert first.get("/v1/bridge/messages?session_id=000000").status_code in {403, 404}
    assert first.post("/v1/bridge/message", json={"session_id": "not-a-code", "sender_role": "deaf", "text": "hi"}).status_code == 422


def test_bridge_returns_the_newest_messages_past_twenty():
    phone = _bridge_device("laptop")
    code = phone.post("/v1/bridge/conversations").json()["code"]
    for index in range(25):
        assert phone.post("/v1/bridge/message", json={
            "session_id": code, "sender_role": "deaf" if index % 2 else "blind", "text": f"message {index}",
        }).status_code == 200
    latest = phone.get(f"/v1/bridge/messages?session_id={code}&limit=20").json()
    assert len(latest) == 20
    assert latest[0]["text"] == "message 5"
    assert latest[-1]["text"] == "message 24"


def test_bridge_retry_with_same_client_id_is_stored_once():
    phone = _bridge_device("retry")
    code = phone.post("/v1/bridge/conversations").json()["code"]
    body = {"session_id": code, "sender_role": "deaf", "text": "HELP", "client_id": "local-123"}
    first = phone.post("/v1/bridge/message", json=body).json()
    second = phone.post("/v1/bridge/message", json=body).json()
    assert first["id"] == second["id"]
    assert len(phone.get(f"/v1/bridge/messages?session_id={code}").json()) == 1


def test_bridge_conversations_expire():
    import observation_store

    code = observation_store.create_bridge("expiring-device", now=1000.0)["code"]
    observation_store.add_bridge_message(code, "expiring-device", "deaf", "HELLO", now=1001.0)
    later = 1001.0 + observation_store.BRIDGE_TTL_SECONDS + 1
    observation_store.cleanup_expired(now=later)
    try:
        observation_store.list_bridge_messages(code, "expiring-device", now=later)
    except observation_store.BridgeNotFoundError:
        pass
    else:
        raise AssertionError("an expired conversation must not be readable")
