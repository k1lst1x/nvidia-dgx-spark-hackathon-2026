"""Regression tests for the repository's externally reachable security boundaries."""

import asyncio
import hashlib
import threading
import time
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
import observation_store
from rate_limit import LimitExceeded, LimitRule, SlidingWindowLimiter, TieredSlidingWindowLimiter


def auth(device: str, secret_label: str | None = None, session: str = "session") -> dict[str, str]:
    secret = hashlib.sha256(f"secret:{secret_label or device}".encode()).hexdigest()
    return {"X-Device-ID": f"device-{secret[:32]}", "X-Session-ID": session, "X-Device-Secret": secret}


@pytest.fixture(autouse=True)
def isolated_security_state(tmp_path, monkeypatch):
    monkeypatch.setattr(observation_store, "DB_PATH", tmp_path / "security.sqlite3")
    observation_store.init_db()
    main._clear_rate_limits_for_tests()
    yield
    main._clear_rate_limits_for_tests()


def _personal(label: str) -> dict[str, object]:
    return {
        "zone_id": "Room 207", "category": "personal_item", "label": label,
        "confidence": 0.9, "privacy_scope": "personal", "verified": False,
    }


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_device_secret_is_bound_to_personal_data_across_sessions():
    client = TestClient(main.app)
    owner = auth("device-owner", session="first")
    created = client.post("/v1/observations", headers=owner, json=_personal("white cane"))
    assert created.status_code == 200

    forged_headers = auth("attacker", session="forged")
    forged_headers["X-Device-ID"] = owner["X-Device-ID"]
    forged = client.get("/v1/personal/observations", headers=forged_headers)
    assert forged.status_code == 401
    assert "does not match" in forged.json()["detail"].lower()

    resumed = client.get(
        "/v1/personal/observations", headers=auth("device-owner", session="second"),
    )
    assert resumed.status_code == 200
    assert [item["label"] for item in resumed.json()] == ["white cane"]


def test_device_scoped_routes_fail_closed_without_device_secret():
    response = TestClient(main.app).get(
        "/v1/personal/observations",
        headers={"X-Device-ID": "known-device", "X-Session-ID": "session"},
    )
    assert response.status_code == 401


def test_legacy_identity_requires_migration_key(monkeypatch):
    observation_store.create_observation(
        zone_id="Room 207", category="personal_item", label="keys", description="",
        confidence=0.9, privacy_scope="personal", verified=False, expires_in_seconds=3600,
        origin="manual", source_device="legacy-device", source_session="old-session",
    )
    client = TestClient(main.app)
    legacy_headers = auth("legacy-secret")
    legacy_headers["X-Device-ID"] = "legacy-device"
    assert client.get("/v1/personal/observations", headers=legacy_headers).status_code == 401

    monkeypatch.setenv("SHARED_SIGHT_TOKEN", "migration-key")
    migrated = client.get(
        "/v1/personal/observations",
        headers={**legacy_headers, "X-Shared-Sight-Key": "migration-key"},
    )
    assert migrated.status_code == 200
    monkeypatch.delenv("SHARED_SIGHT_TOKEN")
    assert client.get("/v1/personal/observations", headers=legacy_headers).status_code == 200


def test_rotating_session_ids_cannot_bypass_peer_inference_limit(monkeypatch):
    monkeypatch.setattr(main, "_general_limits", TieredSlidingWindowLimiter(
        session=LimitRule(100, 60), peer=LimitRule(3, 60), global_budget=LimitRule(100, 60, max_keys=1),
    ))
    monkeypatch.setattr(main._yolo_sign_detector, "detect", lambda _body: [])
    client = TestClient(main.app)
    statuses = []
    for index in range(4):
        response = client.post(
            "/v1/detect-sign", headers={"X-Session-ID": f"rotated-{index}"},
            files={"file": ("frame.png", _png(), "image/png")},
        )
        statuses.append(response.status_code)
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_limiter_evicts_expired_keys_and_caps_live_keys():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10, max_keys=2)
    limiter.check("one", now=0)
    limiter.check("two", now=0)
    with pytest.raises(LimitExceeded):
        limiter.check("three", now=1)
    limiter.check("three", now=11)
    assert limiter.key_count == 1


def test_bridge_join_guesses_are_limited_across_rotated_device_ids(monkeypatch):
    monkeypatch.setattr(main, "_bridge_limits", TieredSlidingWindowLimiter(
        device=LimitRule(100, 300), peer=LimitRule(2, 300), global_budget=LimitRule(100, 60, max_keys=1),
    ))
    client = TestClient(main.app)
    statuses = [
        client.post(f"/v1/bridge/conversations/00000{index}/join", headers=auth(f"rotated-{index}")).status_code
        for index in range(3)
    ]
    assert statuses[:2] == [404, 404]
    assert statuses[2] == 429


def test_detect_sign_rejects_oversized_and_unsupported_uploads_before_inference(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 8)
    monkeypatch.setattr(main._yolo_sign_detector, "detect", lambda body: calls.append(body) or [])
    client = TestClient(main.app)
    oversized = client.post(
        "/v1/detect-sign", headers={"X-Session-ID": "oversized"},
        files={"file": ("frame.jpg", b"123456789", "image/jpeg")},
    )
    unsupported = client.post(
        "/v1/detect-sign", headers={"X-Session-ID": "unsupported"},
        files={"file": ("frame.txt", b"x", "text/plain")},
    )
    assert oversized.status_code == 413
    assert unsupported.status_code == 415
    assert calls == []


def test_detect_sign_uses_global_inference_semaphore(monkeypatch):
    active = 0
    maximum = 0
    lock = threading.Lock()

    def slow_detect(_body):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return []

    monkeypatch.setattr(main, "_slots", asyncio.Semaphore(1))
    monkeypatch.setattr(main._yolo_sign_detector, "detect", slow_detect)
    monkeypatch.setattr(main, "_general_limits", TieredSlidingWindowLimiter(
        session=LimitRule(100, 60), peer=LimitRule(100, 60), global_budget=LimitRule(100, 60, max_keys=1),
    ))

    async def exercise():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(*[
                client.post(
                    "/v1/detect-sign", headers={"X-Session-ID": f"valid-{index}"},
                    files={"file": ("frame.png", _png(), "image/png")},
                )
                for index in range(2)
            ])

    responses = asyncio.run(exercise())
    assert [response.status_code for response in responses] == [200, 200]
    assert maximum == 1


def test_tiered_limiter_rejection_is_atomic():
    limiter = TieredSlidingWindowLimiter(
        session=LimitRule(2, 60), global_budget=LimitRule(1, 60, max_keys=1),
    )
    limiter.check(("session", "first"), ("global_budget", "global"), now=0)
    with pytest.raises(LimitExceeded):
        limiter.check(("session", "second"), ("global_budget", "global"), now=1)
    # The globally rejected request did not consume the second session's tier.
    assert limiter.key_count("session") == 1


def test_invalid_uploads_do_not_consume_inference_budget(monkeypatch):
    limits = TieredSlidingWindowLimiter(
        session=LimitRule(10, 60), peer=LimitRule(10, 60), global_budget=LimitRule(1, 60, max_keys=1),
    )
    monkeypatch.setattr(main, "_general_limits", limits)
    monkeypatch.setattr(main._yolo_sign_detector, "detect", lambda _body: [])
    client = TestClient(main.app)
    invalid = client.post(
        "/v1/detect-sign", headers={"X-Session-ID": "invalid"},
        files={"file": ("frame.txt", b"invalid", "text/plain")},
    )
    valid = client.post(
        "/v1/detect-sign", headers={"X-Session-ID": "valid"},
        files={"file": ("frame.png", _png(), "image/png")},
    )
    assert invalid.status_code == 415
    assert valid.status_code == 200


def test_arbitrary_device_ids_cannot_create_credential_rows():
    client = TestClient(main.app)
    for index in range(3):
        headers = auth(f"attacker-{index}")
        headers["X-Device-ID"] = f"claimed-{index}"
        assert client.get("/v1/personal/observations", headers=headers).status_code == 401
    with observation_store.db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM device_credentials").fetchone()[0] == 0
