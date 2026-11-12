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


def auth(device: str, secret_label: str | None = None, session: str = "session", token: str | None = None) -> dict[str, str]:
    secret = hashlib.sha256(f"secret:{secret_label or device}".encode()).hexdigest()
    device_id = f"device-{hashlib.sha256(secret.encode()).hexdigest()[:32]}"
    headers = {"X-Device-ID": device_id, "X-Session-ID": session, "X-Device-Secret": secret}
    if token:
        headers["X-Shared-Sight-Key"] = token
    return headers


@pytest.fixture(autouse=True)
def isolated_security_state(tmp_path, monkeypatch):
    monkeypatch.setattr(observation_store, "DB_PATH", tmp_path / "security.sqlite3")
    monkeypatch.setattr(main.campus_memory, "DB_PATH", tmp_path / "campus.sqlite3")
    monkeypatch.delenv("SHARED_SIGHT_TOKEN", raising=False)
    monkeypatch.delenv("SHARED_SIGHT_DEVICE_IDS", raising=False)
    monkeypatch.delenv("SHARED_SIGHT_MODERATOR_TOKEN", raising=False)
    observation_store.init_db()
    main.campus_memory.init_db()
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


def test_tampering_with_secret_suffix_is_rejected():
    client = TestClient(main.app)
    owner = auth("owner-device")
    created = client.post("/v1/observations", headers=owner, json=_personal("cane"))
    assert created.status_code == 200

    # Attacker knows owner's device ID and tries to forge secret by altering only the suffix
    tampered_headers = dict(owner)
    # Alter the last 32 characters
    tampered_headers["X-Device-Secret"] = owner["X-Device-Secret"][:32] + "0" * 32
    assert client.get("/v1/personal/observations", headers=tampered_headers).status_code == 401
    assert client.delete(f"/v1/observations/{created.json()['id']}", headers=tampered_headers).status_code == 401


def test_knowing_victim_device_id_cannot_forge_credentials():
    client = TestClient(main.app)
    victim = auth("victim-device")
    created = client.post("/v1/observations", headers=victim, json=_personal("wallet"))
    assert created.status_code == 200

    # Attacker knows victim's public device ID and attempts to access victim's personal data
    attacker = auth("attacker-device")
    attacker["X-Device-ID"] = victim["X-Device-ID"]
    assert client.get("/v1/personal/observations", headers=attacker).status_code == 401
    assert client.delete(f"/v1/observations/{created.json()['id']}", headers=attacker).status_code == 401


def test_empty_token_and_allowlist_fails_closed_for_shared_mutations(monkeypatch):
    monkeypatch.delenv("SHARED_SIGHT_TOKEN", raising=False)
    monkeypatch.delenv("SHARED_SIGHT_DEVICE_IDS", raising=False)
    client = TestClient(main.app)
    caller = auth("anonymous-caller")

    # Creating shared observation fails closed
    shared_obs = {
        "zone_id": "Hallway", "category": "obstacle", "label": "box",
        "confidence": 0.9, "privacy_scope": "shared", "verified": True,
    }
    assert client.post("/v1/observations", headers=caller, json=shared_obs).status_code == 403

    # Reporting campus hazard alert fails closed
    hazard = {
        "location": "Hallway", "category": "hazard", "label": "wet floor",
        "description": "cleaning in progress", "severity": "medium",
    }
    assert client.post("/v1/memory/observe", headers=caller, json=hazard).status_code == 403

    # Establishing baseline in new location fails closed
    new_baseline = {
        "location": "NewRoom",
        "current_objects": [{"label": "door", "x": 0.5, "y": 0.5}],
    }
    assert client.post("/v1/memory/what-changed", headers=caller, json=new_baseline).status_code == 403


def test_ordinary_reporter_cannot_resolve_another_devices_alert(monkeypatch):
    monkeypatch.setenv("SHARED_SIGHT_TOKEN", "shared-token-123")
    client = TestClient(main.app)
    reporter_a = auth("device-a", token="shared-token-123")
    reporter_b = auth("device-b", token="shared-token-123")

    hazard = {
        "location": "Hallway 200", "category": "hazard", "label": "wet floor",
        "description": "Water spill", "severity": "medium",
    }
    res = client.post("/v1/memory/observe", headers=reporter_a, json=hazard)
    assert res.status_code == 200
    alert_id = res.json()["id"]

    # Reporter B cannot resolve Reporter A's alert
    res_b = client.post(f"/v1/memory/resolve/{alert_id}", headers=reporter_b)
    assert res_b.status_code == 403
    assert "moderator" in res_b.json()["detail"].lower()

    # Reporter A (owner) can resolve its own alert
    res_a = client.post(f"/v1/memory/resolve/{alert_id}", headers=reporter_a)
    assert res_a.status_code == 200
    assert res_a.json()["status"] == "resolved"


def test_moderator_can_resolve_any_alert_and_replace_baseline(monkeypatch):
    monkeypatch.setenv("SHARED_SIGHT_TOKEN", "shared-token-123")
    monkeypatch.setenv("SHARED_SIGHT_MODERATOR_TOKEN", "moderator-token-999")
    client = TestClient(main.app)
    reporter = auth("device-reporter", token="shared-token-123")
    moderator = auth("device-mod")
    moderator["X-Moderator-Key"] = "moderator-token-999"

    # Reporter posts alert
    hazard = {
        "location": "Hallway 200", "category": "hazard", "label": "ice",
        "description": "slippery ice", "severity": "high",
    }
    res = client.post("/v1/memory/observe", headers=reporter, json=hazard)
    assert res.status_code == 200
    alert_id = res.json()["id"]

    # Moderator can resolve reporter's alert
    res_mod = client.post(f"/v1/memory/resolve/{alert_id}", headers=moderator)
    assert res_mod.status_code == 200

    # Establish baseline
    base = {
        "location": "Room 500",
        "current_objects": [{"label": "desk", "x": 0.3, "y": 0.5}],
    }
    est = client.post("/v1/memory/what-changed", headers=reporter, json=base)
    assert est.status_code == 200

    # Reporter cannot replace established baseline
    replace_req = {
        "location": "Room 500",
        "current_objects": [{"label": "chair", "x": 0.7, "y": 0.5}],
        "update_baseline": True,
    }
    rep_res = client.post("/v1/memory/what-changed", headers=reporter, json=replace_req)
    assert rep_res.status_code == 403
    assert "moderator" in rep_res.json()["detail"].lower()

    # Moderator CAN replace established baseline
    mod_res = client.post("/v1/memory/what-changed", headers=moderator, json=replace_req)
    assert mod_res.status_code == 200


def test_what_changed_rejects_unbounded_objects(monkeypatch):
    monkeypatch.setenv("SHARED_SIGHT_TOKEN", "token")
    client = TestClient(main.app)
    caller = auth("device-test", token="token")

    # Send 65 objects (max is 64)
    oversized = {
        "location": "Room 100",
        "current_objects": [{"label": f"item-{i}", "x": 0.5, "y": 0.5} for i in range(65)],
    }
    res = client.post("/v1/memory/what-changed", headers=caller, json=oversized)
    assert res.status_code == 422


def test_unisign_endpoint_admission_controls(monkeypatch):
    monkeypatch.setattr(main._unisign_backend, "allow_presence_proxy", True)
    monkeypatch.setattr(main._unisign_backend, "enabled", True)
    client = TestClient(main.app)

    # 1. Unauthenticated request rejected with 401
    valid_payload = {
        "raw_holistic": [{"pose": [{"x": 0.1, "y": 0.2}]}],
        "experimental": True,
    }
    res = client.post("/v1/unisign/interpret", json=valid_payload)
    assert res.status_code == 401

    caller = auth("device-unisign")

    # 2. Oversized payload (> 256 frames) rejected by schema validation
    oversized = {
        "raw_holistic": [{"pose": [{"x": 0.1, "y": 0.2}]} for _ in range(257)],
        "experimental": True,
    }
    assert client.post("/v1/unisign/interpret", headers=caller, json=oversized).status_code == 422

    # 3. Disabled Uni-Sign returns 503
    valid_seq = {
        "raw_holistic": [{"pose": [{"x": 0.1, "y": 0.2}], "left_hand": []}],
        "experimental": True,
    }
    monkeypatch.setattr(main._unisign_backend, "enabled", False)
    assert client.post("/v1/unisign/interpret", headers=caller, json=valid_seq).status_code == 503

