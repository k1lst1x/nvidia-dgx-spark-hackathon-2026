"""Two-device API integration coverage for Shared Sight data flow."""

from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest
from fastapi.testclient import TestClient

import observation_store
from main import app


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setattr(observation_store, "DB_PATH", tmp_path / "shared-sight.sqlite3")
    observation_store.TRACKING_METRICS.clear()
    observation_store.init_db()
    return TestClient(app), TestClient(app)


def headers(device: str, session: str) -> dict[str, str]:
    secret = hashlib.sha256(f"test-secret:{device}".encode()).hexdigest()
    return {"X-Device-ID": f"device-{secret[:32]}", "X-Session-ID": session, "X-Device-Secret": secret}


def observation(label: str, scope: str = "shared", verified: bool = True) -> dict[str, object]:
    return {
        "zone_id": "Room 204", "category": "object", "label": label,
        "description": "User checked the visible object.", "confidence": 0.91,
        "privacy_scope": scope, "verified": verified,
    }


def test_two_devices_share_verified_reports_and_keep_personal_recall_private(clients):
    device_a, device_b = clients
    headers_a = headers("device-a", "session-a")
    headers_b = headers("device-b", "session-b")

    with ThreadPoolExecutor(max_workers=2) as pool:
        posts = [
            pool.submit(device_a.post, "/v1/observations", headers=headers_a, json=observation("chair")),
            pool.submit(device_b.post, "/v1/observations", headers=headers_b, json=observation("backpack")),
        ]
        responses = [future.result() for future in posts]
    assert [response.status_code for response in responses] == [200, 200]
    assert all(response.json()["verification_status"] == "verified" for response in responses)
    assert all(response.json()["privacy_scope"] == "shared" for response in responses)

    with ThreadPoolExecutor(max_workers=2) as pool:
        views = list(pool.map(lambda args: args[0].get("/v1/shared-sight/dashboard?zone_id=Room%20204", headers=args[1]), [(device_a, headers_a), (device_b, headers_b)]))
    assert all(view.status_code == 200 for view in views)
    assert all({item["label"] for item in view.json()["recent_observations"]} == {"chair", "backpack"} for view in views)
    assert all(view.json()["connected_devices"] == 2 for view in views)

    private = device_a.post("/v1/observations", headers=headers_a, json=observation("my umbrella", "personal", False))
    assert private.status_code == 200
    assert private.json()["verification_status"] == "unverified"
    reload_session = headers("device-a", "new-app-session-a")
    assert [item["label"] for item in device_a.get("/v1/personal/observations?query=umbrella", headers=reload_session).json()] == ["my umbrella"]
    assert device_b.get("/v1/personal/observations?query=umbrella", headers=headers_b).json() == []

    changes = device_b.get("/v1/observations/changes?zone_id=Room%20204", headers=headers_b)
    assert changes.status_code == 200
    assert set(changes.json()["added"]) == {"chair", "backpack"}
    assert "my umbrella" not in changes.text


def test_low_confidence_unverified_shared_duplicates_and_tracker_ids_are_rejected(clients):
    client, _ = clients
    auth = headers("device-a", "session-a")

    low = observation("unclear object")
    low["confidence"] = 0.39
    assert client.post("/v1/observations", headers=auth, json=low).status_code == 422

    low_personal = observation("unclear umbrella", "personal", False)
    low_personal["confidence"] = 0.6
    assert client.post("/v1/observations", headers=auth, json=low_personal).status_code == 422

    unverified = observation("unconfirmed chair", verified=False)
    assert client.post("/v1/observations", headers=auth, json=unverified).status_code == 422

    with_track_id = observation("table")
    with_track_id["track_id"] = 17
    assert client.post("/v1/observations", headers=auth, json=with_track_id).status_code == 422

    accepted = client.post("/v1/observations", headers=auth, json=observation("table"))
    assert accepted.status_code == 200
    assert client.post("/v1/observations", headers=auth, json=observation("table")).status_code == 409


def test_concurrent_duplicate_reports_are_serialized(clients):
    client_a, client_b = clients
    request_a = (client_a, headers("device-a", "session-a"))
    request_b = (client_b, headers("device-b", "session-b"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda args: args[0].post("/v1/observations", headers=args[1], json=observation("same chair")),
            [request_a, request_b],
        ))
    assert sorted(response.status_code for response in responses) == [200, 409]


def test_device_allowlist_and_owner_deletion(clients, monkeypatch):
    client, other = clients
    device_a_headers = headers("device-a", "session-a")
    monkeypatch.setenv("SHARED_SIGHT_DEVICE_IDS", device_a_headers["X-Device-ID"])
    saved = client.post("/v1/observations", headers=device_a_headers, json=observation("desk"))
    assert saved.status_code == 200
    assert other.get("/v1/shared-sight/dashboard?zone_id=Room%20204", headers=headers("device-b", "session-b")).status_code == 401
    observation_id = saved.json()["id"]
    assert other.delete(f"/v1/observations/{observation_id}", headers=headers("device-b", "session-b")).status_code == 401
    monkeypatch.setenv("SHARED_SIGHT_DEVICE_IDS", "")
    assert other.delete(f"/v1/observations/{observation_id}", headers=headers("device-b", "session-b")).status_code == 404
    assert client.delete(f"/v1/observations/{observation_id}", headers=headers("device-a", "session-a")).status_code == 200
    assert client.get("/v1/shared-sight/dashboard?zone_id=Room%20204", headers=headers("device-a", "session-a")).json()["recent_observations"] == []

