"""Persistent, privacy-scoped observations for Shared Sight and Personal Recall.

Only semantic labels and metadata are stored. Camera frames and temporary
BoT-SORT IDs are deliberately excluded from this store.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("VISIONMATE_MEMORY_DB", str(Path(__file__).with_name("visionmate_memory.sqlite3"))))
MIN_CONFIDENCE = float(os.getenv("OBSERVATION_MIN_CONFIDENCE", "0.65"))
VERIFIED_SHARED_MIN_CONFIDENCE = float(os.getenv("OBSERVATION_VERIFIED_SHARED_MIN_CONFIDENCE", "0.40"))
DEDUP_SECONDS = 30
# Deaf-Blind bridge: a conversation is one pair of devices (or one laptop) and
# is forgotten after this long without a new message.
BRIDGE_TTL_SECONDS = float(os.getenv("BRIDGE_TTL_HOURS", "12")) * 3600
BRIDGE_MAX_DEVICES = 2
BRIDGE_PAGE_LIMIT = 100
TRACKING_METRICS: deque[tuple[float, int]] = deque(maxlen=1000)


class ObservationAccessError(PermissionError):
    pass


class DuplicateObservationError(ValueError):
    pass


class BridgeNotFoundError(LookupError):
    pass


class BridgeAccessError(PermissionError):
    pass


class BridgeFullError(PermissionError):
    pass


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def db_connection():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                label TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                zone_id TEXT NOT NULL,
                observed_at REAL NOT NULL,
                source_device TEXT NOT NULL,
                source_session TEXT NOT NULL,
                confidence REAL NOT NULL,
                verification_status TEXT NOT NULL CHECK (verification_status IN ('verified','unverified')),
                privacy_scope TEXT NOT NULL CHECK (privacy_scope IN ('personal','shared')),
                room_x REAL,
                room_y REAL,
                expires_at REAL NOT NULL,
                origin TEXT NOT NULL CHECK (origin IN ('manual','detector')),
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_observation_shared_zone_time
                ON observations(privacy_scope, verification_status, zone_id, observed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_observation_owner_time
                ON observations(source_session, privacy_scope, observed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_observation_expiry ON observations(expires_at);
            CREATE TABLE IF NOT EXISTS sight_devices (
                device_id TEXT PRIMARY KEY,
                last_seen REAL NOT NULL,
                last_session TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS device_credentials (
                device_id TEXT PRIMARY KEY,
                secret_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bridge_conversations (
                code TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                created_by TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bridge_members (
                code TEXT NOT NULL REFERENCES bridge_conversations(code) ON DELETE CASCADE,
                device_id TEXT NOT NULL,
                joined_at REAL NOT NULL,
                PRIMARY KEY (code, device_id)
            );
            CREATE TABLE IF NOT EXISTS bridge_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL REFERENCES bridge_conversations(code) ON DELETE CASCADE,
                sender_role TEXT NOT NULL CHECK (sender_role IN ('deaf','blind')),
                text TEXT NOT NULL,
                source_device TEXT NOT NULL,
                client_id TEXT,
                created_at REAL NOT NULL,
                UNIQUE (code, client_id)
            );
            CREATE INDEX IF NOT EXISTS idx_bridge_messages_code_id ON bridge_messages(code, id);
        """)
        # Additive migration for databases created before room sketches existed.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(observations)")}
        if "room_x" not in columns:
            conn.execute("ALTER TABLE observations ADD COLUMN room_x REAL")
        if "room_y" not in columns:
            conn.execute("ALTER TABLE observations ADD COLUMN room_y REAL")


def authorize_device(
    device_id: str | None,
    session_id: str | None,
    device_secret: str | None,
    shared_key: str | None,
) -> tuple[str, str]:
    if not device_id or not session_id or len(device_id) > 100 or len(session_id) > 100:
        raise ObservationAccessError("X-Device-ID and X-Session-ID headers are required")
    if not device_secret or not 64 <= len(device_secret) <= 200:
        raise ObservationAccessError("A valid X-Device-Secret header is required")
    allowed = {value.strip() for value in os.getenv("SHARED_SIGHT_DEVICE_IDS", "").split(",") if value.strip()}
    if allowed and device_id not in allowed:
        raise ObservationAccessError("This device is not authorized for Shared Sight")
    expected_key = os.getenv("SHARED_SIGHT_TOKEN", "")
    if expected_key and not hmac.compare_digest(shared_key or "", expected_key):
        raise ObservationAccessError("Shared Sight access key is invalid")
    supplied_hash = hashlib.sha256(device_secret.encode("utf-8")).hexdigest()
    derived_device_id = f"device-{device_secret[:32]}"
    if hmac.compare_digest(device_id, derived_device_id):
        return device_id, session_id

    # Non-derived IDs are supported only for a one-time migration of databases
    # created before device-bound credentials existed. Fresh identities never
    # allocate a database row, so arbitrary enrollment cannot grow the store.
    with db_connection() as conn:
        credential = conn.execute(
            "SELECT secret_hash FROM device_credentials WHERE device_id = ?", (device_id,),
        ).fetchone()
        if credential is None:
            # Fresh random IDs enroll on first use. An ID that already owns data
            # must prove the deployment's migration key before it can be bound;
            # otherwise an upgrade would let the first caller steal old records.
            owns_existing_state = any((
                conn.execute("SELECT 1 FROM observations WHERE source_device = ? LIMIT 1", (device_id,)).fetchone(),
                conn.execute("SELECT 1 FROM sight_devices WHERE device_id = ? LIMIT 1", (device_id,)).fetchone(),
                conn.execute("SELECT 1 FROM bridge_members WHERE device_id = ? LIMIT 1", (device_id,)).fetchone(),
            ))
            if not owns_existing_state or not expected_key:
                raise ObservationAccessError(
                    "Device ID does not match its secret; legacy IDs require SHARED_SIGHT_TOKEN migration"
                )
            conn.execute(
                "INSERT OR IGNORE INTO device_credentials(device_id, secret_hash, created_at) VALUES(?,?,?)",
                (device_id, supplied_hash, time.time()),
            )
            credential = conn.execute(
                "SELECT secret_hash FROM device_credentials WHERE device_id = ?", (device_id,),
            ).fetchone()
        if not hmac.compare_digest(credential["secret_hash"], supplied_hash):
            raise ObservationAccessError("Device credentials are invalid")
    return device_id, session_id


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "category": row["category"], "label": row["label"],
        "description": row["description"], "zone_id": row["zone_id"],
        "room_x": row["room_x"], "room_y": row["room_y"],
        "timestamp": _iso(row["observed_at"]), "source_device": row["source_device"],
        "source_session": row["source_session"], "confidence": row["confidence"],
        "verification_status": row["verification_status"], "privacy_scope": row["privacy_scope"],
        "expires_at": _iso(row["expires_at"]), "origin": row["origin"],
    }


def cleanup_expired(now: float | None = None) -> int:
    current = now or time.time()
    with db_connection() as conn:
        result = conn.execute("DELETE FROM observations WHERE expires_at <= ?", (current,))
        conn.execute("DELETE FROM sight_devices WHERE last_seen < ?", (current - 86400,))
        conn.execute("DELETE FROM bridge_conversations WHERE expires_at <= ?", (current,))
        return result.rowcount


def touch_device(device_id: str, session_id: str, now: float | None = None) -> None:
    current = now or time.time()
    with db_connection() as conn:
        conn.execute("""
            INSERT INTO sight_devices(device_id,last_seen,last_session) VALUES(?,?,?)
            ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen,last_session=excluded.last_session
        """, (device_id, current, session_id))


def record_tracking_latency(latency_ms: int, now: float | None = None) -> None:
    current = now or time.time()
    TRACKING_METRICS.append((current, max(0, latency_ms)))


def create_observation(
    *, zone_id: str, category: str, label: str, description: str, confidence: float,
    privacy_scope: str, verified: bool, expires_in_seconds: int, origin: str,
    source_device: str, source_session: str, room_x: float | None = None,
    room_y: float | None = None, now: float | None = None,
) -> dict[str, Any]:
    if privacy_scope == "shared" and not verified:
        raise ValueError("Shared Sight accepts only user-verified observations")
    minimum_confidence = (
        VERIFIED_SHARED_MIN_CONFIDENCE if privacy_scope == "shared" and verified else MIN_CONFIDENCE
    )
    if confidence < minimum_confidence:
        raise ValueError(f"Observation confidence must be at least {minimum_confidence:.2f}")
    if (room_x is None) != (room_y is None):
        raise ValueError("room_x and room_y must be provided together")
    current = now or time.time()
    verification = "verified" if verified else "unverified"
    normalized_label = label.strip()
    with db_connection() as conn:
        # Serialize the deduplication read and insert so two devices posting
        # the same report at once cannot both pass the duplicate check.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM observations WHERE expires_at <= ?", (current,))
        duplicate = conn.execute("""
            SELECT id FROM observations
            WHERE zone_id=? AND lower(category)=lower(?) AND lower(label)=lower(?)
              AND privacy_scope=? AND active=1 AND observed_at >= ?
            ORDER BY observed_at DESC LIMIT 1
        """, (zone_id, category, normalized_label, privacy_scope, current - DEDUP_SECONDS)).fetchone()
        if duplicate:
            raise DuplicateObservationError(f"Duplicate of recent observation #{duplicate['id']}")
        cursor = conn.execute("""
            INSERT INTO observations(category,label,description,zone_id,observed_at,source_device,
                source_session,confidence,verification_status,privacy_scope,room_x,room_y,expires_at,origin)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            category.strip().lower(), normalized_label, description.strip(), zone_id.strip(), current,
            source_device, source_session, confidence, verification, privacy_scope,
            room_x, room_y, current + expires_in_seconds, origin,
        ))
        row = conn.execute("SELECT * FROM observations WHERE id=?", (cursor.lastrowid,)).fetchone()
        conn.execute("""
            INSERT INTO sight_devices(device_id,last_seen,last_session) VALUES(?,?,?)
            ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen,last_session=excluded.last_session
        """, (source_device, current, source_session))
        return _serialize(row)


def list_shared(zone_id: str, limit: int = 30, now: float | None = None) -> list[dict[str, Any]]:
    cleanup_expired(now)
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM observations
            WHERE privacy_scope='shared' AND verification_status='verified' AND active=1
              AND zone_id=? AND expires_at > ?
            ORDER BY observed_at DESC LIMIT ?
        """, (zone_id, now or time.time(), max(1, min(limit, 100)))).fetchall()
        return [_serialize(row) for row in rows]


# Words that frame a question but name nothing: "where are my keys" -> "keys".
_RECALL_STOP_WORDS = {
    "a", "an", "the", "my", "mine", "me", "i", "i'm", "im", "you", "your", "is", "are", "was", "were", "be",
    "where", "what", "when", "which", "who", "how", "did", "do", "does", "have", "has", "had", "can", "could",
    "find", "found", "leave", "left", "put", "placed", "see", "seen", "saw", "last", "lost", "of", "to", "in",
    "on", "at", "for", "it", "its", "please", "help", "tell", "show", "remember", "again",
}


def _recall_terms(query: str) -> list[str]:
    """Content words of a question, singular, for matching saved labels."""
    words = re.findall(r"[a-z0-9']+", query.lower())
    terms = []
    for word in words:
        if word in _RECALL_STOP_WORDS or len(word) < 2:
            continue
        stem = word[:-2] if word.endswith("es") and len(word) > 4 else word[:-1] if word.endswith("s") and len(word) > 3 else word
        if stem not in terms:
            terms.append(stem)
    return terms


def search_personal(device_id: str, query: str = "", limit: int = 50, now: float | None = None) -> list[dict[str, Any]]:
    """This device's private notes that best match a question, best match first."""
    cleanup_expired(now)
    current = now or time.time()
    terms = _recall_terms(query)
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM observations
            WHERE privacy_scope='personal' AND source_device=? AND active=1 AND expires_at > ?
            ORDER BY observed_at DESC LIMIT 500
        """, (device_id, current)).fetchall()
    if not terms:
        return [_serialize(row) for row in rows[: max(1, min(limit, 100))]]
    scored = []
    for row in rows:
        text = " ".join(str(row[field] or "").lower() for field in ("label", "description", "category", "zone_id"))
        label = str(row["label"] or "").lower()
        score = sum(2 if term in label else 1 for term in terms if term in text)
        if score:
            scored.append((score, row["observed_at"], row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [_serialize(row) for _score, _at, row in scored[: max(1, min(limit, 100))]]


def delete_owned(observation_id: int, source_device: str) -> bool:
    with db_connection() as conn:
        result = conn.execute("UPDATE observations SET active=0 WHERE id=? AND source_device=?", (observation_id, source_device))
        return result.rowcount > 0


def changes(zone_id: str, compared_seconds: int = 3600, now: float | None = None) -> dict[str, Any]:
    current = now or time.time()
    window = max(60, min(compared_seconds, 86400))
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT label, observed_at FROM observations
            WHERE zone_id=? AND privacy_scope='shared' AND verification_status='verified'
              AND active=1 AND expires_at>? AND observed_at>=?
            ORDER BY observed_at DESC
        """, (zone_id, current, current - window * 2)).fetchall()
    recent = {row["label"] for row in rows if row["observed_at"] >= current - window}
    previous = {row["label"] for row in rows if current - window * 2 <= row["observed_at"] < current - window}
    added = sorted(recent - previous)
    previously_seen = sorted(recent & previous)
    summary = (
        "New verified reports: " + ", ".join(added[:5]) if added else
        ("Recent verified reports are unchanged: " + ", ".join(previously_seen[:5]) if previously_seen else
         f"No verified observations in the last {window // 60} minutes.")
    )
    return {
        "zone_id": zone_id, "generated_at": _iso(current), "compared_seconds": window,
        "added": added, "previously_seen": previously_seen, "summary": summary, "verified_only": True,
    }


def dashboard(zone_id: str, limit: int = 20, now: float | None = None) -> dict[str, Any]:
    current = now or time.time()
    observations = list_shared(zone_id, limit, current)
    with db_connection() as conn:
        connected = conn.execute("SELECT COUNT(*) AS n FROM sight_devices WHERE last_seen >= ?", (current - 120,)).fetchone()["n"]
    values = [latency for timestamp, latency in TRACKING_METRICS if timestamp >= current - 300]
    average = round(sum(values) / len(values)) if values else None
    return {
        "zone_id": zone_id, "connected_devices": connected, "recent_observations": observations,
        "tracking_latency_ms": average, "tracking_requests_last_5m": len(values),
    }


# ---------------------------------------------------------------------------
# Campus alerts: the legacy /v1/memory/* API, now backed by shared observations
# ---------------------------------------------------------------------------
# Before, alerts lived in campus_memory.py while the camera's share button
# wrote here, so phone B never heard phone A's report. Every report path now
# writes and reads these shared, verified observations.

ALERT_TTL_SECONDS = float(os.getenv("CAMPUS_ALERT_TTL_HOURS", "12")) * 3600
_ALERT_CATEGORIES = {"hazard", "obstacle", "sign", "room_status"}
_ALERT_SEVERITY = {"hazard": "high", "obstacle": "medium"}


def _as_alert(row: sqlite3.Row, viewer_device: str, now: float) -> dict[str, Any]:
    category = row["category"] if row["category"] in _ALERT_CATEGORIES else "obstacle"
    return {
        "id": row["id"],
        "location": row["zone_id"],
        "category": category,
        "label": row["label"],
        "description": row["description"] or row["label"],
        "severity": _ALERT_SEVERITY.get(category, "low"),
        # Never expose another phone's identifiers; say whose report it is.
        "reported_by": "you" if row["source_device"] == viewer_device else "another student",
        "created_at": _iso(row["observed_at"]),
        "relative_time": _relative_time(row["observed_at"], now),
        "is_active": True,
    }


def shared_alerts(viewer_device: str, location: str | None = None, limit: int = 20, now: float | None = None) -> list[dict[str, Any]]:
    """Active shared, verified reports, newest first, optionally for one room."""
    current = now or time.time()
    cleanup_expired(current)
    where = "privacy_scope='shared' AND verification_status='verified' AND active=1 AND expires_at > ?"
    params: list[Any] = [current]
    if location and location.strip() and location.strip() != "All Campus":
        where += " AND zone_id LIKE ?"
        params.append(f"%{location.strip()}%")
    params.append(max(1, min(limit, 100)))
    with db_connection() as conn:
        rows = conn.execute(f"SELECT * FROM observations WHERE {where} ORDER BY observed_at DESC LIMIT ?", params).fetchall()
    return [_as_alert(row, viewer_device, current) for row in rows]


def report_alert(
    *, location: str, category: str, label: str, description: str,
    source_device: str, source_session: str, now: float | None = None,
) -> dict[str, Any]:
    """A report a person typed or confirmed: shared and verified by definition."""
    created = create_observation(
        zone_id=location, category=category, label=label, description=description, confidence=1.0,
        privacy_scope="shared", verified=True, expires_in_seconds=int(ALERT_TTL_SECONDS), origin="manual",
        source_device=source_device, source_session=source_session, now=now,
    )
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM observations WHERE id=?", (created["id"],)).fetchone()
    return _as_alert(row, source_device, now or time.time())


def resolve_shared(observation_id: int) -> bool:
    """Mark a shared report as cleared (anyone on campus can clear a hazard)."""
    with db_connection() as conn:
        result = conn.execute(
            "UPDATE observations SET active=0 WHERE id=? AND privacy_scope='shared' AND active=1", (observation_id,),
        )
        return result.rowcount > 0

# ---------------------------------------------------------------------------
# Deaf-Blind bridge
# ---------------------------------------------------------------------------


def _relative_time(timestamp: float, now: float) -> str:
    seconds = max(0, int(now - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def _bridge_conversation(conn: sqlite3.Connection, code: str, device_id: str, now: float) -> dict[str, Any]:
    row = conn.execute(
        "SELECT code, created_at, expires_at FROM bridge_conversations WHERE code = ? AND expires_at > ?",
        (code, now),
    ).fetchone()
    if row is None:
        raise BridgeNotFoundError("No active bridge conversation with that code")
    members = [r["device_id"] for r in conn.execute(
        "SELECT device_id FROM bridge_members WHERE code = ? ORDER BY joined_at", (code,)
    )]
    return {
        "code": row["code"],
        "created_at": _iso(row["created_at"]),
        "expires_at": _iso(row["expires_at"]),
        "devices": len(members),
        "max_devices": BRIDGE_MAX_DEVICES,
        "partner_joined": len(members) >= 2,
        "is_member": device_id in members,
    }


def _require_member(conn: sqlite3.Connection, code: str, device_id: str, now: float) -> dict[str, Any]:
    conversation = _bridge_conversation(conn, code, device_id, now)
    if not conversation["is_member"]:
        raise BridgeAccessError("This device has not joined that bridge conversation")
    return conversation


def create_bridge(device_id: str, now: float | None = None) -> dict[str, Any]:
    """Start a conversation and return its 6-digit code (easy to read aloud)."""
    current = now or time.time()
    with db_connection() as conn:
        conn.execute("DELETE FROM bridge_conversations WHERE expires_at <= ?", (current,))
        for _attempt in range(20):
            code = f"{secrets.randbelow(1_000_000):06d}"
            try:
                conn.execute(
                    "INSERT INTO bridge_conversations(code, created_at, expires_at, created_by) VALUES (?,?,?,?)",
                    (code, current, current + BRIDGE_TTL_SECONDS, device_id),
                )
            except sqlite3.IntegrityError:
                continue
            conn.execute(
                "INSERT INTO bridge_members(code, device_id, joined_at) VALUES (?,?,?)",
                (code, device_id, current),
            )
            return _bridge_conversation(conn, code, device_id, current)
    raise RuntimeError("Could not allocate a bridge code")


def join_bridge(code: str, device_id: str, now: float | None = None) -> dict[str, Any]:
    """Join as the second device. Rejoining from the same device is a no-op."""
    current = now or time.time()
    with db_connection() as conn:
        conversation = _bridge_conversation(conn, code, device_id, current)
        if conversation["is_member"]:
            return conversation
        if conversation["devices"] >= BRIDGE_MAX_DEVICES:
            raise BridgeFullError("That bridge conversation already has two devices")
        conn.execute(
            "INSERT INTO bridge_members(code, device_id, joined_at) VALUES (?,?,?)",
            (code, device_id, current),
        )
        return _bridge_conversation(conn, code, device_id, current)


def bridge_status(code: str, device_id: str, now: float | None = None) -> dict[str, Any]:
    current = now or time.time()
    with db_connection() as conn:
        return _require_member(conn, code, device_id, current)


def _serialize_bridge_message(row: sqlite3.Row, device_id: str, now: float) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["code"],
        "sender_role": row["sender_role"],
        "text": row["text"],
        "client_id": row["client_id"],
        "from_this_device": row["source_device"] == device_id,
        "created_at": _iso(row["created_at"]),
        "relative_time": _relative_time(row["created_at"], now),
    }


def add_bridge_message(
    code: str,
    device_id: str,
    sender_role: str,
    text: str,
    client_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Store a message. A retried send with the same client_id returns the original."""
    current = now or time.time()
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Message text is empty")
    with db_connection() as conn:
        _require_member(conn, code, device_id, current)
        if client_id:
            existing = conn.execute(
                "SELECT * FROM bridge_messages WHERE code = ? AND client_id = ?", (code, client_id)
            ).fetchone()
            if existing is not None:
                return _serialize_bridge_message(existing, device_id, current)
        cursor = conn.execute(
            """INSERT INTO bridge_messages(code, sender_role, text, source_device, client_id, created_at)
               VALUES (?,?,?,?,?,?)""",
            (code, sender_role, cleaned, device_id, client_id, current),
        )
        # Every message keeps the conversation alive for another TTL window.
        conn.execute(
            "UPDATE bridge_conversations SET expires_at = ? WHERE code = ?",
            (current + BRIDGE_TTL_SECONDS, code),
        )
        row = conn.execute("SELECT * FROM bridge_messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _serialize_bridge_message(row, device_id, current)


def list_bridge_messages(
    code: str,
    device_id: str,
    after_id: int | None = None,
    limit: int = 50,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Messages in time order: the newest ``limit``, or only those after ``after_id``."""
    current = now or time.time()
    limit = max(1, min(BRIDGE_PAGE_LIMIT, limit))
    with db_connection() as conn:
        _require_member(conn, code, device_id, current)
        if after_id is not None:
            rows = conn.execute(
                "SELECT * FROM bridge_messages WHERE code = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (code, after_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bridge_messages WHERE code = ? ORDER BY id DESC LIMIT ?",
                (code, limit),
            ).fetchall()[::-1]
        return [_serialize_bridge_message(row, device_id, current) for row in rows]


init_db()
