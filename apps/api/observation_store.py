"""Persistent, privacy-scoped observations for Shared Sight and Personal Recall.

Only semantic labels and metadata are stored. Camera frames and temporary
BoT-SORT IDs are deliberately excluded from this store.
"""

from __future__ import annotations

import hmac
import os
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("VISIONMATE_MEMORY_DB", str(Path(__file__).with_name("visionmate_memory.sqlite3"))))
MIN_CONFIDENCE = float(os.getenv("OBSERVATION_MIN_CONFIDENCE", "0.65"))
DEDUP_SECONDS = 30
TRACKING_METRICS: deque[tuple[float, int]] = deque(maxlen=1000)


class ObservationAccessError(PermissionError):
    pass


class DuplicateObservationError(ValueError):
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
        """)


def authorize_device(device_id: str | None, session_id: str | None, shared_key: str | None) -> tuple[str, str]:
    if not device_id or not session_id or len(device_id) > 100 or len(session_id) > 100:
        raise ObservationAccessError("X-Device-ID and X-Session-ID headers are required")
    allowed = {value.strip() for value in os.getenv("SHARED_SIGHT_DEVICE_IDS", "").split(",") if value.strip()}
    if allowed and device_id not in allowed:
        raise ObservationAccessError("This device is not authorized for Shared Sight")
    expected_key = os.getenv("SHARED_SIGHT_TOKEN", "")
    if expected_key and not hmac.compare_digest(shared_key or "", expected_key):
        raise ObservationAccessError("Shared Sight access key is invalid")
    return device_id, session_id


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"], "category": row["category"], "label": row["label"],
        "description": row["description"], "zone_id": row["zone_id"],
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
    source_device: str, source_session: str, now: float | None = None,
) -> dict[str, Any]:
    if confidence < MIN_CONFIDENCE:
        raise ValueError(f"Observation confidence must be at least {MIN_CONFIDENCE:.2f}")
    if privacy_scope == "shared" and not verified:
        raise ValueError("Shared Sight accepts only user-verified observations")
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
                source_session,confidence,verification_status,privacy_scope,expires_at,origin)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            category.strip().lower(), normalized_label, description.strip(), zone_id.strip(), current,
            source_device, source_session, confidence, verification, privacy_scope,
            current + expires_in_seconds, origin,
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


def search_personal(device_id: str, query: str = "", limit: int = 50, now: float | None = None) -> list[dict[str, Any]]:
    cleanup_expired(now)
    current = now or time.time()
    needle = f"%{query.strip()}%"
    with db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM observations
            WHERE privacy_scope='personal' AND source_device=? AND active=1 AND expires_at > ?
              AND (?='' OR label LIKE ? OR category LIKE ? OR description LIKE ? OR zone_id LIKE ?)
            ORDER BY observed_at DESC LIMIT ?
        """, (device_id, current, query.strip(), needle, needle, needle, needle, max(1, min(limit, 100)))).fetchall()
        return [_serialize(row) for row in rows]


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


init_db()
