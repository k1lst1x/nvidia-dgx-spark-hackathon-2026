"""Campus Memory & Shared Sight Engine for GuideLens.

Provides on-campus, multi-student persistent memory for hazards, room
status, and delta change detection (What Changed?) backed by SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("guidelens.memory")

ALERT_TTL_SECONDS = float(os.getenv("CAMPUS_ALERT_TTL_HOURS", "12")) * 3600
DB_PATH = Path(os.getenv("CAMPUS_MEMORY_DB", str(Path(__file__).with_name("campus_memory.db"))))


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize empty campus memory tables; never invent campus reports."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campus_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                category TEXT NOT NULL,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                reported_by TEXT NOT NULL DEFAULT 'anonymous',
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_location
            ON campus_alerts(location, is_active);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS location_baselines (
                location TEXT PRIMARY KEY,
                objects_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.commit()


def _format_relative_time(timestamp: float) -> str:
    diff = max(0, time.time() - timestamp)
    if diff < 60:
        return "just now"
    minutes = int(diff // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours // 24)
    return f"{days}d ago"


def get_active_alerts(location: str | None = None) -> list[dict[str, Any]]:
    """Retrieve active campus alerts, optionally filtered by location."""
    with _get_connection() as conn:
        if location and location.strip():
            cursor = conn.execute(
                """
                SELECT id, location, category, label, description, severity, reported_by, created_at, is_active
                FROM campus_alerts
                WHERE is_active = 1 AND created_at > ? AND (location LIKE ? OR location = 'All Campus')
                ORDER BY created_at DESC;
                """,
                (time.time() - ALERT_TTL_SECONDS, f"%{location.strip()}%",),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, location, category, label, description, severity, reported_by, created_at, is_active
                FROM campus_alerts
                WHERE is_active = 1 AND created_at > ?
                ORDER BY created_at DESC;
                """,
                (time.time() - ALERT_TTL_SECONDS,),
            )
        rows = cursor.fetchall()
        alerts = []
        for row in rows:
            alerts.append({
                "id": row["id"],
                "location": row["location"],
                "category": row["category"],
                "label": row["label"],
                "description": row["description"],
                "severity": row["severity"],
                "reported_by": row["reported_by"],
                "created_at": datetime.fromtimestamp(row["created_at"], timezone.utc).isoformat(),
                "relative_time": _format_relative_time(row["created_at"]),
                "is_active": bool(row["is_active"]),
            })
        return alerts


def add_alert(
    location: str,
    category: str,
    label: str,
    description: str,
    severity: str = "medium",
    reported_by: str = "anonymous",
) -> dict[str, Any]:
    """Record a new observation from any student phone into campus memory."""
    now = time.time()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO campus_alerts
            (location, category, label, description, severity, reported_by, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1);
            """,
            (location, category, label, description, severity, reported_by, now),
        )
        conn.commit()
        alert_id = cursor.lastrowid

    logger.info("campus_alert_added id=%d location=%s label=%s", alert_id, location, label)
    return {
        "id": alert_id,
        "location": location,
        "category": category,
        "label": label,
        "description": description,
        "severity": severity,
        "reported_by": reported_by,
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "relative_time": "just now",
        "is_active": True,
    }


def resolve_alert(alert_id: int) -> bool:
    """Mark a campus hazard as cleared/resolved."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "UPDATE campus_alerts SET is_active = 0 WHERE id = ?;", (alert_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def has_baseline(location: str) -> bool:
    """Check if a baseline is already established for the given location."""
    with _get_connection() as conn:
        cursor = conn.execute("SELECT 1 FROM location_baselines WHERE location = ? LIMIT 1;", (location,))
        return cursor.fetchone() is not None


def _normalize_object(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return dict(item)


def detect_what_changed(
    location: str,
    current_objects: list[dict[str, Any] | Any],
    update_baseline: bool = False,
) -> dict[str, Any]:
    """Compare currently detected objects against stored baseline for a room/location.

    Returns the geometric delta (what moved, what's new) instead of re-describing
    the static environment.
    """
    now = time.time()
    normalized_objects = [_normalize_object(item) for item in current_objects]
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT objects_json, updated_at FROM location_baselines WHERE location = ?;",
            (location,),
        )
        row = cursor.fetchone()

        if not row:
            # First time visiting this location: save current layout as baseline
            conn.execute(
                "INSERT INTO location_baselines (location, objects_json, updated_at) VALUES (?, ?, ?);",
                (location, json.dumps(normalized_objects), now),
            )
            conn.commit()
            return {
                "has_changed": False,
                "delta_summary": f"Baseline established for {location} with {len(normalized_objects)} objects.",
                "speech": f"Baseline established for {location}. I will only announce what changes from now on.",
                "added_objects": [],
                "removed_objects": [],
                "moved_objects": [],
            }

        baseline_objects = json.loads(row["objects_json"])

        if update_baseline:
            conn.execute(
                "UPDATE location_baselines SET objects_json = ?, updated_at = ? WHERE location = ?;",
                (json.dumps(normalized_objects), now, location),
            )
            conn.commit()

    # Compare current objects with baseline in linear O(N) time using Counter
    baseline_labels = [str(obj.get("label", "")).lower() for obj in baseline_objects if obj.get("label")]
    current_labels = [str(obj.get("label", "")).lower() for obj in normalized_objects if obj.get("label")]

    baseline_counts = Counter(baseline_labels)
    current_counts = Counter(current_labels)

    added: list[str] = [label for label, count in current_counts.items() if count > baseline_counts[label]]
    removed: list[str] = [label for label, count in baseline_counts.items() if count > current_counts[label]]

    # Detect moved obstacles in linear O(N) time via hash lookup
    baseline_x: dict[str, float] = {}
    for b in baseline_objects:
        lbl = str(b.get("label", "")).lower()
        if lbl and lbl not in baseline_x:
            baseline_x[lbl] = float(b.get("x", 0.5))

    moved: list[str] = []
    for curr in normalized_objects:
        lbl = str(curr.get("label", "")).lower()
        if lbl in baseline_x and lbl not in moved:
            curr_x = float(curr.get("x", 0.5))
            if abs(curr_x - baseline_x[lbl]) > 0.20:
                moved.append(lbl)

    has_changed = bool(added or moved)

    if not has_changed:
        speech = f"Everything in {location} is in its usual place. Your path is clear."
        delta_summary = "No changes detected since earlier."
    else:
        parts = []
        if added:
            items_str = ", ".join(added[:2])
            parts.append(f"Something new is in your path: {items_str}")
        if moved:
            items_str = ", ".join(moved[:2])
            parts.append(f"A {items_str} on your usual path has moved")

        speech = f"Notice in {location}: " + ". ".join(parts) + "."
        delta_summary = "; ".join(parts)

    return {
        "has_changed": has_changed,
        "delta_summary": delta_summary,
        "speech": speech,
        "added_objects": added,
        "removed_objects": removed,
        "moved_objects": moved,
    }


# Deaf-Blind bridge messages moved to observation_store.py (the shared store).


# Initialize schema and seed data on module import
init_db()
