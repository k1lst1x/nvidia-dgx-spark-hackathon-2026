"""Pydantic request and response contracts for the VisionMate API."""

from typing import Literal

from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    label: str = Field(max_length=40)
    confidence: float = Field(ge=0, le=1)
    position: str = Field(max_length=30)
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    width: float | None = Field(default=None, ge=0, le=1)
    height: float | None = Field(default=None, ge=0, le=1)
    distance_m: float | None = Field(default=None, gt=0, le=80)
    relative_proximity: float | None = Field(default=None, ge=0, le=1)
    track_id: int | None = Field(default=None, ge=1)
    motion: dict[str, object] | None = None


class ApproachingWarning(BaseModel):
    label: str = Field(max_length=40)
    distance_m: float = Field(gt=0, le=80)


class VisionResult(BaseModel):
    summary: str = Field(max_length=100)
    speech: str = Field(max_length=240)
    scene_key: str = Field(max_length=80)
    urgency: Literal["low", "medium", "high"]
    changed: bool
    objects: list[DetectedObject] = Field(max_length=8)
    approaching: list[ApproachingWarning] = Field(default_factory=list, max_length=3)
    task: Literal["scene", "sign"]


class TrackingResult(BaseModel):
    objects: list[DetectedObject] = Field(max_length=12)
    tracking_latency_ms: int = Field(ge=0)
    detector: Literal["yolo11-botsort", "disabled"]
    metric_depth: bool
    experimental: bool = True
    collision_warnings_enabled: bool = False


class SignSequenceRequest(BaseModel):
    frames: list[str] = Field(min_length=3, max_length=3)
    previous_transcript: str = Field(default="", max_length=600)


class SignTranslation(BaseModel):
    transcript: str = Field(max_length=600)
    segment: str = Field(max_length=180)
    confidence: float = Field(ge=0, le=1)
    is_final: bool
    should_speak: bool


class UniSignRawRequest(BaseModel):
    raw_holistic: list[dict[str, object]] = Field(min_length=1, max_length=1024)
    experimental: bool = False


class CampusAlert(BaseModel):
    id: int
    location: str
    category: str
    label: str
    description: str
    severity: Literal["low", "medium", "high"]
    reported_by: str
    created_at: str
    relative_time: str
    is_active: bool


class CampusAlertCreateRequest(BaseModel):
    location: str = Field(min_length=1, max_length=80)
    category: Literal["hazard", "obstacle", "sign", "room_status"] = "hazard"
    label: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=200)
    severity: Literal["low", "medium", "high"] = "medium"


class WhatChangedRequest(BaseModel):
    location: str = Field(default="Room 204", max_length=80)
    current_objects: list[dict[str, object]] = Field(default_factory=list)
    update_baseline: bool = False


class WhatChangedResponse(BaseModel):
    has_changed: bool
    delta_summary: str
    speech: str
    added_objects: list[str]
    removed_objects: list[str]
    moved_objects: list[str]


class BridgeMessageRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    sender_role: Literal["deaf", "blind"]
    text: str = Field(min_length=1, max_length=500)


class BridgeMessage(BaseModel):
    id: int
    session_id: str
    sender_role: str
    text: str
    created_at: str
    relative_time: str

