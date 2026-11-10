"""Pydantic request and response contracts for the VisionMate API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    spatial: dict[str, object] | None = None


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


class CameraCoach(BaseModel):
    blur: bool
    poor_lighting: bool
    overexposed: bool
    cropped_labels: list[str]
    sharpness_score: float
    brightness_score: float


class TrackingResult(BaseModel):
    objects: list[DetectedObject] = Field(max_length=12)
    tracking_latency_ms: int = Field(ge=0)
    detector: Literal["yolo11-botsort", "disabled"]
    metric_depth: bool
    camera_coach: CameraCoach | None = None
    experimental: bool = True
    collision_warnings_enabled: bool = False
    segmentation: bool = False
    point_cloud: bool = False
    camera_pose: dict[str, object] | None = None


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


class ObservationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    confidence: float = Field(ge=0, le=1)
    privacy_scope: Literal["personal", "shared"] = "personal"
    verified: bool = False
    room_x: float | None = Field(default=None, ge=0, le=1)
    room_y: float | None = Field(default=None, ge=0, le=1)
    expires_in_seconds: int = Field(default=86400, ge=60, le=2592000)
    origin: Literal["manual", "detector"] = "manual"


class Observation(BaseModel):
    id: int
    category: str
    label: str
    description: str
    zone_id: str
    timestamp: str
    source_device: str
    source_session: str
    confidence: float
    verification_status: Literal["verified", "unverified"]
    privacy_scope: Literal["personal", "shared"]
    room_x: float | None = Field(default=None, ge=0, le=1)
    room_y: float | None = Field(default=None, ge=0, le=1)
    expires_at: str
    origin: Literal["manual", "detector"]


class ObservationChanges(BaseModel):
    zone_id: str
    generated_at: str
    compared_seconds: int
    added: list[str]
    previously_seen: list[str]
    summary: str
    verified_only: bool = True


class SharedSightDashboard(BaseModel):
    zone_id: str
    connected_devices: int
    recent_observations: list[Observation]
    tracking_latency_ms: int | None = None
    tracking_requests_last_5m: int = 0

