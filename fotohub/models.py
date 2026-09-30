"""Pydantic response models for the FOTOhub API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field


# --- Enums ---


class VideoJobStatus(str, Enum):
    """Status of an async video generation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatRole(str, Enum):
    """Role in a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# --- Image Generation ---


class ImageResult(BaseModel):
    """Result from image generation."""

    url: str = Field(description="URL of the generated image")
    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")
    model: str = Field(description="Model used for generation")
    seed: Optional[int] = Field(default=None, description="Seed used for generation")
    credits_used: float = Field(default=0, description="Credits consumed")
    generation_time_ms: Optional[int] = Field(
        default=None, description="Generation time in milliseconds"
    )

    model_config = {"extra": "allow"}


class ImageGenerationResponse(BaseModel):
    """Full response from image generation endpoint."""

    success: bool = True
    images: list[ImageResult] = Field(default_factory=list)
    model: str = Field(default="")
    credits_used: float = Field(default=0)

    model_config = {"extra": "allow"}


# --- Video Generation ---


class VideoJob(BaseModel):
    """Video generation job status."""

    job_id: str = Field(description="Unique job identifier")
    status: VideoJobStatus = Field(description="Current job status")
    progress: Optional[float] = Field(
        default=None, description="Progress percentage (0-100)"
    )
    video_url: Optional[str] = Field(
        default=None, description="URL of the generated video (when completed)"
    )
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail URL")
    model: Optional[str] = Field(default=None, description="Model used")
    duration: Optional[float] = Field(
        default=None, description="Video duration in seconds"
    )
    credits_used: Optional[float] = Field(default=None, description="Credits consumed")
    error: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )
    created_at: Optional[datetime] = Field(default=None, description="Job creation time")
    completed_at: Optional[datetime] = Field(
        default=None, description="Job completion time"
    )

    model_config = {"extra": "allow"}


# --- Music Generation ---


class MusicResult(BaseModel):
    """Result from music generation."""

    url: str = Field(description="URL of the generated audio")
    duration: float = Field(description="Duration in seconds")
    model: str = Field(description="Model used for generation")
    credits_used: float = Field(default=0, description="Credits consumed")
    sample_rate: Optional[int] = Field(default=None, description="Sample rate in Hz")
    format: Optional[str] = Field(default=None, description="Audio format (mp3, wav)")

    model_config = {"extra": "allow"}


class MusicGenerationResponse(BaseModel):
    """Full response from music generation endpoint."""

    success: bool = True
    audio: Optional[MusicResult] = None
    credits_used: float = Field(default=0)

    model_config = {"extra": "allow"}


# --- Chat / LLM ---


class ChatMessage(BaseModel):
    """A single chat message."""

    role: ChatRole
    content: str


class ChatChoice(BaseModel):
    """A single chat completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


class ChatUsage(BaseModel):
    """Token usage for a chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    """Response from the chat completion endpoint (OpenAI-compatible)."""

    id: str = Field(default="")
    object: str = Field(default="chat.completion")
    created: int = Field(default=0)
    model: str = Field(default="")
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: Optional[ChatUsage] = None
    credits_used: Optional[float] = Field(default=None)

    model_config = {"extra": "allow"}


class ChatChunk(BaseModel):
    """A single SSE chunk from streaming chat."""

    id: str = Field(default="")
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(default=0)
    model: str = Field(default="")
    choices: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def delta_content(self) -> Optional[str]:
        """Extract the content delta from the first choice."""
        if self.choices and "delta" in self.choices[0]:
            return self.choices[0]["delta"].get("content")
        return None

    model_config = {"extra": "allow"}


# --- Translation ---


class TranslationResult(BaseModel):
    """Result from translation endpoint."""

    translated_text: str = Field(description="Translated text")
    source_language: Optional[str] = Field(
        default=None, description="Detected source language"
    )
    target_language: str = Field(description="Target language")
    credits_used: float = Field(default=0)

    model_config = {"extra": "allow"}


# --- Gabriel (Intent Orchestration) ---


class GabrielResponse(BaseModel):
    """Response from the Gabriel intent orchestration endpoint."""

    intent: str = Field(description="Detected intent")
    response: str = Field(description="Generated response")
    actions: list[dict[str, Any]] = Field(
        default_factory=list, description="Suggested actions"
    )
    context: Optional[dict[str, Any]] = Field(default=None)

    model_config = {"extra": "allow"}


# --- Usage ---


class UsageRecord(BaseModel):
    """A single usage record."""

    date: str = Field(description="Date (YYYY-MM-DD)")
    category: str = Field(description="Usage category (image, video, chat, etc.)")
    credits_used: float = Field(default=0)
    request_count: int = Field(default=0)

    model_config = {"extra": "allow"}


class UsageResponse(BaseModel):
    """Response from the usage analytics endpoint."""

    total_credits_used: float = Field(default=0)
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    records: list[UsageRecord] = Field(default_factory=list)
    daily_breakdown: Optional[list[dict[str, Any]]] = None

    model_config = {"extra": "allow"}


# --- Storage ---


class StorageBucket(BaseModel):
    """A storage bucket."""

    id: str = Field(description="Bucket ID")
    name: str = Field(description="Bucket name")
    region: Optional[str] = Field(default=None, description="AWS region")
    size_bytes: Optional[int] = Field(default=None, description="Total size in bytes")
    object_count: Optional[int] = Field(default=None, description="Number of objects")
    created_at: Optional[datetime] = None

    model_config = {"extra": "allow"}


class BucketListResponse(BaseModel):
    """Response from listing storage buckets."""

    buckets: list[StorageBucket] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class BucketProvisionResponse(BaseModel):
    """Response from S3 bucket provisioning."""

    bucket_id: str = Field(description="Provisioned bucket ID")
    name: str = Field(description="Bucket name")
    region: str = Field(description="AWS region")
    endpoint: Optional[str] = Field(default=None, description="S3 endpoint URL")
    credentials: Optional[dict[str, str]] = Field(
        default=None, description="Access credentials"
    )

    model_config = {"extra": "allow"}


class PresignedUrlResponse(BaseModel):
    """Response containing a presigned URL."""

    url: str = Field(description="Presigned URL")
    expires_at: Optional[datetime] = Field(
        default=None, description="URL expiration time"
    )
    method: str = Field(default="PUT", description="HTTP method for the URL")
    headers: Optional[dict[str, str]] = Field(
        default=None, description="Required headers for the request"
    )

    model_config = {"extra": "allow"}


# --- Video timeline API (/v1/video/projects) ---
#
# Plain TypedDicts, not pydantic models: the timeline methods return the API's
# JSON untouched (camelCase keys, exactly as documented), and these describe it
# for type checkers. Keys marked optional are only present in some responses.


class VideoProjectMedia(TypedDict, total=False):
    """One media item attached to a video project."""

    assetId: str
    kind: str
    name: str
    #: Signed URL of the media; refreshed on every read, expires.
    src: str
    storagePath: str
    #: Seconds.
    duration: float
    durationTicks: int
    hasAudio: bool
    width: int
    height: int


class UnplacedMedia(TypedDict, total=False):
    """A media item that was added to the project but not put on the timeline."""

    assetId: str
    name: str
    reason: str


class VideoProject(TypedDict, total=False):
    """A timeline project, as returned by create / get / list."""

    projectId: str
    title: str
    saveRev: int
    updatedAt: str
    ticksPerSecond: int
    #: Compact, readable summary of the timeline; pass it to the model, not the raw document.
    digest: dict[str, Any]
    media: list[VideoProjectMedia]
    #: Create only: media that could not be placed on the timeline.
    unplacedMedia: list[UnplacedMedia]
    versions: list[dict[str, Any]]
    #: Open the same project in the browser editor.
    editorUrl: str
    #: Only with ``include_doc=True``.
    doc: dict[str, Any]


class ApplyOpsResult(TypedDict, total=False):
    """Result of ``apply_video_ops``.

    A rejected operation is skipped and reported in ``results`` (``accepted`` /
    ``rejected``). ``ok`` is False when no operation was accepted (nothing to
    save) and when the final document would violate the timeline invariants;
    only the latter sets ``rolledBack`` True: the whole batch is then
    discarded, the project and ``saveRev`` are unchanged, and ``violations``
    says why. Both are normal answers (HTTP 200), not exceptions.
    """

    ok: bool
    rolledBack: bool
    dryRun: bool
    violations: list[dict[str, Any]]
    #: One entry per submitted operation, in order (``ok`` plus the reason when rejected).
    results: list[dict[str, Any]]
    summary: Any
    #: How many operations were accepted / rejected.
    accepted: int
    rejected: int
    #: Your ``ref`` names mapped to the ids of the clips the batch created.
    refs: dict[str, str]
    saveRev: int
    digestDelta: dict[str, Any]
    #: False when the automatic version snapshot could not be stored (see ``warnings``).
    versionSaved: bool
    warnings: list[str]


class LintFinding(TypedDict, total=False):
    rule: str
    #: ``error``, ``warn`` or ``info``.
    severity: str
    message: str
    #: First clip the finding is about, and all of them when there are several.
    clipId: str
    clipIds: list[str]
    #: Position on the timeline (ticks and seconds); ``end`` / ``endSeconds`` close the range.
    at: int
    atSeconds: float
    end: int
    endSeconds: float
    #: Numbers and texts from the rule, e.g. ``{"seconds": 1.4}``.
    params: dict[str, Any]
    #: Hint: ``relink``, ``trim-to-content`` or ``none``.
    fix: str
    suggestion: str


class LintCounts(TypedDict):
    error: int
    warn: int
    info: int


class LintResult(TypedDict, total=False):
    """Result of ``lint_video_project``."""

    saveRev: int
    findings: list[LintFinding]
    counts: LintCounts
    #: Whether the checker ran.
    available: bool
    warnings: list[str]


class CaptureFrame(TypedDict):
    """One captured frame, located inside a contact sheet."""

    index: int
    #: Requested time, seconds.
    t: float
    #: Time of the frame actually extracted (quantised to 0.25 s).
    actualT: float
    label: str
    #: Index into ``CaptureResult["sheets"]``.
    sheet: int
    #: Cell rectangle on the sheet, pixels.
    x: int
    y: int
    w: int
    h: int


class CaptureSheet(TypedDict):
    url: str
    width: int
    height: int


class CaptureMissing(TypedDict):
    index: int
    t: float


class CaptureResult(TypedDict, total=False):
    """The ``completed`` payload of a capture job (a :class:`VideoJob` with these keys)."""

    frames: list[CaptureFrame]
    sheets: list[CaptureSheet]
    #: Requested points the engine could not extract.
    missing: list[CaptureMissing]


class AutoEditStage(TypedDict, total=False):
    """One stage of an Auto-Edit run (``signals``, ``cuts``, ``brief``, ``broll``, ``graphics``, ``audio``, ``captions``, ``apply``)."""

    stage: str
    #: ``running``, ``done``, ``skipped`` or ``error``.
    status: str
    #: 0-100 within the stage, when known.
    pct: float
    detail: str


class AutoEditUsage(TypedDict, total=False):
    """Token counters of an Auto-Edit run (no model names) and what they were billed."""

    inputTokens: int
    cacheReadTokens: int
    cacheWriteTokens: int
    outputTokens: int
    #: Stays False while the base fee covers the run's AI tokens (the default): tokens are metered, not billed.
    billed: bool
    units: float
    chargedUsd: float
    chargedCredits: float
    #: Part of the usage that could not be collected.
    uncollectedUsd: float


class AutoEditError(TypedDict, total=False):
    code: str
    message: str


class AutoEditJob(TypedDict, total=False):
    """An Auto-Edit job: the 202 of ``auto_edit_video_project`` (``jobId``, ``status``, ``projectId``, ``billing``) and the ``get_video_job`` view."""

    jobId: str
    kind: str
    projectId: str
    #: ``queued``, ``running``, ``completed``, ``failed`` or ``cancelled``.
    status: str
    #: 0-100.
    progress: int
    stages: list[AutoEditStage]
    #: What was done and skipped; carries ``committed``.
    report: dict[str, Any]
    usage: AutoEditUsage
    #: True once the result is in the project; False while it is a draft (``auto_apply=False``).
    committed: bool
    #: Project revision after the commit.
    saveRev: int
    #: Revision the run started from: the default ``expected_save_rev`` of the apply.
    baseSaveRev: int
    #: Project revision now, on a ``save-conflict``.
    currentSaveRev: int
    #: Seconds left before an unapplied draft expires.
    expiresInSeconds: int
    unchanged: bool
    draftId: str
    error: AutoEditError
    reason: str
    #: ``failed`` / ``cancelled``: whether the charge was returned.
    refunded: bool
    #: Start response (202) only.
    billing: dict[str, Any]
    chargedCredits: float


class ApplyAutoEditResult(TypedDict, total=False):
    """Result of ``apply_video_auto_edit``."""

    jobId: str
    projectId: str
    committed: bool
    saveRev: int
    unchanged: bool
    digest: dict[str, Any]
    versionSaved: bool
    warnings: list[Any]


class VideoJob(TypedDict, total=False):
    """A render or capture job: ``POST .../render|capture`` answers 202 with one, ``GET /v1/video/jobs/{id}`` polls it."""

    jobId: str
    #: ``render`` or ``capture`` (``auto_edit``, see :class:`AutoEditJob`). Only on polls.
    kind: str
    projectId: str
    #: ``queued``, ``running``, ``completed``, ``failed`` or ``cancelled``.
    status: str
    progress: float
    queuePosition: int
    #: Render, ``completed``: the finished file.
    outputUrl: str
    outputSize: int
    #: Render: minutes billed for (start response only).
    billedMinutes: float
    #: Start response (202) only: what the call charged, prepaid USD.
    cost_usd: float
    currency: str
    billing: dict[str, Any]
    #: Only when part of the charge was drawn from subscription credits.
    chargedCredits: float
    #: Capture start response: the requested points.
    cuts: bool
    #: ``failed`` / ``cancelled``.
    error: str
    reason: str
    #: True once the charge of a failed job was returned.
    refunded: bool
    warnings: list[str]
    # Capture, ``completed`` (see CaptureResult):
    saveRev: int
    times: list[float]
    width: int
    height: int
    frames: list[CaptureFrame]
    sheets: list[CaptureSheet]
    missing: list[CaptureMissing]
