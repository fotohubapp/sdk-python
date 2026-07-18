"""Pydantic response models for the FOTOhub API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

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
