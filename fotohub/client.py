"""FOTOhub API client — synchronous and asynchronous.

Covers all 29+ public API endpoints with full type annotations.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
import warnings
from typing import Any, Generator, Optional, Union

import httpx

from .exceptions import (
    AuthError,
    FotoHubError,
    InsufficientFundsError,
    RateLimitError,
    SaveConflictError,
    ServerError,
    TimeoutError,
    ValidationError,
    VideoJobFailedError,
    VideoJobTimeoutError,
)
from .streaming import AsyncChatStream, ChatStream

DEFAULT_BASE_URL = "https://apis.fotohub.app"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_IMAGE_MODEL = "seedream-5-0-260128"
DEFAULT_VIDEO_MODEL = "veo-2"
#: Seedance is the one video family that runs asynchronously (202 + job_id), so
#: it has its own method rather than being reachable through generate_video().
DEFAULT_SEEDANCE_MODEL = "seedance-2-5"
DEFAULT_CHAT_MODEL = "gemini-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4.6"
# Backwards-compat alias — prefer DEFAULT_CLAUDE_MODEL.
DEFAULT_BEDROCK_MODEL = DEFAULT_CLAUDE_MODEL
DEFAULT_MUSIC_MODEL = "minimax"
DEFAULT_SPEECH_MODEL = "google"
SDK_VERSION = "1.11.0"

#: Header the API reads to de-duplicate a retried charged request. The SDK sends
#: one automatically on every guarded POST — see `_idempotency_key_for`.
IDEMPOTENCY_HEADER = "X-Idempotency-Key"

#: Path prefixes the API protects with `X-Idempotency-Key`. Mirrors
#: `_IDEMPOTENT_PREFIXES` in api-server's `main.py`. Sending the header outside
#: these prefixes is harmless (the server ignores it), but generating a key only
#: where it does something keeps request logs honest.
_IDEMPOTENT_PREFIXES: tuple[str, ...] = (
    "/v1/ai/",
    "/v1/images/",
    "/v1/video/",
    "/v1/shorts/",
    "/v1/story/",
    "/v1/3d/",
    "/v1/generate/",
    "/v1/voice/",
)

#: Streaming endpoints, which the server deliberately excludes: a buffered
#: stream cannot be replayed, and holding one back in full before its first byte
#: would defeat streaming. Mirrors `_IDEMPOTENCY_EXCLUDE_PREFIXES` server-side.
_IDEMPOTENCY_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "/v1/ai/chat",
    "/v1/ai/agent/stream",
    "/v1/ai/gabriel",
    "/v1/ai/tts/",
    "/v1/story/generate",
)


#: Timeline POSTs that are free and carry no idempotency key. Digest and lint are
#: read-only. `ops` is not: it is protected by `expected_save_rev` instead (see
#: `_retry_ambiguous`), because a keyed replay of a lost 2xx is not needed for a
#: call whose duplicate the server can refuse with `save-conflict`.
_UNGUARDED_TIMELINE_POST = re.compile(r"^/v1/video/projects/[^/]+/(ops|digest|lint)$")


def _idempotency_key_for(method: str, path: str, stream: bool) -> Optional[str]:
    """A fresh key for one logical call, or None if the call is not guarded.

    Generated per `_request` invocation, NOT per HTTP attempt: that is the whole
    point. This client retries `(429, 500, 502, 503, 504)` and connect timeouts
    up to `max_retries` times by default, and a 504 arriving after a render has
    already started used to bill the same job again on each retry. Reusing one
    key across the attempts of a single call turns those retries into replays.

    A key is deliberately not carried across separate calls to `_request`: two
    calls with the same arguments are two requests the caller asked for, and
    silently collapsing them would make the SDK lose a generation somebody paid
    for.
    """
    if method.upper() not in ("POST", "PUT", "PATCH") or stream:
        return None
    if not path.startswith(_IDEMPOTENT_PREFIXES):
        return None
    if path.startswith(_IDEMPOTENCY_EXCLUDE_PREFIXES):
        return None
    if _UNGUARDED_TIMELINE_POST.match(path):
        return None
    return str(uuid.uuid4())


def _is_idempotency_in_progress(response: httpx.Response) -> bool:
    """Whether a 409 means "a request with this key is still running".

    Decided by the envelope's error code. A body with no readable code (an older
    server, a proxy) counts only when it carries `Retry-After`, which the API
    sends with this 409 and with no other.
    """
    try:
        body = response.json()
    except Exception:
        body = None
    envelope = body.get("error") if isinstance(body, dict) else None
    code = envelope.get("code") if isinstance(envelope, dict) else None
    if isinstance(code, str):
        return code == "idempotency-in-progress"
    return "retry-after" in response.headers


def _as_float(value: Any) -> Optional[float]:
    """A money figure from an error body, or ``None`` if there isn't one.

    Explicitly typed rather than truthy: ``0`` is the commonest balance behind a
    402 and is the single number worth reporting, so a falsy check would drop
    exactly the case the caller most needs. Strings are accepted because a
    gateway may serialize numerics; a bool is not a number here.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_error(body: Any, fallback: str) -> tuple[str, dict[str, Any]]:
    """Pull a human message and any structured fields out of an error body.

    The API is FastAPI, so every error is ``{"detail": ...}`` — either a plain
    string or, on a few endpoints, a dict. Reading only ``error``/``message``
    (as this SDK used to) meant the raw JSON was stringified into the message
    and ``credits_required`` / ``credits_available`` / ``errors`` always came
    back ``None``. The ``error``/``message`` keys are still honoured so a
    gateway or Cloudflare error page keeps working.

    Returns the message plus the dict the structured fields should be read
    from, which is the ``detail`` dict when there is one, else the whole body.
    """
    if not isinstance(body, dict):
        return (str(body) if body else fallback), {}

    detail = body.get("detail")
    if isinstance(detail, dict):
        message = (
            detail.get("message")
            or detail.get("error")
            or detail.get("detail")
            or fallback
        )
        return str(message), detail
    if isinstance(detail, list):
        # FastAPI request-validation errors: [{"loc": [...], "msg": ..., ...}]
        msgs = [
            str(d.get("msg")) for d in detail if isinstance(d, dict) and d.get("msg")
        ]
        return ("; ".join(msgs) if msgs else fallback), body
    if detail:
        return str(detail), body

    message = body.get("error", body.get("message", fallback))
    if isinstance(message, dict):
        # {"error": {"message": ...}} — an upstream provider envelope.
        # The public timeline envelope is {"error": {code, message, details?}}: the
        # structured fields live under `details`, so merge them over the envelope.
        fields = dict(message)
        if isinstance(message.get("details"), dict):
            fields = {**message["details"], **fields}
        return str(message.get("message") or message.get("code") or fallback), fields
    return str(message), body


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    """Omit unset optionals: the API validates strictly and rejects unknown/null shapes."""
    return {k: v for k, v in data.items() if v is not None}


def _video_media_payload(media: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """``[{"url"|"storage_path", "kind"?, "name"?}]`` -> the API's camelCase shape."""
    if media is None:
        return None
    return [{_camel(k): v for k, v in item.items() if v is not None} for item in media]


def _video_project_payload(**kw: Any) -> dict[str, Any]:
    template = kw.get("template")
    return _drop_none({
        "title": kw.get("title"),
        "aspect": kw.get("aspect"),
        "fps": kw.get("fps"),
        "media": _video_media_payload(kw.get("media")),
        "template": {"id": template} if isinstance(template, str) else template,
        "placeMedia": kw.get("place_media"),
    })


def _video_render_payload(**kw: Any) -> dict[str, Any]:
    span = kw.get("time_range")
    return _drop_none({
        "format": kw.get("format"),
        "codec": kw.get("codec"),
        "quality": kw.get("quality"),
        "resolution": kw.get("resolution"),
        "fps": kw.get("fps"),
        "bitrate": kw.get("bitrate"),
        "range": {"in": span[0], "out": span[1]} if span else None,
        "contentCredentials": kw.get("content_credentials"),
        "contentAiDeclared": kw.get("content_ai_declared"),
    })


def _video_capture_payload(**kw: Any) -> dict[str, Any]:
    sheet = kw.get("sheet")
    return _drop_none({
        "times": kw.get("times"),
        "count": kw.get("count"),
        # `cuts: false` means "not chosen": the API wants exactly one selector.
        "cuts": True if kw.get("cuts") else None,
        "width": kw.get("width"),
        "sheet": {_camel(k): v for k, v in sheet.items()} if sheet else None,
    })


def _video_auto_edit_payload(**kw: Any) -> dict[str, Any]:
    return _drop_none({
        "style": kw.get("style"),
        "toggles": kw.get("toggles"),
        "language": kw.get("language"),
        "aspect": kw.get("aspect"),
        "aiBudgetUsd": kw.get("ai_budget_usd"),
        "autoApply": kw.get("auto_apply"),
        "mode": kw.get("mode"),
    })


def _video_source_payload(**kw: Any) -> dict[str, Any]:
    """Analysis source: either a public ``url`` or a project's ``project_id`` + ``media_id``."""
    return _drop_none({
        "url": kw.get("url"),
        "projectId": kw.get("project_id"),
        "mediaId": kw.get("media_id"),
    })


#: Statuses after which a render/capture job will not change again.
_VIDEO_JOB_FAILED = ("failed", "cancelled")


def _video_job_failure(job: dict[str, Any]) -> VideoJobFailedError:
    job_id = job.get("jobId")
    reason = job.get("reason")
    detail = job.get("error") or reason or job.get("status")
    return VideoJobFailedError(
        message=f"Video job {job_id} {job.get('status')}: {detail}",
        job_id=job_id,
        reason=reason,
        refunded=job.get("refunded"),
        code=job.get("code") if isinstance(job.get("code"), str) else None,
        response_body=job,
    )


def _seedance_payload(**kwargs: Any) -> dict[str, Any]:
    """Build a /v1/ai/generate/video body for the Seedance family.

    Shared by the sync and async clients so the two cannot drift. Optional
    fields are omitted rather than sent as null: the API validates the shape of
    what it receives, and an explicit ``"reference_videos": null`` reads as an
    empty reference list, which changes the inferred task type.
    """
    payload: dict[str, Any] = {
        "prompt": kwargs["prompt"],
        "model": kwargs["model"],
        "duration": kwargs["duration"],
        "resolution": kwargs["resolution"],
        "aspect_ratio": kwargs["aspect_ratio"],
        "generate_audio": kwargs["generate_audio"],
    }
    for key in (
        "image_url", "last_frame_url", "reference_images", "reference_videos",
        "reference_audios", "asset_ids", "output_format", "negative_prompt",
        "seed", "callback_url",
    ):
        value = kwargs.get(key)
        if value is not None:
            payload[key] = value
    # Booleans are only sent when true — `smart_ratio: false` is the default and
    # sending it would still be honoured, but it makes request logs read as if
    # the caller had opted out of something.
    if kwargs.get("smart_ratio"):
        payload["smart_ratio"] = True
    if kwargs.get("smart_duration"):
        payload["smart_duration"] = True
    return payload


class _BaseClient:
    """Shared configuration for sync and async clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.environ.get("FOTOHUB_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("FOTOHUB_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "User-Agent": f"fotohub-python/{SDK_VERSION}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key
        return headers

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Raise appropriate exception based on HTTP status code.

        Whatever is raised also carries ``code`` / ``details`` when the body is
        the ``{"error": {"code", "message", "details"}}`` envelope.
        """
        try:
            self._raise_for_status(response)
        except FotoHubError as exc:
            body = exc.response_body
            envelope = body.get("error") if isinstance(body, dict) else None
            if isinstance(envelope, dict):
                if isinstance(envelope.get("code"), str):
                    exc.code = envelope["code"]
                exc.details = envelope.get("details")
            raise

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}

        message, fields = _extract_error(body, response.text)

        if status == 409 and fields.get("code") == "save-conflict":
            rev = fields.get("currentSaveRev")
            raise SaveConflictError(
                message=message,
                status_code=status,
                response_body=body,
                current_save_rev=int(rev) if isinstance(rev, (int, float)) and not isinstance(rev, bool) else None,
            )
        if status == 403 and fields.get("code") == "payment-required":
            # Empty prepaid wallet / no API entitlement: a funds problem, not a bad key.
            raise InsufficientFundsError(
                message=message,
                status_code=status,
                response_body=body,
                required_usd=_as_float(fields.get("required_usd")),
                balance_usd=_as_float(fields.get("balance_usd")),
                shortfall_usd=_as_float(fields.get("shortfall_usd")),
                topup_url=fields.get("topup_url") or None,
                operation=fields.get("operation") or None,
            )
        if status == 401 or status == 403:
            raise AuthError(message=message, status_code=status, response_body=body)
        elif status == 402:
            # `fields` is the server's flat funds payload: required_usd,
            # balance_usd, shortfall_usd, topup_url, charged, operation. Reading
            # only credits_required/credits_available -- which the prepaid API
            # never sends -- left every 402 with no price, no balance and no
            # top-up link, while all three sat in the response body.
            raise InsufficientFundsError(
                message=message,
                status_code=status,
                response_body=body,
                required_usd=_as_float(fields.get("required_usd")),
                balance_usd=_as_float(fields.get("balance_usd")),
                shortfall_usd=_as_float(fields.get("shortfall_usd")),
                topup_url=fields.get("topup_url") or None,
                operation=fields.get("operation") or None,
                # Only a pre-cutover server populates these.
                credits_required=_as_float(fields.get("credits_required")),
                credits_available=_as_float(fields.get("credits_available")),
            )
        elif status == 429:
            retry_after = (
                response.headers.get("retry-after")
                or fields.get("retryAfterSeconds")
                or fields.get("retry_after")
            )
            raise RateLimitError(
                message=message,
                status_code=status,
                response_body=body,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif status == 400 or status == 422:
            raise ValidationError(
                message=message,
                status_code=status,
                response_body=body,
                errors=fields.get("errors") or (body.get("detail") if isinstance(body.get("detail"), list) else None),
            )
        elif status >= 500:
            raise ServerError(message=message, status_code=status, response_body=body)
        else:
            raise FotoHubError(message=message, status_code=status, response_body=body)

    def _should_retry(
        self,
        response: httpx.Response,
        *,
        idempotent: bool = False,
        retry_ambiguous: bool = True,
    ) -> bool:
        """Determine if a request should be retried based on the response.

        `idempotent` adds one 409 to the retryable set: `idempotency-in-progress`.
        On a guarded endpoint the API answers it, with `Retry-After`, when a
        request carrying this same key is still in flight -- which, on a retry,
        is our own earlier attempt. The right move is to wait and collect its
        result. Every other 409 (`save-conflict`, `project-limit`,
        `draft-limit`, ...) is a real answer that repeating cannot change, so it
        is raised at once. Without an idempotency key a 409 is never retried.

        `retry_ambiguous=False` is for writes that are neither keyed nor
        naturally repeatable: after a 5xx the server may already have applied
        the change, so only 429 (rejected before it ran) is safe to repeat.
        """
        status_code = response.status_code
        if status_code == 409:
            return idempotent and _is_idempotency_in_progress(response)
        if status_code == 429:
            return True
        return retry_ambiguous and status_code in (500, 502, 503, 504)

    def _backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay in seconds."""
        return min(2**attempt * 0.5, 30.0)


# ---------------------------------------------------------------------------
# Synchronous Client
# ---------------------------------------------------------------------------


class FotoHub(_BaseClient):
    """Synchronous FOTOhub API client.

    Usage::

        from fotohub import FotoHub

        client = FotoHub(api_key="your-api-key")
        result = client.generate_image(prompt="A sunset over mountains")
        print(result["images"][0]["url"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        super().__init__(api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        stream: bool = False,
        idempotency_key: Optional[str] = None,
        retry_ambiguous: bool = True,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic.

        `retry_ambiguous=False` disables retries after a 5xx or a timeout, for
        unkeyed writes whose first attempt may already have been applied.

        Every retry of a charged POST carries the same `X-Idempotency-Key`, so a
        timeout or a 5xx that arrives after the work has already started is
        replayed rather than charged again. See `_idempotency_key_for`.
        """
        last_exception: Optional[Exception] = None
        idem_key = idempotency_key or _idempotency_key_for(method, path, stream)
        extra_headers = {IDEMPOTENCY_HEADER: idem_key} if idem_key else None

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = self._client.stream(
                        method, path, json=json_data, params=params
                    ).__enter__()
                else:
                    response = self._client.request(
                        method, path, json=json_data, params=params,
                        headers=extra_headers,
                    )

                if response.status_code < 400:
                    return response

                if self._should_retry(
                    response, idempotent=idem_key is not None,
                    retry_ambiguous=retry_ambiguous,
                ) and attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = max(delay, float(retry_after))
                    time.sleep(delay)
                    continue

                self._handle_error_response(response)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                # A read/write timeout may hide an applied write; only a failure
                # to connect at all is known to have changed nothing.
                if not retry_ambiguous and not isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    raise TimeoutError(message=f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                raise TimeoutError(message=f"Request failed: {e}")

        if last_exception:
            raise TimeoutError(message=f"Request failed after {self.max_retries} retries")
        raise FotoHubError("Unexpected retry exhaustion")

    # =========================================================================
    # AI Generation
    # =========================================================================

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        width: int = 1024,
        height: int = 1024,
        aspect_ratio: str = "1:1",
        num_images: int = 1,
        image_size: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Model to use (default: seedream-5-0-260128).
            width: Image width in pixels.
            height: Image height in pixels.
            aspect_ratio: Aspect ratio string (e.g. "1:1", "16:9", "9:16").
            num_images: Whole number of images, 1-8. Charged per image the
                provider actually delivers: every provider caps the count at its
                own maximum, and the difference is refunded automatically.
            image_size: Resolution tier -- "1K", "1.5K", "2K", "3K" or "4K".
                This is priced: 4K costs more than 1K on any model offering it.
                Leave it None to bill the model's 1K base rate; width/height are
                mapped onto a tier when it is omitted.
            negative_prompt: Things to avoid in the image.
            style: Style preset (e.g. "photographic", "cinematic", "anime").
            seed: Random seed for reproducibility.

        Returns:
            Dict with ``images`` (list of URLs), ``model``, ``cost_usd``,
            ``currency`` and a ``billing`` block. There is no ``credits_used``:
            the API is prepaid in USD and reads no credit balance.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged.
            ValidationError: If parameters are invalid.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
        }
        if image_size is not None:
            payload["image_size"] = image_size
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if style is not None:
            payload["style"] = style
        if seed is not None:
            payload["seed"] = seed

        response = self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return response.json()

    def generate_ida_q(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        num_images: int = 1,
        seed: Optional[int] = None,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Generate an image with IDA Q 1.0, FOTOhub's proprietary image model.

        Unlike :meth:`generate_image`, IDA Q 1.0 runs on a self-hosted, single-GPU
        queue and is asynchronous — generation takes 30 seconds to ~3.5 minutes
        depending on ``image_size``. This method submits the job and polls until
        it completes, returning the finished result. Any prompt (including
        non-English text) is automatically translated and restructured for best
        results — see the `IDA Q 1.0 docs <https://docs.fotohub.app/api/ida-q>`_.

        Args:
            prompt: Text description of the desired image. Any language.
            aspect_ratio: One of "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
                "21:9". This is the one thing the render never trades away: where
                the GPU's 2048-per-edge ceiling applies, the resolution gives way
                and the ratio is kept.
            image_size: Resolution tier — "1K" (~30s), "1.5K" (~90s), or "2K"
                (~3.5min). "3K" and "4K" are accepted and capped to "2K"; the model
                renders at most 2048x2048.
            num_images: Number of images to generate (1-2). Higher values are
                clamped to 2 before billing.
            seed: Random seed for reproducibility.
            poll_interval: Seconds to wait between status checks. The poll endpoint
                is rate-limited per ACCOUNT by tier (30/min on the lowest), and the
                default 3s costs 20 of those a minute, so raise this for a 2K
                render or two concurrent jobs will throttle each other.
            timeout: Maximum seconds to wait for completion before raising.

        Returns:
            Dict with ``images`` (list of URLs), ``model``, ``job_id``,
            ``cost_usd`` and ``billing``. IDA Q is self-hosted and renders at
            $0.00, so ``cost_usd`` is 0 -- and because a zero-price operation is
            settled without a balance check, this is the one model an empty wallet
            can still run.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged. Cannot happen at the current $0.00 price.
            TimeoutError: If generation doesn't complete within ``timeout``.
            FotoHubError: If generation fails server-side.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": "ida-q-image",
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_images": num_images,
        }
        if seed is not None:
            payload["seed"] = seed

        submit_response = self._request("POST", "/v1/ai/generate/image", json_data=payload)
        job = submit_response.json()
        job_id = job["job_id"]

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status_response = self._request("GET", f"/v1/ai/generate/image/ida-q/{job_id}")
            status = status_response.json()
            if status["status"] == "completed":
                return {
                    "model": "ida-q-image",
                    "job_id": job_id,
                    # The wallet is charged at submit and the poll route reports
                    # job state only, so the cost has to be carried across or it
                    # is absent from the result the caller actually receives.
                    # This used to carry `credits_used`, a field the prepaid API
                    # stopped returning -- so it was always None.
                    "cost_usd": job.get("cost_usd", (job.get("billing") or {}).get("cost_usd")),
                    "currency": "USD",
                    "billing": job.get("billing"),
                    "images": status.get("images", []),
                    "metadata": status.get("metadata"),
                }
            if status["status"] == "failed":
                raise FotoHubError(status.get("error", "IDA Q 1.0 generation failed"))
            time.sleep(poll_interval)

        raise TimeoutError(message=f"IDA Q 1.0 job {job_id} did not complete within {timeout}s")

    def edit_image(
        self,
        image_url: str,
        prompt: str,
        *,
        mode: str = "inpaint",
        mask_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Edit an existing image using AI.

        Args:
            image_url: URL of the source image.
            prompt: Instruction for the edit.
            mode: Edit mode — "inpaint", "outpaint", "remove_bg", "upscale", "style_transfer".
            mask_url: URL of the mask image (required for inpaint/erase).
            model: Model override.

        Returns:
            Dict with edited image URL and metadata.
        """
        payload: dict[str, Any] = {
            "image_url": image_url,
            "prompt": prompt,
            "mode": mode,
        }
        if mask_url is not None:
            payload["mask_url"] = mask_url
        if model is not None:
            payload["model"] = model

        response = self._request("POST", "/v1/ai/edit/image", json_data=payload)
        return response.json()

    def generate_video(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_VIDEO_MODEL,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        image_url: Optional[str] = None,
        resolution: str = "1080p",
        poll_interval: float = 5.0,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        """Generate a video, waiting for the finished file.

        Most models render inside the request and come back finished. Some
        (Alibaba Wan, xAI Grok) answer immediately with ``status: "processing"``
        and a ``job_id`` instead — so this polls until the job reaches a terminal
        state and returns the completed result either way. The returned dict
        always has ``video_url`` set on success.

        Note that ``duration`` is snapped to a length the provider actually
        renders (Veo accepts only 4/6/8s, Kling 5/10s), and the charge follows
        the snapped value — read ``duration`` on the result, not your request.

        Args:
            prompt: Text description of the desired video.
            model: Video model (default: veo-2).
            duration: Desired duration in seconds.
            aspect_ratio: Aspect ratio (e.g. "16:9", "9:16", "1:1").
            image_url: Reference image for image-to-video generation.
            resolution: Output resolution ("720p", "1080p", "4k").
            poll_interval: Seconds between polls, for the models that queue.
            timeout: How long to keep polling before giving up. The job itself
                is unaffected and may still finish.

        Returns:
            Dict with model, video_url, job_id, status, duration, ``cost_usd``
            and ``currency``. There is no ``credits_used``: the API is prepaid in
            USD.

        Raises:
            FotoHubError: If the generation failed. A failed video is refunded to
                the wallet automatically, so a raise here does not mean you paid
                for nothing delivered.
            TimeoutError: If the job was still processing when ``timeout``
                elapsed.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        if image_url is not None:
            payload["image_url"] = image_url

        result = self._request(
            "POST", "/v1/ai/generate/video", json_data=payload
        ).json()

        job_id = result.get("job_id")
        # Only the queueing models need polling. A finished response already
        # carries the URL, and one without a job_id cannot be polled at all.
        if result.get("video_url") or result.get("status") != "processing" or not job_id:
            return result

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            status = self._request(
                "GET", f"/v1/ai/generate/video/{job_id}"
            ).json()
            state = status.get("status", "")
            if state == "completed":
                # The poll route reports the charge from the job row's
                # `estimated_cost`, which is null on a row written before that
                # column held USD. Fall back to the submit response, which always
                # carries it. Was `credits_used` on both sides -- a field the
                # prepaid API stopped returning, so this copied None onto None.
                if status.get("cost_usd") is None:
                    status["cost_usd"] = result.get("cost_usd")
                    status.setdefault("currency", "USD")
                if status.get("billing") is None and result.get("billing"):
                    status["billing"] = result["billing"]
                return status
            if state in ("failed", "cancelled"):
                raise FotoHubError(
                    message=status.get("error")
                    or status.get("error_message")
                    or f"Video job {job_id} {state}",
                    status_code=500,
                    response_body=status,
                )

        raise TimeoutError(
            message=f"Video job {job_id} did not complete within {timeout}s. "
                    f"It may still finish — poll GET /v1/ai/generate/video/{job_id}."
        )

    def generate_seedance(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_SEEDANCE_MODEL,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        generate_audio: bool = False,
        image_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        reference_images: Optional[list[Any]] = None,
        reference_videos: Optional[list[Any]] = None,
        reference_audios: Optional[list[Any]] = None,
        asset_ids: Optional[list[str]] = None,
        output_format: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        callback_url: Optional[str] = None,
        smart_ratio: bool = False,
        smart_duration: bool = False,
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Generate a video with a Seedance model, waiting for the result.

        Unlike :meth:`generate_video`, the Seedance family is asynchronous: the
        API answers 202 with a ``job_id``, and the render runs in a queue. This
        method submits, polls, and returns the finished job — so the returned
        dict already has ``video_url``.

        ``seedance-2-5`` is the only model on the platform that produces a
        30-second clip in one request, and the only one that accepts a source
        video (``reference_videos``) for editing or extension. Native audio is
        included in its price: $0.2335/s at 720p, $0.103062/s at 480p, the same
        with ``generate_audio`` on or off. It does **not** do 1080p or 4K — those
        return a 400. For higher resolution use ``seedance-2-0-pro`` (up to 4K,
        but capped at 15s).

        Args:
            prompt: Text description of the desired video.
            model: Seedance model id (default: seedance-2-5). Others:
                ``seedance-2-0-pro`` / ``-fast`` / ``-mini``,
                ``seedance-1-5-pro-251215``, ``seedance-1-0-pro-250528``,
                ``seedance-1-0-pro-fast-251015``.
            duration: Seconds. 2.5 takes 4-30, 2.0 takes 4-15, 1.x takes 5-10.
                Pass ``-1`` to match a source clip's length (billed at the
                model's ceiling, since the real length is unknown until the clip
                is decoded).
            resolution: ``"480p"`` or ``"720p"`` on 2.5; ``seedance-2-0-pro``
                also takes ``"1080p"`` and ``"4K"``. Anything the model does not
                accept is a 400, never a silent downgrade — the price scales
                with resolution.
            aspect_ratio: ``16:9``, ``9:16``, ``1:1``, ``4:3``, ``3:4``,
                ``21:9``, or ``adaptive``.
            generate_audio: Native soundtrack. Free on 2.5.
            image_url: First frame (image-to-video).
            last_frame_url: Final frame.
            reference_images: Up to 30 on 2.5 (9 on 2.0). URLs or
                ``{"mimeType": ..., "base64": ...}`` dicts.
            reference_videos: Up to 10 on 2.5 (3 on 2.0). Attaching one switches
                the request to reference / editing / extension mode and raises
                the rate to $0.283421/s at 720p ($0.125607 at 480p), because the
                source frames bill as input tokens. Only 2.5 has that rate; on
                every other model a reference video costs the plain rate. The
                response's ``billing.breakdown.video_input`` says which rate you
                were charged.
            reference_audios: Up to 10 on 2.5 (3 on 2.0). Requires at least one
                image or video reference.
            asset_ids: Pre-registered ``asset://`` portrait ids from
                ``POST /v1/ai/assets/register``, for face consistency.
            output_format: ``"mp4"`` (default) or ``"mov"``. 2.5 only.
            negative_prompt: Recorded on the job.
            seed: Recorded on the job.
            callback_url: HTTPS URL POSTed once the job reaches a terminal state.
            smart_ratio: Let the model pick the aspect ratio.
            smart_duration: Let the model pick the duration.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising. A 30s 720p render
                takes ~4 minutes; the default allows for a queue.

        Returns:
            The finished job dict — ``video_url``, ``thumbnail_url``, ``status``,
            ``cost_usd``, ``currency``, ``duration``, ``resolution``,
            ``task_type``, ``billing``. There is no ``credits_used``.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged.
            TimeoutError: If the job does not finish within ``timeout``.
            FotoHubError: If the render fails. A failed render is refunded to the
                wallet server-side.
        """
        payload = _seedance_payload(
            prompt=prompt, model=model, duration=duration, resolution=resolution,
            aspect_ratio=aspect_ratio, generate_audio=generate_audio,
            image_url=image_url, last_frame_url=last_frame_url,
            reference_images=reference_images, reference_videos=reference_videos,
            reference_audios=reference_audios, asset_ids=asset_ids,
            output_format=output_format, negative_prompt=negative_prompt,
            seed=seed, callback_url=callback_url, smart_ratio=smart_ratio,
            smart_duration=smart_duration,
        )

        submit = self._request(
            "POST", "/v1/ai/generate/video", json_data=payload
        ).json()
        job_id = submit.get("job_id")
        if not job_id:
            # A non-Seedance model was passed: that path is synchronous and has
            # already returned the finished video, so hand it back as-is rather
            # than polling a job that does not exist.
            return submit

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._request(
                "GET", f"/v1/ai/generate/video/{job_id}"
            ).json()
            state = status.get("status", "")
            if state == "completed":
                return status
            if state in ("failed", "cancelled"):
                raise FotoHubError(
                    message=status.get("error")
                    or status.get("error_message")
                    or f"Seedance job {job_id} {state}",
                    status_code=500,
                    response_body=status,
                )
            time.sleep(poll_interval)

        raise TimeoutError(
            message=f"Seedance job {job_id} did not complete within {timeout}s. "
                    f"It may still finish — poll GET /v1/ai/generate/video/{job_id}."
        )

    def register_video_asset(
        self, image_url: str, *, retention_hours: Optional[int] = None
    ) -> dict[str, Any]:
        """Register a hosted portrait as a reusable Seedance asset.

        Free — no credits are charged. Pass the returned ``uri`` (or bare id) in
        ``asset_ids`` on :meth:`generate_seedance` so the same face appears
        across generations.

        A registered face is biometric data. Pass ``retention_hours`` to have it
        self-delete, at the provider and in our records, once that period
        elapses — use it to honour a data-minimisation policy instead of relying
        on remembering to call :meth:`delete_video_asset` yourself.

        Args:
            image_url: HTTPS URL on a FOTOhub host. Upload the file first (e.g.
                via ``POST /v1/photos/upload``); third-party URLs are refused.
            retention_hours: Optional, 1-8760 (1 year). Omit to keep the face
                until you delete it — this is opt-in so an existing integration
                does not silently start losing faces it depends on.

        Returns:
            Dict with ``asset_id``, ``uri``, ``status``, ``retention_hours``,
            ``expires_at``.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if retention_hours is not None:
            payload["retention_hours"] = retention_hours
        response = self._request(
            "POST", "/v1/ai/assets/register", json_data=payload
        )
        return response.json()

    def list_video_assets(
        self, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List the virtual portrait assets registered by this account.

        Each row includes ``retention_hours``, ``expires_at`` and ``purged_at``
        so you can tell an active registration from an erased one without a
        second call.

        Returns:
            Dict with ``assets`` (list) and ``count``.
        """
        response = self._request(
            "GET", "/v1/ai/assets",
            params={"limit": limit, "offset": offset},
        )
        return response.json()

    def get_video_asset(self, asset_id: str) -> dict[str, Any]:
        """Get the current provider status of one registered face.

        Args:
            asset_id: The bare id (not the ``asset://`` uri).

        Raises:
            FotoHubError: 404 if the asset does not belong to this account.
        """
        response = self._request("GET", f"/v1/ai/assets/{asset_id}")
        return response.json()

    def delete_video_asset(self, asset_id: str) -> dict[str, Any]:
        """Delete a registered face, at the provider and here.

        Use this to honour an erasure request immediately, rather than waiting
        on ``retention_hours``. The remote delete happens first; the local
        record is only marked erased once the provider confirms it — so a
        successful return means the face is actually gone, not just that
        deletion was requested.

        Idempotent: calling this on an already-erased asset returns
        ``{"deleted": True, "already_deleted": True}`` rather than raising.

        Args:
            asset_id: The bare id (not the ``asset://`` uri).

        Returns:
            Dict with ``asset_id``, ``deleted``, and either ``reason`` or
            ``already_deleted``.

        Raises:
            FotoHubError: 404 if the asset does not belong to this account;
                502 if the provider delete failed (nothing was recorded as
                deleted — safe to retry).
        """
        response = self._request("DELETE", f"/v1/ai/assets/{asset_id}")
        return response.json()

    def generate_music(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MUSIC_MODEL,
        duration: int = 30,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tempo: int = 120,
        instrumental: bool = True,
    ) -> dict[str, Any]:
        """Generate music from a text description.

        Args:
            prompt: Description of the desired music.
            model: Music generation model (default: minimax).
            duration: Duration in seconds (5-300).
            genre: Genre hint (e.g. "electronic", "classical", "jazz").
            mood: Mood hint (e.g. "happy", "melancholic", "energetic").
            tempo: BPM (40-240, default: 120).
            instrumental: Whether to generate instrumental-only (default: True).

        Returns:
            Dict with ``audio_url``, ``duration``, ``cost_usd``, ``currency`` and
            a ``billing`` block. There is no ``credits_used``.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "tempo": tempo,
            "instrumental": instrumental,
        }
        if genre is not None:
            payload["genre"] = genre
        if mood is not None:
            payload["mood"] = mood

        response = self._request("POST", "/v1/ai/generate/music", json_data=payload)
        return response.json()

    def generate_sfx(
        self,
        prompt: str,
        *,
        duration: int = 5,
    ) -> dict[str, Any]:
        """Generate a short sound effect.

        Args:
            prompt: Description of the sound effect.
            duration: Duration in seconds (1-30, default: 5).

        Returns:
            Dict with audio URL and metadata.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
        }

        response = self._request("POST", "/v1/ai/generate/sfx", json_data=payload)
        return response.json()

    def generate_speech(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        model: str = DEFAULT_SPEECH_MODEL,
        language: str = "pl",
        speed: float = 1.0,
        pitch: int = 0,
    ) -> dict[str, Any]:
        """Generate speech audio from text (TTS).

        Args:
            text: Text to convert to speech.
            voice_id: Voice identifier (provider-specific).
            model: TTS model/provider (default: "google").
            language: Language code (default: "pl").
            speed: Speech speed multiplier (0.5-2.0, default: 1.0).
            pitch: Pitch adjustment in semitones (-10 to 10, default: 0).

        Returns:
            Dict with ``audio_url``, ``characters_processed``, ``cost_usd``,
            ``currency`` and a ``billing`` block. There is no ``credits_used``.
        """
        payload: dict[str, Any] = {
            "text": text,
            "model": model,
            "language": language,
            "speed": speed,
            "pitch": pitch,
        }
        if voice_id is not None:
            payload["voice_id"] = voice_id

        response = self._request("POST", "/v1/ai/generate/speech", json_data=payload)
        return response.json()

    def transcribe(
        self,
        audio_url: str,
        *,
        language: str = "auto",
    ) -> dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio_url: URL of the audio file.
            language: Language code or "auto" for auto-detection.

        Returns:
            Dict with transcribed text, detected language, segments.
        """
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "language": language,
        }

        response = self._request("POST", "/v1/ai/transcribe", json_data=payload)
        return response.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Union[dict[str, Any], ChatStream]:
        """Send a chat completion request (OpenAI-compatible).

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: LLM model (default: gemini-flash). Only ``gemini-flash``,
                ``gemini-pro``, ``gpt-4o`` and ``claude-sonnet`` are accepted;
                anything else is rejected with 400 rather than silently
                substituted.
            temperature: Sampling temperature (0-2, default: 0.7).
            max_tokens: Maximum tokens in the response.
            stream: Not supported -- see Raises.

        Returns:
            Dict with choices, usage, ``cost_usd``, ``currency`` and ``billing``.
            Billed on real token counts at the provider's own per-direction rate,
            so ``billing["cost_usd"]`` scales with the length of the answer --
            fractions of a cent for a short reply. ``billing["legs"]`` splits it
            into input and output. ``billing["basis"]`` is ``"tokens"`` when the
            charge came from the model's own usage figures, or
            ``"flat_fallback"`` when the provider omitted them and one 1K output
            block was charged instead. There is no ``credits_used``.

        Raises:
            ValueError: If ``stream=True``. /v1/ai/chat/completions accepts the
                flag for OpenAI compatibility and then ignores it, returning one
                complete JSON body. ChatStream finds no SSE frames in that body,
                so it yields zero chunks and raises nothing -- an empty result
                for a request that was still billed. Failing before the call
                keeps it free.
        """
        if stream:
            raise ValueError(
                "chat(stream=True) is not supported: /v1/ai/chat/completions never "
                "streams, so the iterator would yield nothing while the request is "
                "still billed. Use POST /v1/ai/agent/stream for token-by-token "
                "output -- see https://docs.fotohub.app/guides/streaming"
            )

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = self._request("POST", "/v1/ai/chat/completions", json_data=payload)
        return response.json()

    def chat_claude(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a chat request to a premium Claude (Anthropic) model.

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: Claude model ID (default: claude-sonnet-4.6).
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in the response.
            system: System prompt (prepended to conversation).

        Returns:
            Dict with response content, usage, stop_reason.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system is not None:
            payload["system"] = system

        response = self._request("POST", "/v1/ai/chat/claude", json_data=payload)
        return response.json()

    def chat_bedrock(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Deprecated alias for :meth:`chat_claude`.

        .. deprecated:: 1.4.0
            Use :meth:`chat_claude` instead. This method will be removed in a
            future release.
        """
        warnings.warn(
            "chat_bedrock() is deprecated; use chat_claude() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat_claude(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

    def analyze_image(
        self,
        image_url: str,
        *,
        features: Optional[list[str]] = None,
        language: Optional[str] = None,
        max_labels: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> dict[str, Any]:
        """Analyze an image using vision models.

        Costs a flat 1 credit however many features are requested, so asking for
        everything in one call is cheaper than one call per feature.

        Args:
            image_url: URL of the image to analyze. Must be publicly reachable,
                max 20MB.
            features: Analysis features to run. Defaults to
                ``["labels", "objects"]``. Valid: ``labels``, ``objects``,
                ``faces``, ``nsfw``, ``colors``, ``ocr``, ``landmarks``,
                ``logos`` (plus the aliases ``text`` for ocr and
                ``safe_search`` for nsfw). An unrecognised name is rejected
                with 400 rather than ignored.
            language: Language hint for OCR. Label names are always English.
            max_labels: Cap on returned labels, 1-50 (default 50).
            min_confidence: Minimum confidence 0-1, applied to labels, objects
                and faces. ``nsfw`` and ``ocr`` carry no numeric score.

        Returns:
            Dict with the requested features at the top level (``labels``,
            ``objects``, ``faces``, ``nsfw``, ``ocr``, ``colors``, ...) plus
            ``auto_tags`` -- the flat deduplicated tag list. There is no
            ``analysis`` wrapper key.

            Faces and content safety return likelihood buckets
            (``VERY_UNLIKELY`` .. ``VERY_LIKELY``), not floats, and no age or
            gender. Bounding boxes carry ``units``: ``"pixels"`` for faces and
            OCR, ``"normalized"`` (0-1 fractions) for objects.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if features is not None:
            payload["features"] = features
        if language is not None:
            payload["language"] = language
        if max_labels is not None:
            payload["max_labels"] = max_labels
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence

        response = self._request("POST", "/v1/ai/analyze/image", json_data=payload)
        return response.json()

    def enhance_prompt(
        self,
        prompt: str,
        *,
        style: str = "photographic",
    ) -> str:
        """Enhance a short prompt into a detailed generation prompt.

        Args:
            prompt: Short input prompt.
            style: Target style ("photographic", "cinematic", "illustration",
                "3d", "anime").

        Returns:
            Enhanced prompt string.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "style": style,
        }

        response = self._request("POST", "/v1/ai/enhance-prompt", json_data=payload)
        data = response.json()
        return data.get("enhanced_prompt", data.get("prompt", prompt))

    # =========================================================================
    # Stability AI Tools
    # =========================================================================

    def stability_tools(self) -> list[dict[str, Any]]:
        """List available Stability AI tools, their price and their inputs.

        Returns:
            List of tool descriptors with ``id``, ``model_id``, ``price_usd``,
            ``currency``, ``unit`` and the three ``requires_*`` flags. There are no
            ``name``/``description``/``parameters`` keys -- those were never sent.

            ``price_usd`` is per image and spans $0.03 (fast upscale) to $0.60
            (creative upscale), so read it before picking a tool. ``credits`` is
            still present but deprecated: a legacy relative weight, not a price,
            and not proportional to one either.
        """
        response = self._request("GET", "/stability/tools")
        data = response.json()
        return data.get("tools", data) if isinstance(data, dict) else data

    def stability_run(
        self,
        tool_id: str,
        image_base64: str,
        *,
        mask: Optional[str] = None,
        prompt: Optional[str] = None,
        reference: Optional[str] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Stability AI tool on an image.

        Args:
            tool_id: The tool identifier (e.g. "remove-background", "upscale").
            image_base64: Base64-encoded input image.
            mask: Base64-encoded mask image (for inpaint/erase).
            prompt: Text prompt (for generation-based tools).
            reference: Base64-encoded reference image (for style transfer).
            seed: Random seed for reproducibility.
            negative_prompt: Things to avoid.
            output_format: Output format ("png", "webp", "jpeg").
            **kwargs: Additional tool-specific parameters.

        Returns:
            Dict with ``image`` (base64, never a URL), ``tool``, ``seed`` (None on
            the tools that do not sample), and the usual ``cost_usd`` / ``currency``
            / ``billing`` charge block. There is no ``credits_used``: these tools are
            paid in USD from the prepaid wallet, from $0.03 to $0.60 per image
            depending on the tool -- see :meth:`stability_tools`.
        """
        payload: dict[str, Any] = {
            "image": image_base64,
            "output_format": output_format,
        }
        if mask is not None:
            payload["mask"] = mask
        if prompt is not None:
            payload["prompt"] = prompt
        if reference is not None:
            payload["reference"] = reference
        if seed is not None:
            payload["seed"] = seed
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        payload.update(kwargs)

        response = self._request("POST", f"/stability/{tool_id}", json_data=payload)
        return response.json()

    def stability_upscale(
        self,
        image_base64: str,
        *,
        type: str = "fast",
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upscale an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.
            type: Upscale type — "fast" (4x, instant), "creative" (4x, slower,
                prompt-guided) or "conservative" (4x, faithful).
            prompt: Optional guiding prompt (used by creative/conservative).

        Returns:
            Dict with upscaled image data.
        """
        tool_id = f"{type}-upscale"
        return self.stability_run(tool_id, image_base64, prompt=prompt)

    def stability_remove_background(
        self,
        image_base64: str,
    ) -> dict[str, Any]:
        """Remove background from an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.

        Returns:
            Dict with transparent-background image data.
        """
        return self.stability_run("remove-background", image_base64)

    def stability_erase(
        self,
        image_base64: str,
        mask_base64: str,
    ) -> dict[str, Any]:
        """Erase regions from an image (content-aware fill).

        Args:
            image_base64: Base64-encoded input image.
            mask_base64: Base64-encoded mask (white = erase).

        Returns:
            Dict with result image data.
        """
        return self.stability_run("erase-object", image_base64, mask=mask_base64)

    def stability_inpaint(
        self,
        image_base64: str,
        mask_base64: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Inpaint masked regions of an image with a prompt.

        Args:
            image_base64: Base64-encoded input image.
            mask_base64: Base64-encoded mask (white = inpaint).
            prompt: What to paint into the masked region.

        Returns:
            Dict with inpainted image data.
        """
        return self.stability_run("inpaint", image_base64, mask=mask_base64, prompt=prompt)

    def stability_outpaint(
        self,
        image_base64: str,
        *,
        left: int = 0,
        right: int = 0,
        up: int = 0,
        down: int = 0,
    ) -> dict[str, Any]:
        """Extend an image beyond its borders (outpainting).

        Args:
            image_base64: Base64-encoded input image.
            left: Pixels to extend left.
            right: Pixels to extend right.
            up: Pixels to extend up.
            down: Pixels to extend down.

        Returns:
            Dict with extended image data.
        """
        return self.stability_run(
            "outpaint", image_base64, left=left, right=right, up=up, down=down
        )

    def stability_search_replace(
        self,
        image_base64: str,
        search_prompt: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Search for an object in an image and replace it.

        Args:
            image_base64: Base64-encoded input image.
            search_prompt: Description of what to find and replace.
            prompt: Description of the replacement.

        Returns:
            Dict with result image data.
        """
        return self.stability_run(
            "search-replace", image_base64, prompt=prompt, search_prompt=search_prompt
        )

    def stability_recolor(
        self,
        image_base64: str,
        search_prompt: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Recolor a specific object in an image.

        Args:
            image_base64: Base64-encoded input image.
            search_prompt: Description of the object to recolor.
            prompt: Description of the new color/appearance.

        Returns:
            Dict with recolored image data.
        """
        return self.stability_run(
            "search-recolor", image_base64, prompt=prompt, search_prompt=search_prompt
        )

    def stability_style_transfer(
        self,
        image_base64: str,
        reference_base64: str,
    ) -> dict[str, Any]:
        """Transfer the style of a reference image onto a source image.

        Args:
            image_base64: Base64-encoded source image.
            reference_base64: Base64-encoded style reference image.

        Returns:
            Dict with style-transferred image data.
        """
        return self.stability_run("style-transfer", image_base64, reference=reference_base64)

    # =========================================================================
    # Billing
    # =========================================================================

    def get_balance(self) -> dict[str, Any]:
        """Get the wallet balance and this month's spend, in USD.

        Returns:
            Dict with ``wallet`` (``balance_usd``, ``pending_usd``,
            ``total_topped_up_usd``, ``currency``), ``spend``
            (``this_month_usd``, ``monthly_limit_usd``, ``remaining_usd``),
            ``billing_model`` and ``api_subscription``. There is no ``credits``
            block: it used to report the web app's subscription counter, telling
            API developers they had hundreds of credits while their spendable
            balance was $0. ``overage`` is the old name for ``spend`` and is
            deprecated -- a prepaid wallet has nothing to exceed.
        """
        response = self._request("GET", "/v1/billing/balance")
        return response.json()

    def get_pricing(self) -> dict[str, Any]:
        """Get pricing for all AI operations.

        Returns:
            Dict with per-model credit costs by category.
        """
        response = self._request("GET", "/v1/billing/pricing")
        return response.json()

    def get_plans(self) -> dict[str, Any]:
        """Deprecated. There are no paid API plans; the list is always empty.

        .. deprecated:: 1.11.0
            Paid API plans were retired on 2026-08-13. The endpoint answers
            ``{"plans": []}`` -- a 200 with nothing to iterate. Rate limits
            follow the prepaid USD wallet, so :meth:`topup_wallet` /
            :meth:`create_topup` is the upgrade path and :meth:`compare_tiers`
            is what to show someone choosing limits.

        Returns:
            ``{"plans": []}``.
        """
        response = self._request("GET", "/v1/billing/plans")
        return response.json()

    def get_credits(self) -> dict[str, Any]:
        """Deprecated. The API has no credits; this returns the wallet.

        Kept because it is a published route, and the endpoint answers 200 with
        ``deprecated: True`` and an explanation rather than a 404. Use
        :meth:`get_balance` for the wallet and :meth:`get_pricing` for
        per-operation USD prices.

        Returns:
            Dict with ``deprecated``, ``message``, ``billing_model``, ``wallet``
            and ``spend``. No credit pools, no expiration: credits exist only in
            the fotohub.app web app and cannot pay for API usage.
        """
        response = self._request("GET", "/v1/billing/credits")
        return response.json()

    def set_overage_limit(
        self,
        hard_limit_usd: float,
        *,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set the hard spending limit for overage charges.

        Args:
            hard_limit_usd: Maximum monthly overage amount in USD. Pass 0 to
                disable (the wallet balance then becomes the only cap).
            project_id: Optional project ID (defaults to account-level).

        Returns:
            Dict confirming the updated limit (``hard_limit_usd``).
        """
        payload: dict[str, Any] = {"hard_limit_usd": hard_limit_usd}
        if project_id is not None:
            payload["project_id"] = project_id

        response = self._request("PUT", "/v1/billing/overage-limit", json_data=payload)
        return response.json()

    def get_topup_packages(self) -> list[dict[str, Any]]:
        """Get available wallet top-up packages.

        A package credits ``total_usd`` -- the amount paid plus any volume bonus.
        From $500 up, the ladder adds extra spendable dollars: 5% at $500 rising
        to 20% at $15 000, so $1 000 paid credits $1 100 and $15 000 credits
        $18 000. The bonus is ordinary balance, spendable on any operation.

        Returns:
            List of packages with ``slug``, ``name`` (the amount PAID),
            ``amount_usd``, ``bonus_usd``, ``total_usd`` and ``bonus_pct``.

            ``bonus_credits`` is gone: the old figure described a credit transfer
            nothing performed, and this product has no credit unit. ``bonus_usd``
            is dollars, and it really is credited.

        Note:
            Use :meth:`get_topup_package_list` when quoting a custom amount --
            it also returns the ladder and the accepted bounds.
        """
        response = self._request("GET", "/v1/billing/topup/packages")
        data = response.json()
        return data.get("packages", data) if isinstance(data, dict) else data

    def get_topup_package_list(self) -> dict[str, Any]:
        """Top-up packages plus the bonus ladder and the custom-amount bounds.

        Returns:
            Dict with ``packages``, ``bonus_tiers`` (highest ``min_usd`` first),
            ``min_usd``, ``max_usd`` and ``notes``.

        Example:
            Quote a bonus for an arbitrary amount without hardcoding the ladder::

                import math

                data = client.get_topup_package_list()

                def bonus_for(usd: float) -> float:
                    for tier in data["bonus_tiers"]:      # already sorted desc
                        if usd >= tier["min_usd"]:
                            # Floored to the cent, matching the server.
                            return math.floor(usd * tier["pct"] * 100) / 100
                    return 0.0

                bonus_for(2500)   # 300.0
        """
        response = self._request("GET", "/v1/billing/topup/packages")
        return response.json()

    def create_topup(self, package: str) -> dict[str, Any]:
        """Purchase a wallet top-up package.

        Args:
            package: Package slug. Starter rungs are ``"topup-50"`` ($15),
                ``"topup-100"`` ($25), ``"topup-250"`` ($60) and
                ``"topup-500"`` ($120) -- historical names that do NOT match
                their amounts. Bonus-earning rungs are ``"scale-500"``,
                ``"scale-1000"``, ``"scale-2000"``, ``"scale-3000"``,
                ``"scale-5000"``, ``"scale-7500"``, ``"scale-10000"`` and
                ``"scale-15000"``, where the number IS the amount in USD.
                Prefer :meth:`get_topup_packages` over a hardcoded slug.

        Returns:
            Dict with checkout_url and the purchased package descriptor.

        Example:
            ::

                topup = client.create_topup("scale-1000")  # pay $1000, get $1100
        """
        payload: dict[str, Any] = {"package": package}

        response = self._request("POST", "/v1/billing/topup", json_data=payload)
        return response.json()

    def get_transactions(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        type_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get credit transaction history.

        Args:
            page: Page number (starting at 1).
            page_size: Items per page (max 100).
            type_filter: Filter by type ("charge", "topup", "refund", "bonus").

        Returns:
            Dict with transactions list and pagination metadata.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if type_filter is not None:
            params["type"] = type_filter

        response = self._request("GET", "/v1/billing/transactions", params=params)
        return response.json()

    def estimate_cost(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Estimate the cost of a set of operations before running them.

        Args:
            operations: List of operation dicts, each with "type", "model",
                and relevant parameters (width, height, duration, etc.).

        Returns:
            Dict with ``total_usd``, ``provider_cost_usd``, ``margin``,
            ``currency``, ``balance_usd``, ``sufficient``, ``priced`` and a
            ``breakdown`` per operation. Because the account is prepaid, read
            ``sufficient`` -- the server's own answer to "can my wallet cover
            this" -- rather than comparing two numbers yourself. An operation with
            no published rate comes back ``priced: false`` with
            ``amount_usd: null``, and then ``total_usd`` covers only the priced
            legs. ``total_credits`` is deprecated and always ``None``.
        """
        payload: dict[str, Any] = {"operations": operations}

        response = self._request("POST", "/v1/billing/estimate", json_data=payload)
        return response.json()

    def get_invoices(self) -> dict[str, Any]:
        """Get billing invoices.

        Returns:
            Dict with list of invoices and their status.
        """
        response = self._request("GET", "/v1/billing/invoices")
        return response.json()

    # =========================================================================
    # 3D Generation
    # =========================================================================

    def generate_3d(
        self,
        mode: str,
        model: str,
        *,
        image: Optional[str] = None,
        prompt: Optional[str] = None,
        quality: str = "standard",
        format: str = "glb",
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate a 3D model from an image or text prompt.

        Synchronous: this returns the finished model. There is no job queue and
        nothing to poll -- `fh-pro-3d` can take ~60s, so give the client a long
        timeout rather than reaching for `wait_for_3d`.

        Charged in USD from the prepaid wallet before the GPU runs. A 402 means
        insufficient funds and nothing was taken; a failure on our side is
        refunded automatically.

        Args:
            mode: Generation mode — "image-to-3d" or "text-to-3d".
            model: 3D model to use. Only "fh-lite-3d" (image) and "fh-text-3d"
                (text) are enabled; "fh-pro-3d" validates but will not render.
            image: Base64-encoded image (required for image-to-3d).
            prompt: Text prompt (required for text-to-3d).
            quality: Output quality — "draft", "standard", "high". Does not
                affect the price.
            format: Output file format — "glb", "obj", "stl", "usdz".
            options: Additional options (texture, pbr, simplify, target_polys).

        Returns:
            Dict with `file_id`, `url` (signed, valid 2h), `stats`, `cost_usd`
            and `billing`. Note there is no `id`, `status`, `poly_count` or
            `thumbnail_url` -- those were documented but never returned.
        """
        payload: dict[str, Any] = {
            "mode": mode,
            "model": model,
            "quality": quality,
            "format": format,
        }
        if image is not None:
            payload["image_base64"] = image
        if prompt is not None:
            payload["prompt"] = prompt
        if options is not None:
            payload["options"] = options

        response = self._request("POST", "/v1/ai/generate/3d", json_data=payload)
        return response.json()

    def get_3d_status(self, job_id: str) -> dict[str, Any]:
        """Fetch a stored 3D asset with a freshly signed download URL.

        Not a status check -- `generate_3d()` is synchronous, so the model is
        already done when it returns. This exists for the expiry: the `url` from
        the generate call dies after 2 hours and this mints a new one. Free.

        Args:
            job_id: The `file_id` returned from generate_3d().

        Returns:
            Dict with a fresh `url`, plus `model`, `format` and `stats`.
            `status` is always "completed" -- an unfinished generation is never
            stored. Raises `NotFoundError` if the id is not yours.
        """
        response = self._request("GET", f"/v1/ai/generate/3d/{job_id}")
        return response.json()

    def wait_for_3d(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Deprecated -- there is nothing to wait for.

        `generate_3d()` is synchronous and returns the finished model, so this
        resolves on its first poll: a stored asset always reports "completed".
        Kept so existing code keeps working. New code should use the
        `generate_3d()` result directly, or `get_3d_status(file_id)` when it
        needs a fresh signed URL.

        Args:
            job_id: The `file_id` returned from generate_3d().
            poll_interval: Seconds between checks. Effectively unused now.
            timeout: Maximum wait in seconds (default: 120.0).

        Returns:
            Dict with the stored result including a fresh `url`.

        Raises:
            TimeoutError: Only if the asset lookup itself keeps failing.
            FotoHubError: If the stored record reports a failure.
        """
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    message=f"3D generation job {job_id} timed out after {timeout}s"
                )

            result = self.get_3d_status(job_id)
            status = result.get("status", "")

            if status == "completed":
                return result
            if status == "failed":
                raise FotoHubError(
                    message=f"3D generation job {job_id} failed",
                    status_code=500,
                    response_body=result,
                )

            time.sleep(poll_interval)

    def list_3d_models(self) -> list[dict[str, Any]]:
        """List available 3D generation models with capabilities and pricing.

        Returns:
            List of 3D models with id, name, `price_usd`, unit, speed, mode,
            `available` and quality. There is no `credits` key: the API is
            prepaid USD, so the catalog quotes dollars.
        """
        response = self._request("GET", "/v1/ai/generate/3d/models")
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    def list_models(self, category: Optional[str] = None) -> list[dict[str, Any]]:
        """List the model catalog with pricing.

        Read `price_unit` -- not `pricing_type` -- to know what `request_price`
        buys. `pricing_type` says "request" on every video model, but their
        price is per second of output.

        Args:
            category: Narrow to one of image, video, audio, text.

        Returns:
            List of models with id, name, request_price, price_unit,
            request_price_per, currency and limits.
        """
        params = {"category": category} if category else None
        response = self._request("GET", "/v1/models", params=params)
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    # =========================================================================
    # Virtual Try-On
    # =========================================================================

    def tryon(
        self,
        person_image_url: str,
        *,
        garment_image_url: Optional[str] = None,
        garment_id: Optional[str] = None,
        category: str = "tops",
        garment_photo_type: Optional[str] = None,
        garments: Optional[list[dict[str, Any]]] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Dress a person photo in a garment.

        Returns immediately with a job_id — a render takes ~11 s, so collect the
        result with wait_for_tryon() or poll get_tryon_status() yourself.

        Args:
            person_image_url: Publicly reachable URL of the person photo.
            garment_image_url: URL of the garment photo. Required unless
                garment_id or garments is given.
            garment_id: A catalogue garment. Supplies the image and overrides
                category and garment_photo_type.
            category: "tops", "bottoms" or "one-pieces".
            garment_photo_type: How the garment was shot — "flat-lay", "model" or
                "auto". Defaults to "flat-lay" server-side.
            garments: Two garments to apply in one job, e.g.
                [{"garment_image_url": ..., "category": "tops"},
                 {"garment_id": ..., "category": "bottoms"}].
                Exactly one top and one bottom, no one-pieces. Billed as one
                chained render (two Vertex passes) rather than two jobs, and
                forces num_images to 1. Order is irrelevant — the top is always
                applied first.
            num_images: Renders to produce, 1-4. Ignored for an outfit.
            seed: Fixed seed for reproducible output.

        Returns:
            Dict with job_id, status, category, ``cost_usd``, ``currency``,
            ``billing``, estimated_seconds and poll_url. The wallet is charged at
            submit, so the poll route never reports the cost.
        """
        payload: dict[str, Any] = {
            "person_image_url": person_image_url,
            "num_images": num_images,
        }
        # An outfit and a single garment are mutually exclusive request shapes;
        # sending both would leave the server to guess which was meant.
        if garments:
            payload["garments"] = garments
        else:
            if garment_image_url is not None:
                payload["garment_image_url"] = garment_image_url
            if garment_id is not None:
                payload["garment_id"] = garment_id
            payload["category"] = category
            if garment_photo_type is not None:
                payload["garment_photo_type"] = garment_photo_type
        if seed is not None:
            payload["seed"] = seed

        response = self._request("POST", "/v1/ai/tryon", json_data=payload)
        return response.json()

    def get_tryon_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a try-on job.

        Args:
            job_id: The job_id returned from tryon().

        Returns:
            Dict with status, progress, images (when completed) and
            error_message (when failed).
        """
        response = self._request("GET", f"/v1/ai/tryon/{job_id}")
        return response.json()

    def wait_for_tryon(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Wait for a try-on job to complete, polling at intervals.

        A partially failed outfit completes rather than fails: the top-only
        render comes back and one credit is refunded. Check
        result["metadata"]["partial_failure"] to detect that — this method
        treats it as success, because a paid-for render did arrive.

        Args:
            job_id: The job_id returned from tryon().
            poll_interval: Seconds between status checks (default: 3.0).
            timeout: Maximum wait time in seconds (default: 120.0).

        Returns:
            Dict with the completed job including images.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            FotoHubError: If the job fails.
        """
        start = time.time()
        while True:
            if time.time() - start >= timeout:
                raise TimeoutError(
                    message=f"Try-on job {job_id} timed out after {timeout}s"
                )

            result = self.get_tryon_status(job_id)
            status = result.get("status", "")

            if status == "completed":
                return result
            if status in ("failed", "cancelled"):
                raise FotoHubError(
                    message=result.get("error_message") or f"Try-on job {job_id} {status}",
                    status_code=500,
                    response_body=result,
                )

            time.sleep(poll_interval)

    # =========================================================================
    # Tier Management
    # =========================================================================

    def get_tier_catalog(self) -> dict[str, Any]:
        """Get the full tier catalog (PAYG + subscription tiers).

        Returns:
            Dict with ``payg`` and ``subscriptions`` lists (slug, name,
            description, limits, access), plus ``currency`` (``"USD"``),
            ``billing_cycle`` (``"prepaid"``), ``subscriptions_retired`` and
            ``overage_policy``.

            Nothing in the catalog has a price: ``price_monthly`` is ``0`` on
            PAYG entries and ``None`` on every ``sub-*`` one, all of which carry
            ``purchasable: False``. The ``subscriptions`` list survives because
            those rows are the live rate-limit definitions for accounts that
            already hold a ``sub-*`` tier -- not an offer.

            Watch ``limits``: ``-1`` is the sentinel for "no cap"
            (``sub-enterprise`` carries it on ``storage_gb``, ``daily_quota``
            and ``tpm``). Render a negative as unlimited, or it reads as a
            negative allowance.
        """
        response = self._request("GET", "/v1/tiers/catalog")
        return response.json()

    def get_current_tier(self) -> dict[str, Any]:
        """Get the current user's tier, limits, and usage stats.

        Returns:
            Dict with ``tier``, ``name``, ``category``, ``limits`` (rpm,
            burst_4h, concurrent_jobs, storage_gb, daily_quota, tpm), ``access``,
            ``usage`` (used_4h, used_period, requests_today), ``wallet``
            (balance_usd, pending_usd, lifetime_spend), ``subscription`` and
            ``upgrade_options``.
        """
        response = self._request("GET", "/v1/tiers/current")
        return response.json()

    def compare_tiers(self) -> dict[str, Any]:
        """Compare every tier side-by-side, flattened for a table.

        It does not tell you which tier is yours -- read ``tier`` from
        :meth:`get_current_tier`.

        Returns:
            Dict with ``tiers`` (flat rows: slug, name, category, description,
            purchasable, upgrade_path, rpm, concurrent_jobs, storage_gb, models,
            priority, sla, support), plus ``currency`` (``"USD"``),
            ``billing_model`` (``"prepaid_wallet_usd"``) and
            ``subscriptions_retired``.

            No row carries a price or a credit grant: every row is
            ``purchasable: False`` with an ``upgrade_path`` of ``"wallet_topup"``
            or, for ``sub-enterprise``, ``"contact_sales"``. Compare on ``rpm``
            and ``concurrent_jobs``; ``storage_gb`` of ``-1`` means uncapped.
        """
        response = self._request("GET", "/v1/tiers/compare")
        return response.json()

    def subscribe_tier(self, tier_slug: str) -> dict[str, Any]:
        """Retired on 2026-08-13 -- always raises.

        .. deprecated:: 1.5.0
            ``POST /v1/tiers/subscribe`` answers **HTTP 410** for every tier.
            There are no paid API plans any more: rate limits follow the prepaid
            wallet, so topping up raises the tier on its own with no monthly
            commitment to cancel. Use :meth:`topup_wallet` or
            :meth:`create_topup` instead -- and the swap is in your favour, since
            from $500 up a top-up earns a 5-20% volume bonus in extra spendable
            dollars.

            ``sub-enterprise`` was never bought this way; it is a contract, via
            :meth:`apply_enterprise`.

        Kept as a raising stub rather than deleted so upgrading gives a clear
        message pointing at the replacement instead of an ``AttributeError``.

        Args:
            tier_slug: Ignored.

        Raises:
            FotoHubError: Always, with ``status_code=410``.
        """
        raise FotoHubError(
            "API subscription plans were retired on 2026-08-13. Rate limits now "
            "follow your prepaid wallet balance, so top up instead: "
            "client.topup_wallet(amount_usd) or client.create_topup(package). "
            "Top-ups from $500 up earn a 5-20% volume bonus in extra spendable "
            "dollars. For sub-enterprise, use client.apply_enterprise(...).",
            status_code=410,
            response_body={"error": "api_subscriptions_retired"},
        )

    def get_wallet(self) -> dict[str, Any]:
        """Get the current wallet balance and spending info.

        Returns:
            Dict with balance, currency, lifetime_spend, auto_topup.
        """
        response = self._request("GET", "/v1/tiers/wallet")
        return response.json()

    def topup_wallet(
        self,
        amount_usd: float,
        *,
        pay_currency: Optional[str] = None,
    ) -> dict[str, Any]:
        """Top up wallet balance (returns a Stripe checkout URL).

        From $500 up the amount earns a volume bonus in extra spendable dollars
        -- 5% at $500, 10% at $1 000, rising to 20% at $15 000 -- credited in the
        same transaction as the payment. The bonus is a function of the amount,
        not of the package: passing ``1000`` here earns the same +$100 as buying
        the ``"scale-1000"`` package.

        Args:
            amount_usd: Amount in USD to add (minimum 10, maximum 15000, whole
                cents only -- $10.005 is rejected rather than rounded).
            pay_currency: Optional Stripe charge currency, ``"usd"`` (default)
                or ``"pln"``. With ``"pln"`` a Polish customer pays by
                BLIK/card/bank transfer while the wallet is still credited
                ``amount_usd``.

        Returns:
            Dict with ``checkout_url``, ``amount_usd`` (charged), ``bonus_usd``,
            ``total_credited_usd`` (the balance increase -- the figure to show a
            customer) and ``pay_currency``.

            ``bonus_credits`` is still on the wire but always ``None`` and
            deprecated: it described a credit transfer nothing performed, and
            this product has no credit unit. It is null rather than 0 because 0
            would read as "this package has no bonus" instead of "there is no
            such thing".

        Example:
            ::

                topup = client.topup_wallet(1000)
                print(topup["amount_usd"], "->", topup["total_credited_usd"])
                # 1000 -> 1100.0
        """
        payload: dict[str, Any] = {"amount_usd": amount_usd}
        if pay_currency is not None:
            payload["pay_currency"] = pay_currency
        response = self._request("POST", "/v1/tiers/wallet/topup", json_data=payload)
        return response.json()

    def apply_enterprise(
        self,
        company_name: str,
        contact_email: str,
        expected_usage: str,
        use_case: str,
        *,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit an enterprise tier application.

        Args:
            company_name: Company or organization name.
            contact_email: Contact email for follow-up.
            expected_usage: Expected monthly usage description.
            use_case: Primary use case description.
            notes: Additional notes or requirements.

        Returns:
            Dict with application id and status.
        """
        payload: dict[str, Any] = {
            "company_name": company_name,
            "contact_email": contact_email,
            "expected_usage": expected_usage,
            "use_case": use_case,
        }
        if notes is not None:
            payload["notes"] = notes

        response = self._request("POST", "/v1/tiers/enterprise/apply", json_data=payload)
        return response.json()

    # =========================================================================
    # Webhooks
    # =========================================================================

    def list_webhooks(self) -> list[dict[str, Any]]:
        """List all webhook endpoints.

        Returns:
            List of webhook configurations.
        """
        response = self._request("GET", "/v1/console/webhooks")
        data = response.json()
        return data.get("webhooks", data) if isinstance(data, dict) else data

    def create_webhook(
        self,
        name: str,
        url: str,
        events: list[str],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create a new webhook endpoint.

        Args:
            name: Human-readable name for the webhook.
            url: The URL to receive webhook events.
            events: List of event types (e.g. ["generation.completed",
                "generation.failed", "credits.low"]).
            headers: Custom headers to include in webhook requests.

        Returns:
            Dict with webhook ID, secret, and configuration.
        """
        payload: dict[str, Any] = {
            "name": name,
            "url": url,
            "events": events,
        }
        if headers is not None:
            payload["headers"] = headers

        response = self._request("POST", "/v1/console/webhooks", json_data=payload)
        return response.json()

    def update_webhook(self, webhook_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an existing webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
            **kwargs: Fields to update (url, events, name, headers, active).

        Returns:
            Dict with updated webhook configuration.
        """
        response = self._request(
            "PATCH", f"/v1/console/webhooks/{webhook_id}", json_data=kwargs
        )
        return response.json()

    def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
        """
        self._request("DELETE", f"/v1/console/webhooks/{webhook_id}")

    def test_webhook(self, webhook_id: str) -> dict[str, Any]:
        """Send a test event to a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            Dict with delivery status and response code.
        """
        response = self._request("POST", f"/v1/console/webhooks/{webhook_id}/test")
        return response.json()

    def get_webhook_logs(self, webhook_id: str) -> list[dict[str, Any]]:
        """Get delivery logs for a webhook.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            List of delivery log entries with status, timestamp, response.
        """
        response = self._request("GET", f"/v1/console/webhooks/{webhook_id}/logs")
        data = response.json()
        return data.get("logs", data) if isinstance(data, dict) else data

    # =========================================================================
    # Gabriel AI Orchestrator
    # =========================================================================

    def gabriel_classify(
        self,
        prompt: str,
        *,
        language: str = "en",
        context: Optional[dict[str, Any]] = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        """Classify user intent and route to the optimal platform feature.

        Args:
            prompt: Natural language request (max 1000 chars).
            language: Language code (default: "en").
            context: Additional context (user_tier, credits_remaining, etc.).
            enhance_prompt: When True, enriches prompt with model-specific knowledge.

        Returns:
            Dict with action, target, params, model_selected, confidence,
            credits_estimated, tips.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "language": language,
        }
        if context is not None:
            payload["context"] = context
        if enhance_prompt:
            payload["enhance_prompt"] = True

        response = self._request("POST", "/v1/ai/gabriel", json_data=payload)
        return response.json()

    def gabriel_stream(
        self,
        prompt: str,
        *,
        language: str = "en",
        context: Optional[dict[str, Any]] = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream orchestration results via SSE.

        Args:
            prompt: Natural language request.
            language: Language code (default: "en").
            context: Additional context.

        Yields:
            Dicts with type (thinking/routing/result) and payload.
        """
        import json as json_mod

        payload: dict[str, Any] = {
            "prompt": prompt,
            "language": language,
        }
        if context is not None:
            payload["context"] = context

        with self._client.stream(
            "POST", "/v1/ai/gabriel/stream", json=payload
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json_mod.loads(data)

    def gabriel_suggest(
        self,
        partial: str,
        *,
        tab: str = "all",
        page: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get lightweight autocomplete suggestions (no auth required).

        Args:
            partial: Partial user input (min 2 chars).
            tab: Current tab context ("all", "image", "video", "audio").
            page: Current page path.

        Returns:
            List of suggestion dicts with text, category, target, icon.
        """
        payload: dict[str, Any] = {
            "partial": partial,
            "tab": tab,
        }
        if page is not None:
            payload["page"] = page

        response = self._request("POST", "/v1/ai/gabriel/suggest", json_data=payload)
        data = response.json()
        return data.get("suggestions", [])

    def gabriel_recommend(
        self,
        *,
        page: Optional[str] = None,
        credits_remaining: Optional[int] = None,
        has_brand: Optional[bool] = None,
        recent_actions: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Get proactive context-aware recommendations (no auth required).

        Args:
            page: Current page path.
            credits_remaining: User's credit balance.
            has_brand: Whether user has a brand kit.
            recent_actions: Last few actions taken.

        Returns:
            List of recommendation dicts with text, target, icon.
        """
        payload: dict[str, Any] = {}
        if page is not None:
            payload["page"] = page
        if credits_remaining is not None:
            payload["credits_remaining"] = credits_remaining
        if has_brand is not None:
            payload["has_brand"] = has_brand
        if recent_actions is not None:
            payload["recent_actions"] = recent_actions

        response = self._request("POST", "/v1/ai/gabriel/recommend", json_data=payload)
        data = response.json()
        return data.get("recommendations", [])

    def translate(
        self,
        text: str,
        target_language: str,
        *,
        source_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Translate text between languages.

        Args:
            text: Text to translate (max 10,000 chars).
            target_language: Target language code (e.g. "en", "pl", "de").
            source_language: Source language (auto-detected if omitted).

        Returns:
            Dict with translated_text, source_language, target_language.
        """
        payload: dict[str, Any] = {
            "text": text,
            "target_language": target_language,
        }
        if source_language is not None:
            payload["source_language"] = source_language

        response = self._request("POST", "/v1/ai/translate", json_data=payload)
        return response.json()

    # =========================================================================
    # Convenience Helpers
    # =========================================================================

    def remove_background(self, image_url: str) -> dict[str, Any]:
        """Remove the background from an image (convenience wrapper).

        Args:
            image_url: URL of the source image.

        Returns:
            Dict with processed image URL and metadata.
        """
        return self.edit_image(image_url, "remove background", mode="remove_bg")

    def upscale_image(self, image_url: str, *, scale: int = 2) -> dict[str, Any]:
        """Upscale an image to higher resolution (convenience wrapper).

        Args:
            image_url: URL of the image to upscale.
            scale: Upscale factor (2 or 4, default: 2).

        Returns:
            Dict with upscaled image URL and metadata.
        """
        return self.edit_image(
            image_url, f"upscale {scale}x", mode="upscale", scale=scale
        )

    def wait_for_video(
        self,
        result: Union[str, dict[str, Any]],
        *,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Return a finished video result.

        .. deprecated:: 1.4.0
            :meth:`generate_video` already waits for the finished ``video_url``,
            polling on your behalf for the models that queue (Wan, Grok). This
            method just returns its result dict unchanged, and will be removed in
            a future release.

        Args:
            result: The dict returned by :meth:`generate_video`.
            poll_interval: Unused (kept for backwards compatibility).
            timeout: Unused (kept for backwards compatibility).

        Returns:
            The finished video result dict.
        """
        warnings.warn(
            "wait_for_video() is deprecated; generate_video() already returns "
            "the finished video_url, polling when the model queues.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(result, dict):
            return result
        raise FotoHubError(
            "wait_for_video() no longer accepts a job_id: generate_video() "
            "polls for you. Pass the dict it returned (or just read its "
            "'video_url')."
        )

    # =========================================================================
    # Video timeline (headless editor API: /v1/video/projects)
    # =========================================================================

    def create_video_project(
        self,
        *,
        title: Optional[str] = None,
        aspect: Optional[str] = None,
        fps: Optional[int] = None,
        media: Optional[list[dict[str, Any]]] = None,
        template: Optional[Union[str, dict[str, Any]]] = None,
        place_media: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a timeline project, optionally seeded with media. Free.

        The project uses the same document as the FOTOhub video editor, so the
        result opens in the browser at ``editorUrl``.

        Args:
            title: Project title.
            aspect: "16:9", "9:16", "1:1", "4:5" or "4:3".
            fps: Frames per second.
            media: Up to 50 items, each ``{"url": "https://..."}`` (public HTTPS;
                FOTOhub copies the file into your storage) or
                ``{"storage_path": "<bucket>/<userId>/..."}``, plus optional
                ``kind`` ("video", "audio", "image") and ``name``.
            template: A template id (or ``{"id": ...}``) to start from.
            place_media: "sequence" (default) lays the media on the timeline one
                after another; "none" only adds them to the project.
            idempotency_key: Override the automatic ``X-Idempotency-Key``. Retrying
                with the same key within 24 h returns the first project instead
                of creating a second one.

        Returns:
            :class:`~fotohub.models.VideoProject` dict: ``projectId``, ``saveRev``,
            ``digest``, ``media``, ``unplacedMedia`` (media that did not fit on the
            timeline) and ``editorUrl``.

        Raises:
            FotoHubError: ``code`` is ``media-blocked`` (URL not allowed),
                ``media-too-large`` or ``media-not-found``.
        """
        response = self._request(
            "POST", "/v1/video/projects",
            json_data=_video_project_payload(
                title=title, aspect=aspect, fps=fps, media=media,
                template=template, place_media=place_media,
            ),
            idempotency_key=idempotency_key,
        )
        return response.json()

    def list_video_projects(self, *, limit: int = 50) -> dict[str, Any]:
        """List your API-created video projects (newest first). Free.

        Returns:
            Dict with ``projects``: ``[{projectId, title, updatedAt, editorUrl}]``.
        """
        response = self._request("GET", "/v1/video/projects", params={"limit": limit})
        return response.json()

    def get_video_project(
        self, project_id: str, *, include_doc: bool = False
    ) -> dict[str, Any]:
        """Fetch a project: digest, media (with fresh URLs), versions and ``saveRev``. Free.

        Args:
            project_id: The ``projectId`` from :meth:`create_video_project`.
            include_doc: Also return the full editor document as ``doc``.

        Returns:
            :class:`~fotohub.models.VideoProject` dict. A project that is not
            yours is reported as not found (HTTP 404), never as forbidden.
        """
        response = self._request(
            "GET", f"/v1/video/projects/{project_id}",
            params={"include": "doc"} if include_doc else None,
        )
        return response.json()

    def delete_video_project(self, project_id: str) -> dict[str, Any]:
        """Delete an API-created project. Free.

        Args:
            project_id: The ``projectId`` to delete.
        """
        response = self._request("DELETE", f"/v1/video/projects/{project_id}")
        return response.json()

    def apply_video_ops(
        self,
        project_id: str,
        ops: list[dict[str, Any]],
        *,
        dry_run: bool = False,
        expected_save_rev: Optional[int] = None,
        label: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply up to 40 editing operations to a project as one atomic batch. Free.

        The operation shapes are listed by :meth:`get_video_ops_catalog`. If any
        operation is rejected the whole batch is rolled back: the project is
        unchanged and the result has ``rolledBack: True`` with ``violations``
        (HTTP 200, not an exception).

        Args:
            project_id: The project to edit.
            ops: Operation objects. Each one is discriminated by ``op``; the
                full list with schemas is :meth:`get_video_ops_catalog`. For
                example ``{"op": "insertClip", "ref": "intro", "at": {...},
                "clip": {...}}``. ``ref`` names a new clip so later operations in
                the batch, and your code (via ``refs`` in the result), can
                address it.
            dry_run: Validate and preview the effect without saving.
            expected_save_rev: The ``saveRev`` you last read. If the project
                changed since (the browser editor, another agent), nothing is
                written and :class:`~fotohub.SaveConflictError` is raised.
                Set it whenever you can: without it a timeout or a 5xx is
                **not** retried automatically (the batch may already have been
                saved, and repeating it would apply it twice), so you get the
                error and must re-read the project yourself.
            label: Name for the version snapshot saved with this change.

        Returns:
            :class:`~fotohub.models.ApplyOpsResult` dict: ``ok``, ``rolledBack``,
            ``violations``, per-operation ``results`` with ``summary``,
            ``accepted`` / ``rejected`` counts, ``refs`` (your ``ref`` names
            mapped to the ids of the clips they created), the new ``saveRev``,
            ``digestDelta``, ``versionSaved`` and ``warnings``.

        Raises:
            SaveConflictError: 409 ``save-conflict``; re-read the project
                (``current_save_rev``) and re-apply.
            ValidationError: 422 ``invalid-ops`` (schema path in ``details``).
        """
        body = _drop_none({
            "ops": ops,
            "dryRun": True if dry_run else None,
            "expectedSaveRev": expected_save_rev,
            "label": label,
        })
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/ops", json_data=body,
            retry_ambiguous=expected_save_rev is not None,
        )
        return response.json()

    def digest_video_project(
        self,
        project_id: str,
        *,
        clip_ids: Optional[list[str]] = None,
        view: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the project digest, or detailed data for up to 10 clips. Free.

        Args:
            project_id: The project to read.
            clip_ids: Return details for these clips (max 10).
            view: "digest" or "clips".
        """
        body = _drop_none({"clipIds": clip_ids, "view": view})
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/digest", json_data=body
        )
        return response.json()

    def lint_video_project(
        self,
        project_id: str,
        *,
        rules: Optional[list[str]] = None,
        severity: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, Any]:
        """Check a project for editing problems (gaps, clipping, overlaps, ...). Free.

        Args:
            project_id: The project to check.
            rules: Only run these rule ids.
            severity: Severities to report: "error", "warn" and/or "info" (one
                string or a list).

        Returns:
            :class:`~fotohub.models.LintResult` dict. While the checker is not
            deployed the call still succeeds, with ``available: False`` and
            ``warnings: ["lint-unavailable"]``. A 501 ``lint-unavailable`` error
            means the whole endpoint is missing.
        """
        body = _drop_none({
            "rules": rules,
            "severity": [severity] if isinstance(severity, str) else severity,
        })
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/lint", json_data=body
        )
        return response.json()

    def capture_video_project(
        self,
        project_id: str,
        *,
        times: Optional[list[float]] = None,
        count: Optional[int] = None,
        cuts: bool = False,
        width: Optional[int] = 640,
        sheet: Optional[dict[str, int]] = None,
        wait: bool = False,
        max_wait: float = 300.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Render still frames of the timeline into contact sheets, to see the edit. Paid, flat fee.

        Give exactly one of ``times``, ``count`` or ``cuts=True``.

        Args:
            project_id: The project to capture.
            times: Timeline positions in seconds (up to 24).
            count: That many frames spread evenly over the timeline.
            cuts: One frame at every cut.
            width: Frame width in pixels (16-1280, default 640).
            sheet: Contact sheet layout ``{"max_cells": 1-12, "max_edge": 256-1568}``.
            wait: Poll until the job finishes and return it (see
                :meth:`wait_for_video_job`).
            max_wait: Seconds to wait when ``wait`` is true.
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            Without ``wait``: a queued :class:`~fotohub.models.VideoJob`
            (``jobId``, ``status``, ``times``, ``width``, ``height``). With
            ``wait``: the completed job, carrying a
            :class:`~fotohub.models.CaptureResult` (``frames``, ``sheets``,
            ``missing``).

        Raises:
            RateLimitError: 429 ``rate-limited`` with ``retry_after``.
            AuthError: 403 ``payment-required`` when the wallet is empty.
        """
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/capture",
            json_data=_video_capture_payload(
                times=times, count=count, cuts=cuts, width=width, sheet=sheet
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    def render_video_project(
        self,
        project_id: str,
        *,
        format: str = "mp4",
        quality: str = "high",
        resolution: Optional[str] = None,
        codec: Optional[str] = None,
        fps: Optional[int] = None,
        bitrate: Optional[str] = None,
        time_range: Optional[tuple[float, float]] = None,
        content_credentials: Optional[bool] = None,
        content_ai_declared: Optional[bool] = None,
        wait: bool = False,
        max_wait: float = 1800.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Render the project to a video file. Paid per output minute.

        The charge is returned automatically if the render fails.

        Args:
            project_id: The project to render.
            format: "mp4", "webm", "mov", "gif", "mp3" or "wav".
            quality: "draft", "standard", "high" or "ultra".
            resolution: "720p", "1080p", "2k" or "4k".
            codec: "h264", "h265" or "prores".
            fps: Output frame rate (1-120).
            bitrate: e.g. "8M" or "800k".
            time_range: Render only ``(start, end)`` seconds of the timeline.
            content_credentials: Embed C2PA content credentials.
            content_ai_declared: Declare AI-generated content in them.
            wait: Poll until the render finishes and return the completed job.
            max_wait: Seconds to wait when ``wait`` is true (default 1800).
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            Without ``wait``: a queued :class:`~fotohub.models.VideoJob`
            (``jobId``, ``billedMinutes``). With ``wait``: the completed job with
            ``outputUrl``.

        Raises:
            VideoJobFailedError: With ``wait``, if the render fails (``refunded``
                tells whether the charge was returned).
            VideoJobTimeoutError: With ``wait``, if ``max_wait`` elapses; the job
                keeps running, poll it with :meth:`get_video_job`.
        """
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/render",
            json_data=_video_render_payload(
                format=format, quality=quality, resolution=resolution, codec=codec,
                fps=fps, bitrate=bitrate, time_range=time_range,
                content_credentials=content_credentials,
                content_ai_declared=content_ai_declared,
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    def auto_edit_video_project(
        self,
        project_id: str,
        *,
        style: Optional[str] = None,
        toggles: Optional[dict[str, Any]] = None,
        language: Optional[str] = None,
        aspect: Optional[str] = None,
        ai_budget_usd: float = 0,
        auto_apply: bool = True,
        mode: str = "auto_edit",
        wait: bool = False,
        max_wait: float = 1800.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Let FOTOhub edit the project for you (server-side Auto-Edit). Paid.

        .. note:: Provisional. The server route ships with the Auto-Edit release
           and its body and result may still change; do not rely on it yet.

        Args:
            project_id: The project to edit.
            style: "viral", "podcast", "explainer", "storytelling" or "captions-only".
            toggles: Feature switches, as in the editor's Auto-Edit panel.
            language: Spoken language ("auto" to detect).
            aspect: Target aspect ratio.
            ai_budget_usd: Cap for AI-generated media, 0-50 (0 = stock only).
            auto_apply: Commit the result; if false it stays a draft.
            mode: "auto_edit" or "cut".
            wait: Poll until finished and return the completed job (with ``report``).
            max_wait: Seconds to wait when ``wait`` is true.
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            A queued :class:`~fotohub.models.VideoJob`, or the finished one with ``wait``.
        """
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/auto-edit",
            json_data=_video_auto_edit_payload(
                style=style, toggles=toggles, language=language, aspect=aspect,
                ai_budget_usd=ai_budget_usd, auto_apply=auto_apply, mode=mode,
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    def apply_video_auto_edit(
        self,
        project_id: str,
        job_id: str,
        *,
        expected_save_rev: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Commit the draft of an Auto-Edit job started with ``auto_apply=False``. Free.

        .. note:: Provisional. The server route ships with the Auto-Edit release
           and its body and result may still change; do not rely on it yet.

        Args:
            project_id: The project the job edited.
            job_id: The ``jobId`` returned by :meth:`auto_edit_video_project`.
            expected_save_rev: The ``saveRev`` you last read; if the project
                changed since, nothing is written and
                :class:`~fotohub.SaveConflictError` is raised (the draft stays).
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            The apply result with the new ``saveRev``.
        """
        body = _drop_none({"expectedSaveRev": expected_save_rev})
        response = self._request(
            "POST", f"/v1/video/projects/{project_id}/auto-edit/{job_id}/apply",
            json_data=body, idempotency_key=idempotency_key,
        )
        return response.json()

    def get_video_job(self, job_id: str) -> dict[str, Any]:
        """Read the state of a render / capture / auto-edit job. Free.

        Returns:
            :class:`~fotohub.models.VideoJob` dict. ``status`` is "queued",
            "running", "completed", "failed" or "cancelled"; a failed job carries
            ``error`` and ``refunded``.
        """
        response = self._request("GET", f"/v1/video/jobs/{job_id}")
        return response.json()

    def wait_for_video_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Poll a render / capture / auto-edit job until it completes.

        Args:
            job_id: The ``jobId`` returned by the render, capture or auto-edit call.
            poll_interval: Seconds between checks (default 3.0).
            timeout: Maximum wait in seconds (default 1800.0).

        Returns:
            The completed :class:`~fotohub.models.VideoJob`.

        Raises:
            VideoJobFailedError: The job ended "failed" or "cancelled".
            VideoJobTimeoutError: ``timeout`` elapsed first (the job may still finish).
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_video_job(job_id)
            status = job.get("status")
            if status == "completed":
                return job
            if status in _VIDEO_JOB_FAILED:
                raise _video_job_failure(job)
            if time.monotonic() + poll_interval > deadline:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} not finished after {timeout}s (last status: {status})",
                    job_id=job_id,
                )
            time.sleep(poll_interval)

    def get_video_ops_catalog(self) -> dict[str, Any]:
        """The JSON schema of every operation :meth:`apply_video_ops` accepts. Free."""
        response = self._request("GET", "/v1/video/ops/catalog")
        return response.json()

    def detect_video_scenes(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        threshold: float = 0.4,
        min_scene_duration: float = 0.5,
    ) -> dict[str, Any]:
        """Find scene cuts in a video. Paid per request.

        Give either ``url`` (public HTTPS) or ``project_id`` + ``media_id``
        (an ``assetId`` from the project's media).

        Args:
            url: Public HTTPS URL of the video.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            threshold: Cut sensitivity 0-1 (default 0.4).
            min_scene_duration: Shortest scene in seconds (default 0.5).
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "threshold": threshold,
            "minSceneDuration": min_scene_duration,
        }
        response = self._request("POST", "/v1/video/detect-scenes", json_data=body)
        return response.json()

    def detect_video_silence(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        noise_floor_db: float = -30.0,
        min_silence_duration: float = 0.3,
    ) -> dict[str, Any]:
        """Find silent ranges in audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            noise_floor_db: Level below which audio counts as silence (default -30).
            min_silence_duration: Shortest silence in seconds (default 0.3).
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "noiseFloorDb": noise_floor_db,
            "minSilenceDuration": min_silence_duration,
        }
        response = self._request("POST", "/v1/video/detect-silence", json_data=body)
        return response.json()

    def detect_video_beats(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Find beats and tempo in audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
        """
        body = _video_source_payload(url=url, project_id=project_id, media_id=media_id)
        response = self._request("POST", "/v1/video/detect-beats", json_data=body)
        return response.json()

    def transcribe_video(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        language: str = "auto",
        hotwords: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Start a transcription job for audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            language: Language code, or "auto" (default).
            hotwords: Up to 50 words/names to favour.

        Returns:
            Dict with the transcription ``jobId``; read it with
            :meth:`get_video_transcription`.
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "language": language,
            **_drop_none({"hotwords": hotwords}),
        }
        response = self._request("POST", "/v1/video/transcribe", json_data=body)
        return response.json()

    def get_video_transcription(self, job_id: str) -> dict[str, Any]:
        """Read a transcription job started by :meth:`transcribe_video`. Free.

        Returns:
            Dict with ``status`` ("queued", "processing", "completed", "failed"),
            ``progress`` and, when completed, ``result``.
        """
        response = self._request("GET", f"/v1/video/transcribe/{job_id}")
        return response.json()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "FotoHub":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Asynchronous Client
# ---------------------------------------------------------------------------


class AsyncFotoHub(_BaseClient):
    """Asynchronous FOTOhub API client.

    Usage::

        from fotohub import AsyncFotoHub

        async with AsyncFotoHub(api_key="your-api-key") as client:
            result = await client.generate_image(prompt="A sunset over mountains")
            print(result["images"][0]["url"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        super().__init__(api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        stream: bool = False,
        idempotency_key: Optional[str] = None,
        retry_ambiguous: bool = True,
    ) -> httpx.Response:
        """Make an async HTTP request with retry logic.

        `retry_ambiguous=False`: see the sync client.

        Same idempotency contract as the sync client: one key per logical call,
        reused across that call's retries. See `_idempotency_key_for`.
        """
        import asyncio

        last_exception: Optional[Exception] = None
        idem_key = idempotency_key or _idempotency_key_for(method, path, stream)
        extra_headers = {IDEMPOTENCY_HEADER: idem_key} if idem_key else None

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = await self._client.stream(
                        method, path, json=json_data, params=params
                    ).__aenter__()
                else:
                    response = await self._client.request(
                        method, path, json=json_data, params=params,
                        headers=extra_headers,
                    )

                if response.status_code < 400:
                    return response

                if self._should_retry(
                    response, idempotent=idem_key is not None,
                    retry_ambiguous=retry_ambiguous,
                ) and attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = max(delay, float(retry_after))
                    await asyncio.sleep(delay)
                    continue

                self._handle_error_response(response)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                # A read/write timeout may hide an applied write; only a failure
                # to connect at all is known to have changed nothing.
                if not retry_ambiguous and not isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    raise TimeoutError(message=f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self._backoff_delay(attempt))
                    continue
                raise TimeoutError(message=f"Request failed: {e}")

        if last_exception:
            raise TimeoutError(message=f"Request failed after {self.max_retries} retries")
        raise FotoHubError("Unexpected retry exhaustion")

    # =========================================================================
    # AI Generation
    # =========================================================================

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        width: int = 1024,
        height: int = 1024,
        aspect_ratio: str = "1:1",
        num_images: int = 1,
        image_size: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Model to use (default: seedream-5-0-260128).
            width: Image width in pixels.
            height: Image height in pixels.
            aspect_ratio: Aspect ratio string (e.g. "1:1", "16:9", "9:16").
            num_images: Whole number of images, 1-8. Charged per image the
                provider actually delivers: every provider caps the count at its
                own maximum, and the difference is refunded automatically.
            image_size: Resolution tier -- "1K", "1.5K", "2K", "3K" or "4K".
                This is priced: 4K costs more than 1K on any model offering it.
                Absent from this client until now, which is why an async caller
                could not ask for "1.5K" at all and had to express a tier through
                width/height. Leave it None to let width/height pick the tier.
            negative_prompt: Things to avoid in the image.
            style: Style preset (e.g. "photographic", "cinematic", "anime").
            seed: Random seed for reproducibility.

        Returns:
            Dict with ``images`` (list of URLs), ``model``, ``cost_usd``,
            ``currency`` and a ``billing`` block. There is no ``credits_used``:
            the API is prepaid in USD and reads no credit balance.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged.
            ValidationError: If parameters are invalid.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
        }
        if image_size is not None:
            payload["image_size"] = image_size
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if style is not None:
            payload["style"] = style
        if seed is not None:
            payload["seed"] = seed

        response = await self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return response.json()

    async def generate_ida_q(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        num_images: int = 1,
        seed: Optional[int] = None,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Generate an image with IDA Q 1.0, FOTOhub's proprietary image model.

        Unlike :meth:`generate_image`, IDA Q 1.0 runs on a self-hosted, single-GPU
        queue and is asynchronous — generation takes 30 seconds to ~3.5 minutes
        depending on ``image_size``. This method submits the job and polls until
        it completes, returning the finished result. Any prompt (including
        non-English text) is automatically translated and restructured for best
        results — see the `IDA Q 1.0 docs <https://docs.fotohub.app/api/ida-q>`_.

        Args:
            prompt: Text description of the desired image. Any language.
            aspect_ratio: One of "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
                "21:9". This is the one thing the render never trades away: where
                the GPU's 2048-per-edge ceiling applies, the resolution gives way
                and the ratio is kept.
            image_size: Resolution tier — "1K" (~30s), "1.5K" (~90s), or "2K"
                (~3.5min). "3K" and "4K" are accepted and capped to "2K"; the model
                renders at most 2048x2048.
            num_images: Number of images to generate (1-2). Higher values are
                clamped to 2 before billing.
            seed: Random seed for reproducibility.
            poll_interval: Seconds to wait between status checks. The poll endpoint
                is rate-limited per ACCOUNT by tier (30/min on the lowest), and the
                default 3s costs 20 of those a minute, so raise this for a 2K
                render or two concurrent jobs will throttle each other.
            timeout: Maximum seconds to wait for completion before raising.

        Returns:
            Dict with ``images`` (list of URLs), ``model``, ``job_id``,
            ``cost_usd`` and ``billing``. IDA Q is self-hosted and renders at
            $0.00, so ``cost_usd`` is 0 -- and because a zero-price operation is
            settled without a balance check, this is the one model an empty wallet
            can still run.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged. Cannot happen at the current $0.00 price.
            TimeoutError: If generation doesn't complete within ``timeout``.
            FotoHubError: If generation fails server-side.
        """
        import asyncio

        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": "ida-q-image",
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "num_images": num_images,
        }
        if seed is not None:
            payload["seed"] = seed

        submit_response = await self._request("POST", "/v1/ai/generate/image", json_data=payload)
        job = submit_response.json()
        job_id = job["job_id"]

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            status_response = await self._request("GET", f"/v1/ai/generate/image/ida-q/{job_id}")
            status = status_response.json()
            if status["status"] == "completed":
                return {
                    "model": "ida-q-image",
                    "job_id": job_id,
                    # See the sync twin: the charge lands at submit, the poll
                    # reports job state only. `credits_used` was always None here
                    # because the prepaid API does not return that field.
                    "cost_usd": job.get("cost_usd", (job.get("billing") or {}).get("cost_usd")),
                    "currency": "USD",
                    "billing": job.get("billing"),
                    "images": status.get("images", []),
                    "metadata": status.get("metadata"),
                }
            if status["status"] == "failed":
                raise FotoHubError(status.get("error", "IDA Q 1.0 generation failed"))
            await asyncio.sleep(poll_interval)

        raise TimeoutError(message=f"IDA Q 1.0 job {job_id} did not complete within {timeout}s")

    async def edit_image(
        self,
        image_url: str,
        prompt: str,
        *,
        mode: str = "inpaint",
        mask_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Edit an existing image using AI.

        Args:
            image_url: URL of the source image.
            prompt: Instruction for the edit.
            mode: Edit mode — "inpaint", "outpaint", "remove_bg", "upscale",
                "style_transfer".
            mask_url: URL of the mask image (required for inpaint/erase).
            model: Model override.
            **kwargs: Additional parameters.

        Returns:
            Dict with edited image URL and metadata.
        """
        payload: dict[str, Any] = {
            "image_url": image_url,
            "prompt": prompt,
            "mode": mode,
        }
        if mask_url is not None:
            payload["mask_url"] = mask_url
        if model is not None:
            payload["model"] = model
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/edit/image", json_data=payload)
        return response.json()

    async def generate_video(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_VIDEO_MODEL,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        image_url: Optional[str] = None,
        resolution: str = "1080p",
        poll_interval: float = 5.0,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        """Generate a video, awaiting the finished file.

        Most models render inside the request and come back finished. Some
        (Alibaba Wan, xAI Grok) answer immediately with ``status: "processing"``
        and a ``job_id`` instead — so this polls until the job reaches a terminal
        state and returns the completed result either way. The returned dict
        always has ``video_url`` set on success.

        Note that ``duration`` is snapped to a length the provider actually
        renders (Veo accepts only 4/6/8s, Kling 5/10s), and the charge follows
        the snapped value — read ``duration`` on the result, not your request.

        Args:
            prompt: Text description of the desired video.
            model: Video model (default: veo-2).
            duration: Desired duration in seconds.
            aspect_ratio: Aspect ratio (e.g. "16:9", "9:16", "1:1").
            image_url: Reference image for image-to-video generation.
            resolution: Output resolution ("720p", "1080p", "4k").
            poll_interval: Seconds between polls, for the models that queue.
            timeout: How long to keep polling before giving up. The job itself
                is unaffected and may still finish.

        Returns:
            Dict with model, video_url, job_id, status, duration, ``cost_usd``
            and ``currency``. There is no ``credits_used``: the API is prepaid in
            USD.

        Raises:
            FotoHubError: If the generation failed. A failed video is refunded to
                the wallet automatically, so a raise here does not mean you paid
                for nothing delivered.
            TimeoutError: If the job was still processing when ``timeout``
                elapsed.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        if image_url is not None:
            payload["image_url"] = image_url

        response = await self._request("POST", "/v1/ai/generate/video", json_data=payload)
        result = response.json()

        job_id = result.get("job_id")
        # Only the queueing models need polling. A finished response already
        # carries the URL, and one without a job_id cannot be polled at all.
        if result.get("video_url") or result.get("status") != "processing" or not job_id:
            return result

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            status_resp = await self._request(
                "GET", f"/v1/ai/generate/video/{job_id}"
            )
            status = status_resp.json()
            state = status.get("status", "")
            if state == "completed":
                # The poll route reports the charge from the job row's
                # `estimated_cost`, which is null on a row written before that
                # column held USD. Fall back to the submit response, which always
                # carries it. Was `credits_used` on both sides -- a field the
                # prepaid API stopped returning, so this copied None onto None.
                if status.get("cost_usd") is None:
                    status["cost_usd"] = result.get("cost_usd")
                    status.setdefault("currency", "USD")
                if status.get("billing") is None and result.get("billing"):
                    status["billing"] = result["billing"]
                return status
            if state in ("failed", "cancelled"):
                raise FotoHubError(
                    message=status.get("error")
                    or status.get("error_message")
                    or f"Video job {job_id} {state}",
                    status_code=500,
                    response_body=status,
                )

        raise TimeoutError(
            message=f"Video job {job_id} did not complete within {timeout}s. "
                    f"It may still finish — poll GET /v1/ai/generate/video/{job_id}."
        )

    async def generate_seedance(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_SEEDANCE_MODEL,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        generate_audio: bool = False,
        image_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        reference_images: Optional[list[Any]] = None,
        reference_videos: Optional[list[Any]] = None,
        reference_audios: Optional[list[Any]] = None,
        asset_ids: Optional[list[str]] = None,
        output_format: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        callback_url: Optional[str] = None,
        smart_ratio: bool = False,
        smart_duration: bool = False,
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Generate a video with a Seedance model, waiting for the result.

        Async counterpart of :meth:`FotoHub.generate_seedance` — same parameters,
        same return shape. ``seedance-2-5`` is the only model that reaches 30
        seconds in a single request (4-30s, 480p/720p, audio included at
        $0.2335/s at 720p) and the only one that accepts a source video —
        attaching one raises the rate to $0.283421/s, since the source frames
        bill as input tokens.

        Returns:
            The finished job dict — ``video_url``, ``thumbnail_url``, ``status``,
            ``cost_usd``, ``currency``, ``duration``, ``resolution``,
            ``task_type``, ``billing``. There is no ``credits_used``.

        Raises:
            InsufficientFundsError: If the prepaid USD wallet cannot cover it.
                Nothing is charged.
            TimeoutError: If the job does not finish within ``timeout``.
            FotoHubError: If the render fails. A failed render is refunded to the
                wallet server-side.
        """
        payload = _seedance_payload(
            prompt=prompt, model=model, duration=duration, resolution=resolution,
            aspect_ratio=aspect_ratio, generate_audio=generate_audio,
            image_url=image_url, last_frame_url=last_frame_url,
            reference_images=reference_images, reference_videos=reference_videos,
            reference_audios=reference_audios, asset_ids=asset_ids,
            output_format=output_format, negative_prompt=negative_prompt,
            seed=seed, callback_url=callback_url, smart_ratio=smart_ratio,
            smart_duration=smart_duration,
        )

        submit_response = await self._request(
            "POST", "/v1/ai/generate/video", json_data=payload
        )
        submit = submit_response.json()
        job_id = submit.get("job_id")
        if not job_id:
            # Non-Seedance model — that path is synchronous and already finished.
            return submit

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status_response = await self._request(
                "GET", f"/v1/ai/generate/video/{job_id}"
            )
            status = status_response.json()
            state = status.get("status", "")
            if state == "completed":
                return status
            if state in ("failed", "cancelled"):
                raise FotoHubError(
                    message=status.get("error")
                    or status.get("error_message")
                    or f"Seedance job {job_id} {state}",
                    status_code=500,
                    response_body=status,
                )
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            message=f"Seedance job {job_id} did not complete within {timeout}s. "
                    f"It may still finish — poll GET /v1/ai/generate/video/{job_id}."
        )

    async def register_video_asset(
        self, image_url: str, *, retention_hours: Optional[int] = None
    ) -> dict[str, Any]:
        """Register a hosted portrait as a reusable Seedance asset.

        Free — no credits are charged. Pass the returned ``uri`` (or bare id) in
        ``asset_ids`` on :meth:`generate_seedance` so the same face appears
        across generations.

        A registered face is biometric data. Pass ``retention_hours`` to have it
        self-delete, at the provider and in our records, once that period
        elapses — use it to honour a data-minimisation policy instead of relying
        on remembering to call :meth:`delete_video_asset` yourself.

        Args:
            image_url: HTTPS URL on a FOTOhub host. Upload the file first;
                third-party URLs are refused.
            retention_hours: Optional, 1-8760 (1 year). Omit to keep the face
                until you delete it.

        Returns:
            Dict with ``asset_id``, ``uri``, ``status``, ``retention_hours``,
            ``expires_at``.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if retention_hours is not None:
            payload["retention_hours"] = retention_hours
        response = await self._request(
            "POST", "/v1/ai/assets/register", json_data=payload
        )
        return response.json()

    async def list_video_assets(
        self, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List the virtual portrait assets registered by this account.

        Returns:
            Dict with ``assets`` (list) and ``count``.
        """
        response = await self._request(
            "GET", "/v1/ai/assets",
            params={"limit": limit, "offset": offset},
        )
        return response.json()

    async def get_video_asset(self, asset_id: str) -> dict[str, Any]:
        """Get the current provider status of one registered face.

        Raises:
            FotoHubError: 404 if the asset does not belong to this account.
        """
        response = await self._request("GET", f"/v1/ai/assets/{asset_id}")
        return response.json()

    async def delete_video_asset(self, asset_id: str) -> dict[str, Any]:
        """Delete a registered face, at the provider and here.

        Idempotent: calling this on an already-erased asset returns
        ``{"deleted": True, "already_deleted": True}`` rather than raising.

        Raises:
            FotoHubError: 404 if the asset does not belong to this account;
                502 if the provider delete failed (safe to retry).
        """
        response = await self._request("DELETE", f"/v1/ai/assets/{asset_id}")
        return response.json()

    async def generate_music(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MUSIC_MODEL,
        duration: int = 30,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tempo: int = 120,
        instrumental: bool = True,
    ) -> dict[str, Any]:
        """Generate music from a text description.

        Args:
            prompt: Description of the desired music.
            model: Music generation model (default: minimax).
            duration: Duration in seconds (5-300).
            genre: Genre hint (e.g. "electronic", "classical", "jazz").
            mood: Mood hint (e.g. "happy", "melancholic", "energetic").
            tempo: BPM (40-240, default: 120).
            instrumental: Whether to generate instrumental-only (default: True).

        Returns:
            Dict with ``audio_url``, ``duration``, ``cost_usd``, ``currency`` and
            a ``billing`` block. There is no ``credits_used``.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "tempo": tempo,
            "instrumental": instrumental,
        }
        if genre is not None:
            payload["genre"] = genre
        if mood is not None:
            payload["mood"] = mood

        response = await self._request("POST", "/v1/ai/generate/music", json_data=payload)
        return response.json()

    async def generate_sfx(
        self,
        prompt: str,
        *,
        duration: int = 5,
    ) -> dict[str, Any]:
        """Generate a short sound effect.

        Args:
            prompt: Description of the sound effect.
            duration: Duration in seconds (1-30, default: 5).

        Returns:
            Dict with audio URL and metadata.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
        }

        response = await self._request("POST", "/v1/ai/generate/sfx", json_data=payload)
        return response.json()

    async def generate_speech(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        model: str = DEFAULT_SPEECH_MODEL,
        language: str = "pl",
        speed: float = 1.0,
        pitch: int = 0,
    ) -> dict[str, Any]:
        """Generate speech audio from text (TTS).

        Args:
            text: Text to convert to speech.
            voice_id: Voice identifier (provider-specific).
            model: TTS model/provider (default: "google").
            language: Language code (default: "pl").
            speed: Speech speed multiplier (0.5-2.0, default: 1.0).
            pitch: Pitch adjustment in semitones (-10 to 10, default: 0).

        Returns:
            Dict with ``audio_url``, ``characters_processed``, ``cost_usd``,
            ``currency`` and a ``billing`` block. There is no ``credits_used``.
        """
        payload: dict[str, Any] = {
            "text": text,
            "model": model,
            "language": language,
            "speed": speed,
            "pitch": pitch,
        }
        if voice_id is not None:
            payload["voice_id"] = voice_id

        response = await self._request("POST", "/v1/ai/generate/speech", json_data=payload)
        return response.json()

    async def transcribe(
        self,
        audio_url: str,
        *,
        language: str = "auto",
    ) -> dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio_url: URL of the audio file.
            language: Language code or "auto" for auto-detection.

        Returns:
            Dict with transcribed text, detected language, segments.
        """
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "language": language,
        }

        response = await self._request("POST", "/v1/ai/transcribe", json_data=payload)
        return response.json()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Union[dict[str, Any], AsyncChatStream]:
        """Send a chat completion request (OpenAI-compatible).

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: LLM model (default: gemini-flash). Only ``gemini-flash``,
                ``gemini-pro``, ``gpt-4o`` and ``claude-sonnet`` are accepted;
                anything else is rejected with 400 rather than silently
                substituted.
            temperature: Sampling temperature (0-2, default: 0.7).
            max_tokens: Maximum tokens in the response.
            stream: Not supported -- see Raises.

        Returns:
            Dict with choices, usage, ``cost_usd``, ``currency`` and ``billing``.
            Billed on real token counts at the provider's own per-direction rate,
            so ``billing["cost_usd"]`` scales with the length of the answer --
            fractions of a cent for a short reply. ``billing["legs"]`` splits it
            into input and output. ``billing["basis"]`` is ``"tokens"`` when the
            charge came from the model's own usage figures, or
            ``"flat_fallback"`` when the provider omitted them and one 1K output
            block was charged instead. There is no ``credits_used``.

        Raises:
            ValueError: If ``stream=True``. /v1/ai/chat/completions accepts the
                flag for OpenAI compatibility and then ignores it, returning one
                complete JSON body. AsyncChatStream finds no SSE frames in that
                body, so it yields zero chunks and raises nothing -- an empty
                result for a request that was still billed. Failing before the
                call keeps it free.
        """
        if stream:
            raise ValueError(
                "chat(stream=True) is not supported: /v1/ai/chat/completions never "
                "streams, so the iterator would yield nothing while the request is "
                "still billed. Use POST /v1/ai/agent/stream for token-by-token "
                "output -- see https://docs.fotohub.app/guides/streaming"
            )

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = await self._request("POST", "/v1/ai/chat/completions", json_data=payload)
        return response.json()

    async def chat_claude(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a chat request to a premium Claude (Anthropic) model.

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: Claude model ID (default: claude-sonnet-4.6).
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in the response.
            system: System prompt (prepended to conversation).

        Returns:
            Dict with response content, usage, stop_reason.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system is not None:
            payload["system"] = system

        response = await self._request("POST", "/v1/ai/chat/claude", json_data=payload)
        return response.json()

    async def chat_bedrock(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Deprecated alias for :meth:`chat_claude`.

        .. deprecated:: 1.4.0
            Use :meth:`chat_claude` instead. This method will be removed in a
            future release.
        """
        warnings.warn(
            "chat_bedrock() is deprecated; use chat_claude() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.chat_claude(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

    async def analyze_image(
        self,
        image_url: str,
        *,
        features: Optional[list[str]] = None,
        language: Optional[str] = None,
        max_labels: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> dict[str, Any]:
        """Analyze an image using vision models.

        Costs a flat 1 credit however many features are requested, so asking for
        everything in one call is cheaper than one call per feature.

        Args:
            image_url: URL of the image to analyze. Must be publicly reachable,
                max 20MB.
            features: Analysis features to run. Defaults to
                ``["labels", "objects"]``. Valid: ``labels``, ``objects``,
                ``faces``, ``nsfw``, ``colors``, ``ocr``, ``landmarks``,
                ``logos`` (plus the aliases ``text`` for ocr and
                ``safe_search`` for nsfw). An unrecognised name is rejected
                with 400 rather than ignored.
            language: Language hint for OCR. Label names are always English.
            max_labels: Cap on returned labels, 1-50 (default 50).
            min_confidence: Minimum confidence 0-1, applied to labels, objects
                and faces. ``nsfw`` and ``ocr`` carry no numeric score.

        Returns:
            Dict with the requested features at the top level (``labels``,
            ``objects``, ``faces``, ``nsfw``, ``ocr``, ``colors``, ...) plus
            ``auto_tags`` -- the flat deduplicated tag list. There is no
            ``analysis`` wrapper key.

            Faces and content safety return likelihood buckets
            (``VERY_UNLIKELY`` .. ``VERY_LIKELY``), not floats, and no age or
            gender. Bounding boxes carry ``units``: ``"pixels"`` for faces and
            OCR, ``"normalized"`` (0-1 fractions) for objects.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if features is not None:
            payload["features"] = features
        if language is not None:
            payload["language"] = language
        if max_labels is not None:
            payload["max_labels"] = max_labels
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence

        response = await self._request("POST", "/v1/ai/analyze/image", json_data=payload)
        return response.json()

    async def enhance_prompt(
        self,
        prompt: str,
        *,
        style: str = "photographic",
    ) -> str:
        """Enhance a short prompt into a detailed generation prompt.

        Args:
            prompt: Short input prompt.
            style: Target style ("photographic", "cinematic", "illustration",
                "3d", "anime").

        Returns:
            Enhanced prompt string.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "style": style,
        }

        response = await self._request("POST", "/v1/ai/enhance-prompt", json_data=payload)
        data = response.json()
        return data.get("enhanced_prompt", data.get("prompt", prompt))

    # =========================================================================
    # Stability AI Tools
    # =========================================================================

    async def stability_tools(self) -> list[dict[str, Any]]:
        """List available Stability AI tools, priced per image in USD.

        Returns:
            Tool descriptors with ``id``, ``model_id``, ``price_usd``, ``currency``,
            ``unit`` and the ``requires_*`` flags. ``credits`` is deprecated.
        """
        response = await self._request("GET", "/stability/tools")
        data = response.json()
        return data.get("tools", data) if isinstance(data, dict) else data

    async def stability_run(
        self,
        tool_id: str,
        image_base64: str,
        *,
        mask: Optional[str] = None,
        prompt: Optional[str] = None,
        reference: Optional[str] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Stability AI tool on an image.

        Args:
            tool_id: The tool identifier (e.g. "remove-background", "upscale").
            image_base64: Base64-encoded input image.
            mask: Base64-encoded mask image (for inpaint/erase).
            prompt: Text prompt (for generation-based tools).
            reference: Base64-encoded reference image (for style transfer).
            seed: Random seed for reproducibility.
            negative_prompt: Things to avoid.
            output_format: Output format ("png", "webp", "jpeg").
            **kwargs: Additional tool-specific parameters.

        Returns:
            Dict with ``image`` (base64, never a URL), ``tool``, ``seed`` (None on
            the tools that do not sample), and the usual ``cost_usd`` / ``currency``
            / ``billing`` charge block. There is no ``credits_used``: these tools are
            paid in USD from the prepaid wallet, from $0.03 to $0.60 per image
            depending on the tool -- see :meth:`stability_tools`.
        """
        payload: dict[str, Any] = {
            "image": image_base64,
            "output_format": output_format,
        }
        if mask is not None:
            payload["mask"] = mask
        if prompt is not None:
            payload["prompt"] = prompt
        if reference is not None:
            payload["reference"] = reference
        if seed is not None:
            payload["seed"] = seed
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        payload.update(kwargs)

        response = await self._request("POST", f"/stability/{tool_id}", json_data=payload)
        return response.json()

    async def stability_upscale(
        self,
        image_base64: str,
        *,
        type: str = "fast",
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upscale an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.
            type: Upscale type — "fast" (4x, instant), "creative" (4x, slower,
                prompt-guided) or "conservative" (4x, faithful).
            prompt: Optional guiding prompt (used by creative/conservative).

        Returns:
            Dict with upscaled image data.
        """
        tool_id = f"{type}-upscale"
        return await self.stability_run(tool_id, image_base64, prompt=prompt)

    async def stability_remove_background(
        self,
        image_base64: str,
    ) -> dict[str, Any]:
        """Remove background from an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.

        Returns:
            Dict with transparent-background image data.
        """
        return await self.stability_run("remove-background", image_base64)

    async def stability_erase(
        self,
        image_base64: str,
        mask_base64: str,
    ) -> dict[str, Any]:
        """Erase regions from an image (content-aware fill).

        Args:
            image_base64: Base64-encoded input image.
            mask_base64: Base64-encoded mask (white = erase).

        Returns:
            Dict with result image data.
        """
        return await self.stability_run("erase-object", image_base64, mask=mask_base64)

    async def stability_inpaint(
        self,
        image_base64: str,
        mask_base64: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Inpaint masked regions of an image with a prompt.

        Args:
            image_base64: Base64-encoded input image.
            mask_base64: Base64-encoded mask (white = inpaint).
            prompt: What to paint into the masked region.

        Returns:
            Dict with inpainted image data.
        """
        return await self.stability_run(
            "inpaint", image_base64, mask=mask_base64, prompt=prompt
        )

    async def stability_outpaint(
        self,
        image_base64: str,
        *,
        left: int = 0,
        right: int = 0,
        up: int = 0,
        down: int = 0,
    ) -> dict[str, Any]:
        """Extend an image beyond its borders (outpainting).

        Args:
            image_base64: Base64-encoded input image.
            left: Pixels to extend left.
            right: Pixels to extend right.
            up: Pixels to extend up.
            down: Pixels to extend down.

        Returns:
            Dict with extended image data.
        """
        return await self.stability_run(
            "outpaint", image_base64, left=left, right=right, up=up, down=down
        )

    async def stability_search_replace(
        self,
        image_base64: str,
        search_prompt: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Search for an object in an image and replace it.

        Args:
            image_base64: Base64-encoded input image.
            search_prompt: Description of what to find and replace.
            prompt: Description of the replacement.

        Returns:
            Dict with result image data.
        """
        return await self.stability_run(
            "search-replace", image_base64, prompt=prompt, search_prompt=search_prompt
        )

    async def stability_recolor(
        self,
        image_base64: str,
        search_prompt: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Recolor a specific object in an image.

        Args:
            image_base64: Base64-encoded input image.
            search_prompt: Description of the object to recolor.
            prompt: Description of the new color/appearance.

        Returns:
            Dict with recolored image data.
        """
        return await self.stability_run(
            "search-recolor", image_base64, prompt=prompt, search_prompt=search_prompt
        )

    async def stability_style_transfer(
        self,
        image_base64: str,
        reference_base64: str,
    ) -> dict[str, Any]:
        """Transfer the style of a reference image onto a source image.

        Args:
            image_base64: Base64-encoded source image.
            reference_base64: Base64-encoded style reference image.

        Returns:
            Dict with style-transferred image data.
        """
        return await self.stability_run(
            "style-transfer", image_base64, reference=reference_base64
        )

    # =========================================================================
    # Billing
    # =========================================================================

    async def get_balance(self) -> dict[str, Any]:
        """Get the wallet balance and this month's spend, in USD.

        Returns:
            Dict with ``wallet`` (``balance_usd``, ``pending_usd``,
            ``total_topped_up_usd``, ``currency``), ``spend``
            (``this_month_usd``, ``monthly_limit_usd``, ``remaining_usd``),
            ``billing_model`` and ``api_subscription``. There is no ``credits``
            block: it used to report the web app's subscription counter, telling
            API developers they had hundreds of credits while their spendable
            balance was $0. ``overage`` is the old name for ``spend`` and is
            deprecated -- a prepaid wallet has nothing to exceed.
        """
        response = await self._request("GET", "/v1/billing/balance")
        return response.json()

    async def get_pricing(self) -> dict[str, Any]:
        """Get pricing for all AI operations.

        Returns:
            Dict with per-model credit costs by category.
        """
        response = await self._request("GET", "/v1/billing/pricing")
        return response.json()

    async def get_plans(self) -> dict[str, Any]:
        """Deprecated. There are no paid API plans; the list is always empty.

        .. deprecated:: 1.11.0
            See :meth:`FotoHub.get_plans`. The endpoint answers
            ``{"plans": []}``; fund the wallet to raise limits.

        Returns:
            ``{"plans": []}``.
        """
        response = await self._request("GET", "/v1/billing/plans")
        return response.json()

    async def get_credits(self) -> dict[str, Any]:
        """Deprecated. The API has no credits; this returns the wallet.

        See :meth:`FotoHub.get_credits`. Answers 200 with ``deprecated: True``,
        ``billing_model``, ``wallet`` and ``spend``.
        """
        response = await self._request("GET", "/v1/billing/credits")
        return response.json()

    async def set_overage_limit(
        self,
        hard_limit_usd: float,
        *,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set the hard spending limit for overage charges.

        Args:
            hard_limit_usd: Maximum monthly overage amount in USD. Pass 0 to
                disable (the wallet balance then becomes the only cap).
            project_id: Optional project ID (defaults to account-level).

        Returns:
            Dict confirming the updated limit (``hard_limit_usd``).
        """
        payload: dict[str, Any] = {"hard_limit_usd": hard_limit_usd}
        if project_id is not None:
            payload["project_id"] = project_id

        response = await self._request(
            "PUT", "/v1/billing/overage-limit", json_data=payload
        )
        return response.json()

    async def get_topup_packages(self) -> list[dict[str, Any]]:
        """Get available wallet top-up packages.

        A package credits ``total_usd`` -- the amount paid plus any volume bonus.
        From $500 up, the ladder adds extra spendable dollars: 5% at $500 rising
        to 20% at $15 000, so $1 000 paid credits $1 100.

        Returns:
            List of packages with ``slug``, ``name`` (the amount PAID),
            ``amount_usd``, ``bonus_usd``, ``total_usd`` and ``bonus_pct``.
            ``bonus_credits`` is gone -- it described a credit transfer nothing
            performed, and this product has no credit unit.

        Note:
            Use :meth:`get_topup_package_list` when quoting a custom amount.
        """
        response = await self._request("GET", "/v1/billing/topup/packages")
        data = response.json()
        return data.get("packages", data) if isinstance(data, dict) else data

    async def get_topup_package_list(self) -> dict[str, Any]:
        """Top-up packages plus the bonus ladder and the custom-amount bounds.

        Returns:
            Dict with ``packages``, ``bonus_tiers`` (highest ``min_usd`` first),
            ``min_usd``, ``max_usd`` and ``notes``. See the sync
            :meth:`FotoHub.get_topup_package_list` for a worked bonus quote.
        """
        response = await self._request("GET", "/v1/billing/topup/packages")
        return response.json()

    async def create_topup(self, package: str) -> dict[str, Any]:
        """Purchase a wallet top-up package.

        Args:
            package: Package slug. Starter rungs ``"topup-50"`` ($15),
                ``"topup-100"`` ($25), ``"topup-250"`` ($60), ``"topup-500"``
                ($120) -- historical names that do NOT match their amounts.
                Bonus-earning rungs ``"scale-500"`` through ``"scale-15000"``,
                where the number IS the amount in USD. Prefer
                :meth:`get_topup_packages` over a hardcoded slug.

        Returns:
            Dict with checkout_url and the purchased package descriptor.
        """
        payload: dict[str, Any] = {"package": package}

        response = await self._request("POST", "/v1/billing/topup", json_data=payload)
        return response.json()

    async def get_transactions(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        type_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get credit transaction history.

        Args:
            page: Page number (starting at 1).
            page_size: Items per page (max 100).
            type_filter: Filter by type ("charge", "topup", "refund", "bonus").

        Returns:
            Dict with transactions list and pagination metadata.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if type_filter is not None:
            params["type"] = type_filter

        response = await self._request("GET", "/v1/billing/transactions", params=params)
        return response.json()

    async def estimate_cost(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Estimate the cost of a set of operations before running them.

        Args:
            operations: List of operation dicts, each with "type", "model",
                and relevant parameters (width, height, duration, etc.).

        Returns:
            Dict with ``total_usd``, ``provider_cost_usd``, ``margin``,
            ``currency``, ``balance_usd``, ``sufficient``, ``priced`` and a
            ``breakdown`` per operation. Because the account is prepaid, read
            ``sufficient`` -- the server's own answer to "can my wallet cover
            this" -- rather than comparing two numbers yourself. An operation with
            no published rate comes back ``priced: false`` with
            ``amount_usd: null``, and then ``total_usd`` covers only the priced
            legs. ``total_credits`` is deprecated and always ``None``.
        """
        payload: dict[str, Any] = {"operations": operations}

        response = await self._request("POST", "/v1/billing/estimate", json_data=payload)
        return response.json()

    async def get_invoices(self) -> dict[str, Any]:
        """Get billing invoices.

        Returns:
            Dict with list of invoices and their status.
        """
        response = await self._request("GET", "/v1/billing/invoices")
        return response.json()

    # =========================================================================
    # 3D Generation
    # =========================================================================

    async def generate_3d(
        self,
        mode: str,
        model: str,
        *,
        image: Optional[str] = None,
        prompt: Optional[str] = None,
        quality: str = "standard",
        format: str = "glb",
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate a 3D model from an image or text prompt.

        Synchronous despite being awaited: the coroutine resolves with the
        finished model, not a job handle. Charged in USD from the prepaid wallet.
        See the sync `generate_3d` for the full contract.
        """
        payload: dict[str, Any] = {
            "mode": mode,
            "model": model,
            "quality": quality,
            "format": format,
        }
        if image is not None:
            payload["image_base64"] = image
        if prompt is not None:
            payload["prompt"] = prompt
        if options is not None:
            payload["options"] = options

        response = await self._request("POST", "/v1/ai/generate/3d", json_data=payload)
        return response.json()

    async def get_3d_status(self, job_id: str) -> dict[str, Any]:
        """Fetch a stored 3D asset with a freshly signed URL. Free.

        Not a status check -- see the sync `get_3d_status`.
        """
        response = await self._request("GET", f"/v1/ai/generate/3d/{job_id}")
        return response.json()

    async def wait_for_3d(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Deprecated -- generation is synchronous, so this returns immediately.

        See the sync `wait_for_3d`.
        """
        import asyncio

        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    message=f"3D generation job {job_id} timed out after {timeout}s"
                )

            result = await self.get_3d_status(job_id)
            status = result.get("status", "")

            if status == "completed":
                return result
            if status == "failed":
                raise FotoHubError(
                    message=f"3D generation job {job_id} failed",
                    status_code=500,
                    response_body=result,
                )

            await asyncio.sleep(poll_interval)

    async def list_3d_models(self) -> list[dict[str, Any]]:
        """List available 3D generation models, priced in USD (`price_usd`)."""
        response = await self._request("GET", "/v1/ai/generate/3d/models")
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    async def list_models(self, category: Optional[str] = None) -> list[dict[str, Any]]:
        """List the model catalog with pricing.

        Read `price_unit` -- not `pricing_type` -- to know what `request_price`
        buys. `pricing_type` says "request" on every video model, but their
        price is per second of output.

        Args:
            category: Narrow to one of image, video, audio, text.

        Returns:
            List of models with id, name, request_price, price_unit,
            request_price_per, currency and limits.
        """
        params = {"category": category} if category else None
        response = await self._request("GET", "/v1/models", params=params)
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    # =========================================================================
    # Virtual Try-On
    # =========================================================================

    async def tryon(
        self,
        person_image_url: str,
        *,
        garment_image_url: Optional[str] = None,
        garment_id: Optional[str] = None,
        category: str = "tops",
        garment_photo_type: Optional[str] = None,
        garments: Optional[list[dict[str, Any]]] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Dress a person photo in a garment. Pass `garments` (one top plus one
        bottom) for a chained outfit, billed as one render of two Vertex passes.

        Returns the same 202 shape as :meth:`FotoHub.tryon` — job_id, status,
        category, ``cost_usd``, ``currency``, ``billing``, estimated_seconds and
        poll_url. The wallet is charged at submit; the poll route never reports it.
        """
        payload: dict[str, Any] = {
            "person_image_url": person_image_url,
            "num_images": num_images,
        }
        if garments:
            payload["garments"] = garments
        else:
            if garment_image_url is not None:
                payload["garment_image_url"] = garment_image_url
            if garment_id is not None:
                payload["garment_id"] = garment_id
            payload["category"] = category
            if garment_photo_type is not None:
                payload["garment_photo_type"] = garment_photo_type
        if seed is not None:
            payload["seed"] = seed

        response = await self._request("POST", "/v1/ai/tryon", json_data=payload)
        return response.json()

    async def get_tryon_status(self, job_id: str) -> dict[str, Any]:
        """Check the status of a try-on job."""
        response = await self._request("GET", f"/v1/ai/tryon/{job_id}")
        return response.json()

    async def wait_for_tryon(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Wait for a try-on job to complete. A partially failed outfit counts as
        success — inspect result["metadata"]["partial_failure"]."""
        import asyncio

        start = time.time()
        while True:
            if time.time() - start >= timeout:
                raise TimeoutError(
                    message=f"Try-on job {job_id} timed out after {timeout}s"
                )

            result = await self.get_tryon_status(job_id)
            status = result.get("status", "")

            if status == "completed":
                return result
            if status in ("failed", "cancelled"):
                raise FotoHubError(
                    message=result.get("error_message") or f"Try-on job {job_id} {status}",
                    status_code=500,
                    response_body=result,
                )

            await asyncio.sleep(poll_interval)

    # =========================================================================
    # Tier Management
    # =========================================================================

    async def get_tier_catalog(self) -> dict[str, Any]:
        """Get the full tier catalog. See :meth:`FotoHub.get_tier_catalog`.

        Nothing here is purchasable and nothing carries a price; ``-1`` in
        ``limits`` means "no cap", not a negative allowance.
        """
        response = await self._request("GET", "/v1/tiers/catalog")
        return response.json()

    async def get_current_tier(self) -> dict[str, Any]:
        """Get the current user's tier, limits, and usage."""
        response = await self._request("GET", "/v1/tiers/current")
        return response.json()

    async def compare_tiers(self) -> dict[str, Any]:
        """Compare all tiers side-by-side. See :meth:`FotoHub.compare_tiers`.

        No row carries a price: every one is ``purchasable: False`` with an
        ``upgrade_path``. Compare on ``rpm`` / ``concurrent_jobs``.
        """
        response = await self._request("GET", "/v1/tiers/compare")
        return response.json()

    async def subscribe_tier(self, tier_slug: str) -> dict[str, Any]:
        """Retired on 2026-08-13 -- always raises.

        .. deprecated:: 1.5.0
            See :meth:`FotoHub.subscribe_tier`. Rate limits follow the prepaid
            wallet now; use :meth:`topup_wallet` or :meth:`create_topup`.

        Raises:
            FotoHubError: Always, with ``status_code=410``.
        """
        raise FotoHubError(
            "API subscription plans were retired on 2026-08-13. Rate limits now "
            "follow your prepaid wallet balance, so top up instead: "
            "await client.topup_wallet(amount_usd) or "
            "await client.create_topup(package). Top-ups from $500 up earn a "
            "5-20% volume bonus in extra spendable dollars. For sub-enterprise, "
            "use await client.apply_enterprise(...).",
            status_code=410,
            response_body={"error": "api_subscriptions_retired"},
        )

    async def get_wallet(self) -> dict[str, Any]:
        """Get the current wallet balance."""
        response = await self._request("GET", "/v1/tiers/wallet")
        return response.json()

    async def topup_wallet(
        self,
        amount_usd: float,
        *,
        pay_currency: Optional[str] = None,
    ) -> dict[str, Any]:
        """Top up wallet balance (returns a Stripe checkout URL).

        ``amount_usd`` is in USD (minimum 10, maximum 15000, whole cents only).
        Pass ``pay_currency="pln"`` to charge in PLN via BLIK/card/bank while
        still crediting the wallet ``amount_usd``.

        From $500 up the amount earns a 5-20% volume bonus in extra spendable
        dollars; ``total_credited_usd`` in the response is the balance increase.
        See the sync :meth:`FotoHub.topup_wallet` for the full response shape.
        """
        payload: dict[str, Any] = {"amount_usd": amount_usd}
        if pay_currency is not None:
            payload["pay_currency"] = pay_currency
        response = await self._request("POST", "/v1/tiers/wallet/topup", json_data=payload)
        return response.json()

    async def apply_enterprise(
        self,
        company_name: str,
        contact_email: str,
        expected_usage: str,
        use_case: str,
        *,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit an enterprise tier application."""
        payload: dict[str, Any] = {
            "company_name": company_name,
            "contact_email": contact_email,
            "expected_usage": expected_usage,
            "use_case": use_case,
        }
        if notes is not None:
            payload["notes"] = notes

        response = await self._request("POST", "/v1/tiers/enterprise/apply", json_data=payload)
        return response.json()

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def list_webhooks(self) -> list[dict[str, Any]]:
        """List all webhook endpoints.

        Returns:
            List of webhook configurations.
        """
        response = await self._request("GET", "/v1/console/webhooks")
        data = response.json()
        return data.get("webhooks", data) if isinstance(data, dict) else data

    async def create_webhook(
        self,
        name: str,
        url: str,
        events: list[str],
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create a new webhook endpoint.

        Args:
            name: Human-readable name for the webhook.
            url: The URL to receive webhook events.
            events: List of event types (e.g. ["generation.completed",
                "generation.failed", "credits.low"]).
            headers: Custom headers to include in webhook requests.

        Returns:
            Dict with webhook ID, secret, and configuration.
        """
        payload: dict[str, Any] = {
            "name": name,
            "url": url,
            "events": events,
        }
        if headers is not None:
            payload["headers"] = headers

        response = await self._request("POST", "/v1/console/webhooks", json_data=payload)
        return response.json()

    async def update_webhook(self, webhook_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update an existing webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
            **kwargs: Fields to update (url, events, name, headers, active).

        Returns:
            Dict with updated webhook configuration.
        """
        response = await self._request(
            "PATCH", f"/v1/console/webhooks/{webhook_id}", json_data=kwargs
        )
        return response.json()

    async def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
        """
        await self._request("DELETE", f"/v1/console/webhooks/{webhook_id}")

    async def test_webhook(self, webhook_id: str) -> dict[str, Any]:
        """Send a test event to a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            Dict with delivery status and response code.
        """
        response = await self._request("POST", f"/v1/console/webhooks/{webhook_id}/test")
        return response.json()

    async def get_webhook_logs(self, webhook_id: str) -> list[dict[str, Any]]:
        """Get delivery logs for a webhook.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            List of delivery log entries with status, timestamp, response.
        """
        response = await self._request("GET", f"/v1/console/webhooks/{webhook_id}/logs")
        data = response.json()
        return data.get("logs", data) if isinstance(data, dict) else data

    # =========================================================================
    # Convenience Helpers
    # =========================================================================

    # =========================================================================
    # Gabriel AI Orchestrator
    # =========================================================================

    async def gabriel_classify(
        self,
        prompt: str,
        *,
        language: str = "en",
        context: Optional[dict[str, Any]] = None,
        enhance_prompt: bool = False,
    ) -> dict[str, Any]:
        """Classify user intent and route to the optimal platform feature."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "language": language,
        }
        if context is not None:
            payload["context"] = context
        if enhance_prompt:
            payload["enhance_prompt"] = True

        response = await self._request("POST", "/v1/ai/gabriel", json_data=payload)
        return response.json()

    async def gabriel_suggest(
        self,
        partial: str,
        *,
        tab: str = "all",
        page: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get lightweight autocomplete suggestions (no auth required)."""
        payload: dict[str, Any] = {
            "partial": partial,
            "tab": tab,
        }
        if page is not None:
            payload["page"] = page

        response = await self._request("POST", "/v1/ai/gabriel/suggest", json_data=payload)
        data = response.json()
        return data.get("suggestions", [])

    async def gabriel_recommend(
        self,
        *,
        page: Optional[str] = None,
        credits_remaining: Optional[int] = None,
        has_brand: Optional[bool] = None,
        recent_actions: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Get proactive context-aware recommendations (no auth required)."""
        payload: dict[str, Any] = {}
        if page is not None:
            payload["page"] = page
        if credits_remaining is not None:
            payload["credits_remaining"] = credits_remaining
        if has_brand is not None:
            payload["has_brand"] = has_brand
        if recent_actions is not None:
            payload["recent_actions"] = recent_actions

        response = await self._request("POST", "/v1/ai/gabriel/recommend", json_data=payload)
        data = response.json()
        return data.get("recommendations", [])

    async def translate(
        self,
        text: str,
        target_language: str,
        *,
        source_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Translate text between languages."""
        payload: dict[str, Any] = {
            "text": text,
            "target_language": target_language,
        }
        if source_language is not None:
            payload["source_language"] = source_language

        response = await self._request("POST", "/v1/ai/translate", json_data=payload)
        return response.json()

    # =========================================================================
    # Convenience Helpers
    # =========================================================================

    async def remove_background(self, image_url: str) -> dict[str, Any]:
        """Remove the background from an image (convenience wrapper).

        Args:
            image_url: URL of the source image.

        Returns:
            Dict with processed image URL and metadata.
        """
        return await self.edit_image(image_url, "remove background", mode="remove_bg")

    async def upscale_image(self, image_url: str, *, scale: int = 2) -> dict[str, Any]:
        """Upscale an image to higher resolution (convenience wrapper).

        Args:
            image_url: URL of the image to upscale.
            scale: Upscale factor (2 or 4, default: 2).

        Returns:
            Dict with upscaled image URL and metadata.
        """
        return await self.edit_image(
            image_url, f"upscale {scale}x", mode="upscale", scale=scale
        )

    async def wait_for_video(
        self,
        result: Union[str, dict[str, Any]],
        *,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Return a finished video result.

        .. deprecated:: 1.4.0
            :meth:`generate_video` already waits for the finished ``video_url``,
            polling on your behalf for the models that queue (Wan, Grok). This
            method just returns its result dict unchanged, and will be removed in
            a future release.

        Args:
            result: The dict returned by :meth:`generate_video`.
            poll_interval: Unused (kept for backwards compatibility).
            timeout: Unused (kept for backwards compatibility).

        Returns:
            The finished video result dict.
        """
        warnings.warn(
            "wait_for_video() is deprecated; generate_video() already returns "
            "the finished video_url, polling when the model queues.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(result, dict):
            return result
        raise FotoHubError(
            "wait_for_video() no longer accepts a job_id: generate_video() "
            "polls for you. Pass the dict it returned (or just read its "
            "'video_url')."
        )

    # =========================================================================
    # Video timeline (headless editor API: /v1/video/projects)
    # =========================================================================

    async def create_video_project(
        self,
        *,
        title: Optional[str] = None,
        aspect: Optional[str] = None,
        fps: Optional[int] = None,
        media: Optional[list[dict[str, Any]]] = None,
        template: Optional[Union[str, dict[str, Any]]] = None,
        place_media: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a timeline project, optionally seeded with media. Free.

        The project uses the same document as the FOTOhub video editor, so the
        result opens in the browser at ``editorUrl``.

        Args:
            title: Project title.
            aspect: "16:9", "9:16", "1:1", "4:5" or "4:3".
            fps: Frames per second.
            media: Up to 50 items, each ``{"url": "https://..."}`` (public HTTPS;
                FOTOhub copies the file into your storage) or
                ``{"storage_path": "<bucket>/<userId>/..."}``, plus optional
                ``kind`` ("video", "audio", "image") and ``name``.
            template: A template id (or ``{"id": ...}``) to start from.
            place_media: "sequence" (default) lays the media on the timeline one
                after another; "none" only adds them to the project.
            idempotency_key: Override the automatic ``X-Idempotency-Key``. Retrying
                with the same key within 24 h returns the first project instead
                of creating a second one.

        Returns:
            :class:`~fotohub.models.VideoProject` dict: ``projectId``, ``saveRev``,
            ``digest``, ``media``, ``unplacedMedia`` (media that did not fit on the
            timeline) and ``editorUrl``.

        Raises:
            FotoHubError: ``code`` is ``media-blocked`` (URL not allowed),
                ``media-too-large`` or ``media-not-found``.
        """
        response = await self._request(
            "POST", "/v1/video/projects",
            json_data=_video_project_payload(
                title=title, aspect=aspect, fps=fps, media=media,
                template=template, place_media=place_media,
            ),
            idempotency_key=idempotency_key,
        )
        return response.json()

    async def list_video_projects(self, *, limit: int = 50) -> dict[str, Any]:
        """List your API-created video projects (newest first). Free.

        Returns:
            Dict with ``projects``: ``[{projectId, title, updatedAt, editorUrl}]``.
        """
        response = await self._request("GET", "/v1/video/projects", params={"limit": limit})
        return response.json()

    async def get_video_project(
        self, project_id: str, *, include_doc: bool = False
    ) -> dict[str, Any]:
        """Fetch a project: digest, media (with fresh URLs), versions and ``saveRev``. Free.

        Args:
            project_id: The ``projectId`` from :meth:`create_video_project`.
            include_doc: Also return the full editor document as ``doc``.

        Returns:
            :class:`~fotohub.models.VideoProject` dict. A project that is not
            yours is reported as not found (HTTP 404), never as forbidden.
        """
        response = await self._request(
            "GET", f"/v1/video/projects/{project_id}",
            params={"include": "doc"} if include_doc else None,
        )
        return response.json()

    async def delete_video_project(self, project_id: str) -> dict[str, Any]:
        """Delete an API-created project. Free.

        Args:
            project_id: The ``projectId`` to delete.
        """
        response = await self._request("DELETE", f"/v1/video/projects/{project_id}")
        return response.json()

    async def apply_video_ops(
        self,
        project_id: str,
        ops: list[dict[str, Any]],
        *,
        dry_run: bool = False,
        expected_save_rev: Optional[int] = None,
        label: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply up to 40 editing operations to a project as one atomic batch. Free.

        The operation shapes are listed by :meth:`get_video_ops_catalog`. If any
        operation is rejected the whole batch is rolled back: the project is
        unchanged and the result has ``rolledBack: True`` with ``violations``
        (HTTP 200, not an exception).

        Args:
            project_id: The project to edit.
            ops: Operation objects. Each one is discriminated by ``op``; the
                full list with schemas is :meth:`get_video_ops_catalog`. For
                example ``{"op": "insertClip", "ref": "intro", "at": {...},
                "clip": {...}}``. ``ref`` names a new clip so later operations in
                the batch, and your code (via ``refs`` in the result), can
                address it.
            dry_run: Validate and preview the effect without saving.
            expected_save_rev: The ``saveRev`` you last read. If the project
                changed since (the browser editor, another agent), nothing is
                written and :class:`~fotohub.SaveConflictError` is raised.
                Set it whenever you can: without it a timeout or a 5xx is
                **not** retried automatically (the batch may already have been
                saved, and repeating it would apply it twice), so you get the
                error and must re-read the project yourself.
            label: Name for the version snapshot saved with this change.

        Returns:
            :class:`~fotohub.models.ApplyOpsResult` dict: ``ok``, ``rolledBack``,
            ``violations``, per-operation ``results`` with ``summary``,
            ``accepted`` / ``rejected`` counts, ``refs`` (your ``ref`` names
            mapped to the ids of the clips they created), the new ``saveRev``,
            ``digestDelta``, ``versionSaved`` and ``warnings``.

        Raises:
            SaveConflictError: 409 ``save-conflict``; re-read the project
                (``current_save_rev``) and re-apply.
            ValidationError: 422 ``invalid-ops`` (schema path in ``details``).
        """
        body = _drop_none({
            "ops": ops,
            "dryRun": True if dry_run else None,
            "expectedSaveRev": expected_save_rev,
            "label": label,
        })
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/ops", json_data=body,
            retry_ambiguous=expected_save_rev is not None,
        )
        return response.json()

    async def digest_video_project(
        self,
        project_id: str,
        *,
        clip_ids: Optional[list[str]] = None,
        view: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the project digest, or detailed data for up to 10 clips. Free.

        Args:
            project_id: The project to read.
            clip_ids: Return details for these clips (max 10).
            view: "digest" or "clips".
        """
        body = _drop_none({"clipIds": clip_ids, "view": view})
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/digest", json_data=body
        )
        return response.json()

    async def lint_video_project(
        self,
        project_id: str,
        *,
        rules: Optional[list[str]] = None,
        severity: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, Any]:
        """Check a project for editing problems (gaps, clipping, overlaps, ...). Free.

        Args:
            project_id: The project to check.
            rules: Only run these rule ids.
            severity: Severities to report: "error", "warn" and/or "info" (one
                string or a list).

        Returns:
            :class:`~fotohub.models.LintResult` dict. While the checker is not
            deployed the call still succeeds, with ``available: False`` and
            ``warnings: ["lint-unavailable"]``. A 501 ``lint-unavailable`` error
            means the whole endpoint is missing.
        """
        body = _drop_none({
            "rules": rules,
            "severity": [severity] if isinstance(severity, str) else severity,
        })
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/lint", json_data=body
        )
        return response.json()

    async def capture_video_project(
        self,
        project_id: str,
        *,
        times: Optional[list[float]] = None,
        count: Optional[int] = None,
        cuts: bool = False,
        width: Optional[int] = 640,
        sheet: Optional[dict[str, int]] = None,
        wait: bool = False,
        max_wait: float = 300.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Render still frames of the timeline into contact sheets, to see the edit. Paid, flat fee.

        Give exactly one of ``times``, ``count`` or ``cuts=True``.

        Args:
            project_id: The project to capture.
            times: Timeline positions in seconds (up to 24).
            count: That many frames spread evenly over the timeline.
            cuts: One frame at every cut.
            width: Frame width in pixels (16-1280, default 640).
            sheet: Contact sheet layout ``{"max_cells": 1-12, "max_edge": 256-1568}``.
            wait: Poll until the job finishes and return it (see
                :meth:`wait_for_video_job`).
            max_wait: Seconds to wait when ``wait`` is true.
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            Without ``wait``: a queued :class:`~fotohub.models.VideoJob`
            (``jobId``, ``status``, ``times``, ``width``, ``height``). With
            ``wait``: the completed job, carrying a
            :class:`~fotohub.models.CaptureResult` (``frames``, ``sheets``,
            ``missing``).

        Raises:
            RateLimitError: 429 ``rate-limited`` with ``retry_after``.
            AuthError: 403 ``payment-required`` when the wallet is empty.
        """
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/capture",
            json_data=_video_capture_payload(
                times=times, count=count, cuts=cuts, width=width, sheet=sheet
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return await self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    async def render_video_project(
        self,
        project_id: str,
        *,
        format: str = "mp4",
        quality: str = "high",
        resolution: Optional[str] = None,
        codec: Optional[str] = None,
        fps: Optional[int] = None,
        bitrate: Optional[str] = None,
        time_range: Optional[tuple[float, float]] = None,
        content_credentials: Optional[bool] = None,
        content_ai_declared: Optional[bool] = None,
        wait: bool = False,
        max_wait: float = 1800.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Render the project to a video file. Paid per output minute.

        The charge is returned automatically if the render fails.

        Args:
            project_id: The project to render.
            format: "mp4", "webm", "mov", "gif", "mp3" or "wav".
            quality: "draft", "standard", "high" or "ultra".
            resolution: "720p", "1080p", "2k" or "4k".
            codec: "h264", "h265" or "prores".
            fps: Output frame rate (1-120).
            bitrate: e.g. "8M" or "800k".
            time_range: Render only ``(start, end)`` seconds of the timeline.
            content_credentials: Embed C2PA content credentials.
            content_ai_declared: Declare AI-generated content in them.
            wait: Poll until the render finishes and return the completed job.
            max_wait: Seconds to wait when ``wait`` is true (default 1800).
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            Without ``wait``: a queued :class:`~fotohub.models.VideoJob`
            (``jobId``, ``billedMinutes``). With ``wait``: the completed job with
            ``outputUrl``.

        Raises:
            VideoJobFailedError: With ``wait``, if the render fails (``refunded``
                tells whether the charge was returned).
            VideoJobTimeoutError: With ``wait``, if ``max_wait`` elapses; the job
                keeps running, poll it with :meth:`get_video_job`.
        """
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/render",
            json_data=_video_render_payload(
                format=format, quality=quality, resolution=resolution, codec=codec,
                fps=fps, bitrate=bitrate, time_range=time_range,
                content_credentials=content_credentials,
                content_ai_declared=content_ai_declared,
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return await self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    async def auto_edit_video_project(
        self,
        project_id: str,
        *,
        style: Optional[str] = None,
        toggles: Optional[dict[str, Any]] = None,
        language: Optional[str] = None,
        aspect: Optional[str] = None,
        ai_budget_usd: float = 0,
        auto_apply: bool = True,
        mode: str = "auto_edit",
        wait: bool = False,
        max_wait: float = 1800.0,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Let FOTOhub edit the project for you (server-side Auto-Edit). Paid.

        .. note:: Provisional. The server route ships with the Auto-Edit release
           and its body and result may still change; do not rely on it yet.

        Args:
            project_id: The project to edit.
            style: "viral", "podcast", "explainer", "storytelling" or "captions-only".
            toggles: Feature switches, as in the editor's Auto-Edit panel.
            language: Spoken language ("auto" to detect).
            aspect: Target aspect ratio.
            ai_budget_usd: Cap for AI-generated media, 0-50 (0 = stock only).
            auto_apply: Commit the result; if false it stays a draft.
            mode: "auto_edit" or "cut".
            wait: Poll until finished and return the completed job (with ``report``).
            max_wait: Seconds to wait when ``wait`` is true.
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            A queued :class:`~fotohub.models.VideoJob`, or the finished one with ``wait``.
        """
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/auto-edit",
            json_data=_video_auto_edit_payload(
                style=style, toggles=toggles, language=language, aspect=aspect,
                ai_budget_usd=ai_budget_usd, auto_apply=auto_apply, mode=mode,
            ),
            idempotency_key=idempotency_key,
        )
        job = response.json()
        return await self.wait_for_video_job(job["jobId"], timeout=max_wait) if wait else job

    async def apply_video_auto_edit(
        self,
        project_id: str,
        job_id: str,
        *,
        expected_save_rev: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Commit the draft of an Auto-Edit job started with ``auto_apply=False``. Free.

        .. note:: Provisional. The server route ships with the Auto-Edit release
           and its body and result may still change; do not rely on it yet.

        Args:
            project_id: The project the job edited.
            job_id: The ``jobId`` returned by :meth:`auto_edit_video_project`.
            expected_save_rev: The ``saveRev`` you last read; if the project
                changed since, nothing is written and
                :class:`~fotohub.SaveConflictError` is raised (the draft stays).
            idempotency_key: Override the automatic ``X-Idempotency-Key``.

        Returns:
            The apply result with the new ``saveRev``.
        """
        body = _drop_none({"expectedSaveRev": expected_save_rev})
        response = await self._request(
            "POST", f"/v1/video/projects/{project_id}/auto-edit/{job_id}/apply",
            json_data=body, idempotency_key=idempotency_key,
        )
        return response.json()

    async def get_video_job(self, job_id: str) -> dict[str, Any]:
        """Read the state of a render / capture / auto-edit job. Free.

        Returns:
            :class:`~fotohub.models.VideoJob` dict. ``status`` is "queued",
            "running", "completed", "failed" or "cancelled"; a failed job carries
            ``error`` and ``refunded``.
        """
        response = await self._request("GET", f"/v1/video/jobs/{job_id}")
        return response.json()

    async def wait_for_video_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Poll a render / capture / auto-edit job until it completes.

        Args:
            job_id: The ``jobId`` returned by the render, capture or auto-edit call.
            poll_interval: Seconds between checks (default 3.0).
            timeout: Maximum wait in seconds (default 1800.0).

        Returns:
            The completed :class:`~fotohub.models.VideoJob`.

        Raises:
            VideoJobFailedError: The job ended "failed" or "cancelled".
            VideoJobTimeoutError: ``timeout`` elapsed first (the job may still finish).
        """
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get_video_job(job_id)
            status = job.get("status")
            if status == "completed":
                return job
            if status in _VIDEO_JOB_FAILED:
                raise _video_job_failure(job)
            if time.monotonic() + poll_interval > deadline:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} not finished after {timeout}s (last status: {status})",
                    job_id=job_id,
                )
            await asyncio.sleep(poll_interval)

    async def get_video_ops_catalog(self) -> dict[str, Any]:
        """The JSON schema of every operation :meth:`apply_video_ops` accepts. Free."""
        response = await self._request("GET", "/v1/video/ops/catalog")
        return response.json()

    async def detect_video_scenes(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        threshold: float = 0.4,
        min_scene_duration: float = 0.5,
    ) -> dict[str, Any]:
        """Find scene cuts in a video. Paid per request.

        Give either ``url`` (public HTTPS) or ``project_id`` + ``media_id``
        (an ``assetId`` from the project's media).

        Args:
            url: Public HTTPS URL of the video.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            threshold: Cut sensitivity 0-1 (default 0.4).
            min_scene_duration: Shortest scene in seconds (default 0.5).
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "threshold": threshold,
            "minSceneDuration": min_scene_duration,
        }
        response = await self._request("POST", "/v1/video/detect-scenes", json_data=body)
        return response.json()

    async def detect_video_silence(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        noise_floor_db: float = -30.0,
        min_silence_duration: float = 0.3,
    ) -> dict[str, Any]:
        """Find silent ranges in audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            noise_floor_db: Level below which audio counts as silence (default -30).
            min_silence_duration: Shortest silence in seconds (default 0.3).
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "noiseFloorDb": noise_floor_db,
            "minSilenceDuration": min_silence_duration,
        }
        response = await self._request("POST", "/v1/video/detect-silence", json_data=body)
        return response.json()

    async def detect_video_beats(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Find beats and tempo in audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
        """
        body = _video_source_payload(url=url, project_id=project_id, media_id=media_id)
        response = await self._request("POST", "/v1/video/detect-beats", json_data=body)
        return response.json()

    async def transcribe_video(
        self,
        *,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        media_id: Optional[str] = None,
        language: str = "auto",
        hotwords: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Start a transcription job for audio or video. Paid per request.

        Args:
            url: Public HTTPS URL of the media.
            project_id: Project holding the media.
            media_id: ``assetId`` of the project media item.
            language: Language code, or "auto" (default).
            hotwords: Up to 50 words/names to favour.

        Returns:
            Dict with the transcription ``jobId``; read it with
            :meth:`get_video_transcription`.
        """
        body = {
            **_video_source_payload(url=url, project_id=project_id, media_id=media_id),
            "language": language,
            **_drop_none({"hotwords": hotwords}),
        }
        response = await self._request("POST", "/v1/video/transcribe", json_data=body)
        return response.json()

    async def get_video_transcription(self, job_id: str) -> dict[str, Any]:
        """Read a transcription job started by :meth:`transcribe_video`. Free.

        Returns:
            Dict with ``status`` ("queued", "processing", "completed", "failed"),
            ``progress`` and, when completed, ``result``.
        """
        response = await self._request("GET", f"/v1/video/transcribe/{job_id}")
        return response.json()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncFotoHub":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
