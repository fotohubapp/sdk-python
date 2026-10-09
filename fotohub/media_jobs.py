"""Upscale Pro and AI video: two job-based namespaces on both clients.

::

    client = FotoHub(api_key="...")
    q = client.upscale_pro.quote("video", width=1280, height=720, fps=30, seconds=18)
    if q["available"]:
        # ``credits`` is null when the wallet pays (no credit price); None skips the check.
        job = client.upscale_pro.video("https://example.com/clip.mp4", quote_credits=q.get("credits"))
        done = client.upscale_pro.wait_for_job(job["job_id"])

    out = client.ai_video.generate("A lighthouse at dusk, waves rolling in")
    video = client.ai_video.wait_for_job(out["job"]["id"])

Like :mod:`fotohub.aiwave`, every method builds one request and hands it to ``client._aiw(...)``,
so the same classes serve the sync and the async client.

``quote_credits`` is optional everywhere. Send back the figure you showed (from the matching quote)
and the API refuses the start with a :class:`~fotohub.PriceChangedError` (409, nothing charged)
when the price moved; omit it and no check is made.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from .aiwave import JOB_TERMINAL, ProgressCallback, _body, _Namespace, _request_id
from .exceptions import ValidationError


def _quote_credits(value: Any) -> Optional[float]:
    """``quote_credits`` checked before any request: ``None`` (no price check) or a finite number >= 0.

    A NaN would fail JSON encoding with a bare ``ValueError``; a negative or infinite figure can never
    match a price. Either way it is the caller's mistake, so it is refused here, nothing sent.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"quote_credits must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValidationError(f"quote_credits must be a finite number >= 0, got {value!r}")
    return value


class UpscalePro(_Namespace):
    """``/v1/upscale/pro``: generative 4K image and video upscaling (Professional plan and up).

    An image usually finishes inside the request (200 with ``result``); otherwise, and always for a
    video, the answer is 202 with a ``job_id`` to poll. Charged once when accepted, refunded
    automatically when a job fails or is cancelled. Prices: :meth:`quote` and ``GET /v1/pricing``.

    The start routes take no idempotency key, so the SDK does not repeat a start after a 5xx or a
    read timeout (the first attempt may already be running); only a 429 (refused before anything
    ran) is retried. Check :meth:`jobs` before starting again by hand.
    """

    _BASE = "/v1/upscale/pro"

    def quote(
        self,
        kind: str,
        *,
        width: int,
        height: int,
        scale: Optional[int] = None,
        fps: Optional[float] = None,
        seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        """Free. Whether your plan may run it, the applied scale, output size, ETA, billed seconds
        and, for a video, ``credits`` (``None`` when the wallet pays). ``fps`` and ``seconds`` are
        required for ``kind="video"``."""
        body = _body(kind=kind, width=width, height=height, scale=scale, fps=fps, seconds=seconds)
        return self._run("POST", f"{self._BASE}/quote", body)

    def image(
        self,
        image_url: str,
        *,
        scale: Optional[int] = None,
        wait: Optional[bool] = None,
        client_ref: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Upscale an image (``https://`` URL or ``data:image/...;base64``). Charged.

        Args:
            scale: 2, 3 or 4 (server default 4), lowered automatically to fit the output cap.
            wait: ``True`` (server default) waits up to ~55 s and answers 200 with ``result``;
                otherwise 202 with ``job_id`` / ``poll_url``.
            client_ref: Up to 2 KB of your own JSON, echoed back in the job.
        """
        body = _body(
            image_url=image_url, scale=scale, wait=wait,
            client_ref=dict(client_ref) if client_ref is not None else None,
        )
        return self._client._aiw("POST", f"{self._BASE}/image", body, None, retry_ambiguous=False)

    def video(
        self,
        video_url: str,
        *,
        scale: Optional[int] = None,
        start_seconds: Optional[float] = None,
        max_seconds: Optional[float] = None,
        quote_credits: Optional[float] = None,
        client_ref: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Upscale a span of a video (``https://`` URL). Charged per billed second; answers 202.

        Args:
            scale: 2 (server default), 3 or 4, lowered automatically to fit the output cap.
            start_seconds: Where the span starts (default 0).
            max_seconds: Span length (default: to the end, within the per-job cap).
            quote_credits: The ``credits`` of :meth:`quote` you confirmed. The charge is measured on
                the file; a higher price is refused with :class:`~fotohub.PriceChangedError`
                (``current_credits``, ``billed_seconds``), a lower one is charged as measured.
            client_ref: Up to 2 KB of your own JSON, echoed back in the job.
        """
        body = _body(
            video_url=video_url, scale=scale, start_seconds=start_seconds, max_seconds=max_seconds,
            quote_credits=_quote_credits(quote_credits), client_ref=dict(client_ref) if client_ref is not None else None,
        )
        return self._client._aiw("POST", f"{self._BASE}/video", body, None, retry_ambiguous=False)

    def jobs(
        self, *, active: Optional[bool] = None, kind: Optional[str] = None, limit: Optional[int] = None
    ) -> dict[str, Any]:
        """Your recent jobs, newest first (``active=True``: unfinished only; ``kind``: image|video)."""
        params = {"active": "true" if active else None, "kind": kind, "limit": limit}
        return self._run("GET", f"{self._BASE}/jobs", params=params)

    def job(self, job_id: str) -> dict[str, Any]:
        """One job: ``status``, ``progress``, ``eta_s``; ``result`` once succeeded; ``error`` and
        ``refunded`` when it failed or was cancelled. Free."""
        return self._run("GET", f"{self._BASE}/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; refunded at once. 409 ``not_cancellable`` when it has
        already finished."""
        return self._run("POST", f"{self._BASE}/jobs/{job_id}/cancel")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the job succeeds. Returns the job view.

        Raises:
            FotoHubError: the job failed or was cancelled (``response_body`` is the job view, with
                ``refunded``).
            TimeoutError: not finished within ``timeout`` seconds (the job keeps running).
        """
        return self._poll(
            f"{self._BASE}/jobs/{job_id}", job_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Upscale Pro job",
        )


class AiVideo(_Namespace):
    """``/v1/video/generations``, ``/v1/video/avatar``, ``/v1/video/dub``: FOTOhub AI video jobs.

    API keys only. Every start takes a ``request_id`` (minted when omitted, and re-sent unchanged on
    the SDK's own retries): a retry with the same id returns the first job and is never billed
    twice. The routes refuse unknown fields, so only the arguments you set are sent.
    """

    MOTION_MODEL = "fotohub-motion-audio"

    def generate(
        self,
        prompt: str,
        *,
        image: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        quote_credits: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """AI Motion video with sound (720p, 5 s). Charged. Returns ``{"job": {...}, "created": ...}``.

        Args:
            prompt: What happens in the clip.
            image: ``photos/<your user id>/...`` first frame (image-to-video); omit for text-to-video.
            duration: Seconds; only 5 is available (the server default).
            resolution: Only ``"720p"`` is available (the server default).
            quote_credits: The price you showed; 409 :class:`~fotohub.PriceChangedError` when stale.
            request_id: Your UUID for this run; minted when omitted.
        """
        body = _body(model=self.MOTION_MODEL, prompt=prompt, image=image, duration=duration,
                     resolution=resolution, seed=seed, quote_credits=_quote_credits(quote_credits))
        body["request_id"] = _request_id(request_id)
        return self._run("POST", "/v1/video/generations", body)

    def avatar(
        self,
        portrait: str,
        *,
        consent: Mapping[str, Any],
        mode: Optional[str] = None,
        audio: Optional[str] = None,
        script: Optional[str] = None,
        language: Optional[str] = None,
        voice_ref: Optional[str] = None,
        voice_consent: Optional[bool] = None,
        prompt: Optional[str] = None,
        gesture: Optional[str] = None,
        seed: Optional[int] = None,
        quote_credits: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """A talking-head clip (up to 5 s) from your own portrait. Charged.

        Args:
            portrait: ``photos/<your user id>/...``: your photo or one of a person who agreed.
            consent: Required: ``{"subject": "self" | "consented_person", "version": ...,
                "locale": "en" | "pl" | "de", "sha256"?: ...}``.
            mode: ``"audio"`` (server default: your speech in ``audio``) or ``"script"`` (``script``).
            audio: Your speech (wav or mp3, up to 5 s) for ``mode="audio"``.
            script: What the avatar says (up to 400 characters) for ``mode="script"``.
            language: ``"pl"`` (server default), ``"en"`` or ``"de"``.
            voice_ref: Your own voice recording for ``mode="script"`` (needs ``voice_consent``).
            gesture: ``"natural"`` (server default), ``"calm"`` or ``"expressive"``.
            quote_credits: The price you showed; 409 :class:`~fotohub.PriceChangedError` when stale.
            request_id: Your UUID for this run; minted when omitted.
        """
        body = _body(portrait=portrait, consent=dict(consent), mode=mode, audio=audio, script=script,
                     language=language, voice_ref=voice_ref, voice_consent=voice_consent, prompt=prompt,
                     gesture=gesture, seed=seed, quote_credits=_quote_credits(quote_credits))
        body["request_id"] = _request_id(request_id)
        return self._run("POST", "/v1/video/avatar", body)

    def dub(
        self,
        source: str,
        languages: Sequence[str],
        *,
        source_language: Optional[str] = None,
        glossary: Optional[Sequence[str]] = None,
        voice_ref: Optional[str] = None,
        voice_consent: Optional[Mapping[str, Any]] = None,
        quote_credits: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Dub your video into up to 5 languages, one job each. Charged.

        Returns ``{"project_id", "created", "jobs": [{"language", "job", "created"}, ...]}``.

        Args:
            source: ``videos/<your user id>/...`` (or ``photos/``, ``cloud-drive/``).
            languages: Target languages.
            source_language: Omit or ``"auto"`` to detect it.
            glossary: Up to 30 names and terms to keep exactly as written.
            voice_ref: Your own voice recording to clone for every speaker (needs ``voice_consent``:
                ``{"version", "locale", "sha256"?}``).
            quote_credits: The TOTAL price you showed, all languages; 409 when stale.
            request_id: Your UUID for this run; minted when omitted.
        """
        body = _body(
            source=source, languages=list(languages), source_language=source_language,
            glossary=list(glossary) if glossary is not None else None, voice_ref=voice_ref,
            voice_consent=dict(voice_consent) if voice_consent is not None else None,
            quote_credits=_quote_credits(quote_credits),
        )
        body["request_id"] = _request_id(request_id)
        return self._run("POST", "/v1/video/dub", body)

    def get(self, job_id: str) -> dict[str, Any]:
        """A job any of the three routes returned: ``status``, ``progress``, ``eta_s``, ``result``
        (``assets`` with URLs) once succeeded, ``error`` and ``refunded`` otherwise. Free."""
        return self._run("GET", f"/v1/video/generations/{job_id}")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll :meth:`get` until the job succeeds. Returns the job.

        Raises:
            FotoHubError: the job failed or was cancelled (``response_body`` is the job, with
                ``refunded``).
            TimeoutError: not finished within ``timeout`` seconds (the job keeps running).
        """
        return self._poll(
            f"/v1/video/generations/{job_id}", job_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Video job",
        )


def attach(client: Any) -> None:
    """Give a client its ``upscale_pro`` and ``ai_video`` namespaces (called from both constructors)."""
    client.upscale_pro = UpscalePro(client)
    client.ai_video = AiVideo(client)
