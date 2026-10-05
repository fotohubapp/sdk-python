"""AI Wave 1 namespaces: Character Studio, Product Shot, video edit by chat, music edits.

Reached as attributes of both clients::

    client = FotoHub(api_key="...")
    client.characters.create(...)
    client.product_shot.create(...)
    client.product_shot.batches.create(...)
    client.video_edit.edit(...)
    client.music_edit.edit_section(...)

    async with AsyncFotoHub(api_key="...") as client:
        job = await client.product_shot.create(...)

The same classes serve both clients: every method builds one request and hands it to
``client._aiw(...)``, which returns the parsed JSON on the sync client and a coroutine on the async
one. ``wait_for_job`` / ``wait`` go through ``client._aiw_wait(...)``, which polls with
``time.sleep`` or ``asyncio.sleep`` accordingly.

Conventions (they come from the API, not from this SDK):

* Every charged call takes a ``request_id``. The SDK mints a UUID when you do not pass one and
  re-sends the *same* body on each of its own retries, so a timeout never bills twice. Pass your own
  ``request_id`` to make a retry safe across process restarts too.
* Every price comes from the server. ``quote*`` methods are free and tell you the price, the route
  and the queue before you commit; ``GET /v1/pricing`` lists the same rates. This module hard-codes
  no price and no limit.
* Paths you pass in (``image_path``, ``source_path``, ``photos``) point into your own storage,
  ``photos/<your user id>/...``, ``videos/<id>/...`` or ``audio/<id>/...``; the API refuses anything
  else with a 422 and does not charge.
* A feature that is not enabled for your account answers 403 ``feature_disabled`` (an
  :class:`~fotohub.AuthError`).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

from .exceptions import FotoHubError

#: Job / turn statuses after which nothing changes any more.
JOB_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
#: Product Shot batch statuses after which nothing changes any more (``paused`` is not terminal).
BATCH_TERMINAL = frozenset({"completed", "completed_with_errors", "failed", "cancelled"})

ProgressCallback = Callable[[dict[str, Any]], None]
Selection = Union[Mapping[str, float], Sequence[float]]


def _body(**fields: Any) -> dict[str, Any]:
    """Only the fields the caller set: the server's own defaults apply to the rest."""
    return {k: v for k, v in fields.items() if v is not None}


def _request_id(value: Any) -> str:
    return str(value) if value is not None else str(uuid.uuid4())


def _selection(value: Optional[Selection]) -> Optional[dict[str, float]]:
    """``(start_s, end_s)`` or ``{"start_s": .., "end_s": ..}`` -> the wire shape."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {"start_s": float(value["start_s"]), "end_s": float(value["end_s"])}
    start, end = value
    return {"start_s": float(start), "end_s": float(end)}


def wait_failure(
    view: Mapping[str, Any], terminal: frozenset, label: str, job_id: str
) -> Optional[FotoHubError]:
    """The error a waiter raises for a finished view, or ``None`` when the view is the answer.

    A job or turn that did not succeed raises (its refund state travels in ``response_body``); a
    batch that finished with failed images is returned, because its ``counts`` are the answer.
    """
    status = view.get("status")
    if terminal != JOB_TERMINAL or status not in ("failed", "cancelled"):
        return None
    err = view.get("error")
    detail = (err.get("message") if isinstance(err, Mapping) else err) or f"{label} {job_id} {status}"
    refund = view.get("refund_message")
    return FotoHubError(message=f"{detail} {refund}".strip() if refund else str(detail), response_body=dict(view))


class _Namespace:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _run(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        return self._client._aiw(method, path, body, _body(**params) if params else None)

    def _poll(
        self,
        path: str,
        job_id: str,
        terminal: Iterable[str],
        poll_interval: float,
        timeout: float,
        on_progress: Optional[ProgressCallback],
        label: str,
    ) -> Any:
        return self._client._aiw_wait(
            path, job_id, frozenset(terminal), poll_interval, timeout, on_progress, label
        )


# ─────────────────────────────────────────────────────────────────────────────
# Character Studio
# ─────────────────────────────────────────────────────────────────────────────


class Characters(_Namespace):
    """``/v1/characters``: a character with an identity sheet, usable in any image generation.

    A character is created from 1-5 photos (``source="photos"``, needs the consent declaration) or
    from a description (``source="prompt"``). Creation runs as a job (``job_id`` in the answer);
    the character is ready when its ``status`` is ``"ready"``.
    """

    def config(self) -> dict[str, Any]:
        """Views, photo limits, consent declaration (all locales + sha256), where each step runs,
        models with their reference caps, your plan's limits and today's usage. Free."""
        return self._run("GET", "/v1/characters/config")

    @staticmethod
    def _create_body(
        name: str,
        source: str,
        kind: str,
        photos: Optional[Sequence[str]],
        prompt: Optional[str],
        description: Optional[str],
        strength: Optional[float],
        consent: Optional[Mapping[str, Any]],
        seed: Optional[int],
    ) -> dict[str, Any]:
        return _body(
            name=name,
            kind=kind,
            source=source,
            photos=list(photos) if photos is not None else None,
            prompt=prompt,
            description=description,
            strength=strength,
            consent=dict(consent) if consent is not None else None,
            seed=seed,
        )

    def quote(
        self,
        name: str,
        *,
        source: str,
        kind: str = "face",
        photos: Optional[Sequence[str]] = None,
        prompt: Optional[str] = None,
        description: Optional[str] = None,
        strength: Optional[float] = None,
        consent: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Price, availability, ETA and your limits for a new character. Never charges."""
        body = self._create_body(name, source, kind, photos, prompt, description, strength, consent, seed)
        return self._run("POST", "/v1/characters/quote", body)

    def create(
        self,
        name: str,
        *,
        source: str,
        kind: str = "face",
        photos: Optional[Sequence[str]] = None,
        prompt: Optional[str] = None,
        description: Optional[str] = None,
        strength: Optional[float] = None,
        consent: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a character and start its identity sheet. Charged.

        Args:
            name: 1-60 characters.
            source: ``"photos"`` or ``"prompt"``.
            kind: ``"face"`` (default) or ``"non_face"`` (mascot, product character, ...).
            photos: 1-5 storage paths, ``photos/<your user id>/...`` (``source="photos"``).
            prompt: The character in a few words (``source="prompt"``).
            consent: Required for photos: ``{"subject": "self" | "consented_person" | "fictional",
                "version": <config()["consent"]["version"]>, "locale": "en" | "pl" | "de",
                "sha256": <config()["consent"]["texts"][locale]["sha256"]>}``.
            strength: How strictly later generations keep the identity, 0..1.
            request_id: Idempotency key; a UUID is generated when omitted.

        Returns:
            ``{"character": {...}, "job": {"job_id": ...}, "eta": ..., "quote": ..., "replayed": bool}``.
            The sheet is refunded in full when fewer views than ``config()["min_accepted_views"]``
            pass the identity check.
        """
        body = self._create_body(name, source, kind, photos, prompt, description, strength, consent, seed)
        body["request_id"] = _request_id(request_id)
        return self._run("POST", "/v1/characters", body)

    def list(self, *, limit: int = 50) -> dict[str, Any]:
        """Your characters, your plan's limits and today's usage."""
        return self._run("GET", "/v1/characters", params={"limit": limit})

    def get(self, character_id: str) -> dict[str, Any]:
        """One character: views with identity scores, references, sheet job state and ETA."""
        return self._run("GET", f"/v1/characters/{character_id}")

    def update(
        self,
        character_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        strength: Optional[float] = None,
        exclude_assets: Optional[Sequence[str]] = None,
        include_assets: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Rename, describe, change the strength, exclude or re-include sheet views. Free."""
        body = _body(
            name=name,
            description=description,
            strength=strength,
            exclude_assets=list(exclude_assets) if exclude_assets is not None else None,
            include_assets=list(include_assets) if include_assets is not None else None,
        )
        return self._run("PATCH", f"/v1/characters/{character_id}", body)

    def delete(self, character_id: str) -> dict[str, Any]:
        """Delete a character: active jobs are cancelled (and refunded), photos and sheet deleted.
        The consent record is retained. Free."""
        return self._run("DELETE", f"/v1/characters/{character_id}")

    def regenerate_sheet(
        self, character_id: str, *, seed: Optional[int] = None, request_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Regenerate the identity sheet. Charged like a creation."""
        body = _body(seed=seed)
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"/v1/characters/{character_id}/sheet", body)

    def resolve(
        self, character_id: str, model: str, *, strength: Optional[float] = None
    ) -> dict[str, Any]:
        """What a generator would send for ``model``: reference URLs, a prompt prefix and the
        strength, so you can drive the model yourself. Free."""
        return self._run(
            "POST", f"/v1/characters/{character_id}/resolve", _body(model=model, strength=strength)
        )

    def quote_generation(
        self,
        character_id: str,
        prompt: str,
        *,
        model: str = "seedream-5-0-260128",
        aspect_ratio: Optional[str] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """The model's own price plus the character-lock fee, availability and ETA. Never charges."""
        body = _body(model=model, prompt=prompt, aspect_ratio=aspect_ratio, strength=strength, seed=seed)
        return self._run("POST", f"/v1/characters/{character_id}/generations/quote", body)

    def generate(
        self,
        character_id: str,
        prompt: str,
        *,
        model: str = "seedream-5-0-260128",
        aspect_ratio: Optional[str] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """An image of the character. Charged: the model's own price now, the character-lock fee
        only when the identity check accepts the result. Returns a job; collect it with
        :meth:`wait_for_job`."""
        body = _body(model=model, prompt=prompt, aspect_ratio=aspect_ratio, strength=strength, seed=seed)
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"/v1/characters/{character_id}/generations", body)

    def jobs(
        self, character_id: str, *, limit: int = 20, kind: Optional[str] = None
    ) -> dict[str, Any]:
        """The character's recent jobs (``kind``: ``"sheet"`` or ``"generation"``)."""
        return self._run(
            "GET", f"/v1/characters/{character_id}/jobs", params={"limit": limit, "kind": kind}
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Status, stage, progress, ETA, result or error (and whether it was refunded)."""
        return self._run("GET", f"/v1/characters/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; refunded once. Free."""
        return self._run("POST", f"/v1/characters/jobs/{job_id}/cancel")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the job succeeds. Returns the job view.

        Raises:
            FotoHubError: the job failed or was cancelled (``response_body`` is the job view, which
                says whether it was refunded).
            TimeoutError: the job did not finish within ``timeout`` seconds (it keeps running).
        """
        return self._poll(
            f"/v1/characters/jobs/{job_id}", job_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Character job",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Product Shot
# ─────────────────────────────────────────────────────────────────────────────


def _shot_settings(
    preset_id: Optional[str],
    prompt: Optional[str],
    params: Optional[Mapping[str, Any]],
    aspects: Optional[Sequence[str]],
    tier: Optional[str],
    scene_model: Optional[str],
) -> dict[str, Any]:
    return _body(
        preset_id=preset_id,
        prompt=prompt,
        params=dict(params) if params is not None else None,
        aspects=list(aspects) if aspects is not None else None,
        tier=tier,
        scene_model=scene_model,
    )


def _consent_fields(consent_us_processing: bool, consent_version: Optional[str]) -> dict[str, Any]:
    if consent_us_processing and not consent_version:
        raise ValueError(
            "consent_version is required with consent_us_processing=True: pass "
            "config()['residency']['consent_version'] of the Product Shot config."
        )
    return _body(consent_us_processing=bool(consent_us_processing), consent_version=consent_version)


class ProductShotBatches(_Namespace):
    """``/v1/aiwave/product-shot/batches``: up to the plan's item ceiling per batch.

    Each image is charged when it starts and refunded individually when it fails; ``cancel``
    refunds what is in flight and never charges what has not started.
    """

    _BASE = "/v1/aiwave/product-shot/batches"

    def quote(
        self,
        skus: int = 1,
        *,
        preset_id: Optional[str] = None,
        prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        aspects: Optional[Sequence[str]] = None,
        tier: Optional[str] = None,
        scene_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Price per image and in total, availability, your ceilings and whether the batch fits."""
        body = _shot_settings(preset_id, prompt, params, aspects, tier, scene_model)
        body["skus"] = skus
        return self._run("POST", f"{self._BASE}/quote", body)

    def create(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        name: Optional[str] = None,
        preset_id: Optional[str] = None,
        prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        aspects: Optional[Sequence[str]] = None,
        tier: Optional[str] = None,
        scene_model: Optional[str] = None,
        consent_us_processing: bool = False,
        consent_version: Optional[str] = None,
        source: Optional[str] = "api",
        external_ref: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a batch. ``items`` are ``{"image_path": ...}`` or ``{"image_url": "https://..."}``
        plus optional ``sku`` / ``external_id`` / ``title``. Needs the US-processing consent."""
        body = _shot_settings(preset_id, prompt, params, aspects, tier, scene_model)
        body.update(_body(name=name, source=source, external_ref=external_ref))
        body["items"] = [dict(i) for i in items]
        body.update(_consent_fields(consent_us_processing, consent_version))
        body["request_id"] = _request_id(request_id)
        return self._run("POST", self._BASE, body)

    def list(self, *, limit: int = 20) -> dict[str, Any]:
        """Your recent batches and your ceilings."""
        return self._run("GET", self._BASE, params={"limit": limit})

    def get(self, batch_id: str) -> dict[str, Any]:
        """Status, counts, progress, spend, pause reason, ETA and the archive state."""
        return self._run("GET", f"{self._BASE}/{batch_id}")

    def items(
        self,
        batch_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Per-image status, result, error, charge and refund state."""
        return self._run(
            "GET",
            f"{self._BASE}/{batch_id}/items",
            params={"status": status, "limit": limit, "offset": offset},
        )

    def cancel(self, batch_id: str) -> dict[str, Any]:
        """Stop the batch. Free."""
        return self._run("POST", f"{self._BASE}/{batch_id}/cancel")

    def retry_failed(self, batch_id: str) -> dict[str, Any]:
        """Queue the failed images again (each is charged when it starts)."""
        return self._run("POST", f"{self._BASE}/{batch_id}/retry-failed")

    def resume(self, batch_id: str, *, accept_price: bool = False) -> dict[str, Any]:
        """Resume a paused batch. Pass ``accept_price=True`` when the pause was a price change."""
        return self._run("POST", f"{self._BASE}/{batch_id}/resume", {"accept_price": bool(accept_price)})

    def build_archive(self, batch_id: str) -> dict[str, Any]:
        """(Re)build the results zip once the batch has finished. Free."""
        return self._run("POST", f"{self._BASE}/{batch_id}/archive")

    def get_archive(self, batch_id: str) -> dict[str, Any]:
        """The zip parts; a 409 ``archive_not_ready`` (:class:`~fotohub.FotoHubError`) while it is built."""
        return self._run("GET", f"{self._BASE}/{batch_id}/archive")

    def wait(
        self,
        batch_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the batch reaches a terminal status (``completed``, ``completed_with_errors``,
        ``failed`` or ``cancelled``) and return its view. A batch with failed images is returned,
        not raised: look at ``counts`` and use :meth:`retry_failed`. A ``paused`` batch keeps
        waiting; resume it (see ``pause`` in the view).

        Raises:
            TimeoutError: not finished within ``timeout`` seconds (the batch keeps running).
        """
        return self._poll(
            f"{self._BASE}/{batch_id}", batch_id, BATCH_TERMINAL, poll_interval, timeout, on_progress,
            "Product Shot batch",
        )


class ProductShot(_Namespace):
    """``/v1/aiwave/product-shot``: a product photo on a clean or generated background.

    The product's own pixels are preserved. Cut-outs are made in the United States; the first
    ``create`` therefore needs the consent text from :meth:`config` (``residency``), see
    ``consent_us_processing`` / ``consent_version``.
    """

    _BASE = "/v1/aiwave/product-shot"

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.batches = ProductShotBatches(client)

    def config(self) -> dict[str, Any]:
        """Presets, aspects, marketplaces, premium scene models, your saved presets, the
        processing-location consent text (``residency``) and limits. Free."""
        return self._run("GET", f"{self._BASE}/config")

    def quote(
        self,
        image_path: str,
        *,
        preset_id: Optional[str] = None,
        prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        aspects: Optional[Sequence[str]] = None,
        tier: Optional[str] = None,
        scene_model: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Price per image and in total, availability and ETA, before any charge."""
        body = _shot_settings(preset_id, prompt, params, aspects, tier, scene_model)
        body.update(_body(image_path=image_path, seed=seed))
        return self._run("POST", f"{self._BASE}/quote", body)

    def create(
        self,
        image_path: str,
        *,
        preset_id: Optional[str] = None,
        prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        aspects: Optional[Sequence[str]] = None,
        tier: Optional[str] = None,
        scene_model: Optional[str] = None,
        seed: Optional[int] = None,
        consent_us_processing: bool = False,
        consent_version: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create one job per requested aspect. Charged per image, all-or-nothing.

        Args:
            image_path: ``photos/<your user id>/...``.
            preset_id: A built-in or saved preset id (default ``studio_white``).
            prompt: A custom scene description instead of a preset.
            aspects: Subset of ``config()["aspects"]``; default ``["1:1"]``.
            tier: ``"standard"`` (default) or ``"premium"`` (``scene_model`` from
                ``config()["premium_models"]``).
            consent_us_processing, consent_version: Required. Set the first to ``True`` and the
                second to ``config()["residency"]["consent_version"]``.

        Returns:
            ``{"request_id", "jobs": [{"job_id", "aspect", "status", "replayed"}], "quote", "eta",
            "residency"}``.
        """
        body = _shot_settings(preset_id, prompt, params, aspects, tier, scene_model)
        body.update(_body(image_path=image_path, seed=seed))
        body.update(_consent_fields(consent_us_processing, consent_version))
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"{self._BASE}/jobs", body)

    def list_jobs(self, *, request_id: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        """Your recent jobs (optionally the jobs of one ``create`` call)."""
        return self._run("GET", f"{self._BASE}/jobs", params={"request_id": request_id, "limit": limit})

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Status, stage, progress, ETA, result or error (and whether it was refunded)."""
        return self._run("GET", f"{self._BASE}/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; refunded once. Free."""
        return self._run("POST", f"{self._BASE}/jobs/{job_id}/cancel")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the job succeeds and return its view.

        Raises:
            FotoHubError: failed or cancelled (``response_body`` is the job view).
            TimeoutError: not finished within ``timeout`` seconds (the job keeps running).
        """
        return self._poll(
            f"{self._BASE}/jobs/{job_id}", job_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Product Shot job",
        )

    def presets(self) -> dict[str, Any]:
        """Your saved presets."""
        return self._run("GET", f"{self._BASE}/presets")

    def create_preset(
        self,
        name: str,
        *,
        prompt: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        kind: Optional[str] = None,
    ) -> dict[str, Any]:
        """Save a preset (``kind``: ``studio``, ``lifestyle``, ``seasonal`` or ``custom``). Free."""
        body = _body(
            name=name, prompt=prompt, params=dict(params) if params is not None else None, kind=kind
        )
        return self._run("POST", f"{self._BASE}/presets", body)

    def delete_preset(self, preset_id: str) -> dict[str, Any]:
        """Delete a saved preset. Free."""
        return self._run("DELETE", f"{self._BASE}/presets/{preset_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Video edit by chat
# ─────────────────────────────────────────────────────────────────────────────


class VideoEdit(_Namespace):
    """``/v1/aiwave/video-edit``: edit or extend a clip with plain-language instructions.

    A *session* is a conversation about one clip; every *turn* is one instruction and produces a new
    version of the clip (the head). ``undo`` / ``revert`` move the head without a charge. Priced per
    billable second of the edited range (see :meth:`quote`).
    """

    _BASE = "/v1/aiwave/video-edit"

    def config(self) -> dict[str, Any]:
        """Limits (selection, clip length, extension), the public route names, price multipliers,
        the AI-generation disclosure, the likeness notice and whether you may pay with credits. Free."""
        return self._run("GET", f"{self._BASE}/config")

    def create_session(
        self,
        source_path: str,
        *,
        duration_s: float,
        width: int,
        height: int,
        fps: Optional[float] = None,
        title: Optional[str] = None,
    ) -> dict[str, Any]:
        """Start a session on one of your clips (``videos/<your user id>/...`` or
        ``photos/<your user id>/...``). Describe the clip with its ``duration_s`` / ``width`` /
        ``height`` (and ``fps``). Free."""
        clip = _body(duration_s=duration_s, width=width, height=height, fps=fps)
        return self._run(
            "POST", f"{self._BASE}/sessions", _body(source_path=source_path, clip=clip, title=title)
        )

    def sessions(self, *, limit: int = 20) -> dict[str, Any]:
        """Your recent sessions."""
        return self._run("GET", f"{self._BASE}/sessions", params={"limit": limit})

    def get_session(self, session_id: str) -> dict[str, Any]:
        """The session, its head clip and every turn with status and ETA."""
        return self._run("GET", f"{self._BASE}/sessions/{session_id}")

    def archive_session(self, session_id: str) -> dict[str, Any]:
        """Archive a session (the outputs stay in your storage). Free."""
        return self._run("DELETE", f"{self._BASE}/sessions/{session_id}")

    @staticmethod
    def _turn_body(
        instruction: str,
        selection: Optional[Selection],
        extend_seconds: Optional[int],
        audio: Optional[str],
        likeness_ack: Optional[bool],
    ) -> dict[str, Any]:
        return _body(
            instruction=instruction,
            selection=_selection(selection),
            extend_seconds=extend_seconds,
            audio=audio,
            likeness_ack=likeness_ack,
        )

    def quote(
        self,
        session_id: str,
        instruction: str,
        *,
        selection: Optional[Selection] = None,
        extend_seconds: Optional[int] = None,
        audio: Optional[str] = None,
        likeness_ack: Optional[bool] = None,
    ) -> dict[str, Any]:
        """The intent the instruction was read as, the route that will run, billable seconds, price,
        how you pay, availability and ETA. Never charges."""
        body = self._turn_body(instruction, selection, extend_seconds, audio, likeness_ack)
        return self._run("POST", f"{self._BASE}/sessions/{session_id}/quote", body)

    def edit(
        self,
        session_id: str,
        instruction: str,
        *,
        selection: Optional[Selection] = None,
        extend_seconds: Optional[int] = None,
        audio: Optional[str] = None,
        likeness_ack: Optional[bool] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """One edit turn. Charged; refunded once if it fails or is cancelled.

        Args:
            instruction: What to change, in plain language (Polish or English).
            selection: ``(start_s, end_s)`` of the part to edit; required for edits, not for
                extensions. At most ``config()["limits"]["max_selection_s"]`` long.
            extend_seconds: Seconds to add when the instruction extends the clip.
            audio: ``"original"`` or ``"edited"``.
            likeness_ack: ``True`` when the quote says ``likeness_ack_required``.

        Returns:
            ``{"turn": {"turn_id", "status", ...}, "replayed", "eta", "quote", "payment"}``. Collect
            the result with :meth:`wait_for_turn`.
        """
        body = self._turn_body(instruction, selection, extend_seconds, audio, likeness_ack)
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"{self._BASE}/sessions/{session_id}/turns", body)

    def undo(self, session_id: str) -> dict[str, Any]:
        """Move the head to the version before the last applied edit. Free."""
        return self._run("POST", f"{self._BASE}/sessions/{session_id}/undo")

    def revert(self, session_id: str, turn_id: Optional[str] = None) -> dict[str, Any]:
        """Move the head to any delivered turn, or to the original clip (``turn_id=None``). Free."""
        return self._run("POST", f"{self._BASE}/sessions/{session_id}/revert", {"turn_id": turn_id})

    def get_turn(self, turn_id: str) -> dict[str, Any]:
        """Status, stage, progress, ETA, result or error (and whether it was refunded)."""
        return self._run("GET", f"{self._BASE}/turns/{turn_id}")

    def cancel_turn(self, turn_id: str) -> dict[str, Any]:
        """Cancel a queued or running turn; refunded once. Free."""
        return self._run("POST", f"{self._BASE}/turns/{turn_id}/cancel")

    def wait_for_turn(
        self,
        turn_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 900.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the turn succeeds and return its view (``result.url`` is the new clip).

        Raises:
            FotoHubError: failed or cancelled (``response_body`` is the turn view).
            TimeoutError: not finished within ``timeout`` seconds (the turn keeps running).
        """
        return self._poll(
            f"{self._BASE}/turns/{turn_id}", turn_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Video edit turn",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Music edits
# ─────────────────────────────────────────────────────────────────────────────


class MusicEdit(_Namespace):
    """``/v1/aiwave/music-edit``: re-do a section or a lyric line of a song, or score a video/image.

    Edits are spliced back into the original: everything outside the edited region stays
    sample-identical. Apply an edit as a *new version*; chain edits with ``parent_job_id``.
    """

    _BASE = "/v1/aiwave/music-edit"

    def config(self) -> dict[str, Any]:
        """Prices per operation, limits, stages, the AI disclosure and where each step runs. Free."""
        return self._run("GET", f"{self._BASE}/config")

    @staticmethod
    def _section_body(
        source_path: str,
        source_duration_s: float,
        start_s: float,
        end_s: float,
        instruction: str,
        lyrics: Optional[str],
        style: Optional[str],
        snap_to_beat: Optional[bool],
        crossfade_ms: Optional[int],
        seed: Optional[int],
        parent_job_id: Optional[str],
    ) -> dict[str, Any]:
        return _body(
            source_path=source_path,
            source_duration_s=source_duration_s,
            start_s=start_s,
            end_s=end_s,
            instruction=instruction,
            lyrics=lyrics,
            style=style,
            snap_to_beat=snap_to_beat,
            crossfade_ms=crossfade_ms,
            seed=seed,
            parent_job_id=parent_job_id,
        )

    @staticmethod
    def _lyrics_body(
        source_path: str,
        source_duration_s: float,
        line: Mapping[str, Any],
        new_text: str,
        style: Optional[str],
        crossfade_ms: Optional[int],
        seed: Optional[int],
        parent_job_id: Optional[str],
    ) -> dict[str, Any]:
        return _body(
            source_path=source_path,
            source_duration_s=source_duration_s,
            line=dict(line),
            new_text=new_text,
            style=style,
            crossfade_ms=crossfade_ms,
            seed=seed,
            parent_job_id=parent_job_id,
        )

    @staticmethod
    def _soundtrack_body(
        source_path: str,
        media_duration_s: Optional[float],
        start_s: Optional[float],
        duration_s: Optional[int],
        mood: Optional[str],
        sfx: Optional[bool],
        keep_original_audio: Optional[bool],
        duck_db: Optional[float],
        seed: Optional[int],
    ) -> dict[str, Any]:
        return _body(
            source_path=source_path,
            media_duration_s=media_duration_s,
            start_s=start_s,
            duration_s=duration_s,
            mood=mood,
            sfx=sfx,
            keep_original_audio=keep_original_audio,
            duck_db=duck_db,
            seed=seed,
        )

    def quote_section(
        self,
        source_path: str,
        *,
        source_duration_s: float,
        start_s: float,
        end_s: float,
        instruction: str,
        lyrics: Optional[str] = None,
        style: Optional[str] = None,
        snap_to_beat: Optional[bool] = None,
        crossfade_ms: Optional[int] = None,
        seed: Optional[int] = None,
        parent_job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Price, availability, ETA and the normalised plan of a section edit. Never charges."""
        body = self._section_body(
            source_path, source_duration_s, start_s, end_s, instruction, lyrics, style, snap_to_beat,
            crossfade_ms, seed, parent_job_id,
        )
        return self._run("POST", f"{self._BASE}/section/quote", body)

    def edit_section(
        self,
        source_path: str,
        *,
        source_duration_s: float,
        start_s: float,
        end_s: float,
        instruction: str,
        lyrics: Optional[str] = None,
        style: Optional[str] = None,
        snap_to_beat: Optional[bool] = None,
        crossfade_ms: Optional[int] = None,
        seed: Optional[int] = None,
        parent_job_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Regenerate ``[start_s, end_s]`` of a song from a plain-language instruction. Charged.

        Send ``lyrics`` for a sung section (the vocal is re-sung with them); without them the new
        section is instrumental. Returns ``{"job", "replayed", "eta", "quote", "payment"}``; collect
        the result with :meth:`wait_for_job`.
        """
        body = self._section_body(
            source_path, source_duration_s, start_s, end_s, instruction, lyrics, style, snap_to_beat,
            crossfade_ms, seed, parent_job_id,
        )
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"{self._BASE}/section", body)

    def quote_lyrics(
        self,
        source_path: str,
        *,
        source_duration_s: float,
        line: Mapping[str, Any],
        new_text: str,
        style: Optional[str] = None,
        crossfade_ms: Optional[int] = None,
        seed: Optional[int] = None,
        parent_job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Price, availability and ETA of a lyric-line edit. Never charges."""
        body = self._lyrics_body(
            source_path, source_duration_s, line, new_text, style, crossfade_ms, seed, parent_job_id
        )
        return self._run("POST", f"{self._BASE}/lyrics/quote", body)

    def edit_lyrics(
        self,
        source_path: str,
        *,
        source_duration_s: float,
        line: Mapping[str, Any],
        new_text: str,
        style: Optional[str] = None,
        crossfade_ms: Optional[int] = None,
        seed: Optional[int] = None,
        parent_job_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Re-sing one line with new words, melody and voice kept. Charged.

        ``line`` is ``{"start_s": .., "end_s": .., "text": <old words, optional>}``.
        """
        body = self._lyrics_body(
            source_path, source_duration_s, line, new_text, style, crossfade_ms, seed, parent_job_id
        )
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"{self._BASE}/lyrics", body)

    def quote_soundtrack(
        self,
        source_path: str,
        *,
        media_duration_s: Optional[float] = None,
        start_s: Optional[float] = None,
        duration_s: Optional[int] = None,
        mood: Optional[str] = None,
        sfx: Optional[bool] = None,
        keep_original_audio: Optional[bool] = None,
        duck_db: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Price, availability and ETA of a soundtrack. Never charges."""
        body = self._soundtrack_body(
            source_path, media_duration_s, start_s, duration_s, mood, sfx, keep_original_audio, duck_db, seed
        )
        return self._run("POST", f"{self._BASE}/soundtrack/quote", body)

    def soundtrack(
        self,
        source_path: str,
        *,
        media_duration_s: Optional[float] = None,
        start_s: Optional[float] = None,
        duration_s: Optional[int] = None,
        mood: Optional[str] = None,
        sfx: Optional[bool] = None,
        keep_original_audio: Optional[bool] = None,
        duck_db: Optional[float] = None,
        seed: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Music (and optionally sound effects, ducked under it) for a video or an image. Charged.

        ``source_path`` is ``videos/<id>/...`` or ``photos/<id>/...``; for a video also send
        ``media_duration_s``.
        """
        body = self._soundtrack_body(
            source_path, media_duration_s, start_s, duration_s, mood, sfx, keep_original_audio, duck_db, seed
        )
        body["request_id"] = _request_id(request_id)
        return self._run("POST", f"{self._BASE}/soundtrack", body)

    def jobs(self, *, op: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        """Your history, newest first (``op``: ``section``, ``lyrics`` or ``soundtrack``)."""
        return self._run("GET", f"{self._BASE}/jobs", params={"op": op, "limit": limit})

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Status, stage, progress, ETA, result or error (and whether it was refunded)."""
        return self._run("GET", f"{self._BASE}/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; refunded once. Free."""
        return self._run("POST", f"{self._BASE}/jobs/{job_id}/cancel")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 4.0,
        timeout: float = 900.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Any:
        """Poll until the job succeeds and return its view.

        Raises:
            FotoHubError: failed or cancelled (``response_body`` is the job view).
            TimeoutError: not finished within ``timeout`` seconds (the job keeps running).
        """
        return self._poll(
            f"{self._BASE}/jobs/{job_id}", job_id, JOB_TERMINAL, poll_interval, timeout, on_progress,
            "Music edit job",
        )


def attach(client: Any) -> None:
    """Give a client its four namespaces (called from both client constructors)."""
    client.characters = Characters(client)
    client.product_shot = ProductShot(client)
    client.video_edit = VideoEdit(client)
    client.music_edit = MusicEdit(client)
