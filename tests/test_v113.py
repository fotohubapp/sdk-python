"""1.13.0: typed price / URL / pricing errors, Kling `mode`, Upscale Pro, AI video, IDA Q Image 2.

HTTP is mocked at the transport; nothing here reaches a server. Bodies mirror what api-server sends
(`{"detail": {...}}`, FastAPI default).
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid

import httpx
import pytest

import fotohub
from fotohub import (
    AsyncFotoHub,
    AuthError,
    FotoHub,
    FotoHubError,
    PriceChangedError,
    PricingNotConfiguredError,
    RateLimitError,
    ServerError,
    UrlBlockedError,
    ValidationError,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
JID = "5b0c9a52-5d1e-4f3e-9f3a-70d0c0a1e0b1"


@pytest.fixture()
def fast(monkeypatch):
    real = asyncio.sleep

    async def instant(_d, *a, **k):
        await real(0)

    monkeypatch.setattr(time, "sleep", lambda _d: None)
    monkeypatch.setattr(asyncio, "sleep", instant)


def _retrying(rec):
    c = FotoHub(api_key="fh_test_key", base_url="https://api.test", max_retries=3)
    c._client = httpx.Client(base_url=c.base_url, headers=c._headers(), transport=httpx.MockTransport(rec.handler))
    return c


# ── version ────────────────────────────────────────────────────────────────────

def test_version_is_1_13_0(client):
    assert fotohub.__version__ == "1.13.0"
    assert client._headers()["User-Agent"] == "fotohub-python/1.13.0"


# ── typed errors ───────────────────────────────────────────────────────────────

UPSCALE_409 = {"detail": {
    "error": "price_changed",
    "message": "The price changed since it was shown: this video is longer or larger than quoted. "
               "Check the new price. Your wallet was not charged for this request.",
    "quoted_credits": 12.0, "current_credits": 18.5, "billed_seconds": 20,
}}
WAVE_409 = {"detail": {
    "error": "price_changed", "quoted_credits": 3.0, "current_credits": 4.2,
    "message": "The price changed since it was shown. Check the new price; you were not charged.",
}}


def test_price_changed_409_from_upscale_pro_is_typed(client, rec):
    rec.queue((409, UPSCALE_409))
    with pytest.raises(PriceChangedError) as ei:
        client.upscale_pro.video("https://example.com/clip.mp4", quote_credits=12.0)
    e = ei.value
    assert isinstance(e, FotoHubError)
    assert e.status_code == 409 and e.code == "price_changed"
    assert (e.quoted_credits, e.current_credits, e.billed_seconds) == (12.0, 18.5, 20)
    assert e.charged is False
    assert "price changed" in e.message


def test_price_changed_409_from_ai_video_has_no_billed_seconds(client, rec):
    rec.queue((409, WAVE_409))
    with pytest.raises(PriceChangedError) as ei:
        client.ai_video.generate("a lighthouse at dusk", quote_credits=3.0)
    assert (ei.value.quoted_credits, ei.value.current_credits, ei.value.billed_seconds) == (3.0, 4.2, None)
    assert len(rec.requests) == 1  # a 409 price_changed is an answer, never retried


def test_price_changed_is_not_retried_even_with_a_retry_budget(rec, fast):
    c = _retrying(rec)
    rec.queue((409, WAVE_409), (409, WAVE_409), (409, WAVE_409))
    with pytest.raises(PriceChangedError):
        c.ai_video.generate("a lighthouse at dusk", quote_credits=3.0)
    assert len(rec.requests) == 1


@pytest.mark.parametrize("code,text", [
    ("url_blocked", "image_url must be a public https URL"),
    ("url_not_allowed", "image_url must be a file from your FOTOhub storage"),
])
def test_url_refusals_are_typed_and_still_validation_errors(client, rec, code, text):
    rec.queue((400, {"detail": {"error": code, "field": "image_urls[2]", "charged": False,
                                "message": f"{text}. Your wallet was not charged for this request."}}))
    with pytest.raises(UrlBlockedError) as ei:
        client.upscale_pro.image("http://10.0.0.1/a.png")
    e = ei.value
    assert isinstance(e, ValidationError)  # an old `except ValidationError` still catches it
    assert e.status_code == 400 and e.code == code
    assert e.field == "image_urls[2]" and e.charged is False


def test_pricing_not_configured_503_is_typed_and_not_retried(rec, fast):
    c = _retrying(rec)
    body = {"detail": {"code": "PRICING_NOT_CONFIGURED",
                       "message": "This model has no video price configured. You were not charged."}}
    rec.queue((503, body), (503, body), (503, body))
    with pytest.raises(PricingNotConfiguredError) as ei:
        c.generate_video("a fox", model="kling-v3")
    e = ei.value
    assert isinstance(e, ServerError)
    assert e.status_code == 503 and e.code == "PRICING_NOT_CONFIGURED" and e.charged is False
    assert len(rec.requests) == 1


def test_pricing_not_configured_inside_a_424_string_is_typed(client, rec):
    raw = ('{"success":false,"error":"Brak ceny","code":"PRICING_NOT_CONFIGURED"} '
           "Your wallet was not charged for this request.")
    rec.queue((424, {"detail": raw}))
    with pytest.raises(PricingNotConfiguredError) as ei:
        client.generate_video("a fox", model="kling-v3")
    assert ei.value.status_code == 424 and ei.value.code == "PRICING_NOT_CONFIGURED"


def test_other_503s_are_still_plain_server_errors_and_retried(rec, fast):
    c = _retrying(rec)
    rec.queue((503, {"detail": "busy"}), (200, {"video_url": "https://cdn.test/v.mp4", "status": "completed"}))
    out = c.generate_video("a fox")
    assert out["video_url"].endswith("v.mp4") and len(rec.requests) == 2


@pytest.mark.parametrize("status,detail,cls", [
    (429, {"error": "The IDA Q queue is full. Try again shortly.", "code": "QUEUE_FULL", "charged": False,
           "eta_seconds": 900, "retry_after": 120}, RateLimitError),
    (429, {"error": "You already have the maximum number of active IDA Q jobs.", "code": "TOO_MANY_ACTIVE",
           "charged": False, "active_limit": 2}, RateLimitError),
    (403, {"error": "Your plan does not include IDA Q Image 2.", "code": "PLAN_REQUIRED", "charged": False},
     AuthError),
    (503, {"error": "IDA Q Image 2 is temporarily unavailable.", "code": "MODEL_DISABLED", "charged": False},
     ServerError),
    (409, {"error": "This job_id is already used. Send a new job_id.", "code": "JOB_ID_CONFLICT",
           "charged": False}, FotoHubError),
    (422, {"error": "num_images must be a whole number from 1 to 1 for HD / max.", "code": "INVALID_REQUEST",
           "charged": False, "message": "Your wallet was not charged for this request."}, ValidationError),
])
def test_ida_q2_refusals_carry_their_code(client, rec, status, detail, cls):
    rec.queue((status, {"detail": detail}))
    with pytest.raises(cls) as ei:
        client.generate_ida_q2("a poster", size_tier="HD", preset="max")
    assert ei.value.code == detail["code"]
    assert ei.value.status_code == status
    if status == 429 and "retry_after" in detail:
        assert ei.value.retry_after == 120


# ── generate_video(mode=) ─────────────────────────────────────────────────────

def test_generate_video_sends_mode_only_when_set(client, rec):
    rec.queue((200, {"video_url": "https://cdn.test/a.mp4"}), (200, {"video_url": "https://cdn.test/b.mp4"}))
    client.generate_video("a fox", model="kling-v3", mode="pro", duration=10)
    assert rec.body()["mode"] == "pro" and rec.body()["duration"] == 10
    client.generate_video("a fox")
    assert "mode" not in rec.body()


async def test_async_generate_video_sends_mode(aclient, rec):
    rec.queue((200, {"video_url": "https://cdn.test/a.mp4"}))
    await aclient.generate_video("a fox", model="kling-v3-omni", mode="standard")
    assert rec.body()["mode"] == "standard"


# ── Upscale Pro ────────────────────────────────────────────────────────────────

def test_upscale_pro_quote_sends_only_what_was_set(client, rec):
    rec.queue((200, {"available": True, "credits": 12.0, "billed_seconds": 20}))
    out = client.upscale_pro.quote("video", width=1280, height=720, scale=2, fps=30, seconds=18)
    assert out["credits"] == 12.0
    assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/upscale/pro/quote")
    assert rec.body() == {"kind": "video", "width": 1280, "height": 720, "scale": 2, "fps": 30, "seconds": 18}
    client.upscale_pro.quote("image", width=800, height=600)
    assert rec.body() == {"kind": "image", "width": 800, "height": 600}


def test_upscale_pro_image_and_video_bodies(client, rec):
    rec.queue((200, {"status": "succeeded", "result": {"url": "https://cdn.test/x.png"}}),
              (202, {"job_id": JID, "status": "queued"}))
    client.upscale_pro.image("https://example.com/a.png", scale=3, wait=False, client_ref={"sku": "A1"})
    assert rec.last.url.path == "/v1/upscale/pro/image"
    assert rec.body() == {"image_url": "https://example.com/a.png", "scale": 3, "wait": False,
                          "client_ref": {"sku": "A1"}}
    out = client.upscale_pro.video("https://example.com/c.mp4", start_seconds=2, max_seconds=10,
                                   quote_credits=12.0)
    assert out["job_id"] == JID
    assert rec.last.url.path == "/v1/upscale/pro/video"
    body = rec.body()
    assert body == {"video_url": "https://example.com/c.mp4", "start_seconds": 2, "max_seconds": 10,
                    "quote_credits": 12.0}
    assert "request_id" not in body  # the route has no such field


def test_upscale_pro_start_is_not_replayed_after_a_5xx(rec, fast):
    """/v1/upscale/* carries no idempotency key server-side: a 5xx may hide a charge, so no blind retry."""
    c = _retrying(rec)
    rec.queue((502, {"detail": "bad gateway"}), (202, {"job_id": JID}))
    with pytest.raises(ServerError):
        c.upscale_pro.video("https://example.com/c.mp4")
    assert len(rec.requests) == 1


def test_upscale_pro_jobs_job_cancel_and_wait(client, rec, fast):
    client.upscale_pro.jobs(active=True, kind="video")
    assert rec.last.url.path == "/v1/upscale/pro/jobs"
    assert dict(rec.last.url.params) == {"active": "true", "kind": "video"}
    client.upscale_pro.jobs()
    assert dict(rec.last.url.params) == {}
    client.upscale_pro.job(JID)
    assert (rec.last.method, rec.last.url.path) == ("GET", f"/v1/upscale/pro/jobs/{JID}")
    client.upscale_pro.cancel(JID)
    assert (rec.last.method, rec.last.url.path) == ("POST", f"/v1/upscale/pro/jobs/{JID}/cancel")
    rec.queue((200, {"job_id": JID, "status": "running"}),
              (200, {"job_id": JID, "status": "succeeded", "result": {"url": "u"}}))
    assert client.upscale_pro.wait_for_job(JID)["result"]["url"] == "u"
    rec.queue((200, {"job_id": JID, "status": "failed", "error_code": "engine_failed",
                     "error": "The upscale failed.", "refunded": True}))
    with pytest.raises(FotoHubError) as ei:
        client.upscale_pro.wait_for_job(JID)
    assert ei.value.response_body["refunded"] is True


# ── AI video ───────────────────────────────────────────────────────────────────

def test_ai_video_generate_body(client, rec):
    rec.queue((200, {"job": {"id": JID, "type": "video.generation", "status": "queued"}, "created": True}))
    out = client.ai_video.generate("a lighthouse at dusk", quote_credits=3.0)
    assert out["job"]["id"] == JID
    assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/video/generations")
    body = rec.body()
    assert body["model"] == "fotohub-motion-audio"
    assert body["prompt"] == "a lighthouse at dusk" and body["quote_credits"] == 3.0
    assert UUID_RE.match(body["request_id"])
    # the route forbids unknown fields: only set ones go out
    assert set(body) == {"model", "prompt", "request_id", "quote_credits"}


def test_ai_video_generate_keeps_the_callers_request_id_and_image(client, rec):
    rid = str(uuid.uuid4())
    client.ai_video.generate("a lighthouse", image="photos/u1/first.jpg", seed=7, request_id=rid)
    body = rec.body()
    assert body["request_id"] == rid and body["image"] == "photos/u1/first.jpg" and body["seed"] == 7


def test_ai_video_avatar_body(client, rec):
    consent = {"subject": "self", "version": "2026-10-05.1", "locale": "en"}
    client.ai_video.avatar("photos/u1/me.jpg", consent=consent, mode="script", script="Hello there",
                           language="en", quote_credits=2.5)
    assert rec.last.url.path == "/v1/video/avatar"
    body = rec.body()
    assert body["portrait"] == "photos/u1/me.jpg" and body["consent"] == consent
    assert body["mode"] == "script" and body["script"] == "Hello there" and body["language"] == "en"
    assert body["quote_credits"] == 2.5 and UUID_RE.match(body["request_id"])
    assert "audio" not in body and "voice_ref" not in body


def test_ai_video_dub_body(client, rec):
    rec.queue((200, {"project_id": "p1", "created": True, "jobs": [{"language": "de", "job": {"id": JID}}]}))
    out = client.ai_video.dub("videos/u1/talk.mp4", ["de", "en"], glossary=["FOTOhub"], quote_credits=9.0)
    assert out["jobs"][0]["job"]["id"] == JID
    assert rec.last.url.path == "/v1/video/dub"
    body = rec.body()
    assert body["source"] == "videos/u1/talk.mp4" and body["languages"] == ["de", "en"]
    assert body["glossary"] == ["FOTOhub"] and body["quote_credits"] == 9.0
    assert UUID_RE.match(body["request_id"])
    assert "review" not in body and "voice_consent" not in body


def test_ai_video_get_and_wait(client, rec, fast):
    client.ai_video.get(JID)
    assert (rec.last.method, rec.last.url.path) == ("GET", f"/v1/video/generations/{JID}")
    rec.queue((200, {"id": JID, "status": "running"}),
              (200, {"id": JID, "status": "succeeded", "result": {"assets": [{"url": "u"}]}}))
    assert client.ai_video.wait_for_job(JID)["result"]["assets"][0]["url"] == "u"
    rec.queue((200, {"id": JID, "status": "failed", "error": {"code": "render_failed", "message": "It failed."},
                     "refunded": True}))
    with pytest.raises(FotoHubError) as ei:
        client.ai_video.wait_for_job(JID)
    assert "It failed." in ei.value.message and ei.value.response_body["refunded"] is True


async def test_async_ai_video_and_upscale_pro(aclient, rec):
    rec.queue((200, {"job": {"id": JID}}), (200, {"available": True}))
    out = await aclient.ai_video.generate("a lighthouse")
    assert out["job"]["id"] == JID and UUID_RE.match(rec.body()["request_id"])
    await aclient.upscale_pro.quote("image", width=10, height=10)
    assert rec.last.url.path == "/v1/upscale/pro/quote"


# ── IDA Q Image 2 ─────────────────────────────────────────────────────────────

def test_generate_ida_q2_submits_and_polls(client, rec, fast):
    rec.queue(
        (202, {"model": "ida-q-image-2", "job_id": JID, "status": "queued", "cost_usd": 0.04,
               "currency": "USD", "billing": {"cost_usd": 0.04}}),
        (200, {"job_id": JID, "status": "running", "progress": 40}),
        (200, {"job_id": JID, "status": "completed", "images": ["https://cdn.test/1.png"],
               "metadata": {"model": "ida-q-image-2"}, "result": {"seed": 3, "width": 1024}}),
    )
    out = client.generate_ida_q2("a typographic poster", size_tier="1K", preset="fast", num_images=2,
                                 aspect_ratio="4:5", style="poster", transparent=False)
    first = rec.body(0)
    assert rec.requests[0].url.path == "/v1/ai/generate/image"
    assert first["model"] == "ida-q-image-2" and first["size_tier"] == "1K" and first["preset"] == "fast"
    assert first["num_images"] == 2 and first["aspect_ratio"] == "4:5" and first["style"] == "poster"
    assert first["transparent"] is False
    assert UUID_RE.match(first["job_id"])
    assert rec.requests[1].url.path == f"/v1/ai/generate/image/ida-q-image-2/{JID}"
    assert out["images"] == ["https://cdn.test/1.png"] and out["cost_usd"] == 0.04
    assert out["job_id"] == JID and out["result"]["seed"] == 3


def test_generate_ida_q2_failed_job_raises_with_its_code(client, rec, fast):
    rec.queue(
        (202, {"job_id": JID, "status": "queued"}),
        (200, {"job_id": JID, "status": "failed", "error": "Blocked by the safety filter.",
               "error_code": "SAFETY_BLOCKED"}),
    )
    with pytest.raises(FotoHubError) as ei:
        client.generate_ida_q2("x", job_id=JID)
    assert rec.body(0)["job_id"] == JID
    assert ei.value.code == "SAFETY_BLOCKED" and "safety" in ei.value.message


async def test_async_generate_ida_q2(aclient, rec, fast):
    rec.queue((202, {"job_id": JID, "status": "queued"}),
              (200, {"job_id": JID, "status": "completed", "images": ["u"]}))
    out = await aclient.generate_ida_q2("a poster")
    assert out["images"] == ["u"]
    assert rec.body(0)["size_tier"] == "1K" and rec.body(0)["preset"] == "balanced"
