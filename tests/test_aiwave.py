"""AI Wave 1 namespaces: request shape (method, path, body), idempotency, waiters, errors.

HTTP is mocked at the transport; nothing here reaches a server.
"""

from __future__ import annotations

import asyncio
import re
import uuid

import httpx
import pytest

from fotohub import AuthError, FotoHubError, InsufficientFundsError, RateLimitError, TimeoutError, ValidationError

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
CID = "0c9d3f0e-6a7b-4c1e-9b8a-3e5f2a1d7c60"
JID = "5b0c9a52-5d1e-4f3e-9f3a-70d0c0a1e0b1"


# ── characters ────────────────────────────────────────────────────────────────

def test_character_create_from_photos_sends_consent_and_a_request_id(client, rec):
    rec.queue((200, {"character": {"id": CID}, "job": {"job_id": JID}, "replayed": False}))
    consent = {"subject": "self", "version": "2026-10-05.1", "locale": "en"}
    out = client.characters.create(
        "Ada", source="photos", photos=["photos/u1/a.jpg", "photos/u1/b.jpg"], consent=consent
    )
    assert out["job"]["job_id"] == JID
    assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/characters")
    body = rec.body()
    assert body["name"] == "Ada" and body["kind"] == "face" and body["source"] == "photos"
    assert body["photos"] == ["photos/u1/a.jpg", "photos/u1/b.jpg"]
    assert body["consent"] == consent
    assert UUID_RE.match(body["request_id"])
    assert "prompt" not in body and "seed" not in body and "strength" not in body


def test_character_create_keeps_the_callers_request_id(client, rec):
    rid = str(uuid.uuid4())
    client.characters.create("Max", source="prompt", prompt="a friendly robot", kind="non_face", request_id=rid)
    assert rec.body()["request_id"] == rid
    assert rec.body()["kind"] == "non_face"


def test_character_quote_has_no_request_id(client, rec):
    client.characters.quote("Ada", source="prompt", prompt="a pirate captain")
    assert rec.last.url.path == "/v1/characters/quote"
    assert "request_id" not in rec.body()


def test_character_generate_defaults_to_the_documented_example_model(client, rec):
    client.characters.generate(CID, "Ada on a bicycle", aspect_ratio="3:4")
    assert rec.last.url.path == f"/v1/characters/{CID}/generations"
    body = rec.body()
    assert body["model"] == "seedream-5-0-260128"
    assert body["prompt"] == "Ada on a bicycle" and body["aspect_ratio"] == "3:4"
    assert re.match(r"^[A-Za-z0-9-]{8,64}$", body["request_id"])


def test_character_free_routes_and_verbs(client, rec):
    client.characters.update(CID, name="Ada L.", exclude_assets=["a1"])
    assert (rec.last.method, rec.last.url.path) == ("PATCH", f"/v1/characters/{CID}")
    assert rec.body() == {"name": "Ada L.", "exclude_assets": ["a1"]}
    client.characters.delete(CID)
    assert rec.last.method == "DELETE"
    client.characters.resolve(CID, "seedream-5-0-260128", strength=0.8)
    assert rec.body() == {"model": "seedream-5-0-260128", "strength": 0.8}
    client.characters.jobs(CID, kind="sheet")
    assert rec.last.url.params["kind"] == "sheet" and rec.last.url.params["limit"] == "20"
    client.characters.list(limit=5)
    assert rec.last.url.params["limit"] == "5"
    client.characters.cancel_job(JID)
    assert (rec.last.method, rec.last.url.path) == ("POST", f"/v1/characters/jobs/{JID}/cancel")


def test_unset_query_params_are_not_sent(client, rec):
    client.characters.jobs(CID)
    assert "kind" not in rec.last.url.params


def test_character_wait_returns_the_succeeded_view_and_reports_progress(client, rec, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.queue(
        (200, {"status": "queued", "progress": 0}),
        (200, {"status": "running", "progress": 0.5}),
        (200, {"status": "succeeded", "result": {"url": "https://x/y.png"}}),
    )
    seen = []
    view = client.characters.wait_for_job(JID, on_progress=seen.append)
    assert view["result"]["url"] == "https://x/y.png"
    assert [v["status"] for v in seen] == ["queued", "running", "succeeded"]
    assert len(rec.requests) == 3


def test_wait_raises_with_the_refund_state_on_a_failed_job(client, rec):
    rec.queue((200, {"status": "failed", "error": "sheet_failed", "refunded": True,
                     "refund_message": "You were not charged for this."}))
    with pytest.raises(FotoHubError) as exc:
        client.characters.wait_for_job(JID)
    assert "sheet_failed" in str(exc.value) and "not charged" in str(exc.value)
    assert exc.value.response_body["refunded"] is True


def test_wait_times_out_but_leaves_the_job_running(client, rec, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.default = (200, {"status": "running"})
    with pytest.raises(TimeoutError):
        client.characters.wait_for_job(JID, timeout=0)
    assert not any(r.method == "POST" for r in rec.requests)    # nothing cancelled


# ── product shot ──────────────────────────────────────────────────────────────

def test_product_shot_create_needs_the_consent_version(client, rec):
    with pytest.raises(ValueError, match="consent_version"):
        client.product_shot.create("photos/u1/shoe.jpg", consent_us_processing=True)
    assert rec.requests == []


def test_product_shot_create_body(client, rec):
    rec.queue((200, {"jobs": [{"job_id": JID, "aspect": "1:1"}]}))
    client.product_shot.create(
        "photos/u1/shoe.jpg", preset_id="studio_white", aspects=["1:1", "4:5"],
        consent_us_processing=True, consent_version="2026-10-04",
    )
    assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/aiwave/product-shot/jobs")
    body = rec.body()
    assert body["image_path"] == "photos/u1/shoe.jpg" and body["aspects"] == ["1:1", "4:5"]
    assert body["consent_us_processing"] is True and body["consent_version"] == "2026-10-04"
    assert UUID_RE.match(body["request_id"]) and "tier" not in body


def test_product_shot_without_consent_sends_false_and_lets_the_server_refuse(client, rec):
    rec.queue((422, {"detail": {"error": "consent_required", "message": "Background removal runs in the US"}}))
    with pytest.raises(ValidationError):
        client.product_shot.create("photos/u1/shoe.jpg")
    assert rec.body()["consent_us_processing"] is False


def test_product_shot_quote_and_presets(client, rec):
    client.product_shot.quote("photos/u1/shoe.jpg", tier="premium", scene_model="m1")
    assert rec.last.url.path == "/v1/aiwave/product-shot/quote"
    assert rec.body() == {"image_path": "photos/u1/shoe.jpg", "tier": "premium", "scene_model": "m1"}
    client.product_shot.create_preset("Autumn", prompt="oak table, golden light", kind="seasonal")
    assert rec.last.url.path == "/v1/aiwave/product-shot/presets"
    client.product_shot.delete_preset(CID)
    assert (rec.last.method, rec.last.url.path) == ("DELETE", f"/v1/aiwave/product-shot/presets/{CID}")
    client.product_shot.list_jobs(request_id=CID)
    assert rec.last.url.params["request_id"] == CID


def test_batches_create_defaults_source_to_api_and_requires_consent_version(client, rec):
    items = [{"image_path": "photos/u1/a.jpg", "sku": "A-1"}, {"image_url": "https://shop.test/b.jpg"}]
    with pytest.raises(ValueError):
        client.product_shot.batches.create(items, consent_us_processing=True)
    client.product_shot.batches.create(
        items, name="Autumn", preset_id="studio_white", consent_us_processing=True, consent_version="2026-10-04"
    )
    assert rec.last.url.path == "/v1/aiwave/product-shot/batches"
    body = rec.body()
    assert body["source"] == "api" and body["items"] == items and body["name"] == "Autumn"
    assert UUID_RE.match(body["request_id"])


def test_batch_control_routes(client, rec):
    b = client.product_shot.batches
    b.quote(50, preset_id="studio_white")
    assert rec.body() == {"preset_id": "studio_white", "skus": 50}
    b.items(CID, status="failed", limit=50)
    assert dict(rec.last.url.params) == {"status": "failed", "limit": "50", "offset": "0"}
    b.retry_failed(CID)
    assert rec.last.url.path.endswith(f"/{CID}/retry-failed")
    b.resume(CID, accept_price=True)
    assert rec.body() == {"accept_price": True}
    b.build_archive(CID)
    assert rec.last.method == "POST" and rec.last.url.path.endswith("/archive")
    b.get_archive(CID)
    assert rec.last.method == "GET"
    b.cancel(CID)
    assert rec.last.url.path.endswith("/cancel")


def test_batch_archive_not_ready_is_a_409_error(client, rec):
    rec.queue((409, {"detail": {"error": "archive_not_ready", "message": "The results archive is not ready yet."}}))
    with pytest.raises(FotoHubError) as exc:
        client.product_shot.batches.get_archive(CID)
    assert exc.value.status_code == 409


def test_batch_wait_returns_a_finished_batch_with_failures_instead_of_raising(client, rec, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.queue(
        (200, {"status": "active", "progress": 0.5}),
        (200, {"status": "completed_with_errors", "counts": {"failed": 4, "succeeded": 46}}),
    )
    view = client.product_shot.batches.wait(CID)
    assert view["counts"]["failed"] == 4


def test_batch_wait_keeps_waiting_while_paused(client, rec, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.queue((200, {"status": "paused", "pause": {"reason": "insufficient_funds"}}), (200, {"status": "completed"}))
    assert client.product_shot.batches.wait(CID)["status"] == "completed"


# ── video edit ────────────────────────────────────────────────────────────────

def test_video_edit_session_and_turn(client, rec):
    client.video_edit.create_session("videos/u1/clip.mp4", duration_s=12.5, width=1920, height=1080, fps=30)
    assert rec.last.url.path == "/v1/aiwave/video-edit/sessions"
    assert rec.body() == {"source_path": "videos/u1/clip.mp4",
                          "clip": {"duration_s": 12.5, "width": 1920, "height": 1080, "fps": 30}}
    rec.queue((200, {"turn": {"turn_id": JID, "status": "queued"}}))
    client.video_edit.edit(CID, "make the sky a sunset", selection=(2, 6.5), likeness_ack=True)
    assert rec.last.url.path == f"/v1/aiwave/video-edit/sessions/{CID}/turns"
    body = rec.body()
    assert body["selection"] == {"start_s": 2.0, "end_s": 6.5} and body["likeness_ack"] is True
    assert UUID_RE.match(body["request_id"]) and "extend_seconds" not in body


def test_video_edit_accepts_a_selection_mapping_and_quote_never_has_a_request_id(client, rec):
    client.video_edit.quote(CID, "slow motion", selection={"start_s": 1, "end_s": 3})
    assert rec.last.url.path == f"/v1/aiwave/video-edit/sessions/{CID}/quote"
    assert rec.body() == {"instruction": "slow motion", "selection": {"start_s": 1.0, "end_s": 3.0}}


def test_video_extend_has_no_selection(client, rec):
    client.video_edit.edit(CID, "continue the scene", extend_seconds=5)
    assert "selection" not in rec.body() and rec.body()["extend_seconds"] == 5


def test_video_revert_to_the_original_sends_an_explicit_null(client, rec):
    client.video_edit.revert(CID)
    assert rec.body() == {"turn_id": None}
    client.video_edit.revert(CID, JID)
    assert rec.body() == {"turn_id": JID}
    client.video_edit.undo(CID)
    assert rec.last.url.path.endswith("/undo")
    client.video_edit.archive_session(CID)
    assert rec.last.method == "DELETE"


def test_video_wait_for_turn(client, rec, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.queue((200, {"status": "running"}), (200, {"status": "succeeded", "result": {"url": "https://x/v.mp4"}}))
    assert client.video_edit.wait_for_turn(JID)["result"]["url"] == "https://x/v.mp4"
    assert rec.last.url.path == f"/v1/aiwave/video-edit/turns/{JID}"


def test_insufficient_funds_on_a_turn_carries_the_amounts(client, rec):
    rec.queue((402, {"detail": {"error": "insufficient_funds", "message": "short", "required_usd": 1.25,
                                "balance_usd": 0.5, "shortfall_usd": 0.75, "topup_url": "https://fotohub.app/top-up"}}))
    with pytest.raises(InsufficientFundsError) as exc:
        client.video_edit.edit(CID, "x", selection=(0, 1))
    assert exc.value.required_usd == 1.25 and exc.value.shortfall_usd == 0.75


# ── music edit ────────────────────────────────────────────────────────────────

def test_music_section_edit(client, rec):
    client.music_edit.edit_section(
        "audio/u1/song.flac", source_duration_s=180, start_s=60, end_s=80,
        instruction="make the chorus more energetic", lyrics="la la la", crossfade_ms=100,
    )
    assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/aiwave/music-edit/section")
    body = rec.body()
    assert body["start_s"] == 60 and body["crossfade_ms"] == 100 and body["lyrics"] == "la la la"
    assert "snap_to_beat" not in body            # the server's default (true) applies
    assert UUID_RE.match(body["request_id"])


def test_music_lyrics_edit_and_quote(client, rec):
    client.music_edit.edit_lyrics(
        "audio/u1/song.flac", source_duration_s=180,
        line={"start_s": 30, "end_s": 34, "text": "old words"}, new_text="new words", parent_job_id=JID,
    )
    assert rec.last.url.path == "/v1/aiwave/music-edit/lyrics"
    assert rec.body()["line"] == {"start_s": 30, "end_s": 34, "text": "old words"}
    assert rec.body()["parent_job_id"] == JID
    client.music_edit.quote_lyrics("audio/u1/song.flac", source_duration_s=180,
                                   line={"start_s": 30, "end_s": 34}, new_text="new words")
    assert rec.last.url.path == "/v1/aiwave/music-edit/lyrics/quote" and "request_id" not in rec.body()


def test_soundtrack_for_a_video_with_sfx_off_by_default(client, rec):
    client.music_edit.soundtrack("videos/u1/clip.mp4", media_duration_s=40, duration_s=30, mood="calm")
    assert rec.last.url.path == "/v1/aiwave/music-edit/soundtrack"
    body = rec.body()
    assert "sfx" not in body and body["mood"] == "calm" and UUID_RE.match(body["request_id"])
    client.music_edit.soundtrack("photos/u1/pic.jpg", sfx=False)
    assert rec.body()["sfx"] is False                       # an explicit False is still sent


def test_music_jobs_filter_and_wait_failure_message(client, rec):
    client.music_edit.jobs(op="section")
    assert rec.last.url.params["op"] == "section"
    rec.queue((200, {"status": "failed", "error": {"code": "engine_failed", "message": "The music engine could not finish this edit."},
                     "refunded": True, "refund_message": "You were not charged for this edit."}))
    with pytest.raises(FotoHubError) as exc:
        client.music_edit.wait_for_job(JID)
    assert "could not finish" in str(exc.value) and "not charged" in str(exc.value)


# ── errors common to every namespace ──────────────────────────────────────────

def test_feature_disabled_is_an_auth_error(client, rec):
    rec.queue((403, {"detail": {"error": "feature_disabled", "feature": "product_shot", "reason": "tier_required"}}))
    with pytest.raises(AuthError) as exc:
        client.product_shot.config()
    assert exc.value.status_code == 403


def test_ceiling_exceeded_retries_the_429_after_the_retry_after_header(rec, monkeypatch):
    from fotohub import FotoHub

    c = FotoHub(api_key="k", base_url="https://api.test", max_retries=2)
    c._client = httpx.Client(base_url=c.base_url, headers=c._headers(), transport=httpx.MockTransport(rec.handler))
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    rec.queue((429, {"detail": {"error": "ceiling_exceeded", "message": "slow down"}}, {"retry-after": "7"}),
              (200, {"jobs": []}))
    assert c.product_shot.list_jobs() == {"jobs": []}
    assert slept and slept[0] >= 7
    rec.queue((429, {"detail": {"error": "ceiling_exceeded"}}, {"retry-after": "1"}),
              (429, {"detail": {"error": "ceiling_exceeded"}}, {"retry-after": "1"}))
    with pytest.raises(RateLimitError):
        c.product_shot.list_jobs()


def test_retry_of_a_charged_call_resends_the_same_request_id(rec, monkeypatch):
    from fotohub import FotoHub

    c = FotoHub(api_key="k", base_url="https://api.test", max_retries=2)
    c._client = httpx.Client(base_url=c.base_url, headers=c._headers(), transport=httpx.MockTransport(rec.handler))
    monkeypatch.setattr("time.sleep", lambda s: None)
    rec.queue((503, {"detail": {"error": "backend_unavailable"}}), (200, {"jobs": [{"job_id": JID}]}))
    c.product_shot.create("photos/u1/x.jpg", consent_us_processing=True, consent_version="v")
    assert len(rec.requests) == 2
    assert rec.body(0)["request_id"] == rec.body(1)["request_id"]


def test_auth_headers_are_sent(client, rec):
    client.music_edit.config()
    assert rec.last.headers["authorization"] == "Bearer fh_test_key"


# ── the async client uses the very same namespaces ────────────────────────────

def test_async_client_returns_awaitables_with_the_same_requests(aclient, rec):
    async def run():
        rec.queue((200, {"character": {"id": CID}, "job": {"job_id": JID}}))
        out = await aclient.characters.create("Ada", source="prompt", prompt="a pirate captain")
        assert out["job"]["job_id"] == JID
        assert (rec.last.method, rec.last.url.path) == ("POST", "/v1/characters")
        assert UUID_RE.match(rec.body()["request_id"])
        await aclient.product_shot.batches.cancel(CID)
        assert rec.last.url.path.endswith(f"/{CID}/cancel")
        await aclient.aclose() if hasattr(aclient, "aclose") else None

    asyncio.run(run())


def test_async_wait_polls_without_blocking_the_loop(aclient, rec, monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    async def run():
        rec.queue((200, {"status": "running"}), (200, {"status": "succeeded", "result": {"url": "u"}}))
        view = await aclient.video_edit.wait_for_turn(JID, poll_interval=2.0)
        assert view["result"]["url"] == "u" and slept == [2.0]
        rec.queue((200, {"status": "cancelled", "refunded": True}))
        with pytest.raises(FotoHubError):
            await aclient.music_edit.wait_for_job(JID)

    asyncio.run(run())
