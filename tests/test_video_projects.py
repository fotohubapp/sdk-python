"""Video timeline API (/v1/video/projects): every method runs against the sync and the async client."""

from __future__ import annotations

import inspect
import json

import pytest

from fotohub import (
    AsyncFotoHub,
    AuthError,
    FotoHub,
    InsufficientFundsError,
    RateLimitError,
    SaveConflictError,
    ValidationError,
    VideoJobFailedError,
    VideoJobTimeoutError,
)
from fotohub.exceptions import FotoHubError

BASE = "https://api.test"
PID = "3f1c2a4e-7b1d-4c8e-9a55-0d6f1e2b3c4d"
JOB = "9b2e6d10-1c3a-4f7e-8d21-5a6b7c8d9e0f"


@pytest.fixture(params=["sync", "async"])
async def api(request):
    """A client bound to `call(name, *args, **kw)`, which awaits when the method is a coroutine."""
    if request.param == "sync":
        client = FotoHub(api_key="fh_test", base_url=BASE, max_retries=1)
    else:
        client = AsyncFotoHub(api_key="fh_test", base_url=BASE, max_retries=1)

    async def call(name, *args, **kw):
        result = getattr(client, name)(*args, **kw)
        return await result if inspect.isawaitable(result) else result

    call.is_async = request.param == "async"
    yield call
    closing = client.close()
    if inspect.isawaitable(closing):
        await closing


INSERT_CLIP = {"op": "insertClip", "ref": "intro", "at": {"trackId": "t1", "time": 0},
               "clip": {"type": "video", "assetId": "a1"}}


def body_of(request) -> dict:
    return json.loads(request.content)


def envelope(code, message="boom", details=None):
    err = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"error": err}


# --- create / read / delete ---


async def test_create_sends_camel_case_and_an_idempotency_key(api, httpx_mock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/video/projects", status_code=201,
        json={"projectId": PID, "saveRev": 1, "unplacedMedia": [{"assetId": "a2", "reason": "too-long"}],
              "editorUrl": f"https://fotohub.app/fh/editor/lite/{PID}"},
    )
    out = await api(
        "create_video_project", title="Promo", aspect="9:16", place_media="none", template="tpl-1",
        media=[{"url": "https://cdn.example.com/a.mp4", "kind": "video"},
               {"storage_path": "videos/u1/x.mp4", "name": "x"}],
    )
    assert out["projectId"] == PID and out["unplacedMedia"][0]["assetId"] == "a2"
    req = httpx_mock.get_requests()[0]
    assert body_of(req) == {
        "title": "Promo", "aspect": "9:16", "placeMedia": "none", "template": {"id": "tpl-1"},
        "media": [{"url": "https://cdn.example.com/a.mp4", "kind": "video"},
                  {"storagePath": "videos/u1/x.mp4", "name": "x"}],
    }
    assert req.headers["X-Idempotency-Key"]


async def test_create_idempotency_key_is_overridable(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects", status_code=201, json={"projectId": PID})
    await api("create_video_project", idempotency_key="my-key-1")
    assert httpx_mock.get_requests()[0].headers["X-Idempotency-Key"] == "my-key-1"


async def test_create_media_blocked_exposes_the_error_code(api, httpx_mock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE}/v1/video/projects", status_code=422,
        json=envelope("media-blocked", "media url rejected: private address"),
    )
    with pytest.raises(ValidationError) as exc:
        await api("create_video_project", media=[{"url": "https://10.0.0.1/a.mp4"}])
    assert exc.value.code == "media-blocked"
    assert "private address" in exc.value.message


async def test_get_project_and_include_doc(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/projects/{PID}", json={"projectId": PID, "saveRev": 4})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/projects/{PID}?include=doc", json={"projectId": PID, "doc": {}})
    assert (await api("get_video_project", PID))["saveRev"] == 4
    assert "doc" in await api("get_video_project", PID, include_doc=True)


async def test_foreign_or_missing_project_is_404_not_found(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/projects/{PID}", status_code=404,
                            json=envelope("not-found", "project not found"))
    with pytest.raises(FotoHubError) as exc:
        await api("get_video_project", PID)
    assert exc.value.status_code == 404 and exc.value.code == "not-found"


async def test_list_and_delete(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/projects?limit=5", json={"projects": [{"projectId": PID}]})
    httpx_mock.add_response(method="DELETE", url=f"{BASE}/v1/video/projects/{PID}", json={"ok": True})
    assert (await api("list_video_projects", limit=5))["projects"][0]["projectId"] == PID
    assert (await api("delete_video_project", PID))["ok"] is True


# --- ops ---


async def test_apply_ops_body_and_no_idempotency_key(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops",
                            json={"ok": True, "rolledBack": False, "saveRev": 6, "versionSaved": True})
    ops = [INSERT_CLIP]
    out = await api("apply_video_ops", PID, ops, dry_run=True, expected_save_rev=5, label="cut intro")
    assert out["saveRev"] == 6
    req = httpx_mock.get_requests()[0]
    assert body_of(req) == {"ops": ops, "dryRun": True, "expectedSaveRev": 5, "label": "cut intro"}
    assert "X-Idempotency-Key" not in req.headers  # free, and a 409 here is a real conflict


async def test_apply_ops_defaults_send_only_ops(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops", json={"ok": True})
    await api("apply_video_ops", PID, [INSERT_CLIP])
    assert body_of(httpx_mock.get_requests()[0]) == {"ops": [INSERT_CLIP]}


async def test_rolled_back_batch_is_a_result_not_an_exception(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops",
                            json={"ok": False, "rolledBack": True, "violations": [{"code": "overlap"}], "saveRev": 5})
    out = await api("apply_video_ops", PID, [INSERT_CLIP])
    assert out["rolledBack"] is True and out["violations"][0]["code"] == "overlap"


async def test_save_conflict_raises_once_and_carries_the_current_rev(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops", status_code=409,
                            json=envelope("save-conflict", "project changed", {"currentSaveRev": 9}))
    with pytest.raises(SaveConflictError) as exc:
        await api("apply_video_ops", PID, [INSERT_CLIP], expected_save_rev=5)
    assert exc.value.current_save_rev == 9 and exc.value.code == "save-conflict"
    assert exc.value.status_code == 409
    assert len(httpx_mock.get_requests()) == 1


async def test_save_conflict_is_not_retried_even_with_retries_enabled(httpx_mock):
    client = FotoHub(api_key="k", base_url=BASE, max_retries=3)
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops", status_code=409,
                            json=envelope("save-conflict", "project changed", {"currentSaveRev": 2}))
    with pytest.raises(SaveConflictError):
        client.apply_video_ops(PID, [INSERT_CLIP])
    assert len(httpx_mock.get_requests()) == 1


async def test_invalid_ops_is_a_validation_error_with_details(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/ops", status_code=422,
                            json=envelope("invalid-ops", "op 0 invalid", {"path": "/ops/0/op"}))
    with pytest.raises(ValidationError) as exc:
        await api("apply_video_ops", PID, [{"op": "nope"}])
    assert exc.value.code == "invalid-ops" and exc.value.details == {"path": "/ops/0/op"}


async def test_digest_lint_and_catalog(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/digest", json={"clips": []})
    for _ in range(2):
        httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/lint", json={"findings": []})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/ops/catalog", json={"ops": {}})
    await api("digest_video_project", PID, clip_ids=["c1"], view="clips")
    await api("lint_video_project", PID, rules=["gap"], severity="warn")
    await api("lint_video_project", PID, severity=["error", "warn"])
    assert (await api("get_video_ops_catalog")) == {"ops": {}}
    digest, lint, lint_list, _ = httpx_mock.get_requests()
    assert body_of(digest) == {"clipIds": ["c1"], "view": "clips"}
    # The timeline service requires an array; a single string is wrapped.
    assert body_of(lint) == {"rules": ["gap"], "severity": ["warn"]}
    assert body_of(lint_list) == {"severity": ["error", "warn"]}
    assert "X-Idempotency-Key" not in digest.headers and "X-Idempotency-Key" not in lint.headers


async def test_lint_unavailable_is_a_result_with_a_warning(api, httpx_mock):
    # The checker not being deployed is a 200 with available:false, not an error.
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/lint",
                            json={"saveRev": 4, "findings": [], "counts": {"error": 0, "warn": 0, "info": 0},
                                  "available": False, "warnings": ["lint-unavailable"]})
    out = await api("lint_video_project", PID)
    assert out["available"] is False and out["warnings"] == ["lint-unavailable"]


async def test_lint_route_missing_is_501_lint_unavailable(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/lint", status_code=501,
                            json=envelope("lint-unavailable", "lint is not available yet"))
    with pytest.raises(FotoHubError) as exc:
        await api("lint_video_project", PID)
    assert exc.value.code == "lint-unavailable" and exc.value.status_code == 501


# --- capture ---


CAPTURE_DONE = {
    "jobId": JOB, "kind": "capture", "status": "completed", "progress": 100,
    "frames": [{"index": 0, "t": 1.0, "actualT": 1.0, "label": "0:01", "sheet": 0, "x": 0, "y": 0, "w": 640, "h": 360}],
    "sheets": [{"url": "https://s/sheet.jpg", "width": 1280, "height": 720}],
    "missing": [],
}


async def test_capture_body_defaults_and_idempotency_key(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=202,
                            json={"jobId": JOB, "status": "queued"})
    out = await api("capture_video_project", PID, times=[1.0, 2.5], sheet={"max_cells": 4, "max_edge": 512})
    assert out == {"jobId": JOB, "status": "queued"}
    req = httpx_mock.get_requests()[0]
    assert body_of(req) == {"times": [1.0, 2.5], "width": 640, "sheet": {"maxCells": 4, "maxEdge": 512}}
    assert req.headers["X-Idempotency-Key"]


async def test_capture_cuts_true_only_when_requested(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=202, json={"jobId": JOB})
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=202, json={"jobId": JOB})
    await api("capture_video_project", PID, cuts=True, width=320)
    await api("capture_video_project", PID, count=6)
    first, second = (body_of(r) for r in httpx_mock.get_requests())
    assert first == {"cuts": True, "width": 320}
    assert second == {"count": 6, "width": 640}


async def test_capture_wait_returns_frames_sheets_and_missing(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=202,
                            json={"jobId": JOB, "status": "queued"})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": "running"})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json=CAPTURE_DONE)
    out = await api("capture_video_project", PID, times=[1.0], wait=True, max_wait=60)
    assert out["frames"][0]["actualT"] == 1.0 and out["sheets"][0]["width"] == 1280 and out["missing"] == []


@pytest.fixture(autouse=True)
def instant_sleep(monkeypatch):
    """Polling waits are real `time.sleep` / `asyncio.sleep`: make them instant, keep the calls observable."""
    import asyncio
    import time

    real_async_sleep = asyncio.sleep

    async def fast_async(_delay, *a, **k):
        await real_async_sleep(0)

    monkeypatch.setattr(time, "sleep", lambda _d: None)
    monkeypatch.setattr(asyncio, "sleep", fast_async)


# --- render ---


async def test_render_defaults_and_full_body(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=202,
                            json={"jobId": JOB, "status": "queued", "billedMinutes": 0.5})
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=202, json={"jobId": JOB})
    out = await api("render_video_project", PID)
    assert out["billedMinutes"] == 0.5
    await api("render_video_project", PID, format="webm", quality="ultra", resolution="4k", codec="h265", fps=30,
              bitrate="8M", time_range=(2, 10), content_credentials=True, idempotency_key="render-1")
    default, full = httpx_mock.get_requests()
    assert body_of(default) == {"format": "mp4", "quality": "high"}
    assert default.headers["X-Idempotency-Key"]
    assert body_of(full) == {"format": "webm", "quality": "ultra", "resolution": "4k", "codec": "h265", "fps": 30,
                             "bitrate": "8M", "range": {"in": 2, "out": 10}, "contentCredentials": True}
    assert full.headers["X-Idempotency-Key"] == "render-1"


async def test_render_wait_returns_the_finished_job(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=202, json={"jobId": JOB})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": "queued", "queuePosition": 1})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": "completed", "outputUrl": "https://s/out.mp4"})
    out = await api("render_video_project", PID, wait=True)
    assert out["outputUrl"] == "https://s/out.mp4"


async def test_render_wait_failed_job_raises_with_refund_state(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=202, json={"jobId": JOB})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}",
                            json={"jobId": JOB, "status": "failed", "error": "encoder crashed", "reason": "engine", "code": "save-conflict", "refunded": True})
    with pytest.raises(VideoJobFailedError) as exc:
        await api("render_video_project", PID, wait=True)
    assert exc.value.job_id == JOB and exc.value.refunded is True and exc.value.reason == "engine"
    assert exc.value.code == "save-conflict"
    assert "encoder crashed" in exc.value.message


async def test_rate_limit_exposes_code_and_retry_after(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=429,
                            headers={"Retry-After": "42"}, json=envelope("rate-limited", "slow down"))
    with pytest.raises(RateLimitError) as exc:
        await api("render_video_project", PID)
    assert exc.value.retry_after == 42 and exc.value.code == "rate-limited"


async def test_rate_limit_retry_after_from_envelope_details(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=429,
                            json=envelope("rate-limited", "slow down", {"retryAfterSeconds": 180}))
    with pytest.raises(RateLimitError) as exc:
        await api("capture_video_project", PID, count=3)
    assert exc.value.retry_after == 180


async def test_payment_required_403_and_402(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=403,
                            json=envelope("payment-required", "wallet is empty"))
    with pytest.raises(InsufficientFundsError) as exc403:
        await api("render_video_project", PID)
    assert exc403.value.code == "payment-required" and exc403.value.status_code == 403

    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/capture", status_code=402,
                            json=envelope("payment-required", "insufficient funds", {"required_usd": 0.05, "balance_usd": 0.01}))
    with pytest.raises(InsufficientFundsError) as exc402:
        await api("capture_video_project", PID, count=3)
    assert exc402.value.code == "payment-required" and exc402.value.required_usd == 0.05
    assert exc402.value.balance_usd == 0.01


async def test_media_not_found_code(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects", status_code=422,
                            json=envelope("media-not-found", "storage path not found"))
    with pytest.raises(ValidationError) as exc:
        await api("create_video_project", media=[{"storage_path": "videos/u/none.mp4"}])
    assert exc.value.code == "media-not-found"


# --- jobs ---


async def test_wait_for_video_job_polls_until_completed(api, httpx_mock):
    for status in ("queued", "running", "running"):
        httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": status})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": "completed"})
    out = await api("wait_for_video_job", JOB, poll_interval=0.01)
    assert out["status"] == "completed" and len(httpx_mock.get_requests()) == 4


async def test_wait_for_video_job_cancelled_raises(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}",
                            json={"jobId": JOB, "status": "cancelled", "reason": "stale", "refunded": False})
    with pytest.raises(VideoJobFailedError) as exc:
        await api("wait_for_video_job", JOB)
    assert exc.value.reason == "stale" and exc.value.refunded is False


async def test_wait_for_video_job_times_out_with_the_job_id(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", json={"jobId": JOB, "status": "running"}, is_reusable=True)
    with pytest.raises(VideoJobTimeoutError) as exc:
        await api("wait_for_video_job", JOB, poll_interval=1.0, timeout=0.0)
    assert exc.value.job_id == JOB


async def test_get_video_job_of_someone_else_is_404(api, httpx_mock):
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/jobs/{JOB}", status_code=404, json=envelope("not-found", "job not found"))
    with pytest.raises(FotoHubError) as exc:
        await api("get_video_job", JOB)
    assert exc.value.code == "not-found"


# --- auto-edit ---


async def test_auto_edit_body_and_idempotency_key(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/auto-edit", status_code=202,
                            json={"jobId": JOB, "kind": "auto_edit", "status": "queued"})
    out = await api("auto_edit_video_project", PID, style="podcast", toggles={"captions": True}, language="pl",
                    ai_budget_usd=2.5, auto_apply=False, mode="cut")
    assert out["kind"] == "auto_edit"
    req = httpx_mock.get_requests()[0]
    assert body_of(req) == {"style": "podcast", "toggles": {"captions": True}, "language": "pl",
                            "aiBudgetUsd": 2.5, "autoApply": False, "mode": "cut"}
    assert req.headers["X-Idempotency-Key"]


# --- analysis ---


async def test_detect_scenes_silence_beats_bodies(api, httpx_mock):
    for path in ("detect-scenes", "detect-silence", "detect-beats"):
        httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/{path}", json={"ok": path})
    await api("detect_video_scenes", url="https://cdn.example.com/a.mp4", threshold=0.5)
    await api("detect_video_silence", project_id=PID, media_id="a1", noise_floor_db=-40, min_silence_duration=0.5)
    await api("detect_video_beats", url="https://cdn.example.com/a.mp3")
    scenes, silence, beats = (body_of(r) for r in httpx_mock.get_requests())
    assert scenes == {"url": "https://cdn.example.com/a.mp4", "threshold": 0.5, "minSceneDuration": 0.5}
    assert silence == {"projectId": PID, "mediaId": "a1", "noiseFloorDb": -40, "minSilenceDuration": 0.5}
    assert beats == {"url": "https://cdn.example.com/a.mp3"}


async def test_transcribe_start_and_status(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/transcribe", json={"jobId": "tr_12345678"})
    httpx_mock.add_response(method="GET", url=f"{BASE}/v1/video/transcribe/tr_12345678", json={"status": "completed", "result": {"text": "hi"}})
    started = await api("transcribe_video", url="https://cdn.example.com/a.mp3", language="pl", hotwords=["FOTOhub"])
    done = await api("get_video_transcription", started["jobId"])
    assert body_of(httpx_mock.get_requests()[0]) == {"url": "https://cdn.example.com/a.mp3", "language": "pl", "hotwords": ["FOTOhub"]}
    assert done["result"]["text"] == "hi"


async def test_plain_403_is_still_an_auth_error(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=403,
                            json=envelope("forbidden", "no access"))
    with pytest.raises(AuthError):
        await api("render_video_project", PID)


# --- retry safety ---


@pytest.fixture(params=["sync", "async"])
async def retrying(request):
    """Like `api` but with the default retry budget, to prove what is and is not retried."""
    client = FotoHub(api_key="k", base_url=BASE, max_retries=3) if request.param == "sync" \
        else AsyncFotoHub(api_key="k", base_url=BASE, max_retries=3)

    async def call(name, *args, **kw):
        result = getattr(client, name)(*args, **kw)
        return await result if inspect.isawaitable(result) else result

    yield call
    closing = client.close()
    if inspect.isawaitable(closing):
        await closing


OPS_URL = f"{BASE}/v1/video/projects/{PID}/ops"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_ops_without_a_rev_is_not_retried_on_5xx(retrying, httpx_mock, status):
    # The service may have saved before answering: a replay would apply the batch twice.
    httpx_mock.add_response(method="POST", url=OPS_URL, status_code=status,
                            json=envelope("timeline-timeout", "no answer", {"retryable": True}))
    httpx_mock.add_response(method="POST", url=OPS_URL, json={"ok": True, "saveRev": 7})
    with pytest.raises(FotoHubError):
        await retrying("apply_video_ops", PID, [INSERT_CLIP])
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_ops_without_a_rev_is_not_retried_on_timeout(retrying, httpx_mock):
    import httpx

    httpx_mock.add_exception(httpx.ReadTimeout("slow"), method="POST", url=OPS_URL)
    httpx_mock.add_response(method="POST", url=OPS_URL, json={"ok": True, "saveRev": 7})
    with pytest.raises(FotoHubError):
        await retrying("apply_video_ops", PID, [INSERT_CLIP])
    assert len(httpx_mock.get_requests()) == 1


async def test_ops_with_a_rev_is_retried_on_5xx(retrying, httpx_mock):
    # A replay of a batch that did save answers 409 save-conflict instead of applying it again.
    httpx_mock.add_response(method="POST", url=OPS_URL, status_code=504, json=envelope("timeline-timeout", "no answer"))
    httpx_mock.add_response(method="POST", url=OPS_URL, json={"ok": True, "saveRev": 6})
    out = await retrying("apply_video_ops", PID, [INSERT_CLIP], expected_save_rev=5)
    assert out["saveRev"] == 6 and len(httpx_mock.get_requests()) == 2


async def test_ops_with_a_rev_replay_after_a_saved_batch_raises_save_conflict(retrying, httpx_mock):
    httpx_mock.add_response(method="POST", url=OPS_URL, status_code=504, json=envelope("timeline-timeout", "no answer"))
    httpx_mock.add_response(method="POST", url=OPS_URL, status_code=409,
                            json=envelope("save-conflict", "changed", {"currentSaveRev": 6}))
    with pytest.raises(SaveConflictError) as exc:
        await retrying("apply_video_ops", PID, [INSERT_CLIP], expected_save_rev=5)
    assert exc.value.current_save_rev == 6 and len(httpx_mock.get_requests()) == 2


async def test_ops_is_still_retried_on_429(retrying, httpx_mock):
    # Rejected before it ran: always safe to repeat.
    httpx_mock.add_response(method="POST", url=OPS_URL, status_code=429, headers={"Retry-After": "0"},
                            json=envelope("rate-limited", "slow down"))
    httpx_mock.add_response(method="POST", url=OPS_URL, json={"ok": True, "saveRev": 6})
    await retrying("apply_video_ops", PID, [INSERT_CLIP])
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.parametrize("code", ["save-conflict", "project-limit", "draft-limit"])
async def test_keyed_409_with_another_code_is_sent_once(retrying, httpx_mock, code):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects", status_code=409, json=envelope(code, "no"))
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects", status_code=201, json={"projectId": PID})
    with pytest.raises(FotoHubError) as exc:
        await retrying("create_video_project", media=[{"storage_path": "videos/u/a.mp4"}])
    assert exc.value.code == code and len(httpx_mock.get_requests()) == 1


async def test_keyed_409_idempotency_in_progress_is_retried(retrying, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=409,
                            headers={"Retry-After": "0"}, json=envelope("idempotency-in-progress", "still running"))
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/render", status_code=202, json={"jobId": JOB})
    out = await retrying("render_video_project", PID)
    reqs = httpx_mock.get_requests()
    assert out["jobId"] == JOB and len(reqs) == 2
    assert reqs[0].headers["X-Idempotency-Key"] == reqs[1].headers["X-Idempotency-Key"]


# --- auto-edit apply (provisional until the server route exists) ---


async def test_apply_video_auto_edit_commits_the_draft(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/auto-edit/{JOB}/apply",
                            json={"ok": True, "saveRev": 8})
    out = await api("apply_video_auto_edit", PID, JOB, expected_save_rev=7)
    req = httpx_mock.get_requests()[0]
    assert out["saveRev"] == 8 and body_of(req) == {"expectedSaveRev": 7}
    assert req.headers["X-Idempotency-Key"]


async def test_apply_video_auto_edit_conflict_is_typed(api, httpx_mock):
    httpx_mock.add_response(method="POST", url=f"{BASE}/v1/video/projects/{PID}/auto-edit/{JOB}/apply", status_code=409,
                            json=envelope("save-conflict", "changed", {"currentSaveRev": 9}))
    with pytest.raises(SaveConflictError) as exc:
        await api("apply_video_auto_edit", PID, JOB)
    assert exc.value.current_save_rev == 9


def test_public_docstrings_use_the_real_op_shape():
    for cls in (FotoHub, AsyncFotoHub):
        doc = inspect.getdoc(cls.apply_video_ops)
        assert '"op": "insertClip"' in doc and "addClip" not in doc and '"type"' not in doc
        assert "provisional" in inspect.getdoc(cls.auto_edit_video_project).lower()
        assert "provisional" in inspect.getdoc(cls.apply_video_auto_edit).lower()
