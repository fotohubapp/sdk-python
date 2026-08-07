<p align="center">
  <img src="https://static.fotohub.app/brand/fotohub-logo-dark.png" alt="FOTOhub" width="280" />
</p>

<p align="center">
  <strong>Official Python SDK for the FOTOhub AI Platform</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/fotohub/"><img src="https://img.shields.io/pypi/v/fotohub?color=blue&label=PyPI" alt="PyPI Version" /></a>
  <a href="https://pypi.org/project/fotohub/"><img src="https://img.shields.io/pypi/pyversions/fotohub" alt="Python Versions" /></a>
  <a href="https://github.com/fotohubapp/sdk-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fotohubapp/sdk-python" alt="License" /></a>
  <a href="https://pypi.org/project/fotohub/"><img src="https://img.shields.io/pypi/dm/fotohub?color=green" alt="Downloads" /></a>
  <a href="https://docs.fotohub.app/sdk/python"><img src="https://img.shields.io/badge/docs-fotohub.app-blue" alt="Documentation" /></a>
</p>

<p align="center">
  Generate images, videos, music, and chat with LLMs — all through a single, unified Python client.<br/>
  Supports 80+ AI models from 10+ providers with built-in credit management.
</p>

---

## Features

- **Image Generation** — 25+ models including SeedDream 5.0, Flux, Imagen, Gemini, GPT Image, and more
- **IDA Q 1.0** — FOTOhub's proprietary self-hosted image model, submitted and polled for you
- **Video Generation** — Veo, Seedance and more; the call blocks and returns the finished URL
- **Music, SFX and Speech** — AI-generated music, sound effects and TTS
- **Virtual Try-On** — dress a person photo in a garment, or a full top + bottom outfit in one call
- **Chat / LLM** — OpenAI-shaped chat completions, plus premium Claude-class models
- **Gabriel AI** — routes a natural-language request to the right feature and model
- **3D Generation** — image-to-mesh and text-to-mesh jobs with polling helpers
- **Stability Tools** — upscale, erase, inpaint, outpaint, recolor, style transfer
- **Billing** — balance, credits, pricing, top-ups, transactions and invoices in USD
- **Webhooks** — register, test and inspect delivery logs
- **Translation** — multi-language translation
- **Sync + Async** — both synchronous and asynchronous clients included
- **Automatic Retries** — exponential backoff with configurable retry logic
- **Fully Typed** — complete type annotations and a `py.typed` marker

## Installation

```bash
pip install fotohub
```

Requires Python 3.9 or higher.

## Quick Start

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

# Generate an image
result = client.generate_image(prompt="A mountain landscape at golden hour")
print(result["images"][0])
```

Every method returns the API's JSON as a plain `dict` — index it, don't use
attribute access.

## Authentication

Get your API key from [fotohub.app/settings/api](https://fotohub.app/settings/api).

```python
from fotohub import FotoHub

# Option 1: Pass directly
client = FotoHub(api_key="fh_live_...")

# Option 2: Environment variable
# export FOTOHUB_API_KEY=fh_live_...
client = FotoHub()
```

The SDK authenticates via both `Authorization: Bearer` and `x-api-key` headers.

## Usage Examples

### Image Generation

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

# Basic generation (default: seedream-5-0-260128)
result = client.generate_image(
    prompt="A serene Japanese garden with cherry blossoms",
)
print(result["images"][0])

# Advanced options
result = client.generate_image(
    prompt="Cyberpunk cityscape at night, neon reflections on wet streets",
    model="flux-2-pro",
    width=1280,
    height=720,
    num_images=2,
    negative_prompt="blurry, low quality",
    seed=42,
)

for url in result["images"]:
    print(url)
print(f"{result['credits_used']} credits, ${result['billing']['usd_charged']}")
```

`guidance_scale` and `steps` are not accepted by this endpoint — passing them
raises `TypeError` in the SDK. Prefer `aspect_ratio` over manual width/height;
the API picks dimensions the chosen model actually supports.

**IDA Q 1.0** (FOTOhub's own model) runs on a single-GPU queue, so it has its
own method that submits and polls for you — 30 s at `1K`, up to ~3.5 min at `2K`:

```python
result = client.generate_ida_q(
    prompt="Portret kobiety w świetle porannym",   # any language
    aspect_ratio="4:3",
    image_size="1.5K",
)
print(result["images"][0])
```

### Video Generation

Video generation is **synchronous**: the request stays open until the render
finishes, so the returned dict already carries the finished `video_url`. There is
no job to poll. Raise `timeout` on the client if your model is a slow one.

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...", timeout=600.0)

result = client.generate_video(
    prompt="A drone shot flying over a tropical beach at sunrise",
    model="veo-3.1-generate-001",
    duration=5,
    aspect_ratio="16:9",
)
print(result["video_url"])
print(f"{result['credits_used']} credits")
```

`wait_for_video()` still exists but is deprecated — it emits a
`DeprecationWarning` and returns the dict you pass it unchanged.

**Image-to-Video:**

```python
result = client.generate_video(
    prompt="Camera slowly zooms in, subtle parallax motion",
    image_url="https://example.com/photo.jpg",
)
```

### Seedance — long clips and video editing

Seedance is the one video family that runs **asynchronously**: the API answers
`202` with a `job_id` instead of a finished video, so `generate_video()` cannot
consume it. Use `generate_seedance()`, which submits the job and then polls until
it finishes.

`seedance-2-5` renders **4–30 seconds in a single clip** — the longest we offer —
and its audio track costs nothing extra. The trade-off is resolution: **480p and
720p only**, so if you need 1080p or 4K stay on `seedance-2-0-pro` (4–15s).

```python
result = client.generate_seedance(
    prompt="A lone hiker crossing a snowfield, wind picking up, wide drone shot",
    duration=30,
    resolution="720p",
    generate_audio=True,        # free
)
print(result["video_url"])
print(f"{result['credits_used']} credits")   # ~435 for 30s @ 720p
```

Rates are per second: **14.5 credits/s at 720p**, **6.4 at 480p**. Draft at 480p
(a 5s test costs 32 credits) and re-render the take you like at 720p.

**Edit or extend an existing video** — pass `reference_videos` and
`duration=-1` to keep the source length:

```python
edited = client.generate_seedance(
    prompt="Make it golden hour, warmer light on the subject's face",
    reference_videos=["https://example.com/clip.mp4"],
    duration=-1,
)
```

A video reference bills at the higher **17.6 credits/s** (720p) because the
source frames are charged as input.

**Face consistency** — register a portrait once, then reuse the asset id:

```python
asset = client.register_video_asset("https://example.com/face.jpg")
result = client.generate_seedance(
    prompt="The same woman walking through a night market, neon reflections",
    asset_ids=[asset["asset_id"]],
    duration=10,
)
```

`generate_seedance()` raises `fotohub.TimeoutError` if the job is still running
when `timeout` (default 1800s) expires — the message carries the `job_id` so you
can keep polling — and `FotoHubError` if the job fails. Note that
`fotohub.TimeoutError` is a `FotoHubError` subclass, **not** Python's built-in
`TimeoutError`; import it as
`from fotohub.exceptions import TimeoutError as FotoHubTimeoutError` if the
distinction matters. `AsyncFotoHub` exposes the same two methods with `await`.

### Music, SFX and Speech

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

track = client.generate_music(
    prompt="Upbeat electronic track with heavy bass and synth arpeggios",
    duration=30,          # integer seconds, capped at 300
    genre="electronic",
    instrumental=True,
)
print(track["audio_url"], track["duration"], track["credits_used"])

sfx = client.generate_sfx(prompt="Heavy wooden door slamming shut", duration=5)
print(sfx["audio_url"])

speech = client.generate_speech(text="Dzień dobry!", language="pl", speed=1.0)
print(speech["audio_url"])
```

Music costs 5 credits up to 30 s, 10 up to 60 s, 25 beyond that.

### Virtual Try-On

A try-on is a job, not a blocking call: a render takes about 11 seconds, so you submit and then wait.

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

job = client.tryon(
    person_image_url="https://example.com/person.jpg",
    garment_image_url="https://example.com/shirt.png",
    category="tops",              # "tops" | "bottoms" | "one-pieces"
    garment_photo_type="flat-lay",
)
result = client.wait_for_tryon(job["job_id"])
print(result["images"][0])
```

Pass `garments` to dress a top **and** a bottom in one job. The API applies the top first, feeds
that render into the second pass, and charges 3 credits instead of 4:

```python
job = client.tryon(
    person_image_url="https://example.com/person.jpg",
    garments=[
        {"garment_image_url": "https://example.com/tee.png", "category": "tops"},
        {"garment_id": "0f1e2d3c-...", "category": "bottoms"},   # or a catalogue id
    ],
)
result = client.wait_for_tryon(job["job_id"], timeout=60)

# If the second pass failed, the top-only render still comes back and one credit
# is refunded — so check before calling it a finished outfit.
partial = (result.get("metadata") or {}).get("partial_failure")
if partial:
    print(f"The {partial['slot']} is missing:", result["images"][0])
```

Exactly one top plus one bottom is required — two tops, three garments, or a `one-pieces` in the
array are rejected with `400`. Hats and shoes are not supported by the model at all.

### Chat Completions (OpenAI-Compatible)

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

response = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)
print(response["choices"][0]["message"]["content"])
print(response["credits_used"])
```

`chat()` accepts exactly four model IDs — `gemini-flash` (default),
`gemini-pro`, `gpt-4o`, `claude-sonnet`. Anything else is rejected with `400`.
For premium Claude-class models use `chat_claude()` or `chat_bedrock()`, which
take a full model ID and a `system=` prompt.

`credits_used` is the authoritative charge; `usage` is passed through from the
provider and can be `{}`.

### Streaming Chat

> **`chat(stream=True)` is not supported and raises `ValueError`.**
> `/v1/ai/chat/completions` accepts the flag for OpenAI compatibility and then
> ignores it, returning one complete JSON body. The stream iterator would find no
> SSE frames in that body and yield nothing while the request was still billed,
> so the SDK refuses before sending.

The one streaming endpoint on the platform is `POST /v1/ai/agent/stream`. It has
no SDK wrapper yet — call it over plain HTTP. Frames carry a `type`
(`text_delta`, `tool_use`, `done`, `error`) and the stream ends at `data: [DONE]`:

```python
import json
import os

import requests

resp = requests.post(
    "https://apis.fotohub.app/v1/ai/agent/stream",
    headers={"Authorization": f"Bearer {os.environ['FOTOHUB_API_KEY']}"},
    json={
        "model": "claude-sonnet-4.6",
        "messages": [{"role": "user", "content": "Write a short poem about the sea."}],
    },
    stream=True,
)
resp.raise_for_status()

for line in resp.iter_lines():
    if not line:
        continue
    payload = line.decode("utf-8")
    if not payload.startswith("data: "):
        continue
    data = payload[6:]
    if data == "[DONE]":          # the only reliable terminator
        break
    frame = json.loads(data)
    if frame["type"] == "text_delta":
        print(frame["text"], end="", flush=True)
    elif frame["type"] == "error":
        raise RuntimeError(frame["message"])
print()
```

Two things to know about that endpoint: the `done` frame is **optional** (it is
omitted when the turn produced no tokens, and replaced by `error` when
generation succeeded but settlement failed), and **abandoning the stream still
bills you** — the server settles the tokens it already generated.

### Gabriel AI (Model Routing)

Gabriel returns a routing *decision*, not a completion: it tells you which
feature and model to use, and you make that call yourself. An API key is
required.

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

decision = client.gabriel_classify(
    "I want to create a logo for my coffee shop",
    language="en",
    context={"user_tier": "pro"},
)
print(decision["action"])             # route | answer | workflow | error
print(decision.get("target"))         # the feature to send the user to
print(decision.get("model_selected"))
print(decision.get("credits_estimated"))

# Autocomplete as the user types, and idle suggestions for a dashboard:
print(client.gabriel_suggest("make me a log", tab="image"))
print(client.gabriel_recommend(credits_remaining=40))
```

### Storage & S3 Buckets

Dedicated S3 buckets live under `/v1/storage/s3/*` and are **not wrapped by this
SDK** — call them over plain HTTP for now:

```python
import httpx

api = httpx.Client(
    base_url="https://apis.fotohub.app",
    headers={"Authorization": "Bearer fh_live_..."},
)

buckets = api.get("/v1/storage/s3/buckets").json()

upload = api.post(
    f"/v1/storage/s3/buckets/{buckets[0]['id']}/objects/presign-upload",
    json={"key": "images/photo.jpg", "content_type": "image/jpeg", "expires_in": 3600},
).json()
# PUT your bytes to upload["url"]
```

### Translation

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

result = client.translate("Hello, how are you?", "pl")
print(result["translated_text"])   # "Cześć, jak się masz?"
print(result["source_language"])   # "auto"
```

`text` and `target_language` are positional. On a provider timeout the endpoint
returns your input text unchanged rather than an error — compare against the
input if that matters to you.

### Billing & Usage

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

balance = client.get_balance()
print(balance["credits"]["remaining_4h"], "credits left in this 4h window")
print(balance["wallet"]["balance"], balance["wallet"]["currency"])   # USD

print(client.get_transactions(page=1, page_size=20))
print(client.get_invoices())
print(client.estimate_cost([{"type": "generate_image", "model": "flux-2-pro", "count": 10}]))
```

Per-endpoint analytics live at `GET /v1/usage` (JWT auth, fixed 30-day window)
and have no SDK helper — see the
[Usage & Analytics docs](https://docs.fotohub.app/api/usage-analytics).

**Topping up.** Either take a fixed package or name your own amount:

```python
for pkg in client.get_topup_packages():
    # The slugs are historical (topup-50 is now the $15 package) — read
    # amount_usd, never the number in the slug.
    print(pkg["slug"], pkg["amount_usd"], f"+{pkg['bonus_pct']}% bonus credits")

session = client.create_topup("topup-100")       # a package — this one is $25
print(session["checkout_url"])

session = client.topup_wallet(40.0)              # an arbitrary amount, $10–$15000
print(session["checkout_url"])
```

The wallet is denominated in **USD**. A customer in Poland can still pay in
złoty — pass `pay_currency="pln"` and Stripe offers BLIK, card and bank transfer
while the wallet is credited the USD amount you asked for:

```python
session = client.topup_wallet(40.0, pay_currency="pln")
```

## Async Client

Every method is available as an async variant via `AsyncFotoHub`:

```python
import asyncio
from fotohub import AsyncFotoHub

async def main():
    async with AsyncFotoHub(api_key="fh_live_...") as client:
        # Generate images concurrently
        results = await asyncio.gather(
            client.generate_image(prompt="A sunset over mountains"),
            client.generate_image(prompt="A forest in morning mist"),
            client.generate_image(prompt="An ocean wave at golden hour"),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"failed: {r}")
            else:
                print(r["images"][0])

asyncio.run(main())
```

`return_exceptions=True` matters here: without it, one failed generation cancels
the gather while the others keep running server-side — and you are still billed
for them.

`await client.chat(..., stream=True)` raises `ValueError` on the async client
too, for the same reason as the sync one.

## Error Handling

The SDK raises typed exceptions for all error conditions:

```python
from fotohub import (
    FotoHub,
    AuthError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
    ServerError,
    TimeoutError,
    VideoJobTimeoutError,
)

client = FotoHub(api_key="fh_live_...")

try:
    result = client.generate_image(prompt="test")
except AuthError as e:
    # Invalid or missing API key (HTTP 401/403)
    print(f"Authentication failed: {e}")
except InsufficientCreditsError as e:
    # Not enough credits (HTTP 402)
    print(f"Out of credits: {e}")
except RateLimitError as e:
    # Too many requests (HTTP 429)
    print(f"Rate limited. Retry after {e.retry_after}s")
except ValidationError as e:
    # Invalid parameters (HTTP 400/422)
    print(f"Invalid request: {e.errors}")
except ServerError as e:
    # Server error (HTTP 5xx)
    print(f"Server error: {e}")
except TimeoutError as e:
    # Request timed out
    print(f"Timed out: {e}")
except VideoJobTimeoutError as e:
    # Video polling exceeded max_wait
    print(f"Video job {e.job_id} timed out")
```

### Exception Hierarchy

| Exception | HTTP Status | Description |
|-----------|-------------|-------------|
| `FotoHubError` | Any | Base exception for all SDK errors |
| `AuthError` | 401, 403 | Invalid or missing API key |
| `InsufficientCreditsError` | 402 | Account lacks sufficient credits |
| `RateLimitError` | 429 | Rate limit exceeded |
| `ValidationError` | 400, 422 | Invalid request parameters |
| `ServerError` | 5xx | Server-side error |
| `TimeoutError` | — | Request timed out or connection failed |
| `VideoJobTimeoutError` | — | Video polling exceeded `max_wait` (legacy — video is synchronous now) |

All exceptions include `status_code` and `response_body` attributes for debugging.

`InsufficientCreditsError.credits_required` / `.credits_available` and
`ValidationError.errors` are populated only when the response carries those keys.
The API returns FastAPI's `{"detail": "..."}` envelope, so in practice they are
`None` / `[]` — read `str(e)` or `e.response_body` for the real reason.

Two failure modes that are **not** exceptions and need an explicit check:

- **`chat(stream=True)`** raises `ValueError` before sending — see
  [Streaming Chat](#streaming-chat).
- **A partially failed try-on outfit completes rather than fails.** The top-only
  render comes back, one credit is refunded, and
  `result["metadata"]["partial_failure"]` says which slot is missing.

## Configuration

```python
from fotohub import FotoHub

client = FotoHub(
    api_key="fh_live_...",
    base_url="https://apis.fotohub.app",  # Custom API endpoint
    timeout=120.0,                         # Request timeout (seconds)
    max_retries=3,                         # Max retry attempts
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOTOHUB_API_KEY` | API key for authentication | — |
| `FOTOHUB_BASE_URL` | Override API base URL | `https://apis.fotohub.app` |

### Retry Behavior

The SDK automatically retries on transient failures:

- **HTTP 429** — Rate limit (respects `Retry-After` header)
- **HTTP 500, 502, 503, 504** — Server errors
- **Connection timeouts** — Network failures

Backoff schedule: `0.5s → 1s → 2s → 4s → ...` (capped at 30s).

## Context Managers

Both clients support context managers for automatic resource cleanup:

```python
# Sync
with FotoHub(api_key="fh_live_...") as client:
    result = client.generate_image(prompt="test")

# Async
async with AsyncFotoHub(api_key="fh_live_...") as client:
    result = await client.generate_image(prompt="test")
```

## API Reference

### `FotoHub` / `AsyncFotoHub`

On `AsyncFotoHub` every method below is a coroutine. The two classes are
otherwise identical, with one exception: `gabriel_stream()` is sync-only.

**Generation**

| Method | Description |
|--------|-------------|
| `generate_image(prompt, *, model, width, height, aspect_ratio, num_images, negative_prompt, style, seed)` | Generate images from text |
| `generate_ida_q(prompt, *, aspect_ratio, image_size, num_images, seed, poll_interval, timeout)` | IDA Q 1.0 — submits and polls to completion |
| `edit_image(image_url, prompt, *, mode, mask_url, model)` | Inpaint, remove background, upscale |
| `remove_background(image_url)` / `upscale_image(image_url, *, scale)` | `edit_image` convenience wrappers |
| `generate_video(prompt, *, model, duration, aspect_ratio, image_url, resolution)` | Generate a video — blocks, returns `video_url`. Not for Seedance |
| `generate_seedance(prompt, *, model, duration, resolution, generate_audio, reference_videos, asset_ids, ...)` | Seedance 4–30s — submits and polls to completion |
| `register_video_asset(image_url)` | Register a face for Seedance `asset_ids` — free |
| `generate_music(prompt, *, model, duration, genre, mood, tempo, instrumental)` | Generate music from text |
| `generate_sfx(prompt, *, duration)` | Generate a sound effect |
| `generate_speech(text, *, voice_id, model, language, speed, pitch)` | Text to speech |
| `transcribe(audio_url, *, language)` | Speech to text |
| `generate_3d(mode, model, *, image, prompt, quality, format, options)` | Start a 3D mesh job |
| `get_3d_status(job_id)` / `wait_for_3d(job_id, ...)` | Poll a 3D job |
| `tryon(person_image_url, *, garment_image_url, garment_id, category, garments, ...)` | Start a virtual try-on job |
| `get_tryon_status(job_id)` / `wait_for_tryon(job_id, ...)` | Poll a try-on job |

**Language**

| Method | Description |
|--------|-------------|
| `chat(messages, *, model, temperature, max_tokens)` | Chat completion. `stream=True` raises `ValueError` |
| `chat_claude(messages, *, model, temperature, max_tokens, system)` | Premium Claude-class chat |
| `chat_bedrock(messages, *, model, temperature, max_tokens, system)` | Chat via Bedrock |
| `analyze_image(image_url, *, features)` | Vision analysis |
| `enhance_prompt(prompt, *, style)` | Rewrite a prompt for image models |
| `translate(text, target_language, *, source_language)` | Translate text |
| `gabriel_classify(prompt, *, language, context, enhance_prompt)` | Route a request to a feature + model |
| `gabriel_stream(prompt, *, language, context)` | Same, streamed as SSE frames |
| `gabriel_suggest(partial, *, tab, page)` | Autocomplete suggestions |
| `gabriel_recommend(*, page, credits_remaining, has_brand, recent_actions)` | Idle recommendations |

**Stability tools**

`stability_tools()`, `stability_run(tool_id, image_base64, ...)`, and the named
wrappers `stability_upscale`, `stability_remove_background`, `stability_erase`,
`stability_inpaint`, `stability_outpaint`, `stability_search_replace`,
`stability_recolor`, `stability_style_transfer`.

**Billing, tiers and webhooks** — all amounts in USD

| Method | Description |
|--------|-------------|
| `get_balance()` | Tier, credit counters, wallet balance, overage |
| `get_credits()` | Credit breakdown and per-operation costs |
| `get_pricing()` / `get_plans()` | Public price catalogue, subscription plans |
| `estimate_cost(operations)` | Price a batch before running it |
| `get_topup_packages()` / `create_topup(package)` | Wallet top-up catalogue and checkout |
| `get_wallet()` / `topup_wallet(amount_usd, *, pay_currency)` | Wallet state and an arbitrary-amount top-up |
| `get_transactions(*, page, page_size, type_filter)` / `get_invoices()` | History |
| `set_overage_limit(hard_limit_usd, *, project_id)` | Hard monthly overage cap |
| `get_tier_catalog()` / `get_current_tier()` / `compare_tiers()` / `subscribe_tier(slug)` | API tier plans |
| `apply_enterprise(company_name, contact_email, expected_usage, use_case, *, notes)` | Enterprise enquiry |
| `list_webhooks()` / `create_webhook(name, url, events, *, headers)` / `update_webhook(id, **kw)` / `delete_webhook(id)` / `test_webhook(id)` / `get_webhook_logs(id)` | Webhook management |

`topup_wallet(pay_currency="pln")` keeps the wallet in USD while letting a Polish
customer pay at Stripe in PLN (BLIK, card, bank transfer).

Storage (`/v1/storage/s3/*`) and per-endpoint analytics (`GET /v1/usage`) have no
SDK wrappers yet — call them over HTTP.

## Type Safety

The SDK ships a `py.typed` marker and full annotations on every parameter.
Responses are returned as `dict[str, Any]` — the API's JSON, unwrapped:

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

result = client.generate_image(prompt="test")
url: str = result["images"][0]
credits: float = result["credits_used"]
charged: float = result["billing"]["usd_charged"]
```

`fotohub.models` also ships Pydantic v2 models (`ImageGenerationResponse`,
`ImageResult`, …), but **no client method returns them** — they are not exported
from the package root and are kept only for callers that want to validate a
payload themselves.

## Requirements

| Dependency | Version |
|-----------|---------|
| Python | >= 3.9 |
| httpx | >= 0.24 |
| pydantic | >= 2.0 |

## Contributing

We welcome contributions! To get started:

```bash
# Clone the repository
git clone https://github.com/fotohubapp/sdk-python.git
cd sdk-python

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy fotohub/
```

Please ensure all tests pass and type checks are clean before submitting a pull request.

## Links

- [Documentation](https://docs.fotohub.app/sdk/python)
- [API Reference](https://docs.fotohub.app/api)
- [FOTOhub Platform](https://fotohub.app)
- [Changelog](https://github.com/fotohubapp/sdk-python/releases)
- [Issue Tracker](https://github.com/fotohubapp/sdk-python/issues)

## License

MIT License. See [LICENSE](LICENSE) for details.
