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

- **Image Generation** — 25+ models including SeedDream 5.0, Flux, SDXL, Midjourney-style, and more
- **Video Generation** — Async job-based video creation with polling and image-to-video
- **Music Generation** — AI-powered music and audio creation
- **Chat / LLM** — OpenAI-compatible chat completions with streaming support
- **Gabriel AI** — Intent orchestration engine for natural-language workflows
- **Storage** — S3 bucket provisioning with presigned upload/download URLs
- **Translation** — Multi-language translation (no auth required)
- **Fully Typed** — Complete type annotations with Pydantic v2 models
- **Sync + Async** — Both synchronous and asynchronous clients included
- **Automatic Retries** — Exponential backoff with configurable retry logic
- **Streaming** — Real-time SSE streaming for chat completions

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
print(result.images[0].url)
```

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

# Basic generation (default: SeedDream 5.0 Lite)
result = client.generate_image(
    prompt="A serene Japanese garden with cherry blossoms",
)
print(result.images[0].url)

# Advanced options
result = client.generate_image(
    prompt="Cyberpunk cityscape at night, neon reflections on wet streets",
    model="flux-1-schnell",
    width=1280,
    height=720,
    num_images=2,
    negative_prompt="blurry, low quality",
    seed=42,
    guidance_scale=7.5,
    steps=30,
)

for image in result.images:
    print(f"{image.url} — {image.width}x{image.height}, {image.credits_used} credits")
```

### Video Generation

Video generation is asynchronous — submit a job and poll for the result.

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

# Start a video generation job
job = client.generate_video(
    prompt="A drone shot flying over a tropical beach at sunrise",
    duration=5.0,
    aspect_ratio="16:9",
)
print(f"Job started: {job.job_id}")

# Poll until complete (blocks up to 10 minutes)
result = client.poll_video(job.job_id, poll_interval=5.0, max_wait=600)

if result.status == "completed":
    print(f"Video: {result.video_url}")
else:
    print(f"Failed: {result.error}")
```

**Image-to-Video:**

```python
job = client.generate_video(
    prompt="Camera slowly zooms in, subtle parallax motion",
    image_url="https://example.com/photo.jpg",
)
```

### Music Generation

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

result = client.generate_music(
    prompt="Upbeat electronic track with heavy bass and synth arpeggios",
    duration=30.0,
)
print(f"Audio: {result.audio.url}")
print(f"Duration: {result.audio.duration}s")
print(f"Credits used: {result.credits_used}")
```

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
print(response.choices[0].message.content)
```

### Streaming Chat

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

stream = client.chat(
    messages=[{"role": "user", "content": "Write a short poem about the sea."}],
    stream=True,
)

for chunk in stream:
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
print()

# Or collect the entire response at once:
stream = client.chat(messages=[...], stream=True)
full_text = stream.collect()
```

### Gabriel AI (Intent Orchestration)

```python
from fotohub import FotoHub

client = FotoHub()  # No API key needed

result = client.gabriel(
    message="I want to create a logo for my coffee shop",
    context={"user_tier": "pro"},
)
print(f"Intent: {result.intent}")
print(f"Response: {result.response}")
print(f"Suggested actions: {result.actions}")
```

### Storage & S3 Buckets

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

# List buckets
buckets = client.list_buckets()
for bucket in buckets.buckets:
    print(f"{bucket.name} ({bucket.region}) — {bucket.object_count} objects")

# Provision a dedicated S3 bucket
provision = client.provision_s3_bucket(name="my-media-bucket", region="eu-central-1")
print(f"Bucket ID: {provision.bucket_id}")
print(f"Endpoint: {provision.endpoint}")

# Presigned upload
upload = client.presign_upload(
    bucket_id="bucket-123",
    key="images/photo.jpg",
    content_type="image/jpeg",
    expires_in=3600,
)
# Use upload.url with PUT request to upload your file

# Presigned download
download = client.presign_download(bucket_id="bucket-123", key="images/photo.jpg")
print(f"Download: {download.url}")
```

### Translation (No Auth Required)

```python
from fotohub import FotoHub

client = FotoHub()  # No API key needed

result = client.translate(text="Hello, how are you?", target_language="pl")
print(result.translated_text)   # "Cześć, jak się masz?"
print(result.source_language)   # "en"
```

### Usage Analytics

```python
from fotohub import FotoHub

client = FotoHub(api_key="fh_live_...")

usage = client.get_usage(start_date="2026-07-01", end_date="2026-07-18", category="image")
print(f"Total credits: {usage.total_credits_used}")
for record in usage.records:
    print(f"  {record.date}: {record.credits_used} credits ({record.request_count} requests)")
```

## Async Client

Every method is available as an async variant via `AsyncFotoHub`:

```python
import asyncio
from fotohub import AsyncFotoHub

async def main():
    async with AsyncFotoHub(api_key="fh_live_...") as client:
        # Generate images concurrently
        import asyncio as aio
        results = await aio.gather(
            client.generate_image(prompt="A sunset over mountains"),
            client.generate_image(prompt="A forest in morning mist"),
            client.generate_image(prompt="An ocean wave at golden hour"),
        )
        for r in results:
            print(r.images[0].url)

        # Async streaming chat
        stream = await client.chat(
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        async for chunk in stream:
            if chunk.delta_content:
                print(chunk.delta_content, end="")
        print()

asyncio.run(main())
```

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
    print(f"Need credits! Required: {e.credits_required}, Available: {e.credits_available}")
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
| `VideoJobTimeoutError` | — | Video polling exceeded `max_wait` |

All exceptions include `status_code` and `response_body` attributes for debugging.

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

| Method | Description |
|--------|-------------|
| `generate_image(prompt, *, model, width, height, num_images, ...)` | Generate images from text |
| `generate_video(prompt, *, model, duration, image_url, aspect_ratio, ...)` | Start video generation job |
| `poll_video(job_id, *, poll_interval, max_wait)` | Poll video job until completion |
| `generate_music(prompt, *, model, duration, ...)` | Generate music from text |
| `chat(messages, *, model, stream, temperature, max_tokens, ...)` | Chat completion (streaming supported) |
| `translate(text, *, target_language, source_language)` | Translate text between languages |
| `gabriel(message, *, context)` | Intent orchestration via Gabriel AI |
| `get_usage(*, start_date, end_date, category)` | Retrieve usage analytics |
| `list_buckets()` | List storage buckets |
| `create_bucket(name, *, region)` | Create a storage bucket |
| `provision_s3_bucket(*, name, region)` | Provision a dedicated S3 bucket |
| `presign_upload(bucket_id, *, key, content_type, expires_in)` | Get presigned upload URL |
| `presign_download(bucket_id, *, key, expires_in)` | Get presigned download URL |

## Type Safety

The SDK ships with a `py.typed` marker and full type annotations. All response models are Pydantic v2 `BaseModel` subclasses:

```python
from fotohub import FotoHub, ImageGenerationResponse, ImageResult

client = FotoHub(api_key="fh_live_...")

result: ImageGenerationResponse = client.generate_image(prompt="test")
image: ImageResult = result.images[0]

# All fields are fully typed
print(image.url)              # str
print(image.width)            # int
print(image.credits_used)     # float
print(image.generation_time_ms)  # Optional[int]
```

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
