"""FOTOhub API client — synchronous and asynchronous.

Covers all 29+ public API endpoints with full type annotations.
"""

from __future__ import annotations

import os
import time
from typing import Any, Generator, Optional, Union

import httpx

from .exceptions import (
    AuthError,
    FotoHubError,
    InsufficientCreditsError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
    VideoJobTimeoutError,
)
from .streaming import AsyncChatStream, ChatStream

DEFAULT_BASE_URL = "https://apis.fotohub.app"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_IMAGE_MODEL = "seedream-5-0-260128"
DEFAULT_VIDEO_MODEL = "veo-2"
DEFAULT_CHAT_MODEL = "gemini-flash"
DEFAULT_BEDROCK_MODEL = "claude-sonnet-4.6"
DEFAULT_MUSIC_MODEL = "minimax"
DEFAULT_SPEECH_MODEL = "google"
SDK_VERSION = "1.3.0"


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
        """Raise appropriate exception based on HTTP status code."""
        status = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}

        message = body.get("error", body.get("message", response.text))

        if status == 401 or status == 403:
            raise AuthError(message=str(message), status_code=status, response_body=body)
        elif status == 402:
            raise InsufficientCreditsError(
                message=str(message),
                status_code=status,
                response_body=body,
                credits_required=body.get("credits_required"),
                credits_available=body.get("credits_available"),
            )
        elif status == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                message=str(message),
                status_code=status,
                response_body=body,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif status == 400 or status == 422:
            raise ValidationError(
                message=str(message),
                status_code=status,
                response_body=body,
                errors=body.get("errors"),
            )
        elif status >= 500:
            raise ServerError(message=str(message), status_code=status, response_body=body)
        else:
            raise FotoHubError(message=str(message), status_code=status, response_body=body)

    def _should_retry(self, status_code: int) -> bool:
        """Determine if a request should be retried based on status code."""
        return status_code in (429, 500, 502, 503, 504)

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
    ) -> httpx.Response:
        """Make an HTTP request with retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = self._client.stream(
                        method, path, json=json_data, params=params
                    ).__enter__()
                else:
                    response = self._client.request(
                        method, path, json=json_data, params=params
                    )

                if response.status_code < 400:
                    return response

                if self._should_retry(response.status_code) and attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = max(delay, float(retry_after))
                    time.sleep(delay)
                    continue

                self._handle_error_response(response)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
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
            num_images: Number of images to generate (1-4).
            negative_prompt: Things to avoid in the image.
            style: Style preset (e.g. "photographic", "cinematic", "anime").
            seed: Random seed for reproducibility.

        Returns:
            Dict with ``images`` list containing URLs, model, credits_used.

        Raises:
            InsufficientCreditsError: If account lacks credits.
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
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if style is not None:
            payload["style"] = style
        if seed is not None:
            payload["seed"] = seed

        response = self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return response.json()

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
    ) -> dict[str, Any]:
        """Start an asynchronous video generation job.

        Args:
            prompt: Text description of the desired video.
            model: Video model (default: veo-2).
            duration: Desired duration in seconds.
            aspect_ratio: Aspect ratio (e.g. "16:9", "9:16", "1:1").
            image_url: Reference image for image-to-video generation.
            resolution: Output resolution ("720p", "1080p", "4k").

        Returns:
            Dict with job_id and initial status.
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

        response = self._request("POST", "/v1/ai/generate/video", json_data=payload)
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
            Dict with audio URL, duration, credits_used.
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
            Dict with audio URL, duration, credits_used.
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
            model: LLM model (default: gemini-flash).
            temperature: Sampling temperature (0-2, default: 0.7).
            max_tokens: Maximum tokens in the response.
            stream: If True, returns a ChatStream iterator yielding SSE chunks.

        Returns:
            Dict with choices, usage if stream=False; ChatStream if stream=True.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            response = self._request("POST", "/v1/ai/chat", json_data=payload, stream=True)
            return ChatStream(response)

        response = self._request("POST", "/v1/ai/chat", json_data=payload)
        return response.json()

    def chat_bedrock(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_BEDROCK_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a chat request via AWS Bedrock (Claude/Anthropic models).

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: Bedrock model ID (default: claude-sonnet-4.6).
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

        response = self._request("POST", "/v1/ai/chat/bedrock", json_data=payload)
        return response.json()

    def analyze_image(
        self,
        image_url: str,
        *,
        features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Analyze an image using vision models.

        Args:
            image_url: URL of the image to analyze.
            features: List of analysis features (e.g. ["caption", "objects",
                "text", "faces", "nsfw", "colors"]).

        Returns:
            Dict with analysis results keyed by feature.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if features is not None:
            payload["features"] = features

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
        """List available Stability AI tools and their capabilities.

        Returns:
            List of tool descriptors with id, name, description, parameters.
        """
        response = self._request("GET", "/v1/ai/stability/tools")
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
            Dict with output image (base64 or URL), seed, credits_used.
        """
        payload: dict[str, Any] = {
            "tool_id": tool_id,
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

        response = self._request("POST", "/v1/ai/stability/run", json_data=payload)
        return response.json()

    def stability_upscale(
        self,
        image_base64: str,
        *,
        type: str = "fast",
    ) -> dict[str, Any]:
        """Upscale an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.
            type: Upscale type — "fast" (2x, instant) or "creative" (4x, slower).

        Returns:
            Dict with upscaled image data.
        """
        return self.stability_run("upscale", image_base64, type=type)

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
        return self.stability_run("erase", image_base64, mask=mask_base64)

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
            "search-and-replace", image_base64, prompt=prompt, search_prompt=search_prompt
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
            "recolor", image_base64, prompt=prompt, search_prompt=search_prompt
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
        """Get current credit balance and plan info.

        Returns:
            Dict with credits_remaining, plan, limits, usage_this_period.
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
        """Get available subscription plans.

        Returns:
            Dict with list of plans, their features and pricing.
        """
        response = self._request("GET", "/v1/billing/plans")
        return response.json()

    def get_credits(self) -> dict[str, Any]:
        """Get detailed credit breakdown (included, bonus, topup).

        Returns:
            Dict with credit pools and expiration info.
        """
        response = self._request("GET", "/v1/billing/credits")
        return response.json()

    def set_overage_limit(
        self,
        hard_limit_pln: float,
        *,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set the hard spending limit for overage charges.

        Args:
            hard_limit_pln: Maximum overage amount in PLN.
            project_id: Optional project ID (defaults to account-level).

        Returns:
            Dict confirming the updated limit.
        """
        payload: dict[str, Any] = {"hard_limit_pln": hard_limit_pln}
        if project_id is not None:
            payload["project_id"] = project_id

        response = self._request("POST", "/v1/billing/overage-limit", json_data=payload)
        return response.json()

    def get_topup_packages(self) -> list[dict[str, Any]]:
        """Get available credit top-up packages.

        Returns:
            List of packages with id, credits, price_pln, bonus.
        """
        response = self._request("GET", "/v1/billing/topup/packages")
        data = response.json()
        return data.get("packages", data) if isinstance(data, dict) else data

    def create_topup(self, package: str) -> dict[str, Any]:
        """Purchase a credit top-up package.

        Args:
            package: Package identifier (e.g. "starter", "pro", "enterprise").

        Returns:
            Dict with payment URL or confirmation.
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
            Dict with total_credits, breakdown per operation.
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

        Args:
            mode: Generation mode — "image-to-3d" or "text-to-3d".
            model: 3D model to use ("triposr", "sf3d", "shap-e", "trellis", "hunyuan3d").
            image: Base64-encoded image (required for image-to-3d).
            prompt: Text prompt (required for text-to-3d).
            quality: Output quality — "draft", "standard", "high".
            format: Output file format — "glb", "obj", "stl", "usdz".
            options: Additional options (texture, pbr, simplify, target_polys).

        Returns:
            Dict with id, url, format, status, billing info.
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
        """Check the status of a 3D generation job.

        Args:
            job_id: The generation ID returned from generate_3d().

        Returns:
            Dict with status, url (if completed), and billing info.
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
        """Wait for a 3D generation job to complete, polling at intervals.

        Args:
            job_id: The generation ID returned from generate_3d().
            poll_interval: Seconds between status checks (default: 3.0).
            timeout: Maximum wait time in seconds (default: 120.0).

        Returns:
            Dict with completed result including url and billing.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            FotoHubError: If the job fails.
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
            List of 3D models with id, name, credits, speed, mode, quality.
        """
        response = self._request("GET", "/v1/ai/generate/3d/models")
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    # =========================================================================
    # Tier Management
    # =========================================================================

    def get_tier_catalog(self) -> dict[str, Any]:
        """Get the full tier catalog (PAYG + subscription tiers).

        Returns:
            Dict with tiers list containing slug, name, type, rpm, credits, price.
        """
        response = self._request("GET", "/v1/tiers/catalog")
        return response.json()

    def get_current_tier(self) -> dict[str, Any]:
        """Get the current user's tier, limits, and usage stats.

        Returns:
            Dict with tier slug, name, limits (rpm, daily_quota, credits_monthly),
            and usage (rpm_used, daily_used, credits_used).
        """
        response = self._request("GET", "/v1/tiers/current")
        return response.json()

    def compare_tiers(self) -> dict[str, Any]:
        """Compare all tiers side-by-side with the current tier highlighted.

        Returns:
            Dict with current tier slug and full tier comparison data.
        """
        response = self._request("GET", "/v1/tiers/compare")
        return response.json()

    def subscribe_tier(self, tier_slug: str) -> dict[str, Any]:
        """Subscribe to a tier (returns checkout URL for payment).

        Args:
            tier_slug: Tier identifier (e.g. "sub-developer", "sub-startup").

        Returns:
            Dict with checkout_url for completing the subscription.
        """
        payload: dict[str, Any] = {"tier": tier_slug}
        response = self._request("POST", "/v1/tiers/subscribe", json_data=payload)
        return response.json()

    def get_wallet(self) -> dict[str, Any]:
        """Get the current wallet balance and spending info.

        Returns:
            Dict with balance, currency, lifetime_spend, auto_topup.
        """
        response = self._request("GET", "/v1/tiers/wallet")
        return response.json()

    def topup_wallet(self, amount: float) -> dict[str, Any]:
        """Top up wallet balance (returns payment session URL).

        Args:
            amount: Amount in PLN to add.

        Returns:
            Dict with session_url for completing payment.
        """
        payload: dict[str, Any] = {"amount": amount}
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
        response = self._request("GET", "/v1/webhooks")
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

        response = self._request("POST", "/v1/webhooks", json_data=payload)
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
            "PATCH", f"/v1/webhooks/{webhook_id}", json_data=kwargs
        )
        return response.json()

    def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
        """
        self._request("DELETE", f"/v1/webhooks/{webhook_id}")

    def test_webhook(self, webhook_id: str) -> dict[str, Any]:
        """Send a test event to a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            Dict with delivery status and response code.
        """
        response = self._request("POST", f"/v1/webhooks/{webhook_id}/test")
        return response.json()

    def get_webhook_logs(self, webhook_id: str) -> list[dict[str, Any]]:
        """Get delivery logs for a webhook.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            List of delivery log entries with status, timestamp, response.
        """
        response = self._request("GET", f"/v1/webhooks/{webhook_id}/logs")
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
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Poll a video generation job until completion or timeout.

        Args:
            job_id: The job ID from generate_video().
            poll_interval: Seconds between status checks (default: 5).
            timeout: Maximum seconds to wait (default: 300).

        Returns:
            Dict with final job status and video_url if completed.

        Raises:
            VideoJobTimeoutError: If timeout is exceeded.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            response = self._request("GET", f"/v1/ai/generate/video/{job_id}")
            job = response.json()

            status = job.get("status", "")
            if status in ("completed", "failed"):
                return job

            time.sleep(poll_interval)

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
    ) -> httpx.Response:
        """Make an async HTTP request with retry logic."""
        import asyncio

        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = await self._client.stream(
                        method, path, json=json_data, params=params
                    ).__aenter__()
                else:
                    response = await self._client.request(
                        method, path, json=json_data, params=params
                    )

                if response.status_code < 400:
                    return response

                if self._should_retry(response.status_code) and attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = max(delay, float(retry_after))
                    await asyncio.sleep(delay)
                    continue

                self._handle_error_response(response)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
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
            num_images: Number of images to generate (1-4).
            negative_prompt: Things to avoid in the image.
            style: Style preset (e.g. "photographic", "cinematic", "anime").
            seed: Random seed for reproducibility.

        Returns:
            Dict with ``images`` list containing URLs, model, credits_used.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
        }
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if style is not None:
            payload["style"] = style
        if seed is not None:
            payload["seed"] = seed

        response = await self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return response.json()

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
    ) -> dict[str, Any]:
        """Start an asynchronous video generation job.

        Args:
            prompt: Text description of the desired video.
            model: Video model (default: veo-2).
            duration: Desired duration in seconds.
            aspect_ratio: Aspect ratio (e.g. "16:9", "9:16", "1:1").
            image_url: Reference image for image-to-video generation.
            resolution: Output resolution ("720p", "1080p", "4k").

        Returns:
            Dict with job_id and initial status.
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
            Dict with audio URL, duration, credits_used.
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
            Dict with audio URL, duration, credits_used.
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
            model: LLM model (default: gemini-flash).
            temperature: Sampling temperature (0-2, default: 0.7).
            max_tokens: Maximum tokens in the response.
            stream: If True, returns an AsyncChatStream iterator yielding SSE chunks.

        Returns:
            Dict with choices, usage if stream=False; AsyncChatStream if stream=True.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            response = await self._request(
                "POST", "/v1/ai/chat", json_data=payload, stream=True
            )
            return AsyncChatStream(response)

        response = await self._request("POST", "/v1/ai/chat", json_data=payload)
        return response.json()

    async def chat_bedrock(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_BEDROCK_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a chat request via AWS Bedrock (Claude/Anthropic models).

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            model: Bedrock model ID (default: claude-sonnet-4.6).
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

        response = await self._request("POST", "/v1/ai/chat/bedrock", json_data=payload)
        return response.json()

    async def analyze_image(
        self,
        image_url: str,
        *,
        features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Analyze an image using vision models.

        Args:
            image_url: URL of the image to analyze.
            features: List of analysis features (e.g. ["caption", "objects",
                "text", "faces", "nsfw", "colors"]).

        Returns:
            Dict with analysis results keyed by feature.
        """
        payload: dict[str, Any] = {"image_url": image_url}
        if features is not None:
            payload["features"] = features

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
        """List available Stability AI tools and their capabilities.

        Returns:
            List of tool descriptors with id, name, description, parameters.
        """
        response = await self._request("GET", "/v1/ai/stability/tools")
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
            Dict with output image (base64 or URL), seed, credits_used.
        """
        payload: dict[str, Any] = {
            "tool_id": tool_id,
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

        response = await self._request("POST", "/v1/ai/stability/run", json_data=payload)
        return response.json()

    async def stability_upscale(
        self,
        image_base64: str,
        *,
        type: str = "fast",
    ) -> dict[str, Any]:
        """Upscale an image using Stability AI.

        Args:
            image_base64: Base64-encoded input image.
            type: Upscale type — "fast" (2x, instant) or "creative" (4x, slower).

        Returns:
            Dict with upscaled image data.
        """
        return await self.stability_run("upscale", image_base64, type=type)

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
        return await self.stability_run("erase", image_base64, mask=mask_base64)

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
            "search-and-replace", image_base64, prompt=prompt, search_prompt=search_prompt
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
            "recolor", image_base64, prompt=prompt, search_prompt=search_prompt
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
        """Get current credit balance and plan info.

        Returns:
            Dict with credits_remaining, plan, limits, usage_this_period.
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
        """Get available subscription plans.

        Returns:
            Dict with list of plans, their features and pricing.
        """
        response = await self._request("GET", "/v1/billing/plans")
        return response.json()

    async def get_credits(self) -> dict[str, Any]:
        """Get detailed credit breakdown (included, bonus, topup).

        Returns:
            Dict with credit pools and expiration info.
        """
        response = await self._request("GET", "/v1/billing/credits")
        return response.json()

    async def set_overage_limit(
        self,
        hard_limit_pln: float,
        *,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set the hard spending limit for overage charges.

        Args:
            hard_limit_pln: Maximum overage amount in PLN.
            project_id: Optional project ID (defaults to account-level).

        Returns:
            Dict confirming the updated limit.
        """
        payload: dict[str, Any] = {"hard_limit_pln": hard_limit_pln}
        if project_id is not None:
            payload["project_id"] = project_id

        response = await self._request(
            "POST", "/v1/billing/overage-limit", json_data=payload
        )
        return response.json()

    async def get_topup_packages(self) -> list[dict[str, Any]]:
        """Get available credit top-up packages.

        Returns:
            List of packages with id, credits, price_pln, bonus.
        """
        response = await self._request("GET", "/v1/billing/topup/packages")
        data = response.json()
        return data.get("packages", data) if isinstance(data, dict) else data

    async def create_topup(self, package: str) -> dict[str, Any]:
        """Purchase a credit top-up package.

        Args:
            package: Package identifier (e.g. "starter", "pro", "enterprise").

        Returns:
            Dict with payment URL or confirmation.
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
            Dict with total_credits, breakdown per operation.
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
        """Generate a 3D model from an image or text prompt."""
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
        """Check the status of a 3D generation job."""
        response = await self._request("GET", f"/v1/ai/generate/3d/{job_id}")
        return response.json()

    async def wait_for_3d(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Wait for a 3D generation job to complete."""
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
        """List available 3D generation models."""
        response = await self._request("GET", "/v1/ai/generate/3d/models")
        data = response.json()
        return data.get("models", data) if isinstance(data, dict) else data

    # =========================================================================
    # Tier Management
    # =========================================================================

    async def get_tier_catalog(self) -> dict[str, Any]:
        """Get the full tier catalog."""
        response = await self._request("GET", "/v1/tiers/catalog")
        return response.json()

    async def get_current_tier(self) -> dict[str, Any]:
        """Get the current user's tier, limits, and usage."""
        response = await self._request("GET", "/v1/tiers/current")
        return response.json()

    async def compare_tiers(self) -> dict[str, Any]:
        """Compare all tiers side-by-side."""
        response = await self._request("GET", "/v1/tiers/compare")
        return response.json()

    async def subscribe_tier(self, tier_slug: str) -> dict[str, Any]:
        """Subscribe to a tier."""
        payload: dict[str, Any] = {"tier": tier_slug}
        response = await self._request("POST", "/v1/tiers/subscribe", json_data=payload)
        return response.json()

    async def get_wallet(self) -> dict[str, Any]:
        """Get the current wallet balance."""
        response = await self._request("GET", "/v1/tiers/wallet")
        return response.json()

    async def topup_wallet(self, amount: float) -> dict[str, Any]:
        """Top up wallet balance."""
        payload: dict[str, Any] = {"amount": amount}
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
        response = await self._request("GET", "/v1/webhooks")
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

        response = await self._request("POST", "/v1/webhooks", json_data=payload)
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
            "PATCH", f"/v1/webhooks/{webhook_id}", json_data=kwargs
        )
        return response.json()

    async def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.
        """
        await self._request("DELETE", f"/v1/webhooks/{webhook_id}")

    async def test_webhook(self, webhook_id: str) -> dict[str, Any]:
        """Send a test event to a webhook endpoint.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            Dict with delivery status and response code.
        """
        response = await self._request("POST", f"/v1/webhooks/{webhook_id}/test")
        return response.json()

    async def get_webhook_logs(self, webhook_id: str) -> list[dict[str, Any]]:
        """Get delivery logs for a webhook.

        Args:
            webhook_id: The webhook identifier.

        Returns:
            List of delivery log entries with status, timestamp, response.
        """
        response = await self._request("GET", f"/v1/webhooks/{webhook_id}/logs")
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
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Poll a video generation job until completion or timeout.

        Args:
            job_id: The job ID from generate_video().
            poll_interval: Seconds between status checks (default: 5).
            timeout: Maximum seconds to wait (default: 300).

        Returns:
            Dict with final job status and video_url if completed.

        Raises:
            VideoJobTimeoutError: If timeout is exceeded.
        """
        import asyncio

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} did not complete within {timeout}s",
                    job_id=job_id,
                )

            response = await self._request("GET", f"/v1/ai/generate/video/{job_id}")
            job = response.json()

            status = job.get("status", "")
            if status in ("completed", "failed"):
                return job

            await asyncio.sleep(poll_interval)

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
