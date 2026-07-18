"""FOTOhub API client — synchronous and asynchronous."""

from __future__ import annotations

import os
import time
from typing import Any, Optional, Union

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
from .models import (
    BucketListResponse,
    BucketProvisionResponse,
    ChatCompletion,
    ChatMessage,
    ChatRole,
    GabrielResponse,
    ImageGenerationResponse,
    MusicGenerationResponse,
    PresignedUrlResponse,
    StorageBucket,
    TranslationResult,
    UsageResponse,
    VideoJob,
)
from .streaming import AsyncChatStream, ChatStream

DEFAULT_BASE_URL = "https://apis.fotohub.app"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_IMAGE_MODEL = "seedream-5-0-260128"
SDK_VERSION = "1.0.0"


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
        self.base_url = (base_url or os.environ.get("FOTOHUB_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
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


class FotoHub(_BaseClient):
    """Synchronous FOTOhub API client.

    Usage:
        from fotohub import FotoHub

        client = FotoHub(api_key="your-api-key")
        result = client.generate_image(prompt="A sunset over mountains")
        print(result.images[0].url)
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

    # --- Image Generation ---

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Model to use. Default: seedream-5-0-260128 (SeedDream 5.0 Lite).
            width: Image width in pixels (default: 1024).
            height: Image height in pixels (default: 1024).
            num_images: Number of images to generate (default: 1).
            negative_prompt: Things to avoid in the image.
            seed: Random seed for reproducibility.
            guidance_scale: How closely to follow the prompt (model-dependent).
            steps: Number of diffusion steps (model-dependent).
            **kwargs: Additional model-specific parameters.

        Returns:
            ImageGenerationResponse with generated image URLs and metadata.

        Raises:
            InsufficientCreditsError: If account lacks credits.
            ValidationError: If parameters are invalid.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "num_images": num_images,
        }
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if steps is not None:
            payload["steps"] = steps
        payload.update(kwargs)

        response = self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return ImageGenerationResponse(**response.json())

    # --- Video Generation ---

    def generate_video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        image_url: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoJob:
        """Start an asynchronous video generation job.

        Args:
            prompt: Text description of the desired video.
            model: Video model to use.
            duration: Desired video duration in seconds.
            image_url: Reference image URL for image-to-video.
            aspect_ratio: Aspect ratio (e.g., "16:9", "9:16", "1:1").
            **kwargs: Additional model-specific parameters.

        Returns:
            VideoJob with job_id and initial status.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        if duration is not None:
            payload["duration"] = duration
        if image_url is not None:
            payload["image_url"] = image_url
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        payload.update(kwargs)

        response = self._request("POST", "/v1/ai/generate/video", json_data=payload)
        return VideoJob(**response.json())

    def poll_video(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> VideoJob:
        """Poll a video generation job until completion or timeout.

        Args:
            job_id: The job ID returned by generate_video().
            poll_interval: Seconds between status checks (default: 5).
            max_wait: Maximum seconds to wait (default: 600).

        Returns:
            VideoJob with final status and video_url (if completed).

        Raises:
            VideoJobTimeoutError: If max_wait is exceeded.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} did not complete within {max_wait}s",
                    job_id=job_id,
                )

            response = self._request("GET", f"/v1/ai/generate/video/{job_id}")
            job = VideoJob(**response.json())

            if job.status in ("completed", "failed"):
                return job

            time.sleep(poll_interval)

    # --- Music Generation ---

    def generate_music(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        **kwargs: Any,
    ) -> MusicGenerationResponse:
        """Generate music from a text description.

        Args:
            prompt: Text description of the desired music.
            model: Music generation model to use.
            duration: Desired duration in seconds.
            **kwargs: Additional model-specific parameters.

        Returns:
            MusicGenerationResponse with audio URL and metadata.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        if duration is not None:
            payload["duration"] = duration
        payload.update(kwargs)

        response = self._request("POST", "/v1/ai/generate/music", json_data=payload)
        return MusicGenerationResponse(**response.json())

    # --- Chat / LLM ---

    def chat(
        self,
        messages: list[Union[dict[str, str], ChatMessage]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, ChatStream]:
        """Send a chat completion request (OpenAI-compatible).

        Args:
            messages: List of chat messages (dicts with 'role' and 'content', or ChatMessage).
            model: LLM model to use.
            stream: If True, returns a ChatStream iterator.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional parameters.

        Returns:
            ChatCompletion if stream=False, ChatStream if stream=True.
        """
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append({"role": msg.role.value, "content": msg.content})
            else:
                formatted_messages.append(msg)

        payload: dict[str, Any] = {"messages": formatted_messages, "stream": stream}
        if model is not None:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(kwargs)

        if stream:
            response = self._request("POST", "/v1/ai/chat", json_data=payload, stream=True)
            return ChatStream(response)

        response = self._request("POST", "/v1/ai/chat", json_data=payload)
        return ChatCompletion(**response.json())

    # --- Translation ---

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranslationResult:
        """Translate text between languages (no auth required).

        Args:
            text: Text to translate.
            target_language: Target language code (e.g., "en", "pl", "de").
            source_language: Source language code (auto-detected if not provided).
            **kwargs: Additional parameters.

        Returns:
            TranslationResult with translated text.
        """
        payload: dict[str, Any] = {
            "text": text,
            "target_language": target_language,
        }
        if source_language is not None:
            payload["source_language"] = source_language
        payload.update(kwargs)

        response = self._request("POST", "/v1/ai/translate", json_data=payload)
        return TranslationResult(**response.json())

    # --- Gabriel (Intent Orchestration) ---

    def gabriel(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GabrielResponse:
        """Send a message to the Gabriel intent orchestration engine (no auth required).

        Args:
            message: User message to process.
            context: Optional context dictionary.
            **kwargs: Additional parameters.

        Returns:
            GabrielResponse with detected intent and response.
        """
        payload: dict[str, Any] = {"message": message}
        if context is not None:
            payload["context"] = context
        payload.update(kwargs)

        response = self._request("POST", "/v1/ai/gabriel", json_data=payload)
        return GabrielResponse(**response.json())

    # --- Usage Analytics ---

    def get_usage(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> UsageResponse:
        """Get usage analytics for your account.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            category: Filter by category (image, video, chat, music).

        Returns:
            UsageResponse with credit usage breakdown.
        """
        params: dict[str, Any] = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if category is not None:
            params["category"] = category

        response = self._request("GET", "/v1/usage", params=params)
        return UsageResponse(**response.json())

    # --- Storage Buckets ---

    def list_buckets(self) -> BucketListResponse:
        """List all storage buckets for your account.

        Returns:
            BucketListResponse with list of buckets.
        """
        response = self._request("GET", "/v1/buckets")
        return BucketListResponse(**response.json())

    def create_bucket(
        self,
        name: str,
        *,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> StorageBucket:
        """Create a new storage bucket.

        Args:
            name: Bucket name.
            region: AWS region (default: eu-central-1).
            **kwargs: Additional parameters.

        Returns:
            StorageBucket with the created bucket details.
        """
        payload: dict[str, Any] = {"name": name}
        if region is not None:
            payload["region"] = region
        payload.update(kwargs)

        response = self._request("POST", "/v1/buckets", json_data=payload)
        return StorageBucket(**response.json())

    def provision_s3_bucket(
        self,
        *,
        name: Optional[str] = None,
        region: str = "eu-central-1",
        **kwargs: Any,
    ) -> BucketProvisionResponse:
        """Provision a dedicated S3 bucket.

        Args:
            name: Desired bucket name (auto-generated if not provided).
            region: AWS region (default: eu-central-1).
            **kwargs: Additional parameters.

        Returns:
            BucketProvisionResponse with bucket details and credentials.
        """
        payload: dict[str, Any] = {"region": region}
        if name is not None:
            payload["name"] = name
        payload.update(kwargs)

        response = self._request("POST", "/v1/storage/s3/buy", json_data=payload)
        return BucketProvisionResponse(**response.json())

    def presign_upload(
        self,
        bucket_id: str,
        *,
        key: str,
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        **kwargs: Any,
    ) -> PresignedUrlResponse:
        """Get a presigned URL for uploading an object.

        Args:
            bucket_id: The bucket ID.
            key: Object key (path within the bucket).
            content_type: MIME type of the file.
            expires_in: URL expiry in seconds (default: 3600).
            **kwargs: Additional parameters.

        Returns:
            PresignedUrlResponse with upload URL and headers.
        """
        payload: dict[str, Any] = {"key": key, "expires_in": expires_in}
        if content_type is not None:
            payload["content_type"] = content_type
        payload.update(kwargs)

        response = self._request(
            "POST",
            f"/v1/storage/s3/buckets/{bucket_id}/objects/presign-upload",
            json_data=payload,
        )
        return PresignedUrlResponse(**response.json())

    def presign_download(
        self,
        bucket_id: str,
        *,
        key: str,
        expires_in: int = 3600,
        **kwargs: Any,
    ) -> PresignedUrlResponse:
        """Get a presigned URL for downloading an object.

        Args:
            bucket_id: The bucket ID.
            key: Object key (path within the bucket).
            expires_in: URL expiry in seconds (default: 3600).
            **kwargs: Additional parameters.

        Returns:
            PresignedUrlResponse with download URL.
        """
        payload: dict[str, Any] = {"key": key, "expires_in": expires_in}
        payload.update(kwargs)

        response = self._request(
            "POST",
            f"/v1/storage/s3/buckets/{bucket_id}/objects/presign-download",
            json_data=payload,
        )
        result = PresignedUrlResponse(**response.json())
        result.method = "GET"
        return result

    # --- Lifecycle ---

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "FotoHub":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncFotoHub(_BaseClient):
    """Asynchronous FOTOhub API client.

    Usage:
        from fotohub import AsyncFotoHub

        async with AsyncFotoHub(api_key="your-api-key") as client:
            result = await client.generate_image(prompt="A sunset over mountains")
            print(result.images[0].url)
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

    # --- Image Generation ---

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Model to use. Default: seedream-5-0-260128 (SeedDream 5.0 Lite).
            width: Image width in pixels (default: 1024).
            height: Image height in pixels (default: 1024).
            num_images: Number of images to generate (default: 1).
            negative_prompt: Things to avoid in the image.
            seed: Random seed for reproducibility.
            guidance_scale: How closely to follow the prompt (model-dependent).
            steps: Number of diffusion steps (model-dependent).
            **kwargs: Additional model-specific parameters.

        Returns:
            ImageGenerationResponse with generated image URLs and metadata.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "num_images": num_images,
        }
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if steps is not None:
            payload["steps"] = steps
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/generate/image", json_data=payload)
        return ImageGenerationResponse(**response.json())

    # --- Video Generation ---

    async def generate_video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        image_url: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoJob:
        """Start an asynchronous video generation job.

        Args:
            prompt: Text description of the desired video.
            model: Video model to use.
            duration: Desired video duration in seconds.
            image_url: Reference image URL for image-to-video.
            aspect_ratio: Aspect ratio (e.g., "16:9", "9:16", "1:1").
            **kwargs: Additional model-specific parameters.

        Returns:
            VideoJob with job_id and initial status.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        if duration is not None:
            payload["duration"] = duration
        if image_url is not None:
            payload["image_url"] = image_url
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/generate/video", json_data=payload)
        return VideoJob(**response.json())

    async def poll_video(
        self,
        job_id: str,
        *,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> VideoJob:
        """Poll a video generation job until completion or timeout.

        Args:
            job_id: The job ID returned by generate_video().
            poll_interval: Seconds between status checks (default: 5).
            max_wait: Maximum seconds to wait (default: 600).

        Returns:
            VideoJob with final status and video_url (if completed).

        Raises:
            VideoJobTimeoutError: If max_wait is exceeded.
        """
        import asyncio

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                raise VideoJobTimeoutError(
                    message=f"Video job {job_id} did not complete within {max_wait}s",
                    job_id=job_id,
                )

            response = await self._request("GET", f"/v1/ai/generate/video/{job_id}")
            job = VideoJob(**response.json())

            if job.status in ("completed", "failed"):
                return job

            await asyncio.sleep(poll_interval)

    # --- Music Generation ---

    async def generate_music(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        **kwargs: Any,
    ) -> MusicGenerationResponse:
        """Generate music from a text description.

        Args:
            prompt: Text description of the desired music.
            model: Music generation model to use.
            duration: Desired duration in seconds.
            **kwargs: Additional model-specific parameters.

        Returns:
            MusicGenerationResponse with audio URL and metadata.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if model is not None:
            payload["model"] = model
        if duration is not None:
            payload["duration"] = duration
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/generate/music", json_data=payload)
        return MusicGenerationResponse(**response.json())

    # --- Chat / LLM ---

    async def chat(
        self,
        messages: list[Union[dict[str, str], ChatMessage]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, AsyncChatStream]:
        """Send a chat completion request (OpenAI-compatible).

        Args:
            messages: List of chat messages.
            model: LLM model to use.
            stream: If True, returns an AsyncChatStream iterator.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional parameters.

        Returns:
            ChatCompletion if stream=False, AsyncChatStream if stream=True.
        """
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append({"role": msg.role.value, "content": msg.content})
            else:
                formatted_messages.append(msg)

        payload: dict[str, Any] = {"messages": formatted_messages, "stream": stream}
        if model is not None:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(kwargs)

        if stream:
            response = await self._request("POST", "/v1/ai/chat", json_data=payload, stream=True)
            return AsyncChatStream(response)

        response = await self._request("POST", "/v1/ai/chat", json_data=payload)
        return ChatCompletion(**response.json())

    # --- Translation ---

    async def translate(
        self,
        text: str,
        *,
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranslationResult:
        """Translate text between languages (no auth required).

        Args:
            text: Text to translate.
            target_language: Target language code (e.g., "en", "pl", "de").
            source_language: Source language code (auto-detected if not provided).
            **kwargs: Additional parameters.

        Returns:
            TranslationResult with translated text.
        """
        payload: dict[str, Any] = {
            "text": text,
            "target_language": target_language,
        }
        if source_language is not None:
            payload["source_language"] = source_language
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/translate", json_data=payload)
        return TranslationResult(**response.json())

    # --- Gabriel (Intent Orchestration) ---

    async def gabriel(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GabrielResponse:
        """Send a message to the Gabriel intent orchestration engine (no auth required).

        Args:
            message: User message to process.
            context: Optional context dictionary.
            **kwargs: Additional parameters.

        Returns:
            GabrielResponse with detected intent and response.
        """
        payload: dict[str, Any] = {"message": message}
        if context is not None:
            payload["context"] = context
        payload.update(kwargs)

        response = await self._request("POST", "/v1/ai/gabriel", json_data=payload)
        return GabrielResponse(**response.json())

    # --- Usage Analytics ---

    async def get_usage(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> UsageResponse:
        """Get usage analytics for your account.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            category: Filter by category (image, video, chat, music).

        Returns:
            UsageResponse with credit usage breakdown.
        """
        params: dict[str, Any] = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if category is not None:
            params["category"] = category

        response = await self._request("GET", "/v1/usage", params=params)
        return UsageResponse(**response.json())

    # --- Storage Buckets ---

    async def list_buckets(self) -> BucketListResponse:
        """List all storage buckets for your account."""
        response = await self._request("GET", "/v1/buckets")
        return BucketListResponse(**response.json())

    async def create_bucket(
        self,
        name: str,
        *,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> StorageBucket:
        """Create a new storage bucket.

        Args:
            name: Bucket name.
            region: AWS region (default: eu-central-1).
            **kwargs: Additional parameters.

        Returns:
            StorageBucket with the created bucket details.
        """
        payload: dict[str, Any] = {"name": name}
        if region is not None:
            payload["region"] = region
        payload.update(kwargs)

        response = await self._request("POST", "/v1/buckets", json_data=payload)
        return StorageBucket(**response.json())

    async def provision_s3_bucket(
        self,
        *,
        name: Optional[str] = None,
        region: str = "eu-central-1",
        **kwargs: Any,
    ) -> BucketProvisionResponse:
        """Provision a dedicated S3 bucket.

        Args:
            name: Desired bucket name (auto-generated if not provided).
            region: AWS region (default: eu-central-1).
            **kwargs: Additional parameters.

        Returns:
            BucketProvisionResponse with bucket details and credentials.
        """
        payload: dict[str, Any] = {"region": region}
        if name is not None:
            payload["name"] = name
        payload.update(kwargs)

        response = await self._request("POST", "/v1/storage/s3/buy", json_data=payload)
        return BucketProvisionResponse(**response.json())

    async def presign_upload(
        self,
        bucket_id: str,
        *,
        key: str,
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        **kwargs: Any,
    ) -> PresignedUrlResponse:
        """Get a presigned URL for uploading an object.

        Args:
            bucket_id: The bucket ID.
            key: Object key (path within the bucket).
            content_type: MIME type of the file.
            expires_in: URL expiry in seconds (default: 3600).
            **kwargs: Additional parameters.

        Returns:
            PresignedUrlResponse with upload URL and headers.
        """
        payload: dict[str, Any] = {"key": key, "expires_in": expires_in}
        if content_type is not None:
            payload["content_type"] = content_type
        payload.update(kwargs)

        response = await self._request(
            "POST",
            f"/v1/storage/s3/buckets/{bucket_id}/objects/presign-upload",
            json_data=payload,
        )
        return PresignedUrlResponse(**response.json())

    async def presign_download(
        self,
        bucket_id: str,
        *,
        key: str,
        expires_in: int = 3600,
        **kwargs: Any,
    ) -> PresignedUrlResponse:
        """Get a presigned URL for downloading an object.

        Args:
            bucket_id: The bucket ID.
            key: Object key (path within the bucket).
            expires_in: URL expiry in seconds (default: 3600).
            **kwargs: Additional parameters.

        Returns:
            PresignedUrlResponse with download URL.
        """
        payload: dict[str, Any] = {"key": key, "expires_in": expires_in}
        payload.update(kwargs)

        response = await self._request(
            "POST",
            f"/v1/storage/s3/buckets/{bucket_id}/objects/presign-download",
            json_data=payload,
        )
        result = PresignedUrlResponse(**response.json())
        result.method = "GET"
        return result

    # --- Lifecycle ---

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncFotoHub":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
