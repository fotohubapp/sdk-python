"""FOTOhub Python SDK — Official client for the FOTOhub AI Platform.

Generate images, videos, music, speech, and more with 25+ AI models through a single API.

Usage::

    from fotohub import FotoHub

    client = FotoHub(api_key="your-api-key")
    result = client.generate_image(prompt="A sunset over mountains")
    print(result["images"][0]["url"])

Async usage::

    from fotohub import AsyncFotoHub

    async with AsyncFotoHub(api_key="your-api-key") as client:
        result = await client.generate_image(prompt="A sunset over mountains")
        print(result["images"][0]["url"])
"""

from .aiwave import Characters, MusicEdit, ProductShot, ProductShotBatches, VideoEdit
from .client import AsyncFotoHub, FotoHub
from .exceptions import (
    AuthError,
    FotoHubError,
    # Deprecated alias for InsufficientFundsError, kept one major version: the
    # API is prepaid in USD and has no credits. Same class object, so an existing
    # `except InsufficientCreditsError` keeps working.
    InsufficientCreditsError,
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

__version__ = "1.12.0"

__all__ = [
    # Clients
    "FotoHub",
    "AsyncFotoHub",
    # AI Wave 1 namespaces (client.characters, .product_shot, .video_edit, .music_edit)
    "Characters",
    "ProductShot",
    "ProductShotBatches",
    "VideoEdit",
    "MusicEdit",
    # Exceptions
    "FotoHubError",
    "AuthError",
    "RateLimitError",
    "InsufficientFundsError",
    "InsufficientCreditsError",  # deprecated alias
    "ValidationError",
    "ServerError",
    "TimeoutError",
    "VideoJobTimeoutError",
    "VideoJobFailedError",
    "SaveConflictError",
    # Streaming
    "ChatStream",
    "AsyncChatStream",
    # Version
    "__version__",
]
