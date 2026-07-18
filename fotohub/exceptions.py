"""Custom exceptions for the FOTOhub SDK."""

from __future__ import annotations

from typing import Any, Optional


class FotoHubError(Exception):
    """Base exception for all FOTOhub SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


class AuthError(FotoHubError):
    """Raised when authentication fails (401/403)."""

    def __init__(
        self,
        message: str = "Authentication failed. Check your API key.",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class RateLimitError(FotoHubError):
    """Raised when the API rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please retry after a delay.",
        *,
        retry_after: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InsufficientCreditsError(FotoHubError):
    """Raised when the account does not have enough credits (402)."""

    def __init__(
        self,
        message: str = "Insufficient credits. Please top up your account.",
        *,
        credits_required: Optional[float] = None,
        credits_available: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.credits_required = credits_required
        self.credits_available = credits_available


class ValidationError(FotoHubError):
    """Raised when the request parameters are invalid (400/422)."""

    def __init__(
        self,
        message: str = "Invalid request parameters.",
        *,
        errors: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.errors = errors or []


class ServerError(FotoHubError):
    """Raised when the server returns a 5xx error."""

    def __init__(
        self,
        message: str = "Server error. Please try again later.",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class TimeoutError(FotoHubError):
    """Raised when a request times out."""

    def __init__(
        self,
        message: str = "Request timed out.",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class VideoJobTimeoutError(FotoHubError):
    """Raised when polling a video job exceeds the maximum wait time."""

    def __init__(
        self,
        message: str = "Video job polling timed out.",
        *,
        job_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.job_id = job_id
