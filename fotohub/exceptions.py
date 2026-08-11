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


class InsufficientFundsError(FotoHubError):
    """Raised when the prepaid USD wallet cannot cover the request (402).

    The FOTOhub API is prepaid in dollars. Credits exist only in the fotohub.app
    web app and can never pay for an API call, so this error is about money: the
    request was refused, **nothing was charged**, and the wallet needs funding at
    :attr:`topup_url`.

    ::

        try:
            client.generate_image(prompt="a cat")
        except InsufficientFundsError as e:
            print(f"Need ${e.shortfall_usd} more — top up: {e.topup_url}")

    ``credits_required`` / ``credits_available`` are deprecated. They are only
    ever populated by a pre-cutover server and are removed in the next major
    version.
    """

    def __init__(
        self,
        message: str = (
            "Insufficient funds in your prepaid wallet. "
            "Top up at https://fotohub.app/console/wallet."
        ),
        *,
        required_usd: Optional[float] = None,
        balance_usd: Optional[float] = None,
        shortfall_usd: Optional[float] = None,
        topup_url: Optional[str] = None,
        operation: Optional[str] = None,
        credits_required: Optional[float] = None,
        credits_available: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        #: USD price of the refused request.
        self.required_usd = required_usd
        #: USD wallet balance at the moment of the refusal.
        self.balance_usd = balance_usd
        #: Smallest top-up that would let this request through.
        self.shortfall_usd = shortfall_usd
        #: Where to add funds.
        self.topup_url = topup_url
        #: The refused operation, e.g. ``generate_image:seedream-5-0-pro``.
        self.operation = operation
        #: Deprecated. Only a pre-cutover server sends this.
        self.credits_required = credits_required
        #: Deprecated. Only a pre-cutover server sends this.
        self.credits_available = credits_available

    @property
    def charged(self) -> bool:
        """Always ``False`` — a refused request is never billed."""
        return False


#: Deprecated alias. The API is prepaid in USD and has no credits, so the error
#: was renamed :class:`InsufficientFundsError`. This is the SAME class object, so
#: an existing ``except InsufficientCreditsError`` keeps working and also catches
#: the new name. Removed in the next major version.
InsufficientCreditsError = InsufficientFundsError


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
