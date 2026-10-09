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
        code: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        #: Machine-readable code from an ``{"error": {"code": ...}}`` envelope
        #: (e.g. ``save-conflict``, ``media-not-found``, ``rate-limited``), else ``None``.
        self.code = code
        #: The envelope's ``details`` (validation paths, ``currentSaveRev``, ...), else ``None``.
        self.details = details

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


class SaveConflictError(FotoHubError):
    """Raised when a video project changed since you read it (409 ``save-conflict``).

    Someone else (the editor in a browser, another agent) saved first, so your
    ``expected_save_rev`` is stale and **nothing was written**. Re-read the
    project with ``get_video_project`` and re-apply your operations on top of
    :attr:`current_save_rev`.
    """

    def __init__(
        self,
        message: str = "The project was modified since you read it.",
        *,
        current_save_rev: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        #: The project's ``saveRev`` at the moment of the conflict.
        self.current_save_rev = current_save_rev


class VideoJobFailedError(FotoHubError):
    """Raised by ``wait_for_video_job`` when a render/capture job ends ``failed`` or ``cancelled``.

    :attr:`refunded` says whether the charge was already returned to your wallet.
    For a failed Auto-Edit run :attr:`code` and :attr:`reason` are the run's
    ``error.code``; after a ``save-conflict`` :attr:`current_save_rev` is the
    project revision now and :attr:`draft_id` the kept draft (apply it with
    ``apply_video_auto_edit(..., expected_save_rev=current_save_rev)``).
    """

    def __init__(
        self,
        message: str = "Video job failed.",
        *,
        job_id: Optional[str] = None,
        reason: Optional[str] = None,
        refunded: Optional[bool] = None,
        current_save_rev: Optional[int] = None,
        draft_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.job_id = job_id
        self.reason = reason
        self.refunded = refunded
        self.current_save_rev = current_save_rev
        self.draft_id = draft_id


class PriceChangedError(FotoHubError):
    """Raised when the price moved away from the quote you confirmed (409 ``price_changed``).

    Sent by the routes that take ``quote_credits`` (Upscale Pro video, the AI video routes):
    the start was refused **before anything was charged**. Show :attr:`current_credits`
    (or quote again), and resend with the new figure as ``quote_credits``. For the AI
    video routes keep the same ``request_id``.
    """

    def __init__(
        self,
        message: str = "The price changed since it was shown.",
        *,
        quoted_credits: Optional[float] = None,
        current_credits: Optional[float] = None,
        billed_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "price_changed")
        super().__init__(message, **kwargs)
        #: The ``quote_credits`` you sent.
        self.quoted_credits = quoted_credits
        #: The price now, in the same unit.
        self.current_credits = current_credits
        #: Upscale Pro video only: the seconds the measured file is billed for.
        self.billed_seconds = billed_seconds

    @property
    def charged(self) -> bool:
        """Always ``False``: a refused start is never billed."""
        return False


class UrlBlockedError(ValidationError):
    """Raised when a URL you passed was refused before any work (400 ``url_blocked``).

    API-key callers must pass public ``https://`` URLs on the default port 443, without
    credentials, whose host resolves to public addresses only. :attr:`field` names the
    parameter (list items as ``image_urls[2]``); the URL is never echoed back. Nothing
    was charged. A subclass of :class:`ValidationError`, so existing handlers still catch it.
    ``code`` is ``url_blocked`` (or ``url_not_allowed`` for browser-session callers).
    """

    def __init__(
        self,
        message: str = "A URL in the request was refused.",
        *,
        field: Optional[str] = None,
        charged: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "url_blocked")
        super().__init__(message, **kwargs)
        #: The request parameter that carried the refused URL.
        self.field = field
        #: Always ``False`` in practice: the URL is checked before billing.
        self.charged = charged


class PricingNotConfiguredError(ServerError):
    """Raised when the model has no price configured (``PRICING_NOT_CONFIGURED``).

    The render was refused before any money moved and is never billed at a guessed
    price. Usually 503; when the refusal comes from further down the pipeline the API
    answers 424 with the code inside the message (also mapped here). Retrying does not
    help until the price is configured; try another model or contact support.
    """

    def __init__(
        self,
        message: str = "This model has no price configured. You were not charged.",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "PRICING_NOT_CONFIGURED")
        super().__init__(message, **kwargs)

    @property
    def charged(self) -> bool:
        """Always ``False``."""
        return False
