"""What can go wrong with a model call, and whether another attempt is worth making."""

from __future__ import annotations

import email.utils
import time
from dataclasses import dataclass

import httpx
import openai
from langchain_core.exceptions import ContextOverflowError, ModelRefusalError

from supportagent.events import FailureReason

# A 429 means "slow down" unless it says the account cannot pay. These codes and words mark the
# second kind: a spend cap, an empty balance, a closed billing account. Waiting never clears them.
_BILLING_CODES = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "account_deactivated",
        "insufficient_funds",
        "payment_required",
    }
)
_BILLING_WORDS = ("billing", "credit", "spend", "budget", "payment", "balance")
# A message that names a time window is a rate limit, whatever else it says. Gemini words its
# per-minute limit as "Quota exceeded ... please check your plan and billing details".
_WINDOW_WORDS = (
    "per minute",
    "per second",
    "per hour",
    "per day",
    "per min",
    "rate limit",
    "rate_limit",
    "requests per",
    "tokens per",
    "retry in",
    "try again in",
)
# 408 request timeout, 409 lock timeout, 425 too early, the usual 5xx, and 520 to 529, which
# proxies and Anthropic use for "overloaded". A 501 or a 505 will not get better.
_RETRYABLE_STATUS = frozenset({409, 425, 500, 502, 503, 504, *range(520, 530)})


@dataclass(frozen=True)
class Failure:
    reason: FailureReason
    retryable: bool  # worth another attempt on the same model
    retry_after_s: float | None = None
    detail: str = ""


class ModelFault(Exception):
    """A failure this client raises itself, already classified."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(f"{failure.reason} {failure.detail}".strip())
        self.failure = failure


class InjectedFault(ModelFault):
    """A failure queued by a tester.

    It takes the same path as the real failure it imitates, except that its retries are not
    charged to the retry budget, which exists to protect a real upstream.
    """


class BadCompletion(ModelFault):
    """The model finished without anything usable: no text and no valid tool call."""


class LLMUnavailable(Exception):
    """Every configured model failed. `failures` lists what happened, in order."""

    def __init__(self, failures: list[Failure]) -> None:
        super().__init__("; ".join(f"{f.reason} {f.detail}".strip() for f in failures))
        self.failures = failures


def _retry_after(exc: openai.APIStatusError) -> float | None:
    """Seconds the server asked us to wait: `retry-after-ms`, or `retry-after` as seconds or a date."""
    headers = exc.response.headers
    try:
        return max(0.0, float(headers["retry-after-ms"]) / 1000)
    except (KeyError, ValueError):
        pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _is_billing(exc: openai.APIError) -> bool:
    if (exc.code or "").lower() in _BILLING_CODES or (exc.type or "").lower() in _BILLING_CODES:
        return True
    message = (exc.message or "").lower()
    return any(word in message for word in _BILLING_WORDS) and not any(
        word in message for word in _WINDOW_WORDS
    )


def _from_status(exc: openai.APIStatusError) -> Failure:
    status = exc.status_code
    detail = f"HTTP {status}"
    if status == 402:
        return Failure("quota", retryable=False, detail=detail)
    if status == 429:
        if _is_billing(exc):
            return Failure("quota", retryable=False, detail=detail)
        return Failure("rate_limited", True, _retry_after(exc), detail)
    if status == 408:
        return Failure("timeout", True, _retry_after(exc), detail)
    if status in _RETRYABLE_STATUS:
        return Failure("server_error", True, _retry_after(exc), detail)
    return Failure("rejected", retryable=False, detail=detail)


def _from_stream(exc: openai.APIError) -> Failure:
    """An error that arrived inside a stream, after HTTP 200. It carries no status code."""
    label = f"{exc.code or ''} {exc.type or ''}".strip().lower()
    detail = f"stream error {label}".strip()
    if _is_billing(exc):
        return Failure("quota", retryable=False, detail=detail)
    if "rate_limit" in label:
        return Failure("rate_limited", retryable=True, detail=detail)
    return Failure("server_error", retryable=True, detail=detail)


def classify(exc: BaseException) -> Failure:
    """Map an exception from a model call to a Failure. Anything unknown is a bug and re-raises."""
    if isinstance(exc, ModelFault):
        return exc.failure
    if isinstance(exc, TimeoutError | openai.APITimeoutError | httpx.TimeoutException):
        return Failure("timeout", retryable=True)
    if isinstance(exc, openai.APIConnectionError | httpx.TransportError):
        return Failure("connection", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        return _from_status(exc)
    if isinstance(
        exc,
        ContextOverflowError
        | ModelRefusalError
        | openai.LengthFinishReasonError
        | openai.ContentFilterFinishReasonError,
    ):
        return Failure("rejected", retryable=False, detail=type(exc).__name__)
    if isinstance(exc, openai.APIError):
        return _from_stream(exc)
    raise exc
