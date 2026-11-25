"""One clock for the whole app, so tests and evals can pin the date."""

from __future__ import annotations

from datetime import UTC, date, datetime

_frozen: datetime | None = None


def freeze(moment: datetime | None) -> None:
    """Pin the clock to `moment`, or pass None to release it."""
    global _frozen
    _frozen = moment


def now() -> datetime:
    return _frozen if _frozen is not None else datetime.now(UTC)


def today() -> date:
    return now().date()
