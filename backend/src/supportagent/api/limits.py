"""Limits that keep one visitor from using up what the deployment pays for.

Every new conversation costs a copy of the mock CRM, and every message costs model calls. Both
limits live in this process's memory or in the session's own stored events, so they need no extra
service. With several backend processes, the limit on new conversations would be per process.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

_SWEEP_ABOVE = 10_000  # keys held before the ones that have gone quiet are dropped


class SlidingWindow:
    """At most `limit` events per key in any stretch of `window_s` seconds."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._seen: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record one event for `key` unless it is over the limit. A refused event is not counted,
        so a client that keeps knocking does not push its own wait further out."""
        now = self._clock()
        if len(self._seen) > _SWEEP_ABOVE:
            self._seen = {k: q for k, q in self._seen.items() if q and now - q[-1] < self.window_s}
        times = self._seen.setdefault(key, deque())
        while times and now - times[0] >= self.window_s:
            times.popleft()
        if len(times) >= self.limit:
            return False
        times.append(now)
        return True


def client_address(direct: str | None, from_proxy: str | None) -> str:
    """The address to count a visitor by.

    `from_proxy` is the value of the header named in CLIENT_ADDRESS_HEADER, in which the host's
    proxy reports the visitor, for example X-Real-IP on Railway. Name only a header that the
    proxy sets on every request, replacing whatever the client sent, or a visitor can pick its
    own address. X-Forwarded-For is not such a header: on Railway its last entry was the same for
    every visitor, which turned the limit per visitor into one limit for the world.
    """
    return (from_proxy or "").strip() or direct or "unknown"
