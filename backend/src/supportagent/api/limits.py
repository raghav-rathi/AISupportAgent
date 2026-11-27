"""Limits that keep one visitor, or all of them together, from using up what the deployment pays for.

Every new conversation costs a copy of the mock CRM, and every message costs model calls. The
limits per visitor live in this process's memory, so with several backend processes each would
count for itself. The daily budget is read from the stored events, so it holds across restarts
and processes.
"""

from __future__ import annotations

import ipaddress
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from psycopg_pool import AsyncConnectionPool

_MAX_KEYS = 20_000  # visitors remembered at once. Beyond it the one seen longest ago is dropped.
# A model call without a configured price still costs something. It counts as this much.
_UNPRICED_CALL_USD = 0.002


class SlidingWindow:
    """At most `limit` events per key in any stretch of `window_s` seconds.

    Every operation takes constant time however many keys there are. Keys are kept in the order
    they were last seen: the front of that order is swept a little on each call, and when there
    are more keys than `max_keys` the oldest goes. An attacker with many addresses can therefore
    make this forget quiet visitors early, and cannot make it slow or large.
    """

    def __init__(
        self,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = _MAX_KEYS,
    ) -> None:
        self.limit, self.window_s, self._clock, self._max_keys = limit, window_s, clock, max_keys
        self._seen: OrderedDict[str, deque[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._seen)

    def _sweep(self, now: float) -> None:
        """Drop up to two keys from the front whose last event has left the window."""
        for _ in range(2):
            if not self._seen:
                return
            key, times = next(iter(self._seen.items()))
            if times and now - times[-1] < self.window_s:
                return
            del self._seen[key]

    def allow(self, key: str) -> bool:
        """Record one event for `key` unless it is over the limit. A refused event is not counted,
        so a client that keeps knocking does not push its own wait further out."""
        now = self._clock()
        self._sweep(now)
        times = self._seen.get(key)
        if times is None:
            times = self._seen[key] = deque()
            if len(self._seen) > self._max_keys:
                self._seen.popitem(last=False)
        while times and now - times[0] >= self.window_s:
            times.popleft()
        if len(times) >= self.limit:
            return False
        times.append(now)
        self._seen.move_to_end(key)
        return True

    def undo(self, key: str) -> None:
        """Take back the newest event of `key`: it was counted for something that then did not
        happen, for example a conversation the server had no room for."""
        times = self._seen.get(key)
        if times:
            times.pop()


class Counter:
    """How many of something each key holds right now, with a cap. For open connections."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._held: dict[str, int] = {}

    def take(self, key: str) -> bool:
        if self._held.get(key, 0) >= self.limit:
            return False
        self._held[key] = self._held.get(key, 0) + 1
        return True

    def give_back(self, key: str) -> None:
        left = self._held.get(key, 0) - 1
        if left > 0:
            self._held[key] = left
        else:
            self._held.pop(key, None)


def client_address(direct: str | None, from_proxy: str | None) -> str:
    """The address to count a visitor by.

    `from_proxy` is the value of the header named in CLIENT_ADDRESS_HEADER, in which the host's
    proxy reports the visitor, for example X-Real-IP on Railway. Name only a header that the
    proxy sets on every request, replacing whatever the client sent, or a visitor can pick its
    own address. X-Forwarded-For is not such a header: on Railway its last entry was the same for
    every visitor, which turned the limit per visitor into one limit for the world.

    The address is put into one form, so that one visitor is one key. An IPv6 address counts by
    its /64, the block a single subscriber is given: counted in full, one home connection would
    have as many keys as it cares to use.
    """
    raw = (from_proxy or "").strip() or (direct or "").strip()
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return raw.lower() or "unknown"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network((address, 64), strict=False).network_address) + "/64"
    return str(address)


class Budget:
    """What every visitor together may spend on models in 24 hours.

    The figure is the cost the stored events report: every model call, every search on a hosted
    reranker, and what speech to text and text to speech charged. It is read at most once per
    `refresh_s`, so a burst can overshoot by what is started within that time. The provider's
    own cap on the key stays the hard limit.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool,
        limit_usd: float | None,
        *,
        refresh_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pool, self.limit_usd, self._refresh_s, self._clock = pool, limit_usd, refresh_s, clock
        self._spent = 0.0
        self._read_at: float | None = None

    async def spent(self) -> float:
        """Dollars spent in the last 24 hours, as of the last reading."""
        now = self._clock()
        if self._read_at is None or now - self._read_at >= self._refresh_s:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT coalesce(sum(CASE WHEN type = 'llm.finished' "
                    "THEN coalesce((payload->>'cost_usd')::float8, %s) "
                    "ELSE coalesce((payload->>'cost_usd')::float8, 0) END), 0) AS usd "
                    "FROM session_events WHERE created_at > now() - interval '24 hours' "
                    "AND type IN ('llm.finished', 'retrieval.result', 'speech.heard', "
                    "'speech.spoken', 'voice.usage')",
                    (_UNPRICED_CALL_USD,),
                )
                self._spent = float((await cur.fetchone())["usd"])
            self._read_at = now
        return self._spent

    async def exhausted(self) -> bool:
        return self.limit_usd is not None and await self.spent() >= self.limit_usd
