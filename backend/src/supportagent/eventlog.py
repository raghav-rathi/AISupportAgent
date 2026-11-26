"""Numbers a session's events and stores them.

Every event gets a sequence number, a timestamp and its turn. A client that reconnects says the last
number it saw and gets what it missed. Token deltas are numbered but not stored: the final message
event carries the full text, and one row per token would be most of the table.

Because deltas are not stored, the highest stored number is not the last number handed out. The
session row keeps that number itself, so a turn that was cut off in the middle of an answer cannot
make the next turn reuse numbers a client has already seen.

The session row also says which turn is running. Starting a turn claims the session, and a second
process that tries the same finds it taken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from supportagent.agent.context import TurnUsage
from supportagent.events import EVENT_ADAPTER, AnyEvent
from supportagent.redact import Redactor

_NOT_STORED = frozenset({"llm.delta"})
# A claim older than this belongs to a process that died in the middle of a turn.
_STALE_CLAIM_S = 600

_PAST_USAGE = """
SELECT
    count(*) FILTER (WHERE type = 'llm.finished') AS llm_calls,
    count(*) FILTER (WHERE type = 'tool.started') AS tool_calls,
    coalesce(sum((payload->>'input_tokens')::bigint) FILTER (WHERE type = 'llm.finished'), 0)
        AS input_tokens,
    coalesce(sum((payload->>'output_tokens')::bigint) FILTER (WHERE type = 'llm.finished'), 0)
        AS output_tokens,
    coalesce(sum((payload->>'cost_usd')::float8) FILTER (WHERE type = 'llm.finished'), 0)
        AS priced_usd,
    count(*) FILTER (WHERE type = 'llm.finished' AND payload->>'cost_usd' IS NULL)
        AS unpriced_calls,
    coalesce(
        sum((payload->>'total_ms')::float8)
            FILTER (WHERE type IN ('turn.failed', 'turn.interrupted')),
        0
    ) AS elapsed_ms
FROM session_events
WHERE session_id = %s AND turn_id = %s
"""


class TurnInProgress(Exception):
    """The session is already running a turn."""


@dataclass(frozen=True)
class PastTurn:
    """The turn a session ran last, for picking it up again."""

    turn_id: str
    mode: str
    usage: TurnUsage  # what its earlier attempts used
    elapsed_ms: float  # how long its earlier attempts ran


class EventLog:
    def __init__(
        self, pool: AsyncConnectionPool, session_id: str, world_id: str, last_seq: int
    ) -> None:
        self._pool = pool
        self.session_id = session_id
        self.world_id = world_id
        self.last_seq = last_seq
        self._unsaved: list[BaseModel] = []

    @classmethod
    async def open(cls, pool: AsyncConnectionPool, session_id: str, world_id: str) -> EventLog:
        """Open the log of a session, creating the session on first use.

        A session that already exists keeps the world it was created with.
        """
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, world_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (session_id, world_id),
            )
            cur = await conn.execute(
                "SELECT world_id, last_seq FROM sessions WHERE id = %s", (session_id,)
            )
            row = await cur.fetchone()
        return cls(pool, session_id, row["world_id"], row["last_seq"])

    async def turn_count(self) -> int:
        """How many customer messages this session has stored."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM session_events "
                "WHERE session_id = %s AND type = 'turn.started'",
                (self.session_id,),
            )
            return (await cur.fetchone())["n"]

    def stamp(self, event: BaseModel, turn_id: str) -> BaseModel:
        """Give the event its place in the stream. Call `save` before the turn is over."""
        self.last_seq += 1
        stamped = event.model_copy(
            update={"seq": self.last_seq, "ts": time.time(), "turn_id": turn_id}
        )
        if stamped.type not in _NOT_STORED:
            self._unsaved.append(stamped)
        return stamped

    async def save(self, redactor: Redactor | None = None, *, release: str | None = None) -> None:
        """Write the events stamped since the last save, and the stream's position.

        With a redactor, each event is masked once more first: by now the redactor knows every PIN
        the turn has used, including ones it learned after an event went out. `release` names a
        turn whose claim on the session ends with this write. Events stay queued until the write
        has committed, so a failed save loses nothing and the next one writes them.
        """
        batch = list(self._unsaved)
        rows = []
        for event in batch:
            if redactor is not None:
                event = redactor.event(event)
            rows.append(
                (self.session_id, event.seq, event.turn_id, event.type, Jsonb(event.model_dump(mode="json")))
            )  # fmt: skip
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if rows:
                await cur.executemany(
                    "INSERT INTO session_events (session_id, seq, turn_id, type, payload) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (session_id, seq) DO NOTHING",
                    rows,
                )
            await cur.execute(
                "UPDATE sessions SET last_seq = greatest(last_seq, %s), "
                "turn_running = turn_running AND turn_id IS DISTINCT FROM %s WHERE id = %s",
                (self.last_seq, release, self.session_id),
            )
        del self._unsaved[: len(batch)]

    async def since(self, seq: int) -> list[AnyEvent]:
        """The stored events after `seq`, in order."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT payload FROM session_events WHERE session_id = %s AND seq > %s ORDER BY seq",
                (self.session_id, seq),
            )
            return [EVENT_ADAPTER.validate_python(row["payload"]) for row in await cur.fetchall()]

    async def claim(self, turn_id: str, mode: str) -> None:
        """Mark the session as running this turn. Raises TurnInProgress when it is taken."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE sessions SET turn_id = %s, turn_mode = %s, turn_running = true, "
                "turn_started_at = now() WHERE id = %s AND (NOT turn_running "
                "OR turn_started_at < now() - make_interval(secs => %s)) RETURNING last_seq",
                (turn_id, mode, self.session_id, _STALE_CLAIM_S),
            )
            row = await cur.fetchone()
        if row is None:
            raise TurnInProgress(self.session_id)
        # Another process may have run turns on this session since this log was opened.
        self.last_seq = max(self.last_seq, row["last_seq"])

    async def last_turn(self) -> PastTurn | None:
        """The turn this session ran last, with what its attempts so far used."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT turn_id, turn_mode FROM sessions WHERE id = %s", (self.session_id,)
            )
            row = await cur.fetchone()
            if row is None or row["turn_id"] is None:
                return None
            cur = await conn.execute(_PAST_USAGE, (self.session_id, row["turn_id"]))
            past = await cur.fetchone()
        elapsed_ms = past.pop("elapsed_ms")
        return PastTurn(row["turn_id"], row["turn_mode"], TurnUsage(**past), elapsed_ms)
