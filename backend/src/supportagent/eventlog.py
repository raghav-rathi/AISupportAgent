"""Numbers a session's events and stores them.

Every event gets a sequence number, a timestamp and its turn. A client that reconnects says the last
number it saw and gets what it missed. Token deltas are numbered but not stored: the final message
event carries the full text, and one row per token would be most of the table. Prompt traces are
stored but not streamed: a client asks for those of one turn when it shows them.

Because not every number is stored or sent, the highest stored number is not the last number handed
out. The session row keeps that number itself, so a turn that was cut off in the middle of an answer
cannot make the next turn reuse numbers a client has already seen.

The session row also says which turn is running and which process runs it. Starting a turn claims
the session. The claim is a lease: the owner renews it while the turn runs, and another process
may take the session only when the lease has run out. So no process ever has to clear claims it
does not own, and two instances that overlap during a deploy cannot both write one conversation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from supportagent.agent.context import TurnUsage
from supportagent.events import EVENT_ADAPTER, NOT_STREAMED, AnyEvent, LlmTrace
from supportagent.redact import Redactor

logger = logging.getLogger(__name__)

_NOT_STORED = frozenset({"llm.delta"})
# A claim that has not been renewed for this long belongs to a process that died in the middle of
# a turn. The owner renews every HEARTBEAT_S, so three renewals have to fail in a row.
LEASE_S = 45.0
HEARTBEAT_S = 15.0
# This process, as the owner of the turns it runs.
INSTANCE = uuid.uuid4().hex[:12]

_PAST_USAGE = """
SELECT
    count(*) FILTER (WHERE type = 'llm.finished') AS llm_calls,
    count(*) FILTER (WHERE type = 'tool.started') AS tool_calls,
    coalesce(sum((payload->>'input_tokens')::bigint) FILTER (WHERE type = 'llm.finished'), 0)
        AS input_tokens,
    coalesce(sum((payload->>'output_tokens')::bigint) FILTER (WHERE type = 'llm.finished'), 0)
        AS output_tokens,
    coalesce(sum((payload->>'cost_usd')::float8) FILTER (WHERE type = 'llm.finished'), 0)
        + coalesce(sum((payload->>'cost_usd')::float8) FILTER (WHERE type = 'retrieval.result'), 0)
        AS priced_usd,
    count(*) FILTER (WHERE type = 'llm.finished' AND payload->>'cost_usd' IS NULL)
        AS unpriced_calls,
    -- Each attempt reports the time of every attempt up to and including itself, so the newest
    -- figure is the total. Adding them up would count the first attempt once per retry.
    coalesce(
        max((payload->>'total_ms')::float8)
            FILTER (WHERE type IN ('turn.failed', 'turn.interrupted')),
        0
    ) AS elapsed_ms,
    count(*) FILTER (WHERE type IN ('turn.failed', 'turn.interrupted')) AS stops
FROM session_events
WHERE session_id = %s AND turn_id = %s
"""


class TurnInProgress(Exception):
    """The session is already running a turn."""


class SessionGone(Exception):
    """The session no longer exists: it was deleted after it went unused for too long."""


@dataclass(frozen=True)
class PastTurn:
    """The turn a session ran last, for picking it up again."""

    turn_id: str
    mode: str
    usage: TurnUsage  # what its earlier attempts used
    elapsed_ms: float  # how long its earlier attempts ran
    stops: int  # how often it has failed or been stopped so far


class EventLog:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        session_id: str,
        world_id: str,
        last_seq: int,
        *,
        secret_hashes: list[str] | None = None,
        owner: str = INSTANCE,
        heard: str | None = None,
    ) -> None:
        self._pool = pool
        self.session_id = session_id
        self.world_id = world_id
        self.last_seq = last_seq
        self.secret_hashes = list(secret_hashes or [])  # see redact.Redactor.hashes
        self.owner = owner
        # What the customer had heard of the last reply when they spoke over it, until a turn
        # has put it into the conversation. See `keep_heard`.
        self.heard = heard
        self._unsaved: list[BaseModel] = []
        # Events stamped by `stamp_apart` and not written yet. See `write_apart`.
        self._apart: list[BaseModel] = []
        # One save at a time. A turn saves at its end, and voice stores events of its own while
        # a turn may be running: two saves that overlap would each drop the other's events from
        # the queue.
        self._saving = asyncio.Lock()

    @classmethod
    async def open(
        cls, pool: AsyncConnectionPool, session_id: str, world_id: str, *, owner: str = INSTANCE
    ) -> EventLog:
        """Open the log of a session, creating the session on first use.

        A session that already exists keeps the world it was created with. Opening counts as
        activity, so a conversation someone has come back to is not purged as unused.
        """
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, world_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (session_id, world_id),
            )
            cur = await conn.execute(
                "UPDATE sessions SET last_active_at = now() WHERE id = %s "
                "RETURNING world_id, last_seq, secret_hashes, heard",
                (session_id,),
            )
            row = await cur.fetchone()
        return cls(
            pool,
            session_id,
            row["world_id"],
            row["last_seq"],
            secret_hashes=row["secret_hashes"],
            owner=owner,
            heard=row["heard"],
        )

    @property
    def unsaved(self) -> list[BaseModel]:
        """Events that have a number and are not in the store yet, in order."""
        return sorted([*self._unsaved, *self._apart], key=lambda event: event.seq)

    async def turn_count(self) -> int:
        """How many customer messages this session has stored."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM session_events "
                "WHERE session_id = %s AND type = 'turn.started'",
                (self.session_id,),
            )
            return (await cur.fetchone())["n"]

    def stamp(self, event: BaseModel, turn_id: str | None) -> BaseModel:
        """Give the event its place in the stream. Call `save` before the turn is over."""
        self.last_seq += 1
        stamped = event.model_copy(
            update={"seq": self.last_seq, "ts": time.time(), "turn_id": turn_id}
        )
        if stamped.type not in _NOT_STORED:
            self._unsaved.append(stamped)
        return stamped

    def stamp_apart(self, event: BaseModel, turn_id: str | None) -> BaseModel:
        """Give an event its place in the stream without adding it to the turn's queue.

        For events that no running turn emits: what voice measured. `write_apart` stores it.
        """
        self.last_seq += 1
        stamped = event.model_copy(
            update={"seq": self.last_seq, "ts": time.time(), "turn_id": turn_id}
        )
        self._apart.append(stamped)
        return stamped

    async def write_apart(self, stamped: BaseModel, redactor: Redactor | None = None) -> None:
        """Store one event of `stamp_apart` now, and leave a running turn's events queued.

        A turn's events wait for the turn's end, where they are masked once more with every
        PIN the turn has learned by then. Voice stores events of its own while a turn runs. If
        writing those took the turn's queue along, the customer's message would be in the
        store before that last pass, and a PIN the first pass did not recognise would stay.
        If the write fails the event stays here, and the next save of any kind writes it.
        """
        async with self._saving:
            if stamped in self._apart:  # not written by a save that got here first
                await self._write([stamped], redactor, None)
                self._apart.remove(stamped)

    async def save(self, redactor: Redactor | None = None, *, release: str | None = None) -> None:
        """Write the events stamped since the last save, and the stream's position.

        With a redactor, each event is masked once more first: by now the redactor knows every PIN
        the turn has used, including ones it learned after an event went out. The hashes of those
        PINs are stored with the session. `release` names a turn whose claim on the session ends
        with this write, if this process still owns it. Events stay queued until the write has
        committed, so a failed save loses nothing and the next one writes them.
        """
        async with self._saving:
            # Events of `stamp_apart` whose own write failed go along.
            apart, turn = list(self._apart), list(self._unsaved)
            await self._write(sorted([*apart, *turn], key=lambda e: e.seq), redactor, release)
            del self._unsaved[: len(turn)]
            for event in apart:
                self._apart.remove(event)

    async def _write(
        self, batch: list[BaseModel], redactor: Redactor | None, release: str | None
    ) -> None:
        rows = []
        for event in batch:
            if redactor is not None:
                event = redactor.event(event)
            rows.append(
                (self.session_id, event.seq, event.turn_id, event.type, Jsonb(event.model_dump(mode="json")))
            )  # fmt: skip
        hashes = redactor.hashes if redactor is not None else []
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if rows:
                await cur.executemany(
                    "INSERT INTO session_events (session_id, seq, turn_id, type, payload) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (session_id, seq) DO NOTHING",
                    rows,
                )
                if cur.rowcount != len(rows):
                    # A number of ours was taken already. Only a second writer on this session
                    # can do that, and the lease on the turn is there to prevent one.
                    logger.error(
                        "session %s: %s of %s events were not stored, their numbers were in use",
                        self.session_id,
                        len(rows) - cur.rowcount,
                        len(rows),
                    )
            await cur.execute(
                "UPDATE sessions SET last_seq = greatest(last_seq, %s), last_active_at = now(), "
                "secret_hashes = ARRAY(SELECT DISTINCT unnest(secret_hashes || %s::text[])), "
                "turn_running = turn_running AND NOT (turn_id IS NOT DISTINCT FROM %s "
                "AND turn_owner IS NOT DISTINCT FROM %s) WHERE id = %s",
                (self.last_seq, hashes, release, self.owner, self.session_id),
            )

    async def since(self, seq: int) -> list[AnyEvent]:
        """The stored events after `seq` that a client is sent, in order."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT payload FROM session_events WHERE session_id = %s AND seq > %s "
                "AND NOT (type = ANY(%s)) ORDER BY seq",
                (self.session_id, seq, list(NOT_STREAMED)),
            )
            return [EVENT_ADAPTER.validate_python(row["payload"]) for row in await cur.fetchall()]

    async def traces(self, turn_id: str) -> list[LlmTrace]:
        """The stored prompt traces of one turn, in order."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT payload FROM session_events WHERE session_id = %s AND turn_id = %s "
                "AND type = 'llm.trace' ORDER BY seq",
                (self.session_id, turn_id),
            )
            return [LlmTrace.model_validate(row["payload"]) for row in await cur.fetchall()]

    async def was_shown(self, chunk_id: str) -> bool:
        """Whether a search of this session handed this passage to the model.

        Only such a passage may be read back. A search that ended in "verify first" or "another
        plan" passed nothing on, so its candidates do not count.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM session_events WHERE session_id = %s AND type = 'retrieval.result' "
                "AND payload->>'band' IN ('answer', 'check') AND payload->'hits' @> %s LIMIT 1",
                (self.session_id, Jsonb([{"chunk_id": chunk_id}])),
            )
            return await cur.fetchone() is not None

    async def keep_heard(self, text: str) -> None:
        """Note what the customer had heard of the last reply when they spoke over it.

        `text` may be empty: they heard none of it. The next new turn takes the note, see
        `claim`. Until then the turn that was cut off cannot be picked up again.
        """
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET heard = %s WHERE id = %s", (text, self.session_id)
            )
        self.heard = text

    async def claim(self, turn_id: str, mode: str, *, resume: bool = False) -> str | None:
        """Mark the session as running this turn, owned by this process.

        Returns what `keep_heard` noted, and clears it: the turn that starts now puts it into
        the conversation. With `resume` the claim is for a turn that is being picked up
        again, and the note stays where it is.

        Raises TurnInProgress when another turn holds a live claim, and SessionGone when the
        session has been deleted.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE sessions s SET turn_id = %s, turn_mode = %s, turn_running = true, "
                "turn_started_at = now(), turn_owner = %s, turn_heartbeat_at = now(), "
                "last_active_at = now(), heard = CASE WHEN %s THEN s.heard END "
                "FROM (SELECT id, heard FROM sessions WHERE id = %s FOR UPDATE) old "
                "WHERE s.id = old.id AND (NOT s.turn_running OR s.turn_heartbeat_at IS NULL "
                "OR s.turn_heartbeat_at < now() - make_interval(secs => %s)) "
                "RETURNING s.last_seq, old.heard",
                (turn_id, mode, self.owner, resume, self.session_id, LEASE_S),
            )
            row = await cur.fetchone()
            if row is None:
                cur = await conn.execute("SELECT 1 FROM sessions WHERE id = %s", (self.session_id,))
                if await cur.fetchone() is None:
                    raise SessionGone(self.session_id)
                raise TurnInProgress(self.session_id)
        # Another process may have run turns on this session since this log was opened.
        self.last_seq = max(self.last_seq, row["last_seq"])
        if resume:
            return None
        self.heard = None
        return row["heard"]

    async def renew(self, turn_id: str) -> bool:
        """Renew this process's claim on the turn. False when the claim is no longer ours."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE sessions SET turn_heartbeat_at = now() WHERE id = %s AND turn_running "
                "AND turn_id = %s AND turn_owner = %s",
                (self.session_id, turn_id, self.owner),
            )
            return cur.rowcount == 1

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
        elapsed_ms, stops = past.pop("elapsed_ms"), past.pop("stops")
        return PastTurn(row["turn_id"], row["turn_mode"], TurnUsage(**past), elapsed_ms, stops)
