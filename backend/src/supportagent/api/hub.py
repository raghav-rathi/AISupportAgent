"""One hub per live session: it runs turns in the background and fans their events out.

A turn does not belong to a socket. If the socket drops, the turn finishes anyway and its events are
stored. A client that comes back says the last sequence number it saw and gets the rest: stored
events from Postgres, then the events of the turn in flight from memory, then the live stream.

The hub keeps what must carry over from turn to turn while the session is live: the event log with
its numbering, and the redactor with the PINs it has learned. Both can be rebuilt from Postgres,
so a hub that is let go and opened again loses nothing.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from supportagent.agent.context import ToolFaults
from supportagent.agent.tools import TOOL_NAMES
from supportagent.api.protocol import FaultInject
from supportagent.eventlog import EventLog, SessionGone, TurnInProgress
from supportagent.events import NOT_STREAMED, TURN_ENDED, ProtocolError, TurnFailed
from supportagent.llm import faults as presets
from supportagent.llm.faults import MAX_PENDING, FaultPlan
from supportagent.runtime import Runtime
from supportagent.session import NothingToResume, can_resume, ensure_world, run_turn

logger = logging.getLogger(__name__)

# Events a client may lag behind before it is cut off and has to reconnect. A long turn has a few
# hundred, most of them token deltas.
_MAX_LAG = 2_000
# Connections to one conversation at a time: a few tabs, not a crowd.
MAX_CLIENTS = 4


class TooManyClients(Exception):
    """This conversation already has as many connections as it may have."""


class Subscription:
    """One client's view of a hub: a bounded queue of events."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[BaseModel] = asyncio.Queue(maxsize=_MAX_LAG)
        self.lost = False  # the client fell too far behind and missed events

    def offer(self, event: BaseModel) -> None:
        if self.lost:
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.lost = True
            # What is queued will not be sent: the client is told to reconnect and gets
            # everything from the store. Dropping it here frees the memory at once.
            self.drain()

    def drain(self) -> list[BaseModel]:
        waiting = []
        while not self.queue.empty():
            waiting.append(self.queue.get_nowait())
        return waiting


class SessionHub:
    def __init__(
        self, rt: Runtime, log: EventLog, on_idle: Callable[[SessionHub], None], turns: int = 0
    ) -> None:
        self.rt = rt
        self.log = log
        self.turns = turns  # customer messages so far, for the limit per conversation
        self.session_id = log.session_id
        self.busy = False  # a turn is running
        self.gone = False  # the session was deleted while this hub was open
        self.redactor = rt.redactor(log)
        self.faults = FaultPlan()
        self.tool_faults = ToolFaults()
        self._on_idle = on_idle
        self._turn_events: list[BaseModel] = []  # the turn in flight, deltas included
        self._subscribers: set[Subscription] = set()
        self._task: asyncio.Task[None] | None = None
        self._working = False  # the task of the last turn has not returned yet

    def _publish(self, event: BaseModel) -> None:
        if event.type in NOT_STREAMED:
            return  # stored by the turn, and read back on request
        if event.type in TURN_ENDED:
            # run_turn has stored the turn by now. The session is free before any client hears of
            # it, so a message sent in reply to this event is not turned away. And a client that
            # attaches from here on reads the turn from the store. The buffer of the turn in
            # flight goes at this moment and not when the turn's task returns: the task still
            # tidies up for a few milliseconds, and a client that arrived in between was sent
            # the finished turn's token deltas, which the store does not hold.
            self.busy = False
            self._turn_events = []
        elif event.seq:
            self._turn_events.append(event)
        for subscriber in self._subscribers:
            subscriber.offer(event)

    def start_turn(
        self, text: str | None, mode: Literal["text", "voice"], client_id: str | None = None
    ) -> bool:
        """Start a turn in the background. `text` None resumes one that stopped early.

        Returns False when a turn is already running. A session handles one turn at a time.
        """
        if self.busy:
            return False
        self.busy = self._working = True
        if text is not None:
            self.turns += 1
        self._turn_events = []
        self._task = asyncio.create_task(self._run(text, mode, client_id, self._task))
        return True

    async def _run(
        self,
        text: str | None,
        mode: Literal["text", "voice"],
        client_id: str | None,
        previous: asyncio.Task[None] | None,
    ) -> None:
        started: str | None = None  # the ID of this turn, once an event of it has gone out
        ended = False  # the event that ends the turn has gone out

        def publish(event: BaseModel) -> None:
            nonlocal started, ended
            started = started or getattr(event, "turn_id", None) or ""
            ended = ended or event.type in TURN_ENDED
            self._publish(event)

        def never_started(code: str, message: str) -> None:
            """Tell the clients the message started no turn, and give its place in the count back."""
            if text is not None:
                self.turns -= 1
            self._publish(ProtocolError(code=code, message=message, client_id=client_id))

        try:
            if previous is not None:
                # The turn before this one has told its clients it is over and may still be
                # tidying up its saved state. Two turns must not write one conversation.
                await asyncio.gather(previous, return_exceptions=True)
            await run_turn(
                self.rt,
                session_id=self.session_id,
                world_id=self.log.world_id,
                text=text,
                on_event=publish,
                mode=mode,
                faults=self.faults,
                tool_faults=self.tool_faults,
                log=self.log,
                redactor=self.redactor,
                client_id=client_id,
            )
        except TurnInProgress:
            # Another process is running a turn on this session.
            never_started("turn_in_progress", "a turn is running")
        except NothingToResume:
            never_started("nothing_to_retry", "no turn to pick up")
        except SessionGone:
            self.gone = True
            never_started("session_expired", "this conversation is no longer stored")
        except asyncio.CancelledError:
            # Cancelled on request or at shutdown. If the turn had begun, run_turn has told the
            # clients. If not, the message that asked for it is answered here.
            if started is None:
                never_started("failed_to_start", "the turn was stopped before it began")
        except Exception:
            # The traceback is for us. If the turn had begun, run_turn has told the clients.
            logger.exception("turn failed in session %s", self.session_id)
            if started is None:
                never_started("failed_to_start", "the turn could not start, try again")
        finally:
            if started is not None and not ended:
                # The turn began and the event that ends it never went out: storing it failed,
                # for example because the session was deleted under it. Clients are told all
                # the same, or they would wait on a turn that is over.
                self._publish(
                    TurnFailed(
                        code="internal",
                        message="the turn could not be saved",
                        retryable=False,
                        turn_id=started or None,
                    )
                )
            if self._task is asyncio.current_task():
                self.busy = self._working = False
                # Stored events come from Postgres from here on. A save that failed keeps its
                # events in the log, where attach finds them.
                self._turn_events = []
            self._on_idle(self)  # the client may have left while the turn ran

    async def resumable(self) -> bool:
        """True when the last turn stopped early and `turn.retry` would pick it up."""
        return not self.busy and await can_resume(self.rt, self.session_id)

    async def retries_used(self) -> int:
        """How often the last turn has failed or been stopped so far."""
        past = await self.log.last_turn()
        return past.stops if past else 0

    def cancel(self) -> bool:
        """Stop the running turn. Returns False when none is running."""
        if not self.busy or self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    async def stop(self) -> None:
        """Stop the running turn and wait until it has been stored and tidied up."""
        task = self._task
        if task is None:
            return
        if self.busy and not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def attach(self, last_seq: int) -> tuple[list[BaseModel], Subscription, int]:
        """Subscribe a client. Returns what it missed, its subscription, and the stream's position.

        Raises TooManyClients when the conversation is at its limit of connections.
        """
        if len(self._subscribers) >= MAX_CLIENTS:
            raise TooManyClients(self.session_id)
        subscription = Subscription()
        # Subscribe before reading anything, so no event can fall between the backlog and the queue.
        self._subscribers.add(subscription)
        try:
            stored = await self.log.since(last_seq)
        except BaseException:
            self.detach(subscription)
            raise
        # No await between here and the return. The sources are read at one moment: what was
        # stored, what a save has not written yet, what the turn in flight has sent so far, and
        # what arrived while we read.
        waiting = [e for e in self.log.unsaved if e.type not in NOT_STREAMED]
        recent = [*waiting, *self._turn_events, *subscription.drain()]
        merged = {e.seq: e for e in [*recent, *stored] if e.seq > last_seq}
        return [merged[seq] for seq in sorted(merged)], subscription, self.log.last_seq

    def detach(self, subscription: Subscription) -> None:
        self._subscribers.discard(subscription)
        self._on_idle(self)

    @property
    def idle(self) -> bool:
        return not self.busy and not self._working and not self._subscribers

    def inject(self, request: FaultInject) -> bool:
        """Queue simulated failures for this session's next calls. False when the request is bad.

        Faults live in this session only, and each queue has a cap.
        """
        if request.kind == "tool_failure":
            name = request.tool or "search_knowledge_base"
            if name not in TOOL_NAMES:
                return False
            queued = self.tool_faults.remaining.get(name, 0) + request.times
            self.tool_faults.remaining[name] = min(MAX_PENDING, queued)
        elif request.kind == "llm_timeout":
            self.faults.add(presets.timeout(delay_s=1.5), request.times)
        elif request.kind == "llm_rate_limit":
            self.faults.add(presets.rate_limit(retry_after_s=1.0), request.times)
        elif request.kind == "llm_server_error":
            self.faults.add(presets.server_error(), request.times)
        elif request.kind == "llm_primary_down":
            # The first model refuses and is not retried, so the fallback model takes the call.
            self.faults.add(presets.quota("primary"), request.times)
        elif request.kind == "llm_down":
            # Every model refuses: one fault per model is enough, because quota errors are not retried.
            self.faults.add(presets.quota("any"), len(self.rt.llm.models) * request.times)
        else:
            models = getattr(self.rt.judge, "model_count", 1)
            self.faults.add(presets.quota("any", purpose="judge"), models * request.times)
        return True


class TooManySessions(Exception):
    """This process is at its limit of live sessions."""


class Hubs:
    """The hubs of the sessions that have a client or a running turn."""

    def __init__(self, rt: Runtime, max_sessions: int = 200) -> None:
        self.rt = rt
        self.max_sessions = max_sessions
        self._hubs: dict[str, SessionHub] = {}
        self._opening = 0  # hubs being built: they hold a place before they exist

    def __len__(self) -> int:
        return len(self._hubs)

    @property
    def full(self) -> bool:
        return len(self._hubs) + self._opening >= self.max_sessions

    async def known(self, session_id: str | None) -> bool:
        """Whether a session with this ID exists. Opening an unknown ID starts a new session."""
        if not session_id:
            return False
        if session_id in self._hubs:
            return not self._hubs[session_id].gone
        async with self.rt.pool.connection() as conn:
            cur = await conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,))
            return await cur.fetchone() is not None

    async def open(self, session_id: str | None) -> tuple[SessionHub, bool]:
        """Find or create the hub for a session. The flag says whether the session already existed."""
        if session_id and session_id in self._hubs and not self._hubs[session_id].gone:
            return self._hubs[session_id], True
        # The place is taken before the first await. Counting only finished hubs would let any
        # number of connections that arrive together all find room.
        if self.full:
            raise TooManySessions
        self._opening += 1
        try:
            world_id = None
            if session_id:
                async with self.rt.pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT world_id FROM sessions WHERE id = %s", (session_id,)
                    )
                    row = await cur.fetchone()
                world_id = row["world_id"] if row else None
            resumed = world_id is not None
            if not resumed:
                # The ID is the only key to the session, so it has to be unguessable.
                session_id = world_id = uuid.uuid4().hex
                await ensure_world(self.rt, world_id)
            log = await EventLog.open(self.rt.pool, session_id, world_id)
            turns = await log.turn_count() if resumed else 0
            hub = self._hubs.setdefault(session_id, SessionHub(self.rt, log, self._release, turns))
            return hub, resumed
        finally:
            self._opening -= 1

    def _release(self, hub: SessionHub) -> None:
        """Forget a hub nobody is using. Its session lives on in Postgres."""
        if hub.idle and self._hubs.get(hub.session_id) is hub:
            del self._hubs[hub.session_id]

    async def close(self) -> None:
        """Stop every running turn and wait until each has been stored. For shutdown."""
        await asyncio.gather(*(hub.stop() for hub in list(self._hubs.values())))
