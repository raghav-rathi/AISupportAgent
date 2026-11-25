"""One hub per live session: it runs turns in the background and fans their events out.

A turn does not belong to a socket. If the socket drops, the turn finishes anyway and its events are
stored. A client that comes back says the last sequence number it saw and gets the rest: stored
events from Postgres, then the events of the turn in flight from memory, then the live stream.

The hub keeps what must carry over from turn to turn while the session is live: the event log with
its numbering, and the redactor with the PINs it has learned.
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
from supportagent.eventlog import EventLog, TurnInProgress
from supportagent.events import TURN_ENDED, ProtocolError
from supportagent.llm import faults as presets
from supportagent.llm.faults import MAX_PENDING, FaultPlan
from supportagent.redact import Redactor
from supportagent.runtime import Runtime
from supportagent.session import NothingToResume, can_resume, ensure_world, run_turn

logger = logging.getLogger(__name__)

# Events a client may lag behind before it is cut off and has to reconnect.
_MAX_LAG = 5_000


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

    def drain(self) -> list[BaseModel]:
        waiting = []
        while not self.queue.empty():
            waiting.append(self.queue.get_nowait())
        return waiting


class SessionHub:
    def __init__(self, rt: Runtime, log: EventLog, on_idle: Callable[[SessionHub], None]) -> None:
        self.rt = rt
        self.log = log
        self.session_id = log.session_id
        self.busy = False  # a turn is running
        self.redactor = Redactor()
        self.faults = FaultPlan()
        self.tool_faults = ToolFaults()
        self._on_idle = on_idle
        self._turn_events: list[BaseModel] = []  # the current or last turn, deltas included
        self._subscribers: set[Subscription] = set()
        self._task: asyncio.Task[None] | None = None

    def _publish(self, event: BaseModel) -> None:
        if event.type in TURN_ENDED:
            # run_turn has stored the turn by now. The session is free before any client hears of
            # it, so a message sent in reply to this event is not turned away.
            self.busy = False
        if event.seq:
            self._turn_events.append(event)
        for subscriber in self._subscribers:
            subscriber.offer(event)

    def start_turn(self, text: str | None, mode: Literal["text", "voice"]) -> bool:
        """Start a turn in the background. `text` None resumes one that stopped early.

        Returns False when a turn is already running. A session handles one turn at a time.
        """
        if self.busy:
            return False
        self.busy = True
        self._turn_events = []
        self._task = asyncio.create_task(self._run(text, mode))
        return True

    async def _run(self, text: str | None, mode: Literal["text", "voice"]) -> None:
        try:
            await run_turn(
                self.rt,
                session_id=self.session_id,
                world_id=self.log.world_id,
                text=text,
                on_event=self._publish,
                mode=mode,
                faults=self.faults,
                tool_faults=self.tool_faults,
                log=self.log,
                redactor=self.redactor,
            )
        except TurnInProgress:
            # Another process is running a turn on this session.
            self._publish(ProtocolError(code="turn_in_progress", message="a turn is running"))
        except NothingToResume:
            self._publish(ProtocolError(code="nothing_to_retry", message="no turn to pick up"))
        except asyncio.CancelledError:
            pass  # cancelled on request or at shutdown. run_turn has told the clients.
        except Exception:
            # run_turn has already told the clients. The traceback is for us.
            logger.exception("turn failed in session %s", self.session_id)
        finally:
            if self._task is asyncio.current_task():
                self.busy = False
            self._on_idle(self)  # the client may have left while the turn ran

    async def resumable(self) -> bool:
        """True when the last turn stopped early and `turn.retry` would pick it up."""
        return not self.busy and await can_resume(self.rt, self.session_id)

    def cancel(self) -> bool:
        """Stop the running turn. Returns False when none is running."""
        if not self.busy or self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    async def stop(self) -> None:
        """Stop the running turn and wait until it has been stored."""
        task = self._task
        if self.cancel() and task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def attach(self, last_seq: int) -> tuple[list[BaseModel], Subscription, int]:
        """Subscribe a client. Returns what it missed, its subscription, and the stream's position."""
        subscription = Subscription()
        # Subscribe before reading anything, so no event can fall between the backlog and the queue.
        self._subscribers.add(subscription)
        try:
            stored = await self.log.since(last_seq)
        except BaseException:
            self.detach(subscription)
            raise
        # No await between here and the return. The three sources are read at one moment: what
        # was stored, what the turn in flight has sent so far, and what arrived while we read.
        recent = [*self._turn_events, *subscription.drain()]
        merged = {e.seq: e for e in [*recent, *stored] if e.seq > last_seq}
        return [merged[seq] for seq in sorted(merged)], subscription, self.log.last_seq

    def detach(self, subscription: Subscription) -> None:
        self._subscribers.discard(subscription)
        self._on_idle(self)

    @property
    def idle(self) -> bool:
        return not self.busy and not self._subscribers

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

    def __len__(self) -> int:
        return len(self._hubs)

    async def open(self, session_id: str | None) -> tuple[SessionHub, bool]:
        """Find or create the hub for a session. The flag says whether the session already existed."""
        if session_id and session_id in self._hubs:
            return self._hubs[session_id], True
        if len(self._hubs) >= self.max_sessions:
            raise TooManySessions
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
        hub = self._hubs.setdefault(session_id, SessionHub(self.rt, log, self._release))
        return hub, resumed

    def _release(self, hub: SessionHub) -> None:
        """Forget a hub nobody is using. Its session lives on in Postgres."""
        if hub.idle and self._hubs.get(hub.session_id) is hub:
            del self._hubs[hub.session_id]

    async def close(self) -> None:
        """Stop every running turn and wait until each has been stored. For shutdown."""
        await asyncio.gather(*(hub.stop() for hub in list(self._hubs.values())))
