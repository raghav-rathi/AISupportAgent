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
from typing import TYPE_CHECKING, Literal

from psycopg.errors import ForeignKeyViolation
from pydantic import BaseModel

from supportagent.agent.context import ToolFaults
from supportagent.agent.early import EarlyCall
from supportagent.agent.graph import first_prompt
from supportagent.agent.tools import TOOL_NAMES, TOOLS
from supportagent.api.protocol import FaultInject
from supportagent.desk.bus import DeskBus
from supportagent.eventlog import EventLog, SessionGone, TurnInProgress
from supportagent.events import NOT_STREAMED, TURN_ENDED, ProtocolError, TurnFailed
from supportagent.llm import faults as presets
from supportagent.llm.faults import MAX_PENDING, FaultPlan
from supportagent.runtime import Runtime
from supportagent.session import NothingToResume, can_resume, ensure_world, run_turn

if TYPE_CHECKING:
    from supportagent.voice.session import VoiceSession

logger = logging.getLogger(__name__)

# Events a client may lag behind before it is cut off and has to reconnect. A long turn has a few
# hundred, most of them token deltas.
_MAX_LAG = 2_000
# Connections to one conversation at a time: a few tabs, not a crowd.
MAX_CLIENTS = 4


class TooManyClients(Exception):
    """This conversation already has as many connections as it may have."""


class _First:
    """Wraps what goes to the front of a client's queue."""

    __slots__ = ("item",)

    def __init__(self, item: BaseModel) -> None:
        self.item = item


class _Lane(asyncio.Queue):
    """A queue in which some things do not wait their turn.

    They go in front of everything that waits, and behind those of their own kind that got
    there first: a pause and the resume that follows it must not change places.
    """

    def _init(self, maxsize: int) -> None:
        super()._init(maxsize)
        self._ahead = 0  # how many at the front went in ahead of their turn

    def _put(self, item: object) -> None:
        if isinstance(item, _First):
            self._queue.insert(self._ahead, item.item)
            self._ahead += 1
        else:
            self._queue.append(item)

    def _get(self) -> object:
        self._ahead = max(0, self._ahead - 1)
        return self._queue.popleft()


class Subscription:
    """One client's view of a hub: a bounded queue of events.

    The connection that has the microphone also gets the reply's audio through this queue, as
    bytes, so that audio and the messages about it reach the client in the order they were
    made. The exception is what is offered with `first`: telling the player to pause or to
    go on. Behind a second of audio that is still queued for a slow connection, a pause
    would come a second late, and where in the audio it arrives does not change what it
    means.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[BaseModel | bytes] = _Lane(maxsize=_MAX_LAG)
        self.lost = False  # the client fell too far behind and missed events

    def offer(self, event: BaseModel | bytes, *, first: bool = False) -> None:
        if self.lost:
            return
        try:
            self.queue.put_nowait(_First(event) if first else event)
        except asyncio.QueueFull:
            self.lost = True
            # What is queued will not be sent: the client is told to reconnect and gets
            # everything from the store. Dropping it here frees the memory at once.
            self.drain()

    def drain(self) -> list[BaseModel]:
        """Take out what is queued. Returns the events among it: audio is of no use later."""
        waiting = []
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not isinstance(item, bytes):
                waiting.append(item)
        return waiting


class SessionHub:
    def __init__(
        self,
        rt: Runtime,
        log: EventLog,
        on_idle: Callable[[SessionHub], None],
        turns: int = 0,
        desk: DeskBus | None = None,
        workspace: str | None = None,
    ) -> None:
        self.rt = rt
        self.log = log
        self.turns = turns  # customer messages so far, for the limit per conversation
        self.session_id = log.session_id
        self.busy = False  # a turn is running
        self.current: str | None = None  # that turn's ID, once its first event has gone out
        self.gone = False  # the session was deleted while this hub was open
        self.redactor = rt.redactor(log)
        self.faults = FaultPlan()
        self.tool_faults = ToolFaults()
        self._on_idle = on_idle
        self._turn_events: list[BaseModel] = []  # the turn in flight, deltas included
        self._subscribers: set[Subscription] = set()
        # Code in this process that follows the session's events as they happen: voice.
        self._listeners: list[Callable[[BaseModel], None]] = []
        self.voice: VoiceSession | None = None  # while this session has the microphone on
        self.speech_down = 0  # simulated failures of the speech service that are still to come
        self.greeted = False  # the voice has said hello in this conversation
        self._task: asyncio.Task[None] | None = None
        self._working = False  # the task of the last turn has not returned yet
        self.desk = desk  # where the open service desks hear of this session's tickets
        self.workspace = workspace  # the workspace this conversation is in, hashed, if any
        self._ticket_ids: set[str] = set()  # the tickets this hub has seen opened

    def _publish(self, event: BaseModel) -> None:
        if event.type == "escalation.created" and self.desk is not None:
            # The ticket is in the database by now. Every open desk of the workspace hears of it.
            self._ticket_ids.add(event.ticket.id)
            self.desk.publish(self.workspace, self.log.world_id, event.ticket.id)
        if event.type in TURN_ENDED and self.desk is not None:
            # And again when a turn is over and stored: the receipt of a new ticket is whole only
            # then, and a desk that has a case open sees what the customer went on to say.
            for ticket_id in sorted(self._ticket_ids):
                self.desk.publish(self.workspace, self.log.world_id, ticket_id)
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
            self.current = None
            self._turn_events = []
        elif event.seq:
            self._turn_events.append(event)
        for subscriber in self._subscribers:
            subscriber.offer(event)
        for listener in list(self._listeners):
            listener(event)

    async def join(self, workspace: str | None) -> None:
        """Put a conversation that is in no workspace yet into the one its browser names.

        It stays in the first one it joined. The service desk of that workspace is where its
        tickets show, and a second browser that learns the conversation's ID does not move it.
        """
        if not workspace or self.workspace is not None:
            return
        async with self.rt.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE sessions SET workspace_id = coalesce(workspace_id, %s) WHERE id = %s "
                "RETURNING workspace_id",
                (workspace, self.session_id),
            )
            row = await cur.fetchone()
        self.workspace = row["workspace_id"] if row else None

    def listen(self, listener: Callable[[BaseModel], None]) -> Callable[[], None]:
        """Call `listener` with every event from now on. Returns the way to stop that."""
        self._listeners.append(listener)

        def stop() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)
            self._on_idle(self)

        return stop

    async def record(self, event: BaseModel, turn_id: str | None) -> BaseModel:
        """Number, store and send an event that no running turn emits: what voice measured.

        The store gets it at once, and it alone. Events of a running turn stay queued for the
        turn's end, where they get their last masking pass, see `EventLog.write_apart`.

        All of it goes through even if whoever asked for it is stopped meanwhile. Voice being
        turned off in that moment is common: a reply ends, the customer hangs up. What voice
        measured is also what it cost, and an event the clients saw has to be in the store.
        """
        return await asyncio.shield(self._record(event, turn_id))

    async def _record(self, event: BaseModel, turn_id: str | None) -> BaseModel:
        # A turn that is ending has numbered its last event and is storing it. This event
        # waits for that one to go out, so that clients get the two in order.
        async with self.log.closing:
            stamped = self.log.stamp_apart(self.redactor.event(event), turn_id)
            for subscriber in self._subscribers:
                subscriber.offer(stamped)
            for listener in list(self._listeners):
                listener(stamped)
        await self.log.write_apart(stamped, self.redactor)
        return stamped

    async def start_early(self, text: str) -> EarlyCall | None:
        """Start the first model call of the spoken turn that `text` would open.

        Voice calls this when the recogniser first thinks the customer may have finished. If
        they have, `start_turn` gets the call and the turn uses it. None when no call can be
        made ahead: a turn is running, so the conversation it would read is not settled, or
        simulated failures are queued, which have to show in the turn they are meant for.
        """
        if self.busy or self._working or self.gone or len(self.faults):
            return None
        state = await self.rt.graph.aget_state({"configurable": {"thread_id": self.session_id}})
        if self.busy or self._working or self.gone:
            return None  # a turn began while the conversation was being read
        prompt = first_prompt(state.values, text, self.log.heard, "voice")
        return EarlyCall(self.rt.llm, prompt, TOOLS, text)

    def let_go(self, early: EarlyCall) -> None:
        """Give up a call of `start_early` that no turn will use. What it cost is stored."""
        early.let_go(lambda finished: self.record(finished, None))

    def start_turn(
        self,
        text: str | None,
        mode: Literal["text", "voice"],
        client_id: str | None = None,
        early: EarlyCall | None = None,
    ) -> bool:
        """Start a turn in the background. `text` None resumes one that stopped early.

        Returns False when a turn is already running. A session handles one turn at a time.
        `early` is the turn's first model call if `start_early` made it ahead.
        """
        if self.busy:
            if early is not None:
                self.let_go(early)
            return False
        self.busy = self._working = True
        if text is not None:
            self.turns += 1
        self._turn_events = []
        self._task = asyncio.create_task(self._run(text, mode, client_id, self._task, early))
        return True

    async def _run(
        self,
        text: str | None,
        mode: Literal["text", "voice"],
        client_id: str | None,
        previous: asyncio.Task[None] | None,
        early: EarlyCall | None = None,
    ) -> None:
        started: str | None = None  # the ID of this turn, once an event of it has gone out
        ended = False  # the event that ends the turn has gone out

        def publish(event: BaseModel) -> None:
            nonlocal started, ended
            started = started or getattr(event, "turn_id", None) or ""
            ended = ended or event.type in TURN_ENDED
            self.current = started or None
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
                early=early,
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
            if early is not None:
                self.let_go(early)  # which does nothing if the turn took it
            if self._task is asyncio.current_task():
                self.busy = self._working = False
                self.current = None
                # Stored events come from Postgres from here on. A save that failed keeps its
                # events in the log, where attach finds them.
                self._turn_events = []
            self._on_idle(self)  # the client may have left while the turn ran

    async def resumable(self) -> bool:
        """True when the last turn stopped early and `turn.retry` would pick it up."""
        return (
            not self.busy and self.log.heard is None and await can_resume(self.rt, self.session_id)
        )

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

    async def stop(self, turn_id: str | None = None) -> None:
        """Stop the running turn and wait until it has been stored and tidied up.

        With `turn_id`, only that turn is stopped. If another one runs by now, it is not the
        caller's to stop: it is left alone, and nothing is waited for.
        """
        task = self._task
        if task is None:
            return
        if self.busy and not task.done():
            if turn_id is not None and self.current != turn_id:
                return
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def interrupt_turn(self, heard: str, turn_id: str | None = None) -> None:
        """Stop the reply the customer spoke over, and keep of it what they heard.

        `heard` is the part of the reply that had been played to them, which may be nothing.
        `turn_id` names the turn the reply belongs to, see `EventLog.keep_heard`.
        If that turn is still running it is cancelled, and this returns once it is stored and
        the session is free. A turn that has started since is not touched. Voice hears of a
        turn's end a moment after the session is free again, and by then a typed message may
        have started the next one: a note about the reply before must not stop it.

        The next turn, spoken or typed, puts `heard` into the conversation in place of the
        full reply, so the model does not assume the customer knows what they were never
        told. The turn that was cut off is not offered for picking up again.
        """
        await self.log.keep_heard(heard, turn_id)
        await self.stop(turn_id)

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
        return not self.busy and not self._working and not self._subscribers and not self._listeners

    def inject(self, request: FaultInject) -> bool:
        """Queue simulated failures for this session's next calls. False when the request is bad.

        Faults live in this session only, and each queue has a cap.
        """
        if request.kind == "speech_down":
            # Voice that is on drops out now. Otherwise the next attempts to turn it on fail.
            times = request.times
            if self.voice is not None:
                self.voice.fail("simulated failure")
                times -= 1
            self.speech_down = min(MAX_PENDING, self.speech_down + times)
        elif request.kind == "tool_failure":
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
        self.desk = DeskBus()
        self._hubs: dict[str, SessionHub] = {}
        self._opening = 0  # hubs being built: they hold a place before they exist

    def __len__(self) -> int:
        return len(self._hubs)

    @property
    def full(self) -> bool:
        return len(self._hubs) + self._opening >= self.max_sessions

    def live(self) -> list[str]:
        """The sessions that are open in this process."""
        return list(self._hubs)

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
            world_id = workspace = None
            if session_id:
                async with self.rt.pool.connection() as conn:
                    cur = await conn.execute(
                        "SELECT world_id, workspace_id FROM sessions WHERE id = %s", (session_id,)
                    )
                    row = await cur.fetchone()
                world_id = row["world_id"] if row else None
                workspace = row["workspace_id"] if row else None
            resumed = world_id is not None
            if not resumed:
                # The ID is the only key to the session, so it has to be unguessable.
                session_id = world_id = uuid.uuid4().hex
                await ensure_world(self.rt, world_id)
            try:
                log = await EventLog.open(self.rt.pool, session_id, world_id)
            except ForeignKeyViolation:
                if not resumed:
                    raise
                # The conversation was purged between the look above and this. The visitor
                # gets a new one, as for any ID the server no longer knows.
                resumed = False
                session_id = world_id = uuid.uuid4().hex
                await ensure_world(self.rt, world_id)
                log = await EventLog.open(self.rt.pool, session_id, world_id)
            turns = await log.turn_count() if resumed else 0
            hub = self._hubs.setdefault(
                session_id,
                SessionHub(self.rt, log, self._release, turns, self.desk, workspace),
            )
            return hub, resumed
        finally:
            self._opening -= 1

    def _release(self, hub: SessionHub) -> None:
        """Forget a hub nobody is using. Its session lives on in Postgres."""
        if hub.idle and self._hubs.get(hub.session_id) is hub:
            del self._hubs[hub.session_id]

    async def tell(self, session_id: str, event: BaseModel) -> bool:
        """Put an event that no turn emits into a conversation, whether or not anyone has it open.

        For what the service desk does to a ticket: the event is numbered and stored with the
        conversation and sent to every connection it has. False when the conversation is gone.
        """
        if not await self.known(session_id):
            return False
        hub, resumed = await self.open(session_id)
        try:
            if resumed:
                await hub.record(event, None)
            return resumed
        finally:
            self._release(hub)

    async def close(self) -> None:
        """Stop every running turn and wait until each has been stored. For shutdown."""
        hubs = list(self._hubs.values())
        await asyncio.gather(
            *(hub.voice.stop("error", "the server is restarting") for hub in hubs if hub.voice),
            return_exceptions=True,
        )
        await asyncio.gather(*(hub.stop() for hub in hubs))
