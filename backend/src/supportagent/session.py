"""Runs one turn of one session and reports it as events. Chat, voice and the CLI all call this."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Literal

from pydantic import BaseModel

from supportagent import clock
from supportagent.agent.context import ToolFaults, TurnUsage
from supportagent.agent.graph import user_turn
from supportagent.crm import world as worlds
from supportagent.eventlog import HEARTBEAT_S, EventLog
from supportagent.events import (
    LlmReset,
    TurnFailed,
    TurnFinished,
    TurnInterrupted,
    TurnStarted,
)
from supportagent.llm.errors import LLMUnavailable
from supportagent.llm.faults import FaultPlan
from supportagent.redact import Redactor
from supportagent.runtime import Runtime

logger = logging.getLogger(__name__)

OnEvent = Callable[[BaseModel], None]


class NothingToResume(Exception):
    """A resume was asked for and the session has no unfinished turn."""


async def ensure_world(rt: Runtime, world_id: str) -> None:
    """Give a new session its private copy of the mock CRM."""
    async with rt.pool.connection() as conn:
        await worlds.create_world(conn, world_id, clock.today())


async def can_resume(rt: Runtime, session_id: str) -> bool:
    """True when the session's last turn stopped before its end and can be picked up."""
    state = await rt.graph.aget_state({"configurable": {"thread_id": session_id}})
    return bool(state.next)


async def _keep_claim(log: EventLog, turn_id: str, turn: asyncio.Task) -> None:
    """Renew the claim on the session while the turn runs. Stop the turn if the claim is lost.

    A claim is lost only when this process stalled for longer than the lease and another one took
    the session over. Two writers on one conversation would overwrite each other, so this one
    gives up.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        try:
            ours = await log.renew(turn_id)
        except Exception:
            logger.warning("could not renew the claim on session %s", log.session_id, exc_info=True)
            continue
        if not ours:
            logger.error("lost the claim on session %s, stopping its turn", log.session_id)
            turn.cancel()
            return


class _Waits:
    """What the customer waited for in one turn, read off the events as they go out."""

    def __init__(self, clock_ms: Callable[[], float]) -> None:
        self._now = clock_ms
        self.first_step_ms: float | None = None
        self.first_text_ms: float | None = None
        self.answer_ms: float | None = None
        # The call whose streamed text is on the customer's screen and not yet confirmed as the
        # answer. If the turn stops here, that text has to be taken back.
        self.open_call: str | None = None

    def note(self, event: BaseModel) -> None:
        kind = event.type
        if kind in ("tool.started", "llm.delta") and self.first_step_ms is None:
            self.first_step_ms = self._now()
        if kind == "llm.delta":
            if self.open_call != event.call_id:
                self.open_call, self.first_text_ms = event.call_id, self._now()
        elif kind == "llm.reset" and event.call_id == self.open_call:
            self.open_call = self.first_text_ms = None
        elif kind == "answer.retracted":
            self.open_call = self.first_text_ms = None
        elif kind == "assistant.message":
            self.open_call = None
            self.answer_ms = self._now()
            if self.first_text_ms is None:  # a held answer appears all at once
                self.first_text_ms = self.answer_ms


async def run_turn(
    rt: Runtime,
    *,
    session_id: str,
    world_id: str,
    text: str | None,
    on_event: OnEvent,
    mode: Literal["text", "voice"] = "text",
    faults: FaultPlan | None = None,
    tool_faults: ToolFaults | None = None,
    log: EventLog | None = None,
    redactor: Redactor | None = None,
    client_id: str | None = None,
) -> bool:
    """Run the agent on one customer message. `text` None resumes a turn that stopped early.

    Every event is masked, numbered, passed to `on_event` as it happens, and stored. The model
    gets the message as typed. A caller that keeps a session open passes its `log` and `redactor`
    so both carry over from turn to turn. `client_id` is echoed in `turn.started`.

    If the customer spoke over the last reply, the session holds what they had heard of it by
    then. A new turn takes that and keeps that much of the reply in the conversation, and no
    more. A turn that was cut off this way is not picked up again.

    Exactly one event ends the turn: `turn.finished`, `turn.failed` or `turn.interrupted`. It is
    stored, and the session is free again, before `on_event` sees it, so a client that reacts to
    it at once finds the session ready. A turn that stops early takes back any text it had
    streamed and not confirmed.

    Returns False when the turn could not finish because every model was down. The session keeps
    its state, and calling again with `text` None picks the turn up where it stopped: same turn
    ID, same mode, usage added to what the first attempt used. A bug in our own code is reported
    as a failed turn and then raised. Raises TurnInProgress when the session is running a turn,
    SessionGone when it has been deleted, and NothingToResume when `text` is None and no turn is
    unfinished. Until one of those or the first event, nothing of the turn exists.
    """
    started = time.perf_counter()
    log = log or await EventLog.open(rt.pool, session_id, world_id)
    redactor = redactor or rt.redactor(log)
    config = {"configurable": {"thread_id": session_id}}
    before = await rt.graph.aget_state(config)
    if text is None:
        past = await log.last_turn()
        if past is None or not before.next or log.heard is not None:
            raise NothingToResume(session_id)
        turn_id, mode, usage, earlier_ms = past.turn_id, past.mode, past.usage, past.elapsed_ms
    else:
        text = text.replace("\x00", "")
        turn_id, usage, earlier_ms = uuid.uuid4().hex[:12], TurnUsage(), 0.0
    heard = await log.claim(turn_id, mode, resume=text is None)
    lease = asyncio.create_task(_keep_claim(log, turn_id, asyncio.current_task()))
    # Until the customer is verified, any number they type may be the PIN the agent asked for.
    redactor.strict = before.values.get("customer") is None
    ctx = rt.context(
        world_id=log.world_id,
        mode=mode,
        faults=faults,
        tool_faults=tool_faults,
        redactor=redactor,
        usage=usage,
    )

    def elapsed_ms() -> float:
        return earlier_ms + (time.perf_counter() - started) * 1000

    waits = _Waits(elapsed_ms)

    def publish(event: BaseModel) -> None:
        for out in redactor.outgoing(event):
            waits.note(out)
            on_event(log.stamp(out, turn_id))

    async def end(event: BaseModel) -> None:
        if waits.open_call is not None:
            publish(LlmReset(call_id=waits.open_call, reason="cancelled"))
        stamped = log.stamp(redactor.event(event), turn_id)
        await log.save(redactor, release=turn_id)
        on_event(stamped)

    async def resumable() -> bool:
        """Whether the graph saved enough of this turn to pick it up again."""
        try:
            return bool((await rt.graph.aget_state(config)).next)
        except Exception:
            return False

    try:
        if text is not None:
            publish(TurnStarted(mode=mode, text=text, client_id=client_id))
        try:
            async for part in rt.graph.astream(
                user_turn(text, heard) if text is not None else None,
                config,
                context=ctx,
                stream_mode="custom",
                version="v2",
            ):
                publish(part["data"])
            after = await rt.graph.aget_state(config)
        except LLMUnavailable as exc:
            await end(
                TurnFailed(
                    code="llm_unavailable", message=str(exc), retryable=True, total_ms=elapsed_ms()
                )
            )
            return False
        except asyncio.CancelledError:
            # A turn the customer spoke over is finished with: they have moved on.
            can_resume = log.heard is None and await resumable()
            await end(TurnInterrupted(total_ms=elapsed_ms(), resumable=can_resume))
            raise
        except Exception:
            await end(
                TurnFailed(
                    code="internal",
                    message="internal error",
                    retryable=await resumable(),
                    total_ms=elapsed_ms(),
                )
            )
            raise
        await end(
            TurnFinished(
                outcome=after.values.get("outcome", "answered"),
                total_ms=elapsed_ms(),
                llm_calls=usage.llm_calls,
                tool_calls=usage.tool_calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                first_step_ms=waits.first_step_ms,
                first_text_ms=waits.first_text_ms,
                answer_ms=waits.answer_ms,
            )
        )
        return True
    finally:
        lease.cancel()
        with suppress(asyncio.CancelledError):
            await lease
        # Covers the paths on which `end` did not run or did not get through. The claim goes
        # either way: a session must not stay taken because its turn broke.
        await log.save(redactor, release=turn_id)
        # Older saved states of the conversation still hold what this turn masked or removed.
        await rt.prune(session_id)
