"""Runs one turn of one session and reports it as events. Chat, voice and the CLI all call this."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from supportagent import clock
from supportagent.agent.context import ToolFaults, TurnUsage
from supportagent.agent.graph import user_turn
from supportagent.crm import world as worlds
from supportagent.eventlog import EventLog
from supportagent.events import TurnFailed, TurnFinished, TurnInterrupted, TurnStarted
from supportagent.llm.errors import LLMUnavailable
from supportagent.llm.faults import FaultPlan
from supportagent.redact import Redactor
from supportagent.runtime import Runtime

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
) -> bool:
    """Run the agent on one customer message. `text` None resumes a turn that stopped early.

    Every event is masked, numbered, passed to `on_event` as it happens, and stored. The model
    gets the message as typed. A caller that keeps a session open passes its `log` and `redactor`
    so both carry over from turn to turn.

    Exactly one event ends the turn: `turn.finished`, `turn.failed` or `turn.interrupted`. It is
    stored, and the session is free again, before `on_event` sees it, so a client that reacts to
    it at once finds the session ready.

    Returns False when the turn could not finish because every model was down. The session keeps
    its state, and calling again with `text` None picks the turn up where it stopped: same turn
    ID, same mode, usage added to what the first attempt used. A bug in our own code is reported
    as a failed turn and then raised. Raises TurnInProgress when the session is running a turn,
    and NothingToResume when `text` is None and no turn is unfinished.
    """
    started = time.perf_counter()
    log = log or await EventLog.open(rt.pool, session_id, world_id)
    redactor = redactor or Redactor()
    config = {"configurable": {"thread_id": session_id}}
    before = await rt.graph.aget_state(config)
    if text is None:
        past = await log.last_turn()
        if past is None or not before.next:
            raise NothingToResume(session_id)
        turn_id, mode, usage, earlier_ms = past.turn_id, past.mode, past.usage, past.elapsed_ms
    else:
        text = text.replace("\x00", "")
        turn_id, usage, earlier_ms = uuid.uuid4().hex[:12], TurnUsage(), 0.0
    await log.claim(turn_id, mode)
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

    def publish(event: BaseModel) -> None:
        on_event(log.stamp(redactor.event(event), turn_id))

    async def end(event: BaseModel) -> None:
        stamped = log.stamp(redactor.event(event), turn_id)
        await log.save(redactor, release=turn_id)
        on_event(stamped)

    try:
        if text is not None:
            publish(TurnStarted(mode=mode, text=text))
        try:
            async for part in rt.graph.astream(
                user_turn(text) if text is not None else None,
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
            await end(TurnInterrupted(total_ms=elapsed_ms()))
            raise
        except Exception:
            await end(
                TurnFailed(
                    code="internal",
                    message="internal error",
                    retryable=False,
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
            )
        )
        return True
    finally:
        # Covers the paths on which `end` did not run or did not get through. The claim goes
        # either way: a session must not stay taken because its turn broke.
        await log.save(redactor, release=turn_id)
        # Older saved states of the conversation still hold what this turn masked or removed.
        await rt.prune(session_id)
