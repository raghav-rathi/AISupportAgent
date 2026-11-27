"""The first model call of a turn, started before the turn itself.

Voice knows what the customer has said a little before it knows that they have finished. The
recogniser sends an early signal when a turn may be over and the final one some hundreds of
milliseconds later: on 34 test clips the gap was 0.24 s at the median and 0.6 s or more for the
turns that took longest to end. The agent's first model call fits into that gap. It has no side
effect: it reads the conversation and proposes an answer or a tool call, and nothing of it is
shown, spoken, stored or run before the turn exists.

So the call is started on the early signal, with exactly the prompt the turn would send, and
kept aside. If the turn then starts with that prompt, the agent takes the call in place of making
it: what the call has emitted so far is replayed into the turn, and the rest arrives as it comes.
If the customer went on speaking, or anything else changed the prompt, the call is let go.

A call that was let go has still been paid for. It is left to finish, so that its usage is known,
and that usage is stored as an event without a turn, where the daily budget counts it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from supportagent.events import LlmFinished, LlmStarted
from supportagent.llm.client import Emit, LLMResult, ResilientLLM

logger = logging.getLogger(__name__)

# A call nobody will use is given this long to finish and report what it used.
_LET_GO_AFTER_S = 6.0
_letting_go: set[asyncio.Task[None]] = set()


def prompt_key(prompt: Sequence[BaseMessage]) -> str:
    """A fingerprint of everything in a prompt that the model reads.

    Two prompts with one key get one answer's worth of work. Message IDs are left out: they
    differ between the conversation as it is stored and as a turn rebuilds it, and the model
    never sees them.
    """
    seen = [
        (
            message.type,
            message.content,
            [
                (call["name"], call["args"], call["id"])
                for call in getattr(message, "tool_calls", [])
            ],
            getattr(message, "tool_call_id", None),
            getattr(message, "name", None),
            getattr(message, "status", None),
        )
        for message in prompt
    ]
    return hashlib.sha256(json.dumps(seen, sort_keys=True, default=str).encode()).hexdigest()


class EarlyCall:
    """One model call that runs ahead of the turn it is meant for."""

    def __init__(
        self,
        llm: ResilientLLM,
        prompt: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
        text: str,
    ) -> None:
        self.text = text  # the customer's message it was made for
        self.key = prompt_key(prompt)
        self.taken = False  # a turn has it, or it has been let go
        self._began = time.perf_counter()
        self._events: list[BaseModel] = []  # what the call has emitted and nobody has taken yet
        self._news = asyncio.Event()  # something was emitted, or the call is over
        self._task = asyncio.create_task(
            llm.complete(prompt, tools=tools, emit=self._emit, hedge=True)
        )
        self._task.add_done_callback(lambda _: self._news.set())

    def _emit(self, event: BaseModel) -> None:
        self._events.append(event)
        self._news.set()

    async def result(self, emit: Emit) -> LLMResult:
        """Give the call to the turn: what it has emitted so far, then the rest as it comes.

        The events are passed on from here, by the turn that takes them, and not by the call as
        it emits them: a turn's events can only be written from inside the turn. Raises what the
        call raises. A turn that is stopped while it waits here stops the call.
        """
        self.taken = True
        head_start = round((time.perf_counter() - self._began) * 1000, 1)
        try:
            while True:
                self._news.clear()
                waiting, self._events = self._events, []
                for event in waiting:
                    if isinstance(event, LlmFinished):
                        event = event.model_copy(update={"early_ms": head_start})
                    emit(event)
                if self._task.done() and not self._events:
                    return self._task.result()
                await self._news.wait()
        except asyncio.CancelledError:
            self._task.cancel()
            raise

    def let_go(self, record: Callable[[LlmFinished], Awaitable[object]]) -> None:
        """Nobody will use this call. `record` stores what it used, once that is known."""
        if self.taken:
            return
        self.taken = True
        task = asyncio.create_task(self._settle(record))
        _letting_go.add(task)
        task.add_done_callback(_letting_go.discard)

    async def _settle(self, record: Callable[[LlmFinished], Awaitable[object]]) -> None:
        finished: LlmFinished | None = None
        try:
            await asyncio.wait_for(self._task, _LET_GO_AFTER_S)
        except Exception:
            pass  # it failed, which costs nothing, or it ran out of time and was stopped
        started = None
        for event in self._events:
            if isinstance(event, LlmStarted):
                started = event
            elif isinstance(event, LlmFinished):
                finished = event
        if finished is None and started is not None and self._task.cancelled():
            # Stopped in mid-stream. The provider charged for the prompt and reports nothing.
            finished = LlmFinished(
                call_id=started.call_id,
                purpose=started.purpose,
                model=started.model,
                attempts=started.attempt,
                used_fallback=False,
                ttft_ms=None,
                total_ms=(time.perf_counter() - self._began) * 1000,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                cancelled=True,
            )
        if finished is None:
            return
        try:
            await record(finished.model_copy(update={"unused": True}))
        except Exception:
            logger.warning("could not store the usage of an early model call", exc_info=True)
