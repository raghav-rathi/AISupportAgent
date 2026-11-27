"""One entry point for every model call.

It adds what a bare SDK call lacks: a first-token timeout, a stall timeout, a timeout per attempt,
a deadline for the whole call, backoff that honors Retry-After, a retry budget per model, failover
to a second model, and fault injection. Every step is reported as an event, so the UI can show
"retrying 1 of 2" or "switched to backup model" from what the client did.

SDK retries must be off on the models passed in. Two layers of retries multiply load on an upstream
that is already struggling.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, message_chunk_to_message
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from supportagent.events import (
    LlmDelta,
    LlmFallback,
    LlmFinished,
    LlmReset,
    LlmRetry,
    LlmStarted,
    LlmTrace,
    Purpose,
)
from supportagent.llm.errors import (
    BadCompletion,
    Failure,
    InjectedFault,
    LLMUnavailable,
    classify,
)
from supportagent.llm.faults import Fault, FaultPlan
from supportagent.llm.pricing import Price
from supportagent.llm.trace import trace_message, trace_prompt

Emit = Callable[[BaseModel], None]


class ChatModel(Protocol):
    """A streaming chat model. `name` identifies it in events. `price` may be None.

    `stream` also accepts `json_schema=` when the caller wants the reply held to a schema.
    """

    name: str
    price: Price | None

    def stream(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool] | None
    ) -> AsyncIterator[AIMessageChunk]: ...


@dataclass(frozen=True)
class LLMPolicy:
    first_token_timeout_s: float = 8.0
    stall_timeout_s: float = 10.0  # longest silence allowed between two chunks
    total_timeout_s: float = 45.0  # one attempt
    deadline_s: float = 90.0  # the whole call: every attempt on every model, waits included
    max_attempts: int = 3  # per model
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 4.0
    max_wait_s: float = 5.0  # a Retry-After longer than this goes to the fallback instead


@dataclass(frozen=True)
class LLMResult:
    message: AIMessage
    call_id: str
    model: str
    attempts: int
    used_fallback: bool
    ttft_ms: float | None
    total_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


class RetryBudget:
    """Caps the extra load retries may add to one model.

    Each first attempt earns `ratio` of a token, each retry spends one, and the bucket holds at
    most `burst`. With ratio 0.1, retries stay near 10% of requests however bad the upstream gets.
    A slow refill of `per_second` gives a quiet service its retries back after an outage.
    """

    def __init__(
        self,
        ratio: float = 0.1,
        burst: float = 5.0,
        per_second: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ratio, self._burst, self._per_second, self._clock = ratio, burst, per_second, clock
        self._tokens = burst
        self._refilled = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._refilled) * self._per_second)
        self._refilled = now

    def record_request(self) -> None:
        self._refill()
        self._tokens = min(self._burst, self._tokens + self._ratio)

    def try_spend(self) -> bool:
        self._refill()
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True


class _AttemptFailed(Exception):
    def __init__(self, failure: Failure, text_went_out: bool, injected: bool) -> None:
        super().__init__(failure.reason)
        self.failure = failure
        self.text_went_out = text_went_out
        self.injected = injected


def _ignore(_: BaseModel) -> None:
    return None


class ResilientLLM:
    def __init__(
        self,
        models: Sequence[ChatModel],
        *,
        policy: LLMPolicy | None = None,
        budget_factory: Callable[[], RetryBudget] = RetryBudget,
        capture_prompts: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """`models[0]` is the primary. The rest are fallbacks, tried in order.

        Each model gets its own retry budget, so a failing primary cannot use up the retries of
        its fallback. With `capture_prompts` every finished call also emits an `llm.trace` event
        that holds the full prompt and response. The session masks PINs in it.
        """
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self.policy = policy or LLMPolicy()
        self.capture_prompts = capture_prompts
        self._budget_factory = budget_factory
        self._budgets: dict[tuple[int, str], RetryBudget] = {}
        self._sleep = sleep
        self._jitter = jitter

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[BaseTool] | None = None,
        purpose: Purpose = "agent",
        emit: Emit = _ignore,
        stream_text: bool = True,
        faults: FaultPlan | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Run one model call to completion. Raises LLMUnavailable when every model failed.

        With `stream_text` false the text is not emitted as deltas. The caller holds it, for
        example until a grounding check has passed. `json_schema` asks the model for a reply in
        that shape and only applies to a call without tools.
        """
        call_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        deadline = started + self.policy.deadline_s
        failures: list[Failure] = []
        attempts = 0

        for index, model in enumerate(self.models):
            if failures and time.perf_counter() >= deadline:
                break
            # One budget per place in the list. Two entries may be the same model on two
            # providers, and the first must not use up the retries of the second.
            budget = self._budgets.setdefault((index, model.name), self._budget_factory())
            credited = False  # this call has earned the budget its share
            last_model = index == len(self.models) - 1
            if index > 0:
                emit(
                    LlmFallback(
                        call_id=call_id,
                        from_model=self.models[index - 1].name,
                        to_model=model.name,
                        reason=failures[-1].reason,
                    )
                )
            for attempt in range(1, self.policy.max_attempts + 1):
                attempts += 1
                emit(
                    LlmStarted(call_id=call_id, purpose=purpose, model=model.name, attempt=attempt)
                )
                role = "primary" if index == 0 else "fallback"
                fault = faults.take(role, purpose) if faults else None
                try:
                    message, ttft_ms = await self._attempt(
                        model, messages, tools, json_schema, call_id, started, deadline, emit,
                        stream_text, fault,
                    )  # fmt: skip
                except _AttemptFailed as failed:
                    failure = failed.failure
                    failures.append(failure)
                    if not credited and not failed.injected:
                        # A simulated failure is no request to the upstream, so it earns nothing.
                        budget.record_request()
                        credited = True
                    if failed.text_went_out:
                        emit(LlmReset(call_id=call_id, reason=failure.reason))
                    wait = self._wait(attempt, failure)
                    if (
                        not failure.retryable
                        or attempt == self.policy.max_attempts
                        # A long wait is a reason to try the next model. With no next model,
                        # waiting is all there is, as long as the deadline allows it.
                        or (
                            (failure.retry_after_s or 0.0) > self.policy.max_wait_s
                            and not last_model
                        )
                        or time.perf_counter() + wait >= deadline
                        or not (failed.injected or budget.try_spend())
                    ):
                        break  # this model is done; try the next one
                    emit(
                        LlmRetry(
                            call_id=call_id,
                            model=model.name,
                            attempt=attempt + 1,
                            max_attempts=self.policy.max_attempts,
                            reason=failure.reason,
                            wait_ms=round(wait * 1000),
                        )
                    )
                    await self._sleep(wait)
                    continue

                if not credited:
                    budget.record_request()
                usage = message.usage_metadata or {}
                input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
                price = getattr(model, "price", None)
                result = LLMResult(
                    message=message,
                    call_id=call_id,
                    model=model.name,
                    attempts=attempts,
                    used_fallback=index > 0,
                    ttft_ms=ttft_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=price.cost(input_tokens, output_tokens) if price else None,
                )
                emit(
                    LlmFinished(
                        call_id=call_id,
                        purpose=purpose,
                        model=result.model,
                        attempts=result.attempts,
                        used_fallback=result.used_fallback,
                        ttft_ms=result.ttft_ms,
                        total_ms=result.total_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=result.cost_usd,
                    )
                )
                if self.capture_prompts:
                    prompt, tool_names = trace_prompt(messages, tools)
                    emit(
                        LlmTrace(
                            call_id=call_id,
                            purpose=purpose,
                            model=result.model,
                            prompt=prompt,
                            tools=tool_names,
                            response=trace_message(result.message),
                        )
                    )
                return result
        raise LLMUnavailable(failures)

    def _wait(self, attempt: int, failure: Failure) -> float:
        """Seconds to wait before the next attempt: Retry-After when given, else capped backoff."""
        if failure.retry_after_s is not None:
            return failure.retry_after_s + 0.25 * self._jitter()
        backoff = min(self.policy.backoff_cap_s, self.policy.backoff_base_s * 2 ** (attempt - 1))
        return backoff * (0.5 + 0.5 * self._jitter())

    async def _attempt(
        self,
        model: ChatModel,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None,
        json_schema: dict[str, Any] | None,
        call_id: str,
        call_started: float,
        deadline: float,
        emit: Emit,
        stream_text: bool,
        fault: Fault | None,
    ) -> tuple[AIMessage, float | None]:
        """One attempt on one model. Returns the response and the time to its first token.

        The time is counted from the start of the call, so it includes failed attempts and waits:
        that is how long the customer looked at an empty reply.
        """
        attempt_started = time.perf_counter()
        first_token_by = attempt_started + self.policy.first_token_timeout_s
        merged: AIMessageChunk | None = None
        ttft_ms: float | None = None
        alive = False  # the model has sent something other than an empty opening chunk
        text_went_out = False
        chunks = 0
        stream: AsyncIterator[AIMessageChunk] | None = None
        try:
            limit = min(self.policy.total_timeout_s, max(0.01, deadline - attempt_started))
            async with asyncio.timeout(limit):
                if fault and fault.after_chunks == 0:
                    await asyncio.sleep(fault.delay_s)
                    raise InjectedFault(fault.failure)
                stream = aiter(
                    model.stream(messages, tools)
                    if json_schema is None
                    else model.stream(messages, tools, json_schema=json_schema)
                )
                while True:
                    patience = (
                        self.policy.stall_timeout_s
                        if alive
                        else max(0.0, first_token_by - time.perf_counter())
                    )
                    try:
                        chunk = await asyncio.wait_for(anext(stream), patience)
                    except StopAsyncIteration:
                        break
                    # Output means text or a piece of a tool call. A chunk can carry a content
                    # block with empty text, or a key whose value is None, and say nothing:
                    # counting those would stop the first-token clock on a model that is stuck.
                    if str(chunk.text) or any(
                        piece.get("name") or piece.get("args") or piece.get("id")
                        for piece in chunk.tool_call_chunks
                    ):
                        alive = True
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - call_started) * 1000
                    elif any(chunk.additional_kwargs.values()):
                        alive = True  # reasoning tokens: no visible output yet, but not stuck
                    merged = chunk if merged is None else merged + chunk
                    text = str(chunk.text)
                    if text and stream_text:
                        emit(LlmDelta(call_id=call_id, text=text))
                        text_went_out = True
                    chunks += 1
                    if fault and chunks == fault.after_chunks:
                        await asyncio.sleep(fault.delay_s)
                        raise InjectedFault(fault.failure)
                message = None if merged is None else message_chunk_to_message(merged)
                if message is None or not (str(message.text).strip() or message.tool_calls):
                    malformed = message is not None and bool(message.invalid_tool_calls)
                    detail = "malformed tool call" if malformed else "empty completion"
                    raise BadCompletion(Failure("server_error", retryable=True, detail=detail))
        except Exception as exc:
            try:
                failure = classify(exc)
            except Exception:
                # A bug of ours, not an outage. The half answer on screen still has to go.
                if text_went_out:
                    emit(LlmReset(call_id=call_id, reason="server_error"))
                raise
            raise _AttemptFailed(failure, text_went_out, isinstance(exc, InjectedFault)) from exc
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                with suppress(Exception):
                    await closer()
        return message, ttft_ms
