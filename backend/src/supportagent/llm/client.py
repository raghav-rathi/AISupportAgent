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
    # For a call that asks for it: if the model has sent nothing after this long, the same
    # request goes out a second time, and the answer is whichever of the two starts first.
    # None: one request, however long its first token takes.
    hedge_after_s: float | None = None


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
    requests: int = 1  # 2 when a second request went out beside a slow first one


def _at_work(chunk: AIMessageChunk) -> bool:
    """Whether a chunk shows the model at work: text, a piece of a tool call, or reasoning.

    A chunk can carry a content block with empty text, or a key whose value is None, and say
    nothing. Counting those would stop the first-token clock on a model that is stuck.
    """
    return bool(
        str(chunk.text)
        or any(
            piece.get("name") or piece.get("args") or piece.get("id")
            for piece in chunk.tool_call_chunks
        )
        or any(chunk.additional_kwargs.values())
    )


async def _drop(stream: AsyncIterator[AIMessageChunk], waiting: asyncio.Future | None) -> None:
    """Give up a stream: stop whoever waits on it, then close it."""
    if waiting is not None:
        waiting.cancel()
        with suppress(BaseException):
            await waiting
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        with suppress(Exception):
            await closer()


async def _first_of_two(
    make: Callable[[], AsyncIterator[AIMessageChunk]],
    delay_s: float,
    on_second: Callable[[], None],
) -> AsyncIterator[AIMessageChunk]:
    """One request's stream, or that of a second one sent beside it if the first is slow.

    Most calls get their first token in a quarter of a second. About one in fourteen took
    more than 0.8 s on the hosted model, and now and then one took five. Nothing in the
    request explains it: the same request sent again answers at the usual speed. So after
    `delay_s` without a sign of life the request is sent a second time, and whichever of the
    two shows the model at work first is the stream. The other is closed. A request that
    fails before the second went out fails the call, as it would without this. Once both
    are out, one failing leaves the other.
    """
    streams = [make()]
    kept: list[list[AIMessageChunk]] = [[]]  # what each sent before it showed any sign
    waiting: dict[asyncio.Future, int] = {asyncio.ensure_future(anext(streams[0])): 0}
    timer: asyncio.Future | None = asyncio.ensure_future(asyncio.sleep(delay_s))
    failed: list[BaseException] = []
    winner: int | None = None
    ended = False
    try:
        while winner is None:
            pending = set(waiting) | ({timer} if timer is not None else set())
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if timer in done:
                timer = None
                on_second()
                streams.append(make())
                kept.append([])
                waiting[asyncio.ensure_future(anext(streams[1]))] = 1
            for task in done:
                if task not in waiting:
                    continue
                index = waiting.pop(task)
                try:
                    chunk = task.result()
                except StopAsyncIteration:
                    winner, ended = index, True  # over without a word: the caller's to judge
                    break
                except Exception as exc:
                    failed.append(exc)
                    if timer is not None or len(failed) == len(streams):
                        raise failed[0] from None
                    continue
                kept[index].append(chunk)
                if _at_work(chunk):
                    winner = index
                    break
                waiting[asyncio.ensure_future(anext(streams[index]))] = index
        for task, index in list(waiting.items()):
            if index != winner:
                del waiting[task]
                await _drop(streams[index], task)
        for chunk in kept[winner]:
            yield chunk
        if not ended:
            async for chunk in streams[winner]:
                yield chunk
    finally:
        if timer is not None:
            timer.cancel()
        tasks = {index: task for task, index in waiting.items()}
        for index, stream in enumerate(streams):
            await _drop(stream, tasks.get(index))


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
        hedge: bool = False,
    ) -> LLMResult:
        """Run one model call to completion. Raises LLMUnavailable when every model failed.

        With `stream_text` false the text is not emitted as deltas. The caller holds it, for
        example until a grounding check has passed. `json_schema` asks the model for a reply in
        that shape and only applies to a call without tools. `hedge` sends the request a second
        time if its first token is slow, see `LLMPolicy.hedge_after_s`: for a call that someone
        is waiting for with nothing to look at, which is a spoken turn's first.
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
                    message, ttft_ms, requests = await self._attempt(
                        model, messages, tools, json_schema, call_id, started, deadline, emit,
                        stream_text, fault, self.policy.hedge_after_s if hedge else None,
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
                cost = price.cost(input_tokens, output_tokens) if price else None
                if cost is not None and requests > 1:
                    # The request that lost was read by its provider all the same. What it
                    # wrote before it was closed is not known, and is little.
                    cost += price.cost(input_tokens, 0) or 0.0
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
                    cost_usd=cost,
                    requests=requests,
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
                        requests=result.requests,
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
        hedge_after_s: float | None = None,
    ) -> tuple[AIMessage, float | None, int]:
        """One attempt on one model. Returns the response, the time to its first token, and
        how many requests it took: two when a second went out beside a slow first.

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
        requests = 1
        stream: AsyncIterator[AIMessageChunk] | None = None

        def make() -> AsyncIterator[AIMessageChunk]:
            if json_schema is None:
                return aiter(model.stream(messages, tools))
            return aiter(model.stream(messages, tools, json_schema=json_schema))

        def second() -> None:
            nonlocal requests
            requests = 2

        try:
            limit = min(self.policy.total_timeout_s, max(0.01, deadline - attempt_started))
            async with asyncio.timeout(limit):
                if fault and fault.after_chunks == 0:
                    await asyncio.sleep(fault.delay_s)
                    raise InjectedFault(fault.failure)
                if hedge_after_s is None:
                    stream = make()
                else:
                    stream = _first_of_two(make, hedge_after_s, second)
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
                    # Output means text or a piece of a tool call, see `_at_work`.
                    if _at_work(chunk):
                        alive = True
                        # Reasoning tokens are no visible output yet, and not a first token.
                        said = str(chunk.text) or any(
                            piece.get("name") or piece.get("args") or piece.get("id")
                            for piece in chunk.tool_call_chunks
                        )
                        if said and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - call_started) * 1000
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
        return message, ttft_ms, requests
