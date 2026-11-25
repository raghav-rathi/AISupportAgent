from __future__ import annotations

import email.utils
import time

import httpx
import pytest
from fakes import ScriptedModel, connection_error, http_error, stream_error, tool_call, usage
from langchain_core.messages import AIMessageChunk

from supportagent.llm import faults
from supportagent.llm.client import LLMPolicy, ResilientLLM, RetryBudget
from supportagent.llm.errors import LLMUnavailable
from supportagent.llm.pricing import Price

FAST = LLMPolicy(first_token_timeout_s=0.05, stall_timeout_s=0.05, total_timeout_s=1.0)
OK = ["Hel", "lo", usage(100, 20)]


class Harness:
    """A client with recorded events and sleeps, and no real waiting."""

    def __init__(self, *models, policy=FAST, budget_factory=RetryBudget, jitter=0.0):
        self.events, self.sleeps = [], []
        self.llm = ResilientLLM(
            models,
            policy=policy,
            budget_factory=budget_factory,
            sleep=self._sleep,
            jitter=lambda: jitter,
        )

    async def _sleep(self, seconds):
        self.sleeps.append(seconds)

    async def complete(self, **kwargs):
        return await self.llm.complete([], emit=self.events.append, **kwargs)

    def types(self):
        return [e.type for e in self.events]

    def of(self, kind):
        return [e for e in self.events if e.type == kind]


async def test_text_streams_as_deltas_and_usage_becomes_cost():
    h = Harness(ScriptedModel("primary", OK, price=Price(input=1.0, output=2.0)))
    result = await h.complete()
    assert result.message.content == "Hello"
    assert [e.text for e in h.of("llm.delta")] == ["Hel", "lo"]
    assert h.types() == ["llm.started", "llm.delta", "llm.delta", "llm.finished"]
    finished = h.of("llm.finished")[0]
    assert (finished.input_tokens, finished.output_tokens) == (100, 20)
    assert finished.cost_usd == pytest.approx(100 * 1.0 / 1e6 + 20 * 2.0 / 1e6)
    assert finished.ttft_ms is not None


async def test_a_model_without_a_price_reports_no_cost_rather_than_zero():
    h = Harness(ScriptedModel("primary", OK))
    await h.complete()
    assert h.of("llm.finished")[0].cost_usd is None


async def test_a_tool_call_arrives_whole_and_sends_no_text():
    chunks = [
        tool_call("issue_refund", '{"invoice_id": "INV-', "c9"),
        tool_call(None, '30417"}', None),
        usage(50, 12),
    ]
    h = Harness(ScriptedModel("primary", chunks))
    result = await h.complete()
    assert result.message.tool_calls == [
        {
            "name": "issue_refund",
            "args": {"invoice_id": "INV-30417"},
            "id": "c9",
            "type": "tool_call",
        }
    ]
    assert h.of("llm.delta") == []


async def test_held_text_is_returned_but_not_streamed():
    h = Harness(ScriptedModel("primary", OK))
    result = await h.complete(stream_text=False)
    assert result.message.content == "Hello"
    assert h.of("llm.delta") == []


async def test_a_schema_reaches_the_model_only_when_the_caller_gives_one():
    model = ScriptedModel("primary", OK)
    h = Harness(model)
    await h.complete()
    await h.complete(json_schema={"name": "verdict", "schema": {"type": "object"}})
    assert model.schemas == [None, {"name": "verdict", "schema": {"type": "object"}}]


async def test_rate_limit_waits_as_long_as_the_server_asked():
    model = ScriptedModel("primary", [http_error(429, retry_after="2")], OK)
    h = Harness(model)
    result = await h.complete()
    assert result.attempts == 2
    assert h.sleeps == [2.0]
    retry = h.of("llm.retry")[0]
    assert (retry.reason, retry.attempt, retry.max_attempts, retry.wait_ms) == (
        "rate_limited",
        2,
        3,
        2000,
    )


async def test_retry_after_is_read_in_milliseconds_and_as_a_date():
    in_two_seconds = email.utils.formatdate(time.time() + 2.0, usegmt=True)
    model = ScriptedModel(
        "primary",
        [http_error(429, headers={"retry-after-ms": "1500"})],
        [http_error(503, retry_after=in_two_seconds)],
        OK,
    )
    h = Harness(model)
    await h.complete()
    assert h.sleeps[0] == 1.5
    assert 0.5 < h.sleeps[1] <= 2.0


async def test_jitter_does_not_turn_an_acceptable_wait_into_a_skipped_retry():
    # The server asks for exactly the longest wait we accept. Jitter comes on top of it.
    model = ScriptedModel("primary", [http_error(429, retry_after="5")], OK)
    h = Harness(model, ScriptedModel("fallback", OK), jitter=1.0)
    result = await h.complete()
    assert result.used_fallback is False
    assert h.sleeps == [5.25]


async def test_a_slow_first_token_times_out_and_is_retried():
    model = ScriptedModel("primary", [0.5, "late"], OK)
    h = Harness(model)
    result = await h.complete()
    assert result.message.content == "Hello"
    assert h.of("llm.retry")[0].reason == "timeout"


async def test_an_empty_opening_chunk_does_not_stop_the_first_token_clock():
    # Providers open a stream with a chunk that carries the role and no text.
    opening = AIMessageChunk(content="")
    model = ScriptedModel("primary", [opening, 0.5, "late"], OK)
    h = Harness(model)
    result = await h.complete()
    assert result.message.content == "Hello"
    assert h.of("llm.retry")[0].reason == "timeout"


async def test_a_stall_in_the_middle_of_the_answer_is_a_timeout_too():
    model = ScriptedModel("primary", ["Hel", 0.5, "lo"], OK)
    h = Harness(model)
    await h.complete()
    assert [e.reason for e in h.of("llm.reset")] == ["timeout"]


async def test_text_from_a_failed_attempt_is_reset_before_the_retry():
    model = ScriptedModel("primary", ["Hel", connection_error()], OK)
    h = Harness(model)
    result = await h.complete()
    assert result.message.content == "Hello"
    assert h.types() == [
        "llm.started",
        "llm.delta",
        "llm.reset",
        "llm.retry",
        "llm.started",
        "llm.delta",
        "llm.delta",
        "llm.finished",
    ]


async def test_an_error_inside_the_stream_is_reset_retried_and_failed_over():
    # HTTP 200, some text, then an error event. The SDK raises it without a status code.
    primary = ScriptedModel("primary", ["Hel", stream_error()])
    fallback = ScriptedModel("fallback", OK)
    h = Harness(primary, fallback)
    result = await h.complete()
    assert result.used_fallback is True
    assert len(primary.calls) == 3
    assert [e.reason for e in h.of("llm.reset")] == ["server_error"] * 3
    assert result.message.content == "Hello"


async def test_a_dropped_socket_below_the_sdk_is_a_connection_failure():
    model = ScriptedModel("primary", [httpx.ReadError("reset by peer")], OK)
    h = Harness(model)
    await h.complete()
    assert h.of("llm.retry")[0].reason == "connection"


async def test_a_completion_with_nothing_in_it_is_retried_then_failed_over():
    primary = ScriptedModel("primary", [AIMessageChunk(content=""), usage(40, 0)])
    h = Harness(primary, ScriptedModel("fallback", OK))
    result = await h.complete()
    assert result.used_fallback is True
    assert len(primary.calls) == 3
    assert h.of("llm.fallback")[0].reason == "server_error"


async def test_the_fallback_takes_over_when_the_primary_keeps_failing():
    primary = ScriptedModel("primary", [http_error(500)])
    fallback = ScriptedModel("fallback", OK)
    h = Harness(primary, fallback)
    result = await h.complete()
    assert (result.model, result.used_fallback, result.attempts) == ("fallback", True, 4)
    assert len(primary.calls) == 3
    switch = h.of("llm.fallback")[0]
    assert (switch.from_model, switch.to_model, switch.reason) == (
        "primary",
        "fallback",
        "server_error",
    )
    assert h.of("llm.finished")[0].used_fallback is True


OUT_OF_CREDIT = "You exceeded your current quota, please check your plan and billing details."
PER_MINUTE = (
    "You exceeded your current quota, please check your plan and billing details. Quota exceeded "
    "for metric: generate_content_requests, limit: 15 requests per minute. Please retry in 2s."
)


@pytest.mark.parametrize(
    "error",
    [
        http_error(429, message=OUT_OF_CREDIT, code="insufficient_quota"),
        http_error(429, message="Your credit balance is too low."),
        http_error(402, message="Spend limit reached"),
        http_error(401, message="bad key"),
        http_error(501, message="not implemented"),
        http_error(429, retry_after="60"),
    ],
    ids=["quota-code", "quota-words", "payment-required", "rejected", "501", "long-retry-after"],
)
async def test_errors_that_waiting_cannot_fix_go_straight_to_the_fallback(error):
    primary = ScriptedModel("primary", [error])
    h = Harness(primary, ScriptedModel("fallback", OK))
    result = await h.complete()
    assert result.used_fallback is True
    assert len(primary.calls) == 1
    assert h.sleeps == []
    assert h.of("llm.retry") == []


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (http_error(429, message=PER_MINUTE), "rate_limited"),
        (http_error(429, message="Rate limit reached on tokens per min (TPM)"), "rate_limited"),
        (http_error(408, message="request timeout"), "timeout"),
        (http_error(529, message="overloaded"), "server_error"),
    ],
    ids=["gemini-per-minute", "openai-tpm", "408", "529"],
)
async def test_errors_that_pass_are_retried_on_the_same_model(error, reason):
    primary = ScriptedModel("primary", [error], OK)
    h = Harness(primary, ScriptedModel("fallback", OK))
    result = await h.complete()
    assert result.used_fallback is False
    assert h.of("llm.retry")[0].reason == reason


async def test_every_model_down_raises_with_the_failures_in_order():
    h = Harness(
        ScriptedModel("primary", [http_error(401)]), ScriptedModel("fallback", [connection_error()])
    )
    with pytest.raises(LLMUnavailable) as raised:
        await h.complete()
    assert [f.reason for f in raised.value.failures] == ["rejected"] + ["connection"] * 3
    assert h.of("llm.finished") == []


async def test_the_deadline_ends_a_call_that_keeps_failing():
    # Each attempt waits 0.05 s for a first token. The call as a whole may take 0.12 s.
    slow = ScriptedModel("primary", [1.0, "late"])
    policy = LLMPolicy(first_token_timeout_s=0.05, deadline_s=0.12, backoff_base_s=0.0)
    h = Harness(slow, ScriptedModel("fallback", [1.0, "late"]), policy=policy)
    started = time.perf_counter()
    with pytest.raises(LLMUnavailable):
        await h.complete()
    assert time.perf_counter() - started < 0.5
    assert len(slow.calls) <= 3


async def test_an_empty_budget_stops_retries():
    model = ScriptedModel("primary", [http_error(500)], OK, [http_error(500)], OK)
    h = Harness(model, budget_factory=lambda: RetryBudget(ratio=0.0, burst=1.0, per_second=0.0))
    assert (await h.complete()).attempts == 2  # the one retry the budget allowed
    with pytest.raises(LLMUnavailable):
        await h.complete()
    assert len(model.calls) == 3


async def test_each_model_has_its_own_budget():
    primary = ScriptedModel("primary", [http_error(500)])
    fallback = ScriptedModel("fallback", [http_error(500)], OK)
    h = Harness(
        primary, fallback, budget_factory=lambda: RetryBudget(ratio=0.0, burst=1.0, per_second=0.0)
    )
    result = await h.complete()
    # The primary spent its one retry. The fallback still had its own.
    assert (len(primary.calls), len(fallback.calls)) == (2, 2)
    assert result.used_fallback is True


def test_a_quiet_budget_refills_with_time():
    now = [0.0]
    budget = RetryBudget(ratio=0.0, burst=2.0, per_second=0.5, clock=lambda: now[0])
    assert budget.try_spend() and budget.try_spend() and not budget.try_spend()
    now[0] = 2.0
    assert budget.try_spend()


async def test_an_injected_fault_takes_the_same_path_as_a_real_failure():
    model = ScriptedModel("primary", OK)
    plan = faults.FaultPlan()
    plan.add(faults.rate_limit(retry_after_s=1.5))
    h = Harness(model)
    result = await h.complete(faults=plan)
    assert result.message.content == "Hello"
    assert len(model.calls) == 1  # the faulted attempt never reached the model
    assert h.sleeps == [1.5]
    assert h.of("llm.retry")[0].reason == "rate_limited"
    assert len(plan) == 0


async def test_injected_faults_do_not_use_up_the_retry_budget():
    model = ScriptedModel("primary", OK, OK, [http_error(500)], OK)
    h = Harness(model, budget_factory=lambda: RetryBudget(ratio=0.0, burst=1.0, per_second=0.0))
    plan = faults.FaultPlan()
    for _ in range(2):  # two calls, each failing twice before it gets through
        plan.add(faults.server_error(), times=2)
        assert (await h.complete(faults=plan)).attempts == 3
    # Four injected failures later the budget still has its one real retry.
    assert (await h.complete()).attempts == 2


async def test_a_fault_aimed_at_the_fallback_leaves_the_primary_alone():
    plan = faults.FaultPlan()
    plan.add(faults.server_error(target="fallback"))
    h = Harness(ScriptedModel("primary", OK), ScriptedModel("fallback", OK))
    result = await h.complete(faults=plan)
    assert result.used_fallback is False
    assert len(plan) == 1


async def test_a_fault_meant_for_the_agent_does_not_hit_the_judge():
    plan = faults.FaultPlan()
    plan.add(faults.quota("any"))
    h = Harness(ScriptedModel("judge", OK))
    result = await h.complete(faults=plan, purpose="judge")
    assert result.attempts == 1
    assert len(plan) == 1


def test_a_session_cannot_queue_faults_without_limit():
    plan = faults.FaultPlan()
    assert plan.add(faults.timeout(), times=15) == 15
    assert plan.add(faults.timeout(), times=15) == faults.MAX_PENDING - 15
    assert len(plan) == faults.MAX_PENDING


async def test_a_bug_in_our_own_code_is_not_swallowed_as_an_outage():
    h = Harness(
        ScriptedModel("primary", ["Hel", ValueError("our bug")]), ScriptedModel("fallback", OK)
    )
    with pytest.raises(ValueError, match="our bug"):
        await h.complete()
    # The half answer is taken off the screen even so.
    assert h.types() == ["llm.started", "llm.delta", "llm.reset"]
