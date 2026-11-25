"""The scope gate and the grounding check, with a scripted model and a scripted judge."""

from __future__ import annotations

import pytest
from agent_harness import (
    ALWAYS_ANSWER,
    ALWAYS_CHECK,
    GATE,
    GROUNDED,
    REFUND_QUESTION,
    UNGROUNDED,
    ScriptedJudge,
    Session,
    of,
    scalar,
)
from fakes import ScriptedModel, call, say
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from supportagent.agent.graph import evidence_texts
from supportagent.agent.judge import JudgeError, Verdict, parse_verdict
from supportagent.agent.tools import CAPABILITIES
from supportagent.llm.errors import Failure, LLMUnavailable


async def test_a_search_that_finds_nothing_goes_to_a_person_and_the_model_never_answers(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query="weather forecast for Boston"),
        say("It will be sunny in Boston tomorrow."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("What's the weather in Boston tomorrow?")

    assert len(model.calls) == 1
    assert of(events, "retrieval.result")[0].band == "escalate"
    ticket = of(events, "escalation.created")[0].ticket
    assert ticket.reason == "low_confidence"
    final = of(events, "assistant.message")[-1]
    assert ticket.id in final.text
    assert final.hits == []  # passages the model never got are not sources of anything
    assert of(events, "llm.delta") == []
    state = await session.state()
    assert state["outcome"] == "escalated"
    assert [t["id"] for t in state["tickets"]] == [ticket.id]
    assert state["messages"][-1].content == final.text
    stored = await scalar(
        pool, "SELECT reason FROM tickets WHERE world_id = %s AND id = %s", world, ticket.id
    )
    assert stored == "low_confidence"


async def test_a_mid_confidence_answer_waits_for_the_grounding_check(pool, retriever, world):
    judge = ScriptedJudge(GROUNDED)
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say("You have 14 days from the invoice date [1]."),
    )
    session = Session(pool, retriever, world, model, judge=judge, gate=ALWAYS_CHECK)
    events = await session.turn(REFUND_QUESTION)

    assert of(events, "llm.delta") == []  # nothing reached the customer before the verdict
    kinds = [e.type for e in events]
    assert kinds.index("guardrail.verdict") < kinds.index("assistant.message")
    verdict = of(events, "guardrail.verdict")[0]
    assert (verdict.verdict, verdict.held) == ("grounded", True)
    message = of(events, "assistant.message")[0]
    assert message.held is True
    assert message.text == "You have 14 days from the invoice date [1]."
    assert message.hits[0].n == 1
    # The judge read the passages the search returned, labelled with the tool they came from.
    evidence = judge.calls[0]["evidence"]
    assert any(text.startswith("tool: search_knowledge_base") for text in evidence)
    assert any("Refund policy" in text for text in evidence)


async def test_a_mid_confidence_answer_that_fails_the_check_never_reaches_the_customer(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say("Refunds take 365 days."),
    )
    session = Session(
        pool, retriever, world, model, judge=ScriptedJudge(UNGROUNDED), gate=ALWAYS_CHECK
    )
    events = await session.turn(REFUND_QUESTION)

    assert of(events, "llm.delta") == []
    assert of(events, "answer.retracted") == []  # nothing was shown, so nothing to take back
    assert of(events, "escalation.created")[0].ticket.reason == "ungrounded_answer"
    shown = [e.text for e in of(events, "assistant.message")]
    assert len(shown) == 1 and "365 days" not in shown[0]
    state = await session.state()
    assert not any("365 days" in str(m.content) for m in state["messages"])
    assert state["outcome"] == "escalated"


async def test_a_streamed_answer_that_fails_the_check_is_retracted(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say("Refunds take 365 days."),
    )
    session = Session(
        pool, retriever, world, model, judge=ScriptedJudge(UNGROUNDED), gate=ALWAYS_ANSWER
    )
    events = await session.turn(REFUND_QUESTION)

    kinds = [e.type for e in events]
    assert "llm.delta" in kinds
    order = [
        kinds.index(k) for k in ("guardrail.verdict", "answer.retracted", "escalation.created")
    ]
    assert order == sorted(order)
    assert of(events, "guardrail.verdict")[0].unsupported == ["refunds take 365 days"]
    state = await session.state()
    assert not any("365 days" in str(m.content) for m in state["messages"])
    assert isinstance(state["messages"][-1], AIMessage)
    assert of(events, "escalation.created")[0].ticket.id in state["messages"][-1].content


async def test_an_answer_given_without_any_search_is_checked_too(pool, retriever, world):
    judge = ScriptedJudge(Verdict(grounded=False, unsupported=["Canberra is the capital"]))
    model = ScriptedModel("m", say("Canberra is the capital of Australia."))
    session = Session(pool, retriever, world, model, judge=judge)
    events = await session.turn("What's the capital of Australia?")
    # Nothing was looked up. All the judge has is what the assistant's tools can do.
    assert judge.calls[0]["evidence"] == [CAPABILITIES]
    assert of(events, "answer.retracted")
    assert (await session.state())["outcome"] == "escalated"


async def test_an_answer_that_cannot_be_checked_is_taken_back_and_checked_on_retry(
    pool, retriever, world
):
    judge = ScriptedJudge(JudgeError("garbled"), GROUNDED)
    model = ScriptedModel("m", say("Could you give me your account number and PIN?"))
    session = Session(pool, retriever, world, model, judge=judge)
    with pytest.raises(LLMUnavailable):
        await session.turn("I want a refund.")
    # The text had streamed. With no verdict it comes off the screen, and no ticket is opened
    # for what is an outage of ours and not a problem with the answer.
    assert of(session.events, "guardrail.verdict")[0].verdict == "skipped"
    assert of(session.events, "answer.retracted")
    assert of(session.events, "escalation.created") == []
    assert await scalar(pool, "SELECT count(*) FROM tickets WHERE world_id = %s", world) == 0

    events = await session.turn(None)
    assert of(events, "guardrail.verdict")[0].verdict == "grounded"
    assert of(events, "assistant.message")[-1].text.startswith("Could you give me")
    assert len(model.calls) == 1  # only the check ran again, not the model
    state = await session.state()
    assert (state["outcome"], state["unchecked"]) == ("answered", None)


async def test_a_held_answer_is_not_released_when_the_judge_is_down(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say("You have 14 days [1]."),
    )
    judge = ScriptedJudge(LLMUnavailable([Failure("timeout", retryable=True)]))
    session = Session(pool, retriever, world, model, judge=judge, gate=ALWAYS_CHECK)
    with pytest.raises(LLMUnavailable):
        await session.turn(REFUND_QUESTION)
    assert of(session.events, "assistant.message") == []
    assert of(session.events, "answer.retracted") == []  # it was never shown
    assert await scalar(pool, "SELECT count(*) FROM tickets WHERE world_id = %s", world) == 0


async def test_an_unchecked_answer_is_gone_when_the_customer_moves_on_instead_of_retrying(
    pool, retriever, world
):
    judge = ScriptedJudge(JudgeError("garbled"), GROUNDED)
    model = ScriptedModel("m", say("Refunds take 365 days."), say("Hello again."))
    session = Session(pool, retriever, world, model, judge=judge)
    with pytest.raises(LLMUnavailable):
        await session.turn("How long do refunds take?")
    await session.turn("Never mind. Hello?")

    # The model's second call did not read the answer nobody checked.
    assert not any("365" in str(m.content) for m in model.calls[1][0])
    messages = (await session.state())["messages"]
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert not any("365" in str(m.content) for m in messages)
    assert messages[-1].content == "Hello again."


async def test_the_judge_reads_the_customers_recent_messages_with_pins_masked(
    pool, retriever, world
):
    judge = ScriptedJudge(GROUNDED)
    model = ScriptedModel("m", say("Which invoice do you mean?"), say("INV-30417, understood."))
    session = Session(pool, retriever, world, model, judge=judge)
    await session.turn("I want a refund for INV-30417. My PIN is 4721.")
    await session.turn("That one.")
    assert judge.calls[1]["question"] == "That one."
    assert judge.calls[1]["earlier"] == ["I want a refund for INV-30417. My PIN is ****."]


def tool_result(name: str, content: str, call_id: str) -> ToolMessage:
    return ToolMessage(content, name=name, tool_call_id=call_id)


def test_the_judge_always_gets_this_turns_results_and_the_latest_account_reading():
    old_searches = [
        tool_result("search_knowledge_base", f"passage {n} " + "x" * 3000, f"s{n}")
        for n in range(8)
    ]
    state = {
        "customer": {"account_number": "10482917", "tier": "pro"},
        "messages": [
            HumanMessage("what is on my account?"),
            tool_result("get_account", '{"invoices": ["INV-30417"]}', "a1"),
            *old_searches,
            HumanMessage("refund INV-30417"),
            tool_result("issue_refund", '{"status": "refunded"}', "r1"),
        ],
    }
    evidence = evidence_texts(state)
    assert evidence[0] == CAPABILITIES
    assert evidence[1].startswith("Verified customer:")
    assert 'tool: get_account\n{"invoices": ["INV-30417"]}' in evidence
    assert evidence[-1] == 'tool: issue_refund\n{"status": "refunded"}'
    # Older passages fill what room is left and no more.
    assert sum(len(text) for text in evidence[2:]) <= 16_000
    assert 3 <= sum(text.startswith("tool: search_knowledge_base") for text in evidence) < 8


def test_one_oversized_result_does_not_push_out_the_smaller_ones():
    state = {
        "messages": [
            HumanMessage("q"),
            tool_result("get_account", "account data", "a1"),
            tool_result("search_knowledge_base", "y" * 20_000, "s1"),
            tool_result("issue_refund", "refund data", "r1"),
        ]
    }
    assert evidence_texts(state) == [
        CAPABILITIES,
        "tool: get_account\naccount data",
        "tool: issue_refund\nrefund data",
    ]


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (None, "escalate"),
        (float("nan"), "escalate"),
        (-4.01, "escalate"),
        (-4.0, "check"),
        (1.99, "check"),
        (2.0, "answer"),
    ],
)
def test_gate_bands(score, band):
    assert GATE.band(score) == band


@pytest.mark.parametrize(
    ("visible", "other", "verified", "band"),
    [
        (None, 5.0, False, "verify"),  # nothing public, and a plan's passage answers
        (None, 5.0, True, "restricted"),
        (-9.0, -0.5, False, "verify"),  # a fair match is enough to ask a visitor to verify
        (-9.0, -0.5, True, "escalate"),  # and too weak to tell a customer whose topic it is
        (-3.5, 7.9, False, "verify"),  # something weak is visible, a plan answers far better
        (-3.5, 7.9, True, "restricted"),
        (1.0, 3.5, False, "check"),  # a little better is not enough
        (-3.5, 1.5, False, "check"),  # better by the gap, yet no confident match itself
        (5.0, 9.0, True, "answer"),  # what the customer may see already answers
        (-9.0, -5.0, False, "escalate"),  # no plan answers
        (-9.0, None, False, "escalate"),
        (-9.0, float("nan"), False, "escalate"),
    ],
)
def test_the_gate_tells_another_plans_answer_from_no_answer(visible, other, verified, band):
    assert GATE.decide(visible, other, verified=verified) == band


def test_a_verdict_is_read_out_of_surrounding_text():
    text = 'Here is my verdict:\n```json\n{"grounded": false, "unsupported": ["x"]}\n```'
    assert parse_verdict(text) == Verdict(grounded=False, unsupported=["x"])
    noisy = 'I looked at {the evidence}. {"grounded": true, "unsupported": []} {"extra": 1}'
    assert parse_verdict(noisy) == Verdict(grounded=True)
    with pytest.raises(JudgeError):
        parse_verdict("I think it is fine.")
    with pytest.raises(JudgeError):
        parse_verdict('{"grounded": "maybe"}')


def test_a_verdict_that_lists_unsupported_statements_is_not_grounded_whatever_it_says():
    verdict = parse_verdict('{"grounded": true, "unsupported": ["the $500 bonus"]}')
    assert verdict == Verdict(grounded=False, unsupported=["the $500 bonus"])
