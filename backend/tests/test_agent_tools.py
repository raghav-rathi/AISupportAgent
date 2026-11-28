"""Tool rules with a scripted model: each one is enforced by code, whatever the model asks for."""

from __future__ import annotations

import json

from agent_harness import (
    ALWAYS_ANSWER,
    DANIEL,
    ELENA,
    GROUNDED,
    MAYA,
    REFUND_QUESTION,
    ScriptedJudge,
    Session,
    invoice_status,
    of,
    refunds,
    scalar,
    steps,
    system_text,
)
from fakes import ScriptedModel, call, calls, say
from langchain_core.messages import ToolMessage

from supportagent.agent.context import ToolFaults
from supportagent.agent.tools import search_knowledge_base


async def test_the_refund_chain_runs_four_tools_and_pays_once(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        call("get_account"),
        call("search_knowledge_base", query=REFUND_QUESTION),
        call("issue_refund", invoice_id="INV-30417", reason="changed_mind"),
        say("Done. I refunded $96.00 for invoice INV-30417 [1]."),
    )
    session = Session(pool, retriever, world, model, judge=ScriptedJudge(GROUNDED))
    events = await session.turn("Account 10482917, PIN 4721. Please refund my last invoice.")

    assert steps(events) == [
        ("verify_identity", "ok"),
        ("get_account", "ok"),
        ("search_knowledge_base", "ok"),
        ("issue_refund", "ok"),
    ]
    assert of(events, "tool.started")[0].args == {"account_number": "10482917", "pin": "****"}
    assert await invoice_status(pool, world, "INV-30417") == "refunded"
    assert await refunds(pool, world) == 1
    state = await session.state()
    assert state["customer"]["tier"] == "pro"
    assert state["outcome"] == "answered"
    assert of(events, "assistant.message")[-1].text.startswith("Done. I refunded $96.00")
    assert (session.ctx.usage.tool_calls, session.ctx.usage.llm_calls) == (4, 5)
    # The model is told who it is talking to only after the tool has verified them.
    assert "has not been verified" in system_text(model, 0)
    assert "Maya Okafor" not in system_text(model, 0)
    assert "Pro plan" in system_text(model, 1) and "Maya Okafor" in system_text(model, 1)


async def test_account_tools_refuse_until_identity_is_verified(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("issue_refund", invoice_id="INV-30417", reason="changed_mind"),
        call("get_account"),
        call("update_account_field", field="billing_email", value="thief@evil.example"),
        say("I need to verify your identity first."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Refund INV-30417 and change the billing email.")
    assert steps(events) == [
        ("issue_refund", "refused"),
        ("get_account", "refused"),
        ("update_account_field", "refused"),
    ]
    assert await refunds(pool, world) == 0
    changed = await scalar(
        pool,
        "SELECT count(*) FROM customers WHERE world_id = %s AND billing_email LIKE %s",
        world,
        "%evil%",
    )
    assert changed == 0


async def test_the_plan_filter_comes_from_session_state_not_from_the_model(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("verify_identity", **ELENA),
        call("search_knowledge_base", query="How long do I have to ask for a refund?"),
        say("Within 30 days [1]."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 53920164, PIN 2458. What is my refund window?")
    hits = of(events, "retrieval.result")[0].hits
    assert hits and all("enterprise" in hit.tiers for hit in hits)
    assert not any("Pro plan refunds" in hit.source for hit in hits)
    # The model cannot ask for another plan's documents: the tool takes a query and nothing else.
    schema = search_knowledge_base.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}


async def test_only_the_first_of_several_tool_calls_in_one_step_runs(pool, retriever, world):
    refund = ("issue_refund", {"invoice_id": "INV-30417", "reason": "other"})
    model = ScriptedModel("m", calls(("verify_identity", MAYA), refund), say("You are verified."))
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 10482917, PIN 4721, refund INV-30417.")
    assert steps(events) == [("verify_identity", "ok")]
    assert await refunds(pool, world) == 0
    messages = (await session.state())["messages"]
    assert len(messages[1].tool_calls) == 1
    assert sum(isinstance(m, ToolMessage) for m in messages) == 1


async def test_a_model_that_loops_is_stopped_and_handed_to_a_person(pool, retriever, world):
    model = ScriptedModel("m", *(call("get_account") for _ in range(10)))
    session = Session(pool, retriever, world, model, max_steps=3)
    events = await session.turn("Show me my account.")
    assert len(model.calls) == 3
    # The third step asked for a tool again. It did not run: no step was left to report it.
    assert len(of(events, "tool.started")) == 2
    assert of(events, "escalation.created")[0].ticket.reason == "step_limit"
    assert (await session.state())["outcome"] == "escalated"


async def test_a_failing_tool_is_retried_and_recovers(pool, retriever, world):
    model = ScriptedModel(
        "m", call("search_knowledge_base", query=REFUND_QUESTION), say("14 days [1].")
    )
    faults = ToolFaults({"search_knowledge_base": 2})
    session = Session(pool, retriever, world, model, tool_faults=faults, gate=ALWAYS_ANSWER)
    events = await session.turn(REFUND_QUESTION)
    tool_events = [e.type for e in events if e.type.startswith("tool.")]
    assert tool_events == [
        "tool.started",
        "tool.error",
        "tool.retry",
        "tool.error",
        "tool.retry",
        "tool.finished",
    ]
    assert [e.attempt for e in of(events, "tool.retry")] == [2, 3]
    assert steps(events) == [("search_knowledge_base", "ok")]


async def test_a_tool_that_keeps_failing_tells_the_model_so(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say("I can't reach the help center right now."),
    )
    faults = ToolFaults({"search_knowledge_base": 3})
    session = Session(pool, retriever, world, model, tool_faults=faults)
    events = await session.turn(REFUND_QUESTION)
    assert steps(events) == [("search_knowledge_base", "failed")]
    assert len(of(events, "tool.error")) == 3
    result = model.calls[1][0][-1]
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert json.loads(result.content)["error"] == "tool_unavailable"


async def test_malformed_tool_arguments_go_back_to_the_model_as_an_error(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        call("issue_refund", invoice_id="INV-30417", reason="because"),
        say("Sorry, let me try again."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 10482917, PIN 4721, refund INV-30417.")
    assert steps(events)[1] == ("issue_refund", "failed")
    assert json.loads(model.calls[2][0][-1].content)["error"] == "invalid_arguments"
    assert await refunds(pool, world) == 0


async def test_three_wrong_pins_lock_verification_and_open_a_ticket(pool, retriever, world):
    wrong = {"account_number": "10482917", "pin": "0000"}
    model = ScriptedModel(
        "m",
        call("verify_identity", **wrong),
        call("verify_identity", **wrong),
        call("verify_identity", **wrong),
        call("verify_identity", **MAYA),
        say("Verification is locked. A specialist will follow up."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 10482917, PIN 0000.")
    assert [outcome for _, outcome in steps(events)] == ["refused"] * 4
    assert of(events, "escalation.created")[0].ticket.reason == "verification_failed"
    state = await session.state()
    assert state.get("customer") is None  # the right PIN came too late
    assert state["failed_verifications"] == 3
    assert json.loads(model.calls[4][0][-1].content)["locked"] is True


async def test_a_call_without_a_pin_checks_nothing_and_costs_no_attempt(pool, retriever, world):
    # Seen on the backup model: for a customer who was verified it called the tool again, with
    # the PIN as the conversation held it by then, a mask, and no account number. Counted as a
    # wrong PIN, three such calls locked a verified customer out.
    masked = {"account_number": "", "pin": "****"}
    model = ScriptedModel(
        "m",
        call("verify_identity", **masked),
        say("What are your account number and PIN?"),
        call("verify_identity", **MAYA),
        call("verify_identity", **masked),
        call("verify_identity", **masked),
        call("verify_identity", **masked),
        call("get_account"),
        say("You are on the Pro plan."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Which plan am I on?")
    assert steps(events) == [("verify_identity", "refused")]
    assert (await session.state()).get("failed_verifications", 0) == 0

    events = await session.turn("Account 10482917, PIN 4721.")
    assert [outcome for _, outcome in steps(events)] == ["ok"] * 5
    assert of(events, "escalation.created") == []
    state = await session.state()
    assert state["customer"]["account_number"] == "10482917"
    assert state.get("failed_verifications", 0) == 0
    told = json.loads(model.calls[4][0][-1].content)
    assert told["verified"] is True and "get_account" in told["message"]


async def test_a_second_refund_request_in_a_fresh_tool_call_pays_nothing(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        call("issue_refund", invoice_id="INV-30417", reason="changed_mind"),
        call("issue_refund", invoice_id="30417", reason="changed_mind"),
        say("That invoice was already refunded."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 10482917, PIN 4721. Refund INV-30417 twice.")
    assert steps(events)[1:] == [("issue_refund", "ok"), ("issue_refund", "refused")]
    assert of(events, "tool.finished")[2].summary == "Refused: already_refunded"
    assert await refunds(pool, world) == 1


async def test_a_refund_over_the_limit_opens_a_billing_ticket_in_the_same_step(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("verify_identity", **DANIEL),
        call("issue_refund", invoice_id="INV-30388", reason="changed_mind"),
        say("That refund is above what I can approve. The billing team has your ticket."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn("Account 20573846, PIN 9034. Refund INV-30388.")
    ticket = of(events, "escalation.created")[0].ticket
    assert (ticket.reason, ticket.priority, ticket.tier) == ("refund_over_limit", "high", "pro")
    assert steps(events)[1] == ("issue_refund", "refused")
    assert json.loads(model.calls[2][0][-1].content)["ticket"]["id"] == ticket.id
    assert await invoice_status(pool, world, "INV-30388") == "paid"
    assert (await session.state())["outcome"] == "escalated"
