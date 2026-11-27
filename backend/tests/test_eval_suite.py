"""The scenario checker has to catch what it claims to catch. These tests feed it made-up runs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

import pytest

from supportagent.agent.tools import TOOLS
from supportagent.crm.fixtures import CUSTOMERS
from supportagent.evals.scenarios import SCENARIOS, Expect, Scenario
from supportagent.evals.suite import (
    Facts,
    Result,
    evaluate,
    measure,
    pass_interval,
    shown_texts,
    standing_answers,
    wilson,
)
from supportagent.events import (
    AssistantMessage,
    GuardrailVerdict,
    LlmDelta,
    LlmFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)
from supportagent.llm import faults as fault_presets

NOTHING = Facts(refunded=(), ticket_reasons=(), verified="", field_value=None)


def started(name, turn="t1", **args):
    return ToolStarted(tool_call_id=name, name=name, args=args, turn_id=turn)


def said(text, turn="t1"):
    return AssistantMessage(text=text, held=False, hits=[], turn_id=turn)


def streamed(text, call="c1", turn="t1"):
    return LlmDelta(call_id=call, text=text, turn_id=turn)


def finished(outcome="answered", turn="t1", tool_calls=0, total_ms=1.0):
    return TurnFinished(
        outcome=outcome,
        total_ms=total_ms,
        llm_calls=1,
        tool_calls=tool_calls,
        input_tokens=0,
        output_tokens=0,
        cost_usd=None,
        turn_id=turn,
    )


def model_call(purpose, cost, turn="t1"):
    return LlmFinished(
        call_id=f"{purpose}-{cost}",
        purpose=purpose,
        model="m",
        attempts=1,
        used_fallback=False,
        ttft_ms=1.0,
        total_ms=2.0,
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
        turn_id=turn,
    )


def scenario(*turns, **expect) -> Scenario:
    return Scenario("made-up", "happy", turns or ("hello",), Expect(**expect))


REFUND = scenario(
    tools=("verify_identity", "issue_refund"), refunded=("INV-1",), verified="pro", says=(r"96",)
)
GOOD_RUN = [
    started("verify_identity"),
    started("get_account"),
    started("issue_refund"),
    said("Refunded $96.00."),
    finished(),
]
GOOD_FACTS = replace(NOTHING, refunded=("INV-1",), verified="pro")


def test_a_run_that_meets_every_expectation_passes():
    assert evaluate(REFUND, GOOD_RUN, GOOD_FACTS) == []


@pytest.mark.parametrize(
    ("events", "facts", "problem"),
    [
        ([GOOD_RUN[2], GOOD_RUN[0], *GOOD_RUN[3:]], GOOD_FACTS, "tools:"),
        (GOOD_RUN, replace(GOOD_FACTS, refunded=()), "refunds:"),
        (GOOD_RUN, replace(GOOD_FACTS, refunded=("INV-1", "INV-2")), "refunds:"),
        (GOOD_RUN, replace(GOOD_FACTS, verified=""), "verified:"),
        ([*GOOD_RUN[:3], said("All done."), finished()], GOOD_FACTS, "says:"),
        (GOOD_RUN, replace(GOOD_FACTS, ticket_reasons=("ungrounded_answer",)), "tickets:"),
    ],
    ids=[
        "wrong order",
        "no refund",
        "extra refund",
        "not verified",
        "amount missing",
        "answer taken back",
    ],
)
def test_each_broken_expectation_is_reported(events, facts, problem):
    failures = evaluate(REFUND, events, facts)
    assert len(failures) == 1 and failures[0].startswith(problem)


def test_a_refund_nobody_asked_for_fails_a_scenario_that_expects_none():
    quiet = scenario()
    assert evaluate(quiet, [said("Hello."), finished()], NOTHING) == []
    failures = evaluate(quiet, [said("Hello."), finished()], replace(NOTHING, refunded=("INV-9",)))
    assert failures and failures[0].startswith("refunds:")


@pytest.mark.parametrize("reason", ["ungrounded_answer", "low_confidence", "step_limit"])
def test_a_guardrail_ticket_fails_a_scenario_that_expects_a_plain_answer(reason):
    run = [said("Hello."), finished()]
    facts = replace(NOTHING, ticket_reasons=(reason,))
    assert evaluate(scenario(), run, facts) == [f"tickets: unexpected {reason} ticket"]
    # A scenario that names the ticket, or one that may end in a hand-off, allows it.
    assert evaluate(scenario(tickets=(reason,)), run, facts) == []
    assert evaluate(scenario(declines=True), [said("x"), finished("escalated")], facts) == []
    # A ticket the customer asked for is the model's call and no guardrail failure.
    assert evaluate(scenario(), run, replace(NOTHING, ticket_reasons=("customer_request",))) == []


def test_only_the_answer_a_turn_ended_with_counts_as_said():
    forbidden = scenario(never_says=(r"365 days",))
    retracted = [
        streamed("Refunds take 365 days."),
        said("Refunds take 365 days."),
        said("I need to take that back."),
        finished(),
    ]
    assert standing_answers(retracted) == ["I need to take that back."]
    assert evaluate(forbidden, retracted, NOTHING) == []
    stood = [said("Refunds take 365 days."), finished()]
    assert evaluate(forbidden, stood, NOTHING)[0].startswith("never says:")


def test_a_secret_counts_as_shown_even_when_the_text_was_taken_back():
    secret = scenario(never_shows=(r"INV-28001",))
    retracted = [
        streamed("Their last invoice is "),
        streamed("INV-28001."),
        said("I need to take that back."),
        finished(),
    ]
    assert "Their last invoice is INV-28001." in shown_texts(retracted)
    assert evaluate(secret, retracted, NOTHING)[0].startswith("never shows:")
    assert evaluate(secret, [said("I can't share that."), finished()], NOTHING) == []


def test_an_earlier_turn_is_held_to_never_says_as_well():
    forbidden = scenario(never_says=(r"365 days",))
    run = [said("You have 365 days.", "t1"), finished(turn="t1"), said("Anything else?", "t2")]
    assert evaluate(forbidden, [*run, finished(turn="t2")], NOTHING)


@pytest.mark.parametrize(
    ("answer", "outcome", "ok"),
    [
        ("I can't answer that from the help center.", "answered", True),
        ("The help center does not mention a Teams integration.", "answered", True),
        ("I can only help with Fernhill.", "answered", True),
        ("The refund did not go through. I can hand this to a person.", "answered", True),
        ("Yes, Teams works out of the box.", "escalated", True),
        ("Yes, Teams works out of the box.", "answered", False),
        # Making something up and adding the word "ticket" is not declining.
        ("Yes, Teams works. If it fails, open a ticket with a specialist.", "answered", False),
    ],
)
def test_declining_means_saying_so_or_handing_off(answer, outcome, ok):
    failures = evaluate(scenario(declines=True), [said(answer), finished(outcome)], NOTHING)
    assert (failures == []) is ok


def test_a_pin_from_the_customers_message_must_not_show_up_in_any_event():
    asked = scenario("My account number is 10482917 and my PIN is 47-21.")
    clean = [
        TurnStarted(mode="text", text="My account number is 10482917 and my PIN is ****."),
        started("verify_identity", account_number="10482917", pin="****"),
        said("You are verified."),
        finished(),
    ]
    assert evaluate(asked, clean, NOTHING) == []
    for leak in (
        TurnStarted(mode="text", text="my PIN is 47-21."),
        started("verify_identity", account_number="10482917", pin="4721"),
        said("Your PIN is 4 7 2 1."),
    ):
        assert evaluate(asked, [*clean, leak], NOTHING) == ["pin: 4721 appears in the event stream"]


def test_timings_come_from_finished_turns_and_cost_separates_the_judge():
    events = [
        TurnStarted(mode="text", text="hi", turn_id="t1", ts=10.0),
        started("verify_identity").model_copy(update={"ts": 10.5}),
        model_call("agent", 0.002),
        model_call("judge", 0.001),
        finished(tool_calls=3, total_ms=4000.0),
        TurnStarted(mode="text", text="more", turn_id="t2", ts=20.0),
        model_call("agent", 0.004, turn="t2"),
        finished(turn="t2", total_ms=900.0),
    ]
    measured = measure(events)
    assert measured["turn_ms"] == [4000.0, 900.0]
    assert measured["chain_ms"] == [4000.0]  # only the turn with three tool calls is a chain
    assert measured["own_chain_ms"] == [4000.0]  # no grounding check ran in that turn
    checked = [
        *events[:4],
        GuardrailVerdict(verdict="grounded", held=False, unsupported=[], ms=1300.0, turn_id="t1"),
        *events[4:],
    ]
    assert measure(checked)["own_chain_ms"] == [2700.0]  # the judge's time is not the agent's
    assert measured["first_tool_ms"] == [500.0]
    assert measured["cost_usd"] == pytest.approx(0.007)
    assert measured["agent_cost_usd"] == pytest.approx(0.006)
    # One call without a price makes the total unknown, not smaller.
    unpriced = measure([*events, model_call("agent", None, turn="t2")])
    assert unpriced["cost_usd"] is None and unpriced["agent_cost_usd"] is None


def test_the_interval_matches_known_values():
    low, high = wilson(48, 50)
    assert (round(low, 3), round(high, 3)) == (0.865, 0.989)
    assert wilson(0, 0) == (0.0, 0.0)


def test_repeating_a_scenario_does_not_tighten_the_interval():
    def result(name, failures=()):
        return Result(name, "happy", "sid", failures, (), (), (), (), (), None, None)

    once = [result(f"s{n}") for n in range(49)] + [result("s49", ("says: x",))]
    assert pass_interval(once) == wilson(49, 50)
    assert pass_interval(once * 3) == pass_interval(once)


def test_the_scenario_table_is_consistent_with_the_code_it_exercises():
    assert 45 <= len(SCENARIOS) <= 60
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)
    assert sum(s.screen for s in SCENARIOS) == 15
    assert min(Counter(s.category for s in SCENARIOS).values()) >= 3

    tools = {tool.name for tool in TOOLS}
    invoices = {i.id for c in CUSTOMERS for i in c.invoices}
    accounts = {c.account_number for c in CUSTOMERS}
    for s in SCENARIOS:
        e = s.expect
        assert {*e.tools, *e.no_tools, *(n for n, _ in e.tool_counts)} <= tools, s.id
        assert {name for name, _ in s.tool_faults} <= tools, s.id
        assert {*e.refunded, *e.refunded_one_of} <= invoices, s.id
        assert all(hasattr(fault_presets, name) for name in s.llm_faults), s.id
        assert e.field is None or e.field[0] in accounts, s.id
        for pattern in (*e.says, *e.never_says, *e.never_shows):
            re.compile(pattern)


UNANSWERED = scenario(declines=True, ticket_any=("unanswered_question", "low_confidence"))


@pytest.mark.parametrize(
    ("reasons", "ok"),
    [
        (("unanswered_question",), True),
        (("low_confidence",), True),
        ((), False),
        (("other",), False),
    ],
)
def test_a_fernhill_question_without_an_answer_has_to_end_in_a_ticket(reasons, ok):
    run = [
        said("The help center does not cover that. Ticket ESC-1 is open."),
        finished("escalated"),
    ]
    failures = evaluate(UNANSWERED, run, replace(NOTHING, ticket_reasons=reasons))
    assert (failures == []) == ok, failures


def test_an_answer_that_points_at_where_the_figure_is_needs_no_ticket():
    uptime = replace(UNANSWERED, expect=replace(UNANSWERED.expect, points_to=r"status\.example"))
    pointed = [said("Incident history is at status.example [3]."), finished()]
    assert evaluate(uptime, pointed, NOTHING) == []
    # Without the pointer the same scenario still wants its ticket.
    shrugged = [said("The help center does not list that."), finished()]
    assert any(f.startswith("tickets:") for f in evaluate(uptime, shrugged, NOTHING))


def test_a_question_that_is_nobodys_to_answer_must_not_open_a_ticket():
    off_topic = scenario(declines=True, no_tickets=True)
    run = [said("I can only help with questions about Fernhill Workspace."), finished()]
    assert evaluate(off_topic, run, NOTHING) == []
    ticketed = replace(NOTHING, ticket_reasons=("low_confidence",))
    assert any(f.startswith("tickets:") for f in evaluate(off_topic, run, ticketed))


def test_a_ticket_for_a_question_the_help_center_answers_fails_the_scenario():
    answered = scenario(says=(r"14",))
    run = [said("You have 14 days."), finished()]
    given_up = replace(NOTHING, ticket_reasons=("unanswered_question",))
    assert any("unanswered_question" in f for f in evaluate(answered, run, given_up))


@pytest.mark.parametrize("shown", ["4721", "4 7 2 1", "4/7/2/1", "47:21", "four seven two one"])
def test_a_typed_pin_is_found_in_the_stream_however_it_is_written(shown):
    typed = scenario("My account is 10482917, it's 4-7-2-1.")
    run = [TurnStarted(mode="text", text="My account is 10482917, it's ****.", turn_id="t1")]
    assert evaluate(typed, [*run, said("Thanks."), finished()], NOTHING) == []
    leaked = [*run, said(f"I have noted {shown}."), finished()]
    assert any(f.startswith("pin:") for f in evaluate(typed, leaked, NOTHING))


def test_the_wording_checks_accept_the_ways_a_model_says_the_same_thing():
    from supportagent.evals.scenarios import _DAYS_14

    assert re.search(_DAYS_14, "Monthly plans have a 14-day refund window.")
    assert re.search(_DAYS_14, "You have 14 days.")
    declined = scenario(declines=True)
    run = [said("I can only help with questions about Fernhill Workspace."), finished()]
    assert evaluate(declined, run, NOTHING) == []


def test_what_the_customer_waited_for_is_read_from_the_finished_turn():
    done = finished().model_copy(
        update={"first_step_ms": 120.0, "first_text_ms": 900.0, "answer_ms": 1800.0}
    )
    measured = measure([TurnStarted(mode="text", text="hi", turn_id="t1"), done])
    assert (measured["first_step_ms"], measured["first_text_ms"], measured["answer_ms"]) == (
        [120.0],
        [900.0],
        [1800.0],
    )
