"""A question that only another plan's passages answer is neither answered from weaker passages
nor given up on. An unverified visitor is asked to verify. A verified customer is told which plan
the topic belongs to. The passages themselves stay out of reach in both cases.
"""

from __future__ import annotations

import json

from agent_harness import ALWAYS_ANSWER, ELENA, MAYA, REFUND_QUESTION, Session, of, scalar, steps
from fakes import ScriptedModel, call, say

P1_QUESTION = "How fast do you respond when we report a P1 outage?"
SIGN_IN_QUESTION = "How do we set up Google sign-in for our company domain?"
SOC2_QUESTION = "Do you have a SOC 2 Type II report?"


async def test_a_plan_question_from_an_unverified_visitor_asks_for_verification(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=P1_QUESTION),
        say("I can answer that for your plan once you're verified. Account number and PIN?"),
        call("verify_identity", **ELENA),
        call("search_knowledge_base", query=P1_QUESTION),
        say("Within 1 hour [1]."),
    )
    session = Session(pool, retriever, world, model)

    events = await session.turn(P1_QUESTION)
    assert of(events, "retrieval.result")[0].band == "verify"
    assert steps(events) == [("search_knowledge_base", "refused")]
    assert of(events, "escalation.created") == []
    assert (await session.state())["outcome"] == "answered"
    assert await scalar(pool, "SELECT count(*) FROM tickets WHERE world_id = %s", world) == 0
    # The model learns that it must verify, and nothing of the passages it may not see yet.
    told = model.calls[1][0][-1].content
    assert json.loads(told)["results"] == []
    assert "verify_identity" in told
    assert "1 hour" not in told and "P1" not in told

    events = await session.turn("My account number is 53920164 and my PIN is 2458.")
    answered = of(events, "retrieval.result")[0]
    assert answered.band in ("answer", "check")
    assert any("Enterprise plan support" in hit.source for hit in answered.hits)


async def test_a_weak_public_match_does_not_stand_in_for_a_plans_answer(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=SIGN_IN_QUESTION),
        say("That depends on your plan. Could you give me your account number and PIN?"),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn(SIGN_IN_QUESTION)

    result = of(events, "retrieval.result")[0]
    # Something public matched well enough to pass the low threshold. A plan's passage matched
    # far better, and that decides.
    assert result.band == "verify"
    assert result.top_score >= result.low
    assert result.plan_score - result.top_score >= 3.0
    assert json.loads(model.calls[1][0][-1].content)["results"] == []
    assert of(events, "assistant.message")[-1].hits == []
    assert of(events, "escalation.created") == []


async def test_a_verified_customer_is_told_when_the_topic_belongs_to_another_plan(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        call("search_knowledge_base", query=SOC2_QUESTION),
        say("That is covered for the Enterprise plan and is not part of your Pro plan."),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn(f"Account 10482917, PIN 4721. {SOC2_QUESTION}")

    assert of(events, "retrieval.result")[0].band == "restricted"
    told = json.loads(model.calls[2][0][-1].content)
    assert (told["results"], told["covered_for_plans"], told["customer_plan"]) == (
        [],
        ["enterprise"],
        "pro",
    )
    assert "SOC" not in told["message"]  # which plan covers it, and nothing of what it says
    # Before this rule the customer was told the help center had no answer and got a ticket.
    assert of(events, "escalation.created") == []
    assert (await session.state())["outcome"] == "answered"


async def test_a_locked_out_visitor_is_not_asked_to_verify_again(pool, retriever, world):
    wrong = {"account_number": "53920164", "pin": "0000"}
    model = ScriptedModel(
        "m",
        call("verify_identity", **wrong),
        call("verify_identity", **wrong),
        call("verify_identity", **wrong),
        call("search_knowledge_base", query=P1_QUESTION),
        say("That depends on your plan and I can't verify you here. Shall I pass it on?"),
    )
    session = Session(pool, retriever, world, model)
    events = await session.turn(P1_QUESTION)
    assert of(events, "retrieval.result")[0].band == "verify"
    told = json.loads(model.calls[4][0][-1].content)["message"]
    assert "locked" in told and "verify_identity" not in told


async def test_a_question_no_plan_answers_still_goes_to_a_person(pool, retriever, world):
    model = ScriptedModel("m", call("search_knowledge_base", query="weather forecast for Boston"))
    session = Session(pool, retriever, world, model)
    events = await session.turn("What's the weather in Boston tomorrow?")
    assert of(events, "retrieval.result")[0].band == "escalate"
    assert of(events, "escalation.created")[0].ticket.reason == "low_confidence"


async def test_citation_numbers_continue_across_the_searches_of_one_turn(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        call("search_knowledge_base", query="How do I reset my password?"),
        say("Refunds are covered in [1] and password resets in [6]."),
    )
    session = Session(pool, retriever, world, model, gate=ALWAYS_ANSWER)
    events = await session.turn("Two questions: refunds and password resets.")

    first, second = of(events, "retrieval.result")
    assert [hit.n for hit in first.hits] == [1, 2, 3, 4, 5]
    assert [hit.n for hit in second.hits] == [6, 7, 8, 9, 10]
    assert json.loads(model.calls[2][0][-1].content)["results"][0]["n"] == 6
    # Each number in the answer points at exactly one passage.
    cited = of(events, "assistant.message")[-1].hits
    assert [hit.n for hit in cited] == list(range(1, 11))
