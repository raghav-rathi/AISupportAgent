"""What a session keeps: memory across turns, and its state through a model outage."""

from __future__ import annotations

import pytest
from agent_harness import MAYA, Session, llm_for, of, steps, system_text
from fakes import ScriptedModel, call, http_error, say
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from supportagent.agent.prompts import for_prompt
from supportagent.llm.errors import LLMUnavailable


async def test_a_model_outage_leaves_the_session_intact_and_the_turn_resumable(
    pool, retriever, world
):
    session = Session(pool, retriever, world, ScriptedModel("down", [http_error(401)]))
    with pytest.raises(LLMUnavailable):
        await session.turn("Hello, are you there?")
    assert [type(m) for m in (await session.state())["messages"]] == [HumanMessage]

    session.args["llm"] = llm_for(ScriptedModel("up", say("Yes. How can I help?")))
    events = await session.turn(None)
    assert of(events, "assistant.message")[0].text == "Yes. How can I help?"
    # The customer's message is there once: resuming did not send it again.
    assert [type(m) for m in (await session.state())["messages"]] == [HumanMessage, AIMessage]


async def test_what_one_turn_establishes_the_next_turn_can_use(pool, retriever, world):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        say("Thanks, you are verified."),
        call("get_account"),
        say("You have three invoices."),
    )
    session = Session(pool, retriever, world, model)
    await session.turn("Account 10482917, PIN 4721.")
    events = await session.turn("What invoices do I have?")
    # The second turn reads the account without verifying again.
    assert steps(events) == [("get_account", "ok")]
    assert "Maya Okafor" in system_text(model, 2)
    kinds = [type(m) for m in (await session.state())["messages"]]
    assert kinds == [HumanMessage, AIMessage, ToolMessage, AIMessage] * 2


def test_search_results_from_earlier_turns_are_left_out_of_the_prompt():
    def search(n):
        return ToolMessage(f"results {n}", tool_call_id=f"s{n}", name="search_knowledge_base")

    account = ToolMessage("account data", tool_call_id="a1", name="get_account")
    earlier = [HumanMessage("first question"), search(1), account, AIMessage("an answer")]
    history = [*earlier, HumanMessage("second question"), search(2), search(3)]
    seen = [m.content for m in for_prompt(history)]
    assert "omitted" in seen[1]
    assert seen[2:4] == ["account data", "an answer"]  # only search results are dropped
    assert seen[5:] == ["results 2", "results 3"]  # this turn keeps all of its own
    assert history[1].content == "results 1"  # the saved history is untouched
