"""run_turn end to end: the graph, the event log and the stored stream, with a scripted model."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from agent_harness import MAYA, llm_for, runtime
from fakes import ScriptedModel, call, http_error, say

from supportagent.eventlog import EventLog, TurnInProgress
from supportagent.session import NothingToResume, can_resume, run_turn


async def turn(rt, session_id, world, text):
    events = []
    done = await run_turn(
        rt, session_id=session_id, world_id=world, text=text, on_event=events.append
    )
    return done, events


async def stored_events(pool, session_id, world):
    return await (await EventLog.open(pool, session_id, world)).since(0)


async def test_a_turn_is_numbered_in_order_and_stored_without_token_deltas(pool, retriever, world):
    model = ScriptedModel("m", call("verify_identity", **MAYA), say("You are verified."))
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    done, events = await turn(rt, session_id, world, "Account 10482917, PIN 4721.")

    assert done is True
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert len({e.turn_id for e in events}) == 1 and events[0].turn_id
    assert (events[0].type, events[-1].type) == ("turn.started", "turn.finished")
    assert events[-1].tool_calls == 1 and events[-1].llm_calls == 2
    assert events[-1].cost_usd is None  # the scripted model has no price, so no cost is claimed
    assert any(e.type == "llm.delta" for e in events)

    stored = await stored_events(pool, session_id, world)
    assert [e.type for e in stored] == [e.type for e in events if e.type != "llm.delta"]
    assert stored == [e for e in events if e.type != "llm.delta"]  # the round trip loses nothing
    started = next(e for e in stored if e.type == "tool.started")
    assert started.args["pin"] == "****"  # the PIN is masked before it is stored
    assert events[0].text == "Account 10482917, PIN ****."


async def test_the_next_turn_continues_the_numbering_and_a_client_can_catch_up(
    pool, retriever, world
):
    model = ScriptedModel("m", say("Hello."), say("Hello again."))
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    _, first = await turn(rt, session_id, world, "Hi")
    _, second = await turn(rt, session_id, world, "Hi again")

    assert second[0].seq == first[-1].seq + 1
    assert second[0].turn_id != first[0].turn_id
    missed = await (await EventLog.open(pool, session_id, world)).since(first[-1].seq)
    assert [e.seq for e in missed] == [e.seq for e in second if e.type != "llm.delta"]


async def test_a_failed_turn_is_reported_stored_and_can_be_resumed(pool, retriever, world):
    rt = runtime(pool, retriever, ScriptedModel("down", [http_error(401)]))
    session_id = uuid.uuid4().hex
    done, events = await turn(rt, session_id, world, "Hello?")
    assert done is False
    failed = events[-1]
    assert (failed.type, failed.code, failed.retryable) == ("turn.failed", "llm_unavailable", True)
    assert (await stored_events(pool, session_id, world))[-1].type == "turn.failed"
    assert await can_resume(rt, session_id)

    rt.llm = llm_for(ScriptedModel("up", say("Yes, I'm here.")))
    done, resumed = await turn(rt, session_id, world, None)
    assert done is True
    assert resumed[0].seq == failed.seq + 1
    assert not any(e.type == "turn.started" for e in resumed)  # same turn, not a new message
    assert next(e for e in resumed if e.type == "assistant.message").text == "Yes, I'm here."
    assert not await can_resume(rt, session_id)


async def test_a_resumed_turn_keeps_its_id_and_adds_to_what_it_had_used(pool, retriever, world):
    model = ScriptedModel(
        "m", call("verify_identity", **MAYA), [http_error(401)], say("You are verified.")
    )
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    done, first = await turn(rt, session_id, world, "Account 10482917, PIN 4721.")
    assert done is False and first[-1].type == "turn.failed"

    done, resumed = await turn(rt, session_id, world, None)
    assert done is True
    assert {e.turn_id for e in resumed} == {first[0].turn_id}
    finished = resumed[-1]
    # One model call and the tool call before the outage, one model call after it.
    assert (finished.llm_calls, finished.tool_calls) == (2, 1)
    assert finished.input_tokens == 500 + 600
    assert finished.total_ms >= first[-1].total_ms


async def test_a_resume_with_no_unfinished_turn_is_refused(pool, retriever, world):
    rt, session_id = runtime(pool, retriever, ScriptedModel("m", say("Hello."))), uuid.uuid4().hex
    with pytest.raises(NothingToResume):
        await turn(rt, session_id, world, None)
    await turn(rt, session_id, world, "Hi")
    with pytest.raises(NothingToResume):
        await turn(rt, session_id, world, None)


async def test_a_session_runs_one_turn_at_a_time_even_from_two_processes(pool, retriever, world):
    rt, session_id = runtime(pool, retriever, ScriptedModel("m", say("Hello."))), uuid.uuid4().hex
    other = await EventLog.open(pool, session_id, world)
    await other.claim("their-turn", "text")  # what a second process does when it starts a turn
    with pytest.raises(TurnInProgress):
        await turn(rt, session_id, world, "Hi")
    await other.save(release="their-turn")
    done, _ = await turn(rt, session_id, world, "Hi")
    assert done is True


async def test_a_cancelled_turn_ends_cleanly_and_the_next_turn_keeps_counting(
    pool, retriever, world
):
    model = ScriptedModel("m", ["Hel", 5.0, "lo."], say("Hello again."))
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    first, streaming = [], asyncio.Event()

    def on_event(event):
        first.append(event)
        if event.type == "llm.delta":
            streaming.set()

    running = asyncio.create_task(
        run_turn(rt, session_id=session_id, world_id=world, text="Hi", on_event=on_event)
    )
    await asyncio.wait_for(streaming.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert first[-1].type == "turn.interrupted"
    assert (await stored_events(pool, session_id, world))[-1].type == "turn.interrupted"

    # The delta that went out had a number and was not stored. The next turn must not reuse it,
    # or a client that has seen it would drop the new turn's events as duplicates.
    done, second = await turn(rt, session_id, world, "Hi again")
    assert done is True
    assert second[0].seq == first[-1].seq + 1
    assert second[-1].type == "turn.finished"
    # The model reads a conversation with a reply in every slot, as hosted models require.
    roles = [type(m).__name__ for m in model.calls[1][0]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]


async def test_a_message_with_a_nul_byte_is_stored_without_it(pool, retriever, world):
    rt, session_id = runtime(pool, retriever, ScriptedModel("m", say("Hello."))), uuid.uuid4().hex
    done, events = await turn(rt, session_id, world, "Hi\x00 there")
    assert done is True and events[0].text == "Hi there"
    assert (await stored_events(pool, session_id, world))[0].text == "Hi there"
