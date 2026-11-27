"""run_turn end to end: the graph, the event log and the stored stream, with a scripted model."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from agent_harness import GROUNDED, MAYA, ScriptedJudge, llm_for, runtime, scalar, shows
from fakes import ScriptedModel, call, http_error, say

from supportagent.eventlog import EventLog, SessionGone, TurnInProgress
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


class SlowJudge:
    """A judge that is still thinking when the customer presses Stop."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def check(self, **_):
        self.started.set()
        await asyncio.sleep(30)


async def test_a_turn_stopped_during_the_check_leaves_no_unchecked_answer_behind(
    pool, retriever, world
):
    model = ScriptedModel("m", say("Yes! Pro includes a 365-day refund window."))
    judge = SlowJudge()
    rt, session_id = runtime(pool, retriever, model, judge=judge), uuid.uuid4().hex
    first: list = []
    running = asyncio.create_task(
        run_turn(rt, session_id=session_id, world_id=world, text="Refunds?", on_event=first.append)
    )
    await asyncio.wait_for(judge.started.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    kinds = [e.type for e in first]
    # The text had streamed. Nothing ever called it the answer, and it is taken back before the
    # turn ends, so no client keeps it on screen and no reload can bring it back.
    assert "llm.delta" in kinds and "assistant.message" not in kinds
    assert kinds[-2:] == ["llm.reset", "turn.interrupted"]
    assert (first[-2].reason, first[-1].resumable) == ("cancelled", True)
    stored = await stored_events(pool, session_id, world)
    assert not shows("365-day", [e.model_dump() for e in stored])

    # Picking the turn up runs the check, and only then is there an answer.
    rt.judge = ScriptedJudge(GROUNDED)
    done, resumed = await turn(rt, session_id, world, None)
    kinds = [e.type for e in resumed]
    assert done is True and len(model.calls) == 1
    assert kinds.index("guardrail.verdict") < kinds.index("assistant.message")


async def test_a_turn_stopped_before_anything_was_saved_says_it_cannot_be_picked_up(
    pool, retriever, world, monkeypatch
):
    rt, session_id = runtime(pool, retriever, ScriptedModel("m", say("Hello."))), uuid.uuid4().hex
    events: list = []
    started = asyncio.Event()
    original = rt.graph.astream

    def stalled(*args, **kwargs):
        async def never():
            started.set()
            await asyncio.sleep(30)
            yield  # pragma: no cover

        return never()

    monkeypatch.setattr(rt.graph, "astream", stalled)
    running = asyncio.create_task(
        run_turn(rt, session_id=session_id, world_id=world, text="Hi", on_event=events.append)
    )
    await asyncio.wait_for(started.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    monkeypatch.setattr(rt.graph, "astream", original)
    assert [e.type for e in events] == ["turn.started", "turn.interrupted"]
    # A client must not offer "Resume" here: the server has nothing to resume.
    assert events[-1].resumable is False and not await can_resume(rt, session_id)


async def test_the_time_of_a_retried_turn_is_not_counted_twice(pool, retriever, world):
    down = [0.05, http_error(401)]
    model = ScriptedModel("m", down, down, down, [0.05, *say("Here I am.")])
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    began = time.perf_counter()
    totals = []
    for text in ("Hello?", None, None, None):
        _, events = await turn(rt, session_id, world, text)
        totals.append(events[-1].total_ms)
    wall_ms = (time.perf_counter() - began) * 1000
    assert events[-1].type == "turn.finished"
    assert totals == sorted(totals)
    # Four attempts of about 50 ms each. Adding up the running totals would report over a second.
    assert 200 <= totals[-1] <= wall_ms


async def test_a_finished_turn_says_what_the_customer_waited_for(pool, retriever, world):
    model = ScriptedModel("m", call("verify_identity", **MAYA), say("You are verified."))
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    _, events = await turn(rt, session_id, world, "Account 10482917, PIN 4721.")
    done = events[-1]
    assert 0 <= done.first_step_ms <= done.first_text_ms <= done.answer_ms <= done.total_ms


async def test_a_claim_is_a_lease_that_only_its_owner_can_end(pool, world):
    session_id = uuid.uuid4().hex
    old = await EventLog.open(pool, session_id, world, owner="old-instance")
    await old.claim("turn-a", "text")
    new = await EventLog.open(pool, session_id, world, owner="new-instance")
    with pytest.raises(TurnInProgress):
        await new.claim("turn-b", "text")
    # A second instance that starts during a deploy can neither clear the claim nor release it.
    await new.save(release="turn-a")
    with pytest.raises(TurnInProgress):
        await new.claim("turn-b", "text")
    assert await old.renew("turn-a")

    # The owner dies and stops renewing. Once the lease has run out the session is free.
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET turn_heartbeat_at = now() - interval '2 minutes' WHERE id = %s",
            (session_id,),
        )
    await new.claim("turn-b", "text")
    # If the old owner was only stalled, it learns here that the session is no longer its own.
    assert not await old.renew("turn-a")
    await old.save(release="turn-a")
    assert await scalar(pool, "SELECT turn_running FROM sessions WHERE id = %s", session_id)
    await new.save(release="turn-b")
    assert not await scalar(pool, "SELECT turn_running FROM sessions WHERE id = %s", session_id)


async def test_a_turn_on_a_deleted_session_says_the_session_is_gone(pool, world):
    log = await EventLog.open(pool, uuid.uuid4().hex, world)
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM sessions WHERE id = %s", (log.session_id,))
    with pytest.raises(SessionGone):
        await log.claim("turn-a", "text")


async def test_a_pin_stays_masked_after_the_session_is_picked_up_by_a_new_process(
    pool, retriever, world
):
    model = ScriptedModel(
        "m",
        call("verify_identity", **MAYA),
        say("You are verified."),
        say("Yes, it did."),
    )
    rt, session_id = runtime(pool, retriever, model), uuid.uuid4().hex
    await turn(rt, session_id, world, "Account 10482917, the number is 4721.")
    hashes = await scalar(pool, "SELECT secret_hashes FROM sessions WHERE id = %s", session_id)
    assert len(hashes) == 1 and "4721" not in hashes[0]

    # `turn` opens the log and builds the redactor afresh, as a restarted backend or a hub that
    # was let go and reopened would. The customer is verified, so nothing is masked on a guess.
    _, events = await turn(rt, session_id, world, "Did 4721 work earlier? I have 1234 cards.")
    assert events[0].text == "Did **** work earlier? I have 1234 cards."
    stored = await stored_events(pool, session_id, world)
    assert not shows("4721", [e.model_dump() for e in [*events, *stored] if e.type != "llm.delta"])
