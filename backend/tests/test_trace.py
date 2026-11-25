"""What a turn leaves behind: a trace of every model call, and no PIN anywhere it is stored."""

from __future__ import annotations

import uuid

import pytest
from agent_harness import MAYA, runtime, scalar, shows
from fakes import ScriptedModel, call, say
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from supportagent.crm import world as worlds
from supportagent.eventlog import EventLog
from supportagent.session import run_turn

# A wrong PIN that no random ID will contain by chance. A PIN is cleared whether or not it matched.
GUESS = "7391846"


@pytest.fixture(scope="module")
async def saver(pool) -> AsyncPostgresSaver:
    """The production checkpointer, which stores every step of a conversation in Postgres."""
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver


async def test_every_model_call_is_traced_and_the_pin_appears_nowhere_in_the_stream(
    pool, retriever, world
):
    model = ScriptedModel("m", call("verify_identity", **MAYA), say("You are verified."))
    rt = runtime(pool, retriever, model, capture_prompts=True)
    session_id, events = uuid.uuid4().hex, []
    await run_turn(
        rt,
        session_id=session_id,
        world_id=world,
        text="Account 10482917, the number is 4721.",
        on_event=events.append,
    )

    traces = [e for e in events if e.type == "llm.trace"]
    assert len(traces) == 2
    first, second = traces
    assert [m.role for m in first.prompt] == ["system", "user"]
    assert "verify_identity" in first.tools
    assert first.response.tool_calls[0]["name"] == "verify_identity"
    # The second call saw the tool call and its result.
    assert [m.role for m in second.prompt] == ["system", "user", "assistant", "tool"]
    assert second.prompt[3].name == "verify_identity"
    assert second.response.content == "You are verified."
    # A trace belongs to the call it describes.
    finished = [e.call_id for e in events if e.type == "llm.finished"]
    assert [t.call_id for t in traces] == finished

    # The model needs the PIN to call the tool. Nothing that is stored or shown may carry it.
    assert "4721" in model.calls[0][0][1].content
    stored = await (await EventLog.open(pool, session_id, world)).since(0)
    assert any(e.type == "llm.trace" for e in stored)
    assert not shows("4721", [e.model_dump() for e in [*events, *stored]])
    assert first.prompt[1].content == "Account 10482917, the number is ****."


async def rows_holding(pool, session_id: str, world: str, needle: str) -> dict[str, int]:
    """How many rows of each table that keeps conversation text contain `needle`."""
    like, raw = f"%{needle}%", needle.encode()
    return {
        "session_events": await scalar(
            pool,
            "SELECT count(*) FROM session_events WHERE session_id = %s AND payload::text LIKE %s",
            session_id,
            like,
        ),
        "tickets": await scalar(
            pool,
            "SELECT count(*) FROM tickets WHERE world_id = %s "
            "AND (summary LIKE %s OR transcript_excerpt LIKE %s)",
            world,
            like,
            like,
        ),
        "checkpoints": await scalar(
            pool,
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s "
            "AND (checkpoint::text LIKE %s OR metadata::text LIKE %s)",
            session_id,
            like,
            like,
        ),
        "checkpoint_blobs": await scalar(
            pool,
            "SELECT count(*) FROM checkpoint_blobs WHERE thread_id = %s "
            "AND position(%s in blob) > 0",
            session_id,
            raw,
        ),
        "checkpoint_writes": await scalar(
            pool,
            "SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s "
            "AND position(%s in blob) > 0",
            session_id,
            raw,
        ),
    }


async def test_no_pin_reaches_the_database_and_a_deleted_world_leaves_nothing(
    pool, retriever, saver, today
):
    world, session_id = f"test-{uuid.uuid4().hex[:12]}", uuid.uuid4().hex
    async with pool.connection() as conn:
        await worlds.create_world(conn, world, today)
    guess = {"account_number": "10482917", "pin": GUESS}
    model = ScriptedModel(
        "m",
        call("verify_identity", **guess),
        say("That did not match. Please check the number."),
        call("verify_identity", **guess),
        call("verify_identity", **guess),
        say("Verification is locked. A specialist will follow up."),
    )
    rt = runtime(pool, retriever, model, checkpointer=saver, capture_prompts=True)

    async def say_to_agent(text: str) -> list:
        events: list = []
        await run_turn(rt, session_id=session_id, world_id=world, text=text, on_event=events.append)
        return events

    try:
        first = await say_to_agent(f"My account is 10482917 and the number you want is {GUESS}")
        second = await say_to_agent("try 7 3 9 1 8 4 6 again please")
        # The third wrong guess locked verification and opened a ticket with a transcript.
        assert any(e.type == "escalation.created" for e in second)
        assert not shows(GUESS, [e.model_dump() for e in [*first, *second]])
        assert not shows("7 3 9 1 8 4 6", [e.model_dump() for e in second])
        for needle in (GUESS, "7 3 9 1 8 4 6"):
            assert await rows_holding(pool, session_id, world, needle) == {
                "session_events": 0,
                "tickets": 0,
                "checkpoints": 0,
                "checkpoint_blobs": 0,
                "checkpoint_writes": 0,
            }
        # One saved state per conversation: the steps before it, which held the PIN, are gone.
        saved = await scalar(
            pool, "SELECT count(*) FROM checkpoints WHERE thread_id = %s", session_id
        )
        assert saved == 1
        # The conversation still works from that one saved state.
        state = await rt.graph.aget_state({"configurable": {"thread_id": session_id}})
        assert state.values["failed_verifications"] == 3
    finally:
        async with pool.connection() as conn:
            await worlds.delete_world(conn, world)

    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        left = await scalar(pool, f"SELECT count(*) FROM {table} WHERE thread_id = %s", session_id)
        assert left == 0, table
    assert await scalar(pool, "SELECT count(*) FROM sessions WHERE id = %s", session_id) == 0


async def test_conversations_whose_session_is_gone_can_be_cleared(pool, retriever, saver, today):
    world, session_id = f"test-{uuid.uuid4().hex[:12]}", uuid.uuid4().hex
    async with pool.connection() as conn:
        await worlds.create_world(conn, world, today)
    rt = runtime(pool, retriever, ScriptedModel("m", say("Hello.")), checkpointer=saver)
    await run_turn(rt, session_id=session_id, world_id=world, text="Hi", on_event=lambda _: None)
    async with pool.connection() as conn:
        # What deleting a world did before it took its conversations along.
        await conn.execute("DELETE FROM worlds WHERE id = %s", (world,))
        assert await worlds.forget_orphaned_threads(conn) == 1
        assert await worlds.forget_orphaned_threads(conn) == 0
    left = await scalar(pool, "SELECT count(*) FROM checkpoints WHERE thread_id = %s", session_id)
    assert left == 0
