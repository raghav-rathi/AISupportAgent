"""What voice may use, and what it may store: the limits per visitor, the budget for speech,
and the masking of what the customer says before and after the app knows their PIN.
"""

from __future__ import annotations

from agent_harness import DANIEL, MAYA, shows
from fakes import ScriptedModel, call, say, usage
from test_ws import serve as serve  # the fixture: the app on a free port
from voice_harness import Caller, ask, eventually, voice
from voice_harness import speech as speech  # the fixture: the fake speech service

from supportagent.api.limits import Budget


async def test_voice_has_a_daily_budget_of_its_own(serve, speech, pool, world):
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (id, world_id) VALUES (%s, %s)", (f"spent-{world}", world)
        )
        await conn.execute(
            "INSERT INTO session_events (session_id, seq, type, payload) "
            "VALUES (%s, 1, 'speech.spoken', '{\"cost_usd\": 1000}'::jsonb)",
            (f"spent-{world}",),
        )
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), **voice(speech, daily_speech_budget_usd=999.0)
    )
    caller = await Caller.open(address)
    await caller.send(type="voice.start")
    refused = await caller.until("error")
    assert refused["code"] == "voice_unavailable" and "budget" in refused["message"]
    # Typed questions are paid for elsewhere and go on.
    await caller.say("Hi")
    assert (await caller.until("assistant.message"))["text"] == "Hello."
    await caller.ws.close()


async def test_a_conversation_has_one_voice_and_the_server_a_limit_of_them(serve, speech):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), **voice(speech, max_voice_sessions=1)
    )
    first = await Caller.open(address)
    await first.call()
    # The same conversation in a second window takes the voice over.
    second = await Caller.open(address, session_id=first.ready["session_id"])
    await second.call()
    assert (await first.until("voice.stopped"))["reason"] == "replaced"
    # Another conversation finds the one line busy.
    other = await Caller.open(address)
    await other.send(type="voice.start")
    assert (await other.until("error"))["code"] == "voice_unavailable"
    await second.send(type="voice.stop")
    assert (await second.until("voice.stopped"))["reason"] == "client"
    await other.call()
    for client in (first, second, other):
        await client.ws.close()


async def test_one_visitor_gets_some_of_the_voice_lines_and_not_all(serve, speech):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")),
        **voice(speech, max_voice_sessions=4, max_voice_sessions_per_address=2),
    )
    callers = [await Caller.open(address) for _ in range(3)]
    await callers[0].call()
    await callers[1].call()
    await callers[2].send(type="voice.start", attempt="third")
    refused = await callers[2].until("error")
    assert refused["code"] == "voice_unavailable" and refused["client_id"] == "third"
    assert "other windows" in refused["message"]
    # A line that is given back can be taken again.
    await callers[0].send(type="voice.stop")
    await callers[0].until("voice.stopped")
    await callers[2].call()
    for caller in callers:
        await caller.ws.close()


async def test_turning_voice_on_and_off_in_a_loop_is_stopped(serve, speech):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), **voice(speech, max_voice_starts_per_minute=3)
    )
    caller = await Caller.open(address)
    for _ in range(3):
        await caller.call()
        await caller.send(type="voice.stop")
        await caller.until("voice.stopped")
    connections = len(speech.listening) + len(speech.speaking)
    await caller.send(type="voice.start")
    refused = await caller.until("error")
    assert refused["code"] == "voice_unavailable" and "too often" in refused["message"]
    assert len(speech.listening) + len(speech.speaking) == connections  # and costs nothing
    await caller.ws.close()


async def test_a_visitor_has_so_much_voice_in_a_day(serve, speech):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")),
        **voice(speech, voice_seconds_per_address_per_day=1.0),
    )
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await caller.mic(15)  # 1.2 s of microphone
    await ear.say("StartOfTurn", "Hi")
    await ear.say("EndOfTurn", "Hi")
    # The turn that counted the time goes on as text. Voice ends, and does not come back today.
    stopped = await caller.until("voice.stopped")
    assert stopped["reason"] == "limit" and "today" in stopped["message"]
    await caller.until("turn.finished")
    await caller.send(type="voice.start")
    refused = await caller.until("error")
    assert refused["code"] == "voice_unavailable" and "voice time" in refused["message"]
    await caller.ws.close()


async def test_what_is_not_a_microphone_is_not_passed_on(serve, speech):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    # Two bytes at a time would be a message to the recogniser for every sample. An odd number
    # of bytes is no audio at all.
    for piece in (b"\x00\x00", b"", b"\x00" * 2561):
        for _ in range(10):
            await caller.ws.send(piece)
    await caller.mic(2)
    await eventually(lambda: ear.audio_bytes == 2 * 2560)
    # A client that keeps at it is no browser, and loses the line.
    for _ in range(40):
        await caller.ws.send(b"\x00\x00")
    assert (await caller.until("voice.stopped"))["reason"] == "error"
    await caller.ws.close()


async def test_a_pin_the_first_pass_did_not_recognise_is_masked_in_what_is_stored(
    serve, speech, pool
):
    """A verified customer names another account and its PIN in passing, with no word that says
    what the second number is. When they said it, nothing marked it as a PIN. The model then
    uses it as one, and from there on it is one. The turn's stored events get their last pass
    at the turn's end, and voice storing an event of its own in the meantime must not have
    written the customer's message before that."""
    said = "Check my wife's account 2 0 5 7 3 8 4 6 and then 9 0 3 4"
    model = ScriptedModel(
        "m",
        call("verify_identity", account_number=MAYA["account_number"], pin=MAYA["pin"]),
        say("You are verified."),
        call("verify_identity", account_number=DANIEL["account_number"], pin=DANIEL["pin"]),
        say("That account is verified now."),
    )
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.say(f"My account number is {MAYA['account_number']}, my PIN is {MAYA['pin']}")
    await caller.until("turn.finished")
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, said)
    turn = await caller.until("turn.started")
    assert DANIEL["pin"] in turn["text"]  # nothing said it was a PIN when it was said
    await caller.until("speech.heard")  # voice has stored an event of its own by now
    await caller.until("turn.finished")
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT payload FROM session_events WHERE session_id = %s AND turn_id = %s",
            (caller.ready["session_id"], turn["turn_id"]),
        )
        stored = [row["payload"] for row in await cur.fetchall()]
    assert any(row["type"] == "turn.started" for row in stored)
    assert not shows(DANIEL["pin"], stored)
    await caller.ws.close()


async def test_a_call_started_early_for_nothing_and_a_call_cut_short_both_count_in_the_budget(
    serve, speech, pool
):
    model = ScriptedModel("m", say("Never used."), ["Let me", 5.0, " think.", usage(10, 3)])
    before = await Budget(pool, None).spent()
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await caller.mic(10)
    await ear.say("StartOfTurn", "And")
    await ear.say("EagerEndOfTurn", "And")
    await eventually(lambda: len(model.calls) == 1)
    await ear.say("TurnResumed", "And")
    await ear.say("EndOfTurn", "And what about refunds?")
    await caller.until("llm.delta")
    await caller.send(type="turn.cancel")
    await caller.until("turn.interrupted")
    await eventually(lambda: len(caller.of("llm.finished")) == 2)
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT payload FROM session_events WHERE session_id = %s AND type = 'llm.finished'",
            (caller.ready["session_id"],),
        )
        stored = [row["payload"] for row in await cur.fetchall()]
    # Neither has a usage report with a price in it here. Both are in the store, where the
    # daily budget counts a call without a price at a fixed amount.
    assert sorted((row["unused"], row["cancelled"]) for row in stored) == [
        (False, True),
        (True, False),
    ]
    assert await Budget(pool, None).spent() - before >= 0.004 - 1e-9
    await caller.ws.close()
