"""Voice, driven over the app's real socket by a client that plays the browser, against a fake
Deepgram that plays the speech service. The agent runs its real graph on a scripted model.

This file has what voice says and when. test_voice_turns.py has who has the floor, and
test_voice_limits.py what voice may use and what it may store.
"""

from __future__ import annotations

import asyncio

import pytest
from agent_harness import (
    ALWAYS_ANSWER,
    ALWAYS_CHECK,
    GROUNDED,
    REFUND_QUESTION,
    UNGROUNDED,
    ScriptedJudge,
    shows,
)
from fake_deepgram import SILENCE
from fakes import ScriptedModel, call, http_error, say, say_then_call
from test_ws import serve as serve  # the fixture: the app on a free port
from voice_harness import QUESTION, REPLY, Caller, SlowJudge, ask, eventually, ms_of, voice
from voice_harness import speech as speech  # the fixture: the fake speech service

from supportagent.agent.judge import Verdict
from supportagent.voice import session as voice_session
from supportagent.voice.session import GREETING, TURN_FAILED


async def test_a_spoken_question_is_answered_aloud_and_a_typed_one_is_not(serve, speech):
    model = ScriptedModel("m", say("Refunds take five days."), say("Thirty days for annual."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    assert caller.ready["voice"] is True
    started = await caller.call(attempt="first")
    assert (started["input_rate"], started["output_rate"]) == (16000, 24000)
    assert started["attempt"] == "first"  # the client can tell which of its requests this answers
    ear = await speech.ear()
    assert speech.keys == ["Token test-key", "Token test-key"]
    assert ear.query["model"] == ["flux-general-en"] and "PIN" in ear.query["keyterm"]
    assert ear.query["eager_eot_threshold"] == ["0.3"]

    await ask(caller, ear, QUESTION)
    turn = await caller.until("turn.started")
    assert (turn["mode"], turn["text"]) == ("voice", QUESTION)
    heard = await caller.until("speech.heard")
    assert heard["turn_id"] == turn["turn_id"] and heard["trigger"] == "model"
    assert heard["listened_s"] == 0.8 and heard["cost_usd"] > 0
    assert heard["end_of_turn_ms"] is not None and heard["speech_ms"] > 0
    await caller.until("turn.finished")
    await caller.hear(ms_of("Refunds take five days."))
    assert speech.replies == ["Refunds take five days."]
    assert ear.audio_bytes == 10 * 2560  # the microphone's audio reached the recogniser

    await caller.played(caller.audio_ms, after="drained")
    spoken = await caller.until("speech.spoken")
    assert spoken["turn_id"] == turn["turn_id"] and spoken["interrupted"] is False
    assert (spoken["chars"], spoken["audio_ms"]) == (23, 1150) and spoken["cost_usd"] > 0
    assert spoken["voice_to_voice_ms"] >= spoken["synthesis_ms"] >= 0

    # Typed into the same conversation: the answer is text, and nothing is said.
    await caller.say("And for annual plans?")
    answer = await caller.until("assistant.message")
    await caller.until("turn.finished")
    assert answer["text"] == "Thirty days for annual." and speech.replies == [
        "Refunds take five days."
    ]
    assert [m.content for m in model.calls[1][0]][-3:-1] == [QUESTION, "Refunds take five days."]
    await caller.send(type="voice.stop")
    assert (await caller.until("voice.stopped"))["attempt"] == "first"
    await caller.ws.close()


async def test_a_reply_that_is_over_before_its_check_leaves_voice_listening(serve, speech):
    rt, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    rt.judge = SlowJudge(GROUNDED)
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "Hi")
    await rt.judge.asked.wait()
    # Six characters of audio are played to their end while the judge still reads.
    await caller.hear(ms_of("Hello."))
    await caller.played(ms_of("Hello."), after="drained")
    await caller.until("speech.spoken")
    rt.judge.let.set()
    await caller.until("turn.finished")
    assert caller.of("voice.state")[-1]["state"] == "thinking"  # as it was while the check ran
    assert (await caller.until("voice.state"))["state"] == "listening"
    assert speech.replies == ["Hello."]  # and the confirmed message is not said a second time
    await caller.ws.close()


async def test_text_the_model_writes_on_the_way_to_a_tool_is_never_said(serve, speech):
    model = ScriptedModel(
        "m",
        say_then_call("Let me look that up.", "search_knowledge_base", query=REFUND_QUESTION),
        say("You have **14 days** to ask [1]."),
    )
    rt, address = await serve(model, **voice(speech))
    rt.gate = ALWAYS_ANSWER
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, REFUND_QUESTION)
    await caller.until("turn.finished")
    # No citation number and no markup either: a voice cannot read them.
    assert speech.replies == ["You have 14 days to ask."]
    assert speech.warnings == [] and speech.interrupts == []
    await caller.ws.close()


async def test_an_answer_the_gate_held_is_said_only_after_its_check(serve, speech):
    model = ScriptedModel(
        "m", call("search_knowledge_base", query=REFUND_QUESTION), say("You have 14 days [1].")
    )
    rt, address = await serve(model, **voice(speech))
    rt.gate, rt.judge = ALWAYS_CHECK, SlowJudge(GROUNDED)
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, REFUND_QUESTION)
    drafted = await caller.until("answer.drafted")
    await rt.judge.asked.wait()
    # Voice has had the drafted answer for a while by the time anything is looked at: a rule
    # that spoke it too early would have sent its text by now.
    await caller.mic(3)
    await asyncio.sleep(0.3)
    assert drafted["held"] is True and caller.audio_ms == 0
    assert speech.replies == [] and not any(side.messages for side in speech.speaking)

    rt.judge.let.set()
    await caller.until("turn.finished")
    await caller.hear(ms_of("You have 14 days."))
    assert speech.replies == ["You have 14 days."]
    await caller.ws.close()


async def test_an_answer_that_fails_its_check_is_cut_off_and_corrected_aloud(serve, speech):
    model = ScriptedModel(
        "m", call("search_knowledge_base", query=REFUND_QUESTION), say("You have 365 days [1].")
    )
    rt, address = await serve(model, **voice(speech))
    rt.gate, rt.judge = ALWAYS_ANSWER, SlowJudge(UNGROUNDED)
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, REFUND_QUESTION)
    # The gate was confident, so the answer is being said while the judge still reads.
    await rt.judge.asked.wait()
    await caller.hear(ms_of("You have 365 days."))
    assert speech.replies == ["You have 365 days."]

    rt.judge.let.set()
    assert await caller.playback() == "flush"  # what the client has buffered must not be heard
    await caller.played(300, after="flush")
    message = await caller.until("assistant.message")
    await caller.until("turn.finished")
    assert speech.warnings == []
    assert speech.replies[1] == message["text"]
    assert message["text"].startswith("I need to take that back.")
    assert "on your screen" in message["text"] and "ESC-" not in message["text"]
    taken_back = caller.of("speech.spoken")[0]
    assert taken_back["interrupted"] is True and taken_back["text_spoken"] is None
    # The correction plays from where the client stood, and its end is noticed.
    await caller.hear(ms_of("You have 365 days.") + ms_of(message["text"]))
    await caller.played(300 + ms_of(message["text"]), after="drained")
    assert (await caller.until("speech.spoken"))["interrupted"] is False
    await caller.ws.close()


async def test_a_sentence_the_check_adds_is_said_after_the_answer_and_counts_as_heard(
    serve, speech
):
    answer = "The help center has nothing on that."
    model = ScriptedModel(
        "m",
        call("search_knowledge_base", query=REFUND_QUESTION),
        say(f"{answer} [1]"),
        say("Okay."),
    )
    rt, address = await serve(model, **voice(speech))
    rt.gate, rt.judge = ALWAYS_ANSWER, ScriptedJudge(Verdict(grounded=True, unresolved=True))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "Do you sell hardware?")
    await caller.until("turn.finished")
    # The answer was said before the check was done. The check found a question left open and
    # opened a ticket: the sentence about it waits until the answer has been heard.
    assert speech.replies == [answer]
    await caller.hear(ms_of(answer))
    await caller.played(ms_of(answer), after="drained")
    assert (await caller.until("speech.spoken"))["interrupted"] is False
    await eventually(lambda: len(speech.replies) == 2)
    added = speech.replies[1]
    assert added.startswith("I've passed this to a specialist") and "ESC-" not in added

    await caller.hear(ms_of(answer) + ms_of(added))
    await ear.say("StartOfTurn", "Stop")
    assert await caller.playback() == "pause"
    await caller.played(ms_of(answer) + ms_of("I've passed"), after="pause")
    cut = await caller.until("speech.spoken")
    # What was heard is the whole answer and the first two words of the added sentence.
    assert cut["text_spoken"] == f"{answer} I've passed"
    await caller.ws.close()


async def test_an_account_number_gets_time_and_no_digit_of_the_pin_is_ever_shown(serve, speech):
    model = ScriptedModel(
        "m",
        say("I can help. What are your account number and PIN?"),
        call("verify_identity", account_number="10482917", pin="4721"),
        say("Thanks, you're verified."),
    )
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "I need a refund")
    await caller.until("turn.finished")
    await caller.hear(ms_of(speech.replies[-1]))
    await caller.played(caller.audio_ms, after="drained")
    await caller.until("speech.spoken")
    # The reply asked for digits: the recogniser now waits out pauses and writes numerals.
    assert ear.configured == [
        {
            "type": "Configure",
            "thresholds": {"eot_threshold": 1.0, "eot_timeout_ms": 3000},
            "numerals": True,
        }
    ]

    # The words arrive a few at a time, and the page shows them as they do. The account
    # number may be read once it is whole. Of the PIN not one digit may, at any point.
    await caller.mic(25)
    shown = []
    for sofar in ("1 0 4", "1 0 4 8 2 9 1 7", "1 0 4 8 2 9 1 7 4", "1 0 4 8 2 9 1 7 4 7 2"):
        await ear.say("StartOfTurn" if not shown else "Update", sofar)
        shown.append((await caller.until("stt.partial"))["text"])
    assert shown == ["****", "10482917", "10482917 ****", "10482917 ****"]
    await ear.say("Update", "1 0 4 8 2 9 1 7 4 7 2 1")
    # All twelve digits are in and no more come: the app ends the turn itself.
    turn = await caller.until("turn.started")
    assert {"type": "ForceEndTurn"} in ear.messages
    assert turn["text"] == "10482917 ****"
    assert caller.of("stt.partial")[-1]["text"] == "10482917 ****"
    heard = await caller.until("speech.heard")
    assert (heard["trigger"], heard["digit_mode"]) == ("manual", True)
    await caller.until("turn.finished")
    assert model.calls[1][0][-1].content == "10482917 4721"  # the model needs the PIN as said
    assert caller.of("tool.finished")[0]["outcome"] == "ok"
    assert not shows("4721", caller.events)
    await eventually(lambda: ear.configured[-1]["numerals"] is False)  # back to usual
    await caller.ws.close()


async def test_asked_for_a_number_the_customer_may_say_something_else_first(serve, speech):
    model = ScriptedModel(
        "m", say("What are your account number and PIN?"), say("Take your time."), say("Thanks.")
    )
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "I need a refund")
    await caller.until("turn.finished")
    await eventually(lambda: bool(ear.configured))
    waits = {"type": "Configure", "thresholds": {"eot_threshold": 1.0}}
    usual = {"type": "Configure", "thresholds": {"eot_threshold": 0.7}}

    # A sentence with no digit in it is the recogniser's to end, as fast as it ends any other.
    await caller.mic(10)
    await ear.say("StartOfTurn", "Sure")
    await eventually(lambda: ear.configured[-1] == usual)
    await ear.say("EndOfTurn", "Sure, one moment.")
    await caller.until("turn.finished")
    # The reply to that asks for nothing, and the number is still to come: the recogniser
    # goes on writing numerals, and waits again from the first digit.
    assert all(change.get("numerals") is not False for change in ear.configured)
    await caller.mic(10)
    await ear.say("StartOfTurn", "1 0 4 8")
    await eventually(lambda: ear.configured[-1] == waits)
    assert (await caller.until("stt.partial"))["text"] == "****"
    await caller.ws.close()


async def test_a_pin_said_alone_or_in_two_halves_is_masked_from_its_first_digit(serve, speech):
    model = ScriptedModel("m", say("Thanks. And what is your PIN?"), say("One moment."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "My account number is 10482917")
    await caller.until("turn.finished")
    await eventually(lambda: bool(ear.configured))

    # "Four seven", a long pause, "two one": the recogniser ends the turn in the pause.
    await caller.mic(10)
    await ear.say("StartOfTurn", "4")
    assert (await caller.until("stt.partial"))["text"] == "****"
    await ear.say("Update", "4 7")
    assert (await caller.until("stt.partial"))["text"] == "****"
    await ear.say("EndOfTurn", "4 7.", trigger="timeout")
    first = await caller.until("turn.started")
    await caller.until("turn.finished")
    await caller.mic(10)
    await ear.say("StartOfTurn", "2 1")
    assert (await caller.until("stt.partial"))["text"] == "****"
    await ear.say("EndOfTurn", "2 1.", trigger="timeout")
    second = await caller.until("turn.started")
    assert (first["text"], second["text"]) == ("****.", "****.")
    assert not shows("47", caller.events) and not shows("21", caller.events)
    await caller.ws.close()


async def test_a_long_silence_in_the_voice_is_cut_short_and_what_was_heard_still_comes_out_right(
    serve, speech
):
    # The voice says "Refunds take ", is silent for two seconds, and goes on with " five days."
    reply = f"Refunds take {SILENCE * 40} five days."
    _, address = await serve(ScriptedModel("m", say(reply)), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    # 3.2 s were made. The client gets the words, 240 ms of the pause, and 40 ms of quiet in
    # front of the voice where it goes on, of which 10 ms are in the piece that has its sound.
    await caller.hear(1550)

    # Stopped 1.2 s into what the client played: that is 2.85 s into what the voice made.
    await caller.send(type="voice.interrupt", played_ms=1200)
    cut = await caller.until("speech.spoken")
    assert speech.interrupts[0]["audio_played_ms"] == 2850
    assert cut["played_ms"] == 1200 and cut["text_spoken"].endswith(" fiv")
    assert cut["audio_ms"] == 1550  # all that was sent of it, by the server's own count
    await caller.ws.close()


async def test_a_reply_starts_with_its_first_sound_whatever_silence_the_voice_begins_with(
    serve, speech
):
    # Half a second of silence, then the words. Measured live, a second and more.
    reply = f"{SILENCE * 10}Refunds take five days."
    _, address = await serve(ScriptedModel("m", say(reply)), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    # The client gets 40 ms of quiet and then the voice: 1.65 s were made, 1.19 s are sent.
    await caller.hear(1190)
    await caller.played(1190, after="drained")
    spoken = await caller.until("speech.spoken")
    assert (spoken["audio_ms"], spoken["interrupted"]) == (1190, False)
    await caller.ws.close()


async def test_when_the_voice_cannot_say_what_was_heard_it_is_reckoned_from_the_time_played(
    serve, speech
):
    _, address = await serve(ScriptedModel("m", say(REPLY)), **voice(speech))
    speech.refuse_interrupts = True
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    await caller.send(type="voice.interrupt", played_ms=600)
    cut = await caller.until("speech.spoken")
    # 600 ms of 3.2 s is twelve characters of 64, cut back to a whole word.
    assert speech.warnings == ["INVALID_INTERRUPT_OFFSET"] and speech.interrupts == []
    assert (cut["interrupted"], cut["played_ms"], cut["text_spoken"]) == (True, 600, "Refunds")
    await caller.ws.close()


async def test_a_voice_connection_that_drops_takes_its_reply_and_not_the_conversation(
    serve, speech
):
    model = ScriptedModel("m", say(REPLY), say("Still here."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    await speech.speaking[0].ws.close()
    assert await caller.playback() == "flush"
    assert (await caller.until("speech.spoken"))["interrupted"] is True
    await caller.played(300, after="flush")

    await ask(caller, ear, "Are you still there?")
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY) + ms_of("Still here."))
    assert speech.replies == [REPLY, "Still here."] and caller.of("voice.stopped") == []
    await caller.ws.close()


async def test_a_voice_that_takes_the_text_and_stays_silent_does_not_hold_the_line(
    serve, speech, monkeypatch
):
    monkeypatch.setattr(voice_session, "_NO_AUDIO_S", 0.3)
    model = ScriptedModel("m", say(REPLY), say("Still here."))
    _, address = await serve(model, **voice(speech))
    speech.mute = True
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    # No audio comes, and the connection does not say that anything is wrong. The reply is
    # given up, voice stays on, and the next one is heard.
    lost = await caller.until("speech.spoken")
    assert (lost["interrupted"], lost["audio_ms"]) == (True, 0)
    await caller.state("listening")
    speech.mute = False
    await caller.played(0, after="flush")
    await ask(caller, ear, "Are you still there?")
    await caller.until("turn.finished")
    await caller.hear(ms_of("Still here."))
    assert caller.of("voice.stopped") == []
    await caller.ws.close()


@pytest.mark.parametrize(
    ("limit", "reason"),
    [({"voice_max_seconds": 0.3}, "limit"), ({"voice_idle_seconds": 0.3}, "idle")],
)
async def test_voice_ends_at_its_limits_and_the_time_listened_is_stored(
    serve, speech, limit, reason
):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech, **limit))
    caller = await Caller.open(address)
    await caller.call()
    await caller.mic(10)
    stopped = await caller.until("voice.stopped")
    assert stopped["reason"] == reason and stopped["message"]
    used = caller.of("voice.usage")[0]
    assert used["listened_s"] == 0.8 and used["cost_usd"] > 0
    # The conversation goes on in text.
    await caller.say("Hi")
    assert (await caller.until("assistant.message"))["text"] == "Hello."
    await caller.ws.close()


async def test_when_speech_is_down_the_conversation_goes_on_in_text(serve, speech):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    caller = await Caller.open(address)
    speech.down = True
    await caller.send(type="voice.start", attempt="a1")
    refused = await caller.until("error")
    assert (refused["code"], refused["client_id"]) == ("voice_unavailable", "a1")
    await caller.say("Hi")
    assert (await caller.until("assistant.message"))["text"] == "Hello."
    await caller.until("turn.finished")

    # A socket that drops once is opened again, and the customer notices nothing.
    speech.down = False
    await caller.call()
    ear = await speech.ear()
    await ear.ws.close()
    ear = await speech.ear(2)
    await ask(caller, ear, "Still there?")
    await caller.until("turn.finished")
    # A second drop soon after means the service is down.
    await ear.ws.close()
    stopped = await caller.until("voice.stopped")
    assert stopped["reason"] == "error" and "unavailable" in stopped["message"]
    await caller.ws.close()


async def test_a_simulated_speech_failure_turns_voice_off_and_fails_the_next_start(serve, speech):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    await caller.send(type="fault.inject", kind="speech_down", times=2)
    assert (await caller.until("voice.stopped"))["reason"] == "error"
    await caller.send(type="voice.start")
    assert (await caller.until("error"))["code"] == "voice_unavailable"
    await caller.call()  # the simulated failures are used up
    await caller.ws.close()


async def test_a_turn_that_fails_says_so_aloud_and_stops_waiting_for_digits(serve, speech):
    model = ScriptedModel("m", say("What are your account number and PIN?"), [http_error(401)])
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "I need a refund")
    await caller.until("turn.finished")
    await eventually(lambda: bool(ear.configured) and ear.configured[-1]["numerals"] is True)
    await caller.hear(ms_of(speech.replies[-1]))
    await caller.played(caller.audio_ms, after="drained")

    await ask(caller, ear, QUESTION)
    await caller.until("turn.failed")
    await eventually(lambda: speech.replies[-1] == TURN_FAILED)
    # Whatever the turn before had asked for, nobody is waiting for a number now.
    await eventually(lambda: ear.configured[-1]["numerals"] is False)
    await caller.ws.close()


async def test_voice_says_hello_once_and_the_greeting_is_no_turn(serve, speech):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), **voice(speech, voice_greeting=True)
    )
    caller = await Caller.open(address)
    await caller.call()
    await speech.ear()
    await caller.hear(ms_of(GREETING))
    await caller.played(caller.audio_ms, after="drained")
    spoken = await caller.until("speech.spoken")
    assert speech.replies == [GREETING] and spoken["turn_id"] is None and spoken["cost_usd"] > 0
    assert caller.of("turn.started") == [] and caller.of("assistant.message") == []
    await caller.send(type="voice.stop")
    await caller.until("voice.stopped")
    session_id = caller.ready["session_id"]
    await caller.ws.close()

    # Coming back to the conversation, on a connection of its own and after the server has let
    # go of everything it held in memory for it: the greeting is paid for per character, and a
    # client that reconnects in a loop must not get one each time.
    for _ in range(2):
        again = await Caller.open(address, session_id=session_id)
        await again.call()
        ear = await speech.ear(len(speech.listening) + 0)
        await again.mic(3)
        await eventually(lambda ear=ear: ear.audio_bytes > 0)
        await again.send(type="voice.stop")
        await again.until("voice.stopped")
        await again.ws.close()
    assert speech.replies == [GREETING]
