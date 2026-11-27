"""Who has the floor: the customer speaking over a reply, over an answer that is still being
checked, over nothing at all, and the assistant's own voice coming back through the microphone.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from agent_harness import GROUNDED, REFUND_QUESTION, UNGROUNDED
from fakes import ScriptedModel, say, usage
from test_ws import serve as serve  # the fixture: the app on a free port
from voice_harness import (
    CUT_OFF,
    LOUD,
    QUESTION,
    REPLY,
    Caller,
    SlowJudge,
    ask,
    eventually,
    ms_of,
    voice,
)
from voice_harness import speech as speech  # the fixture: the fake speech service

from supportagent.agent.judge import JudgeError
from supportagent.voice.session import TOOK_BACK_FAILED

ANSWER = "You have 14 days to ask for your money back."


async def test_speaking_over_a_reply_keeps_only_what_was_heard(serve, speech):
    second = "Annual plans have thirty days."
    model = ScriptedModel("m", say(REPLY), say(second), say("Goodbye."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    first = await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    await caller.played(400)

    await caller.mic(6)
    await ear.say("StartOfTurn", "Wait")
    assert await caller.playback() == "pause"
    await caller.played(600, after="pause")  # where the player stopped
    assert await caller.playback() == "flush"
    cut = await caller.until("speech.spoken")
    assert (cut["interrupted"], cut["played_ms"], cut["text_spoken"]) == (True, 600, "Refunds take")
    assert cut["turn_id"] == first["turn_id"]
    assert speech.interrupts[0]["audio_played_ms"] == 600
    await caller.played(600, after="flush")

    await ear.say("EndOfTurn", "Wait, what about annual plans?")
    await caller.until("turn.finished")
    # The model reads the first reply as far as the customer heard it, and no further.
    history = [m.content for m in model.calls[1][0]]
    assert history[-2:] == [f"Refunds take {CUT_OFF}", "Wait, what about annual plans?"]
    assert not any("anything else" in text for text in history[1:]), history[1:]

    # Both clocks moved to where the first reply was cut. A second interruption half a second
    # into the second reply is half a second of its text.
    await caller.hear(ms_of(REPLY) + ms_of(second))
    await caller.mic(6)
    await ear.say("StartOfTurn", "No")
    assert await caller.playback() == "pause"
    await caller.played(600 + 500, after="pause")
    again = await caller.until("speech.spoken")
    assert (again["played_ms"], again["text_spoken"]) == (500, "Annual pla")
    assert speech.warnings == []
    await caller.ws.close()


async def test_speaking_while_the_agent_thinks_stops_that_turn(serve, speech):
    model = ScriptedModel("m", ["Let me", 5.0, " think.", usage(10, 3)], say("Cancelled."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "Refund my last invoice")
    await caller.until("llm.delta")

    await ear.say("StartOfTurn", "Actually")
    await caller.until("turn.interrupted")
    # The model call that was running is stored as one of unknown cost, for the budget.
    running = [e for e in caller.of("llm.finished") if e["cancelled"]]
    assert len(running) == 1 and running[0]["cost_usd"] is None

    await ear.say("EndOfTurn", "Actually, forget it.")
    await caller.until("turn.finished")
    history = [m.content for m in model.calls[1][0]]
    assert history[-3:] == [
        "Refund my last invoice",
        "(The previous request was interrupted before I could reply.)",
        "Actually, forget it.",
    ]
    assert speech.replies == ["Cancelled."]  # nothing of the stopped turn was said
    await caller.ws.close()


async def test_a_turn_without_a_word_stops_neither_the_thinking_nor_the_reply(serve, speech):
    model = ScriptedModel("m", ["Refunds take", 0.4, " five days.", usage(10, 3)])
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("llm.delta")
    # A cough while the agent thinks: the recogniser opens a turn and finds no word in it.
    await ear.say("StartOfTurn", "")
    await ear.say("EndOfTurn", "")
    await caller.until("turn.finished")
    await caller.hear(ms_of("Refunds take five days."))
    # And another while the reply plays.
    await ear.say("StartOfTurn", "")
    await ear.say("Update", "")
    await ear.say("EndOfTurn", "")
    await caller.played(caller.audio_ms, after="drained")
    spoken = await caller.until("speech.spoken")
    assert spoken["interrupted"] is False and spoken["audio_ms"] == 1150
    assert caller.of("turn.interrupted") == [] and caller.of("voice.playback") == []
    assert len(caller.of("turn.started")) == 1
    await caller.ws.close()


async def test_the_assistants_own_voice_in_the_microphone_is_not_an_interruption(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY)), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))

    # The recogniser's first report has one word, as the live one's does. One word of the reply
    # says nothing yet: playback waits for more.
    await ear.say("StartOfTurn", "Refunds")
    assert await caller.playback() == "pause"
    await ear.say("Update", "Refunds take five")
    assert await caller.playback() == "resume"
    # The player moves on, and says so. Words it has not got to are not the reply's echo.
    await caller.played(1800)
    await ear.say("Update", "Refunds take five days. Is there")
    await ear.say("EndOfTurn", "Refunds take five days. Is there anything else")
    await caller.played(ms_of(REPLY), after="drained")
    spoken = await caller.until("speech.spoken")
    assert spoken["interrupted"] is False and speech.interrupts == []
    assert len(caller.of("turn.started")) == 1 and len(caller.of("stt.partial")) == 2
    await caller.ws.close()


async def test_a_word_of_the_reply_and_nothing_after_it_gives_the_customer_the_floor(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY), say("Okay.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    # "Refunds" is in the reply, and it may as well be the customer. No more words come to
    # tell, so after a moment the reply stops for them.
    await ear.say("StartOfTurn", "Refunds")
    assert await caller.playback() == "pause"
    await caller.played(600, after="pause")
    assert await caller.playback() == "flush"
    assert (await caller.until("speech.spoken"))["text_spoken"] == "Refunds take"
    await caller.ws.close()


async def test_what_the_customer_says_after_an_echo_in_the_same_breath_is_heard(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY), say("Goodbye.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))

    await ear.say("StartOfTurn", "Refunds take five")
    assert [await caller.playback(), await caller.playback()] == ["pause", "resume"]
    # The microphone goes on hearing the reply, and then the customer answers over it. The
    # recogniser hands both over as one sentence.
    await ear.say("Update", "Refunds take five days. No thanks")
    assert await caller.playback() == "pause"
    await caller.played(1300, after="pause")
    assert await caller.playback() == "flush"
    await ear.say("EndOfTurn", "Refunds take five days. No thanks, that is all.")
    turn = await caller.until("turn.started")
    assert turn["text"] == "No thanks, that is all."
    await caller.until("turn.finished")
    await caller.ws.close()


async def test_without_echo_cancellation_only_the_button_interrupts(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY)), **voice(speech))
    caller = await Caller.open(address)
    await caller.call(echo_cancellation=False)
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))

    # The microphone hears the reply and maybe the customer: nobody can tell, so it is not taken.
    await caller.mic(8)
    await caller.mic(3, LOUD)
    await ear.say("StartOfTurn", "Wait")
    await ear.say("EndOfTurn", "Wait a moment")
    await caller.send(type="voice.interrupt", played_ms=600)
    cut = await caller.until("speech.spoken")
    assert (cut["interrupted"], cut["text_spoken"]) == (True, "Refunds take")
    # The one thing the client was told about its player is the answer to its own Stop.
    assert [e["action"] for e in caller.of("voice.playback")] == ["flush"]
    assert len(caller.of("turn.started")) == 1
    await caller.ws.close()


async def test_stop_is_always_answered_so_the_client_knows_when_to_play_again(serve, speech):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    await speech.ear()
    # Nothing is playing. The client has dropped its audio all the same, and waits for this.
    await caller.send(type="voice.interrupt", played_ms=0)
    assert await caller.playback() == "flush"
    await caller.ws.close()


async def test_the_microphone_getting_loud_pauses_the_reply_before_the_recogniser_has_a_word(
    serve, speech
):
    _, address = await serve(ScriptedModel("m", say(REPLY), say("Okay.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    await caller.played(300)

    # Half a second of a quiet room while the reply plays, then a door slams: loud, and the
    # recogniser hears no speech in it. Playback pauses at once and goes on a moment later.
    await caller.mic(8)
    await caller.mic(3, LOUD)
    assert await caller.playback() == "pause"
    await caller.played(500, after="pause")
    assert await caller.playback() == "resume"
    assert caller.of("speech.spoken") == []

    # Then the customer: playback pauses on their voice, and the recogniser's word confirms it.
    await caller.mic(3, LOUD)
    assert await caller.playback() == "pause"
    await caller.played(900, after="pause")
    await ear.say("StartOfTurn", "Wait")
    assert await caller.playback() == "flush"
    cut = await caller.until("speech.spoken")
    assert (cut["played_ms"], cut["text_spoken"]) == (900, "Refunds take five")
    await caller.ws.close()


async def test_a_reply_that_ends_while_its_player_is_paused_does_not_leave_it_paused(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY), say("Okay.")), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    # The customer starts on a word the reply also has, just as its last sound leaves the
    # loudspeaker. The pause reaches a player that has nothing left.
    await ear.say("StartOfTurn", "Refunds")
    assert await caller.playback() == "pause"
    await caller.played(ms_of(REPLY), after="drained")
    assert await caller.playback() == "resume"  # or the next reply would never be heard
    assert (await caller.until("speech.spoken"))["interrupted"] is False
    await ear.say("EndOfTurn", "Refunds? Okay.")
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY) + ms_of("Okay."))
    await caller.ws.close()


async def answering(serve, speech, judge, *after: list):
    """A voice call in which the first answer uses no tool, so it is said while it is checked.
    Returns the caller, the recogniser's side, the runtime and the model."""
    model = ScriptedModel("m", say(ANSWER), *after)
    rt, address = await serve(model, **voice(speech))
    rt.judge = judge
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, REFUND_QUESTION)
    await rt.judge.asked.wait()
    await caller.hear(ms_of(ANSWER))
    return caller, ear, rt, model


async def test_speaking_over_an_answer_that_is_being_checked_does_not_get_it_past_the_check(
    serve, speech
):
    caller, ear, rt, model = await answering(serve, speech, SlowJudge(GROUNDED), say("Thirty."))
    first = caller.of("turn.started")[0]["turn_id"]
    await caller.played(300)
    await ear.say("StartOfTurn", "And")
    assert await caller.playback() == "pause"
    await caller.played(500, after="pause")
    assert await caller.playback() == "flush"
    cut = await caller.until("speech.spoken")
    assert cut["text_spoken"] == "You have 1"
    await caller.played(500, after="flush")
    # The voice has stopped. The check has not: what they say next waits for its verdict.
    await ear.say("EndOfTurn", "And for annual plans?")
    await caller.state("thinking")
    assert caller.of("turn.interrupted") == [] and len(caller.of("turn.started")) == 1

    rt.judge.let.set()
    done = await caller.until("turn.finished")
    assert done["turn_id"] == first and done["outcome"] == "answered"
    second = await caller.until("turn.started")
    assert second["text"] == "And for annual plans?"
    await caller.until("turn.finished")
    assert len(rt.judge.calls) == 2  # both answers were checked
    # The first answer passed, and stays in the conversation as far as it was heard.
    history = [m.content for m in model.calls[1][0]]
    assert history[-2:] == [f"You have 1 {CUT_OFF}", "And for annual plans?"]
    assert speech.replies == [ANSWER, "Thirty."]
    await caller.ws.close()


async def test_an_answer_that_fails_its_check_after_the_customer_spoke_over_it_is_still_taken_back(
    serve, speech
):
    caller, ear, rt, model = await answering(
        serve, speech, SlowJudge(UNGROUNDED, GROUNDED), say("Thirty.")
    )
    await caller.played(300)
    await ear.say("StartOfTurn", "And")
    assert await caller.playback() == "pause"
    await caller.played(500, after="pause")
    assert await caller.playback() == "flush"
    await caller.until("speech.spoken")
    await caller.played(500, after="flush")
    await ear.say("EndOfTurn", "And for annual plans?")
    await caller.state("thinking")

    rt.judge.let.set()
    correction = (await caller.until("assistant.message"))["text"]
    assert correction.startswith("I need to take that back.")
    await caller.until("turn.finished")
    await caller.until("turn.finished")
    # They heard the start of an answer that did not stand. Before anything else they hear
    # that, and the answer to what they said next waits until they have.
    await eventually(lambda: len(speech.replies) == 2)
    assert speech.replies[1] == correction
    await caller.hear(ms_of(ANSWER) + ms_of(correction))
    await asyncio.sleep(0.1)
    assert len(speech.replies) == 2
    await caller.played(500 + ms_of(correction), after="drained")
    await eventually(lambda: len(speech.replies) == 3)
    assert speech.replies[2] == "Thirty."
    # And the model reads the correction, not the words that were heard of the wrong answer.
    history = [m.content for m in model.calls[1][0]]
    assert history[-2:] == [correction, "And for annual plans?"]
    assert not any("14 days" in str(text) for text in history[-3:])
    await caller.ws.close()


async def test_an_answer_heard_to_its_end_before_its_check_is_kept_when_the_customer_speaks(
    serve, speech
):
    caller, ear, rt, model = await answering(serve, speech, SlowJudge(GROUNDED), say("Thirty."))
    await caller.played(ms_of(ANSWER), after="drained")
    assert (await caller.until("speech.spoken"))["interrupted"] is False
    # All of it has been heard and the judge still reads. The customer goes on.
    await ear.say("StartOfTurn", "Okay")
    await ear.say("EndOfTurn", "Okay, and for annual plans?")
    await caller.state("thinking")
    assert caller.of("turn.interrupted") == []
    rt.judge.let.set()
    await caller.until("turn.finished")
    await caller.until("turn.finished")
    history = [m.content for m in model.calls[1][0]]
    assert history[-2:] == [ANSWER, "Okay, and for annual plans?"]
    await caller.ws.close()


async def test_speech_does_not_stop_a_typed_message_and_waits_its_turn(serve, speech):
    model = ScriptedModel("m", ["Typed", 0.5, " answer.", usage(10, 3)], say("Spoken answer."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await caller.say("A typed question")
    await caller.until("llm.delta")
    # Someone says something while the typed message is being answered.
    await ask(caller, ear, "And a spoken one")
    typed = await caller.until("turn.finished")
    assert typed["outcome"] == "answered" and caller.of("turn.interrupted") == []
    spoken = await caller.until("turn.started")
    assert (spoken["mode"], spoken["text"]) == ("voice", "And a spoken one")
    await caller.until("turn.finished")
    assert speech.replies == ["Spoken answer."]
    await caller.ws.close()


async def test_stop_ends_a_typed_message_the_way_its_own_stop_would(serve, speech):
    model = ScriptedModel("m", ["Typed", 5.0, " answer.", usage(10, 3)], say("Again."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    await speech.ear()
    await caller.say("A typed question")
    await caller.until("llm.delta")
    await caller.send(type="voice.interrupt", played_ms=0)
    stopped = await caller.until("turn.interrupted")
    assert stopped["resumable"] is True  # not spoken over: stopped, and it can be picked up
    await caller.ws.close()


async def test_the_model_starts_reading_before_the_customer_has_finished(serve, speech):
    model = ScriptedModel(
        "m", say("Refunds take five days."), say("Never used."), say("Thirty days.")
    )
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()

    await caller.mic(10)
    await ear.say("StartOfTurn", "How")
    await ear.say("Update", QUESTION)
    await ear.say("EagerEndOfTurn", QUESTION)
    # The recogniser thinks they may be done. The model is asked now, and nothing else happens.
    await eventually(lambda: len(model.calls) == 1)
    assert caller.of("turn.started") == [] and speech.replies == []
    await ear.say("EndOfTurn", QUESTION)
    await caller.until("turn.finished")
    # They were done: the turn takes that call, and makes none of its own.
    assert len(model.calls) == 1 and speech.replies == ["Refunds take five days."]
    used = caller.of("llm.finished")[0]
    assert used["early_ms"] is not None and used["unused"] is False and used["turn_id"]
    await caller.played(ms_of(speech.replies[0]), after="drained")
    await caller.until("speech.spoken")

    await caller.mic(10)
    await ear.say("StartOfTurn", "And")
    await ear.say("Update", "And for")
    await ear.say("EagerEndOfTurn", "And for")
    await eventually(lambda: len(model.calls) == 2)
    # They were not done. That call is for nothing, and what it cost is on the books.
    await ear.say("TurnResumed", "And for")
    await ear.say("Update", "And for annual plans?")
    await ear.say("EndOfTurn", "And for annual plans?")
    await caller.until("turn.finished")
    assert len(model.calls) == 3 and speech.replies[-1] == "Thirty days."
    assert [m.content for m in model.calls[2][0]][-1] == "And for annual plans?"
    await eventually(lambda: any(e["unused"] for e in caller.of("llm.finished")))
    wasted = next(e for e in caller.of("llm.finished") if e["unused"])
    assert wasted["turn_id"] is None and wasted["seq"] > 0 and wasted["cost_usd"] is None
    await caller.ws.close()


async def test_a_call_that_is_still_running_when_the_turn_starts_goes_on_inside_the_turn(
    serve, speech
):
    slow = ["Refunds", 0.4, " take five", 0.2, " days.", usage(600, 30)]
    model = ScriptedModel("m", slow)
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await caller.mic(10)
    await ear.say("StartOfTurn", "How")
    await ear.say("EagerEndOfTurn", QUESTION)
    await eventually(lambda: len(model.calls) == 1)
    # The model has written a word. The turn starts and takes the call as it is: the words
    # written so far at once, the rest as they come, and the usage at the end.
    await ear.say("EndOfTurn", QUESTION)
    done = await caller.until("turn.finished")
    assert done["outcome"] == "answered" and done["input_tokens"] == 600
    assert len(model.calls) == 1 and speech.replies == ["Refunds take five days."]
    assert "".join(e["text"] for e in caller.of("llm.delta")) == "Refunds take five days."
    await caller.ws.close()


async def test_two_windows_that_ask_for_the_voice_at_once_get_one_between_them(serve, speech):
    _, address = await serve(ScriptedModel("m", say("Hello.")), **voice(speech))
    first = await Caller.open(address)
    second = await Caller.open(address, session_id=first.ready["session_id"])
    await first.send(type="voice.start", attempt="a")
    await second.send(type="voice.start", attempt="b")

    async def settle(client: Caller) -> bool:
        """Read what the window is told until nothing more comes. True if it has the voice."""
        with suppress(TimeoutError):
            while True:
                async with asyncio.timeout(0.5):
                    await client.next()
        told = [e["type"] for e in client.events if e["type"] in ("voice.started", "voice.stopped")]
        return told[-1:] == ["voice.started"]

    # Which of them the server hears first is chance. That one of them ends up with the voice
    # and the other is told it lost it, or never had it, is not.
    has_it = [await settle(first), await settle(second)]
    assert sorted(has_it) == [False, True]
    winner = first if has_it[0] else second
    ear = next(side for side in speech.listening if side.ws.close_code is None)
    await ask(winner, ear, "Hi")
    await winner.until("turn.finished")
    await winner.hear(ms_of("Hello."))
    await asyncio.sleep(0.2)
    assert speech.replies == ["Hello."]  # one voice says it, once
    for client in (first, second):
        await client.ws.close()


async def test_a_report_from_before_the_latest_flush_is_not_taken_for_the_answer_to_it(
    serve, speech
):
    model = ScriptedModel("m", say(REPLY), say("Annual plans have thirty days."), say("Okay."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    await caller.send(type="voice.interrupt", played_ms=600, epoch=0)
    first = await caller.until("voice.playback")
    assert (first["action"], first["id"]) == ("flush", 1)
    await caller.until("speech.spoken")

    # The client's word on that flush is still on its way when the next reply is cut as well.
    await ask(caller, ear, "And for annual plans?")
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY) + 400)
    await caller.mic(6)
    await ear.say("StartOfTurn", "Wait")
    pause = await caller.until("voice.playback")
    assert (pause["action"], pause["id"]) == ("pause", 1)
    second = await caller.until("voice.playback")
    assert (second["action"], second["id"]) == ("flush", 2)
    # Now the first report arrives. It says where the player stood two replies ago.
    await caller.send(type="voice.playhead", played_ms=600, after="flush", epoch=1)
    await caller.send(type="voice.playhead", played_ms=900, after="flush", epoch=2)
    await ear.say("EndOfTurn", "Wait, never mind.")
    await caller.until("turn.finished")

    # The third reply is measured from the second report: 100 ms of it were played.
    await caller.hear(ms_of(REPLY) + 400 + ms_of("Okay."))
    await caller.send(type="voice.interrupt", played_ms=1000, epoch=2)
    cut = [e for e in caller.of("speech.spoken") if e["interrupted"]]
    while len(cut) < 3:
        cut.append(await caller.until("speech.spoken"))
    assert (cut[-1]["played_ms"], cut[-1]["text_spoken"]) == (100, "Ok")
    await caller.ws.close()


async def test_a_stop_pressed_on_a_reply_that_has_been_cut_since_stops_nothing_else(serve, speech):
    model = ScriptedModel("m", say(REPLY), say("Annual plans have thirty days."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    # The customer speaks over the reply, which cuts it, and says something new.
    await caller.mic(6)
    await ear.say("StartOfTurn", "Wait")
    assert await caller.playback() == "pause"
    await caller.played(600, after="pause")
    assert await caller.playback() == "flush"
    await caller.send(type="voice.playhead", played_ms=600, after="flush", epoch=1)
    await ear.say("EndOfTurn", "Wait, and for annual plans?")
    await caller.until("turn.finished")
    second = "Annual plans have thirty days."
    await caller.hear(ms_of(REPLY) + ms_of(second))
    # Their Stop was pressed on the first reply, before its flush had reached them. It
    # arrives now, while the second reply plays, and must not cut that one.
    await caller.send(type="voice.interrupt", played_ms=600, epoch=0)
    await caller.played(600 + ms_of(second), after="drained")
    spoken = await caller.until("speech.spoken")
    while spoken["interrupted"]:
        spoken = await caller.until("speech.spoken")
    assert spoken["audio_ms"] == ms_of(second)
    assert [e["action"] for e in caller.of("voice.playback")] == ["pause", "flush"]
    await caller.ws.close()


async def test_without_echo_cancellation_what_is_said_after_stop_in_the_same_breath_is_heard(
    serve, speech
):
    model = ScriptedModel("m", say(REPLY), say("Done."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call(echo_cancellation=False)
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))
    # The microphone hears the reply, and the recogniser opens a turn for it. Nobody can tell
    # it from the customer, so it is not taken.
    await caller.mic(6)
    await ear.say("StartOfTurn", "Refunds")
    await ear.say("Update", "Refunds take five days.")
    await asyncio.sleep(0.1)  # nothing tells a client that a turn is being passed over
    # The customer presses Stop and speaks at once. The recogniser is still in that turn.
    await caller.send(type="voice.interrupt", played_ms=1300, epoch=0)
    assert await caller.playback() == "flush"
    await caller.send(type="voice.playhead", played_ms=1300, after="flush", epoch=1)
    await ear.say("Update", "Refunds take five days. Actually, cancel my plan")
    await ear.say("EndOfTurn", "Refunds take five days. Actually, cancel my plan.")
    turn = await caller.until("turn.started")
    # Their words, without the reply's own in front of them.
    assert turn["text"] == "Actually, cancel my plan."
    await caller.until("turn.finished")
    await caller.ws.close()


class NoVerdict(SlowJudge):
    """A judge that, once let, fails to give a verdict."""

    async def check(self, **request):
        self.asked.set()
        await self.let.wait()
        raise JudgeError("no verdict")


async def test_a_turn_that_fails_after_some_of_its_answer_was_heard_says_so_and_remembers_it(
    serve, speech
):
    model = ScriptedModel("m", say(ANSWER), say("Let me look again."))
    rt, address = await serve(model, **voice(speech))
    rt.judge = NoVerdict()
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, REFUND_QUESTION)
    await rt.judge.asked.wait()
    await caller.hear(ms_of(ANSWER))
    await caller.played(500)

    rt.judge.let.set()
    assert await caller.playback() == "flush"
    await caller.send(type="voice.playhead", played_ms=500, after="flush", epoch=1)
    failed = await caller.until("turn.failed")
    assert failed["retryable"] is True
    # Not the usual apology: they heard words that nobody could check, and are told so.
    await eventually(lambda: len(speech.replies) == 2)
    assert speech.replies[1] == TOOK_BACK_FAILED
    await caller.hear(ms_of(ANSWER) + ms_of(TOOK_BACK_FAILED))
    await caller.played(500 + ms_of(TOOK_BACK_FAILED), after="drained")

    rt.judge = SlowJudge(GROUNDED)
    rt.judge.let.set()
    await ask(caller, ear, "So how long do I have?")
    await caller.until("turn.finished")
    # The answer that was never checked is gone from the conversation. What they heard of it
    # is there, as words that were never verified.
    history = [str(m.content) for m in model.calls[1][0]]
    assert history[-1] == "So how long do I have?"
    assert 'The customer heard this much of it: "You have 1"' in history[-2]
    assert "never verified" in history[-2] and ANSWER not in history
    await caller.ws.close()
