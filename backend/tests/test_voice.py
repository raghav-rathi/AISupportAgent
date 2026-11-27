"""Voice, driven over the app's real socket by a client that plays the browser, against a fake
Deepgram that plays the speech service. The agent runs its real graph on a scripted model.
"""

from __future__ import annotations

import asyncio
import json
import time

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
from fake_deepgram import MS_PER_CHAR, SILENCE, FakeDeepgram
from fakes import ScriptedModel, call, http_error, say, say_then_call, usage
from pydantic import SecretStr
from test_ws import Client
from test_ws import serve as serve  # the fixture: the app on a free port

from supportagent.agent.judge import Verdict
from supportagent.voice.session import GREETING, TURN_FAILED

REPLY = "Refunds take five days. Is there anything else I can do for you?"
QUESTION = "How long do refunds take?"
CUT_OFF = "(The customer interrupted me here and heard nothing after this.)"


@pytest.fixture
async def speech():
    async with FakeDeepgram() as fake:
        yield fake


def voice(speech: FakeDeepgram, **more) -> dict:
    """Settings for an app whose speech service is the fake."""
    return {
        "deepgram_api_key": SecretStr("test-key"),
        "deepgram_base_url": speech.url,
        "voice_greeting": False,
        **more,
    }


async def eventually(check, within: float = 3.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "did not happen in time"
        await asyncio.sleep(0.01)


class Caller(Client):
    """The browser's side of voice: it sends microphone audio, takes the reply's audio, and
    says how much of it has been played."""

    def __init__(self, ws):
        super().__init__(ws)
        self.audio_ms = 0.0  # audio received since voice started

    async def _frame(self) -> dict | None:
        frame = await asyncio.wait_for(self.ws.recv(), 5)
        if isinstance(frame, bytes):
            self.audio_ms += len(frame) / 48  # mono PCM16 at 24 kHz
            return None
        event = json.loads(frame)
        self.events.append(event)
        return event

    async def next(self):
        while True:
            event = await self._frame()
            if event is not None and event["type"] != "session.caught_up":
                return event

    async def hear(self, ms: float) -> None:
        """Take frames until this much audio has arrived in all."""
        while self.audio_ms < ms - 0.01:
            await self._frame()

    async def call(self, **options) -> dict:
        await self.send(type="voice.start", **options)
        return await self.until("voice.started")

    async def mic(self, frames: int = 10) -> None:
        """Send microphone audio: frames of 80 ms, as the browser does."""
        for _ in range(frames):
            await self.ws.send(bytes(2560))

    async def played(self, ms: float, after: str = "tick") -> None:
        await self.send(type="voice.playhead", played_ms=ms, buffered_ms=0, after=after)

    async def playback(self) -> str:
        return (await self.until("voice.playback"))["action"]

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]


async def ask(caller: Caller, ear, text: str) -> None:
    """The customer says one thing, from the first word to the end of their turn."""
    arrived = ear.audio_bytes
    await caller.mic(10)
    await eventually(lambda: ear.audio_bytes >= arrived + 10 * 2560)
    await ear.say("StartOfTurn", text.split()[0])
    await ear.say("Update", text)
    await ear.say("EndOfTurn", text)


def ms_of(text: str) -> float:
    return len(text) * MS_PER_CHAR


async def test_a_spoken_question_is_answered_aloud_and_a_typed_one_is_not(serve, speech):
    model = ScriptedModel("m", say("Refunds take five days."), say("Thirty days for annual."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    assert caller.ready["voice"] is True
    started = await caller.call()
    assert (started["input_rate"], started["output_rate"]) == (16000, 24000)
    ear = await speech.ear()
    assert speech.keys == ["Token test-key", "Token test-key"]
    assert ear.query["model"] == ["flux-general-en"] and "PIN" in ear.query["keyterm"]

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


class SlowJudge(ScriptedJudge):
    """A judge that gives its verdict only when the test lets it."""

    def __init__(self, *verdicts):
        super().__init__(*verdicts)
        self.asked, self.let = asyncio.Event(), asyncio.Event()

    async def check(self, **request):
        self.asked.set()
        await self.let.wait()
        return await super().check(**request)


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
    assert drafted["held"] is True and speech.replies == [] and caller.audio_ms == 0

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


async def test_speaking_while_the_agent_thinks_stops_that_turn_for_good(serve, speech):
    model = ScriptedModel("m", ["Let me", 5.0, " think.", usage(10, 3)], say("Cancelled."))
    _, address = await serve(model, **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, "Refund my last invoice")
    await caller.until("llm.delta")

    await ear.say("StartOfTurn", "Actually")
    stopped = await caller.until("turn.interrupted")
    assert stopped["resumable"] is False  # the customer has moved on: nothing to pick up
    await caller.send(type="turn.retry")
    assert (await caller.until("error"))["code"] == "nothing_to_retry"

    await ear.say("EndOfTurn", "Actually, forget it.")
    await caller.until("turn.finished")
    history = [m.content for m in model.calls[1][0]]
    assert history[-3:] == [
        "Refund my last invoice",
        "(The customer interrupted before hearing any of my reply.)",
        "Actually, forget it.",
    ]
    assert speech.replies == ["Cancelled."]  # nothing of the stopped turn was said
    await caller.ws.close()


async def test_the_assistants_own_voice_in_the_microphone_is_not_an_interruption(serve, speech):
    _, address = await serve(ScriptedModel("m", say(REPLY)), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.finished")
    await caller.hear(ms_of(REPLY))

    await ear.say("StartOfTurn", "Refunds take five")
    assert [await caller.playback(), await caller.playback()] == ["pause", "resume"]
    await ear.say("Update", "Refunds take five days. Is there")
    await ear.say("EndOfTurn", "Refunds take five days. Is there anything else")
    await caller.played(ms_of(REPLY), after="drained")
    spoken = await caller.until("speech.spoken")
    assert spoken["interrupted"] is False and speech.interrupts == []
    assert len(caller.of("turn.started")) == 1 and len(caller.of("stt.partial")) == 2
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
    await ear.say("StartOfTurn", "Wait")
    await ear.say("EndOfTurn", "Wait a moment")
    await caller.send(type="voice.interrupt", played_ms=600)
    cut = await caller.until("speech.spoken")
    assert (cut["interrupted"], cut["text_spoken"]) == (True, "Refunds take")
    assert caller.of("voice.playback") == [] and len(caller.of("turn.started")) == 1
    await caller.ws.close()


async def test_an_account_number_gets_time_and_the_pin_stays_masked(serve, speech):
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

    await caller.mic(25)
    await ear.say("StartOfTurn", "1 0 4 8")
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
    # 3.2 s were made. The client gets the words, 240 ms of the pause, and the 80 ms before the
    # voice goes on.
    await caller.hear(1600)
    await asyncio.sleep(0.2)
    assert caller.audio_ms == 1600

    # Stopped 1.2 s into what the client played: that is 2.8 s into what the voice made.
    await caller.send(type="voice.interrupt", played_ms=1200)
    cut = await caller.until("speech.spoken")
    assert speech.interrupts[0]["audio_played_ms"] == 2800
    assert cut["played_ms"] == 1200 and cut["text_spoken"].endswith(" fi")
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
    await caller.send(type="voice.start")
    assert (await caller.until("error"))["code"] == "voice_unavailable"
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
    assert stopped["reason"] == "error" and "text" in stopped["message"]
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


async def test_a_turn_that_fails_says_so_aloud(serve, speech):
    _, address = await serve(ScriptedModel("down", [http_error(401)]), **voice(speech))
    caller = await Caller.open(address)
    await caller.call()
    ear = await speech.ear()
    await ask(caller, ear, QUESTION)
    await caller.until("turn.failed")
    await eventually(lambda: speech.replies == [TURN_FAILED])
    await caller.ws.close()


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
    await caller.call()
    await speech.ear(2)
    await caller.mic(3)
    await asyncio.sleep(0.2)
    assert speech.replies == [GREETING]  # not a second time in the same conversation
    await caller.ws.close()
