"""What the voice tests share: a client that plays the browser's side of voice, and the fake
speech service as a fixture. The agent behind it runs its real graph on a scripted model."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from agent_harness import ScriptedJudge
from fake_deepgram import MS_PER_CHAR, FakeDeepgram
from pydantic import SecretStr
from test_ws import Client

REPLY = "Refunds take five days. Is there anything else I can do for you?"
QUESTION = "How long do refunds take?"
CUT_OFF = "(The customer interrupted me here and heard nothing after this.)"
QUIET = bytes(2560)  # 80 ms of microphone with nobody speaking
LOUD = b"\x40\x1f" * 1280  # 80 ms at a level only a voice close to the microphone has


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


def ms_of(text: str) -> float:
    return len(text) * MS_PER_CHAR


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

    async def mic(self, frames: int = 10, piece: bytes = QUIET) -> None:
        """Send microphone audio: frames of 80 ms, as the browser does."""
        for _ in range(frames):
            await self.ws.send(piece)

    async def played(self, ms: float, after: str = "tick") -> None:
        await self.send(type="voice.playhead", played_ms=ms, buffered_ms=0, after=after)

    async def playback(self) -> str:
        return (await self.until("voice.playback"))["action"]

    async def state(self, wanted: str) -> None:
        while (await self.until("voice.state"))["state"] != wanted:
            pass

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


class SlowJudge(ScriptedJudge):
    """A judge that gives each verdict only when the test lets it."""

    def __init__(self, *verdicts):
        super().__init__(*verdicts)
        self.asked, self.let = asyncio.Event(), asyncio.Event()

    async def check(self, **request):
        self.asked.set()
        await self.let.wait()
        return await super().check(**request)
