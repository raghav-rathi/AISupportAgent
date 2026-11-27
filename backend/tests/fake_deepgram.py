"""A stand-in for Deepgram's two streaming sockets, on a local port.

It does what the live service was seen to do in the cases the voice session depends on, and
nothing more. `/v2/listen` takes audio and says only what a test tells it to say. `/v2/speak`
turns text into audio at a fixed number of milliseconds per character, so a test can tell which
characters a customer has heard after so many milliseconds of playback. Every character is a
tone, except `^`, which is silence: a test writes the voice's pauses into the text.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

MS_PER_CHAR = 50
SILENCE = "^"
_BYTES_PER_MS = 48  # mono PCM16 at 24 kHz
_FRAME_BYTES = 3840  # 80 ms, the size the live service sends
_TONE = b"\x00\x10" * (MS_PER_CHAR * _BYTES_PER_MS // 2)  # one character, well above silence
_HUSH = bytes(MS_PER_CHAR * _BYTES_PER_MS)


class Listening:
    """One `/v2/listen` connection. A test makes it speak as the recogniser with `say`."""

    def __init__(self, ws: ServerConnection, query: dict[str, list[str]]) -> None:
        self.ws, self.query = ws, query
        self.audio_bytes = 0
        self.messages: list[dict] = []  # what the app sent, apart from audio
        self.index = 0
        self.transcript = ""
        self.in_turn = False

    @property
    def configured(self) -> list[dict]:
        return [m for m in self.messages if m["type"] == "Configure"]

    async def run(self) -> None:
        await self.ws.send(json.dumps({"type": "Connected"}))
        async for raw in self.ws:
            if isinstance(raw, bytes):
                self.audio_bytes += len(raw)
                continue
            message = json.loads(raw)
            self.messages.append(message)
            if message["type"] == "Configure":
                await self.ws.send(json.dumps({"type": "ConfigureSuccess"}))
            elif message["type"] == "ForceEndTurn" and self.in_turn:
                await self.say("EndOfTurn", self.transcript, trigger="manual")
            elif message["type"] == "CloseStream":
                break

    async def say(
        self,
        event: str,
        transcript: str,
        *,
        confidence: float = 0.3,
        trigger: str | None = None,
        silent_s: float = 0.4,
    ) -> None:
        """Send one TurnInfo. The words end `silent_s` before the audio received so far."""
        now_s = self.audio_bytes / 32_000
        words = transcript.split()
        end = max(0.0, now_s - silent_s)
        start = max(0.0, end - 0.3 * len(words))
        step = (end - start) / len(words) if words else 0.0
        info = {
            "type": "TurnInfo",
            "event": event,
            "turn_index": self.index,
            "audio_window_start": start,
            "audio_window_end": now_s,
            "transcript": transcript,
            "words": [
                {
                    "word": word,
                    "confidence": 0.99,
                    "start": start + step * i,
                    "end": start + step * (i + 1),
                }
                for i, word in enumerate(words)
            ],
            "end_of_turn_confidence": confidence,
        }
        self.transcript, self.in_turn = transcript, True
        if event == "EndOfTurn":
            info["trigger"] = trigger or "model"
            info["end_of_turn_confidence"] = max(confidence, 0.8)
            self.in_turn = False
        await self.ws.send(json.dumps(info))
        if event == "EndOfTurn":
            self.index += 1


class Speaking:
    """One `/v2/speak` connection."""

    def __init__(self, ws: ServerConnection, query: dict[str, list[str]], fake: FakeDeepgram):
        self.ws, self.query, self.fake, self.pace = ws, query, fake, fake.pace
        self.messages: list[dict] = []  # what the app sent
        self.warnings: list[str] = []  # what this side refused
        self.replies: list[str] = []  # the text of every reply, in order, as far as it got
        self.interrupts: list[dict] = []  # what each accepted interrupt answered
        self.start_ms = 0.0  # where the reply in progress begins on this socket's clock
        self.made_ms = 0.0  # audio made for it so far
        self.owed = bytearray()  # audio for text that has arrived and is not sent yet
        self.cut_ms = 0.0  # where the last interrupt put the clock
        self.active = False  # a reply can be interrupted
        self.flushed = False  # its text is complete
        self.told = False  # its metadata has gone out
        self._wake = asyncio.Event()

    @property
    def text(self) -> str:
        """The reply in progress, or the last one."""
        return self.replies[-1] if self.replies else ""

    async def run(self) -> None:
        await self.ws.send(json.dumps({"type": "Connected"}))
        if self.fake.over_quota:
            await self._send({"type": "Error", "code": "QUOTA_EXCEEDED", "description": "no"})
            return
        maker = asyncio.create_task(self._make())
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                self.messages.append(message)
                kind = message["type"]
                if kind == "Speak":
                    if self.active and self.flushed:
                        await self._finish()  # the reply before this one
                        self.start_ms += self.made_ms
                        self.active = False
                    if not self.active:
                        self.made_ms = 0.0
                        self.owed.clear()
                        self.active, self.flushed, self.told = True, False, False
                        self.replies.append("")
                        await self._send({"type": "SpeechStarted", "speech_id": "s"})
                    self.replies[-1] += message["text"]
                    if self.fake.mute:
                        continue  # the text is taken and no audio ever comes
                    for char in message["text"]:
                        self.owed += _HUSH if char == SILENCE else _TONE
                    self._wake.set()
                elif kind == "Flush":
                    if self.active and not self.flushed and not self.fake.mute:
                        self.flushed = True
                        self._wake.set()
                elif kind == "Interrupt":
                    await self._interrupt(message.get("playback_offset"))
                elif kind == "Close":
                    break
        finally:
            maker.cancel()
            with suppress(asyncio.CancelledError):
                await maker

    async def _send(self, message: dict) -> None:
        with suppress(ConnectionClosed):
            await self.ws.send(json.dumps(message))

    def _metadata(self) -> dict:
        return {
            "audio_duration_ms": self.made_ms,
            "input_character_count": len(self.text),
            "billable_character_count": len(self.text),
        }

    async def _frame(self, *, paced: bool) -> None:
        frame = bytes(self.owed[:_FRAME_BYTES])
        del self.owed[:_FRAME_BYTES]
        ms = len(frame) / _BYTES_PER_MS
        self.made_ms += ms
        with suppress(ConnectionClosed):
            await self.ws.send(frame)
        if paced and self.pace:
            await asyncio.sleep(ms / 1000 / self.pace)

    async def _finish(self) -> None:
        """Make what is left of a reply whose text is complete, and say that it is done."""
        while self.owed:
            await self._frame(paced=False)
        if not self.told:
            self.told = True
            await self._send({"type": "Flushed"})
            await self._send({"type": "SpeechMetadata", **self._metadata()})

    async def _make(self) -> None:
        """Turn owed audio into frames. With a pace, no faster than that many times real time."""
        while True:
            if self.owed:
                await self._frame(paced=True)
            elif self.active and self.flushed and not self.told:
                await self._finish()
            else:
                self._wake.clear()
                await self._wake.wait()

    async def _warn(self, code: str) -> None:
        self.warnings.append(code)
        await self._send({"type": "Warning", "code": code, "description": code})

    async def _interrupt(self, offset: dict | None) -> None:
        if not self.active:
            await self._warn("NO_AUDIO_GENERATED")
            return
        if self.fake.refuse_interrupts:
            await self._warn("INVALID_INTERRUPT_OFFSET")
            return
        made_until = self.start_ms + self.made_ms
        spoken = remaining = None
        played = made_until
        if offset is not None:
            played = float(offset["value"])
            if not self.cut_ms < played <= made_until or played < self.start_ms:
                await self._warn("INVALID_INTERRUPT_OFFSET")
                return
            chars = int((played - self.start_ms) / MS_PER_CHAR)
            spoken, remaining = self.text[:chars], self.text[chars:]
        metadata = self._metadata()
        self.owed.clear()
        self.active, self.cut_ms = False, played
        self.start_ms, self.made_ms = played, 0.0
        answer = {"type": "SpeechInterrupted", "audio_played_ms": played, "metadata": metadata}
        if spoken is not None:
            answer |= {"text_spoken": spoken, "text_remaining": remaining}
        self.interrupts.append(answer)
        await self._send(answer)


class FakeDeepgram:
    def __init__(self) -> None:
        self.listening: list[Listening] = []
        self.speaking: list[Speaking] = []
        self.down = False  # refuse new connections
        self.pace: float | None = None  # see Speaking._make
        self.mute = False  # take text and make no audio, like a voice that has hung
        self.refuse_interrupts = False  # answer every interrupt with a warning
        self.over_quota = False  # let a voice connection open, then say no and close it
        self.keys: list[str | None] = []  # the Authorization header of each connection
        self._server = None
        self._arrived = asyncio.Condition()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def __aenter__(self) -> FakeDeepgram:
        self._server = await serve(
            self._handle, "127.0.0.1", 0, process_request=self._gate, ping_interval=None
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    def _gate(self, connection, request):
        if self.down:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "down\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        parts = urlsplit(ws.request.path)
        query = parse_qs(parts.query)
        self.keys.append(ws.request.headers.get("Authorization"))
        if parts.path == "/v2/listen":
            side = Listening(ws, query)
            self.listening.append(side)
        else:
            side = Speaking(ws, query, self)
            self.speaking.append(side)
        async with self._arrived:
            self._arrived.notify_all()
        with suppress(ConnectionClosed):
            await side.run()

    @property
    def replies(self) -> list[str]:
        """The text of every reply the app has had spoken, in order. The app speaks each on a
        connection of its own, and opens them in the order it uses them."""
        return [reply for side in self.speaking for reply in side.replies]

    @property
    def interrupts(self) -> list[dict]:
        return [answer for side in self.speaking for answer in side.interrupts]

    @property
    def warnings(self) -> list[str]:
        return [code for side in self.speaking for code in side.warnings]

    async def ear(self, number: int = 1) -> Listening:
        """The recogniser's socket, once the app has opened this many of them."""
        async with self._arrived:
            await asyncio.wait_for(
                self._arrived.wait_for(lambda: len(self.listening) >= number and self.speaking), 5
            )
        return self.listening[-1]
