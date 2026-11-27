"""A scripted caller: WAV files as the microphone, over the app's own WebSocket.

It does what the browser does. It sends 80 ms frames of microphone audio in real time, silence
between the things it says. It takes the reply's audio and plays it into nothing, at the speed
of sound: a playhead moves through what has arrived, stops when told to pause, and drops what is
buffered when told to flush. It reports that playhead as the browser's player does. So the server
cannot tell it from a browser, and what it measures is what a customer would wait.

Run with a script, a JSON list of steps:

    uv run python -m supportagent.cli.voice_client ws://127.0.0.1:8791/ws script.json

    [{"say": "wav/hello.wav"},                       wait until nobody speaks, then say it
     {"say": "wav/long.wav", "hear_out": false},    say it and do not wait for the reply to end
     {"say": "wav/wait.wav", "over_ms": 1500},       say it 1.5 s into the reply that is playing
     {"type": "What is my refund limit?"},           send a typed message
     {"say": "wav/long.wav", "stop_at_ms": 1200},   say it, and press Stop 1.2 s into the reply
     {"wait_s": 2}]

WAV files are mono PCM16 at 16 kHz. Paths are relative to the script. Times are in milliseconds
on this client's clock: `speech end` is the moment the last sample above silence was sent.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import sys
import time
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect

_FRAME_S = 0.08
_IN_RATE = 16_000
_FRAME_BYTES = int(_IN_RATE * _FRAME_S) * 2
_SILENCE = bytes(_FRAME_BYTES)
_LOUD = 500  # a sample above this is speech, in a file made by a synthesiser


def read_wav(path: Path) -> tuple[bytes, float, float]:
    """The samples of a WAV file, and where its speech starts and ends, in seconds."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, _IN_RATE):
            raise SystemExit(f"{path}: need mono PCM16 at {_IN_RATE} Hz")
        pcm = wav.readframes(wav.getnframes())
    samples = array.array("h", pcm)
    loud = [i for i, sample in enumerate(samples) if abs(sample) > _LOUD]
    if not loud:
        raise SystemExit(f"{path}: no speech in it")
    return pcm, loud[0] / _IN_RATE, (loud[-1] + 1) / _IN_RATE


def save_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


@dataclass
class Player:
    """The browser's player, without a loudspeaker: a playhead that moves at the speed of sound."""

    rate: int = 24_000
    played: int = 0  # samples
    paused: bool = False
    moved_at: float = field(default_factory=time.monotonic)
    heard: bytearray = field(default_factory=bytearray)  # the audio that was played
    pending: bytearray = field(default_factory=bytearray)  # the audio that is buffered

    def advance(self) -> None:
        """Play what the time since the last call allows. Whole samples, so nothing drifts."""
        now = time.monotonic()
        buffered = len(self.pending) // 2
        count = 0 if self.paused else min(int((now - self.moved_at) * self.rate), buffered)
        self.heard += self.pending[: count * 2]
        del self.pending[: count * 2]
        self.played += count
        # Time that played nothing is gone. Time that played part of a sample is kept.
        playing = not self.paused and count < buffered
        self.moved_at = self.moved_at + count / self.rate if playing else now

    def take(self, audio: bytes) -> None:
        self.advance()
        self.pending += audio

    def flush(self) -> None:
        self.advance()
        self.pending.clear()
        self.paused = False

    @property
    def played_ms(self) -> float:
        return self.played / self.rate * 1000

    @property
    def buffered_ms(self) -> float:
        return len(self.pending) / 2 / self.rate * 1000


class Caller:
    def __init__(self, ws, *, verbose: bool) -> None:
        self.ws, self.verbose = ws, verbose
        self.player = Player()
        self.events: list[dict] = []
        self.session_id = ""
        self.began = time.monotonic()
        self.speech: bytes = b""  # what is left of the utterance being said
        self.first_audio_at: float | None = None  # of the reply in progress
        self.paused_at: float | None = None
        self.stop_at_ms: float | None = None  # press Stop this far into the next reply
        self.reply_from_ms = 0.0  # the playhead at the start of the reply in progress
        self.reply_from_byte = 0  # the same place in the audio that was played
        self.cuts: Path | None = None  # where to write what was played of a reply cut off
        self.arrived = asyncio.Condition()
        self.was_playing = False
        self.closed = False

    def now(self) -> float:
        return time.monotonic()

    async def send(self, **message: Any) -> None:
        await self.ws.send(json.dumps(message))

    async def report(self, after: str = "tick") -> None:
        self.player.advance()
        await self.send(
            type="voice.playhead",
            played_ms=round(self.player.played_ms, 1),
            buffered_ms=round(self.player.buffered_ms, 1),
            after=after,
        )

    async def microphone(self) -> None:
        """Send a frame every 80 ms, and with every other one say where the playhead is."""
        due, tick = time.monotonic(), 0
        while not self.closed:
            frame, self.speech = self.speech[:_FRAME_BYTES], self.speech[_FRAME_BYTES:]
            await self.ws.send(frame.ljust(_FRAME_BYTES, b"\0") if frame else _SILENCE)
            tick += 1
            self.player.advance()
            playing = self.player.buffered_ms > 0
            if self.was_playing and not playing:
                await self.report("drained")
            elif playing and tick % 2 == 0:
                await self.report()
            self.was_playing = playing
            press = self.stop_at_ms
            if press is not None and self.first_audio_at is not None:
                if self.player.played_ms - self.reply_from_ms >= press:
                    self.stop_at_ms = None
                    self.player.flush()
                    await self.send(
                        type="voice.interrupt", played_ms=round(self.player.played_ms, 1)
                    )
                    self.note({"type": "client.stop", "played_ms": self.player.played_ms})
            due += _FRAME_S
            await asyncio.sleep(max(0.0, due - time.monotonic()))

    def note(self, event: dict) -> None:
        event["at"] = self.now()
        self.events.append(event)
        if self.verbose and event["type"] not in ("llm.delta", "voice.state"):
            brief = {k: v for k, v in event.items() if k not in ("seq", "ts", "at", "hits")}
            print(f"  {event['at'] - self.began:7.2f}  {json.dumps(brief)[:200]}", file=sys.stderr)

    async def receive(self) -> None:
        try:
            async for frame in self.ws:
                if isinstance(frame, bytes):
                    if self.first_audio_at is None:
                        self.first_audio_at = self.now()
                        self.player.advance()
                        self.reply_from_ms = self.player.played_ms
                        self.reply_from_byte = len(self.player.heard)
                        self.note({"type": "client.first_audio"})
                    self.player.take(frame)
                else:
                    event = json.loads(frame)
                    kind = event["type"]
                    if kind == "voice.playback":
                        action = event["action"]
                        if action == "pause":
                            self.player.advance()
                            self.player.paused, self.paused_at = True, self.now()
                        elif action == "resume":
                            self.player.advance()
                            self.player.paused = False
                        else:
                            self.player.flush()
                        await self.report(action)
                    elif kind == "session.ready":
                        self.session_id = event["session_id"]
                    self.note(event)
                async with self.arrived:
                    self.arrived.notify_all()
        finally:
            self.closed = True
            async with self.arrived:
                self.arrived.notify_all()

    async def until(self, check, within: float = 60.0):
        async with self.arrived:
            await asyncio.wait_for(self.arrived.wait_for(lambda: check() or self.closed), within)
        return check()

    def since(self, mark: int, kind: str) -> list[dict]:
        return [e for e in self.events[mark:] if e["type"] == kind]

    async def quiet(self) -> None:
        """Wait until no turn is running and no reply is playing."""

        while not self.closed:
            self.player.advance()
            left_s = self.player.buffered_ms / 1000
            states = [e["state"] for e in self.events if e["type"] == "voice.state"]
            if left_s <= 0 and (not states or states[-1] == "listening"):
                return
            # Until the buffer has run out, or something arrives that may change the picture.
            async with self.arrived:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.arrived.wait(), max(left_s, 0.05))

    async def say(
        self,
        path: Path,
        over_ms: float | None = None,
        hear_out: bool = True,
        stop_at_ms: float | None = None,
    ) -> dict:
        """Say one clip and report what came of it.

        `over_ms` says it that long into the reply that is playing. `hear_out` false returns as
        soon as the turn has started, for a next step that speaks over the reply. `stop_at_ms`
        presses Stop that long into the reply.
        """
        pcm, onset_s, end_s = read_wav(path)
        mark = len(self.events)
        if over_ms is None:
            await self.quiet()
            await asyncio.sleep(0.6)
            mark = len(self.events)
            self.first_audio_at = None
            self.stop_at_ms = stop_at_ms
        else:
            await self.until(lambda: self.first_audio_at is not None)
            await asyncio.sleep(max(0.0, self.first_audio_at + over_ms / 1000 - self.now()))
        self.paused_at = None
        sent_at = self.now()
        self.speech = pcm
        onset_at, end_at = sent_at + onset_s, sent_at + end_s
        result: dict[str, Any] = {"say": path.name, "speech_ms": round((end_s - onset_s) * 1000)}
        if over_ms is not None:
            cut = await self.until(lambda: self.since(mark, "speech.spoken"), 20)
            if self.paused_at is not None:
                result["pause_after_onset_ms"] = round((self.paused_at - onset_at) * 1000)
            if cut:
                result |= self.keep_cut(mark)
                result |= {
                    "cut_played_ms": cut[0]["played_ms"],
                    "cut_text_spoken": cut[0]["text_spoken"],
                    "cut_after_onset_ms": round((cut[0]["at"] - onset_at) * 1000),
                }
            mark, self.first_audio_at = len(self.events), None
        started = await self.until(lambda: self.since(mark, "turn.started"), 30)
        if not started:
            return result | {"error": "no turn started"}
        turn_id = started[0]["turn_id"]
        result["heard"] = started[0]["text"]
        result["turn_after_speech_ms"] = round((started[0]["at"] - end_at) * 1000)
        if not hear_out:
            return result  # the next step speaks over the reply to this one

        def over() -> bool:
            done = [e for e in self.since(mark, "speech.spoken") if e.get("turn_id") == turn_id]
            failed = self.since(mark, "turn.failed") or self.since(mark, "turn.interrupted")
            return bool(done or failed)

        await self.until(over, 90)
        events = [e for e in self.events[mark:] if e.get("turn_id") == turn_id]
        heard = next((e for e in events if e["type"] == "speech.heard"), {})
        spoken = next((e for e in events if e["type"] == "speech.spoken"), {})
        message = next((e for e in events if e["type"] == "assistant.message"), {})
        finished = next((e for e in events if e["type"] == "turn.finished"), {})
        first = self.since(mark, "client.first_audio")
        result |= {
            "reply": message.get("text"),
            "tools": sum(e["type"] == "tool.started" for e in events),
            "outcome": finished.get("outcome"),
            "end_of_turn_ms": heard.get("end_of_turn_ms"),
            "trigger": heard.get("trigger"),
            "digit_mode": heard.get("digit_mode"),
            "server_voice_to_voice_ms": spoken.get("voice_to_voice_ms"),
            "synthesis_ms": spoken.get("synthesis_ms"),
            "interrupted": spoken.get("interrupted"),
            "text_spoken": spoken.get("text_spoken"),
        }
        if first:
            result["first_audio_after_speech_ms"] = round((first[0]["at"] - end_at) * 1000)
            result["first_audio_after_turn_ms"] = round((first[0]["at"] - started[0]["at"]) * 1000)
        if spoken.get("interrupted"):
            result |= self.keep_cut(mark)
        return result

    def keep_cut(self, mark: int) -> dict:
        """Write what was played of the reply that was just cut off, to compare with the record."""
        if self.cuts is None:
            return {}
        self.player.advance()
        name = f"cut_{self.session_id[:6]}_{mark}.wav"
        played = bytes(self.player.heard[self.reply_from_byte :])
        save_wav(self.cuts / name, played, self.player.rate)
        return {"cut_wav": name}

    async def type(self, text: str) -> dict:
        await self.quiet()
        mark = len(self.events)
        await self.send(type="user.message", text=text, mode="text")
        await self.until(
            lambda: self.since(mark, "turn.finished") or self.since(mark, "turn.failed"), 90
        )
        message = self.since(mark, "assistant.message")
        return {
            "type": text,
            "reply": message[0]["text"] if message else None,
            "tools": [event["name"] for event in self.since(mark, "tool.started")],
        }


async def run(args: argparse.Namespace, steps: list[dict], folder: Path) -> list[dict]:
    headers = {"Origin": args.origin} if args.origin else {}
    async with connect(args.url, additional_headers=headers, max_size=2**22) as ws:
        caller = Caller(ws, verbose=args.verbose)
        caller.cuts = Path(args.cuts) if args.cuts else None
        await caller.send(type="hello", session_id=args.session, last_seq=0)
        tasks = [asyncio.create_task(caller.receive())]
        await caller.until(lambda: caller.session_id)
        await caller.send(type="voice.start", echo_cancellation=not args.no_echo_cancellation)
        started = await caller.until(
            lambda: [e for e in caller.events if e["type"] in ("voice.started", "error")], 15
        )
        if not started or started[0]["type"] == "error":
            raise SystemExit(f"voice did not start: {started}")
        caller.player.rate = started[0]["output_rate"]
        tasks.append(asyncio.create_task(caller.microphone()))
        await asyncio.sleep(1.0)  # a greeting, if the server has one, has begun by now
        results = []
        for step in steps:
            if "say" in step:
                results.append(
                    await caller.say(
                        folder / step["say"],
                        step.get("over_ms"),
                        step.get("hear_out", True),
                        step.get("stop_at_ms"),
                    )
                )
            elif "type" in step:
                results.append(await caller.type(step["type"]))
            elif "wait_s" in step:
                await asyncio.sleep(step["wait_s"])
            if caller.closed:
                break
        await caller.quiet()
        await caller.send(type="voice.stop")
        await caller.until(lambda: [e for e in caller.events if e["type"] == "voice.stopped"], 5)
        if args.save:
            save_wav(Path(args.save), bytes(caller.player.heard), caller.player.rate)
        costs = [
            e.get("cost_usd") or 0
            for e in caller.events
            if e["type"] in ("speech.heard", "speech.spoken", "voice.usage", "turn.finished")
        ]
        results.append({"session_id": caller.session_id, "cost_usd": round(sum(costs), 5)})
        caller.closed = True
        for task in tasks:
            task.cancel()
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", help="the backend's WebSocket, for example ws://127.0.0.1:8791/ws")
    parser.add_argument("script", help="a JSON file with the steps")
    parser.add_argument("--origin", default="http://localhost:3210")
    parser.add_argument("--session", default=None, help="go on with this conversation")
    parser.add_argument("--save", default=None, help="write the audio that was played to this WAV")
    parser.add_argument(
        "--cuts", default=None, help="a folder for what was played of each cut reply"
    )
    parser.add_argument("--no-echo-cancellation", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print every event to stderr")
    args = parser.parse_args()
    script = Path(args.script)
    for result in asyncio.run(run(args, json.loads(script.read_text()), script.parent)):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
