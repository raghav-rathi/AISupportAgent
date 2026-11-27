"""A scripted caller: WAV files as the microphone, over the app's own WebSocket.

It does what the browser's side of voice does, as far as a program without a sound card can. It
sends microphone audio in 80 ms pieces, each when its last sample has been "recorded", with
silence between the things it says. It takes the reply's audio and plays it into nothing, at the
speed of sound: a playhead moves through what has arrived, stops when told to pause, and drops
what is buffered when told to flush. It reports that playhead as the browser's player does, and
after pressing Stop it plays nothing until the server has answered.

What it measures is on its own clock, and three things about it are worth knowing:

- The customer "stops speaking" at the last sample of the clip that is louder than its noise.
- A reply's "first sound" is the moment the player reaches the first sample that is not silence.
  The first piece of audio arrives a little earlier and begins with up to 40 ms of quiet.
- It has no loudspeaker and no microphone. A browser adds the delay of its audio output to every
  figure, 10 to 40 ms on built-in loudspeakers and 150 to 300 ms on Bluetooth, and nothing here
  can show an echo.

Run with a script, a JSON list of steps:

    uv run python -m supportagent.cli.voice_client ws://127.0.0.1:8791/ws script.json

    [{"say": "wav/hello.wav"},                       wait until nobody speaks, then say it
     {"say": "wav/long.wav", "hear_out": false},    say it and do not wait for the reply to end
     {"say": "wav/wait.wav", "over_ms": 1500},       say it 1.5 s into the reply that is playing
     {"type": "What is my refund limit?"},           send a typed message
     {"say": "wav/long.wav", "stop_at_ms": 1200},   say it, and press Stop 1.2 s into the reply
     {"wait_s": 2}]

It prints one line of JSON per step. To turn the lines of many calls into the figures the README
reports:

    uv run python -m supportagent.cli.voice_client --summary calls/*.jsonl

WAV files are mono PCM16 at 16 kHz. Paths are relative to the script.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import statistics
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
_AUDIBLE = 200  # and one above this, in a reply, is sound: the voice's silence stays under 100
_P95_NEEDS = 20  # fewer values than this have no 95th percentile worth the name


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


def first_sound(audio: bytes) -> int | None:
    """The index of the first sample of a piece of audio that is not silence, if it has one."""
    samples = memoryview(audio[: len(audio) & ~1]).cast("h")
    return next((i for i, sample in enumerate(samples) if abs(sample) > _AUDIBLE), None)


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

    def take(self, audio: bytes) -> float:
        """Buffer a piece of audio. Returns when its first sample will be played, if nothing
        stops the player before."""
        self.advance()
        due = self.moved_at + len(self.pending) / 2 / self.rate
        self.pending += audio
        return due

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
        self.live_from = 0  # where in `events` the replay of an older conversation ends
        self.session_id = ""
        self.began = time.monotonic()
        self.speech: bytes = b""  # what is left of the utterance being said
        self.speech_at: float | None = None  # when its first sample was "recorded"
        self.taken_up = asyncio.Event()  # the microphone has begun on it
        self.first_audio_at: float | None = None  # of the reply in progress: its first piece
        self.first_sound_at: float | None = None  # and its first sample that is not silence
        self.paused_at: float | None = None
        self.stop_at_ms: float | None = None  # press Stop this far into the next reply
        self.turn_mark = 0  # where in `events` the turn that was started last begins
        self.one_moment = False  # a tool runs, and what is playing is "one moment"
        self.muted = False  # Stop was pressed, and the server has not answered yet
        self.epoch = 0  # the number of the last flush acted on, see events.VoicePlayback
        self.stopped_in = 0  # the epoch Stop was pressed in
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
            epoch=self.epoch,
        )

    async def microphone(self) -> None:
        """Send 80 ms of audio every 80 ms, as a microphone that has just recorded it.

        The browser's recorder hands a piece over when its last sample is in. So does this: the
        piece that holds the first 80 ms of a clip goes out 80 ms after the clip "began".
        """
        due = time.monotonic()
        while not self.closed:
            began, due = due, due + _FRAME_S
            frame, self.speech = self.speech[:_FRAME_BYTES], self.speech[_FRAME_BYTES:]
            if frame and self.speech_at is None:
                self.speech_at = began  # the clip's first sample is this old when it is sent
                self.taken_up.set()
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            await self.ws.send(frame.ljust(_FRAME_BYTES, b"\0") if frame else _SILENCE)
            self.player.advance()
            playing = self.player.buffered_ms > 0
            if self.was_playing and not playing:
                await self.report("drained")
            elif playing:
                await self.report()
            self.was_playing = playing
            press = self.stop_at_ms
            if press is not None and self.first_audio_at is not None and not self.one_moment:
                if self.player.played_ms - self.reply_from_ms >= press:
                    self.stop_at_ms = None
                    self.player.flush()
                    self.muted = True  # until a later flush of the server: see VoiceInterrupt
                    self.stopped_in = self.epoch
                    await self.send(
                        type="voice.interrupt",
                        played_ms=round(self.player.played_ms, 1),
                        epoch=self.epoch,
                    )
                    self.note({"type": "client.stop", "played_ms": self.player.played_ms})

    def note(self, event: dict, at: float | None = None) -> None:
        event["at"] = self.now() if at is None else at
        self.events.append(event)
        if self.verbose and event["type"] not in ("llm.delta", "voice.state"):
            brief = {k: v for k, v in event.items() if k not in ("seq", "ts", "at", "hits")}
            print(f"  {event['at'] - self.began:7.2f}  {json.dumps(brief)[:200]}", file=sys.stderr)

    async def receive(self) -> None:
        try:
            async for frame in self.ws:
                if isinstance(frame, bytes):
                    if self.muted:
                        continue  # a piece of the reply that Stop was pressed on
                    if self.first_audio_at is None:
                        self.first_audio_at = self.now()
                        self.player.advance()
                        self.reply_from_ms = self.player.played_ms
                        self.reply_from_byte = len(self.player.heard)
                        self.note({"type": "client.first_audio"})
                    due = self.player.take(frame)
                    if self.first_sound_at is None:
                        sample = first_sound(frame)
                        if sample is not None:
                            self.first_sound_at = due + sample / self.player.rate
                            self.note({"type": "client.first_sound"}, self.first_sound_at)
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
                            self.epoch = event.get("id", self.epoch + 1)
                            if self.epoch > self.stopped_in:
                                self.muted = False
                        # The browser reports after each of the three, a resume as a tick.
                        await self.report("tick" if action == "resume" else action)
                    elif kind == "tool.started":
                        self.one_moment = True  # what plays until the answer is not the reply
                    elif kind == "assistant.message" and self.first_audio_at is None:
                        self.one_moment = False  # nothing was said while the tool ran
                    elif kind in ("turn.failed", "turn.interrupted"):
                        self.one_moment = False
                    elif kind == "speech.spoken" and event.get("filler"):
                        # "One moment" has been heard out. The audio after it is the answer,
                        # and is timed as a reply of its own.
                        self.first_audio_at = self.first_sound_at = None
                        self.one_moment = False
                    elif kind == "session.ready":
                        self.session_id = event["session_id"]
                    elif kind == "session.caught_up":
                        self.live_from = len(self.events) + 1
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
        presses Stop that long into the reply. Where a turn uses a tool, the reply in both is
        the answer and not the "one moment" before it.
        """
        pcm, onset_s, end_s = read_wav(path)
        if over_ms is not None:
            await self.answer_begins()
        mark = len(self.events)
        if over_ms is None:
            await self.quiet()
            await asyncio.sleep(0.6)
            mark = len(self.events)
            self.first_audio_at = self.first_sound_at = None
            self.stop_at_ms = stop_at_ms
        else:
            await asyncio.sleep(max(0.0, self.first_sound_at + over_ms / 1000 - self.now()))
        self.paused_at = None
        self.taken_up.clear()
        self.speech_at, self.speech = None, pcm
        await asyncio.wait_for(self.taken_up.wait(), 5)  # with the microphone's next piece
        onset_at, end_at = self.speech_at + onset_s, self.speech_at + end_s
        result: dict[str, Any] = {"say": path.name, "speech_ms": round((end_s - onset_s) * 1000)}
        if over_ms is not None:
            cut = await self.until(lambda: self.since(mark, "speech.spoken"), 20)
            # Every attempt is in the output with what came of it, also the ones that paused
            # nothing: a figure made only of the attempts that worked would flatter.
            if self.paused_at is not None:
                result["pause_after_onset_ms"] = round((self.paused_at - onset_at) * 1000)
                result["interruption"] = "paused"
            else:
                result["interruption"] = "cut without a pause" if cut else "nothing was cut"
            if cut:
                result |= self.keep_cut(mark)
                result |= {
                    "cut_played_ms": cut[0]["played_ms"],
                    "cut_text_spoken": cut[0]["text_spoken"],
                    "cut_after_onset_ms": round((cut[0]["at"] - onset_at) * 1000),
                }
            mark = len(self.events)
            self.first_audio_at = self.first_sound_at = None
        started = await self.until(lambda: self.since(mark, "turn.started"), 30)
        if not started:
            return result | {"error": "no turn started"}
        turn_id = started[0]["turn_id"]
        self.turn_mark = mark
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
        model = next((e for e in events if e["type"] == "llm.finished"), {})
        result |= {
            "reply": message.get("text"),
            "tools": sum(e["type"] == "tool.started" for e in events),
            "outcome": finished.get("outcome"),
            "end_of_turn_ms": heard.get("end_of_turn_ms"),
            "trigger": heard.get("trigger"),
            "digit_mode": heard.get("digit_mode"),
            "model_first_token_ms": model.get("ttft_ms"),
            "model_ms": model.get("total_ms"),
            "model_started_early_ms": model.get("early_ms"),
            "server_voice_to_voice_ms": spoken.get("voice_to_voice_ms"),
            "synthesis_ms": spoken.get("synthesis_ms"),
            "interrupted": spoken.get("interrupted"),
            "text_spoken": spoken.get("text_spoken"),
        }
        for name, kind in (
            ("first_audio", "client.first_audio"),
            ("first_sound", "client.first_sound"),
        ):
            noted = self.since(mark, kind)
            if noted:
                result[f"{name}_after_speech_ms"] = round((noted[0]["at"] - end_at) * 1000)
                result[f"{name}_after_turn_ms"] = round((noted[0]["at"] - started[0]["at"]) * 1000)
        sounds = self.since(mark, "client.first_sound")
        if any(e.get("filler") for e in self.since(mark, "speech.spoken")) and len(sounds) > 1:
            # The first sound was "one moment". The answer came after it.
            result["answer_after_speech_ms"] = round((sounds[-1]["at"] - end_at) * 1000)
        if spoken.get("interrupted"):
            result |= self.keep_cut(mark)
        return result

    async def answer_begins(self) -> None:
        """Wait until the answer of the turn that was just started can be heard."""
        await self.until(lambda: self.first_sound_at is not None)
        if self.since(self.turn_mark, "tool.started"):
            # What plays is "one moment". The answer follows once that has been heard out.
            await self.until(
                lambda: any(e.get("filler") for e in self.since(self.turn_mark, "speech.spoken")),
                30,
            )
            await self.until(lambda: self.first_sound_at is not None, 30)

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
        await caller.send(
            type="voice.start", echo_cancellation=not args.no_echo_cancellation, attempt="caller"
        )
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
        # What this call cost: its turns, its speech, and model calls that were started early
        # and not used. Events of an older conversation that was picked up are not this call's.
        live = caller.events[caller.live_from :]
        costs = [
            e.get("cost_usd") or 0
            for e in live
            if e["type"] in ("speech.heard", "speech.spoken", "voice.usage", "turn.finished")
            or (e["type"] == "llm.finished" and e.get("unused"))
        ]
        early = [e for e in live if e["type"] == "llm.finished" and e.get("early_ms") is not None]
        unused = [e for e in live if e["type"] == "llm.finished" and e.get("unused")]
        results.append(
            {
                "session_id": caller.session_id,
                "cost_usd": round(sum(costs), 5),
                "early_calls_used": len(early),
                "early_calls_unused": len(unused),
            }
        )
        caller.closed = True
        for task in tasks:
            task.cancel()
        return results


def _figures(values: list[float]) -> str:
    """n, median, 95th percentile and range of a list of milliseconds, as one line of text."""
    if not values:
        return "none"
    ordered = sorted(values)
    text = f"n {len(ordered)}, median {statistics.median(ordered):.0f} ms"
    if len(ordered) >= _P95_NEEDS:
        # The nearest rank: the smallest value that 95 in 100 are at or under.
        text += f", p95 {ordered[max(0, -(-95 * len(ordered) // 100) - 1)]:.0f} ms"
    else:
        text += f", p95 not given for fewer than {_P95_NEEDS} values"
    return f"{text}, least {ordered[0]:.0f}, most {ordered[-1]:.0f}"


def summary(paths: list[Path]) -> None:
    """Print the figures of many calls: the output of this program, one JSON object per line."""
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text().splitlines():
            with suppress(ValueError):
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    turns = [r for r in rows if "first_sound_after_speech_ms" in r and not r.get("error")]
    plain = [r for r in turns if r.get("tools") == 0]
    tooled = [r for r in turns if r.get("tools")]

    def of(group: list[dict], key: str) -> list[float]:
        return [r[key] for r in group if r.get(key) is not None]

    print(f"{len(paths)} files, {len(turns)} spoken turns that were answered aloud")
    print("Turns without tools:")
    for label, key in (
        ("  the customer stops, first sound of the reply", "first_sound_after_speech_ms"),
        ("  the customer stops, the turn starts", "turn_after_speech_ms"),
        ("  the turn starts, first sound", "first_sound_after_turn_ms"),
        ("  the customer stops, first piece of audio arrives", "first_audio_after_speech_ms"),
        ("  last word to end of turn, by the server's clock", "end_of_turn_ms"),
        ("  last word to first audio, by the server's clock", "server_voice_to_voice_ms"),
        ("  text to the voice, first audio back", "synthesis_ms"),
        ("  the model's first call, start to finish", "model_ms"),
        ("  how long before the turn that call began", "model_started_early_ms"),
    ):
        print(f"{label}: {_figures(of(plain, key))}")
    early = sum(r.get("model_started_early_ms") is not None for r in plain)
    print(f"  turns whose first model call began before the turn: {early} of {len(plain)}")
    print("Turns with tools:")
    print(
        f"  the customer stops, first sound: {_figures(of(tooled, 'first_sound_after_speech_ms'))}"
    )
    print(
        "  the customer stops, first sound of the answer: "
        f"{_figures(of(tooled, 'answer_after_speech_ms') or of(tooled, 'first_sound_after_speech_ms'))}"
    )
    tried = [r for r in rows if "interruption" in r]
    print(f"Speaking over a reply, {len(tried)} attempts:")
    for outcome in sorted({r["interruption"] for r in tried}):
        print(f"  {outcome}: {sum(r['interruption'] == outcome for r in tried)}")
    print(f"  speech begins, playback is paused: {_figures(of(tried, 'pause_after_onset_ms'))}")
    print(f"  speech begins, the reply is cut: {_figures(of(tried, 'cut_after_onset_ms'))}")
    calls = [r for r in rows if "session_id" in r]
    spent = sum(r.get("cost_usd") or 0 for r in calls)
    used = sum(r.get("early_calls_used") or 0 for r in calls)
    unused = sum(r.get("early_calls_unused") or 0 for r in calls)
    print(f"{len(calls)} calls, ${spent:.2f} at list prices")
    print(f"Model calls started early: {used} used, {unused} not used")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", help="the backend's WebSocket, or the first file with --summary")
    parser.add_argument("script", nargs="*", help="a JSON file with the steps")
    parser.add_argument("--summary", action="store_true", help="read output files, print figures")
    parser.add_argument("--origin", default="http://localhost:3210")
    parser.add_argument("--session", default=None, help="go on with this conversation")
    parser.add_argument("--save", default=None, help="write the audio that was played to this WAV")
    parser.add_argument(
        "--cuts", default=None, help="a folder for what was played of each cut reply"
    )
    parser.add_argument("--no-echo-cancellation", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print every event to stderr")
    args = parser.parse_args()
    if args.summary:
        summary([Path(args.url), *map(Path, args.script)])
        return
    if len(args.script) != 1:
        parser.error("one script")
    script = Path(args.script[0])
    for result in asyncio.run(run(args, json.loads(script.read_text()), script.parent)):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
