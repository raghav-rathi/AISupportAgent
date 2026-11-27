"""One customer's voice session: the microphone in, the same agent, a voice out.

A voice session sits on a session hub next to the connections that read its events. It hears the
customer through Deepgram's recogniser, starts a turn of the agent for each thing they say, and
speaks the agent's answer. Four rules decide what it does.

1. Nothing unconfirmed is spoken. Text the model writes while it may still call a tool is never
   said. An answer is said from the moment its step is known to be the answer. An answer the
   gate held is said only once its grounding check has passed. An answer that fails the check
   after it began is cut off, and the correction is said in its place.
2. The customer can always interrupt. When they start to speak, playback pauses: first on the
   level of the microphone alone, a quarter of a second in, and for good once the recogniser has
   a word. If what the recogniser hears is the assistant's own voice coming back through the
   microphone, playback goes on. Otherwise the reply is cancelled, and the conversation keeps of
   it only the words that had been played.
3. An interruption gets no answer past its check. If the customer speaks over an answer whose
   grounding check is still running, the voice stops and the check runs on. What they said waits
   the second that takes. If the check fails, they hear the correction before anything else.
4. Numbers get time. While the customer gives an account number or a PIN the recogniser waits
   out their pauses, and the turn ends when the digits are in.

All of it runs in one loop that takes one thing at a time: a message from the recogniser, an
event of the agent's turn, a report from the client's player, a tick of the clock. Audio does not
go through the loop. The microphone's audio goes straight to the recogniser, and the voice's
audio straight to the client.

The agent's first model call starts before the turn does. The recogniser says early that a turn
may be over, and that is when the call goes out, on the words heard so far. If the turn then ends
on those words, it takes the call. See agent/early.py.

Each reply is spoken on a connection of its own, opened ahead of time. On a connection that has
spoken before, the voice begins the next reply with a silence, a second long in most trials and
over four in the worst: it treats the two replies as one speech and pauses between them. The
first reply on a connection starts at once.

The voice also pauses for a second or more between sentences, in most sessions. Silence beyond
a short pause is left out of what the client gets. So the client's playhead and the voice's own
clock run apart, and each reply keeps the places where audio was left out, to say in the voice's
terms where the customer stopped listening.
"""

from __future__ import annotations

import asyncio
import logging
import math
import operator
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from supportagent.events import (
    TURN_ENDED,
    ProtocolError,
    SpeechHeard,
    SpeechSpoken,
    SttPartial,
    VoicePlayback,
    VoiceStarted,
    VoiceState,
    VoiceStopped,
    VoiceUsage,
)
from supportagent.voice import text as wording
from supportagent.voice.deepgram import (
    FluxListener,
    FluxSpeaker,
    SpeechConfig,
    SpeechError,
    Spoken,
    Turn,
)

if TYPE_CHECKING:
    from supportagent.agent.early import EarlyCall
    from supportagent.api.hub import SessionHub, Subscription

logger = logging.getLogger(__name__)

GREETING = "Hi, this is Fernhill support. I'm an AI assistant. How can I help?"
TURN_FAILED = "Sorry, something went wrong on my side. Please say that again."

_TICK_S = 0.1
# One or two words that the reply also contains do not say who is speaking. Playback stays
# paused this long for more words, then the customer gets the floor.
_ECHO_WAIT_S = 0.7
# Playback that was paused on the microphone's level alone waits this long for the recogniser
# to report a turn. In trials it did so 0.6 to 0.7 s after speech began.
_HUSH_WAIT_S = 0.9
# The microphone counts as the customer speaking when it is this many times louder than the
# loudest it has been of late while a reply played, and at least this loud, twice in a row.
_HUSH_RATIO = 2.5
_HUSH_MIN_RMS = 600.0
_HUSH_CHUNKS = 2
_HUSH_ARMED_AFTER = 6  # pieces of microphone audio heard during playback before the level counts
_HUSH_MISSES = 2  # pauses that were for nothing before the level is no longer trusted
_DIGITS_WANTED = 12  # an account number and a PIN
_DIGITS_SETTLED_S = 0.6  # no new digit for this long: they are all in
_MIC_SILENT_S = 8.0  # no audio from the client for this long: the microphone is gone
_AHEAD_S = 5.0  # audio may arrive this much faster than time passes, and no more
_FRAME_BYTES = (320, 32_000)  # one piece of microphone audio: 10 ms to 1 s
_BAD_FRAMES = 50  # pieces of another size before the client is taken for no microphone
_RECONNECT_ONCE_PER_S = 30.0
_YOUNG_S = 10.0  # a spare connection that closes sooner than this was not closed for idling
# The voice's silence is a noise floor of 15 to 30 on the 16-bit scale. Its quietest speech
# measured ten times that.
_SILENT_RMS = 50.0
# How much of a silence the client gets, and how much quiet stays in front of the voice when it
# goes on after a silence that was left out.
_PAUSE_MS = 240.0
_LEAD_MS = 40.0
_NO_AUDIO_S = 5.0  # a reply whose audio does not come, or stops coming, has lost its connection
# An answer the customer spoke over gets this long to finish its grounding check. The check took
# 0.8 s at the median on the hosted judge.
_LEAVE_S = 4.0
_WAIT_FOR_TYPED_S = 20.0  # a spoken message waits this long for a typed one's turn to end
_ACCOUNT_EVERY_S = 30.0  # microphone time is stored and counted at least this often
_INBOX = 2_000  # things waiting for the loop. A client that fills it is not a player.
# How the recogniser takes a turn. In digit mode it never ends a turn by itself: the turn ends
# when the digits are in, when a sentence without digits is plainly over, or after 3 s of
# silence. Tried against the live service: at any lower threshold a 1.5 s pause inside an account
# number ended the turn.
_USUAL = {"eot_threshold": 0.7, "eot_timeout_ms": 5000, "numerals": False}
_DIGITS = {"eot_threshold": 1.0, "eot_timeout_ms": 3000, "numerals": True}

StopReason = Literal["client", "idle", "limit", "replaced", "error"]
Admit = Callable[[str], Awaitable[ProtocolError | None]]
# Told how many seconds of microphone were just used. Returns why voice has to end now, if it
# has: the day's budget for speech is spent, or this visitor's time is.
Spend = Callable[[float], Awaitable[str | None]]


def _rms(audio: bytes) -> float:
    samples = memoryview(audio[: len(audio) & ~1]).cast("h")
    if not len(samples):
        return 0.0
    return math.sqrt(sum(map(operator.mul, samples, samples)) / len(samples))


def _silent(audio: bytes) -> bool:
    """Whether a piece of the voice's audio holds its noise floor and nothing else."""
    return _rms(audio) < _SILENT_RMS


def _quiet_head(audio: bytes, rate: int) -> int:
    """How many bytes of silence a piece of the voice's audio begins with, in steps of 5 ms."""
    step = rate // 200 * 2
    for start in range(0, len(audio), step):
        if not _silent(audio[start : start + step]):
            return start
    return len(audio)


def _same(one: str, other: str) -> bool:
    return " ".join(one.split()) == " ".join(other.split())


def _retrieve(task: asyncio.Task[None]) -> None:
    """Look at a task's failure, so that one nobody waits for is not reported as forgotten."""
    if not task.cancelled():
        task.exception()


async def _no_limit(_: float) -> str | None:
    return None


@dataclass(frozen=True)
class VoiceLimits:
    max_seconds: float = 600.0  # one voice session
    idle_seconds: float = 120.0  # nobody says anything
    greeting: bool = True
    max_chars: int = 2_000  # of one spoken message, as for a typed one


@dataclass(eq=False)
class _Line:
    """One connection to the synthesis, and the reply it speaks once it has one."""

    born: float
    speaker: FluxSpeaker | None = None
    opening: asyncio.Task[None] | None = None
    reply: _Reply | None = None


@dataclass(eq=False)
class _Reply:
    """One reply on its way to the customer's ear."""

    turn_id: str | None  # None for wording of our own that belongs to no turn
    began: float  # when its text went to the synthesis
    after: float | None  # when the customer stopped speaking, if this answers them
    line: _Line
    text: str = ""  # what has gone to the synthesis
    before: str = ""  # an earlier piece of the same answer, heard to its end
    must: bool = False  # a correction: what comes next waits for it instead of replacing it
    first_audio: float | None = None  # when its first sound went to the client
    last_audio: float = 0.0  # when the synthesis last sent a piece of it, silent or not
    sent_ms: float = 0.0  # audio passed on to the client
    made_ms: float = 0.0  # audio the synthesis has made, its silences included
    quiet_ms: float = _PAUSE_MS  # of the silence in progress, what was passed on
    held: bytes = b""  # the end of a silence that was left out, for the lead before the voice
    odd: bytes = b""  # half a sample at the end of a piece, waiting for its other half
    # Where audio was left out: the place in what the client got, and in what was made.
    skips: list[tuple[float, float]] = field(default_factory=list)
    made: Spoken | None = None  # the synthesis has made all of its audio
    due: float | None = None  # by when the client must have played it
    over: bool = False  # played to the end, or cut

    def in_voice_time(self, heard_ms: float) -> float:
        """Where the voice's own clock stood when the client had played this much of the reply."""
        sent, made = 0.0, 0.0
        for skip in self.skips:
            if skip[0] > heard_ms:
                break
            sent, made = skip
        return made + (heard_ms - sent)

    @property
    def speaking(self) -> str:
        """Everything of the answer the customer can have heard so far, this piece included."""
        return f"{self.before} {self.text}".strip()


@dataclass(frozen=True)
class _Queued:
    """Something to say when the reply that is playing has been heard to its end."""

    turn_id: str | None
    words: str
    before: str
    must: bool


@dataclass
class _Said:
    """What the customer just said, until the turn it started has a name."""

    turn: Turn
    at: float  # when the recogniser called the turn over
    ended_at: float | None  # when the last word ended, on our clock
    digit_mode: bool


@dataclass
class _Leaving:
    """A turn the customer spoke over while its answer was not confirmed. It is left to finish."""

    turn_id: str
    heard: str  # what they had heard of the answer
    since: float
    retracted: bool = False  # the answer failed its check
    final: str = ""  # the turn's last word, as the voice would say it
    stopped: bool = False  # it took too long and has been cancelled


@dataclass
class _Pending:
    """A spoken message that waits for a turn to end before it starts its own."""

    said: str
    early: EarlyCall | None
    since: float


class VoiceSession:
    def __init__(
        self,
        hub: SessionHub,
        subscription: Subscription,
        config: SpeechConfig,
        limits: VoiceLimits,
        *,
        admit: Admit,
        spend: Spend = _no_limit,
        barge_in: bool = True,
        attempt: str = "",
        on_closed: Callable[[VoiceSession], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`subscription` is the connection with the microphone: it alone gets the audio.

        `admit` says whether a message may start a turn, the way a typed one is checked.
        `spend` is told of microphone time as it is used. `barge_in` false means the client
        could not switch echo cancellation on. The microphone then hears the reply, so speech
        during a reply is not taken as an interruption. `attempt` names the client's request,
        and comes back in what the session tells it about starting and stopping.
        """
        self._hub, self._sub, self._config, self._limits = hub, subscription, config, limits
        self._admit, self._spend, self._barge_in, self._attempt = admit, spend, barge_in, attempt
        self._on_closed, self._clock = on_closed, clock
        self._inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=_INBOX)
        self._tasks: list[asyncio.Task[None]] = []
        self._hearing: asyncio.Task[None] | None = None
        self._closing: set[asyncio.Task[None]] = set()  # connections being closed
        self._lines: set[_Line] = set()  # connections that have a reply and are not closed
        self._listeners = 0  # goes up when the recogniser's socket is opened again
        self._listener = FluxListener(config)
        self._spare = self._line()  # the connection the next reply will be spoken on
        self._spare_failed_at: float | None = None
        self._unlisten: Callable[[], None] | None = None
        self._stopped = False
        self._status_now: str | None = None
        self._began = self._active_at = self._mic_at = self._accounted_at = clock()
        self._reconnected_at: float | None = None
        self._line_failed_at: float | None = None
        # The microphone.
        self._fed_s = 0.0  # audio received since voice started
        self._unbilled_s = 0.0  # of that, what no stored event has accounted for
        self._heard_s = 0.0  # audio sent to the recogniser's socket of the moment
        self._mic_clock = float("inf")  # our clock when that socket's audio began
        self._bad_frames = 0
        # Its level while a reply plays, for pausing before the recogniser has a word.
        self._room = 0.0  # the loudest of late, fading
        self._room_heard = 0  # pieces that went into that
        self._loud = 0  # pieces in a row that were far louder
        self._hushed_at: float | None = None  # playback is paused on the level alone
        self._hush_misses = 0
        self._leaky = False  # the microphone has heard the reply in this session
        # The customer's turn, as the recogniser reports it.
        self._utterance: int | None = None  # the turn they are in
        self._deaf: int | None = None  # a turn that is not taken, see `barge_in`
        self._worded = False  # the turn has a word
        self._paused_at: float | None = None  # playback is paused until we know who speaks
        self._echo_of: str | None = None  # the turn so far is the reply itself, with this text
        self._over_echo: str | None = None  # the customer spoke over an echo of this text
        self._digit_mode = False
        self._waits_out = False  # in digit mode: the recogniser is told never to end the turn
        self._owed_digits = 0  # replies to go that keep digit mode on without asking again
        self._digits_seen = 0
        self._digits_at = 0.0
        self._ending = False  # we have asked the recogniser to end the turn
        self._early: EarlyCall | None = None  # the model call started on the turn so far
        self._said: _Said | None = None
        self._speech_end: float | None = None
        self._pending: _Pending | None = None
        # The agent's turn.
        self._voice_turns: set[str] = set()  # turns that began with speech
        self._cut_turns: set[str] = set()  # turns the customer has spoken over and stopped
        self._confirmed: set[str] = set()  # turns whose last word has arrived
        self._turn_id: str | None = None  # the voice turn that is running or ran last
        self._drafts: dict[str, str] = {}  # text of each model call so far
        self._said_text: dict[str, str] = {}  # what was handed to the voice or queued, by turn
        self._heard_text: dict[str, str] = {}  # of that, what was played to its end
        self._leaving: _Leaving | None = None
        self._owed: str | None = None  # a correction the customer has not heard yet
        # The reply.
        self._reply: _Reply | None = None
        self._next: list[_Queued] = []  # what is said after it
        self._played = 0.0  # the client's last report
        self._client_paused = False  # the client was told to pause, and not yet to go on
        self._paused_since = 0.0
        self._pause_seen = asyncio.Event()  # it has said where it paused
        self._playhead_posted = False
        # The client's playhead at the start of the reply in progress. None between a cut and
        # the client's word on where it stood.
        self._client_at: float | None = 0.0

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _line(self) -> _Line:
        """A connection to the synthesis that is not open yet."""
        line = _Line(born=self._clock())
        line.speaker = FluxSpeaker(
            self._config,
            on_audio=lambda audio: self._audio(line, audio),
            on_spoken=lambda spoken: self._post("spoken", (line, spoken)),
            on_failed=lambda error: self._post("line_failed", (line, error)),
        )
        return line

    def _open(self, line: _Line) -> asyncio.Task[None]:
        if line.opening is None:
            line.opening = asyncio.create_task(line.speaker.open())
            line.opening.add_done_callback(_retrieve)
        return line.opening

    def _warm(self) -> None:
        """Have the connection for the reply after this one ready when it is needed.

        Unless the service has just closed a spare one as soon as it was open: then the next is
        opened when a reply needs it, and no sooner, so that a service that turns connections
        away is not asked for one after another.
        """
        self._spare = self._line()
        failed = self._spare_failed_at
        if failed is None or self._spare.born - failed >= _RECONNECT_ONCE_PER_S:
            self._open(self._spare)

    def _hang_up(self, line: _Line) -> None:
        """Close a connection whose reply is over, without making anyone wait for it."""
        self._lines.discard(line)
        if line.opening is not None:
            line.opening.cancel()
        task = asyncio.create_task(line.speaker.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    # Starting and stopping.

    async def start(self) -> None:
        """Open the speech sockets and take the hub's events. Raises SpeechError if that fails,
        or if voice was stopped while they opened."""
        if self._hub.speech_down > 0:
            self._hub.speech_down -= 1
            raise SpeechError("simulated failure")
        # Both are waited for, also when one fails: a socket that opens after the failure of
        # the other would belong to nobody.
        opened = await asyncio.gather(
            self._listener.open(), self._open(self._spare), return_exceptions=True
        )
        failed = next((result for result in opened if isinstance(result, BaseException)), None)
        if failed is not None or self._stopped:
            await asyncio.gather(
                self._listener.close(), self._spare.speaker.close(), return_exceptions=True
            )
            if self._stopped:
                # Stopped while its sockets opened, which also cancels the one that was still
                # opening. Whoever stopped it has told the client.
                raise SpeechError("stopped while it was starting")
            if isinstance(failed, SpeechError):
                raise failed
            raise SpeechError(f"could not start: {type(failed).__name__}") from failed
        self._began = self._active_at = self._mic_at = self._accounted_at = self._clock()
        self._hub.redactor.spoken = True
        self._unlisten = self._hub.listen(lambda event: self._post("event", event))
        self._tell(
            VoiceStarted(
                input_rate=self._config.input_rate,
                output_rate=self._config.output_rate,
                attempt=self._attempt,
            )
        )
        self._status("listening")
        self._hearing = asyncio.create_task(self._hear(self._listener, self._listeners))
        self._tasks = [asyncio.create_task(self._run()), asyncio.create_task(self._tick())]
        # A conversation is greeted once, at its very beginning. A stored event of any kind
        # says that this is not it: the greeting itself leaves one. Going by memory alone would
        # greet again on every reconnect, and each greeting is paid for.
        if (
            self._limits.greeting
            and self._hub.turns == 0
            and not self._hub.greeted
            and self._hub.log.last_seq == 0
        ):
            self._hub.greeted = True
            self._post("greet", None)

    async def stop(self, reason: StopReason = "client", message: str = "") -> None:
        """Turn voice off. A turn that is running goes on, and its answer appears as text."""
        await self._shutdown(reason, message)

    def fail(self, message: str) -> None:
        """Act as if the speech service had dropped out for good."""
        self._post("down", SpeechError(message))

    async def _shutdown(self, reason: StopReason, message: str) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._unlisten is not None:
            self._unlisten()
        if self._hub.voice is self:
            self._hub.voice = None
        self._hub.redactor.expecting = self._hub.redactor.spoken = False
        me = asyncio.current_task()
        for task in (*self._tasks, self._hearing, self._spare.opening):
            if task is not None and task is not me:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        early, self._early = self._early, None
        pending, self._pending = self._pending, None
        for call in (early, pending.early if pending is not None else None):
            if call is not None:
                self._hub.let_go(call)
        reply = self._reply
        await asyncio.gather(
            self._listener.close(),
            *(line.speaker.close() for line in {self._spare, *self._lines}),
            *self._closing,
            return_exceptions=True,
        )
        try:
            if reply is not None and not reply.over:
                # Cut off by the end of voice. The full text is on the screen.
                await self._account(reply, interrupted=True)
            # A turn that was being left to finish its check finishes as a typed one would:
            # its answer, or what replaces it, is on the screen.
            self._leaving = None
            if self._unbilled_s > 0:
                seconds, self._unbilled_s = self._unbilled_s, 0.0
                await self._hub.record(
                    VoiceUsage(listened_s=round(seconds, 2), cost_usd=self._listen_cost(seconds)),
                    None,
                )
                await self._spend(seconds)
        except Exception:
            logger.exception("could not store the last of a voice session's usage")
        self._tell(VoiceStopped(reason=reason, message=message, attempt=self._attempt))
        if self._on_closed is not None:
            self._on_closed(self)

    # What comes in from outside the loop.

    def _post(self, kind: str, payload: Any) -> None:
        if self._stopped:
            return
        try:
            self._inbox.put_nowait((kind, payload))
        except asyncio.QueueFull:
            # Nothing of ours fills it. A client that sends reports by the thousand does.
            logger.warning("voice in session %s was flooded", self._hub.session_id)
            task = asyncio.create_task(self._shutdown("error", "it ran into a problem"))
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)

    async def feed(self, audio: bytes) -> None:
        """Microphone audio from the client: mono PCM16 at the input rate."""
        if self._stopped:
            return
        size = len(audio)
        if size % 2 or not _FRAME_BYTES[0] <= size <= _FRAME_BYTES[1]:
            # Not what a microphone sends. A piece of two bytes would be a message to the
            # recogniser for every sample, and an empty one would keep a dead line looking
            # alive.
            self._bad_frames += 1
            if self._bad_frames == _BAD_FRAMES:
                self.fail("the client's audio is not a microphone's")
            return
        now = self._clock()
        seconds = size / 2 / self._config.input_rate
        if self._fed_s + seconds > now - self._began + _AHEAD_S:
            return  # faster than a microphone can produce it: not a microphone
        self._fed_s += seconds
        self._unbilled_s += seconds
        self._heard_s += seconds
        self._mic_at = now
        # The audio up to second `heard_s` was here by `now`. The smallest gap between the two
        # is the best reading of when this socket's audio began, since a delay can only add.
        self._mic_clock = min(self._mic_clock, now - self._heard_s)
        self._weigh_level(audio)
        await self._listener.send(audio)

    def _weigh_level(self, audio: bytes) -> None:
        """Pause playback when the microphone gets loud while a reply plays.

        The recogniser reports a turn once it has a word, 0.6 to 0.7 s after the customer
        began. Their voice in the microphone is there at once. With echo cancellation on, the
        reply itself leaves little in the microphone, so a level far above what has been there
        while the reply played is the customer. Playback pauses on that, and goes on if no turn
        follows. Where the reply does leak into the microphone the bar rises with it, and after
        an echo has been recognised as one, or two pauses were for nothing, the level is no
        longer used.
        """
        reply = self._reply
        if (
            not self._barge_in
            or self._leaky
            or self._hush_misses >= _HUSH_MISSES
            or reply is None
            or reply.over
            or reply.first_audio is None
            or self._client_paused
            or self._utterance is not None
        ):
            self._loud = 0
            return
        level = _rms(audio)
        armed = self._room_heard >= _HUSH_ARMED_AFTER
        if armed and level >= max(_HUSH_MIN_RMS, self._room * _HUSH_RATIO):
            self._loud += 1
            if self._loud >= _HUSH_CHUNKS:
                self._loud = 0
                self._post("hush", None)
            return
        self._loud = 0
        self._room = max(level, self._room * 0.98)
        self._room_heard += 1

    def playhead(self, played_ms: float, after: str) -> None:
        """The client's report of how much audio it has played since voice started."""
        # Taken at once and not in turn: the loop may be waiting for exactly this number. And
        # one wake-up serves any number of reports, so a client cannot queue them without end.
        self._played = max(self._played, played_ms)
        if after == "pause":
            self._pause_seen.set()
        elif after == "flush" and self._client_at is None:
            self._client_at = played_ms
        if not self._playhead_posted:
            self._playhead_posted = True
            self._post("playhead", None)

    def interrupt(self, played_ms: float) -> None:
        """The customer stopped the reply by hand, and their player has dropped the rest."""
        self._post("stop_pressed", played_ms)

    def _audio(self, line: _Line, audio: bytes) -> None:
        """A piece of a reply's audio from the synthesis: on to the client, without long silences.

        A silence is passed on up to `_PAUSE_MS`. What comes after that is left out. When the
        voice goes on, `_LEAD_MS` of quiet goes in front of it, so that a soft first sound is
        not clipped, and no more: the rest would only be the customer waiting. A reply starts
        as if such a silence had just been passed, so the silence before its first sound goes
        the same way.
        """
        reply = line.reply
        if reply is None or reply.over or self._stopped:
            return
        if reply.odd:
            audio, reply.odd = reply.odd + audio, b""
        if len(audio) % 2:
            audio, reply.odd = audio[:-1], audio[-1:]  # the client reads whole samples
        if not audio:
            return
        rate = self._config.output_rate
        reply.last_audio = self._clock()
        ms = len(audio) / 2 / rate * 1000
        lead = int(_LEAD_MS / 1000 * rate) * 2
        if _silent(audio):
            if reply.quiet_ms < _PAUSE_MS:
                self._pass(reply, audio, reply.made_ms)
                reply.quiet_ms += ms
            else:
                reply.held = (reply.held + audio)[-lead:]
        else:
            start = 0
            if reply.quiet_ms >= _PAUSE_MS:
                # The voice goes on after a silence that was left out.
                head = _quiet_head(audio, rate)
                if head >= lead:
                    start = head - lead
                else:
                    kept = reply.held[max(0, len(reply.held) - (lead - head)) :]
                    self._pass(reply, kept, reply.made_ms - len(kept) / 2 / rate * 1000)
            self._pass(reply, audio[start:], reply.made_ms + start / 2 / rate * 1000)
            reply.held = b""
            reply.quiet_ms = 0.0
        reply.made_ms += ms

    def _pass(self, reply: _Reply, audio: bytes, made_at: float) -> None:
        """Send one piece of audio to the client. `made_at` is its place in what the voice made."""
        if not audio:
            return
        if reply.first_audio is None:
            reply.first_audio = self._clock()
            self._status("speaking")
        if abs(reply.in_voice_time(reply.sent_ms) - made_at) > 0.5:
            reply.skips.append((reply.sent_ms, made_at))
        reply.sent_ms += len(audio) / 2 / self._config.output_rate * 1000
        self._sub.offer(audio)

    def _tell(self, event: BaseModel) -> None:
        self._sub.offer(event)

    def _pause_client(self) -> None:
        self._pause_seen.clear()
        self._client_paused = True
        self._paused_since = self._clock()
        self._tell(VoicePlayback(action="pause"))

    def _let_play(self) -> None:
        """Nobody is weighing who speaks any more: a player that was paused for that goes on."""
        self._paused_at = self._hushed_at = None
        if self._client_paused:
            self._client_paused = False
            reply = self._audible()
            if reply is not None and reply.due is not None:
                reply.due += self._clock() - self._paused_since  # it plays that much later
            self._tell(VoicePlayback(action="resume"))

    def _flush_client(self) -> None:
        """The client drops what it has buffered and says where its playhead stands."""
        self._client_paused = False
        self._tell(VoicePlayback(action="flush"))
        self._client_at = None  # until its report comes, see `playhead`

    def _status(self, state: str) -> None:
        if state != self._status_now:
            self._status_now = state
            self._tell(VoiceState(state=state))

    def _rest(self) -> None:
        """The status when no reply is being spoken."""
        if self._audible() is not None and self._audible().first_audio is not None:
            return
        if self._utterance is not None and self._utterance != self._deaf:
            self._status("hearing")
        elif self._pending is not None or (
            self._hub.busy and self._turn_id is not None and self._leaving is None
        ):
            self._status("thinking")
        else:
            self._status("listening")

    # The loop.

    async def _hear(self, listener: FluxListener, number: int) -> None:
        try:
            async for turn in listener.turns():
                self._post("turn", turn)
            error = SpeechError("speech to text closed the connection")
        except SpeechError as exc:
            error = exc
        except Exception:
            # Anything else must end the same way. A reader that dies in silence leaves voice
            # on and deaf, with the microphone still streaming.
            logger.exception("the recogniser's reader failed")
            error = SpeechError("speech to text: its messages could not be handled")
        self._post("listener_failed", (number, error))

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(_TICK_S)
            self._post("tick", None)

    async def _run(self) -> None:
        while not self._stopped:
            kind, payload = await self._inbox.get()
            try:
                await getattr(self, f"_on_{kind}")(payload)
            except SpeechError as exc:
                await self._on_down(exc)
            except Exception:
                logger.exception("voice failed in session %s", self._hub.session_id)
                await self._shutdown("error", "it ran into a problem")

    async def _on_greet(self, _: None) -> None:
        await self._speak(None, GREETING)

    async def _on_hush(self, _: None) -> None:
        reply = self._audible()
        if (
            reply is not None
            and reply.first_audio is not None
            and not self._client_paused
            and self._utterance is None
        ):
            self._pause_client()
            self._hushed_at = self._clock()

    async def _on_tick(self, _: None) -> None:
        now = self._clock()
        if self._hushed_at is not None and now - self._hushed_at >= _HUSH_WAIT_S:
            # The microphone got loud and the recogniser heard no speech: a cough, a door.
            self._hush_misses += 1
            self._let_play()
        if self._paused_at is not None and now - self._paused_at >= _ECHO_WAIT_S:
            if self._worded:
                await self._cut_in()
            else:
                self._let_play()  # a turn that has no word after all this time is no speech
        if (
            self._digit_mode
            and self._utterance is not None
            and not self._ending
            and self._digits_seen >= _DIGITS_WANTED
            and now - self._digits_at >= _DIGITS_SETTLED_S
        ):
            self._ending = True
            await self._listener.end_turn()
        await self._check_waits(now)
        reply = self._audible()
        if reply is not None:
            if reply.due is not None and now >= reply.due:
                await self._played_out(reply)  # the client never said so
            elif reply.made is None and now - max(reply.began, reply.last_audio) >= _NO_AUDIO_S:
                # Its audio does not come, and the connection has not said that it broke.
                await self._on_line_failed((reply.line, SpeechError("the voice went silent")))
        elif self._utterance is None and not self._hub.busy and self._pending is None:
            if now - self._active_at >= self._limits.idle_seconds:
                await self._shutdown("idle", "nobody spoke for a while")
                return
        else:
            self._active_at = now
        if now - self._began >= self._limits.max_seconds:
            minutes = max(1, round(self._limits.max_seconds / 60))
            await self._shutdown("limit", f"a voice session lasts at most {minutes} minutes here")
        elif now - self._mic_at >= _MIC_SILENT_S:
            await self._shutdown("error", "no audio arrived from the microphone")
        elif now - self._accounted_at >= _ACCOUNT_EVERY_S:
            await self._count_listening(now)

    async def _check_waits(self, now: float) -> None:
        """Things that wait for a turn to end do not wait for ever."""
        leaving = self._leaving
        if leaving is not None and not leaving.stopped and now - leaving.since >= _LEAVE_S:
            # Its check is taking too long. What was heard of it is kept as unverified.
            leaving.stopped = True
            await self._hub.interrupt_turn(leaving.heard)
        pending = self._pending
        if pending is not None and leaving is None and now - pending.since >= _WAIT_FOR_TYPED_S:
            self._pending = self._said = None
            if pending.early is not None:
                self._hub.let_go(pending.early)
            self._tell(
                ProtocolError(
                    code="turn_in_progress", message="wait for the current turn to finish"
                )
            )
            self._rest()

    async def _count_listening(self, now: float) -> None:
        """Store the microphone time no turn has accounted for, and see if there is more to use.

        A session in which nobody speaks would otherwise show in the day's spend only when it
        ends, ten minutes later.
        """
        self._accounted_at = now
        if self._unbilled_s <= 0 or self._said is not None:
            return  # nothing to count, or a turn is about to count it
        seconds, self._unbilled_s = self._unbilled_s, 0.0
        await self._hub.record(
            VoiceUsage(listened_s=round(seconds, 2), cost_usd=self._listen_cost(seconds)), None
        )
        why = await self._spend(seconds)
        if why is not None:
            await self._shutdown("limit", why)

    # The speech service failing.

    async def _on_down(self, error: SpeechError) -> None:
        logger.warning("speech is down in session %s: %s", self._hub.session_id, error)
        await self._shutdown("error", "speech is unavailable right now")

    async def _on_listener_failed(self, failure: tuple[int, SpeechError]) -> None:
        """The recogniser's socket broke. It is opened again, once. If that fails, voice is over."""
        number, error = failure
        if number != self._listeners:
            return  # a socket that has been replaced already
        now = self._clock()
        logger.warning("the recogniser dropped in session %s: %s", self._hub.session_id, error)
        again = self._reconnected_at is None or now - self._reconnected_at > _RECONNECT_ONCE_PER_S
        if again and self._hub.speech_down == 0:
            self._reconnected_at = now
            # The turn it was in is gone with it, and so is whatever was being weighed.
            self._utterance = self._deaf = self._echo_of = self._over_echo = None
            self._worded = self._ending = False
            self._forget_early()
            self._let_play()
            old = self._listener
            self._listeners += 1
            self._listener = FluxListener(self._config)
            if self._hearing is not None:
                self._hearing.cancel()
            await old.close()
            try:
                await self._listener.open()
            except SpeechError as exc:
                error = exc
            else:
                # The new socket counts its audio from zero.
                self._heard_s, self._mic_clock = 0.0, float("inf")
                if self._digit_mode:
                    await self._listener.configure(**_DIGITS)
                self._hearing = asyncio.create_task(self._hear(self._listener, self._listeners))
                self._rest()
                return
        await self._on_down(error)

    async def _on_line_failed(self, failure: tuple[_Line, SpeechError]) -> None:
        """A connection to the synthesis broke.

        A spare one that nobody has spoken on is replaced, and that is all: the service closes
        connections that sit idle. One that was speaking takes its reply with it. The text of
        that reply is on the screen, and the next one gets a new connection. Twice in half a
        minute means the service is down.
        """
        line, error = failure
        now = self._clock()
        if line is self._spare:
            if now - line.born < _YOUNG_S:
                self._spare_failed_at = now  # closed as soon as it was open: see `_warm`
            self._warm()
            return
        reply = line.reply
        if reply is None or reply.over:
            return
        logger.warning("the voice dropped in session %s: %s", self._hub.session_id, error)
        again = self._line_failed_at is None or now - self._line_failed_at > _RECONNECT_ONCE_PER_S
        self._line_failed_at = now
        if not again or self._hub.speech_down > 0:
            await self._on_down(error)
            return
        self._paused_at = self._hushed_at = None
        self._next.clear()
        self._flush_client()
        self._hang_up(line)
        await self._account(reply, interrupted=True)
        self._rest()

    # The customer speaks.

    def _audible(self) -> _Reply | None:
        """The reply the customer is hearing, or about to."""
        reply = self._reply
        return reply if reply is not None and not reply.over else None

    def _words(self, turn: Turn) -> str:
        """The customer's message as the turn so far gives it."""
        said = wording.tidy_digits(turn.transcript)
        if self._over_echo is not None:
            said = wording.without_echo(said, self._over_echo)
        return said.strip()[: self._limits.max_chars]

    def _forget_early(self) -> None:
        early, self._early = self._early, None
        if early is not None:
            self._hub.let_go(early)

    async def _on_turn(self, turn: Turn) -> None:
        if turn.event == "StartOfTurn":
            await self._speech_started(turn)
        elif turn.index != self._utterance or turn.index == self._deaf:
            if turn.event == "EndOfTurn" and turn.index == self._utterance:
                self._utterance = self._deaf = None
        elif turn.event == "Update":
            await self._speech_goes_on(turn)
        elif turn.event == "EagerEndOfTurn":
            await self._speech_may_be_over(turn)
        elif turn.event == "TurnResumed":
            self._forget_early()
        elif turn.event == "EndOfTurn":
            await self._speech_ended(turn)

    async def _speech_started(self, turn: Turn) -> None:
        now = self._clock()
        self._forget_early()
        self._utterance, self._active_at = turn.index, now
        self._digits_seen, self._digits_at, self._ending = 0, now, False
        self._echo_of = self._over_echo = None
        self._worded = False
        hushed, self._hushed_at = self._hushed_at is not None, None
        reply = self._audible()
        if reply is not None and reply.first_audio is not None and not self._barge_in:
            self._deaf = turn.index  # it may be the reply itself, and we cannot tell
            return
        if wording.words(turn.transcript):
            await self._first_words(turn)
        elif hushed:
            # Playback is paused already, on the microphone's level. The words decide.
            self._paused_at = now
        else:
            self._paused_at = None

    async def _first_words(self, turn: Turn) -> None:
        """The turn has its first word. Until then it may be a cough, and stops nothing."""
        self._worded = True
        reply = self._audible()
        if reply is None or reply.first_audio is None:
            self._let_play()
            self._status("hearing")
            await self._note(turn)
            await self._cut_in()  # they have more to say, and nothing has been heard yet
            return
        # Either the customer speaks over the reply, or the microphone hears the reply.
        if self._leaky and wording.sounds_like_echo(turn.transcript, reply.speaking) is not False:
            # It has heard the reply before in this session. Playback is not stopped for that
            # again: it goes on until words come that the reply does not have.
            self._let_play()
            self._echo_of = reply.speaking
            return
        if not self._client_paused:
            self._pause_client()
        self._paused_at = self._clock()
        await self._weigh(turn, reply.speaking)

    async def _weigh(self, turn: Turn, speaking: str) -> None:
        """Decide whose voice the recogniser is hearing while playback is paused."""
        verdict = wording.sounds_like_echo(turn.transcript, speaking)
        if verdict is True and self._spoke_after(turn, speaking):
            verdict, self._over_echo = False, speaking  # the reply's words, and then their own
        if verdict is True:
            self._leaky = True
            self._let_play()
            self._echo_of = speaking
        elif verdict is False:
            await self._cut_in()
            await self._note(turn)

    def _spoke_after(self, turn: Turn, speaking: str) -> bool:
        """Whether a transcript that begins with the reply's words goes on with the customer's."""
        rest = wording.without_echo(wording.tidy_digits(turn.transcript), speaking)
        return wording.is_speech(rest)

    async def _speech_goes_on(self, turn: Turn) -> None:
        if not self._worded:
            if wording.words(turn.transcript):
                await self._first_words(turn)
            return
        if self._paused_at is not None:
            reply = self._audible()
            if reply is not None:
                await self._weigh(turn, reply.speaking)
                return
            self._let_play()  # the reply ended while its player was paused
        if self._echo_of is not None:
            echo = wording.sounds_like_echo(turn.transcript, self._echo_of) is not False
            if echo and not self._spoke_after(turn, self._echo_of):
                return
            # Words the reply does not have: the customer is speaking over it.
            self._over_echo, self._echo_of = self._echo_of, None
            if self._audible() is not None and not self._client_paused:
                self._pause_client()
            await self._cut_in()
        await self._note(turn)

    async def _note(self, turn: Turn) -> None:
        """Show the customer's words so far, and give a number the time it needs."""
        now = self._clock()
        said = self._words(turn)
        self._active_at = now
        if not self._digit_mode and wording.gives_credentials(said):
            await self._digits(True)
        self._tell(SttPartial(text=self._hub.redactor.text(said, customer=True)))
        if not self._digit_mode or self._ending:
            return
        digits = wording.digit_count(said)
        if digits:
            self._forget_early()  # a number may not be finished
        if digits != self._digits_seen:
            self._digits_seen, self._digits_at = digits, now
        # A sentence without a digit in it is the recogniser's to end, the way it ends any
        # other: it looks at every piece of audio, and this loop only at what it reports four
        # times a second. Measured live, turns ended half a second sooner that way. The moment
        # a digit shows, the recogniser is told to wait again.
        if bool(digits) != self._waits_out:
            self._waits_out = bool(digits)
            threshold = _DIGITS["eot_threshold"] if digits else _USUAL["eot_threshold"]
            await self._listener.configure(eot_threshold=threshold)
        if digits == 0 and turn.confidence >= _USUAL["eot_threshold"]:
            # Not a number after all, and the sentence is over: no need to wait out the silence.
            self._ending = True
            await self._listener.end_turn()

    async def _digits(self, on: bool) -> None:
        if on != self._digit_mode:
            self._digit_mode = on
            # What is shown of the customer's words follows: every short number is masked.
            # And stays masked when the recogniser goes back to its usual ways, for as long
            # as nobody is verified: a PIN said in two halves has its second half in a turn
            # of its own, which may come after a reply that asked for nothing.
            redactor = self._hub.redactor
            redactor.expecting = on or (redactor.expecting and redactor.strict)
            self._waits_out = on
            await self._listener.configure(**(_DIGITS if on else _USUAL))

    async def _speech_may_be_over(self, turn: Turn) -> None:
        """The recogniser thinks the customer may have finished. The model can start reading."""
        if (
            not self._worded
            or self._paused_at is not None
            or self._echo_of is not None
            or self._leaving is not None
            or self._pending is not None
        ):
            return
        said = self._words(turn)
        if not said or (self._digit_mode and wording.digit_count(said)):
            return
        if self._early is not None and self._early.text == said:
            return
        self._forget_early()
        if await self._admit(said) is None:
            self._early = await self._hub.start_early(said)

    async def _speech_ended(self, turn: Turn) -> None:
        self._utterance = None
        early, self._early = self._early, None
        handed = False
        try:
            said = await self._heard_out(turn)
            if said:
                if early is not None and early.text != said:
                    self._hub.let_go(early)
                    early = None
                handed = True
                await self._take(turn, said, early)
        finally:
            if early is not None and not handed:
                self._hub.let_go(early)

    async def _heard_out(self, turn: Turn) -> str:
        """What the customer said in the turn that just ended, or nothing if it was not them."""
        if not self._worded:
            if not wording.words(turn.transcript):
                self._let_play()  # a turn without a word: a cough. Nothing stops for it.
                self._rest()
                return ""
            await self._first_words(turn)
        if self._paused_at is not None:
            # A word or two, over before it was clear whose they were.
            reply = self._audible()
            if (
                reply is not None
                and wording.sounds_like_echo(turn.transcript, reply.speaking)
                and not self._spoke_after(turn, reply.speaking)
            ):
                self._leaky = True
                self._let_play()
                return ""
            await self._cut_in()
        if self._echo_of is not None:
            echo, self._echo_of = self._echo_of, None
            if not self._spoke_after(turn, echo):
                self._rest()  # the microphone heard the reply and nothing else
                return ""
            self._over_echo = echo  # the reply's words, and then the customer's
            await self._cut_in()
        said = self._words(turn)
        if not said:
            self._let_play()
            self._rest()
        return said

    async def _take(self, turn: Turn, said: str, early: EarlyCall | None) -> None:
        """Start a turn for what the customer said, or line it up behind the one that runs."""
        refusal = await self._admit(said)
        if refusal is not None:
            if early is not None:
                self._hub.let_go(early)
            self._tell(refusal)
            await self._shutdown("limit", refusal.message)
            return
        now = self._clock()
        ended = None
        if turn.last_word_end is not None and self._mic_clock != float("inf"):
            ended = min(now, self._mic_clock + turn.last_word_end)
        self._said = _Said(turn, now, ended, self._digit_mode)
        if self._digit_mode and wording.digit_count(said):
            self._owed_digits = 0  # they have given a number: the next reply decides again
        self._speech_end = ended if ended is not None else now
        waiting, self._pending = self._pending, None
        if waiting is not None:
            # They had said something that is still waiting, and now said more.
            said = f"{waiting.said} {said}"[: self._limits.max_chars]
            for call in (waiting.early, early):
                if call is not None:
                    self._hub.let_go(call)
            early = None
        if self._hub.busy:
            # A turn is running that is not theirs to stop by speaking: an answer whose check
            # is being left to finish, or a message someone typed. This one starts when it ends.
            self._pending = _Pending(said, early, now)
            self._status("thinking")
            return
        await self._begin(said, early)

    async def _begin(self, said: str, early: EarlyCall | None) -> None:
        owed, self._owed = self._owed, None
        if owed:
            # Before anything else they hear that an answer they heard the start of was wrong.
            await self._speak(None, owed, must=True)
        if self._hub.start_turn(said, "voice", early=early):
            self._status("thinking")
        else:
            self._said = None
            self._rest()

    async def _cut_in(self, played_ms: float | None = None, *, pressed: bool = False) -> None:
        """The customer has the floor: stop the reply, and the turn behind it as far as is right.

        A turn that has said nothing yet is cancelled. A turn whose answer was being said, or
        had been said, before its grounding check ended is left to finish that check. A turn that
        is over has its reply cut down to what was heard. A typed message's turn is not theirs
        to stop by speaking, only by pressing Stop.
        """
        self._paused_at = self._hushed_at = None
        reply = self._audible()
        queued = {item.turn_id for item in self._next}
        self._next.clear()
        heard: str | None = None
        turn_id: str | None = None
        if reply is not None:
            heard = await self._cut(reply, played_ms) or ""
            turn_id = reply.turn_id
        else:
            self._let_play()
        # Their own turn, if it is the one that runs. A typed message's turn is not.
        running = self._turn_id if self._hub.busy and self._turn_id is not None else None
        if turn_id is not None and turn_id != running and running is None:
            # The reply's turn is over and stored. It is cut down to what was heard of it.
            whole = turn_id not in queued and _same(heard or "", self._said_text.get(turn_id, ""))
            if not whole:
                await self._hub.interrupt_turn(heard or "")
        if running is None:
            if pressed and self._hub.busy:
                self._hub.cancel()  # a typed message's turn, stopped the way its own Stop would
        elif self._leaving is not None and self._leaving.turn_id == running:
            pass  # already being left to finish
        elif running in self._said_text:
            # Its answer was being said, or had been, and is not confirmed yet.
            was = heard if turn_id == running else self._heard_text.get(running)
            self._leaving = _Leaving(running, was or "", self._clock())
        else:
            # Nothing of it has been said. It is stopped, and leaves no word behind.
            self._cut_turns.add(running)
            await self._hub.stop()
        self._rest()

    async def _on_stop_pressed(self, played_ms: float) -> None:
        self._played = max(self._played, played_ms)
        flushed = self._audible() is not None
        await self._cut_in(played_ms, pressed=True)
        if not flushed:
            # The client plays nothing until it has this, see `VoiceInterrupt`.
            self._flush_client()

    # The agent answers.

    async def _on_event(self, event: BaseModel) -> None:
        kind = event.type
        turn_id = getattr(event, "turn_id", None)
        if kind == "error":
            await self._on_refusal(event)
            return
        if kind == "turn.started":
            if event.mode == "voice":
                self._voice_turns.add(turn_id)
                self._turn_id, self._drafts = turn_id, {}
                await self._count_heard(turn_id)
            else:
                # The customer has typed instead. What is being said belongs to an older turn.
                self._turn_id = None
                self._next.clear()
                reply = self._audible()
                if reply is not None:
                    await self._cut(reply, keep=False)
                self._rest()
            return
        if kind == "assistant.message":
            # A reply that asks for the account number or the PIN, spoken or typed: what the
            # customer says next is a number, and the recogniser has to give it time. It keeps
            # that time for two more replies if what they said first was "one moment" and the
            # reply to that asks for nothing: the number is still to come.
            if wording.asks_for_credentials(event.text):
                self._owed_digits = 2
                await self._digits(True)
            elif self._digit_mode and self._owed_digits > 0 and self._hub.redactor.strict:
                self._owed_digits -= 1
            else:
                await self._digits(False)
        elif kind in ("turn.failed", "turn.interrupted"):
            self._owed_digits = 0
            await self._digits(False)  # whatever was asked for, the turn that asked is gone
        leaving = self._leaving
        if leaving is not None and turn_id == leaving.turn_id:
            await self._on_leaving(leaving, event)
            return
        if turn_id is None or turn_id not in self._voice_turns or turn_id in self._cut_turns:
            if kind in TURN_ENDED:
                await self._turn_over()
            return
        if kind == "llm.delta":
            self._drafts[event.call_id] = self._drafts.get(event.call_id, "") + event.text
        elif kind == "llm.reset":
            self._drafts.pop(event.call_id, None)
        elif kind == "answer.drafted":
            if not event.held:
                await self._say(turn_id, self._drafts.get(event.call_id, ""))
        elif kind == "answer.retracted":
            await self._withdraw(turn_id)
        elif kind == "assistant.message":
            self._confirmed.add(turn_id)
            await self._say(turn_id, event.text)
        elif kind in ("turn.failed", "turn.interrupted"):
            await self._withdraw(turn_id)
            if kind == "turn.failed":
                await self._speak(None, TURN_FAILED)
            await self._turn_over()
        elif kind == "turn.finished":
            await self._turn_over()

    async def _on_refusal(self, event: BaseModel) -> None:
        """A spoken message started no turn: the session was taken, or could not be read."""
        if self._said is None or getattr(event, "client_id", None) is not None:
            return  # not about a spoken message
        if event.code not in ("turn_in_progress", "failed_to_start", "session_expired"):
            return
        self._said = None
        if event.code == "failed_to_start":
            await self._speak(None, TURN_FAILED)
        else:
            self._rest()

    async def _on_leaving(self, leaving: _Leaving, event: BaseModel) -> None:
        """An event of the turn the customer spoke over while its answer was unconfirmed."""
        kind = event.type
        if kind == "answer.retracted":
            leaving.retracted = True
        elif kind == "assistant.message":
            self._confirmed.add(leaving.turn_id)
            leaving.final = wording.speakable(event.text)
        elif kind in TURN_ENDED:
            self._leaving = None
            if kind != "turn.finished":
                # The check never gave a verdict. What was heard stays in the conversation as
                # words nobody verified, see `repair`.
                await self._hub.interrupt_turn(leaving.heard)
            elif leaving.retracted:
                self._owed = leaving.final  # they heard the start of an answer that did not stand
            elif not _same(leaving.heard, leaving.final):
                await self._hub.interrupt_turn(leaving.heard)  # it stands, as far as it was heard
            await self._turn_over()

    async def _turn_over(self) -> None:
        """A turn has ended. Say what is owed, and start what was waiting for it."""
        if self._hub.busy:
            return  # the end of an older turn, heard late
        pending, self._pending = self._pending, None
        if pending is not None:
            await self._begin(pending.said, pending.early)
        elif self._owed and self._utterance is None:
            owed, self._owed = self._owed, None
            await self._speak(None, owed, must=True)
        elif self._audible() is None:
            self._rest()

    async def _count_heard(self, turn_id: str) -> None:
        said, self._said = self._said, None
        if said is None:
            return
        seconds, self._unbilled_s = self._unbilled_s, 0.0
        self._accounted_at = self._clock()
        turn = said.turn
        spoke = None
        if turn.first_word_start is not None and turn.last_word_end is not None:
            spoke = round((turn.last_word_end - turn.first_word_start) * 1000)
        waited = None
        if said.ended_at is not None:
            waited = round(max(0.0, said.at - said.ended_at) * 1000)
        await self._hub.record(
            SpeechHeard(
                speech_ms=spoke,
                end_of_turn_ms=waited,
                confidence=turn.confidence,
                trigger=turn.trigger or "model",
                digit_mode=said.digit_mode,
                listened_s=round(seconds, 2),
                cost_usd=self._listen_cost(seconds),
            ),
            turn_id,
        )
        why = await self._spend(seconds)
        if why is not None:
            await self._shutdown("limit", why)  # the turn goes on, and answers as text

    def _listen_cost(self, seconds: float) -> float:
        return round(seconds / 60 * self._config.stt_usd_per_minute, 6)

    async def _say(self, turn_id: str, text: str) -> None:
        """Speak `text` as the answer of a turn, or the part of it that has not been said.

        An answer goes to the synthesis whole, the moment it is known to be the answer. Sent
        without its end, a reply of one sentence stayed silent until the grounding check was
        done a second later: the voice starts a sentence only when it sees what follows it,
        or is told that nothing does. When the confirmed message has more to it than the
        answer had, a sentence about a ticket, that part is said after the first.
        """
        words = wording.speakable(text)
        said = self._said_text.get(turn_id, "")
        if not words.startswith(said):
            logger.warning("a confirmed answer differs from what was spoken")
            return
        rest = words[len(said) :].strip()
        if rest:
            await self._speak(turn_id, rest, before=said)

    async def _speak(
        self, turn_id: str | None, words: str, before: str = "", *, must: bool = False
    ) -> None:
        """Hand one complete piece of text to the synthesis, as the reply now being spoken.

        It takes the connection that was opened for it and has another one opened for the
        reply after it. It waits its turn behind a reply that has to be heard to its end: a
        correction, or the first piece of the same answer.
        """
        if turn_id is not None:
            self._said_text[turn_id] = f"{before} {words}".strip()
        current = self._audible()
        if current is not None:
            if current.must or (before and current.turn_id == turn_id):
                self._next.append(_Queued(turn_id, words, before, must))
                return
            await self._settle(current)
        line = self._spare
        self._warm()
        after = self._speech_end if turn_id is not None and not before else None
        for last_try in (False, True):
            try:
                await self._open(line)
                reply = _Reply(
                    turn_id=turn_id,
                    began=self._clock(),
                    after=after,
                    line=line,
                    text=words,
                    before=before,
                    must=must,
                )
                reply.last_audio = reply.began
                line.reply = self._reply = reply
                self._lines.add(line)
                await line.speaker.speak(words)
                await line.speaker.flush()
                return
            except SpeechError:
                # It did not open, perhaps long ago, or closed while it waited. One more try on
                # a new one, now that it is needed. If that fails too, speech is down.
                if line.reply is not None:
                    line.reply.over = True
                self._hang_up(line)
                if last_try:
                    raise
                line = self._line()

    async def _settle(self, reply: _Reply) -> None:
        """Make room for the next reply: this one has been played, or it is cut."""
        if (
            reply.made is not None
            and self._client_at is not None
            and self._played >= self._client_at + reply.sent_ms - 150
        ):
            self._next.clear()  # what replaces it is said now, not after something else
            await self._played_out(reply)
        else:
            await self._cut(reply, keep=False)

    async def _withdraw(self, turn_id: str) -> None:
        """Stop saying an answer that did not stand. What replaces it is said in full."""
        self._next = [item for item in self._next if item.turn_id != turn_id]
        reply = self._audible()
        if reply is not None and reply.turn_id == turn_id:
            self._paused_at = self._hushed_at = None
            await self._cut(reply, keep=False)
        self._said_text.pop(turn_id, None)
        self._heard_text.pop(turn_id, None)

    async def _cut(
        self, reply: _Reply, played_ms: float | None = None, *, keep: bool = True
    ) -> str | None:
        """Stop a reply in mid-speech. Returns what the customer had heard of it.

        With `keep` false nobody needs to know, and None comes back. `played_ms` is where the
        client's player stood if it has said so itself, as it does when Stop is pressed.
        """
        reply.over = True
        heard: str | None = None
        heard_ms = 0.0
        result = None
        try:
            if self._client_paused and keep:
                # Playback is stopping at this moment. Where it stopped is what was heard, and
                # the client's word on that is on its way.
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._pause_seen.wait(), 0.3)
            position = self._played if played_ms is None else played_ms
            if self._client_at is not None and reply.first_audio is not None:
                heard_ms = max(0.0, min(position - self._client_at, reply.sent_ms))
            if keep and heard_ms > 0:
                # The voice is told where playback stood, on its own clock, and says which
                # words that was. Its clock began with this reply: the connection has spoken
                # nothing else.
                result = await reply.line.speaker.interrupt(reply.in_voice_time(heard_ms))
                heard = result.text_spoken if result is not None else None
            if keep and heard is None:
                # The voice could not say. Nothing played means nothing heard. Otherwise this
                # is an estimate from the time played, at the pace the voice speaks, cut at a
                # word.
                total_ms = reply.made.audio_ms if reply.made is not None else 0.0
                voice_ms = reply.in_voice_time(heard_ms)
                chars = len(reply.text) * voice_ms / total_ms if total_ms else heard_ms * 0.014
                heard = reply.text[: int(chars)].rsplit(" ", 1)[0] if heard_ms else ""
            if keep:
                heard = f"{reply.before} {heard}".strip()
        finally:
            # Also when voice is being stopped in the middle of this: the client must not go on
            # playing, the connection must not stay open, and the reply is paid for once.
            self._flush_client()
            self._hang_up(reply.line)  # which also stops what the voice was still making
            await self._account(
                reply,
                made=result.spoken if result is not None else None,
                interrupted=True,
                played_ms=heard_ms if keep else None,
                text_spoken=heard if keep else None,
            )
        return heard if keep else None

    async def _on_spoken(self, done: tuple[_Line, Spoken]) -> None:
        line, spoken = done
        reply = line.reply
        if reply is None or reply.over or reply.made is not None:
            return
        reply.made = spoken
        # Audio arrives faster than it plays. The client says when it has played the last of it,
        # and if it never does, the reply is over once its audio must have run out.
        reply.due = self._clock() + reply.sent_ms / 1000 + 3.0
        await self._check_played()

    async def _on_playhead(self, _: None) -> None:
        self._playhead_posted = False
        await self._check_played()

    async def _check_played(self) -> None:
        reply = self._audible()
        if reply is None or reply.made is None or self._client_at is None:
            return
        if self._played >= self._client_at + reply.sent_ms - 5:
            await self._played_out(reply)

    async def _played_out(self, reply: _Reply) -> None:
        reply.over = True
        if self._client_at is not None:
            self._client_at += reply.sent_ms
        self._active_at = self._clock()
        # A player that was paused for a word that never came would stay paused, and the next
        # reply would queue behind nothing and never be heard.
        self._let_play()
        self._hang_up(reply.line)
        if reply.turn_id is not None:
            self._heard_text[reply.turn_id] = reply.speaking
        await self._account(reply, made=reply.made, interrupted=False)
        if self._next:
            item = self._next.pop(0)
            await self._speak(item.turn_id, item.words, item.before, must=item.must)
        else:
            self._rest()

    async def _account(
        self,
        reply: _Reply,
        *,
        made: Spoken | None = None,
        interrupted: bool,
        played_ms: float | None = None,
        text_spoken: str | None = None,
    ) -> None:
        """Store what a reply cost and how fast it came, once, when it is over."""
        reply.over = True
        chars = made.chars if made is not None and made.chars else len(reply.text)
        first = reply.first_audio
        await self._hub.record(
            SpeechSpoken(
                chars=chars,
                audio_ms=round(reply.sent_ms),
                synthesis_ms=round((first - reply.began) * 1000) if first is not None else None,
                voice_to_voice_ms=(
                    round((first - reply.after) * 1000)
                    if first is not None and reply.after is not None
                    else None
                ),
                interrupted=interrupted,
                played_ms=round(played_ms) if played_ms is not None else None,
                text_spoken=text_spoken,
                cost_usd=round(chars / 1000 * self._config.tts_usd_per_1k_chars, 6),
            ),
            reply.turn_id,
        )
