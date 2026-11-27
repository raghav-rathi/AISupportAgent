"""One customer's voice session: the microphone in, the same agent, a voice out.

A voice session sits on a session hub next to the connections that read its events. It hears the
customer through Deepgram's recogniser, starts a turn of the agent for each thing they say, and
speaks the agent's answer. Three rules decide what it does.

1. Nothing unconfirmed is spoken. Text the model writes while it may still call a tool is never
   said. An answer is said from the moment its step is known to be the answer. An answer the
   gate held is said only once its grounding check has passed. An answer that fails the check
   after it began is cut off, and the correction is said in its place.
2. The customer can always interrupt. When they start to speak, playback pauses at once. If what
   the recogniser hears is the assistant's own voice coming back through the microphone,
   playback goes on. Otherwise the reply is cancelled, and the conversation keeps of it only the
   words that had been played.
3. Numbers get time. While the customer gives an account number or a PIN the recogniser waits
   out their pauses, and the turn ends when the digits are in.

All of it runs in one loop that takes one thing at a time: a message from the recogniser, an
event of the agent's turn, a report from the client's player, a tick of the clock. Audio does not
go through the loop. The microphone's audio goes straight to the recogniser, and the voice's
audio straight to the client.

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
import operator
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from supportagent.events import (
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
    from supportagent.api.hub import SessionHub, Subscription

logger = logging.getLogger(__name__)

GREETING = "Hi, this is Fernhill support. I'm an AI assistant. How can I help?"
TURN_FAILED = "Sorry, something went wrong on my side. Please say that again."

_TICK_S = 0.1
# One or two words that the reply also contains do not say who is speaking. Playback stays
# paused this long for more words, then the customer gets the floor.
_ECHO_WAIT_S = 0.7
_DIGITS_WANTED = 12  # an account number and a PIN
_DIGITS_SETTLED_S = 0.6  # no new digit for this long: they are all in
_MIC_SILENT_S = 8.0  # no audio from the client for this long: the microphone is gone
_AHEAD_S = 5.0  # audio may arrive this much faster than time passes, and no more
_RECONNECT_ONCE_PER_S = 30.0
# The voice's silence is a noise floor of 15 to 30 on the 16-bit scale. Its quietest speech
# measured ten times that.
_SILENT_RMS = 50
# How much of a silence the client gets: this, and the last 80 ms before the voice goes on.
_PAUSE_MS = 240.0
# How the recogniser takes a turn. In digit mode it never ends a turn by itself: the turn ends
# when the digits are in, when a sentence without digits is plainly over, or after 3 s of
# silence. Tried against the live service: at any lower threshold a 1.5 s pause inside an account
# number ended the turn.
_USUAL = {"eot_threshold": 0.7, "eot_timeout_ms": 5000, "numerals": False}
_DIGITS = {"eot_threshold": 1.0, "eot_timeout_ms": 3000, "numerals": True}

StopReason = Literal["client", "idle", "limit", "replaced", "error"]
Admit = Callable[[str], Awaitable[ProtocolError | None]]


def _silent(audio: bytes) -> bool:
    """Whether a piece of the voice's audio holds its noise floor and nothing else."""
    samples = memoryview(audio[: len(audio) & ~1]).cast("h")
    return sum(map(operator.mul, samples, samples)) < _SILENT_RMS**2 * len(samples)


@dataclass(frozen=True)
class VoiceLimits:
    max_seconds: float = 600.0  # one voice session
    idle_seconds: float = 120.0  # nobody says anything
    greeting: bool = True
    max_chars: int = 2_000  # of one spoken message, as for a typed one


@dataclass(eq=False)
class _Line:
    """One connection to the synthesis, and the reply it speaks once it has one."""

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
    then: str | None = None  # a piece of the same answer to say when this one has played
    first_audio: float | None = None  # when its first sound went to the client
    sent_ms: float = 0.0  # audio passed on to the client
    made_ms: float = 0.0  # audio the synthesis has made, its silences included
    quiet_ms: float = _PAUSE_MS  # of the silence in progress, what was passed on. None at first.
    held: bytes | None = None  # the newest piece of silence that was left out
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


@dataclass
class _Said:
    """What the customer just said, until the turn it started has a name."""

    turn: Turn
    at: float  # when the recogniser called the turn over
    ended_at: float | None  # when the last word ended, on our clock
    digit_mode: bool


class VoiceSession:
    def __init__(
        self,
        hub: SessionHub,
        subscription: Subscription,
        config: SpeechConfig,
        limits: VoiceLimits,
        *,
        admit: Admit,
        barge_in: bool = True,
        on_closed: Callable[[VoiceSession], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`subscription` is the connection with the microphone: it alone gets the audio.

        `admit` says whether a message may start a turn, the way a typed one is checked.
        `barge_in` false means the client could not switch echo cancellation on. The microphone
        then hears the reply, so speech during a reply is not taken as an interruption.
        """
        self._hub, self._sub, self._config, self._limits = hub, subscription, config, limits
        self._admit, self._barge_in, self._on_closed, self._clock = (
            admit,
            barge_in,
            on_closed,
            clock,
        )
        self._inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._hearing: asyncio.Task[None] | None = None
        self._closing: set[asyncio.Task[None]] = set()  # connections being closed
        self._listeners = 0  # goes up when the recogniser's socket is opened again
        self._listener = FluxListener(config)
        self._spare = self._line()  # the connection the next reply will be spoken on
        self._unlisten: Callable[[], None] | None = None
        self._stopped = False
        self._status_now: str | None = None
        self._began = self._active_at = self._mic_at = clock()
        self._reconnected_at: float | None = None
        self._line_failed_at: float | None = None
        # The microphone.
        self._fed_s = 0.0  # audio received since voice started
        self._unbilled_s = 0.0  # of that, what no stored event has accounted for
        self._mic_clock = float("inf")  # our clock at the moment the microphone's audio began
        # The customer's turn, as the recogniser reports it.
        self._utterance: int | None = None  # the turn they are in
        self._deaf: int | None = None  # a turn that is not taken, see `barge_in`
        self._paused_at: float | None = None  # playback is paused until we know who speaks
        self._echo_of: str | None = None  # the turn so far is the reply itself, with this text
        self._over_echo: str | None = None  # the customer spoke over an echo of this text
        self._digit_mode = False
        self._digits_seen = 0
        self._digits_at = 0.0
        self._ending = False  # we have asked the recogniser to end the turn
        self._said: _Said | None = None
        self._speech_end: float | None = None
        # The agent's turn.
        self._voice_turns: set[str] = set()  # turns that began with speech
        self._cut_turns: set[str] = set()  # turns the customer has spoken over
        self._turn_id: str | None = None
        self._drafts: dict[str, str] = {}  # text of each model call so far
        self._said_text: dict[str, str] = {}  # what has gone to the synthesis, by turn
        # The reply.
        self._reply: _Reply | None = None
        self._played = 0.0  # the client's last report
        self._pause_pending = False  # the client has been told to pause and has not answered
        self._pause_seen = asyncio.Event()
        # The client's playhead at the start of the reply in progress. None between a cut and
        # the client's word on where it stood.
        self._client_at: float | None = 0.0

    def _line(self) -> _Line:
        """A connection to the synthesis that is not open yet."""
        line = _Line()
        line.speaker = FluxSpeaker(
            self._config,
            on_audio=lambda audio: self._audio(line, audio),
            on_spoken=lambda spoken: self._post("spoken", (line, spoken)),
            on_failed=lambda error: self._post("line_failed", (line, error)),
        )
        return line

    def _warm(self) -> None:
        """Open the connection for the reply after this one, so it is there when needed."""
        self._spare = self._line()
        self._spare.opening = asyncio.create_task(self._spare.speaker.open())

    def _hang_up(self, line: _Line) -> None:
        """Close a connection whose reply is over, without making anyone wait for it."""
        if line.opening is not None:
            line.opening.cancel()
        task = asyncio.create_task(line.speaker.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    # Starting and stopping.

    async def start(self) -> None:
        """Open the speech sockets and take the hub's events. Raises SpeechError if that fails."""
        if self._hub.speech_down > 0:
            self._hub.speech_down -= 1
            raise SpeechError("simulated failure")
        try:
            await asyncio.gather(self._listener.open(), self._spare.speaker.open())
        except SpeechError:
            await asyncio.gather(self._listener.close(), self._spare.speaker.close())
            raise
        self._began = self._active_at = self._mic_at = self._clock()
        self._hub.voice = self
        self._unlisten = self._hub.listen(lambda event: self._post("event", event))
        self._tell(
            VoiceStarted(input_rate=self._config.input_rate, output_rate=self._config.output_rate)
        )
        self._status("listening")
        self._hearing = asyncio.create_task(self._hear(self._listener, self._listeners))
        self._tasks = [asyncio.create_task(self._run()), asyncio.create_task(self._tick())]
        if self._limits.greeting and self._hub.turns == 0 and not self._hub.greeted:
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
        me = asyncio.current_task()
        for task in (*self._tasks, self._hearing, self._spare.opening):
            if task is not None and task is not me:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        reply = self._reply
        lines = [self._spare] + ([reply.line] if reply is not None and not reply.over else [])
        await asyncio.gather(
            self._listener.close(),
            *(line.speaker.close() for line in lines),
            *self._closing,
            return_exceptions=True,
        )
        try:
            if reply is not None and not reply.over:
                # Cut off by the end of voice. The full text is on the screen.
                await self._account(reply, interrupted=True)
            if self._unbilled_s > 0:
                seconds, self._unbilled_s = self._unbilled_s, 0.0
                await self._hub.record(
                    VoiceUsage(listened_s=round(seconds, 2), cost_usd=self._listen_cost(seconds)),
                    None,
                )
        except Exception:
            logger.exception("could not store the last of a voice session's usage")
        self._tell(VoiceStopped(reason=reason, message=message))
        if self._on_closed is not None:
            self._on_closed(self)

    # What comes in from outside the loop.

    def _post(self, kind: str, payload: Any) -> None:
        if not self._stopped:
            self._inbox.put_nowait((kind, payload))

    async def feed(self, audio: bytes) -> None:
        """Microphone audio from the client: mono PCM16 at the input rate."""
        if self._stopped:
            return
        now = self._clock()
        seconds = len(audio) / 2 / self._config.input_rate
        if self._fed_s + seconds > now - self._began + _AHEAD_S:
            return  # faster than a microphone can produce it: not a microphone
        self._fed_s += seconds
        self._unbilled_s += seconds
        self._mic_at = now
        # The audio up to second `fed_s` was here by `now`. The smallest gap between the two is
        # the best reading of when the microphone's audio began, since a delay can only add.
        self._mic_clock = min(self._mic_clock, now - self._fed_s)
        await self._listener.send(audio)

    def playhead(self, played_ms: float, after: str) -> None:
        """The client's report of how much audio it has played since voice started."""
        # Taken at once and not in turn: the loop may be waiting for exactly this number.
        self._played = max(self._played, played_ms)
        if after == "pause":
            self._pause_seen.set()
        self._post("playhead", (played_ms, after))

    def interrupt(self, played_ms: float) -> None:
        """The customer stopped the reply by hand, and their player has dropped the rest."""
        self._post("stop_pressed", played_ms)

    def _audio(self, line: _Line, audio: bytes) -> None:
        """A piece of a reply's audio from the synthesis: on to the client, without long silences.

        A silence is passed on up to `_PAUSE_MS`. What comes after that is left out, except the
        last piece before the voice goes on, so that a soft first sound is not clipped. A reply
        starts as if such a silence had just been passed: one before its first sound goes too.
        """
        reply = line.reply
        if reply is None or reply.over or self._stopped:
            return
        ms = len(audio) / 2 / self._config.output_rate * 1000
        silent = _silent(audio)
        if silent and reply.quiet_ms >= _PAUSE_MS:
            reply.held = audio
        else:
            if reply.held is not None and not silent:
                held_ms = len(reply.held) / 2 / self._config.output_rate * 1000
                self._pass(reply, reply.held, reply.made_ms - held_ms)
            self._pass(reply, audio, reply.made_ms)
            reply.held = None
            reply.quiet_ms = reply.quiet_ms + ms if silent else 0.0
        reply.made_ms += ms

    def _pass(self, reply: _Reply, audio: bytes, made_at: float) -> None:
        """Send one piece of audio to the client. `made_at` is its place in what the voice made."""
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
        self._pause_pending = True
        self._tell(VoicePlayback(action="pause"))

    def _resume_client(self) -> None:
        self._pause_pending = False
        self._tell(VoicePlayback(action="resume"))

    def _flush_client(self) -> None:
        """The client drops what it has buffered and says where its playhead stands."""
        self._pause_pending = False
        self._tell(VoicePlayback(action="flush"))
        self._client_at = None  # until its report comes, see `_on_playhead`

    def _status(self, state: str) -> None:
        if state != self._status_now:
            self._status_now = state
            self._tell(VoiceState(state=state))

    def _rest(self) -> None:
        """The status when no reply is being spoken."""
        if self._utterance is not None and self._utterance != self._deaf:
            self._status("hearing")
        else:
            self._status("thinking" if self._hub.busy and self._turn_id else "listening")

    # The loop.

    async def _hear(self, listener: FluxListener, number: int) -> None:
        try:
            async for turn in listener.turns():
                self._post("turn", turn)
            error = SpeechError("speech to text closed the connection")
        except SpeechError as exc:
            error = exc
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
                await self._shutdown("error", "voice ran into a problem, go on in text")

    async def _on_greet(self, _: None) -> None:
        await self._speak(None, GREETING)

    async def _on_tick(self, _: None) -> None:
        now = self._clock()
        if self._paused_at is not None and now - self._paused_at >= _ECHO_WAIT_S:
            await self._cut_in()
        if (
            self._digit_mode
            and self._utterance is not None
            and not self._ending
            and self._digits_seen >= _DIGITS_WANTED
            and now - self._digits_at >= _DIGITS_SETTLED_S
        ):
            self._ending = True
            await self._listener.end_turn()
        reply = self._reply
        if reply is not None and not reply.over:
            if reply.due is not None and now >= reply.due:
                await self._played_out(reply)  # the client never said so
        elif self._utterance is None and not self._hub.busy:
            if now - self._active_at >= self._limits.idle_seconds:
                await self._shutdown("idle", "voice turned off after a while of silence")
                return
        else:
            self._active_at = now
        if now - self._began >= self._limits.max_seconds:
            minutes = round(self._limits.max_seconds / 60)
            await self._shutdown(
                "limit", f"a voice session lasts at most {minutes} minutes here, start another"
            )
        elif now - self._mic_at >= _MIC_SILENT_S:
            await self._shutdown("error", "no audio is arriving from the microphone")

    # The speech service failing.

    async def _on_down(self, error: SpeechError) -> None:
        logger.warning("speech is down in session %s: %s", self._hub.session_id, error)
        await self._shutdown("error", "speech is unavailable right now, go on in text")

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
            self._utterance = self._deaf = self._paused_at = self._echo_of = None
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
                if self._digit_mode:
                    await self._listener.configure(**_DIGITS)
                self._hearing = asyncio.create_task(self._hear(self._listener, self._listeners))
                if self._audible() is None:
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
        if line is self._spare:
            self._warm()
            return
        reply = line.reply
        if reply is None or reply.over:
            return
        now = self._clock()
        logger.warning("the voice dropped in session %s: %s", self._hub.session_id, error)
        again = self._line_failed_at is None or now - self._line_failed_at > _RECONNECT_ONCE_PER_S
        self._line_failed_at = now
        if not again or self._hub.speech_down > 0:
            await self._on_down(error)
            return
        self._flush_client()
        self._hang_up(line)
        await self._account(reply, interrupted=True)
        self._rest()

    # The customer speaks.

    def _audible(self) -> _Reply | None:
        """The reply the customer is hearing, or about to."""
        reply = self._reply
        return reply if reply is not None and not reply.over else None

    async def _on_turn(self, turn: Turn) -> None:
        if turn.event == "StartOfTurn":
            await self._speech_started(turn)
        elif turn.index != self._utterance or turn.index == self._deaf:
            if turn.event == "EndOfTurn" and turn.index == self._utterance:
                self._utterance = self._deaf = None
        elif turn.event == "Update":
            await self._speech_goes_on(turn)
        elif turn.event == "EndOfTurn":
            await self._speech_ended(turn)

    async def _speech_started(self, turn: Turn) -> None:
        now = self._clock()
        self._utterance, self._active_at = turn.index, now
        self._digits_seen, self._digits_at, self._ending = 0, now, False
        self._echo_of = self._over_echo = self._paused_at = None
        reply = self._audible()
        if reply is not None and reply.first_audio is not None:
            if not self._barge_in:
                self._deaf = turn.index  # it may be the reply itself, and we cannot tell
                return
            # Either the customer speaks over the reply, or the microphone hears the reply.
            self._pause_client()
            self._paused_at = now
            await self._weigh(turn, reply.text)
            return
        self._status("hearing")
        await self._note(turn)
        if reply is not None or self._hub.busy:
            await self._cut_in()  # they have more to say, and nothing has been heard yet

    async def _weigh(self, turn: Turn, speaking: str) -> None:
        """Decide whose voice the recogniser is hearing while playback is paused."""
        verdict = wording.sounds_like_echo(turn.transcript, speaking)
        if verdict is True:
            self._resume_client()
            self._paused_at, self._echo_of = None, speaking
        elif verdict is False:
            await self._cut_in()
            await self._note(turn)

    async def _speech_goes_on(self, turn: Turn) -> None:
        if self._paused_at is not None:
            reply = self._audible()
            if reply is not None:
                await self._weigh(turn, reply.text)
            return
        if self._echo_of is not None:
            if wording.sounds_like_echo(turn.transcript, self._echo_of) is not False:
                return
            # Words the reply does not have: the customer is speaking over it.
            self._over_echo, self._echo_of = self._echo_of, None
            if self._audible() is not None:
                self._pause_client()
            await self._cut_in()
        await self._note(turn)

    async def _note(self, turn: Turn) -> None:
        """Show the customer's words so far, and give a number the time it needs."""
        now = self._clock()
        said = wording.tidy_digits(turn.transcript)
        if self._over_echo is not None:
            said = wording.without_echo(said, self._over_echo)
        self._active_at = now
        self._tell(SttPartial(text=self._hub.redactor.text(said, customer=True)))
        if not self._digit_mode and wording.gives_credentials(said):
            await self._digits(True)
        if not self._digit_mode or self._ending:
            return
        digits = wording.digit_count(said)
        if digits != self._digits_seen:
            self._digits_seen, self._digits_at = digits, now
        if digits == 0 and turn.confidence >= _USUAL["eot_threshold"]:
            # Not a number after all, and the sentence is over: no need to wait out the silence.
            self._ending = True
            await self._listener.end_turn()

    async def _digits(self, on: bool) -> None:
        if on != self._digit_mode:
            self._digit_mode = on
            await self._listener.configure(**(_DIGITS if on else _USUAL))

    async def _speech_ended(self, turn: Turn) -> None:
        self._utterance = None
        if self._paused_at is not None:
            # A word or two, over before it was clear whose they were.
            reply = self._audible()
            if reply is not None and wording.sounds_like_echo(turn.transcript, reply.text):
                self._resume_client()
                self._paused_at = None
                return
            await self._cut_in()
        if self._echo_of is not None:
            self._echo_of = None  # the microphone heard the reply and nothing else
            self._rest()
            return
        said = wording.tidy_digits(turn.transcript)
        if self._over_echo is not None:
            said = wording.without_echo(said, self._over_echo)
        said = said.strip()[: self._limits.max_chars]
        if not said:
            self._rest()
            return
        refusal = await self._admit(said)
        if refusal is not None:
            self._tell(refusal)
            await self._shutdown("limit", refusal.message)
            return
        if self._hub.busy:
            await self._hub.interrupt_turn("")  # a turn nobody is listening to any more
        now = self._clock()
        ended = None
        if turn.last_word_end is not None and self._mic_clock != float("inf"):
            ended = min(now, self._mic_clock + turn.last_word_end)
        self._said = _Said(turn, now, ended, self._digit_mode)
        self._speech_end = ended if ended is not None else now
        if self._hub.start_turn(said, "voice"):
            self._status("thinking")
        else:
            self._said = None
            self._rest()

    async def _cut_in(self, played_ms: float | None = None, *, flushed: bool = False) -> None:
        """The customer has the floor: stop the reply and the turn behind it."""
        self._paused_at = None
        reply = self._audible()
        heard = ""
        if reply is not None:
            heard = await self._cut(reply, played_ms, flushed=flushed) or ""
        turn_id = reply.turn_id if reply is not None else self._turn_id
        if self._hub.busy or (reply is not None and reply.turn_id is not None):
            if turn_id is not None:
                self._cut_turns.add(turn_id)
            await self._hub.interrupt_turn(heard if reply is not None and reply.turn_id else "")
        self._rest()

    async def _on_stop_pressed(self, played_ms: float) -> None:
        self._played = max(self._played, played_ms)
        if self._audible() is not None or self._hub.busy:
            await self._cut_in(played_ms, flushed=True)

    # The agent answers.

    async def _on_event(self, event: BaseModel) -> None:
        kind = event.type
        turn_id = getattr(event, "turn_id", None)
        if kind == "turn.started":
            if event.mode == "voice":
                self._voice_turns.add(turn_id)
                self._turn_id, self._drafts = turn_id, {}
                await self._count_heard(turn_id)
            else:
                # The customer has typed instead. What is being said belongs to an older turn.
                self._turn_id = None
                reply = self._audible()
                if reply is not None:
                    await self._cut(reply, keep=False)
            return
        if turn_id is None or turn_id not in self._voice_turns or turn_id in self._cut_turns:
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
            await self._say(turn_id, event.text)
            # A reply that asks for the account number or the PIN: what the customer says next
            # is a number, and the recogniser has to give it time.
            await self._digits(wording.asks_for_credentials(event.text))
        elif kind in ("turn.failed", "turn.interrupted"):
            await self._withdraw(turn_id)
            if kind == "turn.failed":
                await self._speak(None, TURN_FAILED)
            else:
                self._rest()

    async def _count_heard(self, turn_id: str) -> None:
        said, self._said = self._said, None
        if said is None:
            return
        seconds, self._unbilled_s = self._unbilled_s, 0.0
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
        if not rest:
            return
        current = self._audible()
        if said and current is not None and current.turn_id == turn_id:
            current.then = rest  # once the piece that is playing has been heard
        else:
            await self._speak(turn_id, rest, before=said)

    async def _speak(self, turn_id: str | None, words: str, before: str = "") -> None:
        """Hand one complete piece of text to the synthesis, as the reply now being spoken.

        It takes the connection that was opened for it and has another one opened for the
        reply after it.
        """
        current = self._audible()
        if current is not None:
            await self._settle(current)
        line = self._spare
        self._warm()
        try:
            if line.opening is not None:
                await line.opening
        except SpeechError:
            # It did not open, perhaps long ago. One more try, now that it is needed.
            line = self._line()
            await line.speaker.open()
        after = self._speech_end if turn_id is not None and not before else None
        reply = _Reply(
            turn_id=turn_id, began=self._clock(), after=after, line=line, text=words, before=before
        )
        line.reply = self._reply = reply
        if turn_id is not None:
            self._said_text[turn_id] = f"{before} {words}".strip()
        await line.speaker.speak(words)
        await line.speaker.flush()

    async def _settle(self, reply: _Reply) -> None:
        """Make room for the next reply: this one has been played, or it is cut."""
        if (
            reply.made is not None
            and self._client_at is not None
            and self._played >= self._client_at + reply.sent_ms - 150
        ):
            await self._played_out(reply)
        else:
            await self._cut(reply, keep=False)

    async def _withdraw(self, turn_id: str) -> None:
        """Stop saying an answer that did not stand. What replaces it is said in full."""
        reply = self._audible()
        if reply is not None and reply.turn_id == turn_id:
            await self._cut(reply, keep=False)
        self._said_text.pop(turn_id, None)

    async def _cut(
        self,
        reply: _Reply,
        played_ms: float | None = None,
        *,
        flushed: bool = False,
        keep: bool = True,
    ) -> str | None:
        """Stop a reply in mid-speech. Returns what the customer had heard of it.

        With `keep` false nobody needs to know, and None comes back. `flushed` says the client
        has dropped its buffer already, at `played_ms`.
        """
        reply.over = True
        if self._pause_pending and keep:
            # Playback is stopping at this moment. Where it stopped is what was heard, and the
            # client's word on that is on its way.
            with suppress(TimeoutError):
                await asyncio.wait_for(self._pause_seen.wait(), 0.3)
        position = self._played if played_ms is None else played_ms
        heard_ms = 0.0
        if self._client_at is not None and reply.first_audio is not None:
            heard_ms = max(0.0, min(position - self._client_at, reply.sent_ms))
        heard: str | None = None
        result = None
        if keep and heard_ms > 0:
            # The voice is told where playback stood, on its own clock, and says which words
            # that was. Its clock began with this reply: the connection has spoken nothing else.
            result = await reply.line.speaker.interrupt(reply.in_voice_time(heard_ms))
            heard = result.text_spoken if result is not None else None
        if flushed:
            self._pause_pending = False
            self._client_at = position
        else:
            self._flush_client()
        if keep and heard is None:
            # The voice could not say. Nothing played means nothing heard. Otherwise this is an
            # estimate from the time played, at the pace the voice speaks, cut at a word.
            total_ms = reply.made.audio_ms if reply.made is not None else 0.0
            voice_ms = reply.in_voice_time(heard_ms)
            chars = len(reply.text) * voice_ms / total_ms if total_ms else heard_ms * 0.014
            heard = reply.text[: int(chars)].rsplit(" ", 1)[0] if heard_ms else ""
        if keep:
            heard = f"{reply.before} {heard}".strip()
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

    async def _on_playhead(self, report: tuple[float, str]) -> None:
        played_ms, after = report
        if after == "flush" and self._client_at is None:
            self._client_at = played_ms
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
        self._hang_up(reply.line)
        await self._account(reply, made=reply.made, interrupted=False)
        if reply.then:
            await self._speak(reply.turn_id, reply.then, f"{reply.before} {reply.text}".strip())
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
