"""Deepgram's two streaming sockets, as this app uses them.

`FluxListener` is speech to text on `/v2/listen`: microphone audio in, turn events out. The model
itself says when the customer starts to speak and when they have finished, so there is no voice
activity detector and no end-of-turn timer of our own.

`FluxSpeaker` is text to speech on `/v2/speak`: text in, audio out. On an interruption it says
which words had been played, which is what goes into the conversation in place of the full reply.

What the service does was read in its documentation and then tried against the live service,
because three things matter here and the documentation is thin or wrong on them:

- An `Interrupt` still works after the turn's audio has all been sent, as long as the customer
  is still listening to it. Audio arrives about 1.4 times faster than it plays.
- An `Interrupt` sent right after the text, before any audio exists, cancels the turn. The
  documentation says it would be ignored.
- The offset in an `Interrupt` counts the audio the customer has heard since the socket opened,
  not since the reply began, and must be larger than the one before. An `Interrupt` without an
  offset is always accepted, and the service then counts everything it has made as played.
- The voice starts on a sentence when it has seen what follows it, or a `Flush`. A reply of
  one sentence that was sent without a `Flush` gave no audio for 1.05 to 1.1 s, until the
  `Flush` came. With it the first audio came after 0.11 to 0.19 s.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from supportagent.config import Settings

logger = logging.getLogger(__name__)

# Warnings that answer an Interrupt the service will not act on.
_NOT_INTERRUPTED = frozenset(
    {"NO_ACTIVE_SPEECH", "NO_AUDIO_GENERATED", "INTERRUPT_IN_PROGRESS", "INVALID_INTERRUPT_OFFSET"}
)


class SpeechError(Exception):
    """A speech socket could not be opened, or closed in the middle of a session."""


@dataclass(frozen=True)
class SpeechConfig:
    api_key: str
    base_url: str = "wss://api.deepgram.com"
    stt_model: str = "flux-general-en"
    tts_model: str = "flux-haley-en"
    keyterms: tuple[str, ...] = ()
    input_rate: int = 16_000  # what the customer's microphone is resampled to
    output_rate: int = 24_000  # what the reply is synthesised at
    connect_timeout_s: float = 6.0
    stt_usd_per_minute: float = 0.0065
    tts_usd_per_1k_chars: float = 0.045

    @classmethod
    def from_settings(cls, settings: Settings) -> SpeechConfig | None:
        """The speech setup of this deployment, or None when it has no key for it."""
        if settings.deepgram_api_key is None:
            return None
        return cls(
            api_key=settings.deepgram_api_key.get_secret_value(),
            base_url=settings.deepgram_base_url.rstrip("/"),
            stt_model=settings.stt_model,
            tts_model=settings.tts_model,
            keyterms=tuple(settings.stt_keyterms),
            stt_usd_per_minute=settings.stt_usd_per_minute,
            tts_usd_per_1k_chars=settings.tts_usd_per_1k_chars,
        )


@dataclass(frozen=True)
class Turn:
    """One message of the recogniser about the turn the customer is in."""

    event: str  # Update, StartOfTurn, EagerEndOfTurn, TurnResumed or EndOfTurn
    index: int
    transcript: str
    confidence: float  # that no more speech is coming in this turn
    window_end: float  # seconds of audio transcribed so far
    first_word_start: float | None
    last_word_end: float | None
    trigger: str | None  # on EndOfTurn: model, timeout or manual


@dataclass(frozen=True)
class Spoken:
    """One reply as the synthesis accounts for it."""

    audio_ms: float
    chars: int  # billable characters


@dataclass(frozen=True)
class Interrupted:
    """What the customer had heard of a reply when it was cut off."""

    played_ms: float  # on the socket's own clock: audio heard since it opened
    text_spoken: str | None  # None when the service was not told how far playback had got
    spoken: Spoken


async def _open(url: str, config: SpeechConfig) -> ClientConnection:
    try:
        return await connect(
            url,
            additional_headers={"Authorization": f"Token {config.api_key}"},
            open_timeout=config.connect_timeout_s,
            max_size=2**20,
        )
    except (InvalidHandshake, OSError, TimeoutError) as exc:
        raise SpeechError(f"could not connect: {type(exc).__name__}") from exc


class FluxListener:
    """Speech to text for one voice session."""

    def __init__(self, config: SpeechConfig) -> None:
        self._config = config
        self._ws: ClientConnection | None = None
        self._closing = False

    async def open(self) -> None:
        query = [
            ("model", self._config.stt_model),
            ("encoding", "linear16"),
            ("sample_rate", str(self._config.input_rate)),
            *(("keyterm", term) for term in self._config.keyterms),
        ]
        self._ws = await _open(
            f"{self._config.base_url}/v2/listen?{urlencode(query)}", self._config
        )

    async def send(self, audio: bytes) -> None:
        if self._ws is not None and not self._closing:
            with suppress(ConnectionClosed):
                await self._ws.send(audio)

    async def _say(self, message: dict) -> None:
        if self._ws is not None and not self._closing:
            with suppress(ConnectionClosed):
                await self._ws.send(json.dumps(message))

    async def configure(
        self,
        *,
        eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        numerals: bool | None = None,
    ) -> None:
        """Change how the turn in progress and the ones after it are taken. It applies at once."""
        message: dict = {"type": "Configure"}
        thresholds = {
            key: value
            for key, value in (("eot_threshold", eot_threshold), ("eot_timeout_ms", eot_timeout_ms))
            if value is not None
        }
        if thresholds:
            message["thresholds"] = thresholds
        if numerals is not None:
            message["numerals"] = numerals
        await self._say(message)

    async def end_turn(self) -> None:
        """End the customer's turn now, on what has been transcribed so far."""
        await self._say({"type": "ForceEndTurn"})

    async def turns(self) -> AsyncIterator[Turn]:
        """The recogniser's messages until the socket closes. Raises SpeechError if it breaks."""
        assert self._ws is not None, "open() first"
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "TurnInfo":
                    spoken = message.get("words") or []
                    yield Turn(
                        event=message["event"],
                        index=int(message["turn_index"]),
                        transcript=message.get("transcript") or "",
                        confidence=float(message.get("end_of_turn_confidence") or 0.0),
                        window_end=float(message.get("audio_window_end") or 0.0),
                        first_word_start=float(spoken[0]["start"]) if spoken else None,
                        last_word_end=float(spoken[-1]["end"]) if spoken else None,
                        trigger=message.get("trigger"),
                    )
                elif kind == "Error":
                    raise SpeechError(f"speech to text: {message.get('code')}")
                elif kind == "ConfigureFailure":
                    logger.warning("the recogniser refused a setting: %s", message.get("code"))
        except ConnectionClosed as exc:
            if not self._closing:
                raise SpeechError("speech to text closed the connection") from exc

    async def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            with suppress(Exception):
                await self._ws.close()


class FluxSpeaker:
    """Text to speech for one voice session.

    `on_audio` gets each piece of audio as it arrives. `on_spoken` is called when a reply's audio
    has all been sent. `on_failed` is called once if the socket breaks.
    """

    def __init__(
        self,
        config: SpeechConfig,
        *,
        on_audio: Callable[[bytes], None],
        on_spoken: Callable[[Spoken], None],
        on_failed: Callable[[SpeechError], None],
    ) -> None:
        self._config = config
        self._on_audio, self._on_spoken, self._on_failed = on_audio, on_spoken, on_failed
        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._closing = False
        self._discarding = False  # between an Interrupt and its answer: audio already on its way
        self._answer: asyncio.Future[Interrupted | None] | None = None

    async def open(self) -> None:
        query = {
            "model": self._config.tts_model,
            "encoding": "linear16",
            "sample_rate": str(self._config.output_rate),
        }
        self._ws = await _open(f"{self._config.base_url}/v2/speak?{urlencode(query)}", self._config)
        self._reader = asyncio.create_task(self._read())

    async def _say(self, message: dict) -> None:
        if self._ws is None or self._closing:
            return
        try:
            await self._ws.send(json.dumps(message))
        except ConnectionClosed as exc:
            raise SpeechError("text to speech closed the connection") from exc

    async def speak(self, text: str) -> None:
        """Add text to the reply being spoken. The first text starts a reply.

        The service adds no space between two pieces, so the caller keeps the spaces of its text.
        """
        await self._say({"type": "Speak", "text": text})

    async def flush(self) -> None:
        """The reply's text is complete: the rest of its audio is made and the reply ends."""
        await self._say({"type": "Flush"})

    async def interrupt(self, offset_ms: float | None) -> Interrupted | None:
        """Cancel the reply being spoken. Returns what had been heard, or None if nothing was.

        `offset_ms` is how much audio the customer has heard since this socket opened. Without
        it the reply is cancelled all the same, and the service cannot say which words were
        heard.
        """
        if self._ws is None or self._closing:
            return None
        message: dict = {"type": "Interrupt"}
        if offset_ms is not None:
            message["playback_offset"] = {"type": "time_ms", "value": round(offset_ms)}
        self._answer = asyncio.get_running_loop().create_future()
        self._discarding = True
        try:
            await self._say(message)
            return await asyncio.wait_for(self._answer, 2.0)
        except TimeoutError:
            logger.warning("no answer to an interrupt within 2 s")
            return None
        finally:
            self._discarding = False
            self._answer = None

    def _settle(self, result: Interrupted | None) -> None:
        if self._answer is not None and not self._answer.done():
            self._answer.set_result(result)

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    if not self._discarding:
                        self._on_audio(raw)
                    continue
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "SpeechMetadata":
                    self._on_spoken(_spoken(message))
                elif kind == "SpeechInterrupted":
                    self._settle(
                        Interrupted(
                            played_ms=float(message.get("audio_played_ms") or 0.0),
                            text_spoken=message.get("text_spoken"),
                            spoken=_spoken(message.get("metadata") or {}),
                        )
                    )
                elif kind == "Warning":
                    if message.get("code") in _NOT_INTERRUPTED:
                        self._settle(None)
                    else:
                        logger.info("text to speech warned: %s", message.get("code"))
                elif kind == "Error":
                    raise SpeechError(f"text to speech: {message.get('code')}")
            if not self._closing:
                raise SpeechError("text to speech closed the connection")
        except ConnectionClosed:
            if not self._closing:
                self._fail(SpeechError("text to speech closed the connection"))
        except SpeechError as exc:
            self._fail(exc)

    def _fail(self, error: SpeechError) -> None:
        self._settle(None)
        self._on_failed(error)

    async def close(self) -> None:
        self._closing = True
        self._settle(None)
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.send(json.dumps({"type": "Close"}))
            with suppress(Exception):
                await self._ws.close()
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._reader


def _spoken(metadata: dict) -> Spoken:
    return Spoken(
        audio_ms=float(metadata.get("audio_duration_ms") or 0.0),
        chars=int(metadata.get("billable_character_count") or 0),
    )
