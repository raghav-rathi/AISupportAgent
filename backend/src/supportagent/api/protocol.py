"""What a client may send over the WebSocket. The server answers with the events in events.py.

Everything here is a JSON text frame. A binary frame is microphone audio, and the server takes it
only between `voice.start` and `voice.stop`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class Hello(BaseModel):
    """The first message on every connection.

    Without `session_id` the server starts a new session. With one, it resumes that session and
    sends the events after `last_seq`. The ID is the only key to a session, so treat it as a secret.
    """

    type: Literal["hello"]
    session_id: str | None = Field(default=None, max_length=64)
    last_seq: int = Field(default=0, ge=0)


class UserMessage(BaseModel):
    """One customer message.

    The server answers each with exactly one of two things: a `turn.started` that carries the
    same `client_id`, or an `error` that carries it. A client that sees neither after a
    reconnect knows the message never arrived.
    """

    type: Literal["user.message"]
    text: str
    mode: Literal["text", "voice"] = "text"
    client_id: str | None = Field(default=None, max_length=64)


class TurnRetry(BaseModel):
    """Pick up the turn that stopped early: every model was down, or it was cancelled."""

    type: Literal["turn.retry"]


class TurnCancel(BaseModel):
    """Stop the turn that is running. The server answers with `turn.interrupted`."""

    type: Literal["turn.cancel"]


class TraceGet(BaseModel):
    """Ask for the prompt traces of one turn. The server answers with `trace.list`."""

    type: Literal["trace.get"]
    turn_id: str = Field(max_length=64)


class PassageGet(BaseModel):
    """Ask for the text of a help center passage. The server answers with `passage.detail`, and
    only for a passage that a search of this conversation handed to the model."""

    type: Literal["passage.get"]
    chunk_id: str = Field(max_length=200)


class FaultInject(BaseModel):
    """Queue a failure for the next model or tool calls of this session, to show how it recovers.

    The server acts on it only when fault injection is switched on in its settings.
    """

    type: Literal["fault.inject"]
    kind: Literal[
        "llm_timeout",
        "llm_rate_limit",
        "llm_server_error",
        "llm_primary_down",
        "llm_down",
        "judge_down",
        "tool_failure",
        "speech_down",
    ]
    tool: str | None = None  # which tool fails, for tool_failure
    times: int = Field(default=1, ge=1, le=10)


class VoiceStart(BaseModel):
    """Turn voice on for this connection. The server answers with `voice.started` or an `error`.

    After that the client sends mono PCM16 at 16 kHz as binary frames, 80 ms each, and gets the
    reply's audio back the same way. A session has one voice connection: starting voice on
    another one stops the first.
    """

    type: Literal["voice.start"]
    # What the microphone track reports. Without echo cancellation the microphone hears the
    # reply, so speech during a reply cannot be taken as an interruption, and the customer
    # interrupts by hand.
    echo_cancellation: bool = True


class VoiceStop(BaseModel):
    type: Literal["voice.stop"]


class VoicePlayhead(BaseModel):
    """How much of the audio the server sent has been played, counted from `voice.started`.

    Sent a few times a second while audio plays, and at once after a pause, a flush or the end
    of the audio. The server needs it to know what the customer heard before an interruption.
    """

    type: Literal["voice.playhead"]
    played_ms: float = Field(ge=0)
    buffered_ms: float = Field(default=0, ge=0)
    # What the report follows: the regular tick, a pause or a flush the server asked for, or
    # the last of the audio having been played.
    after: Literal["tick", "pause", "flush", "drained"] = "tick"


class VoiceInterrupt(BaseModel):
    """The customer stopped the reply by hand. The client has dropped what it had buffered."""

    type: Literal["voice.interrupt"]
    played_ms: float = Field(ge=0)


ClientMessage = Annotated[
    Hello
    | UserMessage
    | TurnRetry
    | TurnCancel
    | FaultInject
    | TraceGet
    | PassageGet
    | VoiceStart
    | VoiceStop
    | VoicePlayhead
    | VoiceInterrupt,
    Field(discriminator="type"),
]

CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
