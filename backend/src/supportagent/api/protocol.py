"""What a client may send over the WebSocket. The server answers with the events in events.py."""

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
    type: Literal["user.message"]
    text: str
    mode: Literal["text", "voice"] = "text"


class TurnRetry(BaseModel):
    """Pick up the turn that stopped early: every model was down, or it was cancelled."""

    type: Literal["turn.retry"]


class TurnCancel(BaseModel):
    """Stop the turn that is running. The server answers with `turn.interrupted`."""

    type: Literal["turn.cancel"]


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
    ]
    tool: str | None = None  # which tool fails, for tool_failure
    times: int = Field(default=1, ge=1, le=10)


ClientMessage = Annotated[
    Hello | UserMessage | TurnRetry | TurnCancel | FaultInject, Field(discriminator="type")
]

CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
