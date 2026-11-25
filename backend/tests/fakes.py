"""Test doubles for the model layer."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Sequence

import httpx
import openai
from langchain_core.messages import AIMessageChunk, BaseMessage

from supportagent.llm.pricing import Price

_REQUEST = httpx.Request("POST", "https://llm.test/v1/chat/completions")
_call_ids = itertools.count(1)


class ScriptedModel:
    """Plays one scripted behaviour per call and repeats the last one after that.

    A behaviour is a list of steps. A string or an AIMessageChunk is yielded, a float is a pause in
    seconds, and an exception is raised.
    """

    def __init__(self, name: str, *behaviours: list, price: Price | None = None) -> None:
        self.name = name
        self.price = price
        self.behaviours = list(behaviours)
        self.calls: list[tuple[list[BaseMessage], list[str]]] = []
        self.schemas: list[dict | None] = []  # the json_schema of each call

    async def stream(self, messages: Sequence[BaseMessage], tools, json_schema=None):
        index = min(len(self.calls), len(self.behaviours) - 1)
        self.calls.append((list(messages), [t.name for t in tools or []]))
        self.schemas.append(json_schema)
        for step in self.behaviours[index]:
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif isinstance(step, str):
                yield AIMessageChunk(content=step)
            else:
                yield step


def usage(input_tokens: int, output_tokens: int) -> AIMessageChunk:
    """The last chunk of a stream, which carries only token counts."""
    return AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def tool_call(name: str | None, args: str, call_id: str | None = "call-1", index: int = 0):
    """A chunk that carries one tool call, or a piece of one. `args` is JSON text."""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": index}],
    )


def call(name: str, **args) -> list:
    """A whole model turn that asks for one tool, with a fresh call ID."""
    return [tool_call(name, json.dumps(args), f"call-{next(_call_ids)}"), usage(500, 15)]


def calls(*requests: tuple[str, dict]) -> list:
    """A model turn that asks for several tools at once."""
    chunks = [
        tool_call(name, json.dumps(args), f"call-{next(_call_ids)}", index)
        for index, (name, args) in enumerate(requests)
    ]
    return [*chunks, usage(500, 30)]


def say(text: str) -> list:
    """A whole model turn that answers in two chunks."""
    middle = len(text) // 2
    return [text[:middle], text[middle:], usage(600, 30)]


def say_then_call(text: str, name: str, **args) -> list:
    """A model turn that writes some text and then asks for a tool anyway."""
    return [text, tool_call(name, json.dumps(args), f"call-{next(_call_ids)}"), usage(500, 25)]


def http_error(
    status: int,
    *,
    retry_after: str | None = None,
    message: str = "upstream error",
    code: str | None = None,
    headers: dict[str, str] | None = None,
):
    """The exception the OpenAI SDK raises for an HTTP error status."""
    headers = {**(headers or {}), **({"retry-after": retry_after} if retry_after else {})}
    response = httpx.Response(status, headers=headers, request=_REQUEST)
    kind = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        429: openai.RateLimitError,
    }.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return kind(message, response=response, body={"message": message, "code": code})


def stream_error(message: str = "The server had an error", code: str | None = "server_error"):
    """What the SDK raises when a stream that began with HTTP 200 carries an error event."""
    return openai.APIError(message, _REQUEST, body={"message": message, "code": code})


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)
