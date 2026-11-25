"""The adapter over langchain-openai, against a fake endpoint. No real model is involved."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from supportagent.llm.openai_model import OpenAIChatModel


def _chunk(text: str, finish: str | None = None) -> bytes:
    body = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _model(silence_timeout_s: float, pause_s: float) -> OpenAIChatModel:
    """A model whose endpoint sends "Hel", stays silent for `pause_s`, then sends "lo"."""

    async def body():
        yield _chunk("Hel")
        await asyncio.sleep(pause_s)
        yield _chunk("lo", finish="stop")
        yield b"data: [DONE]\n\n"

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    return OpenAIChatModel(
        "m",
        model="m",
        base_url="http://model.test/v1",
        api_key="k",
        silence_timeout_s=silence_timeout_s,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )


async def _text(model: OpenAIChatModel) -> str:
    return "".join([str(chunk.text) async for chunk in model.stream([HumanMessage("hi")], None)])


async def test_the_configured_silence_limit_replaces_the_librarys_own(monkeypatch):
    # langchain-openai ends a stream that is silent for 120 seconds unless it is told otherwise.
    # A local server sends a tool call in one piece, and on a slow machine that takes longer. The
    # variable stands in for the 120 seconds: a model that left the library's limit alone would
    # time out here.
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "0.05")
    assert await _text(_model(silence_timeout_s=5.0, pause_s=0.3)) == "Hello"


async def test_a_stream_silent_for_longer_than_the_limit_ends_in_a_timeout():
    with pytest.raises(TimeoutError):
        await _text(_model(silence_timeout_s=0.05, pause_s=0.5))
