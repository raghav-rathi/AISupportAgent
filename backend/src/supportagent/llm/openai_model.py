"""A chat model behind any OpenAI-compatible endpoint: OpenRouter, a provider, or a local server."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from supportagent.llm.pricing import Price


class OpenAIChatModel:
    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        strict_tools: bool = False,
        json_schema: bool = True,
        extra_body: dict[str, Any] | None = None,
        price: Price | None = None,
        silence_timeout_s: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """`name` is the label in events. `model` is the ID the endpoint expects.

        `extra_body` carries provider-specific fields, such as the reasoning effort or the
        provider routing OpenRouter should follow. `json_schema` false is for an endpoint that rejects structured
        output: a caller's schema is then left out and the prompt alone asks for JSON.
        `silence_timeout_s` only stops a dead connection and must be longer than the client's own
        first-token and stall limits. It is set twice: as the socket's read limit, and as the
        library's own limit on the gap between two chunks. That one is 120 seconds unless it is
        set, which is shorter than a slow local model takes to write one tool call.
        """
        self.name = name
        self.price = price
        self._strict_tools = strict_tools
        self._json_schema = json_schema
        self._extra_body = extra_body or {}
        self._chat = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=0,  # the resilient client owns retries and deadlines
            timeout=httpx.Timeout(silence_timeout_s, connect=5.0),
            stream_chunk_timeout=silence_timeout_s,
            http_async_client=http_client,
            stream_usage=True,
            extra_body=extra_body,
        )

    def stream(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[AIMessageChunk]:
        if tools:
            # One tool call per step: the graph relies on each result before choosing the next.
            bound = self._chat.bind_tools(
                list(tools), parallel_tool_calls=False, strict=self._strict_tools or None
            )
            return bound.astream(list(messages))
        if json_schema and self._json_schema:
            # The schema goes in the request body. Given as `response_format`, langchain-openai
            # switches to the SDK's parsing stream, which drops the token usage, so the judge's
            # calls would report no cost.
            format_ = {"type": "json_schema", "json_schema": json_schema}
            shaped = self._chat.bind(extra_body={**self._extra_body, "response_format": format_})
            return shaped.astream(list(messages))
        return self._chat.astream(list(messages))
