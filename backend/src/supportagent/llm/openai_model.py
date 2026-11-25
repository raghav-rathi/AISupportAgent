"""A chat model behind any OpenAI-compatible endpoint: the AI Gateway, a provider, or a local server."""

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
    ) -> None:
        """`name` is the label in events. `model` is the ID the endpoint expects.

        `extra_body` carries provider-specific fields, such as the reasoning effort or the
        gateway's provider order. `json_schema` false is for an endpoint that rejects structured
        output: a caller's schema is then left out and the prompt alone asks for JSON.
        `silence_timeout_s` only stops a dead socket and must be longer than the client's own
        first-token and stall limits.
        """
        self.name = name
        self.price = price
        self._strict_tools = strict_tools
        self._json_schema = json_schema
        self._chat = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=0,  # the resilient client owns retries and deadlines
            timeout=httpx.Timeout(silence_timeout_s, connect=5.0),
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
            shaped = self._chat.bind(
                response_format={"type": "json_schema", "json_schema": json_schema}
            )
            return shaped.astream(list(messages))
        return self._chat.astream(list(messages))
