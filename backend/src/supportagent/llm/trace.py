"""Turns a model call into a trace: the prompt and the response as plain messages.

Nothing is masked here. A trace is an event like any other, and the session passes every event
through its redactor before the event is sent or stored. The redactor has to see the PIN in a
verify_identity call to find the same PIN in the customer's message.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from supportagent.events import TraceMessage


def trace_message(message: BaseMessage) -> TraceMessage:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "tool"
    return TraceMessage(
        role=role,
        content=str(message.text),
        tool_calls=[
            {"id": call["id"], "name": call["name"], "args": call["args"]}
            for call in getattr(message, "tool_calls", [])
        ],
        name=message.name if role == "tool" else None,
    )


def trace_prompt(
    messages: Sequence[BaseMessage], tools: Sequence[BaseTool] | None
) -> tuple[list[TraceMessage], list[str]]:
    return [trace_message(m) for m in messages], [tool.name for tool in tools or []]
