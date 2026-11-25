"""The agent loop.

    begin -> agent -> tools -> agent -> ... -> check -> end
                        |                        |
                        +------> escalate <------+

- `begin` resets the turn's working memory and repairs what an unfinished turn left behind.
- `agent` calls the model. It may answer or ask for one tool.
- `tools` runs that tool through the guard.
- After a knowledge base search that found nothing relevant, the turn goes straight to `escalate`.
  The model gets no chance to answer from memory.
- `check` runs the grounding check on the answer. An answer that fails is taken back and the turn
  goes to `escalate`. An answer that cannot be checked is taken back and the turn fails, so it can
  be retried.
- `escalate` opens a ticket and replies with fixed wording. No model is involved.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel

from supportagent.agent.context import AgentContext
from supportagent.agent.guard import guard
from supportagent.agent.judge import JudgeError
from supportagent.agent.prompts import for_prompt, system_prompt
from supportagent.agent.state import AgentState, Escalation
from supportagent.agent.tickets import open_ticket, recent_customer_texts
from supportagent.agent.tools import CAPABILITIES, TOOLS
from supportagent.events import (
    AnswerRetracted,
    AssistantMessage,
    EscalationCreated,
    GuardrailVerdict,
    LlmFinished,
    LlmReset,
    RetrievalHit,
    Ticket,
)
from supportagent.llm.client import Emit
from supportagent.llm.errors import Failure, LLMUnavailable

_HAND_OFF = "I've passed your question to a specialist as ticket {ticket}. They will follow up."
_ESCALATION_TEXT = {
    "low_confidence": "I can't answer that from the Fernhill help center, and I won't guess. "
    + _HAND_OFF,
    "ungrounded_answer": "I need to take that back. I couldn't confirm it against the Fernhill "
    "help center. " + _HAND_OFF,
    "step_limit": "This needs more steps than I can take here, so I've passed it to a specialist "
    "as ticket {ticket}. They will follow up.",
}
# For an answer taken back in a turn that had already opened a ticket.
_ALREADY_WITH_A_PERSON = (
    "I need to take that back. I couldn't confirm it against the Fernhill help center. "
    "Your case is already with a specialist as ticket {ticket}. They will follow up."
)
# What the model reads in place of a reply the customer never got. The customer does not see it.
_INTERRUPTED = "(The previous request was interrupted before I could reply.)"
# Tools that change something. If a turn runs out of steps, the customer is told which of them ran.
_ACTIONS = frozenset({"issue_refund", "update_account_field"})

# How much evidence the judge reads: at most this many tool results and this many characters.
_EVIDENCE_RESULTS = 12
_EVIDENCE_CHARS = 16_000


def _emitter(runtime: Runtime[AgentContext]) -> Emit:
    """Send events to the stream and keep the turn's usage totals."""
    usage = runtime.context.usage

    def emit(event: BaseModel) -> None:
        if isinstance(event, LlmFinished):
            usage.add(event)
        runtime.stream_writer(event)

    return emit


def _held(state: AgentState) -> bool:
    """True when a search this turn landed between the thresholds: the answer waits for the check."""
    return any(record["band"] == "check" for record in state.get("evidence") or [])


def _turn_start(messages: Sequence[AnyMessage]) -> int:
    """Index of the customer message that opened the current turn."""
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)


def repair(messages: Sequence[AnyMessage], unchecked: str | None) -> list[AnyMessage]:
    """Updates that make the stored conversation valid again when a new turn starts.

    A turn can stop anywhere: every model down, a crash, a request cancelled by the customer. It
    may leave a tool call without a result, which hosted models reject, a tool result or a
    question with no reply after it, or an answer that never passed its grounding check and that
    the customer never got. `messages` ends with the new customer message.
    """
    *earlier, new = messages
    if not isinstance(new, HumanMessage):
        return []
    updates: list[AnyMessage] = []
    answered = {m.tool_call_id for m in earlier if isinstance(m, ToolMessage)}
    last: AnyMessage | None = None  # the message before the new one, once the updates apply
    for message in earlier:
        if unchecked and message.id == unchecked:
            updates.append(RemoveMessage(id=message.id))
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = [call for call in message.tool_calls if call["id"] in answered]
            if len(calls) != len(message.tool_calls):
                message = message.model_copy(
                    update={"tool_calls": calls, "content": "" if calls else _INTERRUPTED}
                )
                updates.append(message)
        last = message
    if last is not None and not (isinstance(last, AIMessage) and not last.tool_calls):
        # The turn before this one ended without a reply. The note goes where the reply would be,
        # which means taking the new message out and putting it back after the note.
        updates += [RemoveMessage(id=new.id), AIMessage(_INTERRUPTED), HumanMessage(new.content)]
    return updates


async def begin(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "steps": 0,
        "evidence": None,
        "pending_escalation": None,
        "outcome": "answered",
        "tickets_before": len(state.get("tickets") or []),
        "unchecked": None,
    }
    repairs = repair(state.get("messages") or [], state.get("unchecked"))
    if repairs:
        update["messages"] = repairs
    return update


async def agent(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx, emit = runtime.context, _emitter(runtime)
    steps = state.get("steps", 0) + 1
    streamed = not _held(state)
    prompt = [
        SystemMessage(system_prompt(state.get("customer"), ctx.mode)),
        *for_prompt(state["messages"]),
    ]
    result = await ctx.llm.complete(
        prompt, tools=TOOLS, emit=emit, stream_text=streamed, faults=ctx.faults
    )
    message = result.message
    if message.id is None:
        message = message.model_copy(update={"id": uuid.uuid4().hex})
    if not message.tool_calls:
        return {"messages": [message], "steps": steps, "unchecked": message.id}

    if streamed and str(message.text).strip():
        # Text that comes with a tool call is not the answer and never reaches the check.
        # Clients drop it, and it is not kept.
        emit(LlmReset(call_id=result.call_id, reason="tool_call"))
    if steps >= ctx.max_steps:
        # Running this tool would leave no step to tell the customer what it did.
        asked = recent_customer_texts(state["messages"], ctx.redactor, count=1)
        return {
            "steps": steps,
            "pending_escalation": Escalation(
                reason="step_limit",
                priority="normal",
                summary=f"The assistant did not finish within {ctx.max_steps} steps. "
                f"Last customer message: {asked[-1] if asked else ''}",
            ),
        }
    # One tool call per step: the model sees each result before it chooses the next call. The
    # provider's raw copy of the calls goes too, so a dropped call cannot come back from it.
    extra = {k: v for k, v in message.additional_kwargs.items() if k != "tool_calls"}
    message = message.model_copy(
        update={"content": "", "tool_calls": message.tool_calls[:1], "additional_kwargs": extra}
    )
    return {"messages": [message], "steps": steps}


def after_agent(state: AgentState) -> Literal["tools", "check", "escalate"]:
    if state.get("pending_escalation"):
        return "escalate"
    return "tools" if state["messages"][-1].tool_calls else "check"


def after_tools(state: AgentState) -> Literal["agent", "escalate"]:
    return "escalate" if state.get("pending_escalation") else "agent"


def evidence_texts(state: AgentState) -> list[str]:
    """What the judge may count as support: what the tools can do, who the customer is, and tool
    results, oldest first, each named by its tool.

    This turn's results come first when space is short, then the latest reading of the account,
    then older results. One that does not fit is skipped and smaller ones after it still go in.
    """
    messages = state["messages"]
    start = _turn_start(messages)
    results = [(i, m) for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    this_turn = [pair for pair in reversed(results) if pair[0] > start]
    older = [pair for pair in reversed(results) if pair[0] < start]
    account = [pair for pair in older if pair[1].name == "get_account"][:1]
    ordered = this_turn + account + [pair for pair in older if pair not in account]
    kept: list[tuple[int, str]] = []
    size = 0
    for index, message in ordered:
        text = f"tool: {message.name}\n{message.content}"
        if len(kept) < _EVIDENCE_RESULTS and size + len(text) <= _EVIDENCE_CHARS:
            kept.append((index, text))
            size += len(text)
    standing = [CAPABILITIES]
    customer = state.get("customer")
    if customer:
        standing.append("Verified customer: " + json.dumps(customer))
    return [*standing, *(text for _, text in sorted(kept))]


async def check(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx, emit = runtime.context, _emitter(runtime)
    answer = state["messages"][-1]
    text = str(answer.text).strip()
    held = _held(state)
    hits = [RetrievalHit(**hit) for record in state.get("evidence") or [] for hit in record["hits"]]
    final = AssistantMessage(text=text, held=held, hits=hits)
    asked = recent_customer_texts(state["messages"], ctx.redactor)

    def take_back(why: str) -> dict[str, Any]:
        if not held:
            emit(AnswerRetracted(reason=why))
        return {
            "messages": [RemoveMessage(id=answer.id)],
            "unchecked": None,
            "pending_escalation": Escalation(
                reason="ungrounded_answer",
                priority="normal",
                summary=f"The assistant's answer failed its grounding check ({why}). "
                f"Customer question: {asked[-1] if asked else ''}",
            ),
        }

    if not text:
        return take_back("empty answer")
    if not held:
        emit(final)  # the text already streamed; this marks it complete
    if ctx.judge is None:
        # No judge is configured, which only a development setup may do. The answer goes out
        # unchecked and the stream says so.
        emit(GuardrailVerdict(verdict="skipped", held=held, unsupported=[], ms=0.0))
        if held:
            emit(final)
        return {"unchecked": None}

    started = time.perf_counter()
    try:
        verdict = await ctx.judge.check(
            question=asked[-1] if asked else "",
            earlier=asked[:-1],
            answer=text,
            evidence=evidence_texts(state),
            emit=emit,
            faults=ctx.faults,
        )
    except (LLMUnavailable, JudgeError) as exc:
        # No verdict, so no answer: what streamed comes off the screen and the turn fails in a
        # way the customer can retry. The answer stays in state as unchecked. A retry runs this
        # check again, and a new message instead removes the answer.
        ms = (time.perf_counter() - started) * 1000
        emit(GuardrailVerdict(verdict="skipped", held=held, unsupported=[], ms=ms))
        if not held:
            emit(AnswerRetracted(reason="the grounding check could not run"))
        if isinstance(exc, JudgeError):
            failure = Failure("server_error", retryable=True, detail=f"judge: {exc}")
            raise LLMUnavailable([failure]) from exc
        raise

    ms = (time.perf_counter() - started) * 1000
    emit(
        GuardrailVerdict(
            verdict="grounded" if verdict.grounded else "ungrounded",
            held=held,
            unsupported=verdict.unsupported,
            ms=ms,
        )
    )
    if not verdict.grounded:
        return take_back("; ".join(verdict.unsupported) or "unsupported statements")
    if held:
        emit(final)
    return {"unchecked": None}


def after_check(state: AgentState) -> Literal["escalate", "__end__"]:
    return "escalate" if state.get("pending_escalation") else END


def _done_this_turn(state: AgentState) -> list[str]:
    """What the turn's actions achieved, in the words their tools reported."""
    messages = state["messages"]
    return [
        message.artifact["summary"]
        for message in messages[_turn_start(messages) :]
        if isinstance(message, ToolMessage)
        and message.name in _ACTIONS
        and (message.artifact or {}).get("outcome") == "ok"
    ]


async def escalate(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    request = state["pending_escalation"]
    reason = request["reason"]
    opened = (state.get("tickets") or [])[state.get("tickets_before", 0) :]
    new_tickets: list[dict[str, Any]] = []
    if reason == "ungrounded_answer" and opened:
        # A tool already handed this turn to a person. A second ticket would only be noise.
        text = _ALREADY_WITH_A_PERSON.format(ticket=Ticket(**opened[-1]).id)
    else:
        ticket = await open_ticket(
            runtime.context,
            state,
            reason=reason,
            priority=request["priority"],
            summary=request["summary"],
        )
        runtime.stream_writer(EscalationCreated(ticket=ticket))
        new_tickets = [ticket.model_dump()]
        text = _ESCALATION_TEXT[reason].format(ticket=ticket.id)
        done = _done_this_turn(state) if reason == "step_limit" else []
        if done:
            text += " Done so far: " + "; ".join(done) + "."
    runtime.stream_writer(AssistantMessage(text=text, held=True, hits=[]))
    return {
        "messages": [AIMessage(text)],
        "tickets": new_tickets,
        "pending_escalation": None,
        "outcome": "escalated",
        "unchecked": None,
    }


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("begin", begin)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS, awrap_tool_call=guard, handle_tool_errors=False))
    builder.add_node("check", check)
    builder.add_node("escalate", escalate)
    builder.add_edge(START, "begin")
    builder.add_edge("begin", "agent")
    builder.add_conditional_edges("agent", after_agent)
    builder.add_conditional_edges("tools", after_tools)
    builder.add_conditional_edges("check", after_check)
    builder.add_edge("escalate", END)
    return builder.compile(checkpointer=checkpointer)


def user_turn(text: str) -> dict[str, Any]:
    """The graph input for one customer message."""
    return {"messages": [HumanMessage(text)]}
