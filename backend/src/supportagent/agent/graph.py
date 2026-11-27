"""The agent loop.

    begin -> agent -> tools -> agent -> ... -> check -> end
                        |                        |
                        +------> escalate <------+

- `begin` resets the turn's working memory and repairs what an unfinished turn left behind.
- `agent` calls the model. It may answer or ask for one tool.
- `tools` runs that tool through the guard.
- After a knowledge base search that found nothing relevant, the turn goes straight to `escalate`.
  The model gets no chance to answer from memory.
- `check` runs the grounding check on the answer. Only an answer that passed is sent as the final
  message, stored, and shown in a prompt trace. One that fails is taken back and the turn goes to
  `escalate`. One that cannot be checked is taken back and the turn fails, so it can be retried.
- `escalate` opens a ticket and replies with fixed wording. No model is involved.
"""

from __future__ import annotations

import json
import re
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
from supportagent.agent.tools import CAPABILITIES, TOOLS, was_forgotten
from supportagent.crm import policy
from supportagent.events import (
    AnswerRetracted,
    AssistantMessage,
    EscalationCreated,
    GuardrailVerdict,
    LlmFinished,
    LlmReset,
    LlmTrace,
    RetrievalHit,
    Ticket,
    TraceMessage,
)
from supportagent.llm.client import Emit
from supportagent.llm.errors import Failure, LLMUnavailable
from supportagent.llm.trace import trace_prompt
from supportagent.redact import scrub_messages

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

# How much evidence the judge reads. Everything this turn's tools returned goes in, up to a cap
# that only several searches in one turn can reach. Results of earlier turns have a smaller
# budget of their own.
_EVIDENCE_RESULTS = 12
_EVIDENCE_CHARS = 16_000
_TURN_EVIDENCE_CHARS = 48_000
# What a prompt trace shows in place of text that was never approved for the customer.
_WITHHELD = "[withheld: this answer did not pass its grounding check]"
_DROPPED = "[dropped: text the model sent along with a tool call]"
_REPLY_IN_JUDGE_REQUEST = re.compile(r"(<reply-[0-9a-f]+>\n).*?(\n</reply-[0-9a-f]+>)", re.DOTALL)


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
    if not messages or not isinstance(messages[-1], HumanMessage):
        return []
    *earlier, new = messages
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
                # What the provider attached to the calls goes with them: its own raw copy,
                # a signature that vouches for calls no longer there, calls it could not parse.
                message = message.model_copy(
                    update={
                        "tool_calls": calls,
                        "content": "" if calls else _INTERRUPTED,
                        "invalid_tool_calls": [],
                        "additional_kwargs": {},
                    }
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
        "unchecked_call": None,
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
    traces: list[LlmTrace] = []

    def emit_but_keep_traces(event: BaseModel) -> None:
        # What this call wrote may be an answer nobody has checked yet. Its trace waits until
        # that is known.
        if isinstance(event, LlmTrace):
            traces.append(event)
        else:
            emit(event)

    result = await ctx.llm.complete(
        prompt, tools=TOOLS, emit=emit_but_keep_traces, stream_text=streamed, faults=ctx.faults
    )
    message = result.message
    if message.id is None:
        message = message.model_copy(update={"id": uuid.uuid4().hex})
    if not message.tool_calls:
        # The check node writes this call's trace once it has a verdict on the answer.
        waiting = {"call_id": result.call_id, "model": result.model}
        return {
            "messages": [message],
            "steps": steps,
            "unchecked": message.id,
            "unchecked_call": waiting,
        }

    if streamed and str(message.text):
        # Text that comes with a tool call is not the answer and never reaches the check.
        # Clients drop it, and it is not kept.
        emit(LlmReset(call_id=result.call_id, reason="tool_call"))
    for trace in traces:
        said = trace.response.content
        emit(trace.model_copy(update={"response": trace.response.model_copy(update={"content": _DROPPED if said.strip() else ""})}))  # fmt: skip
    if steps >= ctx.max_steps:
        # Running this tool would leave no step to tell the customer what it did.
        update: dict[str, Any] = {"steps": steps}
        dropped = message.tool_calls[0]
        if dropped["name"] == "verify_identity":
            # The call will never run, so nothing else would take the PIN out of the conversation.
            for key, value in (dropped.get("args") or {}).items():
                if key != "account_number":
                    ctx.redactor.learn(value)
            scrubbed = scrub_messages(state["messages"], ctx.redactor)
            if scrubbed:
                update["messages"] = scrubbed
        asked = recent_customer_texts(state["messages"], ctx.redactor, count=1)
        update["pending_escalation"] = Escalation(
            reason="step_limit",
            priority="normal",
            summary=f"The assistant did not finish within {ctx.max_steps} steps. "
            f"Last customer message: {asked[-1] if asked else ''}",
        )
        return update
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


def _as_evidence(message: ToolMessage) -> str:
    """One tool result as the judge reads it.

    A call that did not run gives its error code and nothing else. Its message is partly built
    from what the model sent, and nothing the model or the customer wrote may stand as evidence.
    """
    if message.status != "error":
        return f"tool: {message.name}\n{message.content}"
    try:
        code = str(json.loads(str(message.content)).get("error", "failed"))
    except (ValueError, AttributeError):
        code = "failed"
    return f"tool: {message.name}\n" + json.dumps({"error": code, "note": "This call did not run."})


def _who(state: AgentState) -> str:
    """Who the assistant is talking to, as far as it has been established."""
    customer = state.get("customer")
    if customer:
        return "Verified customer: " + json.dumps(customer)
    if state.get("failed_verifications", 0) >= policy.MAX_VERIFICATION_ATTEMPTS:
        return (
            "Verification state: locked. Identity verification failed too often in this "
            "conversation, and the assistant cannot read or change any account in it."
        )
    return (
        "Verification state: not verified. The assistant cannot read or change an account "
        "until verify_identity has succeeded with an account number and a PIN."
    )


def evidence_texts(state: AgentState) -> list[str]:
    """What the judge may count as support: what the tools can do, who the customer is, and tool
    results, oldest first, each named by its tool.

    Everything this turn's tools returned comes first and goes in whole. An answer may cite any
    of it, and leaving a cited passage out would get a correct answer taken back. Of earlier
    turns, the latest reading of the account and the results of actions fill what space is left.
    Earlier searches stay out: the model no longer sees them either, and their citation numbers
    belong to another turn. So do results of an account that is no longer the verified one.
    """
    messages = state["messages"]
    start = _turn_start(messages)
    results = [
        (i, m)
        for i, m in enumerate(messages)
        if isinstance(m, ToolMessage) and not was_forgotten(m)
    ]
    kept: list[tuple[int, str]] = []
    size = 0
    for index, message in reversed([pair for pair in results if pair[0] > start]):
        text = _as_evidence(message)
        if kept and size + len(text) > _TURN_EVIDENCE_CHARS:
            continue  # only an older search of a turn with several can be this large
        kept.append((index, text))
        size += len(text)
    older = [
        pair
        for pair in reversed(results)
        if pair[0] < start and pair[1].name != "search_knowledge_base" and pair[1].status != "error"
    ]
    account = [pair for pair in older if pair[1].name == "get_account"][:1]
    # Earlier turns have a budget of their own, so a turn with large results does not push the
    # account reading out.
    earlier = earlier_size = 0
    for index, message in account + [pair for pair in older if pair not in account]:
        text = _as_evidence(message)
        if earlier < _EVIDENCE_RESULTS and earlier_size + len(text) <= _EVIDENCE_CHARS:
            kept.append((index, text))
            earlier += 1
            earlier_size += len(text)
    return [CAPABILITIES, _who(state), *(text for _, text in sorted(kept))]


def _without_reply(trace: LlmTrace) -> LlmTrace:
    """The judge's trace with the reply it was shown taken out of its prompt."""
    prompt = [
        message.model_copy(
            update={
                "content": _REPLY_IN_JUDGE_REQUEST.sub(
                    lambda found: f"{found.group(1)}reply> {_WITHHELD}{found.group(2)}",
                    message.content,
                )
            }
        )
        for message in trace.prompt
    ]
    return trace.model_copy(update={"prompt": prompt})


async def check(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Run the grounding check on the answer the agent node wrote.

    Nothing says "this is the answer" before the verdict. An answer on a confident search has
    streamed already, as text in flight that the client still marks as unconfirmed. The final
    message, the stored copy and the prompt traces all come after the verdict, so a turn that is
    stopped during the check leaves no unchecked answer behind, and a rejected answer's text is
    in no trace.
    """
    ctx, emit = runtime.context, _emitter(runtime)
    answer = state["messages"][-1]
    text = str(answer.text).strip()
    held = _held(state)
    hits = [RetrievalHit(**hit) for record in state.get("evidence") or [] for hit in record["hits"]]
    final = AssistantMessage(text=text, held=held, hits=hits)
    asked = recent_customer_texts(state["messages"], ctx.redactor)
    done = {"unchecked": None, "unchecked_call": None}

    def trace_answer(approved: bool) -> None:
        """The trace of the agent call that wrote this answer, rebuilt from the saved state."""
        call = state.get("unchecked_call")
        if not call or not ctx.llm.capture_prompts:
            return
        sent = [
            SystemMessage(system_prompt(state.get("customer"), ctx.mode)),
            *for_prompt(state["messages"][:-1]),
        ]
        prompt, tool_names = trace_prompt(sent, TOOLS)
        emit(
            LlmTrace(
                call_id=call["call_id"],
                purpose="agent",
                model=call["model"],
                prompt=prompt,
                tools=tool_names,
                response=TraceMessage(role="assistant", content=text if approved else _WITHHELD),
            )
        )

    def take_back(why: str) -> dict[str, Any]:
        if not held:
            emit(AnswerRetracted(reason=why))
        return {
            **done,
            "messages": [RemoveMessage(id=answer.id)],
            "pending_escalation": Escalation(
                reason="ungrounded_answer",
                priority="normal",
                summary=f"The assistant's answer failed its grounding check ({why}). "
                f"Customer question: {asked[-1] if asked else ''}",
            ),
        }

    if not text:
        trace_answer(approved=True)  # nothing to withhold
        return take_back("empty answer")
    if ctx.judge is None:
        # No judge is configured, which only a development setup may do. The answer goes out
        # unchecked and the stream says so.
        emit(GuardrailVerdict(verdict="skipped", held=held, unsupported=[], ms=0.0))
        trace_answer(approved=True)
        emit(final)
        return done

    judge_traces: list[LlmTrace] = []

    def emit_but_keep_traces(event: BaseModel) -> None:
        # The judge's prompt quotes the answer. Whether that may be shown depends on its verdict.
        if isinstance(event, LlmTrace):
            judge_traces.append(event)
        else:
            emit(event)

    started = time.perf_counter()
    try:
        verdict = await ctx.judge.check(
            question=asked[-1] if asked else "",
            earlier=asked[:-1],
            answer=text,
            evidence=evidence_texts(state),
            emit=emit_but_keep_traces,
            faults=ctx.faults,
        )
    except (LLMUnavailable, JudgeError) as exc:
        # No verdict, so no answer: what streamed comes off the screen and the turn fails in a
        # way the customer can retry. The answer stays in state as unchecked. A retry runs this
        # check again, and a new message instead removes the answer.
        ms = (time.perf_counter() - started) * 1000
        emit(GuardrailVerdict(verdict="skipped", held=held, unsupported=[], ms=ms))
        for trace in judge_traces:
            emit(_without_reply(trace))
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
    for trace in judge_traces:
        emit(trace if verdict.grounded else _without_reply(trace))
    trace_answer(approved=verdict.grounded)
    if not verdict.grounded:
        return take_back("; ".join(verdict.unsupported) or "unsupported statements")
    emit(final)
    return done


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
    # Whatever ends the turn here, an action that went through stays done. The customer has to
    # hear of it: a refund that was issued and never mentioned would be asked for again.
    done = _done_this_turn(state)
    if done:
        text += " Done so far: " + "; ".join(done) + "."
    runtime.stream_writer(AssistantMessage(text=text, held=True, hits=[]))
    return {
        "messages": [AIMessage(text)],
        "tickets": new_tickets,
        "pending_escalation": None,
        "outcome": "escalated",
        "unchecked": None,
        "unchecked_call": None,
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
