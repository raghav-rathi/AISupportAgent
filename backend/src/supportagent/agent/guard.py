"""Runs around every tool call: start and finish events, a timeout, retries, and a last resort.

ToolNode calls `guard` for each tool it executes, so the events the UI shows as step indicators
come from the execution itself and cannot drift from what ran.

The guard lets no exception out except a cancellation. Every tool call gets a result message, even
when the tool broke. A call left without a result would stay in the saved conversation that way,
and hosted models reject a history with an unanswered tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import psycopg
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.types import Command
from psycopg_pool import PoolTimeout

from supportagent.agent.context import TransientToolError
from supportagent.events import ToolError, ToolFinished, ToolRetry, ToolStarted
from supportagent.kb.hosted import HostedModelError
from supportagent.redact import scrub_messages

logger = logging.getLogger(__name__)

# Failures that may pass on a second try: a simulated fault, a slow tool, the database, and a
# hosted embedding or reranking model.
_RETRYABLE = (
    TransientToolError,
    TimeoutError,
    psycopg.OperationalError,
    PoolTimeout,
    HostedModelError,
)
_GIVE_UP = (
    "Do not call it again in this turn. Tell the customer it did not go through and offer to "
    "hand the conversation to a person."
)


def _failed(request: ToolCallRequest, payload: dict[str, str]) -> ToolMessage | Command:
    """The result of a call that did not run: an error message for the model."""
    call = request.tool_call
    message = ToolMessage(
        json.dumps(payload), tool_call_id=call["id"], name=call["name"], status="error"
    )
    if call["name"] != "verify_identity":
        return message
    # The call carried a PIN, and the tool did not get to clear it from the conversation.
    stored = (request.state or {}).get("messages", [])
    replaced = scrub_messages(stored, request.runtime.context.redactor)
    return Command(update={"messages": [*replaced, message]}) if replaced else message


def _what_was_wrong(exc: ToolInvocationError) -> str:
    """Why the arguments were refused, without the arguments.

    The library's own message quotes every argument the model sent. Those are the model's words,
    often the customer's: a PIN under the wrong name, or text laid out to pass for a policy. The
    reply goes into the conversation and from there to the grounding judge, so it names the
    fields and the rule each one broke and nothing else.
    """
    errors = exc.filtered_errors
    if errors is None:
        errors = exc.source.errors(include_input=False, include_url=False, include_context=False)
    problems = [
        f"{'.'.join(str(part) for part in error.get('loc', ())) or 'arguments'}: "
        f"{error.get('msg', 'not valid')}"
        for error in errors
    ]
    return "; ".join(problems)[:500] or "the arguments are not valid"


def _artifact(result: ToolMessage | Command, call_id: str) -> dict[str, Any]:
    """The outcome, the summary and any receipt a tool attached to its message.

    A tool's update can hold other tool messages too: copies of earlier results that it
    replaces, as verify_identity does when the customer changes. The call's own is the one
    that answers it.
    """
    message: Any = result
    if isinstance(result, Command):
        messages = (result.update or {}).get("messages") or []
        message = next(
            (m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id),
            None,
        )
    return getattr(message, "artifact", None) or {}


async def guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    ctx, emit = request.runtime.context, request.runtime.stream_writer
    call = request.tool_call
    name, call_id = call["name"], call["id"]
    args = call.get("args") or {}
    started = time.perf_counter()
    ctx.usage.tool_calls += 1
    if name == "verify_identity":
        # From here on the PIN is masked wherever it shows, whether or not the tool gets to run.
        # Every argument but the account number counts: a model that names the field `code`
        # has still been handed the PIN.
        for key, value in args.items():
            if key != "account_number":
                ctx.redactor.learn(value)
    emit(ToolStarted(tool_call_id=call_id, name=name, args=ctx.redactor.data(args)))

    def finished(outcome: str, summary: str, receipt: dict[str, str] | None = None) -> None:
        emit(
            ToolFinished(
                tool_call_id=call_id,
                name=name,
                outcome=outcome,
                summary=summary,
                duration_ms=(time.perf_counter() - started) * 1000,
                receipt=receipt,
            )
        )

    if request.tool is None:
        finished("failed", "Unknown tool")
        return await handler(request)  # the engine replies with the list of valid tools

    for attempt in range(1, ctx.tool_attempts + 1):
        try:
            async with asyncio.timeout(ctx.tool_timeout_s):
                ctx.tool_faults.check(name)
                result = await handler(request)
        except ToolInvocationError as exc:
            # The model sent arguments the tool cannot accept. It gets the reason and may try again.
            finished("failed", "Invalid arguments")
            reason = ctx.redactor.text(_what_was_wrong(exc))
            return _failed(request, {"error": "invalid_arguments", "message": reason})
        except _RETRYABLE as exc:
            emit(ToolError(tool_call_id=call_id, name=name, message=str(exc) or type(exc).__name__))
            if attempt == ctx.tool_attempts:
                break
            wait = ctx.tool_backoff_s * 2 ** (attempt - 1)
            emit(
                ToolRetry(
                    tool_call_id=call_id,
                    name=name,
                    attempt=attempt + 1,
                    max_attempts=ctx.tool_attempts,
                    wait_ms=round(wait * 1000),
                )
            )
            await asyncio.sleep(wait)
            continue
        except Exception as exc:
            # A bug or a failure nobody planned for. The traceback is for us. The model and the
            # customer learn only that the step did not go through.
            logger.exception("tool %s failed", name)
            emit(ToolError(tool_call_id=call_id, name=name, message=type(exc).__name__))
            finished("failed", f"{name} hit an internal error")
            return _failed(
                request,
                {
                    "error": "tool_failed",
                    "message": f"{name} could not run because of an internal error. {_GIVE_UP}",
                },
            )
        artifact = _artifact(result, call_id)
        finished(
            artifact.get("outcome", "ok"), artifact.get("summary", ""), artifact.get("receipt")
        )
        return result

    finished("failed", f"{name} is unavailable")
    return _failed(
        request,
        {
            "error": "tool_unavailable",
            "message": f"{name} failed {ctx.tool_attempts} times. {_GIVE_UP}",
        },
    )
