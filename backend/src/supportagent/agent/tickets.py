"""Escalation tickets. Tools and the escalate node open them through the same function.

A ticket is read by a person outside this conversation, so everything written into one goes through
the session's redactor first.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from supportagent.agent.context import AgentContext
from supportagent.agent.state import AgentState
from supportagent.crm import repo
from supportagent.events import Ticket
from supportagent.redact import MASK, Redactor

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_ACCOUNT_NUMBER = re.compile(r"(?<!\d)\d{8}(?!\d)")


def last_customer_text(messages: Sequence[AnyMessage]) -> str:
    """The customer's latest message as typed. Mask it before it goes anywhere but a model."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


def question_only(masked: str) -> str:
    """The customer's message without the sentences that only say who they are.

    `masked` has been through the redactor. A sentence that holds a masked PIN or an 8-digit
    account number gives credentials. Next to the question it blurs a search: on the hosted
    reranker the passage that answers "Do you have a SOC 2 Type II report we can get?" scored
    1.33 for the question alone and 0.00 with "My account number is ... and my PIN is ..." in
    front of it. It has no place in a ticket's summary either. A message in which every
    sentence is of that kind is returned whole.
    """
    sentences = [part.strip() for part in _SENTENCE_END.split(masked) if part.strip()]
    kept = [s for s in sentences if MASK not in s and not _ACCOUNT_NUMBER.search(s)]
    return " ".join(kept) if kept else masked


def recent_customer_texts(
    messages: Sequence[AnyMessage], redactor: Redactor, count: int = 3
) -> list[str]:
    """The customer's last few messages, oldest first, with any PIN masked."""
    texts = [str(m.text) for m in messages if isinstance(m, HumanMessage)]
    return [redactor.text(text, customer=True) for text in texts[-count:]]


def transcript_excerpt(
    messages: Sequence[AnyMessage], redactor: Redactor, lines: int = 6, limit: int = 300
) -> str:
    """The last few things the customer and the assistant said, for the person who picks this up."""
    spoken = []
    for message in messages:
        text = str(message.text).strip()
        if not text:
            continue
        if isinstance(message, HumanMessage):
            spoken.append(f"Customer: {redactor.text(text, customer=True)[:limit]}")
        elif isinstance(message, AIMessage):
            spoken.append(f"Assistant: {redactor.text(text)[:limit]}")
    return "\n".join(spoken[-lines:])


async def open_ticket(
    ctx: AgentContext,
    state: AgentState,
    *,
    reason: str,
    priority: str,
    summary: str,
    tool_call_id: str | None = None,
    attribute: bool = True,
) -> Ticket:
    """Store a ticket for a person to pick up and return it.

    `tool_call_id` makes a retried tool call return the ticket it already opened. `attribute`
    false leaves the verified customer off the ticket, for a ticket that is not about them.
    """
    customer = state.get("customer") if attribute else None
    account_number = customer["account_number"] if customer else None
    tier = customer["tier"] if customer else None
    summary = ctx.redactor.text(summary)[:2000]
    async with ctx.pool.connection() as conn:
        ticket_id = await repo.create_ticket(
            conn,
            world_id=ctx.world_id,
            account_number=account_number,
            tier=tier,
            reason=reason,
            priority=priority,
            summary=summary,
            transcript_excerpt=transcript_excerpt(state.get("messages", []), ctx.redactor),
            tool_call_id=tool_call_id,
        )
    return Ticket(
        id=ticket_id,
        reason=reason,
        priority=priority,
        summary=summary,
        account_number=account_number,
        tier=tier,
    )
