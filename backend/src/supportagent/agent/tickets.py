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
# A comma followed by a space ends a clause. A comma between digits, as in "1,000", does not.
_CLAUSE_END = re.compile(r",\s+")
_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_PIN = re.compile(r"\bPIN\b", re.IGNORECASE)
_ONLY_CREDENTIALS = re.compile(rf"^(?:[\d\s.,:;]|{re.escape(MASK)}|\band\b)+$", re.IGNORECASE)


def _names_credentials(clause: str) -> bool:
    return bool(_ACCOUNT_NUMBER.search(clause) or _PIN.search(clause))


def _gives_credentials(clause: str) -> bool:
    """True for a clause that hands over an account number or a PIN.

    A clause that only mentions a PIN, as in "what if I forget my PIN?", asks something and
    stays. So does one in which a mask hides some other number, as in "ISO ****".
    """
    if _ACCOUNT_NUMBER.search(clause):
        return True
    if _PIN.search(clause) and MASK in clause:
        return True
    return MASK in clause and bool(_ONLY_CREDENTIALS.match(clause.strip()))


def _clauses(sentence: str) -> list[str]:
    """The sentence split at commas, and at "and" where one side names an account number or PIN."""
    clauses = []
    for part in _CLAUSE_END.split(sentence):
        pieces = _AND.split(part)
        if len(pieces) > 1 and any(_names_credentials(p) for p in pieces):
            clauses.extend(pieces)
        else:
            clauses.append(part)
    return [c.strip() for c in clauses if c.strip()]


def _without_credentials(sentence: str) -> str:
    clauses = _clauses(sentence)
    kept = [c for c in clauses if not _gives_credentials(c)]
    if len(kept) == len(clauses):
        return sentence
    text = ", ".join(kept)
    return text[:1].upper() + text[1:]


def last_customer_text(messages: Sequence[AnyMessage]) -> str:
    """The customer's latest message as typed. Mask it before it goes anywhere but a model."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


def question_only(masked: str) -> str:
    """The customer's message without the clauses that only say who they are.

    `masked` has been through the redactor. A clause that holds an 8-digit account number, a
    masked PIN, or nothing but digits and masks gives credentials. Next to the question it blurs
    a search: on the hosted reranker the passage that answers "Do you have a SOC 2 Type II report
    we can get?" scored 1.33 for the question alone and 0.00 with "My account number is ... and
    my PIN is ..." in front of it. It has no place in a ticket's summary either. A mask alone does
    not make a clause a credential: while nobody is verified the redactor also masks numbers such
    as the one in "ISO 27001". A message that holds nothing but credentials is returned whole.
    """
    sentences = [part.strip() for part in _SENTENCE_END.split(masked) if part.strip()]
    kept = [text for text in (_without_credentials(s) for s in sentences) if text]
    return " ".join(kept) if kept else masked


def recent_customer_texts(
    messages: Sequence[AnyMessage], redactor: Redactor, count: int = 3
) -> list[str]:
    """The customer's last few messages, oldest first, with any PIN masked."""
    texts = [str(m.text) for m in messages if isinstance(m, HumanMessage)]
    return [redactor.text(text, customer=True) for text in texts[-count:]]


def transcript_excerpt(
    messages: Sequence[AnyMessage],
    redactor: Redactor,
    lines: int = 6,
    limit: int = 300,
    *,
    current_turn_only: bool = False,
) -> str:
    """The last few things the customer and the assistant said, for the person who picks this up.

    `current_turn_only` starts the excerpt at the customer's latest message. A ticket that is not
    about the verified customer must not carry what they said before.
    """
    if current_turn_only:
        starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        messages = messages[starts[-1] :] if starts else []
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
            transcript_excerpt=transcript_excerpt(
                state.get("messages", []), ctx.redactor, current_turn_only=not attribute
            ),
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
