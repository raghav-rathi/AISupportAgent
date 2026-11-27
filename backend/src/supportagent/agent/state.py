"""Session state: what the graph carries between steps and the checkpointer saves between turns."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Customer(TypedDict):
    """Set by verify_identity. Tools read the plan and the account from here."""

    account_number: str
    company_name: str
    contact_name: str
    tier: str
    status: str


class Evidence(TypedDict):
    """One knowledge base search of the current turn."""

    tool_call_id: str
    query: str  # with any PIN masked
    band: str  # answer | check | verify | restricted | escalate
    top_score: float | None
    hits: list[dict[str, Any]]  # the passages the model was given, as RetrievalHit dicts


class Escalation(TypedDict):
    """A hand-off the graph has decided on and the escalate node will carry out."""

    reason: str
    priority: str
    summary: str
    # For an answer that failed its check: the customer got none of it, so there is nothing
    # to take back, only an answer that is not given.
    unseen: NotRequired[bool]


def _append_or_reset(current: list | None, update: list | None) -> list:
    """List reducer: a list appends, None clears."""
    return [] if update is None else [*(current or []), *update]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer: Customer | None
    # Wrong PINs in this session. It never goes back down: a correct PIN for one account must not
    # buy fresh guesses at another.
    failed_verifications: int
    tickets: Annotated[list[dict[str, Any]], operator.add]  # Ticket as dicts, whole session

    # Working memory of the current turn. The begin node resets it.
    steps: int
    evidence: Annotated[list[Evidence], _append_or_reset]
    pending_escalation: Escalation | None
    outcome: str  # answered | escalated
    tickets_before: int  # how many tickets the session had when the turn began
    # ID of an answer that has not passed its grounding check yet. If the turn stops before the
    # check, the next turn removes that answer from the history.
    unchecked: str | None
    # The model call that wrote that answer: its call ID and model, for the prompt trace that is
    # written once the answer has a verdict.
    unchecked_call: dict[str, str] | None
    # What the customer had heard of the last reply when they interrupted it by voice. It comes
    # in with the next message, and the begin node uses it once and clears it.
    heard: str | None
