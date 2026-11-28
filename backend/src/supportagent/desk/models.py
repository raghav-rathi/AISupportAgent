"""What the desk sends and takes over its socket, `/desk`.

The desk speaks first, with `desk.hello`. It names a workspace by a key only the browser holds.
A workspace is the conversations that browser started, so one visitor's desk shows the tickets
of that visitor's own conversations and nobody else's.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from supportagent.events import (
    AccountStatus,
    Band,
    EscalationReason,
    Priority,
    TicketStatus,
    Tier,
)

Channel = Literal["text", "voice"]
ActivityKind = Literal["created", "assigned", "status", "priority", "note", "reply", "refund"]
DeskErrorCode = Literal[
    "bad_message", "not_found", "not_allowed", "too_long", "rate_limited", "busy", "refused"
]

_MAX_BODY = 2_000


class DeskTicket(BaseModel):
    """One ticket as the inbox lists it."""

    key: str  # the desk's name for the ticket. It says nothing about its conversation.
    id: str  # ESC-1A2B3C, what the customer was told
    status: TicketStatus
    priority: Priority
    reason: EscalationReason
    summary: str
    channel: Channel
    assignee: str | None
    account_number: str | None
    company: str | None
    contact: str | None
    tier: Tier | None
    created_at: float  # seconds since the epoch
    updated_at: float
    replies: int  # replies a person has sent to the customer


class Activity(BaseModel):
    """One thing that happened to a ticket. `public` is what the customer may read."""

    id: int
    kind: ActivityKind
    author: str
    public: bool
    body: str
    at: float
    meta: dict[str, str]


class ReceiptStep(BaseModel):
    name: str
    outcome: Literal["ok", "refused", "failed"]
    summary: str


class ReceiptSearch(BaseModel):
    query: str
    band: Band
    best: str | None  # the passage that scored highest, if any was returned
    score: float | None


class Receipt(BaseModel):
    """What the assistant did in the turn that opened the ticket, and why it stopped there.

    Built from the stored events of that turn, so it says what happened and not what a model
    says happened. Nothing here is written by a model except `told`.
    """

    asked: str  # the customer's message, as stored: PINs masked
    channel: Channel
    why: str  # one sentence on why this is with a person
    steps: list[ReceiptStep]
    searches: list[ReceiptSearch]
    check: Literal["grounded", "ungrounded", "skipped"] | None
    unsupported: list[str]  # what the grounding check could not find evidence for
    told: str  # what the customer was told at the end of the turn
    # For a spoken reply the customer cut off: the words they heard of it. None when the reply
    # was typed or was heard to its end.
    heard: str | None


class Line(BaseModel):
    """One thing said in the conversation since the hand-off."""

    role: Literal["customer", "assistant"]
    text: str
    mode: Channel
    at: float


class AccountInvoice(BaseModel):
    id: str
    amount_cents: int
    description: str
    issued_on: str
    status: Literal["paid", "open", "refunded"]


class AccountRefund(BaseModel):
    id: str
    invoice_id: str
    amount_cents: int
    reason: str
    refunded_on: str


class AccountChange(BaseModel):
    field: str
    old_value: str
    new_value: str
    at: float


class AccountView(BaseModel):
    """The customer behind a ticket, as the mock CRM of its conversation has them."""

    account_number: str
    company: str
    contact: str
    email: str
    phone: str
    billing_email: str
    tier: Tier
    billing_cycle: Literal["monthly", "annual"] | None
    status: AccountStatus
    seats: int
    customer_since: str
    renews_on: str | None
    invoices: list[AccountInvoice]
    refunds: list[AccountRefund]
    changes: list[AccountChange]


class DeskStats(BaseModel):
    """The workspace in numbers."""

    open: int
    pending: int
    resolved: int
    unassigned: int  # open and nobody's
    conversations: int
    turns: int  # customer messages the assistant finished a turn for
    handed_off: int  # of those, the ones that ended with a person
    first_reply_s: float | None  # median time from a ticket to a person's first reply


# What the server sends.


class DeskReady(BaseModel):
    type: Literal["desk.ready"] = "desk.ready"
    agent: str  # the name this desk acts under, as the server took it
    tickets: list[DeskTicket]
    stats: DeskStats


class DeskTicketChanged(BaseModel):
    """A ticket is new or has changed. Sent to every desk of the workspace."""

    type: Literal["desk.ticket"] = "desk.ticket"
    ticket: DeskTicket
    stats: DeskStats


class DeskDetail(BaseModel):
    type: Literal["desk.detail"] = "desk.detail"
    ticket: DeskTicket
    activity: list[Activity]
    receipt: Receipt | None  # None when the turn that opened the ticket is no longer stored
    before: list[str]  # the last lines of the conversation before the hand-off
    after: list[Line]  # the conversation from the hand-off on
    account: AccountView | None  # None for a ticket no verified customer is on


class DeskError(BaseModel):
    type: Literal["desk.error"] = "desk.error"
    code: DeskErrorCode
    message: str
    key: str | None = None


DeskEvent = Annotated[
    DeskReady | DeskTicketChanged | DeskDetail | DeskError, Field(discriminator="type")
]
DESK_EVENT_ADAPTER: TypeAdapter[DeskEvent] = TypeAdapter(DeskEvent)


# What the desk sends.


class DeskHello(BaseModel):
    """The first message. `workspace` is the key the browser keeps. `agent` is a display name."""

    type: Literal["desk.hello"]
    workspace: str = Field(min_length=16, max_length=64)
    agent: str = Field(default="", max_length=60)


class DeskOpen(BaseModel):
    type: Literal["desk.open"]
    key: str = Field(max_length=64)


class DeskAssign(BaseModel):
    """Give the ticket to someone, or to nobody with `assignee` null."""

    type: Literal["desk.assign"]
    key: str = Field(max_length=64)
    assignee: str | None = Field(default=None, max_length=60)


class DeskSetStatus(BaseModel):
    type: Literal["desk.status"]
    key: str = Field(max_length=64)
    status: TicketStatus


class DeskSetPriority(BaseModel):
    type: Literal["desk.priority"]
    key: str = Field(max_length=64)
    priority: Priority


class DeskNote(BaseModel):
    """A note for colleagues. The customer never sees it."""

    type: Literal["desk.note"]
    key: str = Field(max_length=64)
    body: str = Field(min_length=1, max_length=_MAX_BODY)


class DeskReply(BaseModel):
    """A reply to the customer. It appears in the conversation the ticket came from."""

    type: Literal["desk.reply"]
    key: str = Field(max_length=64)
    body: str = Field(min_length=1, max_length=_MAX_BODY)
    status: TicketStatus | None = None  # set along with the reply, for "reply and resolve"


class DeskRefund(BaseModel):
    """Approve a refund as the billing team: above the support limit, or on a suspended account."""

    type: Literal["desk.refund"]
    key: str = Field(max_length=64)
    invoice_id: str = Field(max_length=32)


DeskMessage = Annotated[
    DeskHello
    | DeskOpen
    | DeskAssign
    | DeskSetStatus
    | DeskSetPriority
    | DeskNote
    | DeskReply
    | DeskRefund,
    Field(discriminator="type"),
]
DESK_CLIENT_ADAPTER: TypeAdapter[DeskMessage] = TypeAdapter(DeskMessage)
