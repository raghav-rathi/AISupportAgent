"""Every event the backend sends to a client.

One ordered stream per session carries chat tokens, tool steps, guardrail decisions, failures and
usage. The chat UI, the voice adapter and the observability drawer all read the same stream. The
session adds `seq`, `ts` and `turn_id` when it publishes an event, after masking any PIN in it.

`session.ready` and `error` answer a client directly. They are not part of the stored stream and
keep `seq` 0.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

Purpose = Literal["agent", "judge"]
FailureReason = Literal[
    "timeout", "rate_limited", "server_error", "connection", "quota", "rejected"
]
# What a knowledge base search means for the turn:
#   answer      the best passage is clearly relevant: answer, and check the answer afterwards
#   check       somewhat relevant: hold the answer until the grounding check has passed
#   verify      a passage for some plan answers this much better than anything an unverified
#               visitor may see: the customer has to verify first
#   restricted  the same for a verified customer: the help center covers this for another plan
#   escalate    the help center has no answer: the turn goes to a person
Band = Literal["answer", "check", "verify", "restricted", "escalate"]
EscalationReason = Literal[
    "low_confidence",
    "ungrounded_answer",
    "refund_over_limit",
    "account_suspended",
    "verification_failed",
    "billing_dispute",
    "security",
    "privacy_request",
    "customer_request",
    "step_limit",
    "other",
]
Priority = Literal["low", "normal", "high", "urgent"]
Tier = Literal["free", "pro", "enterprise"]
AccountStatus = Literal["active", "past_due", "suspended", "cancelled"]


class Ticket(BaseModel):
    id: str
    reason: EscalationReason
    priority: Priority
    summary: str
    account_number: str | None = None
    tier: Tier | None = None


class CustomerSummary(BaseModel):
    account_number: str
    company_name: str
    contact_name: str
    tier: Tier
    status: AccountStatus


class RetrievalHit(BaseModel):
    n: int  # the citation number the model sees, unique within the turn
    chunk_id: str
    source: str  # document title and heading path
    score: float
    tiers: list[Tier]


class TraceMessage(BaseModel):
    """One message of a prompt or a response, with secrets masked."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] = []
    name: str | None = None  # the tool, on a tool result


class _Event(BaseModel):
    seq: int = 0
    ts: float = 0.0
    turn_id: str | None = None


class SessionReady(_Event):
    """The reply to a client's hello. Events after `last_seq` follow if the client missed any."""

    type: Literal["session.ready"] = "session.ready"
    session_id: str
    last_seq: int
    customer: CustomerSummary | None
    tickets: list[Ticket]
    busy: bool  # a turn is running right now
    resumable: bool  # the last turn stopped early and `turn.retry` will pick it up


class ProtocolError(_Event):
    """The client sent something the server could not act on."""

    type: Literal["error"] = "error"
    code: Literal[
        "bad_message",
        "turn_in_progress",
        "nothing_to_retry",
        "nothing_to_cancel",
        "too_long",
        "rate_limited",
        "not_allowed",
    ]
    message: str


class TurnStarted(_Event):
    type: Literal["turn.started"] = "turn.started"
    mode: Literal["text", "voice"]
    text: str  # the customer's message, with any PIN masked


class LlmStarted(_Event):
    type: Literal["llm.started"] = "llm.started"
    call_id: str
    purpose: Purpose
    model: str
    attempt: int


class LlmDelta(_Event):
    type: Literal["llm.delta"] = "llm.delta"
    call_id: str
    text: str


class LlmReset(_Event):
    """Clients drop the text this call has streamed.

    Either the attempt failed after some text went out, or the text came with a tool call
    (`tool_call`) and so is not part of the answer.
    """

    type: Literal["llm.reset"] = "llm.reset"
    call_id: str
    reason: FailureReason | Literal["tool_call"]


class LlmRetry(_Event):
    type: Literal["llm.retry"] = "llm.retry"
    call_id: str
    model: str
    attempt: int  # the attempt about to start
    max_attempts: int
    reason: FailureReason
    wait_ms: int


class LlmFallback(_Event):
    type: Literal["llm.fallback"] = "llm.fallback"
    call_id: str
    from_model: str
    to_model: str
    reason: FailureReason


class LlmFinished(_Event):
    type: Literal["llm.finished"] = "llm.finished"
    call_id: str
    purpose: Purpose
    model: str
    attempts: int
    used_fallback: bool
    ttft_ms: float | None  # from the start of the call, failed attempts included
    total_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None  # None when the model has no price configured


class LlmTrace(_Event):
    """The full prompt and response of one model call, for the observability drawer."""

    type: Literal["llm.trace"] = "llm.trace"
    call_id: str
    purpose: Purpose
    model: str
    prompt: list[TraceMessage]
    tools: list[str]  # names of the tools offered to the model
    response: TraceMessage


class ToolStarted(_Event):
    type: Literal["tool.started"] = "tool.started"
    tool_call_id: str
    name: str
    args: dict[str, Any]  # secrets already masked


class ToolError(_Event):
    """One attempt failed. A `tool.retry` or a `tool.finished` follows."""

    type: Literal["tool.error"] = "tool.error"
    tool_call_id: str
    name: str
    message: str


class ToolRetry(_Event):
    type: Literal["tool.retry"] = "tool.retry"
    tool_call_id: str
    name: str
    attempt: int  # the attempt about to start
    max_attempts: int
    wait_ms: int


class ToolFinished(_Event):
    """`ok` the tool did its job, `refused` a rule stopped it, `failed` it could not run."""

    type: Literal["tool.finished"] = "tool.finished"
    tool_call_id: str
    name: str
    outcome: Literal["ok", "refused", "failed"]
    summary: str
    duration_ms: float


class RetrievalResult(_Event):
    type: Literal["retrieval.result"] = "retrieval.result"
    tool_call_id: str
    query: str
    band: Band
    top_score: float | None  # best passage the customer's plan may see
    plan_score: float | None  # best passage written for other plans. The passage is never sent.
    low: float
    high: float
    hits: list[RetrievalHit]
    embed_ms: float  # the three timings add up every search pass this call made
    search_ms: float
    rerank_ms: float


class GuardrailVerdict(_Event):
    type: Literal["guardrail.verdict"] = "guardrail.verdict"
    verdict: Literal["grounded", "ungrounded", "skipped"]
    held: bool  # the answer waited for this verdict before going out
    unsupported: list[str]
    ms: float


class AnswerRetracted(_Event):
    """A streamed answer did not pass its grounding check. Clients remove it."""

    type: Literal["answer.retracted"] = "answer.retracted"
    reason: str


class AssistantMessage(_Event):
    """The final text of the turn. Clients show it in place of whatever they streamed.

    When `held` is true no deltas went out before it.
    """

    type: Literal["assistant.message"] = "assistant.message"
    text: str
    held: bool
    hits: list[RetrievalHit]  # what the citation numbers in `text` point at


class EscalationCreated(_Event):
    type: Literal["escalation.created"] = "escalation.created"
    ticket: Ticket


class SessionState(_Event):
    type: Literal["session.state"] = "session.state"
    customer: CustomerSummary | None
    tickets: list[Ticket]


class TurnFinished(_Event):
    type: Literal["turn.finished"] = "turn.finished"
    outcome: Literal["answered", "escalated"]
    total_ms: float
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when a model without a configured price took part


class TurnFailed(_Event):
    """The turn could not finish. The session and its history are intact.

    When `retryable` is true, `turn.retry` picks the turn up where it stopped.
    """

    type: Literal["turn.failed"] = "turn.failed"
    code: Literal["llm_unavailable", "internal"]
    message: str
    retryable: bool
    total_ms: float = 0.0


class TurnInterrupted(_Event):
    """The turn was stopped before it finished, by the client or by a server shutdown."""

    type: Literal["turn.interrupted"] = "turn.interrupted"
    total_ms: float


AnyEvent = Annotated[
    SessionReady
    | ProtocolError
    | TurnStarted
    | LlmStarted
    | LlmDelta
    | LlmReset
    | LlmRetry
    | LlmFallback
    | LlmFinished
    | LlmTrace
    | ToolStarted
    | ToolError
    | ToolRetry
    | ToolFinished
    | RetrievalResult
    | GuardrailVerdict
    | AnswerRetracted
    | AssistantMessage
    | EscalationCreated
    | SessionState
    | TurnFinished
    | TurnFailed
    | TurnInterrupted,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)

# The events that end a turn. Exactly one of them closes every turn.
TURN_ENDED = frozenset({"turn.finished", "turn.failed", "turn.interrupted"})
