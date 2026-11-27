"""Every event the backend sends to a client.

One ordered stream per session carries chat tokens, tool steps, guardrail decisions, failures and
usage. The chat UI, the voice adapter and the observability drawer all read the same stream. The
session adds `seq`, `ts` and `turn_id` when it publishes an event, after masking any PIN in it.

`session.ready`, `session.caught_up`, `error`, `trace.list` and `passage.detail` answer a client
directly. They are not part of the stored stream and keep `seq` 0. So do the voice messages that
only the connection with the microphone needs: `voice.started`, `voice.stopped`, `voice.state`,
`voice.playback` and `stt.partial`. Audio itself travels as binary frames next to these, in both
directions.

`llm.trace` is stored and numbered but not streamed: a prompt trace is most of a turn's bytes and
only the observability drawer reads it, so a client asks for the traces of a turn when it
shows them.
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
    "unanswered_question",
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
    fault_injection: bool = False  # this server acts on `fault.inject`, so a UI may offer it
    voice: bool = False  # this server can listen and speak, so a UI may offer the microphone


class SessionCaughtUp(_Event):
    """Everything the client had missed has been sent. What follows happens now."""

    type: Literal["session.caught_up"] = "session.caught_up"


class ProtocolError(_Event):
    """The client sent something the server could not act on.

    `client_id` is set when the error answers one `user.message`: that message started no turn.
    """

    type: Literal["error"] = "error"
    code: Literal[
        "bad_message",
        "turn_in_progress",
        "nothing_to_retry",
        "nothing_to_cancel",
        "too_long",
        "rate_limited",
        "not_allowed",
        "too_many_retries",
        "budget_exhausted",
        "failed_to_start",
        "session_expired",
        "not_found",
        "voice_unavailable",
    ]
    message: str
    client_id: str | None = None


class TurnStarted(_Event):
    type: Literal["turn.started"] = "turn.started"
    mode: Literal["text", "voice"]
    text: str  # the customer's message, with any PIN masked
    client_id: str | None = None  # the ID the client gave this message, so it knows it arrived


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
    (`tool_call`) and so is not part of the answer, or the turn was stopped while the text was
    still unconfirmed (`cancelled`).
    """

    type: Literal["llm.reset"] = "llm.reset"
    call_id: str
    reason: FailureReason | Literal["tool_call", "cancelled"]


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
    """The full prompt and response of one model call, for the observability drawer.

    Stored, not streamed. An answer that has not passed its grounding check is not in here: the
    trace of the call that wrote it is made after the verdict, and a rejected answer's text is
    replaced by a note.
    """

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
    # What an action did, in fields a UI can lay out without reading the model's prose. Set by
    # issue_refund when the refund went through.
    receipt: dict[str, str] | None = None


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
    # What a hosted embedder and reranker charged for this search. None when they did not say
    # or when both run in this process.
    cost_usd: float | None = None


class GuardrailVerdict(_Event):
    type: Literal["guardrail.verdict"] = "guardrail.verdict"
    verdict: Literal["grounded", "ungrounded", "skipped"]
    held: bool  # the answer waited for this verdict before going out
    unsupported: list[str]
    ms: float


class AnswerDrafted(_Event):
    """The model's step ended without a tool call, so what it wrote is the turn's answer, pending
    its grounding check. The text itself is not in here.

    `held` says whether the answer waits for the verdict before anyone gets it. Voice starts
    to speak an answer that is not held at this point, and never earlier: text that turns out
    to come with a tool call is not an answer and must not be heard.
    """

    type: Literal["answer.drafted"] = "answer.drafted"
    call_id: str
    held: bool


class AnswerRetracted(_Event):
    """A streamed answer did not pass its grounding check. Clients remove it."""

    type: Literal["answer.retracted"] = "answer.retracted"
    reason: str


class AssistantMessage(_Event):
    """The final text of the turn. Clients show it in place of whatever they streamed.

    It is sent only after the grounding check has passed, or for wording the app wrote itself.
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
    # What the customer waited for, in milliseconds from their message. Time between a failed
    # attempt and its retry is not counted.
    first_step_ms: float | None = None  # the first sign of work: a tool step or streamed text
    first_text_ms: float | None = None  # the first text of the answer that stood
    answer_ms: float | None = None  # the complete answer, after its grounding check


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
    # False when the turn was stopped before anything of it was saved: there is nothing to pick
    # up, and the message has to be sent again.
    resumable: bool = True


VoiceStatus = Literal["listening", "hearing", "thinking", "speaking"]


class SpeechHeard(_Event):
    """How a spoken message was taken. Follows the `turn.started` of a voice turn."""

    type: Literal["speech.heard"] = "speech.heard"
    speech_ms: float | None  # from the first word to the last
    # From the end of the last word until the recogniser said the turn was over, on our clock.
    end_of_turn_ms: float | None
    confidence: float  # the recogniser's own, between 0 and 1
    trigger: str  # model, timeout or manual: what ended the turn
    digit_mode: bool  # the recogniser was waiting out pauses, for an account number or a PIN
    listened_s: float  # microphone audio sent to the recogniser since the last count
    cost_usd: float


class SpeechSpoken(_Event):
    """A reply as audio, once all of it has been played or the customer cut it off."""

    type: Literal["speech.spoken"] = "speech.spoken"
    chars: int  # what the synthesis is billed for
    audio_ms: float  # audio that was made for this reply
    synthesis_ms: float | None  # from the text going out to the first audio coming back
    # From the end of the customer's last word to the first audio on its way to them.
    voice_to_voice_ms: float | None
    interrupted: bool
    played_ms: float | None  # how much of the audio the customer heard, when interrupted
    # What the customer heard before they interrupted. The rest of the reply was not said.
    text_spoken: str | None = None
    cost_usd: float


class VoiceUsage(_Event):
    """Stored when voice stops: the microphone time no turn has accounted for yet."""

    type: Literal["voice.usage"] = "voice.usage"
    listened_s: float
    cost_usd: float


class VoiceStarted(_Event):
    """The reply to `voice.start`. Send microphone audio now, as binary frames."""

    type: Literal["voice.started"] = "voice.started"
    input_rate: int  # what the server expects: mono PCM16 at this rate
    output_rate: int  # what the server sends: mono PCM16 at this rate


class VoiceStopped(_Event):
    """Voice is off. The conversation goes on in text."""

    type: Literal["voice.stopped"] = "voice.stopped"
    reason: Literal["client", "idle", "limit", "replaced", "error"]
    message: str = ""


class VoiceState(_Event):
    type: Literal["voice.state"] = "voice.state"
    state: VoiceStatus


class VoicePlayback(_Event):
    """What the client has to do with the audio it is playing.

    `pause` when the customer may be starting to speak, `resume` when that was nothing, `flush`
    when what is buffered must not be heard. The client answers each with a `voice.playhead`.
    """

    type: Literal["voice.playback"] = "voice.playback"
    action: Literal["pause", "resume", "flush"]


class SttPartial(_Event):
    """The customer's words so far in the turn they are speaking, with any PIN masked."""

    type: Literal["stt.partial"] = "stt.partial"
    text: str


class TraceList(_Event):
    """The reply to `trace.get`: the prompt traces of one turn, in order. `turn_id` names the turn."""

    type: Literal["trace.list"] = "trace.list"
    traces: list[LlmTrace]


class PassageDetail(_Event):
    """The reply to `passage.get`: one help center passage this conversation was shown."""

    type: Literal["passage.detail"] = "passage.detail"
    chunk_id: str
    source: str  # document title and heading path
    tiers: list[Tier]
    content: str


AnyEvent = Annotated[
    SessionReady
    | SessionCaughtUp
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
    | AnswerDrafted
    | AnswerRetracted
    | AssistantMessage
    | EscalationCreated
    | SessionState
    | TurnFinished
    | TurnFailed
    | TurnInterrupted
    | TraceList
    | PassageDetail
    | SpeechHeard
    | SpeechSpoken
    | VoiceUsage
    | VoiceStarted
    | VoiceStopped
    | VoiceState
    | VoicePlayback
    | SttPartial,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)

# The events that end a turn. Exactly one of them closes every turn.
TURN_ENDED = frozenset({"turn.finished", "turn.failed", "turn.interrupted"})
# Stored and numbered, and sent only when a client asks for them.
NOT_STREAMED = frozenset({"llm.trace"})
