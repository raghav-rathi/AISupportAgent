"""The seven tools the agent can call.

Rules that hold for every tool:

- The customer and the plan come from session state, never from the model's arguments.
- A tool that reads or changes an account refuses until verify_identity has succeeded.
- Policy is enforced here, in code. When policy sends a case to a person, the tool opens the ticket.
- Each tool returns a Command. Its message carries a short summary for the UI in `artifact`.

This module does not use postponed annotations: the tool decorator reads the real types.
"""

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from supportagent import clock
from supportagent.agent.context import AgentContext
from supportagent.agent.state import (
    NO_LONGER_VERIFIED,
    AgentState,
    Customer,
    Escalation,
    Evidence,
)
from supportagent.agent.tickets import open_question, open_ticket
from supportagent.crm import policy, repo
from supportagent.desk import store as desk
from supportagent.events import (
    Band,
    CustomerSummary,
    EscalationCreated,
    RetrievalHit,
    RetrievalResult,
    SessionState,
    Ticket,
)
from supportagent.kb.retrieval import SearchResult
from supportagent.redact import MASK, digits_of, scrub_messages

# The tool decorator builds a pydantic schema from the signature. Naming AgentContext and
# AgentState here would make pydantic build validators for the pool, the models and the whole
# message history. `Any` passes them through untouched, and each tool types its own locals.
Runtime = ToolRuntime[Any, Any]

# Tools whose results hold one customer's data. When that customer stops being the verified
# one, the results are taken out of the conversation the model and the judge read.
_ACCOUNT_TOOLS = frozenset(
    {
        "verify_identity",
        "get_account",
        "issue_refund",
        "update_account_field",
        "get_ticket_status",
    }
)
# How much a search result is worth to the turn. The search tool keeps the best of its wordings.
_WORTH: dict[str, int] = {"answer": 3, "check": 2, "verify": 1, "restricted": 1, "escalate": 0}

_REFUND_TICKET_REASON = {
    "over_limit": "refund_over_limit",
    "account_suspended": "account_suspended",
    "free_plan": "billing_dispute",
}


def _reply(
    runtime: Runtime,
    payload: dict[str, Any],
    *,
    summary: str,
    outcome: str = "ok",
    update: dict[str, Any] | None = None,
    replace: Sequence[AnyMessage] = (),
    receipt: dict[str, str] | None = None,
) -> Command:
    """Build the tool's result: JSON for the model, a summary for the UI, and any state changes.

    `replace` holds copies of stored messages. Each takes the place of the original with its ID.
    `receipt` is what an action did, in fields a UI can show without reading the model's reply.
    """
    artifact: dict[str, Any] = {"outcome": outcome, "summary": summary}
    if receipt:
        artifact["receipt"] = receipt
    message = ToolMessage(
        json.dumps(payload, default=str), tool_call_id=runtime.tool_call_id, artifact=artifact
    )
    return Command(update={**(update or {}), "messages": [*replace, message]})


def _not_verified(runtime: Runtime) -> Command:
    return _reply(
        runtime,
        {
            "error": "verification_required",
            "message": "Verify the customer's identity with verify_identity first.",
        },
        outcome="refused",
        summary="Blocked: identity not verified",
    )


async def _hand_off(
    runtime: Runtime, *, reason: str, priority: str, summary: str, attribute: bool = True
) -> Ticket:
    ticket = await open_ticket(
        runtime.context,
        runtime.state,
        reason=reason,
        priority=priority,
        summary=summary,
        tool_call_id=runtime.tool_call_id,
        attribute=attribute,
    )
    runtime.stream_writer(EscalationCreated(ticket=ticket))
    return ticket


def _ticket_update(ticket: Ticket) -> dict[str, Any]:
    return {"tickets": [ticket.model_dump()], "outcome": "escalated"}


def _ticket_payload(ticket: Ticket, team: str | None = None) -> dict[str, Any]:
    """A ticket as the model and the grounding judge read it: spelled out, nothing to infer."""
    payload = {
        "id": ticket.id,
        "created": True,
        "priority": ticket.priority,
        "follow_up": "A specialist will follow up with the customer about this ticket.",
    }
    return {**payload, "team": team} if team else payload


def _forget_account(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Copies of the account results so far with their content taken out.

    For the moment the verified customer changes or is locked out. Whoever is typing now may be
    someone else, and the old results would still feed the model's prompt and the judge's
    evidence. What the assistant already said stays: it was said to a verified customer.
    """
    forgotten = []
    for message in messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name not in _ACCOUNT_TOOLS
            or message.content == NO_LONGER_VERIFIED
        ):
            continue
        artifact = dict(message.artifact or {})
        if message.name in ("verify_identity", "get_account"):
            artifact["summary"] = "Account reading, no longer available"  # it named the company
        forgotten.append(
            message.model_copy(update={"content": NO_LONGER_VERIFIED, "artifact": artifact})
        )
    return forgotten


def _session_state(state: AgentState, customer: Customer | None, *new: Ticket) -> SessionState:
    return SessionState(
        customer=CustomerSummary(**customer) if customer else None,
        tickets=[*(Ticket(**t) for t in state.get("tickets", [])), *new],
    )


@tool
async def verify_identity(account_number: str, pin: str, runtime: Runtime) -> Command:
    """Check a customer's 8-digit account number and PIN.

    Returns the customer's plan and contract status when they match. Call it before any tool that
    reads or changes an account. Ask the customer for both values. Never guess them.
    """
    ctx: AgentContext = runtime.context
    state: AgentState = runtime.state
    failed = state.get("failed_verifications", 0)
    # This call is the last use of the PIN. Whatever the outcome, the stored conversation loses it.
    ctx.redactor.learn(pin)
    scrubbed = scrub_messages(state.get("messages", []), ctx.redactor)
    locked = {
        "verified": False,
        "locked": True,
        "message": "Verification is locked for this conversation after too many failed attempts.",
    }
    if failed >= policy.MAX_VERIFICATION_ATTEMPTS:
        return _reply(
            runtime, locked, outcome="refused", summary="Verification locked", replace=scrubbed
        )

    account = re.sub(r"\D", "", account_number)
    if not account or not digits_of(pin):
        # A call without an account number or without a PIN checks nothing. It is no guess and
        # costs no attempt. A model makes such a call now and then for a customer who is
        # verified already, with the PIN as the conversation holds it by then, which is a mask.
        # Counted as wrong PINs, three of those calls locked a verified customer out.
        verified = state.get("customer")
        if verified:
            return _reply(
                runtime,
                {
                    "verified": True,
                    "message": "The customer is verified already in this conversation. Do not "
                    "ask for the account number or the PIN again. Use get_account to read "
                    "their account.",
                },
                summary=f"Already verified: {verified['company_name']}",
                replace=scrubbed,
            )
        return _reply(
            runtime,
            {
                "verified": False,
                "message": "No account number and PIN were given. Ask the customer for both.",
            },
            outcome="refused",
            summary="Nothing to check: account number or PIN missing",
            replace=scrubbed,
        )
    async with ctx.pool.connection() as conn:
        record = await repo.verify_pin(conn, ctx.world_id, account, digits_of(pin))

    if record is None:
        failed += 1
        left = policy.MAX_VERIFICATION_ATTEMPTS - failed
        if left > 0:
            return _reply(
                runtime,
                {"verified": False, "attempts_left": left},
                outcome="refused",
                summary=f"No match, {left} attempts left",
                update={"failed_verifications": failed},
                replace=scrubbed,
            )
        # Locked. A customer verified earlier in this conversation is no longer trusted either,
        # and the ticket is not put on their account: whoever is guessing may be someone else.
        ticket = await _hand_off(
            runtime,
            reason="verification_failed",
            priority="normal",
            summary=f"Identity verification failed {failed} times in one conversation. "
            f"Last account number tried: {account or 'none'}.",
            attribute=False,
        )
        ctx.redactor.strict = True
        runtime.stream_writer(_session_state(state, None, ticket))
        return _reply(
            runtime,
            {**locked, "ticket": _ticket_payload(ticket)},
            outcome="refused",
            summary=f"Verification locked, ticket {ticket.id}",
            update={"failed_verifications": failed, "customer": None, **_ticket_update(ticket)},
            replace=[*scrubbed, *_forget_account(state.get("messages", []))],
        )

    customer = Customer(
        account_number=record.account_number,
        company_name=record.company_name,
        contact_name=record.contact_name,
        tier=record.tier,
        status=record.status,
    )
    ctx.redactor.strict = False
    before = state.get("customer")
    if before and before["account_number"] != record.account_number:
        # Another account now. What the tools returned about the first one goes.
        scrubbed = [*scrubbed, *_forget_account(state.get("messages", []))]
    runtime.stream_writer(_session_state(state, customer))
    return _reply(
        runtime,
        {
            "verified": True,
            "customer": {
                "contact_name": record.contact_name,
                "company": record.company_name,
                "plan": record.tier,
                "contract_status": record.status,
            },
        },
        summary=f"Verified {record.company_name}: {record.tier} plan, {record.status}",
        # The count of wrong PINs stays as it is. A correct PIN does not hand out new guesses.
        update={"customer": customer},
        replace=scrubbed,
    )


@tool
async def get_account(runtime: Runtime) -> Command:
    """Read the verified customer's account: plan, contract status, seats, renewal date and the
    most recent invoices with their IDs, amounts, dates and status.
    """
    ctx: AgentContext = runtime.context
    customer: Customer | None = runtime.state.get("customer")
    if customer is None:
        return _not_verified(runtime)
    async with ctx.pool.connection() as conn:
        record = await repo.get_customer(conn, ctx.world_id, customer["account_number"])
        invoices = await repo.list_invoices(conn, ctx.world_id, customer["account_number"])
    if record is None:
        return _reply(
            runtime,
            {"error": "account_not_found", "message": "This account no longer exists."},
            outcome="failed",
            summary="Account not found",
        )
    today = clock.today()
    payload = {
        "account": {
            "company": record.company_name,
            "contact_name": record.contact_name,
            "plan": record.tier,
            "billing_cycle": record.billing_cycle,
            "contract_status": record.status,
            "seats": record.seats,
            "billing_email": record.billing_email,
            "customer_since": record.customer_since,
            "renews_on": record.renews_on,
        },
        "invoices": [
            {
                "id": invoice.id,
                "amount": policy.usd(invoice.amount_cents),
                "description": invoice.description,
                "issued_on": invoice.issued_on,
                "age_days": (today - invoice.issued_on).days,
                "status": invoice.status,
            }
            for invoice in invoices
        ],
    }
    summary = (
        f"{record.company_name}: {record.tier} plan, {record.status}, {len(invoices)} invoices"
    )
    return _reply(runtime, payload, summary=summary)


def _band(ctx: AgentContext, result: SearchResult, tier: str | None) -> Band:
    return ctx.gate.decide(result.top_score, result.plan_score, verified=tier is not None)


def _plans(tiers: Sequence[str]) -> str:
    """ "the Pro plan", "the Pro and Enterprise plans"."""
    names = [tier.capitalize() for tier in policy.TIERS if tier in tiers]
    if not names:
        return "another plan"
    return f"the {' and '.join(names)} plan{'s' if len(names) > 1 else ''}"


def _covering_plans(ctx: AgentContext, result: SearchResult) -> tuple[str, ...]:
    """Every plan that has a confident match, not only the plan of the single best passage.

    A topic often has one section per plan. Naming only the best-scoring one tells a Free
    customer "Pro only" about something Enterprise has too.
    """
    confident = {
        tier for score, tiers in result.plan_hits if score >= ctx.gate.high for tier in tiers
    }
    return tuple(tier for tier in policy.TIERS if tier in (confident or set(result.plan_tiers)))


def _searchable(text: str) -> str:
    """Text as it may leave for a search: one line, and nothing where a PIN was masked."""
    return " ".join(text.replace("\x00", " ").replace(MASK, " ").split())


@tool
async def search_knowledge_base(query: str, runtime: Runtime) -> Command:
    """Search the Fernhill help center: plans, pricing, billing, refunds, security, integrations,
    the API, storage, apps and troubleshooting.

    Results are limited to what applies to the customer's plan. Call this before answering any
    question about Fernhill, and answer only from what it returns.
    """
    ctx: AgentContext = runtime.context
    state: AgentState = runtime.state
    customer = state.get("customer")
    tier = customer["tier"] if customer else None
    # Both wordings are masked before they go anywhere. A hosted embedder or reranker is a third
    # party, and the customer's message may still hold the PIN they typed next to the question.
    # The customer's own words are searched without the clauses that give credentials, and
    # when the latest message gave nothing else, the question is the one before it.
    question = _searchable(open_question(state.get("messages", []), ctx.redactor))
    query = _searchable(ctx.redactor.text(query, customer=True))
    # The model's wording first. If it does not find an answer, the customer's own words get
    # one try, and the better of the two results stands.
    wordings = list({text.lower(): text for text in (query, question) if text}.values())
    if not wordings:
        return _reply(
            runtime,
            {"error": "empty_query", "message": "Say what to search for."},
            outcome="refused",
            summary="Empty query",
        )

    result: SearchResult | None = None
    band: Band = "escalate"
    embed_ms = search_ms = rerank_ms = 0.0
    cost_usd: float | None = None
    for wording in wordings:
        found = await ctx.retriever.search(ctx.pool, wording, tier)
        embed_ms += found.embed_ms
        search_ms += found.search_ms
        rerank_ms += found.rerank_ms
        if found.cost_usd is not None:
            cost_usd = (cost_usd or 0.0) + found.cost_usd
            ctx.usage.add_search(found.cost_usd)
        found_band = _band(ctx, found, tier)
        if result is None or _WORTH[found_band] > _WORTH[band]:
            result, band, query = found, found_band, wording
        if _WORTH[band] >= _WORTH["check"]:
            break

    usable = band in ("answer", "check")
    # Citation numbers continue across the searches of one turn, so each points at one passage.
    offset = sum(len(record["hits"]) for record in state.get("evidence") or []) if usable else 0
    hits = [
        RetrievalHit(
            n=n,
            chunk_id=hit.chunk_id,
            source=" > ".join((hit.doc_title, *hit.heading_path)),
            score=hit.score,
            tiers=list(hit.tiers),
        )
        for n, hit in enumerate(result.hits, start=offset + 1)
    ]
    masked_query = query  # masked above, before it was searched
    runtime.stream_writer(
        RetrievalResult(
            tool_call_id=runtime.tool_call_id,
            query=masked_query,
            band=band,
            top_score=result.top_score,
            plan_score=result.plan_score,
            low=ctx.gate.low,
            high=ctx.gate.high,
            hits=hits,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            cost_usd=cost_usd,
        )
    )
    # Only passages the model was given count as sources of the answer.
    evidence = Evidence(
        tool_call_id=runtime.tool_call_id,
        query=masked_query,
        band=band,
        top_score=result.top_score,
        hits=[hit.model_dump() for hit in hits] if usable else [],
    )

    if band == "verify":
        # A passage for some plan answers this and nothing public does. Only that one fact leaves
        # the search: the passage itself is never returned.
        if state.get("failed_verifications", 0) >= policy.MAX_VERIFICATION_ATTEMPTS:
            message = (
                "The answer depends on the customer's plan, and verification is locked for this "
                "conversation. Say so and offer to pass the question to a person."
            )
        else:
            message = (
                "The answer depends on the customer's plan. Ask for their account number and "
                "PIN, call verify_identity, then search again. If they have no account or cannot "
                "verify, say that the answer depends on the plan and offer to pass the question "
                "to a person."
            )
        return _reply(
            runtime,
            {"results": [], "message": message},
            outcome="refused",
            summary="Depends on the plan: verification needed",
            update={"evidence": [evidence]},
        )
    if band == "restricted":
        covering = _covering_plans(ctx, result)
        plans = _plans(covering)
        return _reply(
            runtime,
            {
                "results": [],
                "covered_for_plans": list(covering),
                "customer_plan": tier,
                "message": f"The help center covers this for {plans} only. It is not "
                f"part of the customer's {tier.capitalize()} plan. Tell them that and nothing "
                "more about it. Offer to pass the question to a person if they want details.",
            },
            outcome="refused",
            summary=f"Covered for {plans} only",
            update={"evidence": [evidence]},
        )
    if band == "escalate":
        # Nothing in the help center answers this. The graph hands the turn to a person.
        asked = question or query
        return _reply(
            runtime,
            {"results": [], "message": "The help center has nothing relevant to this question."},
            summary="No relevant help center content",
            update={
                "evidence": [evidence],
                "pending_escalation": Escalation(
                    reason="low_confidence",
                    priority="normal",
                    summary=f"The help center has no answer to: {asked}",
                ),
            },
        )
    payload = {
        "results": [
            {"n": listed.n, "source": listed.source, "content": hit.content}
            for listed, hit in zip(hits, result.hits, strict=True)
        ]
    }
    return _reply(
        runtime,
        payload,
        summary=f"{len(hits)} passages, best: {hits[0].source}",
        update={"evidence": [evidence]},
    )


@tool
async def issue_refund(
    invoice_id: str,
    reason: Literal["duplicate_charge", "billing_error", "service_issue", "changed_mind", "other"],
    runtime: Runtime,
) -> Command:
    """Refund one invoice in full to the payment method that paid it.

    The refund policy is enforced here. The result says whether the refund went through, was
    refused, or went to the billing team as a ticket. Get the invoice ID from get_account.
    """
    ctx: AgentContext = runtime.context
    customer: Customer | None = runtime.state.get("customer")
    if customer is None:
        return _not_verified(runtime)
    digits = re.sub(r"\D", "", invoice_id)
    invoice = f"INV-{digits}" if digits else invoice_id.strip().upper()[:40]

    async with ctx.pool.connection() as conn:
        outcome = await repo.refund_invoice(
            conn,
            world_id=ctx.world_id,
            account_number=customer["account_number"],
            invoice_id=invoice,
            reason=reason,
            tool_call_id=runtime.tool_call_id,
            today=clock.today(),
        )
    decision = outcome.decision
    if decision.allowed:
        amount = policy.usd(outcome.amount_cents)
        return _reply(
            runtime,
            {
                "status": "refunded",
                "refund_id": outcome.refund_id,
                "invoice_id": invoice,
                "amount": amount,
            },
            summary=f"Refunded {amount} for {invoice} as {outcome.refund_id}",
            receipt={
                "kind": "refund",
                "refund_id": outcome.refund_id,
                "invoice_id": invoice,
                "amount": amount,
                "date": clock.today().isoformat(),
            },
        )

    payload = {
        "status": "refused",
        "code": decision.code,
        "invoice_id": invoice,
        "message": decision.message,
    }
    if not decision.escalate:
        return _reply(runtime, payload, outcome="refused", summary=f"Refused: {decision.code}")
    ticket = await _hand_off(
        runtime,
        reason=_REFUND_TICKET_REASON.get(decision.code, "billing_dispute"),
        priority="high",
        summary=f"Refund request for {invoice}. {decision.message}",
    )
    payload["ticket"] = _ticket_payload(ticket, team="billing")
    return _reply(
        runtime,
        payload,
        outcome="refused",
        summary=f"Refused: {decision.code}, ticket {ticket.id}",
        update=_ticket_update(ticket),
    )


@tool
async def update_account_field(
    field: Literal["billing_email", "contact_name", "contact_phone", "marketing_opt_in"],
    value: str,
    runtime: Runtime,
) -> Command:
    """Change one detail on the verified customer's account: the billing email, the contact name,
    the contact phone number, or whether they get marketing emails (yes or no).
    """
    ctx: AgentContext = runtime.context
    state: AgentState = runtime.state
    customer: Customer | None = state.get("customer")
    if customer is None:
        return _not_verified(runtime)
    if customer["status"] == "suspended":
        message = "The account is suspended. Changes need a review by the billing team."
        ticket = await _hand_off(
            runtime,
            reason="account_suspended",
            priority="high",
            summary=f"Requested a change of {field} on a suspended account.",
        )
        return _reply(
            runtime,
            {
                "status": "refused",
                "message": message,
                "ticket": _ticket_payload(ticket, team="billing"),
            },
            outcome="refused",
            summary=f"Refused: account suspended, ticket {ticket.id}",
            update=_ticket_update(ticket),
        )
    try:
        async with ctx.pool.connection() as conn:
            change = await repo.update_field(
                conn,
                world_id=ctx.world_id,
                account_number=customer["account_number"],
                field=field,
                value=value,
            )
    except repo.FieldError as exc:
        return _reply(
            runtime,
            {"status": "refused", "message": str(exc)},
            outcome="refused",
            summary=f"Refused: {exc}",
        )
    update = None
    if field in customer:
        # The session keeps a copy of what it shows the model and the UI about the customer.
        refreshed = Customer(**{**customer, field: change.new_value})
        update = {"customer": refreshed}
        runtime.stream_writer(_session_state(state, refreshed))
    return _reply(
        runtime,
        {
            "status": "updated",
            "field": field,
            "old_value": change.old_value,
            "new_value": change.new_value,
        },
        summary=f"{field}: {change.old_value} to {change.new_value}",
        update=update,
    )


@tool
async def create_escalation_ticket(
    reason: Literal[
        "customer_request",
        "billing_dispute",
        "security",
        "privacy_request",
        "unanswered_question",
        "other",
    ],
    summary: str,
    priority: Literal["low", "normal", "high", "urgent"],
    runtime: Runtime,
) -> Command:
    """Hand the conversation to a human specialist and get a ticket ID.

    Use it when the customer asks for a person, reports a security problem, disputes a charge,
    makes a legal or privacy request, or needs something no other tool can do. Use the reason
    unanswered_question when a search of the help center did not answer a question about
    Fernhill. Write the summary for the specialist: what the customer wants and what has been
    tried.
    """
    # A security report is always urgent, whatever the model chose.
    ticket = await _hand_off(
        runtime,
        reason=reason,
        priority="urgent" if reason == "security" else priority,
        summary=summary,
    )
    return _reply(
        runtime,
        {"ticket": _ticket_payload(ticket)},
        summary=f"Ticket {ticket.id}, {ticket.priority} priority",
        update=_ticket_update(ticket),
    )


# What a ticket's status means, in words a model can pass on and a judge can check.
_TICKET_STATUS = {
    "open": "open, with the support team",
    "pending": "waiting for the customer's answer",
    "resolved": "resolved",
}


@tool
async def get_ticket_status(runtime: Runtime) -> Command:
    """Read the support tickets of this conversation: their status, who at the support team has
    picked them up, and what the team has replied.

    Call it when the customer asks about a ticket, about a case that went to a person, or
    whether anyone has answered yet. Report only what it returns.
    """
    ctx: AgentContext = runtime.context
    state: AgentState = runtime.state
    customer = state.get("customer")
    async with ctx.pool.connection() as conn:
        found = await desk.for_customer(
            conn, ctx.world_id, customer["account_number"] if customer else None
        )
    tickets = [
        {
            "id": ticket["id"],
            "opened_on": ticket["opened"].isoformat(),
            "about": ticket["summary"],
            "status": _TICKET_STATUS[ticket["status"]],
            "picked_up_by": ticket["assignee"] or "nobody yet",
            "replies_from_the_support_team": ticket["replies"],
        }
        for ticket in found
    ]
    if not tickets:
        return _reply(
            runtime,
            {"tickets": [], "message": "No ticket has been opened in this conversation."},
            summary="No tickets in this conversation",
        )
    replies = sum(len(ticket["replies"]) for ticket in found)
    count = f"{len(tickets)} ticket{'s' if len(tickets) != 1 else ''}"
    said = f"{replies} repl{'ies' if replies != 1 else 'y'} from the support team"
    return _reply(runtime, {"tickets": tickets}, summary=f"{count}, {said}")


TOOLS = [
    verify_identity,
    get_account,
    search_knowledge_base,
    issue_refund,
    update_account_field,
    create_escalation_ticket,
    get_ticket_status,
]
TOOL_NAMES = frozenset(t.name for t in TOOLS)


def _on_one_line(text: str) -> str:
    return " ".join(text.split())


# What the grounding judge is told about the assistant itself. Without it a true statement
# such as "I can change the billing email but not the plan" has nothing to stand on.
# The first two sentences are there because the list alone was not enough: asked by voice what
# it can help with, the assistant said "questions about Fernhill Workspace plans, billing and
# policies, and account updates and refunds once you are verified", and the judge took that
# back about every other time.
CAPABILITIES = (
    "The assistant is the support assistant for Fernhill Workspace. It answers questions about "
    "Fernhill's plans, pricing, billing, refunds, policies and product from the help center. "
    "For a customer it has verified, it can read the account, refund an invoice, update contact "
    "details and hand the case to a person. It can read the status of the conversation's "
    "tickets and what the support team has replied on them.\n"
    "What the assistant's tools can do. It can do nothing else to an account.\n"
) + ("\n".join(f"- {t.name}: {_on_one_line(t.description)}" for t in TOOLS))
