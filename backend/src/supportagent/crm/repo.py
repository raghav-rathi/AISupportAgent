"""Queries on the mock CRM. Every function works inside one world."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import date

from psycopg import AsyncConnection, sql

from supportagent.crm import policy

_CUSTOMER_COLS = (
    "account_number, company_name, contact_name, contact_email, contact_phone, billing_email, "
    "tier, billing_cycle, status, seats, marketing_opt_in, customer_since, renews_on"
)
_INVOICE_COLS = "id, amount_cents, description, period_start, period_end, issued_on, status"

_EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
)
_PHONE = re.compile(r"\+?[0-9][0-9 ()-]{5,22}")
# A name starts with a letter and holds letters, digits, spaces and a little punctuation. The name
# is shown to the model, so nothing that could pass for markup or a new line is allowed.
_NAME = re.compile(r"[^\W\d_][\w .,'’-]{0,79}")


@dataclass(frozen=True)
class Customer:
    account_number: str
    company_name: str
    contact_name: str
    contact_email: str
    contact_phone: str
    billing_email: str
    tier: str
    billing_cycle: str | None
    status: str
    seats: int
    marketing_opt_in: bool
    customer_since: date
    renews_on: date | None


@dataclass(frozen=True)
class Invoice:
    id: str
    amount_cents: int
    description: str
    period_start: date
    period_end: date
    issued_on: date
    status: str


@dataclass(frozen=True)
class RefundOutcome:
    decision: policy.RefundDecision
    invoice_id: str
    refund_id: str | None = None
    amount_cents: int | None = None
    replayed: bool = False  # True when this call repeated one that had already paid


@dataclass(frozen=True)
class FieldChange:
    field: str
    old_value: str
    new_value: str


class FieldError(ValueError):
    """The requested account change is not allowed or the value is malformed."""


def hash_pin(account_number: str, pin: str) -> str:
    """Mock data only. A real system would use a slow, salted password hash."""
    return hashlib.sha256(f"{account_number}:{pin}".encode()).hexdigest()


async def verify_pin(
    conn: AsyncConnection, world_id: str, account_number: str, pin: str
) -> Customer | None:
    """Return the customer when the account number and PIN match, else None."""
    cur = await conn.execute(
        f"SELECT pin_hash, {_CUSTOMER_COLS} FROM customers "
        "WHERE world_id = %s AND account_number = %s",
        (world_id, account_number),
    )
    row = await cur.fetchone()
    expected = row.pop("pin_hash") if row else ""
    if not hmac.compare_digest(expected, hash_pin(account_number, pin)):
        return None
    return Customer(**row)


async def get_customer(
    conn: AsyncConnection, world_id: str, account_number: str
) -> Customer | None:
    cur = await conn.execute(
        f"SELECT {_CUSTOMER_COLS} FROM customers WHERE world_id = %s AND account_number = %s",
        (world_id, account_number),
    )
    row = await cur.fetchone()
    return Customer(**row) if row else None


async def list_invoices(
    conn: AsyncConnection, world_id: str, account_number: str, limit: int = 6
) -> list[Invoice]:
    cur = await conn.execute(
        f"SELECT {_INVOICE_COLS} FROM invoices WHERE world_id = %s AND account_number = %s "
        "ORDER BY issued_on DESC, id DESC LIMIT %s",
        (world_id, account_number, limit),
    )
    return [Invoice(**row) for row in await cur.fetchall()]


_NOT_FOUND = policy.RefundDecision(
    "invoice_not_found",
    allowed=False,
    escalate=False,
    message="No invoice with that number exists on this account.",
)


async def refund_invoice(
    conn: AsyncConnection,
    *,
    world_id: str,
    account_number: str,
    invoice_id: str,
    reason: str,
    tool_call_id: str | None,
    today: date,
) -> RefundOutcome:
    """Refund one invoice in full if policy allows it. Safe to call any number of times.

    The customer row is locked first and held for the whole decision, so every refund on one
    account runs one after the other. That matters for a duplicate pair: each invoice is refundable
    only while its twin is still paid, and two requests that each looked at the other's invoice
    without that lock would both go through. The refund row also carries a unique key built from
    customer and invoice, so a retry pays at most once.
    """
    key = f"refund:{account_number}:{invoice_id}"
    async with conn.transaction():
        cur = await conn.execute(
            f"SELECT {_CUSTOMER_COLS} FROM customers "
            "WHERE world_id = %s AND account_number = %s FOR UPDATE",
            (world_id, account_number),
        )
        row = await cur.fetchone()
        if row is None:
            return RefundOutcome(_NOT_FOUND, invoice_id)
        customer = Customer(**row)
        if customer.tier == "free":
            # A Free workspace has no invoices, so this has to come before the invoice lookup.
            return RefundOutcome(policy.FREE_PLAN_REFUND, invoice_id)

        cur = await conn.execute(
            f"SELECT {_INVOICE_COLS} FROM invoices "
            "WHERE world_id = %s AND id = %s AND account_number = %s FOR UPDATE",
            (world_id, invoice_id, account_number),
        )
        row = await cur.fetchone()
        if row is None:
            # Also covers an invoice that belongs to someone else. The reply is the same on purpose.
            return RefundOutcome(_NOT_FOUND, invoice_id)
        invoice = Invoice(**row)

        cur = await conn.execute(
            "SELECT id, amount_cents, tool_call_id, refunded_on FROM refunds "
            "WHERE world_id = %s AND idempotency_key = %s",
            (world_id, key),
        )
        existing = await cur.fetchone()
        if existing is not None:
            if tool_call_id is not None and existing["tool_call_id"] == tool_call_id:
                decision = policy.RefundDecision("ok", True, False, "Refund approved.")
                replayed = True
            else:
                decision = policy.RefundDecision(
                    "already_refunded",
                    allowed=False,
                    escalate=False,
                    message=f"This invoice was already refunded on "
                    f"{existing['refunded_on'].isoformat()} as {existing['id']}. "
                    "No second refund was issued.",
                )
                replayed = False
            return RefundOutcome(
                decision, invoice_id, existing["id"], existing["amount_cents"], replayed
            )

        cur = await conn.execute(
            "SELECT EXISTS (SELECT 1 FROM invoices WHERE world_id = %s AND account_number = %s "
            "AND id <> %s AND period_start = %s AND period_end = %s AND amount_cents = %s "
            "AND status = 'paid') AS dup",
            (
                world_id,
                account_number,
                invoice_id,
                invoice.period_start,
                invoice.period_end,
                invoice.amount_cents,
            ),
        )
        is_duplicate = (await cur.fetchone())["dup"]

        decision = policy.evaluate_refund(
            tier=customer.tier,
            billing_cycle=customer.billing_cycle,
            account_status=customer.status,
            invoice_status=invoice.status,
            amount_cents=invoice.amount_cents,
            issued_on=invoice.issued_on,
            today=today,
            is_duplicate=is_duplicate,
        )
        if not decision.allowed:
            return RefundOutcome(decision, invoice_id)

        refund_id = "RF-" + hashlib.sha1(f"{world_id}:{key}".encode()).hexdigest()[:8].upper()
        await conn.execute(
            "INSERT INTO refunds (world_id, id, invoice_id, account_number, amount_cents, reason, "
            "idempotency_key, tool_call_id, refunded_on) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                world_id,
                refund_id,
                invoice_id,
                account_number,
                invoice.amount_cents,
                reason,
                key,
                tool_call_id,
                today,
            ),
        )
        await conn.execute(
            "UPDATE invoices SET status = 'refunded' WHERE world_id = %s AND id = %s",
            (world_id, invoice_id),
        )
    return RefundOutcome(decision, invoice_id, refund_id, invoice.amount_cents)


def _clean(field: str, value: str) -> str | bool:
    if field not in policy.UPDATABLE_FIELDS:
        raise FieldError(f"Support cannot change {field!r} in chat.")
    text = " ".join(value.split())  # also folds line breaks and tabs into single spaces
    if field == "billing_email":
        if not _EMAIL.fullmatch(text) or ".." in text or len(text) > 254:
            raise FieldError("That is not a valid email address.")
        return text.lower()
    if field == "contact_phone":
        digits = sum(ch.isdigit() for ch in text)
        if not _PHONE.fullmatch(text) or not 7 <= digits <= 15:
            raise FieldError("That is not a valid phone number.")
        return text
    if field == "contact_name":
        if not _NAME.fullmatch(text):
            raise FieldError(
                "A contact name starts with a letter and has up to 80 letters, digits, spaces, "
                "hyphens, apostrophes, commas and periods."
            )
        return text
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    raise FieldError("Use yes or no for marketing emails.")


async def update_field(
    conn: AsyncConnection, *, world_id: str, account_number: str, field: str, value: str
) -> FieldChange:
    """Change one whitelisted field and record the change. Raises FieldError when not allowed."""
    new = _clean(field, value)
    column = sql.Identifier(field)  # safe: _clean accepted only a whitelisted name
    async with conn.transaction():
        cur = await conn.execute(
            sql.SQL(
                "SELECT status, {} AS old FROM customers "
                "WHERE world_id = %s AND account_number = %s FOR UPDATE"
            ).format(column),
            (world_id, account_number),
        )
        row = await cur.fetchone()
        if row is None:
            raise LookupError(account_number)
        if row["status"] == "suspended":
            raise FieldError("The account is suspended. Changes need a review by the billing team.")
        await conn.execute(
            sql.SQL(
                "UPDATE customers SET {} = %s WHERE world_id = %s AND account_number = %s"
            ).format(column),
            (new, world_id, account_number),
        )
        await conn.execute(
            "INSERT INTO account_changes (world_id, account_number, field, old_value, new_value) "
            "VALUES (%s, %s, %s, %s, %s)",
            (world_id, account_number, field, str(row["old"]), str(new)),
        )
    return FieldChange(field, str(row["old"]), str(new))


async def create_ticket(
    conn: AsyncConnection,
    *,
    world_id: str,
    account_number: str | None,
    tier: str | None,
    reason: str,
    priority: str,
    summary: str,
    transcript_excerpt: str,
    tool_call_id: str | None = None,
) -> str:
    """Store an escalation ticket and return its ID.

    With a `tool_call_id` the ticket is stored once per call: a tool that is retried after its
    first attempt already wrote the ticket gets that ticket's ID back.
    """
    for _ in range(4):  # a fresh random ID on the rare clash with an existing one
        cur = await conn.execute(
            "INSERT INTO tickets (world_id, id, account_number, tier, reason, priority, summary, "
            "transcript_excerpt, tool_call_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING RETURNING id",
            (
                world_id,
                f"ESC-{secrets.token_hex(3).upper()}",
                account_number,
                tier,
                reason,
                priority,
                summary,
                transcript_excerpt,
                tool_call_id,
            ),
        )
        row = await cur.fetchone()
        if row is None and tool_call_id is not None:
            cur = await conn.execute(
                "SELECT id FROM tickets WHERE world_id = %s AND tool_call_id = %s",
                (world_id, tool_call_id),
            )
            row = await cur.fetchone()
        if row is not None:
            return row["id"]
    raise RuntimeError("could not allocate a ticket ID")
