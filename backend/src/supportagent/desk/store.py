"""The desk's reads and writes. Every one of them is scoped to a workspace.

A ticket is reached through the conversation it came from, and that conversation has to belong
to the workspace that asks. A key from another workspace finds nothing, the same as a key that
does not exist.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from supportagent.crm import policy, repo
from supportagent.desk.models import (
    AccountChange,
    AccountInvoice,
    AccountRefund,
    AccountView,
    Activity,
    DeskStats,
    DeskTicket,
)
from supportagent.events import CaseUpdated

_TICKETS = """
SELECT t.key, t.id, t.status, t.priority, t.reason, t.summary, t.channel, t.assignee,
       t.account_number, t.tier, t.transcript_excerpt, t.world_id, s.id AS session_id,
       extract(epoch FROM t.created_at)::float8 AS created_at,
       extract(epoch FROM t.updated_at)::float8 AS updated_at,
       c.company_name AS company, c.contact_name AS contact,
       (SELECT count(*) FROM ticket_activity a
         WHERE a.world_id = t.world_id AND a.ticket_id = t.id AND a.kind = 'reply') AS replies
FROM tickets t
JOIN LATERAL (
    SELECT id FROM sessions WHERE world_id = t.world_id AND workspace_id = %(workspace)s LIMIT 1
) s ON true
LEFT JOIN customers c ON c.world_id = t.world_id AND c.account_number = t.account_number
"""
_SHOWN = 200  # the newest tickets a desk lists


def workspace_id(key: str) -> str:
    """What the database keeps of a workspace key. The key itself stays with the browser."""
    return hashlib.sha256(f"workspace:{key}".encode()).hexdigest()[:32]


def agent_name(given: str) -> str:
    """A display name on one line. It is shown to the customer next to a reply."""
    name = " ".join(given.replace("\x00", " ").split())[:40]
    return name or "Support agent"


@dataclass(frozen=True)
class Found:
    """A ticket and where it lives. Only `ticket` leaves the server."""

    ticket: DeskTicket
    world_id: str
    session_id: str
    before: list[str]


@dataclass
class Change:
    """What an action did: where, and what of it the customer is told."""

    found: Found
    updates: list[CaseUpdated] = field(default_factory=list)
    refused: str | None = None  # why nothing was done, in words for the desk


def _found(row: dict[str, Any]) -> Found:
    world_id, session_id = row.pop("world_id"), row.pop("session_id")
    before = [line for line in row.pop("transcript_excerpt").splitlines() if line.strip()]
    return Found(DeskTicket(**row), world_id, session_id, before)


async def tickets(conn: AsyncConnection, workspace: str) -> list[DeskTicket]:
    cur = await conn.execute(
        _TICKETS + "ORDER BY t.created_at DESC, t.id LIMIT %(limit)s",
        {"workspace": workspace, "limit": _SHOWN},
    )
    return [_found(row).ticket for row in await cur.fetchall()]


async def by_key(
    conn: AsyncConnection, workspace: str, key: str, *, lock: bool = False
) -> Found | None:
    cur = await conn.execute(
        _TICKETS + "WHERE t.key = %(key)s" + (" FOR UPDATE OF t" if lock else ""),
        {"workspace": workspace, "key": key},
    )
    row = await cur.fetchone()
    return _found(row) if row else None


async def by_id(
    conn: AsyncConnection, workspace: str, world_id: str, ticket_id: str
) -> Found | None:
    cur = await conn.execute(
        _TICKETS + "WHERE t.world_id = %(world)s AND t.id = %(id)s",
        {"workspace": workspace, "world": world_id, "id": ticket_id},
    )
    row = await cur.fetchone()
    return _found(row) if row else None


async def activity(conn: AsyncConnection, found: Found) -> list[Activity]:
    cur = await conn.execute(
        "SELECT id, kind, author, public, body, meta, extract(epoch FROM created_at)::float8 AS at "
        "FROM ticket_activity WHERE world_id = %s AND ticket_id = %s ORDER BY id",
        (found.world_id, found.ticket.id),
    )
    return [Activity(**row) for row in await cur.fetchall()]


async def account(conn: AsyncConnection, found: Found) -> AccountView | None:
    number = found.ticket.account_number
    customer = await repo.get_customer(conn, found.world_id, number) if number else None
    if customer is None:
        return None
    invoices = await repo.list_invoices(conn, found.world_id, number, limit=12)
    cur = await conn.execute(
        "SELECT id, invoice_id, amount_cents, reason, refunded_on FROM refunds "
        "WHERE world_id = %s AND account_number = %s ORDER BY created_at DESC",
        (found.world_id, number),
    )
    refunds = await cur.fetchall()
    cur = await conn.execute(
        "SELECT field, coalesce(old_value, '') AS old_value, coalesce(new_value, '') AS new_value, "
        "extract(epoch FROM created_at)::float8 AS at FROM account_changes "
        "WHERE world_id = %s AND account_number = %s ORDER BY id DESC LIMIT 20",
        (found.world_id, number),
    )
    changes = await cur.fetchall()
    return AccountView(
        account_number=customer.account_number,
        company=customer.company_name,
        contact=customer.contact_name,
        email=customer.contact_email,
        phone=customer.contact_phone,
        billing_email=customer.billing_email,
        tier=customer.tier,
        billing_cycle=customer.billing_cycle,
        status=customer.status,
        seats=customer.seats,
        customer_since=customer.customer_since.isoformat(),
        renews_on=customer.renews_on.isoformat() if customer.renews_on else None,
        invoices=[
            AccountInvoice(
                id=i.id,
                amount_cents=i.amount_cents,
                description=i.description,
                issued_on=i.issued_on.isoformat(),
                status=i.status,
            )
            for i in invoices
        ],
        refunds=[
            AccountRefund(**{**r, "refunded_on": r["refunded_on"].isoformat()}) for r in refunds
        ],
        changes=[AccountChange(**c) for c in changes],
    )


async def stats(conn: AsyncConnection, workspace: str) -> DeskStats:
    cur = await conn.execute(
        """
        WITH mine AS (SELECT id, world_id FROM sessions WHERE workspace_id = %(workspace)s),
        t AS (SELECT t.* FROM tickets t JOIN mine m ON m.world_id = t.world_id),
        turns AS (
            SELECT e.session_id, e.payload->>'outcome' AS outcome
            FROM session_events e JOIN mine m ON m.id = e.session_id
            WHERE e.type = 'turn.finished'
        )
        SELECT
            (SELECT count(*) FROM t WHERE status = 'open') AS open,
            (SELECT count(*) FROM t WHERE status = 'pending') AS pending,
            (SELECT count(*) FROM t WHERE status = 'resolved') AS resolved,
            (SELECT count(*) FROM t WHERE status = 'open' AND assignee IS NULL) AS unassigned,
            (SELECT count(DISTINCT session_id) FROM turns) AS conversations,
            (SELECT count(*) FROM turns) AS turns,
            (SELECT count(*) FROM turns WHERE outcome = 'escalated') AS handed_off,
            ARRAY(
                SELECT extract(epoch FROM r.first - t.created_at)::float8
                FROM t JOIN LATERAL (
                    SELECT min(created_at) AS first FROM ticket_activity a
                    WHERE a.world_id = t.world_id AND a.ticket_id = t.id AND a.kind = 'reply'
                ) r ON r.first IS NOT NULL
            ) AS first_replies
        """,
        {"workspace": workspace},
    )
    row = await cur.fetchone()
    waits = row.pop("first_replies")
    return DeskStats(**row, first_reply_s=statistics.median(waits) if waits else None)


async def _log(
    conn: AsyncConnection,
    found: Found,
    kind: str,
    author: str,
    body: str = "",
    *,
    public: bool,
    meta: dict[str, str] | None = None,
) -> None:
    await conn.execute(
        "INSERT INTO ticket_activity (world_id, ticket_id, kind, author, public, body, meta) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (found.world_id, found.ticket.id, kind, author, public, body, Jsonb(meta or {})),
    )
    await conn.execute(
        "UPDATE tickets SET updated_at = now() WHERE world_id = %s AND id = %s",
        (found.world_id, found.ticket.id),
    )


def _told(found: Found, kind: str, author: str, body: str = "", **now: Any) -> CaseUpdated:
    """What the customer's conversation is told of a change. `now` is what the change set."""
    ticket = found.ticket
    return CaseUpdated(
        ticket_id=ticket.id,
        status=now.get("status", ticket.status),
        assignee=now["assignee"] if "assignee" in now else ticket.assignee,
        kind=kind,
        author=author,
        body=body,
    )


async def _set(conn: AsyncConnection, found: Found, column: str, value: str | None) -> None:
    assert column in ("status", "priority", "assignee")
    await conn.execute(
        f"UPDATE tickets SET {column} = %s WHERE world_id = %s AND id = %s",
        (value, found.world_id, found.ticket.id),
    )


async def assign(
    conn: AsyncConnection, workspace: str, key: str, agent: str, assignee: str | None
) -> Change | None:
    """Give the ticket to `assignee`, or to nobody. The customer is told who has it."""
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        change = Change(found)
        if found.ticket.assignee == assignee:
            return change
        await _set(conn, found, "assignee", assignee)
        await _log(conn, found, "assigned", agent, assignee or "", public=True)
        change.updates.append(_told(found, "assigned", agent, assignee=assignee))
        return change


async def set_status(
    conn: AsyncConnection, workspace: str, key: str, agent: str, status: str
) -> Change | None:
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        change = Change(found)
        if found.ticket.status == status:
            return change
        await _set(conn, found, "status", status)
        await _log(conn, found, "status", agent, status, public=True)
        change.updates.append(_told(found, "status", agent, status=status))
        return change


async def set_priority(
    conn: AsyncConnection, workspace: str, key: str, agent: str, priority: str
) -> Change | None:
    """Priority is the desk's own business: nothing of it reaches the customer."""
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        if found.ticket.priority != priority:
            await _set(conn, found, "priority", priority)
            await _log(conn, found, "priority", agent, priority, public=False)
        return Change(found)


async def note(
    conn: AsyncConnection, workspace: str, key: str, agent: str, body: str
) -> Change | None:
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        await _log(conn, found, "note", agent, body, public=False)
        return Change(found)


async def reply(
    conn: AsyncConnection, workspace: str, key: str, agent: str, body: str, status: str | None
) -> Change | None:
    """A reply to the customer. Whoever replies to a ticket nobody has takes it."""
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        change = Change(found)
        now: dict[str, Any] = {}
        if found.ticket.assignee is None:
            await _set(conn, found, "assignee", agent)
            await _log(conn, found, "assigned", agent, agent, public=True)
            now["assignee"] = agent
        await _log(conn, found, "reply", agent, body, public=True)
        if status is not None and status != found.ticket.status:
            await _set(conn, found, "status", status)
            await _log(conn, found, "status", agent, status, public=True)
            now["status"] = status
        change.updates.append(_told(found, "reply", agent, body, **now))
        return change


async def refund(
    conn: AsyncConnection, workspace: str, key: str, agent: str, invoice_id: str, today: date
) -> Change | None:
    """Refund one invoice of the ticket's customer as the billing team.

    The same transaction as the assistant's refund, with the two limits lifted that are the
    billing team's to lift. Everything else of the policy holds, and so does "at most once".
    """
    async with conn.transaction():
        found = await by_key(conn, workspace, key, lock=True)
        if found is None:
            return None
        change = Change(found)
        number = found.ticket.account_number
        if number is None:
            change.refused = (
                "No verified customer is on this ticket, so there is no invoice to refund."
            )
            return change
        outcome = await repo.refund_invoice(
            conn,
            world_id=found.world_id,
            account_number=number,
            invoice_id=invoice_id.strip().upper(),
            reason="approved_by_billing",
            tool_call_id=None,
            today=today,
            by_billing=True,
        )
        if not outcome.decision.allowed:
            change.refused = outcome.decision.message
            return change
        amount = policy.usd(outcome.amount_cents or 0)
        body = f"Refund {outcome.refund_id} of {amount} for {outcome.invoice_id} was approved."
        receipt = {
            "invoice_id": outcome.invoice_id,
            "refund_id": outcome.refund_id or "",
            "amount": amount,
        }
        await _log(conn, found, "refund", agent, body, public=True, meta=receipt)
        change.updates.append(_told(found, "refund", agent, body))
        return change


async def for_customer(
    conn: AsyncConnection, world_id: str, account_number: str | None
) -> list[dict[str, Any]]:
    """The tickets of one conversation as its customer may know them, for the assistant.

    Those on the verified customer's account, and those no account is on. A ticket on another
    account of the same conversation is not this customer's to hear about.
    """
    cur = await conn.execute(
        "SELECT id, status, assignee, summary, created_at::date AS opened FROM tickets "
        "WHERE world_id = %s AND (account_number IS NULL OR account_number = %s) "
        "ORDER BY created_at",
        (world_id, account_number),
    )
    found = await cur.fetchall()
    cur = await conn.execute(
        "SELECT ticket_id, kind, author, body, created_at::date AS on_date FROM ticket_activity "
        "WHERE world_id = %s AND public AND kind IN ('reply', 'refund') ORDER BY id",
        (world_id,),
    )
    said: dict[str, list[dict[str, str]]] = {}
    for row in await cur.fetchall():
        said.setdefault(row["ticket_id"], []).append(
            {"from": row["author"], "date": row["on_date"].isoformat(), "text": row["body"]}
        )
    return [{**ticket, "replies": said.get(ticket["id"], [])} for ticket in found]
