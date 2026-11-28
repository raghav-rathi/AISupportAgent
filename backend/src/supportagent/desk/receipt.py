"""The hand-off receipt: what the assistant did before it gave a case to a person.

Whoever picks a ticket up wants four answers before reading a transcript. What did the customer
ask, what did the assistant do about it, why did it stop, and what has the customer been told.
All four are in the stored events of the turn that opened the ticket, so the receipt is read
from them and no model writes it.
"""

from __future__ import annotations

from typing import Any

from psycopg import AsyncConnection

from supportagent.desk.models import Line, Receipt, ReceiptSearch, ReceiptStep
from supportagent.desk.store import Found

# Why a ticket of each kind is with a person, as a sentence the events alone can stand behind.
_WHY = {
    "refund_over_limit": "The refund is above what the assistant may approve. No refund was issued.",
    "account_suspended": "The account is suspended, and a refund on it needs the billing team. "
    "No refund was issued.",
    "verification_failed": "Three wrong PINs in one conversation. Verification is locked for it, "
    "and nobody is verified.",
    "low_confidence": "A search of the help center found nothing on this question, so the "
    "assistant did not answer it.",
    "ungrounded_answer": "The assistant wrote an answer that did not pass its grounding check. "
    "The answer was withheld.",
    "unanswered_question": "The assistant's answer says the help center lacks something the "
    "customer asked about.",
    "customer_request": "The customer asked for a person.",
    "billing_dispute": "The customer disputes a charge.",
    "security": "The customer reported a security problem.",
    "privacy_request": "The customer made a legal or privacy request.",
    "step_limit": "The assistant used every step a turn may take without reaching an answer.",
    "other": "The assistant passed this on as something only a person can settle.",
}
_READ = (
    "turn.started",
    "tool.finished",
    "retrieval.result",
    "guardrail.verdict",
    "assistant.message",
    "speech.spoken",
)
_AFTER = 60  # lines of the conversation a desk is shown


async def _turn_of(conn: AsyncConnection, found: Found) -> tuple[str, int] | None:
    """The turn that opened the ticket, and where it begins in the session's stream."""
    cur = await conn.execute(
        "SELECT turn_id FROM session_events WHERE session_id = %s AND type = 'escalation.created' "
        "AND payload->'ticket'->>'id' = %s ORDER BY seq LIMIT 1",
        (found.session_id, found.ticket.id),
    )
    row = await cur.fetchone()
    if row is None or row["turn_id"] is None:
        return None
    cur = await conn.execute(
        "SELECT min(seq) AS seq FROM session_events WHERE session_id = %s AND turn_id = %s",
        (found.session_id, row["turn_id"]),
    )
    return row["turn_id"], (await cur.fetchone())["seq"]


async def receipt(conn: AsyncConnection, found: Found) -> Receipt | None:
    turn = await _turn_of(conn, found)
    if turn is None:
        return None
    cur = await conn.execute(
        "SELECT type, payload FROM session_events WHERE session_id = %s AND turn_id = %s "
        "AND type = ANY(%s) ORDER BY seq",
        (found.session_id, turn[0], list(_READ)),
    )
    asked, told, mode = "", "", found.ticket.channel
    steps: list[ReceiptStep] = []
    searches: list[ReceiptSearch] = []
    check: str | None = None
    unsupported: list[str] = []
    heard: str | None = None
    for row in await cur.fetchall():
        kind, event = row["type"], row["payload"]
        if kind == "turn.started":
            asked, mode = event.get("text", ""), event.get("mode", mode)
        elif kind == "tool.finished":
            steps.append(
                ReceiptStep(
                    name=event["name"], outcome=event["outcome"], summary=event.get("summary", "")
                )
            )
        elif kind == "retrieval.result":
            hits: list[dict[str, Any]] = event.get("hits") or []
            searches.append(
                ReceiptSearch(
                    query=event.get("query", ""),
                    band=event["band"],
                    best=hits[0]["source"] if hits else None,
                    score=event.get("top_score"),
                )
            )
        elif kind == "guardrail.verdict":
            check, unsupported = event["verdict"], list(event.get("unsupported") or [])
        elif kind == "assistant.message":
            told = event.get("text", "")
        elif kind == "speech.spoken" and not event.get("filler"):
            cut = event.get("interrupted") and event.get("text_spoken") is not None
            heard = event["text_spoken"] if cut else None
    return Receipt(
        asked=asked,
        channel=mode,
        why=_WHY.get(found.ticket.reason, _WHY["other"]),
        steps=steps,
        searches=searches,
        check=check,
        unsupported=unsupported,
        told=told,
        heard=heard,
    )


async def after(conn: AsyncConnection, found: Found) -> list[Line]:
    """What was said from the turn of the hand-off on, the customer's later messages included."""
    turn = await _turn_of(conn, found)
    if turn is None:
        return []
    cur = await conn.execute(
        "SELECT type, turn_id, payload->>'text' AS text, payload->>'mode' AS mode, "
        "extract(epoch FROM created_at)::float8 AS at FROM session_events "
        "WHERE session_id = %s AND seq >= %s AND type IN ('turn.started', 'assistant.message') "
        "ORDER BY seq LIMIT %s",
        (found.session_id, turn[1], _AFTER),
    )
    modes: dict[str, str] = {}
    lines: list[Line] = []
    for row in await cur.fetchall():
        if row["type"] == "turn.started":
            modes[row["turn_id"]] = row["mode"] or "text"
        if not (row["text"] or "").strip():
            continue
        lines.append(
            Line(
                role="customer" if row["type"] == "turn.started" else "assistant",
                text=row["text"],
                mode=modes.get(row["turn_id"], "text"),
                at=row["at"],
            )
        )
    return lines
