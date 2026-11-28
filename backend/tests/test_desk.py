"""The service desk, driven over its real socket next to a customer's conversation.

A scripted model opens the tickets. Everything after that is the desk's own code: who sees a
ticket, what a person does to it, and what of that reaches the customer.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from agent_harness import DANIEL, MAYA, scalar
from fakes import ScriptedModel, call, say
from test_ws import ORIGIN, Client
from test_ws import serve as serve  # the fixture: the app on a free port
from websockets.asyncio.client import connect

PERSON = call(
    "create_escalation_ticket",
    reason="customer_request",
    summary="Wants a person.",
    priority="normal",
)
OVER_LIMIT = (
    call("verify_identity", **DANIEL),
    call("issue_refund", invoice_id="INV-30388", reason="billing_error"),
    say("That is above what I can refund. I have passed it to the billing team."),
)


def new_workspace() -> str:
    """The key one browser would make up for itself."""
    return f"workspace-{uuid.uuid4().hex}"


class Customer(Client):
    """A conversation whose browser names a workspace."""

    @classmethod
    async def open(cls, address, workspace, session_id=None):
        client = cls(await connect(f"ws://{address}/ws", origin=ORIGIN))
        await client.send(type="hello", session_id=session_id, last_seq=0, workspace=workspace)
        client.ready = await client.next()
        return client


class Desk:
    def __init__(self, ws):
        self.ws, self.events, self.ready = ws, [], None

    @classmethod
    async def open(cls, address, workspace, agent="Sam Carter"):
        desk = cls(await connect(f"ws://{address}/desk", origin=ORIGIN))
        desk.workspace = workspace
        await desk.send(type="desk.hello", workspace=workspace, agent=agent)
        desk.ready = await desk.next()
        return desk

    async def send(self, **message):
        await self.ws.send(json.dumps(message))

    async def next(self):
        event = json.loads(await asyncio.wait_for(self.ws.recv(), 5))
        self.events.append(event)
        return event

    async def until(self, kind):
        while True:
            event = await self.next()
            if event["type"] == kind:
                return event

    async def do(self, kind, **message):
        """One action, and the ticket as it stands afterwards."""
        await self.send(type=kind, **message)
        return await self.until("desk.detail")


async def handed_off(serve, *script, text="I'd like to talk to a person."):
    """A conversation in which the assistant has just opened a ticket, and a desk that saw it."""
    rt, address = await serve(ScriptedModel("m", *script))
    desk = await Desk.open(address, new_workspace())
    assert desk.ready["tickets"] == []
    customer = await Customer.open(address, desk.workspace)
    await customer.say(text)
    ticket = (await desk.until("desk.ticket"))["ticket"]
    await customer.until("turn.finished")
    return rt, address, desk, customer, ticket


async def test_a_ticket_shows_on_the_desk_of_its_workspace_the_moment_it_is_opened(serve):
    _, address, desk, customer, ticket = await handed_off(serve, PERSON, say("Ticket opened."))
    assert (ticket["reason"], ticket["status"], ticket["assignee"]) == (
        "customer_request",
        "open",
        None,
    )
    assert ticket["id"].startswith("ESC-") and ticket["channel"] == "text"
    # The desk's name for a ticket says nothing about the conversation it came from.
    assert customer.ready["session_id"] not in json.dumps(ticket)

    # Once the turn is over the desk hears of the ticket again: its receipt is whole only now.
    assert (await desk.until("desk.ticket"))["stats"]["turns"] == 1

    # A desk that opens later finds it, with the workspace in numbers.
    late = await Desk.open(address, desk.workspace)
    assert [t["key"] for t in late.ready["tickets"]] == [ticket["key"]]
    counts = late.ready["stats"]
    assert (counts["open"], counts["unassigned"], counts["turns"], counts["handed_off"]) == (
        1,
        1,
        1,
        1,
    )
    for socket in (desk, late, customer):
        await socket.ws.close()


async def test_another_workspace_neither_sees_a_ticket_nor_can_touch_it(serve):
    _, address, desk, customer, ticket = await handed_off(serve, PERSON, say("Ticket opened."))
    stranger = await Desk.open(address, new_workspace())
    assert stranger.ready["tickets"] == [] and stranger.ready["stats"]["open"] == 0
    # Knowing the key is not enough: the ticket has to be in the workspace that asks.
    for message in (
        {"type": "desk.open"},
        {"type": "desk.reply", "body": "Hello from someone else."},
        {"type": "desk.status", "status": "resolved"},
        {"type": "desk.refund", "invoice_id": "INV-30388"},
    ):
        await stranger.send(key=ticket["key"], **message)
        refused = await stranger.until("desk.error")
        assert (refused["code"], refused["key"]) == ("not_found", ticket["key"])
    detail = await desk.do("desk.open", key=ticket["key"])
    assert detail["ticket"]["status"] == "open" and len(detail["activity"]) == 1
    for socket in (desk, stranger, customer):
        await socket.ws.close()


async def test_a_reply_reaches_the_customer_where_they_asked_and_a_note_never_does(serve, pool):
    _, address, desk, customer, ticket = await handed_off(serve, PERSON, say("Ticket opened."))
    key = ticket["key"]
    noted = await desk.do("desk.note", key=key, body="Checked with legal: clause 4 applies.")
    assert [(a["kind"], a["public"]) for a in noted["activity"]] == [
        ("created", True),
        ("note", False),
    ]
    replied = await desk.do("desk.reply", key=key, body="Hi, I'm looking into your contract.")
    # Replying to a ticket nobody has takes it.
    assert replied["ticket"]["assignee"] == "Sam Carter" and replied["ticket"]["replies"] == 1

    # The first thing the customer hears from the desk is the reply. The note was not for them.
    told = await customer.until("case.updated")
    assert (told["kind"], told["author"], told["body"]) == (
        "reply",
        "Sam Carter",
        "Hi, I'm looking into your contract.",
    )
    assert (told["ticket_id"], told["status"], told["assignee"]) == (
        ticket["id"],
        "open",
        "Sam Carter",
    )
    assert told["seq"] > 0 and told["turn_id"] is None

    # It is part of the conversation: a reload gets it from the store, and the note is in
    # nothing the conversation holds.
    again = await Customer.open(address, desk.workspace, customer.ready["session_id"])
    assert (await again.until("case.updated"))["body"] == told["body"]
    assert (
        await scalar(
            pool,
            "SELECT count(*) FROM session_events WHERE session_id = %s AND payload::text LIKE %s",
            customer.ready["session_id"],
            "%legal%",
        )
        == 0
    )
    for socket in (desk, customer, again):
        await socket.ws.close()


async def test_who_has_a_ticket_and_where_it_stands_is_told_to_every_desk_and_the_customer(
    serve,
):
    _, address, desk, customer, ticket = await handed_off(serve, PERSON, say("Ticket opened."))
    key = ticket["key"]
    second = await Desk.open(address, desk.workspace, agent="Priya")
    await desk.do("desk.assign", key=key, assignee="Sam Carter")
    assert (await second.until("desk.ticket"))["ticket"]["assignee"] == "Sam Carter"
    assert (await customer.until("case.updated"))["kind"] == "assigned"

    # Priority is the desk's own business.
    raised = await desk.do("desk.priority", key=key, priority="urgent")
    assert raised["ticket"]["priority"] == "urgent"
    assert raised["activity"][-1]["public"] is False

    done = await desk.do("desk.reply", key=key, body="All sorted.", status="resolved")
    assert done["ticket"]["status"] == "resolved"
    changed = await second.until("desk.ticket")
    while changed["ticket"]["status"] != "resolved":
        changed = await second.until("desk.ticket")
    assert (changed["stats"]["open"], changed["stats"]["resolved"]) == (0, 1)
    assert changed["stats"]["first_reply_s"] is not None
    # One message to the customer for the reply that closed it, with where the ticket stands.
    told = await customer.until("case.updated")
    assert (told["kind"], told["status"], told["body"]) == ("reply", "resolved", "All sorted.")
    for socket in (desk, second, customer):
        await socket.ws.close()


async def test_the_receipt_says_what_the_assistant_did_and_why_it_stopped(serve):
    _, address, desk, customer, ticket = await handed_off(
        serve, *OVER_LIMIT, text="Account 20573846, PIN 9034. Please refund INV-30388."
    )
    detail = await desk.do("desk.open", key=ticket["key"])
    receipt = detail["receipt"]
    # Read from the stored events: the PIN is masked there, and so it is here.
    assert receipt["asked"] == "Account 20573846, PIN ****. Please refund INV-30388."
    assert [(s["name"], s["outcome"]) for s in receipt["steps"]] == [
        ("verify_identity", "ok"),
        ("issue_refund", "refused"),
    ]
    assert receipt["why"].startswith("The refund is above what the assistant may approve")
    assert receipt["told"].startswith("That is above what I can refund")
    # This app runs without a judge, and the receipt says so: nothing checked the answer.
    assert (receipt["check"], receipt["heard"], receipt["channel"]) == ("skipped", None, "text")
    # The customer behind it, from the conversation's own copy of the CRM.
    account = detail["account"]
    assert (account["company"], account["tier"]) == ("Quillon Freight", "pro")
    assert ("INV-30388", 144_000, "paid") in [
        (i["id"], i["amount_cents"], i["status"]) for i in account["invoices"]
    ]
    assert [line["role"] for line in detail["after"]] == ["customer", "assistant"]
    for socket in (desk, customer):
        await socket.ws.close()


async def test_billing_approves_at_the_desk_what_the_assistant_may_not(serve):
    _, address, desk, customer, ticket = await handed_off(
        serve, *OVER_LIMIT, text="Account 20573846, PIN 9034. Please refund INV-30388."
    )
    key = ticket["key"]
    assert ticket["reason"] == "refund_over_limit" and ticket["company"] == "Quillon Freight"
    approved = await desk.do("desk.refund", key=key, invoice_id="INV-30388")
    assert [(r["invoice_id"], r["amount_cents"]) for r in approved["account"]["refunds"]] == [
        ("INV-30388", 144_000)
    ]
    assert approved["activity"][-1]["kind"] == "refund" and approved["activity"][-1]["public"]
    told = await customer.until("case.updated")
    assert told["kind"] == "refund" and "$1,440.00 for INV-30388 was approved" in told["body"]

    # At most once, here as for the assistant. And an invoice that is not the customer's own
    # is no invoice.
    await desk.send(type="desk.refund", key=key, invoice_id="INV-30388")
    assert "already refunded" in (await desk.until("desk.error"))["message"]
    await desk.send(type="desk.refund", key=key, invoice_id="INV-30417")
    assert (await desk.until("desk.error"))["code"] == "refused"
    for socket in (desk, customer):
        await socket.ws.close()


async def test_the_assistant_reads_a_persons_reply_with_its_ticket_tool(serve):
    script = (
        call("verify_identity", **MAYA),
        PERSON,
        say("I have opened a ticket."),
        call("get_ticket_status"),
        say("Sam Carter replied: it is refunded."),
    )
    rt, address, desk, customer, ticket = await handed_off(
        serve, *script, text="Account 10482917, PIN 4721. I want a person."
    )
    await desk.do("desk.reply", key=ticket["key"], body="It is refunded.", status="resolved")
    await customer.until("case.updated")
    await customer.say("Any news on my ticket?")
    step = await customer.until("tool.finished")
    assert (step["name"], step["summary"]) == (
        "get_ticket_status",
        "1 ticket, 1 reply from the support team",
    )
    await customer.until("turn.finished")
    read = json.loads(rt.llm.models[0].calls[-1][0][-1].content)["tickets"][0]
    assert (read["id"], read["status"], read["picked_up_by"]) == (
        ticket["id"],
        "resolved",
        "Sam Carter",
    )
    assert [r["text"] for r in read["replies_from_the_support_team"]] == ["It is refunded."]
    for socket in (desk, customer):
        await socket.ws.close()


async def test_a_conversation_stays_in_the_workspace_it_joined_first(serve):
    _, address, desk, customer, ticket = await handed_off(serve, PERSON, say("Ticket opened."))
    # Someone who learned the conversation's ID opens it from another browser. That moves
    # nothing: the ticket is still on the first desk and not on theirs.
    other = new_workspace()
    moved = await Customer.open(address, other, customer.ready["session_id"])
    assert moved.ready["tickets"][0]["id"] == ticket["id"]
    assert (await Desk.open(address, other)).ready["tickets"] == []
    assert len((await Desk.open(address, desk.workspace)).ready["tickets"]) == 1
    for socket in (desk, customer, moved):
        await socket.ws.close()


async def test_a_desk_that_does_not_say_hello_or_comes_from_elsewhere_is_turned_away(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    socket = await connect(f"ws://{address}/desk", origin=ORIGIN)
    await socket.send(json.dumps({"type": "desk.open", "key": "x"}))
    await socket.wait_closed()
    assert socket.close_code == 1008
    # A workspace key is the only thing that opens a desk, so a short one is refused.
    socket = await connect(f"ws://{address}/desk", origin=ORIGIN)
    await socket.send(json.dumps({"type": "desk.hello", "workspace": "short"}))
    await socket.wait_closed()
    assert socket.close_code == 1008
