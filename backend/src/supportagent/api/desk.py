"""The service desk's socket, `/desk`.

A desk names its workspace, gets the workspace's tickets, and from then on hears of every ticket
that is opened or changed in it. Everything it does to a ticket is one message, answered with the
ticket as it stands afterwards. What the customer may know of it goes into the conversation the
ticket came from, as a stored event, so the customer sees a person's reply where they asked.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from psycopg import AsyncConnection
from pydantic import BaseModel, ValidationError

from supportagent import clock
from supportagent.api.hub import Hubs, TooManySessions
from supportagent.api.limits import SlidingWindow
from supportagent.config import Settings
from supportagent.desk import receipt as receipts
from supportagent.desk import store
from supportagent.desk.models import (
    DESK_CLIENT_ADAPTER,
    DeskAssign,
    DeskDetail,
    DeskError,
    DeskHello,
    DeskNote,
    DeskOpen,
    DeskReady,
    DeskRefund,
    DeskReply,
    DeskSetPriority,
    DeskSetStatus,
    DeskTicketChanged,
)

logger = logging.getLogger(__name__)

_POLICY_VIOLATION = 1008
_FELL_BEHIND = 4000
_IDLE = 4001
_OUTBOX = 500  # messages a desk may be behind before it is cut off and has to load again
_ACTIONS_PER_MINUTE = 120  # a person does not click faster. A script is slowed to this.


async def _text(socket: WebSocket) -> str:
    """The next text frame. A binary one counts as an empty message, which nothing accepts."""
    message = await socket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    return message.get("text") or ""


async def _detail(conn: AsyncConnection, found: store.Found) -> DeskDetail:
    return DeskDetail(
        ticket=found.ticket,
        activity=await store.activity(conn, found),
        receipt=await receipts.receipt(conn, found),
        before=found.before,
        after=await receipts.after(conn, found),
        account=await store.account(conn, found),
    )


async def serve_desk(socket: WebSocket, hubs: Hubs, settings: Settings) -> None:
    try:
        async with asyncio.timeout(settings.hello_timeout_s):
            hello = DESK_CLIENT_ADAPTER.validate_json(await _text(socket))
    except (ValidationError, WebSocketDisconnect, TimeoutError):
        hello = None
    if not isinstance(hello, DeskHello):
        with suppress(RuntimeError):
            await socket.close(
                code=_POLICY_VIOLATION, reason="the first message must be desk.hello"
            )
        return

    pool = hubs.rt.pool
    workspace = store.workspace_id(hello.workspace)
    agent = store.agent_name(hello.agent)
    changes, unsubscribe = hubs.desk.subscribe(workspace)
    outbox: asyncio.Queue[BaseModel] = asyncio.Queue(maxsize=_OUTBOX)
    actions = SlidingWindow(_ACTIONS_PER_MINUTE, 60.0)
    behind = False

    def tell(event: BaseModel) -> None:
        nonlocal behind
        if outbox.full():
            behind = True  # the reader stopped reading. It loads everything again when it is back.
            return
        outbox.put_nowait(event)

    async def pump() -> None:
        while not behind:
            event = await outbox.get()
            async with asyncio.timeout(settings.send_timeout_s):
                await socket.send_text(event.model_dump_json())
        with suppress(RuntimeError):
            await socket.close(code=_FELL_BEHIND, reason="fell behind")

    async def watch() -> None:
        """Pass on every change in the workspace, as the ticket stands at that moment."""
        while True:
            world_id, ticket_id = await changes.get()
            try:
                async with pool.connection() as conn:
                    found = await store.by_id(conn, workspace, world_id, ticket_id)
                    if found is not None:
                        counts = await store.stats(conn, workspace)
                        tell(DeskTicketChanged(ticket=found.ticket, stats=counts))
            except Exception:
                logger.exception("could not read a changed ticket for a desk")

    def refuse(code: str, message: str, key: str | None = None) -> None:
        tell(DeskError(code=code, message=message, key=key))

    async def act(message: BaseModel) -> None:
        key = message.key
        async with pool.connection() as conn:
            if isinstance(message, DeskOpen):
                found = await store.by_key(conn, workspace, key)
                if found is None:
                    refuse("not_found", "no such ticket in this workspace", key)
                else:
                    tell(await _detail(conn, found))
                return
            if isinstance(message, DeskAssign):
                to = store.agent_name(message.assignee) if message.assignee else None
                change = await store.assign(conn, workspace, key, agent, to)
            elif isinstance(message, DeskSetStatus):
                change = await store.set_status(conn, workspace, key, agent, message.status)
            elif isinstance(message, DeskSetPriority):
                change = await store.set_priority(conn, workspace, key, agent, message.priority)
            elif isinstance(message, (DeskNote, DeskReply)):
                body = message.body.replace("\x00", "").strip()
                if not body:
                    refuse("bad_message", "there is nothing in it", key)
                    return
                if isinstance(message, DeskNote):
                    change = await store.note(conn, workspace, key, agent, body)
                else:
                    change = await store.reply(conn, workspace, key, agent, body, message.status)
            elif isinstance(message, DeskRefund):
                change = await store.refund(
                    conn, workspace, key, agent, message.invoice_id, clock.today()
                )
            else:
                refuse("bad_message", "not a message this desk understands")
                return
        if change is None:
            refuse("not_found", "no such ticket in this workspace", key)
            return
        if change.refused is not None:
            refuse("refused", change.refused, key)
            return
        found = change.found
        # The change is stored. Now the customer's conversation is told what it may know, and
        # after that every desk of the workspace, this one included.
        for update in change.updates:
            try:
                await hubs.tell(found.session_id, update)
            except TooManySessions:
                refuse(
                    "busy",
                    "saved on the ticket, but the server is too busy to show it in the "
                    "customer's conversation right now",
                    key,
                )
        hubs.desk.publish(workspace, found.world_id, found.ticket.id)
        async with pool.connection() as conn:
            fresh = await store.by_key(conn, workspace, key)
            if fresh is not None:
                tell(await _detail(conn, fresh))

    async with pool.connection() as conn:
        tell(
            DeskReady(
                agent=agent,
                tickets=await store.tickets(conn, workspace),
                stats=await store.stats(conn, workspace),
            )
        )
    tasks = [asyncio.create_task(pump()), asyncio.create_task(watch())]
    try:
        while True:
            try:
                async with asyncio.timeout(settings.idle_timeout_s):
                    raw = await _text(socket)
            except TimeoutError:
                await socket.close(code=_IDLE, reason="idle")
                return
            if tasks[0].done():
                return  # the writer closed the socket
            try:
                message = DESK_CLIENT_ADAPTER.validate_json(raw)
            except ValidationError:
                refuse("bad_message", "not a message this desk understands")
                continue
            if isinstance(message, DeskHello):
                refuse("bad_message", "this desk has said hello already")
            elif not actions.allow("desk"):
                refuse("rate_limited", "too many actions, wait a moment", message.key)
            else:
                await act(message)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
