"""The backend app: a health check and one WebSocket that carries everything else.

Run with: uv run uvicorn --factory supportagent.api.app:create_app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlsplit

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from supportagent.api.hub import Hubs, Subscription, TooManySessions
from supportagent.api.protocol import (
    CLIENT_ADAPTER,
    FaultInject,
    Hello,
    TurnCancel,
    TurnRetry,
    UserMessage,
)
from supportagent.config import Settings, get_settings
from supportagent.events import CustomerSummary, ProtocolError, SessionReady, Ticket
from supportagent.runtime import Runtime, open_runtime

_POLICY_VIOLATION = 1008  # WebSocket close codes
_TRY_AGAIN_LATER = 1013


async def _pump(socket: WebSocket, backlog: list[BaseModel], subscription: Subscription) -> None:
    """The only writer on this socket: the backlog first, then live events, each at most once."""
    sent = 0
    for event in backlog:
        await socket.send_text(event.model_dump_json())
        sent = max(sent, event.seq)
    while True:
        if subscription.lost and subscription.queue.empty():
            # The client read too slowly and events were dropped. It reconnects with the last
            # number it saw and gets them from the store.
            await socket.close(code=_TRY_AGAIN_LATER, reason="fell behind, reconnect")
            return
        event = await subscription.queue.get()
        if 0 < event.seq <= sent:
            continue  # it was in the backlog already
        await socket.send_text(event.model_dump_json())
        sent = max(sent, event.seq)


def _normal(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def origin_allowed(origin: str | None, host: str | None, settings: Settings) -> bool:
    """Whether a page from `origin` may open a session.

    Browsers always send Origin, and a page cannot forge it. A client that sends none is not a
    browser, so the check protects nobody there. It is still refused when `require_origin` is set.
    """
    if not origin or not origin.strip():
        return not settings.require_origin
    if host and urlsplit(origin).netloc.lower() == host.lower():
        return True  # the page was served by this backend
    return _normal(origin) in {_normal(allowed) for allowed in settings.allowed_origins}


async def _receive_text(socket: WebSocket) -> str | None:
    """The next text frame, or None for a binary one. Raises WebSocketDisconnect at the end."""
    message = await socket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    return message.get("text")


def create_app(rt: Runtime | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass a ready runtime. In production the app opens its own."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with _serving_runtime(rt, settings) as runtime:
            hubs = app.state.hubs = Hubs(runtime, settings.max_sessions)
            try:
                yield
            finally:
                # Turns still running are stopped and stored while the database is still there.
                await hubs.close()

    app = FastAPI(title="Support agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET"]
    )

    @app.get("/health")
    async def health(response: Response) -> dict[str, bool]:
        """Healthy means this process can reach its database."""
        try:
            async with asyncio.timeout(2.0), app.state.hubs.rt.pool.connection() as conn:
                await conn.execute("SELECT 1")
        except Exception:
            response.status_code = 503
            return {"ok": False}
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        headers = socket.headers
        if not origin_allowed(headers.get("origin"), headers.get("host"), settings):
            await socket.close(code=_POLICY_VIOLATION)
            return
        await socket.accept()
        hubs: Hubs = socket.app.state.hubs

        try:
            async with asyncio.timeout(settings.hello_timeout_s):
                hello = CLIENT_ADAPTER.validate_json(await _receive_text(socket) or "")
        except (ValidationError, WebSocketDisconnect, TimeoutError):
            hello = None
        if not isinstance(hello, Hello):
            with suppress(RuntimeError):
                await socket.close(code=_POLICY_VIOLATION, reason="the first message must be hello")
            return

        try:
            hub, resumed = await hubs.open(hello.session_id)
        except TooManySessions:
            await socket.close(code=_TRY_AGAIN_LATER, reason="too many sessions, try again later")
            return
        backlog, subscription, newest = await hub.attach(hello.last_seq if resumed else 0)
        pump: asyncio.Task[None] | None = None

        def refuse(code: str, message: str) -> None:
            subscription.offer(ProtocolError(code=code, message=message))

        try:
            state = await hubs.rt.graph.aget_state({"configurable": {"thread_id": hub.session_id}})
            customer = state.values.get("customer")
            ready = SessionReady(
                session_id=hub.session_id,
                last_seq=newest,
                customer=CustomerSummary(**customer) if customer else None,
                tickets=[Ticket(**t) for t in state.values.get("tickets", [])],
                busy=hub.busy,
                resumable=not hub.busy and bool(state.next),
                fault_injection=settings.fault_injection,
            )
            pump = asyncio.create_task(_pump(socket, [ready, *backlog], subscription))
            while True:
                raw = await _receive_text(socket)
                try:
                    message = CLIENT_ADAPTER.validate_json(raw or "")
                except ValidationError:
                    refuse("bad_message", "not a message this server understands")
                    continue
                if isinstance(message, UserMessage):
                    text = message.text.replace("\x00", "")
                    if len(text) > settings.max_message_chars:
                        refuse("too_long", f"at most {settings.max_message_chars} characters")
                    elif not text.strip():
                        refuse("bad_message", "the message is empty")
                    elif not hub.start_turn(text, message.mode):
                        refuse("turn_in_progress", "wait for the current turn to finish")
                elif isinstance(message, TurnRetry):
                    if hub.busy:
                        refuse("turn_in_progress", "wait for the current turn to finish")
                    elif not await hub.resumable() or not hub.start_turn(None, "text"):
                        refuse("nothing_to_retry", "no unfinished turn to pick up")
                elif isinstance(message, TurnCancel):
                    if not hub.cancel():
                        refuse("nothing_to_cancel", "no turn is running")
                elif isinstance(message, FaultInject):
                    if not settings.fault_injection:
                        refuse("not_allowed", "fault injection is switched off on this server")
                    elif not hub.inject(message):
                        refuse("bad_message", "no tool with that name")
                else:
                    refuse("bad_message", "hello is only valid as the first message")
        except WebSocketDisconnect:
            pass
        finally:
            # Stop the writer and collect its result, so a send that failed on a dead socket
            # does not surface later as an unhandled task error.
            if pump is not None:
                pump.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pump
            # Detaching also lets the hub go once it has no client and no running turn.
            hub.detach(subscription)

    return app


@asynccontextmanager
async def _serving_runtime(rt: Runtime | None, settings: Settings) -> AsyncIterator[Runtime]:
    """The runtime the app serves from: the one it was given, or one it opens and closes itself."""
    if rt is not None:
        yield rt
        return
    async with open_runtime(settings) as opened:
        async with opened.pool.connection() as conn:
            # Turns that a previous process was running when it stopped are over. Their sessions
            # must not stay marked as busy.
            await conn.execute("UPDATE sessions SET turn_running = false WHERE turn_running")
        yield opened
