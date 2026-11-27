"""The backend app: a health check and one WebSocket that carries everything else.

Run with: uv run uvicorn --factory supportagent.api.app:create_app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from supportagent.api.hub import Hubs, SessionHub, Subscription, TooManyClients, TooManySessions
from supportagent.api.limits import Budget, Counter, SlidingWindow, client_address
from supportagent.api.protocol import (
    CLIENT_ADAPTER,
    FaultInject,
    Hello,
    PassageGet,
    TraceGet,
    TurnCancel,
    TurnRetry,
    UserMessage,
)
from supportagent.config import Settings, get_settings
from supportagent.crm import world as worlds
from supportagent.eventlog import LEASE_S
from supportagent.events import (
    CustomerSummary,
    PassageDetail,
    ProtocolError,
    SessionCaughtUp,
    SessionReady,
    Ticket,
    TraceList,
)
from supportagent.runtime import Runtime, open_runtime

logger = logging.getLogger(__name__)

# WebSocket close codes. The ones from 4000 are this app's own, so a client can tell why it was
# closed without reading the reason text, and decide whether coming straight back makes sense.
POLICY_VIOLATION = 1008  # a page from another origin, or a first message that is not hello
FELL_BEHIND = 4000  # the client read too slowly: reconnect now and catch up from the store
IDLE = 4001  # nothing was sent for a long time: reconnect when the visitor is back
TOO_MANY_CONVERSATIONS = 4002  # this address started too many: wait
AT_CAPACITY = 4003  # the server has no room for another conversation: wait
TOO_MANY_CONNECTIONS = 4004  # this address or this conversation has too many open: close some
SESSION_EXPIRED = 4005  # the conversation was deleted: start a new one

_PURGE_EVERY_S = 3600.0
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


async def _purge_forever(rt: Runtime, ttl_hours: float) -> None:
    """Delete conversations nobody has used for the limit, now and then while the app runs."""
    while True:
        try:
            async with rt.pool.connection() as conn:
                gone = await worlds.purge_worlds(
                    conn, timedelta(hours=ttl_hours), timedelta(seconds=LEASE_S)
                )
                orphaned = await worlds.forget_orphaned_threads(conn)
            if gone or orphaned:
                logger.info(
                    "purged %s conversations unused for %s hours and %s saved states with no "
                    "session",
                    gone,
                    ttl_hours,
                    orphaned,
                )
        except Exception:
            logger.exception("purging old conversations failed")
        await asyncio.sleep(_PURGE_EVERY_S)


async def _pump(
    socket: WebSocket, backlog: list[BaseModel], subscription: Subscription, send_timeout_s: float
) -> None:
    """The only writer on this socket: the backlog first, then live events, each at most once.

    A client that cannot take one event within `send_timeout_s` is closed. Without that limit a
    stalled reader would keep its queue, and the memory in it, for as long as it liked.
    """

    async def send(event: BaseModel) -> None:
        async with asyncio.timeout(send_timeout_s):
            await socket.send_text(event.model_dump_json())

    sent = 0
    try:
        for event in backlog:
            await send(event)
            sent = max(sent, event.seq)
        while True:
            if subscription.lost:
                # The client read too slowly and events were dropped. It reconnects with the
                # last number it saw and gets them from the store.
                await socket.close(code=FELL_BEHIND, reason="fell behind, reconnect")
                return
            event = await subscription.queue.get()
            if 0 < event.seq <= sent:
                continue  # it was in the backlog already
            await send(event)
            sent = max(sent, event.seq)
    except TimeoutError:
        with suppress(Exception):
            await socket.close(code=FELL_BEHIND, reason="fell behind, reconnect")


def _normal(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def origin_allowed(
    origin: str | None, host: str | None, settings: Settings, *, over_tls: bool = False
) -> bool:
    """Whether a page from `origin` may open a session.

    Browsers always send Origin, and a page cannot forge it. A client that sends none is not a
    browser, so the check protects nobody there. It is still refused when `require_origin` is set.

    A page served by this backend's own host is allowed, over http or https and nothing else. When
    the request itself came over TLS, as it does behind a hosting proxy, the page must be https
    too: a page loaded in the clear can have been altered on the way.
    """
    if not origin or not origin.strip():
        return not settings.require_origin
    if _normal(origin) in {_normal(allowed) for allowed in settings.allowed_origins}:
        return True
    parts = urlsplit(origin.strip())
    same_host = bool(host) and parts.netloc.lower() == host.lower()
    return same_host and parts.scheme.lower() in (("https",) if over_tls else ("http", "https"))


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
            app.state.budget = Budget(runtime.pool, settings.daily_budget_usd)
            purge = (
                asyncio.create_task(_purge_forever(runtime, settings.session_ttl_hours))
                if settings.session_ttl_hours is not None
                else None
            )
            try:
                yield
            finally:
                if purge is not None:
                    purge.cancel()
                    with suppress(asyncio.CancelledError):
                        await purge
                # Turns still running are stopped and stored while the database is still there.
                await hubs.close()

    # No interactive docs and no schema: the app has one public route, and listing its internals
    # helps nobody who should be here.
    app = FastAPI(
        title="Support agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    new_sessions = SlidingWindow(settings.max_new_sessions_per_hour, 3600.0)
    connections = Counter(settings.max_connections_per_address)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_normal(origin) for origin in settings.allowed_origins],
        allow_methods=["GET"],
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/health")
    async def health(response: Response) -> dict[str, bool]:
        """Healthy means this process has loaded its models and can reach its database.

        The app answers nothing before its start-up has finished, so the first healthy reply also
        says the embedding model is warm. A host that waits for it sends no visitor to a process
        that would make them wait for the weights.
        """
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
        over_tls = headers.get("x-forwarded-proto", socket.url.scheme).lower() in ("https", "wss")
        if not origin_allowed(
            headers.get("origin"), headers.get("host"), settings, over_tls=over_tls
        ):
            await socket.close(code=POLICY_VIOLATION)
            return
        visitor = client_address(
            socket.client.host if socket.client else None,
            headers.get(settings.client_address_header) if settings.client_address_header else None,
        )
        await socket.accept()
        if not connections.take(visitor):
            await socket.close(code=TOO_MANY_CONNECTIONS, reason="too many connections")
            return
        try:
            await _serve(socket, visitor)
        finally:
            connections.give_back(visitor)

    async def _serve(socket: WebSocket, visitor: str) -> None:
        hubs: Hubs = socket.app.state.hubs
        budget: Budget = socket.app.state.budget

        try:
            async with asyncio.timeout(settings.hello_timeout_s):
                hello = CLIENT_ADAPTER.validate_json(await _receive_text(socket) or "")
        except (ValidationError, WebSocketDisconnect, TimeoutError):
            hello = None
        if not isinstance(hello, Hello):
            with suppress(RuntimeError):
                await socket.close(code=POLICY_VIOLATION, reason="the first message must be hello")
            return

        # Coming back to a conversation is free. Starting one is counted per visitor, and an ID
        # the server does not know starts one. Room is checked first: a visitor turned away
        # because the server is full has started nothing and is charged nothing.
        known = await hubs.known(hello.session_id)
        if not known:
            if hubs.full:
                await socket.close(code=AT_CAPACITY, reason="too many sessions, try again later")
                return
            if not new_sessions.allow(visitor):
                await socket.close(
                    code=TOO_MANY_CONVERSATIONS,
                    reason="too many new conversations, try again later",
                )
                return
        try:
            hub, resumed = await hubs.open(hello.session_id if known else None)
        except TooManySessions:
            if not known:
                new_sessions.undo(visitor)
            await socket.close(code=AT_CAPACITY, reason="too many sessions, try again later")
            return
        try:
            backlog, subscription, newest = await hub.attach(hello.last_seq if resumed else 0)
        except TooManyClients:
            await socket.close(
                code=TOO_MANY_CONNECTIONS, reason="too many connections to this conversation"
            )
            return
        pump: asyncio.Task[None] | None = None

        def tell(event: BaseModel) -> None:
            subscription.offer(event)

        def refuse(code: str, message: str, client_id: str | None = None) -> None:
            tell(ProtocolError(code=code, message=message, client_id=client_id))

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
            pump = asyncio.create_task(
                _pump(
                    socket,
                    [ready, *backlog, SessionCaughtUp()],
                    subscription,
                    settings.send_timeout_s,
                )
            )
            while True:
                try:
                    async with asyncio.timeout(settings.idle_timeout_s):
                        raw = await _receive_text(socket)
                except TimeoutError:
                    if hub.busy:
                        continue  # a turn is running for this client: that is not idle
                    await socket.close(code=IDLE, reason="idle")
                    return
                if pump.done():
                    return  # the writer closed the socket: the client fell behind
                try:
                    message = CLIENT_ADAPTER.validate_json(raw or "")
                except ValidationError:
                    refuse("bad_message", "not a message this server understands")
                    continue
                if isinstance(message, UserMessage):
                    await _on_message(hub, budget, message, refuse)
                elif isinstance(message, TurnRetry):
                    await _on_retry(hub, budget, refuse)
                elif isinstance(message, TurnCancel):
                    if not hub.cancel():
                        refuse("nothing_to_cancel", "no turn is running")
                elif isinstance(message, FaultInject):
                    if not settings.fault_injection:
                        refuse("not_allowed", "fault injection is switched off on this server")
                    elif not hub.inject(message):
                        refuse("bad_message", "no tool with that name")
                elif isinstance(message, TraceGet):
                    traces = await hub.log.traces(message.turn_id)
                    tell(TraceList(turn_id=message.turn_id, traces=traces))
                elif isinstance(message, PassageGet):
                    await _on_passage(hub, message, tell, refuse)
                else:
                    refuse("bad_message", "hello is only valid as the first message")
                if hub.gone:
                    await socket.close(code=SESSION_EXPIRED, reason="this conversation is gone")
                    return
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

    Refuse = Callable[..., None]

    async def _on_message(
        hub: SessionHub, budget: Budget, message: UserMessage, refuse: Refuse
    ) -> None:
        """Start a turn for one customer message, or say why not. Either way the client hears
        about this message by its `client_id`."""
        text, client_id = message.text.replace("\x00", ""), message.client_id
        if len(text) > settings.max_message_chars:
            refuse("too_long", f"at most {settings.max_message_chars} characters", client_id)
        elif not text.strip():
            refuse("bad_message", "the message is empty", client_id)
        elif hub.busy:
            refuse("turn_in_progress", "wait for the current turn to finish", client_id)
        elif hub.turns >= settings.max_turns_per_session:
            limit = settings.max_turns_per_session
            refuse(
                "rate_limited",
                f"this conversation has reached its limit of {limit} messages, start a new one",
                client_id,
            )
        elif await budget.exhausted():
            refuse("budget_exhausted", "the demo has used its budget for today", client_id)
        elif not hub.start_turn(text, message.mode, client_id):
            refuse("turn_in_progress", "wait for the current turn to finish", client_id)

    async def _on_retry(hub: SessionHub, budget: Budget, refuse: Refuse) -> None:
        """Pick an unfinished turn up, within the limits a new message would also meet."""
        if hub.busy:
            refuse("turn_in_progress", "wait for the current turn to finish")
            return
        can = await hub.resumable()
        used = await hub.retries_used() if can else 0
        if hub.busy:  # another connection got there while this one was asking the database
            refuse("turn_in_progress", "wait for the current turn to finish")
        elif not can:
            refuse("nothing_to_retry", "no unfinished turn to pick up")
        elif used > settings.max_retries_per_turn:
            # Each pick-up runs model calls. Without a limit one message could be run for ever.
            refuse("too_many_retries", "this message was retried too often, send it again")
        elif await budget.exhausted():
            refuse("budget_exhausted", "the demo has used its budget for today")
        elif not hub.start_turn(None, "text"):
            refuse("turn_in_progress", "wait for the current turn to finish")

    async def _on_passage(
        hub: SessionHub, message: PassageGet, tell: Callable[[BaseModel], None], refuse: Refuse
    ) -> None:
        """Send back one passage, if a search of this conversation gave it to the model.

        That rule is the plan filter again: a search only returns what the customer's plan may
        read, so a passage it never returned is not handed out here either.
        """
        if not await hub.log.was_shown(message.chunk_id):
            refuse("not_found", "no such passage in this conversation")
            return
        async with hub.rt.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT doc_title, heading_path, tiers, content FROM kb_chunks WHERE id = %s",
                (message.chunk_id,),
            )
            row = await cur.fetchone()
        if row is None:  # the help center changed since
            refuse("not_found", "this passage is no longer in the help center")
            return
        tell(
            PassageDetail(
                chunk_id=message.chunk_id,
                source=" > ".join((row["doc_title"], *row["heading_path"])),
                tiers=row["tiers"],
                content=row["content"],
            )
        )

    return app


@asynccontextmanager
async def _serving_runtime(rt: Runtime | None, settings: Settings) -> AsyncIterator[Runtime]:
    """The runtime the app serves from: the one it was given, or one it opens and closes itself.

    Nothing is cleared at start-up. A turn that a dead process left claimed frees itself when its
    lease runs out, and a turn that a live process is still finishing keeps its claim.
    """
    if rt is not None:
        yield rt
        return
    async with open_runtime(settings) as opened:
        # Load the local models before the first request, not during it.
        await opened.retriever.warm()
        yield opened
