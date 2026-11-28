"""The backend app: a health check and two WebSockets. `/ws` carries a customer's conversation,
`/desk` the service desk of a workspace.

Run with: uv run uvicorn --factory supportagent.api.app:create_app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from supportagent.api.desk import serve_desk
from supportagent.api.hub import Hubs, SessionHub, Subscription, TooManyClients, TooManySessions
from supportagent.api.limits import Allowance, Budget, Counter, SlidingWindow, client_address
from supportagent.api.protocol import (
    CLIENT_ADAPTER,
    FaultInject,
    Hello,
    PassageGet,
    TraceGet,
    TurnCancel,
    TurnRetry,
    UserMessage,
    VoiceInterrupt,
    VoicePlayhead,
    VoiceStart,
    VoiceStop,
)
from supportagent.config import Settings, get_settings
from supportagent.crm import world as worlds
from supportagent.desk.store import workspace_id
from supportagent.eventlog import LEASE_S
from supportagent.events import (
    TURN_ENDED,
    CustomerSummary,
    PassageDetail,
    ProtocolError,
    SessionCaughtUp,
    SessionReady,
    Ticket,
    TraceList,
)
from supportagent.kb.retrieval import may_read
from supportagent.runtime import Runtime, open_runtime
from supportagent.voice.deepgram import SpeechConfig, SpeechError
from supportagent.voice.session import VoiceLimits, VoiceSession

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


async def _purge_forever(rt: Runtime, hubs: Hubs, ttl_hours: float) -> None:
    """Delete conversations nobody has used for the limit, now and then while the app runs."""
    while True:
        try:
            async with rt.pool.connection() as conn:
                gone = await worlds.purge_worlds(
                    conn,
                    # Not sooner than a day: the daily budgets are read from the stored
                    # events, and a conversation purged earlier would take its spend along.
                    timedelta(hours=max(ttl_hours, 24.0)),
                    timedelta(seconds=LEASE_S),
                    keep=hubs.live(),
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

    Events go out as text frames. The reply's audio, for the connection with the microphone,
    goes out as binary frames in between, in the order it was made.

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
            if isinstance(event, bytes):
                async with asyncio.timeout(send_timeout_s):
                    await socket.send_bytes(event)
                continue
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


async def _receive(socket: WebSocket) -> str | bytes:
    """The next frame: text is a message, bytes are microphone audio.

    Raises WebSocketDisconnect at the end.
    """
    message = await socket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    return text if text is not None else message.get("bytes") or b""


def create_app(rt: Runtime | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass a ready runtime. In production the app opens its own."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with _serving_runtime(rt, settings) as runtime:
            hubs = app.state.hubs = Hubs(runtime, settings.max_sessions)
            app.state.budget = Budget(
                runtime.pool, settings.daily_budget_usd, settings.daily_speech_budget_usd
            )
            purge = (
                asyncio.create_task(_purge_forever(runtime, hubs, settings.session_ttl_hours))
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
    speech = SpeechConfig.from_settings(settings)  # None: this server has no voice
    voice_limits = VoiceLimits(
        max_seconds=settings.voice_max_seconds,
        idle_seconds=settings.voice_idle_seconds,
        greeting=settings.voice_greeting,
        max_chars=settings.max_message_chars,
    )
    voices: set[VoiceSession] = set()  # the ones that are on, in this process
    # What one visitor may use of voice. Every start opens two connections to the speech
    # service, and every second of an open microphone is paid for whether anyone speaks or not.
    voice_starts = SlidingWindow(settings.max_voice_starts_per_minute, 60.0)
    voice_lines = Counter(settings.max_voice_sessions_per_address)
    voice_time = Allowance(settings.voice_seconds_per_address_per_day)
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

    async def _admit(socket: WebSocket) -> str | None:
        """Accept a socket from an allowed origin and count it. Returns who it is from."""
        headers = socket.headers
        over_tls = headers.get("x-forwarded-proto", socket.url.scheme).lower() in ("https", "wss")
        if not origin_allowed(
            headers.get("origin"), headers.get("host"), settings, over_tls=over_tls
        ):
            await socket.close(code=POLICY_VIOLATION)
            return None
        visitor = client_address(
            socket.client.host if socket.client else None,
            headers.get(settings.client_address_header) if settings.client_address_header else None,
        )
        await socket.accept()
        if not connections.take(visitor):
            await socket.close(code=TOO_MANY_CONNECTIONS, reason="too many connections")
            return None
        return visitor

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        visitor = await _admit(socket)
        if visitor is None:
            return
        try:
            await _serve(socket, visitor)
        finally:
            connections.give_back(visitor)

    @app.websocket("/desk")
    async def desk(socket: WebSocket) -> None:
        visitor = await _admit(socket)
        if visitor is None:
            return
        try:
            await serve_desk(socket, socket.app.state.hubs, settings)
        finally:
            connections.give_back(visitor)

    async def _serve(socket: WebSocket, visitor: str) -> None:
        hubs: Hubs = socket.app.state.hubs
        budget: Budget = socket.app.state.budget

        try:
            async with asyncio.timeout(settings.hello_timeout_s):
                first = await _receive(socket)
                hello = CLIENT_ADAPTER.validate_json(first if isinstance(first, str) else "")
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
        voice: VoiceSession | None = None  # this connection's, while its microphone is on

        def tell(event: BaseModel) -> None:
            subscription.offer(event)

        def refuse(
            code: str, message: str, client_id: str | None = None, chunk_id: str | None = None
        ) -> None:
            tell(ProtocolError(code=code, message=message, client_id=client_id, chunk_id=chunk_id))

        try:
            await hub.join(workspace_id(hello.workspace) if hello.workspace else None)
            state = await hubs.rt.graph.aget_state({"configurable": {"thread_id": hub.session_id}})
            customer = state.values.get("customer")
            # A turn that is storing its last event still counts as running, and that event is
            # already among what this client is about to be sent. Nothing comes after it to
            # say that the session is free, so this message has to.
            ending = any(e.type in TURN_ENDED and e.turn_id == hub.current for e in backlog)
            busy = hub.busy and not ending
            ready = SessionReady(
                session_id=hub.session_id,
                last_seq=newest,
                customer=CustomerSummary(**customer) if customer else None,
                tickets=[Ticket(**t) for t in state.values.get("tickets", [])],
                busy=busy,
                resumable=not busy and bool(state.next) and hub.log.heard is None,
                fault_injection=settings.fault_injection,
                voice=speech is not None,
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
                        raw = await _receive(socket)
                except TimeoutError:
                    if hub.busy:
                        continue  # a turn is running for this client: that is not idle
                    await socket.close(code=IDLE, reason="idle")
                    return
                if pump.done():
                    return  # the writer closed the socket: the client fell behind
                if isinstance(raw, bytes):
                    # Microphone audio. Without voice on it is dropped, and nothing breaks.
                    if voice is not None:
                        await voice.feed(raw)
                    continue
                try:
                    message = CLIENT_ADAPTER.validate_json(raw)
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
                elif isinstance(message, VoiceStart):
                    if voice is not None:
                        await voice.stop("replaced")
                    voice = await _on_voice_start(
                        hub, budget, subscription, message, refuse, visitor
                    )
                elif isinstance(message, VoiceStop):
                    if voice is not None:
                        await voice.stop("client")
                        voice = None
                elif isinstance(message, VoicePlayhead):
                    if voice is not None:
                        voice.playhead(message.played_ms, message.after, message.epoch)
                elif isinstance(message, VoiceInterrupt):
                    if voice is not None:
                        voice.interrupt(message.played_ms, message.epoch)
                else:
                    refuse("bad_message", "hello is only valid as the first message")
                if hub.gone:
                    await socket.close(code=SESSION_EXPIRED, reason="this conversation is gone")
                    return
        except WebSocketDisconnect:
            pass
        finally:
            if voice is not None:
                with suppress(Exception):
                    await voice.stop("client")
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
        blocked = await _blocked(hub, budget, text)
        if blocked is not None:
            refuse(*blocked, client_id)
        elif not hub.start_turn(text, message.mode, client_id):
            refuse("turn_in_progress", "wait for the current turn to finish", client_id)

    async def _blocked(
        hub: SessionHub, budget: Budget, text: str, *, spoken: bool = False
    ) -> tuple[str, str] | None:
        """Why this message may not start a turn, as an error code and its wording.

        A spoken message meets the same limits as a typed one, except that speech may arrive
        while a turn runs: the customer has then spoken over it, and voice stops that turn.
        """
        if len(text) > settings.max_message_chars:
            return "too_long", f"at most {settings.max_message_chars} characters"
        if not text.strip():
            return "bad_message", "the message is empty"
        if hub.busy and not spoken:
            return "turn_in_progress", "wait for the current turn to finish"
        if hub.turns >= settings.max_turns_per_session:
            limit = settings.max_turns_per_session
            return (
                "rate_limited",
                f"this conversation has reached its limit of {limit} messages, start a new one",
            )
        if await budget.exhausted():
            return "budget_exhausted", "the demo has used its budget for today"
        if spoken and await budget.speech_exhausted():
            return "budget_exhausted", "voice has used its budget for today"
        return None

    async def _on_voice_start(
        hub: SessionHub,
        budget: Budget,
        subscription: Subscription,
        message: VoiceStart,
        refuse: Refuse,
        visitor: str,
    ) -> VoiceSession | None:
        """Turn voice on for this connection, or say why not. The refusal names the attempt."""

        def no(code: str, why: str) -> None:
            refuse(code, why, message.attempt)

        if speech is None:
            return no("voice_unavailable", "it is not set up on this server")
        if not voice_starts.allow(visitor):
            return no("voice_unavailable", "it was turned on too often, try again in a minute")
        if await budget.exhausted():
            return no("budget_exhausted", "the demo has used its budget for today")
        if await budget.speech_exhausted():
            return no("voice_unavailable", "it has used its budget for today")
        left = voice_time.left(visitor)
        if left <= 0:
            return no("voice_unavailable", "this address has used its voice time for today")
        # Whoever has the conversation's voice gives it up first. Looked at again after each
        # wait: a voice that is being stopped has left its place already, and another window
        # that asked in that moment may have taken it. Taken over without a second look, that
        # one stayed on, and the conversation had two voices.
        while (other := hub.voice) is not None:
            await other.stop("replaced", "it is now on in another window")
            if hub.voice is other:
                hub.voice = None  # it has stopped, whatever its shutdown left standing
        if len(voices) >= settings.max_voice_sessions:
            return no("voice_unavailable", "all voice lines are busy")
        if not voice_lines.take(visitor):
            return no("voice_unavailable", "it is already on in other windows of yours")

        async def spend(seconds: float) -> str | None:
            """Count microphone time against the visitor's day, and look at the day's budget."""
            voice_time.use(visitor, seconds)
            if voice_time.left(visitor) <= 0:
                return "this address has used its voice time for today"
            if await budget.speech_exhausted():
                return "voice has used its budget for today"
            return None

        def closed(ended: VoiceSession) -> None:
            voices.discard(ended)
            voice_lines.give_back(visitor)

        async def admit(text: str) -> ProtocolError | None:
            blocked = await _blocked(hub, budget, text, spoken=True)
            return ProtocolError(code=blocked[0], message=blocked[1]) if blocked else None

        voice = VoiceSession(
            hub,
            subscription,
            speech,
            dataclasses.replace(voice_limits, max_seconds=min(voice_limits.max_seconds, left)),
            admit=admit,
            spend=spend,
            barge_in=message.echo_cancellation,
            attempt=message.attempt,
            on_closed=closed,
        )
        # Both places are taken before the sockets open: the count of voices, and the
        # conversation's one voice. Two windows that ask at the same moment would otherwise
        # both find the conversation free, and it would get two voices.
        voices.add(voice)
        hub.voice = voice
        try:
            await voice.start()
        except SpeechError as exc:
            if hub.voice is voice:
                hub.voice = None
            if not voice.stopped:
                closed(voice)
                logger.warning("voice could not start: %s", exc)
                no("voice_unavailable", "speech is unavailable right now")
            return None  # stopped while it was starting: the stop has told the client
        except BaseException:
            # A fault of ours, or the server going down. The places go back all the same: a
            # voice that never started must not hold one of the few lines for good.
            if hub.voice is voice:
                hub.voice = None
            if not voice.stopped:
                closed(voice)
            raise
        return voice

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
            refuse("not_found", "no such passage in this conversation", chunk_id=message.chunk_id)
            return
        async with hub.rt.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT doc_title, heading_path, tiers, content FROM kb_chunks WHERE id = %s",
                (message.chunk_id,),
            )
            row = await cur.fetchone()
        if row is None:  # the help center changed since
            refuse(
                "not_found",
                "this passage is no longer in the help center",
                chunk_id=message.chunk_id,
            )
            return
        # The plan filter once more, for who the conversation is with now. After a lockout or
        # a change of account, a passage that an earlier customer's plan could read is not
        # handed out to whoever has the conversation next.
        state = await hub.rt.graph.aget_state({"configurable": {"thread_id": hub.session_id}})
        customer = state.values.get("customer")
        if not may_read(row["tiers"], customer["tier"] if customer else None):
            refuse("not_found", "no such passage in this conversation", chunk_id=message.chunk_id)
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
