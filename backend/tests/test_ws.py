"""The WebSocket server, driven by a real client over a real socket, with a scripted model."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import uvicorn
from agent_harness import ALWAYS_ANSWER, FAST, REFUND_QUESTION, llm_for, runtime
from fakes import ScriptedModel, call, http_error, say, usage
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from supportagent.api.app import (
    AT_CAPACITY,
    IDLE,
    SESSION_EXPIRED,
    TOO_MANY_CONNECTIONS,
    TOO_MANY_CONVERSATIONS,
    create_app,
    origin_allowed,
)
from supportagent.api.limits import Budget, Counter, SlidingWindow, client_address
from supportagent.config import get_settings
from supportagent.crm import world as worlds
from supportagent.llm.client import ResilientLLM

ORIGIN = "http://ui.test"


@pytest.fixture
async def serve(pool, retriever):
    """Start the app on a free port with the given model. Returns the runtime and the address."""
    running = []

    async def start(model, capture_prompts=False, **overrides):
        rt = runtime(pool, retriever, model, capture_prompts=capture_prompts)
        settings = get_settings().model_copy(
            update={
                "allowed_origins": [ORIGIN],
                "max_message_chars": 200,
                "fault_injection": True,
                **overrides,
            }
        )
        # As in production: uvicorn does not rewrite the client address from X-Forwarded-For.
        rt.app = create_app(rt, settings)
        config = uvicorn.Config(
            rt.app,
            host="127.0.0.1",
            port=0,
            log_level="error",
            proxy_headers=False,
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        running.append((server, task))
        port = server.servers[0].sockets[0].getsockname()[1]
        return rt, f"127.0.0.1:{port}"

    yield start
    for server, task in running:
        server.should_exit = True
        await task


class Client:
    def __init__(self, ws):
        self.ws, self.events, self.ready = ws, [], None

    @classmethod
    async def open(cls, address, session_id=None, last_seq=0, origin=ORIGIN, headers=None):
        socket = await connect(f"ws://{address}/ws", origin=origin, additional_headers=headers)
        client = cls(socket)
        hello = {"type": "hello", "session_id": session_id, "last_seq": last_seq}
        await client.ws.send(json.dumps(hello))
        client.ready = await client.next()
        return client

    async def send(self, **message):
        await self.ws.send(json.dumps(message))

    async def say(self, text, client_id=None):
        await self.send(type="user.message", text=text, client_id=client_id)

    async def next(self):
        """The next event. The marker that ends a replay is noted and passed over."""
        while True:
            event = json.loads(await asyncio.wait_for(self.ws.recv(), 5))
            self.events.append(event)
            if event["type"] != "session.caught_up":
                return event

    async def until(self, kind):
        while True:
            event = await self.next()
            if event["type"] == kind:
                return event

    def stream(self):
        """The numbered events received so far, without direct replies such as session.ready."""
        return [e for e in self.events if e["seq"]]

    def errors(self):
        return [e["code"] for e in self.events if e["type"] == "error"]


async def test_a_turn_streams_numbered_events_to_the_client(serve):
    _, address = await serve(ScriptedModel("m", say("Hello there.")))
    async with httpx.AsyncClient() as http:
        assert (await http.get(f"http://{address}/health")).json() == {"ok": True}

    client = await Client.open(address)
    assert (client.ready["type"], client.ready["busy"]) == ("session.ready", False)
    assert client.ready["last_seq"] == 0 and client.ready["customer"] is None
    assert len(client.ready["session_id"]) == 32  # the only key to the session
    await client.say("Hi")
    await client.until("turn.finished")
    turn = client.stream()
    assert (turn[0]["type"], turn[0]["text"]) == ("turn.started", "Hi")
    assert [e["seq"] for e in turn] == list(range(1, len(turn) + 1))
    assert next(e for e in turn if e["type"] == "assistant.message")["text"] == "Hello there."
    await client.ws.close()


async def test_a_client_that_drops_mid_turn_gets_the_rest_exactly_once(serve):
    slow = ScriptedModel("m", ["Hel", 0.25, "lo ", 0.25, "there.", usage(10, 3)])
    _, address = await serve(slow)
    first = await Client.open(address)
    await first.say("Hi")
    await first.until("llm.delta")
    seen = first.stream()
    await first.ws.close()

    second = await Client.open(address, first.ready["session_id"], last_seq=seen[-1]["seq"])
    assert second.ready["busy"] is True
    await second.until("turn.finished")
    everything = seen + second.stream()
    assert [e["seq"] for e in everything] == list(range(1, len(everything) + 1))
    text = "".join(e["text"] for e in everything if e["type"] == "llm.delta")
    assert text == "Hello there."
    await second.ws.close()


async def test_a_client_that_returns_later_gets_what_it_missed_from_storage(serve):
    _, address = await serve(ScriptedModel("m", say("Hello there.")))
    first = await Client.open(address)
    await first.say("Hi")
    done = await first.until("turn.finished")
    await first.ws.close()
    stored = [e for e in first.stream() if e["type"] != "llm.delta"]

    # This client saw only the first event of the turn before it went away.
    second = await Client.open(address, first.ready["session_id"], last_seq=stored[0]["seq"])
    assert (second.ready["last_seq"], second.ready["busy"]) == (done["seq"], False)
    replay = [await second.next() for _ in stored[1:]]
    assert replay == stored[1:]
    await second.ws.close()


async def test_a_client_that_arrives_while_a_finished_turn_is_tidied_up_reads_it_from_storage(
    serve, monkeypatch
):
    rt, address = await serve(ScriptedModel("m", say("Hello there.")))

    async def slow(_session_id):
        await asyncio.sleep(0.3)

    # The turn tells its clients that it is over, then tidies up its saved state for a while.
    monkeypatch.setattr(rt, "prune", slow)
    first = await Client.open(address)
    await first.say("Hi")
    await first.until("turn.finished")
    second = await Client.open(address, first.ready["session_id"])
    stored = [e for e in first.stream() if e["type"] != "llm.delta"]
    # What the store holds, and none of the token deltas the hub had buffered for the turn.
    assert [await second.next() for _ in stored] == stored
    assert second.ready["busy"] is False
    await first.ws.close()
    await second.ws.close()


async def test_a_second_message_during_a_turn_is_refused(serve):
    _, address = await serve(ScriptedModel("m", ["Hel", 0.25, "lo.", usage(10, 3)]))
    client = await Client.open(address)
    await client.say("first")
    await client.say("second")
    await client.until("turn.finished")
    assert client.errors() == ["turn_in_progress"]
    assert sum(e["type"] == "turn.started" for e in client.events) == 1
    await client.ws.close()


async def test_the_session_is_free_the_moment_the_client_hears_that_the_turn_ended(serve):
    _, address = await serve(ScriptedModel("m", say("One."), say("Two.")))
    client = await Client.open(address)
    await client.say("first")
    await client.until("turn.finished")
    await client.say("second")  # sent at once, with no pause
    await client.until("turn.finished")
    assert client.errors() == []
    texts = [e["text"] for e in client.events if e["type"] == "assistant.message"]
    assert texts == ["One.", "Two."]
    await client.ws.close()


async def test_bad_messages_are_refused_and_the_connection_survives(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    client = await Client.open(address)
    await client.ws.send("this is not json")
    assert (await client.next())["code"] == "bad_message"
    await client.ws.send(b"\x00\x01 audio, one day")
    assert (await client.next())["code"] == "bad_message"
    await client.say("x" * 500)
    assert (await client.next())["code"] == "too_long"
    await client.send(type="turn.retry")
    assert (await client.next())["code"] == "nothing_to_retry"
    await client.send(type="turn.cancel")
    assert (await client.next())["code"] == "nothing_to_cancel"
    await client.say("Hi")
    assert (await client.until("assistant.message"))["text"] == "Hello."
    await client.ws.close()


async def test_a_failed_turn_is_retried_over_the_socket_without_waiting(serve):
    rt, address = await serve(ScriptedModel("down", [http_error(401)]))
    client = await Client.open(address)
    await client.say("Hello?")
    assert (await client.until("turn.failed"))["retryable"] is True

    rt.llm = llm_for(ScriptedModel("up", say("Yes, I'm here.")))
    await client.send(type="turn.retry")  # the failed turn is already stored and released
    assert (await client.until("assistant.message"))["text"] == "Yes, I'm here."
    assert client.errors() == []
    assert sum(e["type"] == "turn.started" for e in client.events) == 1
    await client.ws.close()


async def test_a_failed_turn_can_still_be_retried_after_the_client_reconnects(serve):
    rt, address = await serve(ScriptedModel("down", [http_error(401)]))
    first = await Client.open(address)
    await first.say("Hello?")
    failed = await first.until("turn.failed")
    await first.ws.close()

    # Nothing in memory remembers the failure now. The saved conversation does.
    rt.llm = llm_for(ScriptedModel("up", say("Yes, I'm here.")))
    second = await Client.open(address, first.ready["session_id"], last_seq=failed["seq"])
    assert second.ready["resumable"] is True
    await second.send(type="turn.retry")
    answer = await second.until("assistant.message")
    assert answer["text"] == "Yes, I'm here."
    assert answer["turn_id"] == failed["turn_id"]
    await second.ws.close()


async def test_a_cancelled_turn_is_reported_and_the_next_one_arrives_in_full(serve):
    model = ScriptedModel("m", ["Hel", 5.0, "lo.", usage(10, 3)], say("Hello again."))
    _, address = await serve(model)
    client = await Client.open(address)
    await client.say("Hi")
    await client.until("llm.delta")
    await client.send(type="turn.cancel")
    interrupted = await client.until("turn.interrupted")

    await client.say("Hi again")
    finished = await client.until("turn.finished")
    # Every event of the second turn reached the client: none was taken for a duplicate.
    second = [e for e in client.stream() if e["seq"] > interrupted["seq"]]
    assert [e["seq"] for e in second] == list(range(interrupted["seq"] + 1, finished["seq"] + 1))
    assert next(e for e in second if e["type"] == "assistant.message")["text"] == "Hello again."
    await client.ws.close()


async def test_an_injected_fault_shows_up_as_a_retry_in_the_stream(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    client = await Client.open(address)
    await client.send(type="fault.inject", kind="llm_server_error")
    await client.send(type="fault.inject", kind="tool_failure", tool="drop_all_tables")
    await client.say("Hi")
    await client.until("turn.finished")
    kinds = [e["type"] for e in client.events]
    assert kinds.index("llm.retry") < kinds.index("assistant.message")
    assert client.errors() == ["bad_message"]  # only real tools can be made to fail
    await client.ws.close()


async def test_a_primary_outage_switches_to_the_fallback_model_and_says_so(serve):
    rt, address = await serve(ScriptedModel("first", say("From the first.")))

    async def no_sleep(_seconds):
        return None

    models = [ScriptedModel("first", say("From the first.")), ScriptedModel("second", say("Hi."))]
    rt.llm = ResilientLLM(models, policy=FAST, sleep=no_sleep, jitter=lambda: 0.0)
    client = await Client.open(address)
    assert client.ready["fault_injection"] is True  # a UI may offer the controls
    await client.send(type="fault.inject", kind="llm_primary_down")
    await client.say("Hi")
    await client.until("turn.finished")
    switch = next(e for e in client.events if e["type"] == "llm.fallback")
    assert (switch["from_model"], switch["to_model"]) == ("first", "second")
    assert next(e for e in client.events if e["type"] == "assistant.message")["text"] == "Hi."
    # The fault was for one call. The next turn is back on the first model.
    await client.say("Again")
    await client.until("turn.finished")
    assert [e["text"] for e in client.events if e["type"] == "assistant.message"][-1] == (
        "From the first."
    )
    await client.ws.close()


async def test_faults_cannot_be_injected_unless_the_server_allows_it(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")), fault_injection=False)
    client = await Client.open(address)
    assert client.ready["fault_injection"] is False
    await client.send(type="fault.inject", kind="llm_down")
    assert (await client.next())["code"] == "not_allowed"
    await client.say("Hi")
    await client.until("turn.finished")
    assert not any(e["type"] in ("llm.retry", "turn.failed") for e in client.events)
    await client.ws.close()


async def test_one_sessions_faults_do_not_touch_another_session(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    noisy, quiet = await Client.open(address), await Client.open(address)
    for _ in range(10):
        await noisy.send(type="fault.inject", kind="llm_down", times=10)
    await noisy.say("Hi")
    assert (await noisy.until("turn.failed"))["retryable"] is True
    await quiet.say("Hi")
    await quiet.until("turn.finished")
    assert not any(e["type"] in ("llm.retry", "llm.fallback") for e in quiet.events)
    await noisy.ws.close()
    await quiet.ws.close()


async def test_a_page_from_an_unknown_origin_cannot_open_a_session(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    with pytest.raises(InvalidStatus):
        await Client.open(address, origin="http://evil.test")
    # A client that is not a browser sends no Origin. It is let in unless the server says no.
    client = await Client.open(address, origin=None)
    assert client.ready["type"] == "session.ready"
    await client.ws.close()

    _, strict = await serve(ScriptedModel("m", say("Hello.")), require_origin=True)
    with pytest.raises(InvalidStatus):
        await Client.open(strict, origin=None)


def test_origins_are_compared_without_case_or_trailing_slash_and_same_origin_is_let_in():
    settings = get_settings().model_copy(
        update={"allowed_origins": ["https://Support.Example/"], "require_origin": True}
    )
    host = "api.example"
    assert origin_allowed("https://support.example", host, settings)
    assert origin_allowed("https://support.example/", host, settings)
    assert origin_allowed("http://api.example", host, settings)  # served by the backend itself
    # Behind a proxy that ends TLS, a page loaded in the clear is not the backend's own page.
    assert not origin_allowed("http://api.example", host, settings, over_tls=True)
    assert origin_allowed("https://api.example", host, settings, over_tls=True)
    assert not origin_allowed("javascript://api.example", host, settings)
    assert not origin_allowed("ftp://api.example", host, settings)
    assert not origin_allowed("https://evil.example", host, settings)
    assert not origin_allowed("https://support.example.evil.test", host, settings)
    assert not origin_allowed("", host, settings)
    assert not origin_allowed(None, host, settings)
    nobody = settings.model_copy(update={"allowed_origins": []})
    assert not origin_allowed("https://support.example", host, nobody)


async def test_one_address_cannot_start_conversations_without_end(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")), max_new_sessions_per_hour=2)
    first = await Client.open(address)
    second = await Client.open(address)
    with pytest.raises(ConnectionClosed) as refused:
        await Client.open(address)
    assert refused.value.rcvd.code == TOO_MANY_CONVERSATIONS
    # Coming back to a conversation that exists is not a new one.
    again = await Client.open(address, session_id=first.ready["session_id"])
    assert again.ready["session_id"] == first.ready["session_id"]
    # An ID the server does not know would start a conversation, so it is counted like one.
    with pytest.raises(ConnectionClosed):
        await Client.open(address, session_id="0" * 32)
    for client in (first, second, again):
        await client.ws.close()


async def test_a_conversation_has_a_limit_of_messages_that_outlives_the_connection(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")), max_turns_per_session=2)
    client = await Client.open(address)
    for text in ("one", "two"):
        await client.say(text)
        await client.until("turn.finished")
    await client.say("three")
    refusal = await client.next()
    assert (refusal["code"], "limit of 2 messages" in refusal["message"]) == ("rate_limited", True)
    await client.ws.close()

    # The count is read from the stored conversation, so reconnecting does not reset it.
    back = await Client.open(address, session_id=client.ready["session_id"])
    await back.say("four")
    assert (await back.until("error"))["code"] == "rate_limited"
    assert sum(e["type"] == "turn.started" for e in back.events) == 2
    await back.ws.close()


async def test_visitors_are_told_apart_by_the_header_the_hosts_proxy_sets(serve):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")),
        max_new_sessions_per_hour=1,
        client_address_header="x-real-ip",
    )
    first = await Client.open(address, headers={"X-Real-IP": "203.0.113.9"})
    # Every test client connects from the same socket address. The header is what tells them apart.
    second = await Client.open(address, headers={"X-Real-IP": "198.51.100.4"})
    with pytest.raises(ConnectionClosed):
        await Client.open(address, headers={"X-Real-IP": "203.0.113.9"})
    await first.ws.close()
    await second.ws.close()


async def test_without_that_setting_no_header_can_buy_a_visitor_more_conversations(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")), max_new_sessions_per_hour=1)
    first = await Client.open(address, headers={"X-Real-IP": "203.0.113.9"})
    for forged in ({"X-Real-IP": "198.51.100.4"}, {"X-Forwarded-For": "198.51.100.4"}):
        with pytest.raises(ConnectionClosed):
            await Client.open(address, headers=forged)
    await first.ws.close()


def test_the_window_lets_the_limit_through_and_forgets_after_its_time():
    now = [0.0]
    window = SlidingWindow(limit=2, window_s=60.0, clock=lambda: now[0])
    assert [window.allow("a"), window.allow("a"), window.allow("a")] == [True, True, False]
    assert window.allow("b")  # another visitor has its own count
    now[0] = 59.0
    assert not window.allow("a")  # a refused attempt was not counted, and the first two still are
    now[0] = 60.0
    assert window.allow("a")


async def frames_until_caught_up(client) -> list[dict]:
    """Everything the server sends a reconnecting client, up to the marker that ends the replay."""
    frames = []
    while not frames or frames[-1]["type"] != "session.caught_up":
        frames.append(json.loads(await asyncio.wait_for(client.ws.recv(), 5)))
    return frames


async def test_every_message_is_answered_under_its_own_id(serve):
    _, address = await serve(ScriptedModel("m", ["Hel", 0.25, "lo.", usage(10, 3)]))
    client = await Client.open(address)
    await client.say("first", client_id="m-1")
    assert (await client.until("turn.started"))["client_id"] == "m-1"
    await client.say("second", client_id="m-2")
    refused = await client.until("error")
    assert (refused["code"], refused["client_id"]) == ("turn_in_progress", "m-2")
    await client.until("turn.finished")
    await client.say("x" * 500, client_id="m-3")
    refused = await client.until("error")
    assert (refused["code"], refused["client_id"]) == ("too_long", "m-3")
    await client.ws.close()


async def test_a_returning_client_is_told_where_the_replay_ends(serve):
    _, address = await serve(ScriptedModel("m", say("Hello there.")), capture_prompts=True)
    first = await Client.open(address)
    await first.say("Hi", client_id="m-1")
    await first.until("turn.finished")
    await first.ws.close()

    back = await connect(f"ws://{address}/ws", origin=ORIGIN)
    await back.send(json.dumps({"type": "hello", "session_id": first.ready["session_id"]}))
    client = Client(back)
    kinds = [frame["type"] for frame in await frames_until_caught_up(client)]
    assert kinds[0] == "session.ready" and kinds[-2] == "turn.finished"
    # A prompt trace is most of a turn's bytes. It is stored, and sent only when asked for.
    assert "llm.trace" not in kinds and "llm.delta" not in kinds
    await back.close()


async def test_prompt_traces_are_sent_on_request_and_only_to_their_own_conversation(serve):
    _, address = await serve(ScriptedModel("m", say("Hello there.")), capture_prompts=True)
    client = await Client.open(address)
    await client.say("Hi")
    done = await client.until("turn.finished")
    assert not any(e["type"] == "llm.trace" for e in client.events)

    await client.send(type="trace.get", turn_id=done["turn_id"])
    reply = await client.until("trace.list")
    assert (reply["turn_id"], reply["seq"]) == (done["turn_id"], 0)
    assert [t["response"]["content"] for t in reply["traces"]] == ["Hello there."]

    other = await Client.open(address)
    await other.send(type="trace.get", turn_id=done["turn_id"])
    assert (await other.until("trace.list"))["traces"] == []
    await client.ws.close()
    await other.ws.close()


async def test_a_passage_is_read_back_only_by_a_conversation_whose_search_returned_it(serve):
    model = ScriptedModel(
        "m", call("search_knowledge_base", query=REFUND_QUESTION), say("14 days [1].")
    )
    rt, address = await serve(model)
    rt.gate = ALWAYS_ANSWER
    client = await Client.open(address)
    await client.say(REFUND_QUESTION)
    await client.until("turn.finished")
    hit = next(e for e in client.events if e["type"] == "retrieval.result")["hits"][0]

    await client.send(type="passage.get", chunk_id=hit["chunk_id"])
    passage = await client.until("passage.detail")
    assert passage["source"] == hit["source"] and len(passage["content"]) > 40
    await client.send(type="passage.get", chunk_id="no-such-chunk")
    assert (await client.until("error"))["code"] == "not_found"
    # Another conversation has not been shown this passage. Its plan might not allow it.
    other = await Client.open(address)
    await other.send(type="passage.get", chunk_id=hit["chunk_id"])
    assert (await other.until("error"))["code"] == "not_found"
    await client.ws.close()
    await other.ws.close()


async def test_candidates_of_a_search_that_asked_for_verification_cannot_be_read_back(serve):
    question = "How fast do you respond when we report a P1 outage?"
    model = ScriptedModel("m", call("search_knowledge_base", query=question), say("Please verify."))
    _, address = await serve(model)
    client = await Client.open(address)
    await client.say(question)
    await client.until("turn.finished")
    search = next(e for e in client.events if e["type"] == "retrieval.result")
    assert search["band"] == "verify"
    for hit in search["hits"]:
        await client.send(type="passage.get", chunk_id=hit["chunk_id"])
        assert (await client.until("error"))["code"] == "not_found"
    await client.ws.close()


async def test_a_turn_can_be_picked_up_only_so_many_times(serve):
    model = ScriptedModel("down", [http_error(401)])
    _, address = await serve(model, max_retries_per_turn=1, max_turns_per_session=1)
    client = await Client.open(address)
    await client.say("Hello?")
    await client.until("turn.failed")
    await client.send(type="turn.retry")
    await client.until("turn.failed")
    # Without a limit, failing and retrying would run model calls for ever on one message.
    await client.send(type="turn.retry")
    assert (await client.until("error"))["code"] == "too_many_retries"
    assert len(model.calls) == 2
    await client.ws.close()


async def test_stopping_and_resuming_is_counted_like_any_other_pick_up(serve):
    model = ScriptedModel("m", ["Hel", 5.0, "lo.", usage(10, 3)])
    _, address = await serve(model, max_retries_per_turn=1)
    client = await Client.open(address)
    await client.say("Hi")
    for _ in range(2):
        await client.until("llm.delta")
        await client.send(type="turn.cancel")
        assert (await client.until("turn.interrupted"))["resumable"] is True
        await client.send(type="turn.retry")
    assert (await client.until("error"))["code"] == "too_many_retries"
    assert len(model.calls) == 2
    await client.ws.close()


async def test_connections_that_arrive_together_cannot_all_find_room(serve):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), max_sessions=3, max_connections_per_address=50
    )

    async def attempt():
        try:
            return await Client.open(address)
        except ConnectionClosed as closed:
            return closed.rcvd.code

    results = await asyncio.gather(*(attempt() for _ in range(10)))
    opened = [r for r in results if isinstance(r, Client)]
    assert len(opened) == 3
    assert {r for r in results if not isinstance(r, Client)} == {AT_CAPACITY}
    for client in opened:
        await client.ws.close()


async def test_being_turned_away_for_lack_of_room_does_not_use_up_a_visitors_allowance(serve):
    _, address = await serve(
        ScriptedModel("m", say("Hello.")), max_sessions=1, max_new_sessions_per_hour=2
    )
    first = await Client.open(address)
    for _ in range(3):
        with pytest.raises(ConnectionClosed) as refused:
            await Client.open(address)
        assert refused.value.rcvd.code == AT_CAPACITY
    await first.ws.close()
    await asyncio.sleep(0.1)  # the server lets the first conversation's hub go
    second = await Client.open(address)  # the second of two allowed, not the fifth attempt
    assert second.ready["type"] == "session.ready"
    await second.ws.close()


async def test_a_conversation_and_an_address_each_have_a_limit_of_open_connections(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")), max_connections_per_address=6)
    first = await Client.open(address)
    session_id = first.ready["session_id"]
    tabs = [first] + [await Client.open(address, session_id) for _ in range(3)]
    with pytest.raises(ConnectionClosed) as refused:
        await Client.open(address, session_id)
    assert refused.value.rcvd.code == TOO_MANY_CONNECTIONS
    tabs += [await Client.open(address) for _ in range(2)]  # six open from this address now
    with pytest.raises(ConnectionClosed) as refused:
        await Client.open(address)
    assert refused.value.rcvd.code == TOO_MANY_CONNECTIONS
    for client in tabs:
        await client.ws.close()


async def test_an_idle_connection_is_closed_and_one_with_a_running_turn_is_not(serve):
    slow = ScriptedModel("m", ["Hel", 0.25, "lo ", 0.25, "there.", usage(10, 3)])
    _, address = await serve(slow, idle_timeout_s=0.3)
    client = await Client.open(address)
    await client.say("Hi")
    await client.until("turn.finished")  # the turn outlasted the idle limit and was not cut off
    with pytest.raises(ConnectionClosed) as closed:
        await asyncio.wait_for(client.ws.recv(), 3)
    assert closed.value.rcvd.code == IDLE


async def test_no_turn_starts_once_the_daily_budget_is_spent(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    client = await Client.open(address)
    await client.say("Hi")  # one model call without a price, counted at a fraction of a cent
    await client.until("turn.finished")
    await client.ws.close()

    _, capped = await serve(ScriptedModel("m", say("Hello.")), daily_budget_usd=1e-9)
    client = await Client.open(capped)
    await client.say("Hi", client_id="m-1")
    refused = await client.until("error")
    assert (refused["code"], refused["client_id"]) == ("budget_exhausted", "m-1")
    await client.ws.close()


async def test_the_budget_adds_up_model_calls_and_hosted_searches(pool, world):
    session_id = f"budget-{world}"
    before = await Budget(pool, 1.0).spent()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (id, world_id) VALUES (%s, %s)", (session_id, world)
        )
        for seq, kind, cost in (
            (1, "llm.finished", "0.5"),
            (2, "retrieval.result", "0.25"),
            (3, "llm.finished", "null"),
            (4, "retrieval.result", "null"),
            (5, "tool.finished", "9"),
        ):
            await conn.execute(
                "INSERT INTO session_events (session_id, seq, type, payload) VALUES (%s, %s, %s, %s::jsonb)",
                (session_id, seq, kind, f'{{"cost_usd": {cost}}}'),
            )
    budget = Budget(pool, 0.7)
    # A model call without a price still counts for something. A search that named no cost ran
    # on a local model and counts for nothing.
    assert await budget.spent() - before == pytest.approx(0.5 + 0.25 + 0.002)
    assert await budget.exhausted()
    assert not await Budget(pool, None).exhausted()


async def test_the_http_side_gives_nothing_away(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    async with httpx.AsyncClient() as http:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await http.get(f"http://{address}{path}")).status_code == 404
        health = await http.get(f"http://{address}/health")
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["cache-control"] == "no-store"


async def test_behind_tls_a_page_loaded_in_the_clear_cannot_open_a_session(serve):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    over_tls = {"X-Forwarded-Proto": "https"}
    with pytest.raises(InvalidStatus):
        await Client.open(address, origin=f"http://{address}", headers=over_tls)
    client = await Client.open(address, origin=f"https://{address}", headers=over_tls)
    assert client.ready["type"] == "session.ready"
    await client.ws.close()


async def test_a_conversation_deleted_under_an_open_connection_says_so(serve, pool):
    _, address = await serve(ScriptedModel("m", say("Hello.")))
    client = await Client.open(address)
    gone = client.ready["session_id"]
    async with pool.connection() as conn:
        await worlds.delete_world(conn, gone)
    await client.say("Hi", client_id="m-1")
    refused = await client.until("error")
    assert (refused["code"], refused["client_id"]) == ("session_expired", "m-1")
    await client.say("Hi again")
    with pytest.raises(ConnectionClosed) as closed:
        await client.until("turn.started")
    assert closed.value.rcvd.code == SESSION_EXPIRED
    # Coming back with the dead ID starts a new conversation.
    fresh = await Client.open(address, gone)
    assert fresh.ready["session_id"] != gone
    await fresh.ws.close()


async def test_a_turn_that_cannot_be_stored_still_ends_for_its_client(serve, pool):
    _, address = await serve(ScriptedModel("m", ["Hel", 0.4, "lo.", usage(10, 3)]))
    client = await Client.open(address)
    await client.say("Hi")
    started = await client.until("llm.started")
    async with pool.connection() as conn:
        await worlds.delete_world(conn, client.ready["session_id"])
    # The turn runs on, and storing its end fails. The client must not be left waiting.
    failed = await client.until("turn.failed")
    assert (failed["code"], failed["retryable"]) == ("internal", False)
    assert failed["turn_id"] == started["turn_id"]
    await client.ws.close()


async def test_a_message_that_could_not_start_a_turn_is_answered_and_not_counted(
    serve, monkeypatch
):
    rt, address = await serve(ScriptedModel("m", say("Hello.")), max_turns_per_session=1)
    client = await Client.open(address)
    original = rt.graph.aget_state

    async def broken(*_args, **_kwargs):
        raise RuntimeError("the database blinked")

    monkeypatch.setattr(rt.graph, "aget_state", broken)
    await client.say("Hi", client_id="m-1")
    refused = await client.until("error")
    assert (refused["code"], refused["client_id"]) == ("failed_to_start", "m-1")
    monkeypatch.setattr(rt.graph, "aget_state", original)
    # The message that never started has not used up the one this conversation is allowed.
    await client.say("Hi", client_id="m-2")
    assert (await client.until("turn.started"))["client_id"] == "m-2"
    await client.until("turn.finished")
    await client.ws.close()


def test_the_window_stays_fast_and_small_however_many_addresses_knock():
    now = [0.0]
    window = SlidingWindow(limit=1, window_s=3600.0, clock=lambda: now[0], max_keys=1000)
    began = time.perf_counter()
    for n in range(20_000):
        window.allow(f"visitor-{n}")
    # The sweep this replaces rebuilt its whole table on every call past 10,000 addresses: about
    # 0.75 ms a call, 15 seconds for this loop.
    assert time.perf_counter() - began < 1.0
    assert len(window) == 1000
    assert not window.allow("visitor-19999")  # the newest are the ones still remembered
    now[0] = 4000.0
    for _ in range(600):
        window.allow("someone who keeps talking")
    assert len(window) == 1  # the quiet ones went, a few per call


def test_an_event_that_did_not_happen_can_be_taken_back():
    window = SlidingWindow(limit=1, window_s=60.0, clock=lambda: 0.0)
    assert window.allow("a")
    window.undo("a")
    assert [window.allow("a"), window.allow("a")] == [True, False]


def test_one_visitor_is_one_key_however_the_address_is_written():
    forms = (
        "2001:db8::1",
        "2001:0db8:0:0:0:0:0:1",
        "2001:DB8::1",
        "2001:db8::2",
        "2001:db8::ffff:9",
    )
    # One subscriber has a whole /64 of IPv6 addresses to pick from.
    assert {client_address(None, form) for form in forms} == {"2001:db8::/64"}
    assert client_address(None, "2001:db8:0:1::1") == "2001:db8:0:1::/64"
    assert client_address(None, "::ffff:203.0.113.9") == client_address("203.0.113.9", None)
    assert client_address("198.51.100.4", "  ") == "198.51.100.4"
    assert client_address(None, None) == "unknown"
    assert client_address("testclient", None) == "testclient"


def test_a_counter_caps_what_one_key_holds_at_a_time():
    counter = Counter(limit=2)
    assert [counter.take("a"), counter.take("a"), counter.take("a")] == [True, True, False]
    counter.give_back("a")
    assert counter.take("a") and counter.take("b")
