"""The WebSocket server, driven by a real client over a real socket, with a scripted model."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import uvicorn
from agent_harness import FAST, llm_for, runtime
from fakes import ScriptedModel, http_error, say, usage
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from supportagent.api.app import create_app, origin_allowed
from supportagent.config import get_settings
from supportagent.llm.client import ResilientLLM

ORIGIN = "http://ui.test"


@pytest.fixture
async def serve(pool, retriever):
    """Start the app on a free port with the given model. Returns the runtime and the address."""
    running = []

    async def start(model, **overrides):
        rt = runtime(pool, retriever, model)
        settings = get_settings().model_copy(
            update={
                "allowed_origins": [ORIGIN],
                "max_message_chars": 200,
                "fault_injection": True,
                **overrides,
            }
        )
        config = uvicorn.Config(
            create_app(rt, settings), host="127.0.0.1", port=0, log_level="error"
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
    async def open(cls, address, session_id=None, last_seq=0, origin=ORIGIN):
        client = cls(await connect(f"ws://{address}/ws", origin=origin))
        hello = {"type": "hello", "session_id": session_id, "last_seq": last_seq}
        await client.ws.send(json.dumps(hello))
        client.ready = await client.next()
        return client

    async def send(self, **message):
        await self.ws.send(json.dumps(message))

    async def say(self, text):
        await self.send(type="user.message", text=text)

    async def next(self):
        event = json.loads(await asyncio.wait_for(self.ws.recv(), 5))
        self.events.append(event)
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
    assert not origin_allowed("https://evil.example", host, settings)
    assert not origin_allowed("https://support.example.evil.test", host, settings)
    assert not origin_allowed("", host, settings)
    assert not origin_allowed(None, host, settings)
    nobody = settings.model_copy(update={"allowed_origins": []})
    assert not origin_allowed("https://support.example", host, nobody)
