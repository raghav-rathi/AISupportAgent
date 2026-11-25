"""A conversation harness for agent tests: the real graph, tools, database and retriever, with a
scripted model in place of an LLM.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from supportagent.agent.context import AgentContext, Gate
from supportagent.agent.graph import build_graph, user_turn
from supportagent.agent.judge import Verdict
from supportagent.llm.client import LLMPolicy, ResilientLLM
from supportagent.runtime import Runtime

GATE = Gate(low=-4.0, high=2.0, gap=3.0)  # fits the small reranker these tests run on
ALWAYS_CHECK = Gate(low=-100.0, high=100.0, gap=1000.0)
ALWAYS_ANSWER = Gate(low=-100.0, high=-100.0, gap=1000.0)
FAST = LLMPolicy(first_token_timeout_s=0.5, stall_timeout_s=0.5, total_timeout_s=5.0)

GROUNDED = Verdict(grounded=True)
UNGROUNDED = Verdict(grounded=False, unsupported=["refunds take 365 days"])

MAYA = {"account_number": "10482917", "pin": "4721"}  # Pro monthly, INV-30417 is 5 days old
DANIEL = {"account_number": "20573846", "pin": "9034"}  # Pro annual, INV-30388 is $1,440
ELENA = {"account_number": "53920164", "pin": "2458"}  # Enterprise
JUN = {"account_number": "75148302", "pin": "3390"}  # Free
REFUND_QUESTION = "How long do I have to get my money back on a monthly plan?"


class ScriptedJudge:
    """Returns the scripted verdicts in order and repeats the last. An exception is raised."""

    def __init__(self, *verdicts):
        self.verdicts, self.calls = list(verdicts), []

    async def check(self, *, question, answer, evidence, emit, faults=None, earlier=()):
        self.calls.append(
            {"question": question, "answer": answer, "evidence": evidence, "earlier": list(earlier)}
        )
        verdict = self.verdicts[min(len(self.calls), len(self.verdicts)) - 1]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


async def _no_sleep(_seconds):
    return None


def llm_for(model, **options) -> ResilientLLM:
    return ResilientLLM([model], policy=FAST, sleep=_no_sleep, jitter=lambda: 0.0, **options)


def runtime(pool, retriever, model, *, judge=None, checkpointer=None, **llm_options) -> Runtime:
    """A runtime around a scripted model. Without a checkpointer the conversation lives in memory."""
    return Runtime(
        pool=pool,
        graph=build_graph(checkpointer or InMemorySaver()),
        retriever=retriever,
        llm=llm_for(model, **llm_options),
        judge=judge,
        gate=GATE,
        checkpointer=checkpointer,
    )


class Session:
    """One conversation: a graph with memory, a scripted model, and the events of each turn."""

    def __init__(self, pool, retriever, world, model, *, judge=None, gate=GATE, **settings):
        self.graph = build_graph(InMemorySaver())
        self.model = model
        self.config = {"configurable": {"thread_id": uuid.uuid4().hex}}
        self.args = {
            "pool": pool,
            "retriever": retriever,
            "llm": llm_for(model),
            "judge": judge,
            "gate": gate,
            "world_id": world,
            "tool_backoff_s": 0.0,
            **settings,
        }
        self.ctx: AgentContext | None = None
        self.events: list = []  # the events of the latest turn, also when the turn raised

    async def turn(self, text: str | None):
        """Send one customer message, or resume the interrupted turn when `text` is None."""
        self.ctx = AgentContext(**self.args)
        self.events = []
        graph_input = user_turn(text) if text is not None else None
        async for part in self.graph.astream(
            graph_input, self.config, context=self.ctx, stream_mode="custom", version="v2"
        ):
            self.events.append(part["data"])
        return self.events

    async def state(self):
        return (await self.graph.aget_state(self.config)).values


def of(events, kind):
    return [e for e in events if e.type == kind]


def steps(events):
    return [(e.name, e.outcome) for e in of(events, "tool.finished")]


def system_text(model, call_index):
    """The system prompt the model received on its nth call."""
    return model.calls[call_index][0][0].content


def strings(value: Any, key: str = "") -> Iterator[str]:
    """Every free-text string in nested data. Identifiers are left out: a random ID can hold any
    four digits by chance."""
    if isinstance(value, str):
        if key != "id" and not key.endswith("_id"):
            yield value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from strings(item, name)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from strings(item, key)


def shows(secret: str, *values: Any) -> bool:
    """True when the secret appears in any free-text string of the values."""
    return any(secret in text for value in values for text in strings(value))


async def scalar(pool, query, *params):
    async with pool.connection() as conn:
        cur = await conn.execute(query, params)
        return next(iter((await cur.fetchone()).values()))


async def refunds(pool, world):
    return await scalar(pool, "SELECT count(*) FROM refunds WHERE world_id = %s", world)


async def invoice_status(pool, world, invoice):
    return await scalar(
        pool, "SELECT status FROM invoices WHERE world_id = %s AND id = %s", world, invoice
    )
