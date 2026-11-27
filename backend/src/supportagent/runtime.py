"""Builds the long-lived parts of the app once: pool, models, retriever, graph, checkpointer."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from supportagent import db
from supportagent.agent.context import AgentContext, Gate, ToolFaults, TurnUsage
from supportagent.agent.early import EarlyCall
from supportagent.agent.graph import build_graph
from supportagent.agent.judge import Judge, LLMJudge
from supportagent.config import ModelSettings, Settings
from supportagent.kb.factory import make_embedder, make_reranker
from supportagent.kb.hosted import MissingKey
from supportagent.kb.retrieval import Retriever
from supportagent.llm.client import LLMPolicy, ResilientLLM
from supportagent.llm.faults import FaultPlan
from supportagent.llm.openai_model import OpenAIChatModel
from supportagent.llm.pricing import Price
from supportagent.redact import Redactor, session_key

if TYPE_CHECKING:
    from supportagent.eventlog import EventLog

logger = logging.getLogger(__name__)


class NotConfigured(Exception):
    """A setting the app cannot run without is missing."""


def build_model(settings: Settings, spec: ModelSettings) -> OpenAIChatModel:
    key = spec.api_key or settings.openrouter_api_key
    if key is None:
        raise NotConfigured(f"no API key for {spec.model}: set its API_KEY or OPENROUTER_API_KEY")
    priced = spec.input_price is not None and spec.output_price is not None
    return OpenAIChatModel(
        spec.model,
        model=spec.model,
        base_url=spec.base_url or settings.openrouter_base_url,
        api_key=key.get_secret_value(),
        strict_tools=spec.strict_tools,
        json_schema=spec.json_schema,
        extra_body=spec.extra_body,
        price=Price(spec.input_price, spec.output_price) if priced else None,
        # This limit only catches a dead connection, so it sits above our own limits.
        silence_timeout_s=max(
            60.0, settings.llm_first_token_timeout_s + 10.0, settings.llm_stall_timeout_s + 10.0
        ),
    )


def llm_policy(settings: Settings) -> LLMPolicy:
    return LLMPolicy(
        first_token_timeout_s=settings.llm_first_token_timeout_s,
        stall_timeout_s=settings.llm_stall_timeout_s,
        total_timeout_s=settings.llm_total_timeout_s,
        deadline_s=settings.llm_deadline_s,
        hedge_after_s=settings.voice_hedge_after_s,
    )


def _retriever(settings: Settings) -> Retriever:
    try:
        return Retriever(
            make_embedder(settings.embed_model, settings),
            make_reranker(settings.rerank_model, settings),
        )
    except MissingKey as exc:
        raise NotConfigured(str(exc)) from exc


async def _require_current_vectors(pool: AsyncConnectionPool, embed_model: str) -> None:
    """Refuse to start when the knowledge base has no current vectors for the embedding model.

    Search would still return something, from the full-text leg alone or from vectors of text
    that has since changed, and nothing would show that half of it was missing.
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) AS chunks, count(e.chunk_id) AS current FROM kb_chunks c "
            "LEFT JOIN kb_embeddings e ON e.chunk_id = c.id AND e.model = %s "
            "AND e.content_hash = c.content_hash",
            (embed_model,),
        )
        row = await cur.fetchone()
    if row["chunks"] == 0 or row["current"] < row["chunks"]:
        raise NotConfigured(
            f"{row['chunks'] - row['current']} of {row['chunks']} knowledge base chunks have no current "
            f"vector for {embed_model}: run `python -m supportagent.cli.ingest`"
        )


@dataclass
class Runtime:
    pool: AsyncConnectionPool
    graph: CompiledStateGraph
    retriever: Retriever
    llm: ResilientLLM
    judge: Judge | None
    gate: Gate
    checkpointer: BaseCheckpointSaver | None = None
    secret_key: str | None = None  # keys the stored hashes of PINs, see redact.session_key

    def context(
        self,
        *,
        world_id: str,
        mode: Literal["text", "voice"] = "text",
        faults: FaultPlan | None = None,
        tool_faults: ToolFaults | None = None,
        redactor: Redactor | None = None,
        usage: TurnUsage | None = None,
        early: EarlyCall | None = None,
    ) -> AgentContext:
        """The context for one turn of one session."""
        return AgentContext(
            pool=self.pool,
            retriever=self.retriever,
            llm=self.llm,
            judge=self.judge,
            gate=self.gate,
            world_id=world_id,
            mode=mode,
            faults=faults or FaultPlan(),
            tool_faults=tool_faults or ToolFaults(),
            redactor=redactor or Redactor(),
            usage=usage or TurnUsage(),
            early=early,
        )

    def redactor(self, log: EventLog) -> Redactor:
        """The redactor for one session, knowing the PINs earlier processes hashed for it."""
        key = session_key(self.secret_key, log.session_id)
        return Redactor(key=key, known=log.secret_hashes)

    async def prune(self, session_id: str) -> None:
        """Keep only the newest saved state of a session's conversation.

        The checkpointer stores a full copy of the conversation at every step. Nothing reads the
        older copies, and they would keep text that a later step masked or took back.
        """
        if self.checkpointer is None:
            return
        try:
            await self.checkpointer.aprune([session_id], strategy="keep_latest")
        except NotImplementedError:
            pass  # a checkpointer that keeps nothing on disk
        except psycopg.Error:
            # Two prunes of one session can collide. The next turn prunes again.
            logger.warning("could not prune session %s", session_id, exc_info=True)


@asynccontextmanager
async def open_runtime(settings: Settings) -> AsyncIterator[Runtime]:
    if settings.llm is None:
        raise NotConfigured("no agent model: set LLM__MODEL")
    if settings.judge is None and not settings.allow_unchecked_answers:
        raise NotConfigured(
            "no grounding judge: set JUDGE__MODEL, or ALLOW_UNCHECKED_ANSWERS=true to run without"
        )
    shared = {"capture_prompts": settings.capture_prompts, "policy": llm_policy(settings)}
    models = [build_model(settings, settings.llm)]
    if settings.fallback:
        models.append(build_model(settings, settings.fallback))
    judge = None
    if settings.judge:
        judge = LLMJudge(ResilientLLM([build_model(settings, settings.judge)], **shared))
    retriever = _retriever(settings)

    await db.migrate(settings.database_url)
    pool = db.make_pool(settings.database_url)
    await pool.open()
    try:
        await _require_current_vectors(pool, retriever.embedder.name)
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield Runtime(
            pool=pool,
            graph=build_graph(checkpointer),
            retriever=retriever,
            llm=ResilientLLM(models, **shared),
            judge=judge,
            gate=Gate(low=settings.gate_low, high=settings.gate_high, gap=settings.gate_gap),
            checkpointer=checkpointer,
            secret_key=settings.secret_key.get_secret_value() if settings.secret_key else None,
        )
    finally:
        await pool.close()
        await retriever.aclose()
