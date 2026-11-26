"""Shared fixtures. Tests run against a scratch database rebuilt from the migrations each session."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import AsyncConnectionPool

from supportagent import clock, db
from supportagent.config import Settings, get_settings
from supportagent.crm import world as worlds
from supportagent.kb.chunker import load_kb
from supportagent.kb.ingest import ingest
from supportagent.kb.models import FastEmbedEmbedder, FastEmbedReranker
from supportagent.kb.retrieval import Retriever

TODAY = date(2026, 11, 24)


@pytest.fixture(scope="session", autouse=True)
def frozen_clock() -> None:
    clock.freeze(datetime(TODAY.year, TODAY.month, TODAY.day, 12, tzinfo=UTC))


@pytest.fixture(scope="session")
def today() -> date:
    return TODAY


@pytest.fixture(scope="session")
async def dsn() -> str:
    base = get_settings().database_url
    parts = conninfo_to_dict(base)
    test_db = f"{parts['dbname']}_test"
    async with await psycopg.AsyncConnection.connect(base, autocommit=True) as admin:
        await admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(test_db))
        )
        await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db)))
    test_dsn = make_conninfo(**{**parts, "dbname": test_db})
    await db.migrate(test_dsn)
    return test_dsn


@pytest.fixture(scope="session")
async def pool(dsn: str) -> AsyncIterator[AsyncConnectionPool]:
    pool = db.make_pool(dsn)
    await pool.open()
    yield pool
    await pool.close()


@pytest.fixture
async def world(pool: AsyncConnectionPool) -> AsyncIterator[str]:
    world_id = f"test-{uuid.uuid4().hex[:12]}"
    async with pool.connection() as conn:
        await worlds.create_world(conn, world_id, TODAY)
    yield world_id
    async with pool.connection() as conn:
        await worlds.delete_world(conn, world_id)


@pytest.fixture(scope="session")
async def retriever(pool: AsyncConnectionPool) -> Retriever:
    """The real knowledge base, ingested once, behind the small local models.

    The models are the defaults of the settings, not the values of a .env file, which may name
    hosted models that tests must not call.
    """
    settings = get_settings()
    embedder = FastEmbedEmbedder(Settings.model_fields["embed_model"].default)
    async with pool.connection() as conn:
        await ingest(conn, load_kb(settings.kb_dir), embedder)
    return Retriever(embedder, FastEmbedReranker(Settings.model_fields["rerank_model"].default))
