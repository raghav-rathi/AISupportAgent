"""Postgres access: one async pool, plain SQL, forward-only migrations."""

from __future__ import annotations

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_MIGRATION_LOCK = 724_001  # advisory lock id, so two processes never migrate at once


def migration_files() -> list[tuple[str, str]]:
    """Name and SQL text of every migration, in order."""
    folder = resources.files("supportagent") / "sql"
    files = sorted((f for f in folder.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


async def migrate(dsn: str) -> list[str]:
    """Apply the migrations this database has not seen. Returns the names applied."""
    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK,))
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            cur = await conn.execute("SELECT name FROM schema_migrations")
            done = {row[0] for row in await cur.fetchall()}
            for name, text in migration_files():
                if name in done:
                    continue
                async with conn.transaction():
                    await conn.execute(text)
                    await conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
                applied.append(name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))
    return applied


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


def make_pool(dsn: str, *, max_size: int = 10) -> AsyncConnectionPool:
    """Build a closed pool. Call `await pool.open()` after `migrate`, which creates the vector type.

    Connections autocommit and return rows as dicts. The LangGraph checkpointer needs the same setup,
    so it can share this pool. Each connection is checked when it is handed out, because a hosted
    database closes idle ones and the pool would otherwise pass on a dead socket.
    """
    return AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=max_size,
        open=False,
        configure=_configure,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    )
