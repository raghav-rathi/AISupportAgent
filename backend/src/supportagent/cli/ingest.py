"""Apply migrations, then load the knowledge base documents into Postgres.

Run with: uv run python -m supportagent.cli.ingest [--embedder NAME]

Each embedding model keeps its own vectors, so a second model can be ingested beside the first and
the two compared. A name that starts with "gateway:" is served by the AI Gateway.
"""

from __future__ import annotations

import argparse
import asyncio

from supportagent import db
from supportagent.config import get_settings
from supportagent.kb.chunker import load_kb
from supportagent.kb.factory import make_embedder
from supportagent.kb.ingest import ingest


async def run(embedder_name: str | None) -> None:
    settings = get_settings()
    embedder = make_embedder(embedder_name or settings.embed_model, settings)
    applied = await db.migrate(settings.database_url)
    if applied:
        print("migrations applied:", ", ".join(applied))
    chunks = load_kb(settings.kb_dir)
    pool = db.make_pool(settings.database_url, max_size=2)
    await pool.open()
    try:
        async with pool.connection() as conn:
            report = await ingest(conn, chunks, embedder)
    finally:
        await pool.close()
    print(
        f"{report.chunks} chunks from {len({c.doc_id for c in chunks})} documents: "
        f"{report.written} written, {report.deleted} deleted, "
        f"{report.embedded} embedded with {embedder.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", help="defaults to EMBED_MODEL")
    asyncio.run(run(parser.parse_args().embedder))


if __name__ == "__main__":
    main()
