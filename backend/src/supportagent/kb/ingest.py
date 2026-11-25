"""Load chunks into Postgres. Running it twice changes nothing and embeds nothing."""

from __future__ import annotations

from dataclasses import dataclass

from psycopg import AsyncConnection

from supportagent.kb.chunker import Chunk
from supportagent.kb.models import Embedder

_UPSERT_CHUNK = """
INSERT INTO kb_chunks
    (id, doc_id, doc_title, heading_path, ordinal, tiers, kind, content, embed_text, content_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    doc_title = EXCLUDED.doc_title,
    heading_path = EXCLUDED.heading_path,
    ordinal = EXCLUDED.ordinal,
    tiers = EXCLUDED.tiers,
    kind = EXCLUDED.kind,
    content = EXCLUDED.content,
    embed_text = EXCLUDED.embed_text,
    content_hash = EXCLUDED.content_hash,
    updated_at = now()
"""

_UPSERT_EMBEDDING = """
INSERT INTO kb_embeddings (chunk_id, model, content_hash, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (chunk_id, model) DO UPDATE SET
    content_hash = EXCLUDED.content_hash,
    embedding = EXCLUDED.embedding
"""


@dataclass(frozen=True)
class IngestReport:
    chunks: int  # chunks in the knowledge base after this run
    written: int  # chunk rows inserted or updated
    deleted: int  # chunk rows removed because their section is gone
    embedded: int  # vectors computed for this embedding model


async def ingest(conn: AsyncConnection, chunks: list[Chunk], embedder: Embedder) -> IngestReport:
    """Make the tables match `chunks` exactly, embedding only what is new or changed."""
    cur = await conn.execute("SELECT id, content_hash FROM kb_chunks")
    stored = {row["id"]: row["content_hash"] for row in await cur.fetchall()}
    cur = await conn.execute(
        "SELECT chunk_id, content_hash FROM kb_embeddings WHERE model = %s", (embedder.name,)
    )
    embedded = {row["chunk_id"]: row["content_hash"] for row in await cur.fetchall()}

    changed = [c for c in chunks if stored.get(c.id) != c.content_hash]
    stale = [c for c in chunks if embedded.get(c.id) != c.content_hash]
    # Embed before opening the transaction, so no lock is held while the model runs.
    vectors = await embedder.embed_documents([c.embed_text for c in stale]) if stale else []

    async with conn.transaction(), conn.cursor() as cur:
        if changed:
            await cur.executemany(
                _UPSERT_CHUNK,
                [
                    (
                        c.id,
                        c.doc_id,
                        c.doc_title,
                        list(c.heading_path),
                        c.ordinal,
                        list(c.tiers),
                        c.kind,
                        c.content,
                        c.embed_text,
                        c.content_hash,
                    )
                    for c in changed
                ],
            )
        await cur.execute(
            "DELETE FROM kb_chunks WHERE NOT (id = ANY(%s))", ([c.id for c in chunks],)
        )
        deleted = cur.rowcount
        if stale:
            await cur.executemany(
                _UPSERT_EMBEDDING,
                [
                    (c.id, embedder.name, c.content_hash, vector)
                    for c, vector in zip(stale, vectors, strict=True)
                ],
            )
    return IngestReport(len(chunks), len(changed), deleted, len(stale))
