"""Hybrid retrieval: dense search and full-text search fused with RRF in SQL, then a reranker.

The caller passes the customer's plan from session state. It is a filter in the query, so a chunk
written for another plan never reaches the model.

One search also learns whether a chunk written for another plan would answer better. The reranker
scores a few of those chunks in the same batch, and the result carries only their best score and
the plans it belongs to. The scope gate uses that to tell "the help center has nothing" from "the
help center has this for another plan". The chunks themselves never leave this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from psycopg_pool import AsyncConnectionPool

from supportagent.crm.policy import TIERS
from supportagent.kb.models import Embedder, Reranker

# Reciprocal rank fusion: score = sum over legs of 1 / (k + rank). 60 is the constant from the RRF paper.
_RRF_K = 60

# Chunks are ranked twice, once among those the plan may see and once among the rest. A long
# question rarely contains every word of the right chunk, so the query's terms are OR-ed and
# ts_rank_cd orders the matches. The OR query is built from the parsed lexemes, each one quoted, so
# punctuation inside a term cannot change its structure.
_SEARCH = """
WITH scoped AS (
    SELECT id, tiers @> %(need)s::text[] AS visible FROM kb_chunks
),
terms AS (
    SELECT (
        SELECT string_agg(quote_literal(lexeme), ' | ')
        FROM unnest(to_tsvector('english', %(qtext)s))
    )::tsquery AS tsq
),
dense AS (
    SELECT e.chunk_id AS id, s.visible,
           row_number() OVER (
               PARTITION BY s.visible ORDER BY e.embedding <=> %(qvec)s, e.chunk_id
           ) AS rnk
    FROM kb_embeddings e
    JOIN scoped s ON s.id = e.chunk_id
    WHERE e.model = %(model)s
),
lexical AS (
    SELECT c.id, s.visible,
           row_number() OVER (
               PARTITION BY s.visible ORDER BY ts_rank_cd(c.fts, t.tsq) DESC, c.id
           ) AS rnk
    FROM kb_chunks c
    JOIN scoped s ON s.id = c.id
    CROSS JOIN terms t
    WHERE c.fts @@ t.tsq
),
fused AS (
    SELECT id, visible, d.rnk AS dense_rank, l.rnk AS lexical_rank,
           coalesce(1.0 / (%(rrf_k)s + d.rnk), 0) + coalesce(1.0 / (%(rrf_k)s + l.rnk), 0) AS rrf
    FROM (SELECT * FROM dense WHERE rnk <= %(k)s) d
    FULL OUTER JOIN (SELECT * FROM lexical WHERE rnk <= %(k)s) l USING (id, visible)
),
ranked AS (
    SELECT *, row_number() OVER (PARTITION BY visible ORDER BY rrf DESC, id) AS pos FROM fused
)
SELECT c.id, c.doc_id, c.doc_title, c.heading_path, c.tiers, c.kind, c.content, c.embed_text,
       r.visible, r.dense_rank, r.lexical_rank, r.rrf
FROM ranked r
JOIN kb_chunks c USING (id)
WHERE r.pos <= CASE WHEN r.visible THEN %(k)s ELSE %(probe)s END
ORDER BY r.visible DESC, r.rrf DESC, c.id
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: tuple[str, ...]
    tiers: tuple[str, ...]
    kind: str
    content: str
    dense_rank: int | None
    lexical_rank: int | None
    rrf: float
    score: float | None = None  # the reranker's raw score


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]  # what the plan may see, best first, after reranking
    fused: list[Hit]  # the same candidates in RRF order, before reranking
    plan_score: float | None  # the reranker's best score among chunks written for other plans
    plan_tiers: tuple[str, ...]  # the plans that best other chunk is written for
    embed_ms: float
    search_ms: float
    rerank_ms: float
    # Score and plans of every chunk written for other plans that was scored. No text: the
    # chunks themselves never leave this module.
    plan_hits: tuple[tuple[float, tuple[str, ...]], ...] = ()
    cost_usd: float | None = None  # what a hosted reranker charged, when it says

    @property
    def top_score(self) -> float | None:
        """The reranker's score for the best chunk: the signal the scope gate reads."""
        return self.hits[0].score if self.hits else None


_NOTHING = SearchResult([], [], None, (), 0.0, 0.0, 0.0)


class Retriever:
    def __init__(
        self, embedder: Embedder, reranker: Reranker, *, candidates: int = 20, probe: int = 8
    ) -> None:
        """`candidates` chunks the plan may see and `probe` chunks it may not go to the reranker."""
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.probe = probe

    async def warm(self) -> None:
        """Load the models that run in this process, so the first search does not wait for them."""
        for model in (self.embedder, self.reranker):
            warm = getattr(model, "warm", None)
            if warm is not None:
                await warm()

    async def aclose(self) -> None:
        """Close the HTTP clients of hosted models."""
        for model in (self.embedder, self.reranker):
            close = getattr(model, "aclose", None)
            if close is not None:
                await close()

    async def search(
        self, pool: AsyncConnectionPool, query: str, tier: str | None, *, limit: int = 5
    ) -> SearchResult:
        """Find the chunks most relevant to `query` among those the plan may see.

        `tier` None means an unverified visitor, who sees only chunks written for every plan.
        A database connection is held for the one SQL query and not while the models run.
        """
        query = " ".join(query.replace("\x00", " ").split())
        if not query:
            return _NOTHING  # a reranker scores an empty query at random
        started = time.perf_counter()
        qvec = await self.embedder.embed_query(query)
        embedded = time.perf_counter()

        async with pool.connection() as conn:
            cur = await conn.execute(
                _SEARCH,
                {
                    "need": [tier] if tier else list(TIERS),
                    "qvec": qvec,
                    "qtext": query,
                    "model": self.embedder.name,
                    "k": self.candidates,
                    "probe": self.probe,
                    "rrf_k": _RRF_K,
                },
            )
            rows = await cur.fetchall()
        searched = time.perf_counter()

        documents = [row["embed_text"] for row in rows]
        priced = getattr(self.reranker, "score_with_cost", None)
        if priced is not None:
            scores, cost_usd = await priced(query, documents)
        else:
            scores, cost_usd = await self.reranker.score(query, documents), None
        reranked = time.perf_counter()

        fused: list[Hit] = []
        scored: list[Hit] = []
        plan_hits: list[tuple[float, tuple[str, ...]]] = []
        for row, score in zip(rows, scores, strict=True):
            usable = score is not None and math.isfinite(score)
            if not row["visible"]:
                if usable:
                    plan_hits.append((score, tuple(row["tiers"])))
                continue
            hit = Hit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                doc_title=row["doc_title"],
                heading_path=tuple(row["heading_path"]),
                tiers=tuple(row["tiers"]),
                kind=row["kind"],
                content=row["content"],
                dense_rank=row["dense_rank"],
                lexical_rank=row["lexical_rank"],
                rrf=float(row["rrf"]),
            )
            fused.append(hit)
            if usable:  # a score that is not a number ranks nothing
                scored.append(replace(hit, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        plan_hits.sort(key=lambda hit: hit[0], reverse=True)
        return SearchResult(
            hits=scored[:limit],
            fused=fused,
            plan_score=plan_hits[0][0] if plan_hits else None,
            plan_tiers=plan_hits[0][1] if plan_hits else (),
            embed_ms=(embedded - started) * 1000,
            search_ms=(searched - embedded) * 1000,
            rerank_ms=(reranked - searched) * 1000,
            plan_hits=tuple(plan_hits),
            cost_usd=cost_usd,
        )
