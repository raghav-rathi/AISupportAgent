from __future__ import annotations

from dataclasses import replace

import pytest

from supportagent.config import Settings, get_settings
from supportagent.kb.chunker import load_kb
from supportagent.kb.ingest import ingest
from supportagent.kb.models import FastEmbedEmbedder, FastEmbedReranker
from supportagent.kb.retrieval import Retriever

ALL = ("free", "pro", "enterprise")

# Where each plan's refund terms live.
PLAN_SECTIONS = {
    "pro": ("refund-policy", ("Pro plan refunds",)),
    "enterprise": ("refund-policy", ("Enterprise plan refunds",)),
    "free": ("refund-policy", ("Free plan",)),
}


class CountingEmbedder:
    """Wraps the real embedder and counts the documents it was asked to embed."""

    def __init__(self, inner):
        self.inner, self.name, self.documents = inner, inner.name, 0

    async def embed_documents(self, texts):
        self.documents += len(texts)
        return await self.inner.embed_documents(texts)

    async def embed_query(self, text):
        return await self.inner.embed_query(text)


@pytest.fixture(scope="module")
def kb():
    return load_kb(get_settings().kb_dir)


@pytest.fixture(scope="module")
def embedder():
    return CountingEmbedder(FastEmbedEmbedder(Settings.model_fields["embed_model"].default))


@pytest.fixture(scope="module")
async def retriever(pool, kb, embedder):
    async with pool.connection() as conn:
        await ingest(conn, kb, embedder)
    return Retriever(embedder, FastEmbedReranker(Settings.model_fields["rerank_model"].default))


async def search(pool, retriever, query, tier, limit=5):
    return await retriever.search(pool, query, tier, limit=limit)


def headings(result):
    return [(h.doc_id, h.heading_path) for h in result.hits]


async def test_a_second_ingest_writes_nothing_and_embeds_nothing(pool, kb, embedder, retriever):
    before = embedder.documents
    async with pool.connection() as conn:
        report = await ingest(conn, kb, embedder)
    assert (report.chunks, report.written, report.deleted, report.embedded) == (len(kb), 0, 0, 0)
    assert embedder.documents == before


async def test_ingest_follows_edits_and_removals(pool, kb, embedder, retriever):
    edited = replace(kb[0], content="Edited.", embed_text="Edited.", content_hash="edited")
    before = embedder.documents
    try:
        async with pool.connection() as conn:
            report = await ingest(conn, [edited, *kb[2:]], embedder)
            cur = await conn.execute("SELECT count(*) AS n FROM kb_embeddings")
            vectors = (await cur.fetchone())["n"]
        assert (report.written, report.deleted, report.embedded) == (1, 1, 1)
        assert embedder.documents == before + 1
        assert vectors == len(kb) - 1  # the removed chunk took its vector with it
    finally:
        async with pool.connection() as conn:
            restored = await ingest(conn, kb, embedder)
    assert (restored.written, restored.deleted, restored.embedded) == (2, 0, 2)


@pytest.mark.parametrize("tier", list(PLAN_SECTIONS))
async def test_the_same_question_reaches_the_section_for_the_callers_plan(pool, retriever, tier):
    # This checks hybrid search and the plan filter. How well a reranker then orders the candidates
    # is measured per model by supportagent.evals.retrieval, not asserted here.
    result = await search(pool, retriever, "How long do I have to ask for a refund?", tier)
    candidates = [(h.doc_id, h.heading_path) for h in result.fused]
    assert PLAN_SECTIONS[tier] in candidates[:5]
    other_plans = {section for plan, section in PLAN_SECTIONS.items() if plan != tier}
    assert not other_plans & set(candidates)
    assert all(tier in hit.tiers for hit in result.fused)


async def test_an_unverified_visitor_sees_only_content_written_for_every_plan(pool, retriever):
    result = await search(pool, retriever, "What is the refund window and the support limit?", None)
    assert result.fused
    assert all(hit.tiers == ALL for hit in result.fused)


async def test_exact_tokens_are_found_by_the_lexical_leg(pool, retriever):
    result = await search(pool, retriever, "X-Fernhill-Signature", "pro")
    top = result.hits[0]
    assert (top.doc_id, top.heading_path) == ("integrations", ("Webhooks",))
    assert top.lexical_rank == 1

    result = await search(pool, retriever, "we get invalid_audience", "enterprise")
    expected = ("sso-and-provisioning", ("SAML single sign-on", "SAML troubleshooting"))
    assert headings(result)[0] == expected


async def test_hits_come_back_best_first_with_scores(pool, retriever):
    result = await search(
        pool, retriever, "My card was declined, what happens next?", "pro", limit=3
    )
    scores = [hit.score for hit in result.hits]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert result.top_score == scores[0]
    assert headings(result)[0] == ("billing-and-invoices", ("When a payment fails",))


async def test_a_query_with_no_searchable_words_still_answers(pool, retriever):
    result = await search(pool, retriever, "is it the", "pro")
    assert all(hit.lexical_rank is None for hit in result.fused)
    assert len(result.hits) == 5
