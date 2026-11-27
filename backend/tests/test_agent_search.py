"""What the search tool sends out, which wording it keeps, and how it names other plans."""

from __future__ import annotations

import json

from agent_harness import GATE, JUN, MAYA, Session, of
from fakes import ScriptedModel, call, say

from supportagent.kb.retrieval import Hit, SearchResult


class Recording:
    """A retriever that notes every query before it passes it on."""

    def __init__(self, inner) -> None:
        self.inner, self.queries = inner, []

    async def search(self, pool, query, tier, **options):
        self.queries.append(query)
        return await self.inner.search(pool, query, tier, **options)


class Canned:
    """A retriever that returns prepared results, one per search."""

    def __init__(self, *results: SearchResult) -> None:
        self.results, self.queries = list(results), []

    async def search(self, pool, query, tier, **options):
        self.queries.append(query)
        return self.results[min(len(self.queries), len(self.results)) - 1]


def found(score: float | None, *others: tuple[float, tuple[str, ...]], cost=None) -> SearchResult:
    """A search result: the best visible score, and the scores and plans of other-plan chunks."""
    hits = []
    if score is not None:
        hits = [
            Hit("c1", "d1", "Doc", ("Section",), ("pro",), "prose", "The text.", 1, 1, 0.1, score)
        ]
    ranked = tuple(sorted(others, reverse=True))
    return SearchResult(
        hits=hits,
        fused=hits,
        plan_score=ranked[0][0] if ranked else None,
        plan_tiers=ranked[0][1] if ranked else (),
        embed_ms=1.0,
        search_ms=1.0,
        rerank_ms=1.0,
        plan_hits=ranked,
        cost_usd=cost,
    )


async def test_no_pin_leaves_in_a_search_query(pool, retriever, world):
    # A hosted reranker is a third party. The customer's own words go out as the second
    # wording, and they may still hold the PIN typed next to the question.
    recording = Recording(retriever)
    model = ScriptedModel(
        "m", call("search_knowledge_base", query="nonexistent xyzzy topic, PIN 4721")
    )
    session = Session(pool, recording, world, model)
    events = await session.turn("Account 10482917, my PIN is 4721. Tell me about xyzzy.")
    assert len(recording.queries) == 2
    assert not any("4721" in query or "*" in query for query in recording.queries), (
        recording.queries
    )
    # The second wording is the question alone: the sentence that gave the credentials only
    # blurs a search.
    assert recording.queries[1] == "Tell me about xyzzy."
    assert "4721" not in of(events, "retrieval.result")[0].query


async def test_the_customers_own_words_get_a_try_when_the_models_wording_only_finds_another_plan(
    pool, retriever, world
):
    canned = Canned(found(None, (GATE.high + 1, ("enterprise",))), found(GATE.high + 3))
    model = ScriptedModel(
        "m", call("verify_identity", **MAYA), call("search_knowledge_base", query="sla"), say("[1]")
    )
    session = Session(pool, canned, world, model)
    events = await session.turn("Account 10482917, PIN 4721. How fast do you answer an outage?")
    result = of(events, "retrieval.result")[0]
    assert len(canned.queries) == 2  # "restricted" on the first wording did not end the search
    assert (result.band, result.query) == ("answer", canned.queries[1])


async def test_every_plan_with_a_confident_match_is_named_not_only_the_best(pool, retriever, world):
    others = ((GATE.high + 3, ("pro",)), (GATE.high + 1, ("enterprise",)), (GATE.low, ("free",)))
    model = ScriptedModel(
        "m",
        call("verify_identity", **JUN),
        call("search_knowledge_base", query="audit log"),
        say("ok"),
    )
    session = Session(pool, Canned(found(None, *others)), world, model)
    events = await session.turn("Account 75148302, PIN 3390. How long is the audit log kept?")
    assert of(events, "retrieval.result")[0].band == "restricted"
    told = json.loads(model.calls[2][0][-1].content)
    assert told["covered_for_plans"] == ["pro", "enterprise"]
    assert "the Pro and Enterprise plans only" in told["message"]


async def test_what_a_hosted_search_cost_is_reported_and_counted(pool, retriever, world):
    canned = Canned(found(None, cost=0.0004), found(GATE.high + 3, cost=0.0005))
    model = ScriptedModel("m", call("search_knowledge_base", query="first wording"), say("[1]"))
    session = Session(pool, canned, world, model)
    events = await session.turn("the customer's own wording")
    assert abs(of(events, "retrieval.result")[0].cost_usd - 0.0009) < 1e-9
    assert abs(session.ctx.usage.priced_usd - 0.0009) < 1e-9
