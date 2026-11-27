"""The OpenRouter embedder and reranker against a fake transport, and the tool guard's handling of
their failures. Nothing here reaches the network.
"""

from __future__ import annotations

import json
import math

import httpx
import numpy as np
import pytest
from agent_harness import ALWAYS_ANSWER, REFUND_QUESTION, Session, of, steps
from fakes import ScriptedModel, call, say
from pydantic import SecretStr

from supportagent.config import get_settings
from supportagent.kb.factory import make_embedder, make_reranker
from supportagent.kb.hosted import (
    HostedEmbedder,
    HostedModelError,
    HostedModelRejected,
    HostedReranker,
    MissingKey,
)


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ranked(*pairs: tuple[int, float]):
    """A rerank endpoint that answers with these (index, score) pairs, in this order."""

    def handler(_request: httpx.Request) -> httpx.Response:
        results = [{"index": index, "relevance_score": score} for index, score in pairs]
        return httpx.Response(200, json={"id": "gen_1", "results": results})

    return handler


def reranker(handler) -> HostedReranker:
    return HostedReranker("cohere/rerank-5-pro", api_key="k", client=client(handler))


async def test_the_reranker_sends_the_documented_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(request.url),
            auth=request.headers["authorization"],
            body=json.loads(request.content),
        )
        return ranked((0, 0.5), (1, 0.5))(request)

    await reranker(handler).score("refund window", ["first", "second"])
    assert seen == {
        "url": "https://openrouter.ai/api/v1/rerank",
        "auth": "Bearer k",
        "body": {
            "model": "cohere/rerank-5-pro",
            "query": "refund window",
            "documents": ["first", "second"],
            "top_n": 2,
        },
    }


async def test_scores_come_back_in_document_order_not_in_ranking_order():
    # The service lists the best document first. Document 1 is the best one here.
    scores = await reranker(ranked((1, 0.9), (0, 0.5), (2, 0.1))).score("q", ["a", "b", "c"])
    assert scores == pytest.approx([0.0, math.log(9), -math.log(9)])


async def test_scores_of_exactly_zero_and_one_stay_finite():
    scores = await reranker(ranked((0, 1.0), (1, 0.0))).score("q", ["a", "b"])
    assert all(math.isfinite(s) for s in scores) and scores[0] > 10 > -10 > scores[1]


async def test_a_reply_that_skips_a_document_is_an_error_not_a_zero():
    with pytest.raises(HostedModelError, match="1 of 2"):
        await reranker(ranked((0, 0.9))).score("q", ["a", "b"])


@pytest.mark.parametrize(
    "pairs",
    [
        ((0, 0.9), (0, 0.8), (1, 0.5)),  # index 0 twice, every document still named
        ((0, 0.9), (2, 0.5)),  # index 2 for two documents
    ],
)
async def test_a_reply_that_repeats_an_index_or_names_one_out_of_range_is_an_error(pairs):
    with pytest.raises(HostedModelError, match="of 2 documents"):
        await reranker(ranked(*pairs)).score("q", ["a", "b"])


async def test_a_cost_of_true_is_no_price():
    def handler(_request: httpx.Request) -> httpx.Response:
        results = [{"index": 0, "relevance_score": 0.5}]
        return httpx.Response(200, json={"results": results, "usage": {"cost": True}})

    _, cost = await reranker(handler).score_with_cost("q", ["a"])
    assert cost is None


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_busy_or_broken_service_raises_the_retryable_error(status):
    with pytest.raises(HostedModelError, match=str(status)):
        await reranker(lambda _request: httpx.Response(status)).score("q", ["a"])


async def test_a_timeout_raises_the_retryable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(HostedModelError, match="ReadTimeout"):
        await reranker(handler).score("q", ["a"])


async def test_a_rejected_key_is_not_retried_as_if_it_would_pass():
    for status in (400, 401, 402, 404):
        with pytest.raises(HostedModelRejected, match=str(status)):
            await reranker(lambda _request, s=status: httpx.Response(s)).score("q", ["a"])


async def test_the_embedder_batches_keeps_order_and_instructs_queries_only():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        data = [
            {"index": i, "embedding": [float(len(text)), 0.0]}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": data[::-1]})  # out of order on purpose

    embedder = HostedEmbedder(
        "alibaba/qwen3-embedding-8b",
        api_key="k",
        query_instruction="Find the passage that answers the question",
        batch_size=2,
        client=client(handler),
    )
    vectors = await embedder.embed_documents(["a", "bb", "ccc"])
    assert [float(v[0]) for v in vectors] == [1.0, 2.0, 3.0]
    assert vectors[0].dtype == np.float32
    assert [body["input"] for body in bodies] == [["a", "bb"], ["ccc"]]

    await embedder.embed_query("hi")
    assert bodies[-1] == {
        "model": "alibaba/qwen3-embedding-8b",
        "input": ["Instruct: Find the passage that answers the question\nQuery:hi"],
    }


def test_an_openrouter_name_needs_the_key_and_loses_its_prefix():
    settings = get_settings()
    without = settings.model_copy(update={"openrouter_api_key": None})
    with pytest.raises(MissingKey, match="OPENROUTER_API_KEY"):
        make_reranker("openrouter:cohere/rerank-5-pro", without)
    with pytest.raises(MissingKey):
        make_embedder("openrouter:qwen/qwen3-embedding-8b", without)

    keyed = settings.model_copy(update={"openrouter_api_key": SecretStr("k")})
    assert make_reranker("openrouter:cohere/rerank-5-pro", keyed).name == "cohere/rerank-5-pro"
    embedder = make_embedder("openrouter:qwen/qwen3-embedding-8b", keyed)
    assert embedder.name == "qwen/qwen3-embedding-8b"


class Flaky:
    """A retriever whose hosted reranker is down for the first few searches."""

    def __init__(self, inner, failures: int) -> None:
        self.inner, self.failures = inner, failures

    async def search(self, *args, **kwargs):
        if self.failures:
            self.failures -= 1
            raise HostedModelError("HTTP 503")
        return await self.inner.search(*args, **kwargs)


async def test_the_tool_guard_retries_a_hosted_model_outage(pool, retriever, world):
    model = ScriptedModel(
        "m", call("search_knowledge_base", query=REFUND_QUESTION), say("14 days [1].")
    )
    session = Session(pool, Flaky(retriever, failures=2), world, model, gate=ALWAYS_ANSWER)
    events = await session.turn(REFUND_QUESTION)
    assert [e.message for e in of(events, "tool.error")] == ["HTTP 503", "HTTP 503"]
    assert [e.attempt for e in of(events, "tool.retry")] == [2, 3]
    assert steps(events) == [("search_knowledge_base", "ok")]
