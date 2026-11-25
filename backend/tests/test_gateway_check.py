"""The first-contact check has to tell a working model from a misconfigured one. These tests feed
it scripted models, so they say nothing about any real endpoint.
"""

from __future__ import annotations

import numpy as np
from fakes import ScriptedModel, call, http_error, say

from supportagent.cli.gateway_check import (
    check_chat,
    check_embedder,
    check_judge,
    check_reranker,
)
from supportagent.kb.hosted import HostedModelRejected
from supportagent.llm.pricing import Price

VERIFY = call("verify_identity", account_number="10482917", pin="0000")


async def test_a_model_that_calls_the_tool_passes_and_reports_what_it_used():
    model = ScriptedModel("m", VERIFY, price=Price(input=1.0, output=2.0))
    row = await check_chat("candidate", model)
    assert (row.ok, row.kind, row.name) == (True, "chat", "candidate")
    assert "500 in and 15 out" in row.detail and "$0.00053" in row.detail
    # The call is the agent's own: its system prompt and all six tools.
    messages, tools = model.calls[0]
    assert "support assistant for Fernhill" in messages[0].content
    assert len(tools) == 6


async def test_a_model_that_answers_in_text_or_leaks_its_reasoning_fails():
    chatty = await check_chat("candidate", ScriptedModel("m", say("You are on the Pro plan.")))
    assert not chatty.ok and "expected one verify_identity call" in chatty.detail
    leaky = await check_chat("candidate", ScriptedModel("m", say("<think>hmm</think> Sure.")))
    assert not leaky.ok and "reasoning leaked" in leaky.detail


async def test_a_rejected_request_shows_the_providers_own_words():
    refused = http_error(400, message="provider 'cerebrus' is not available for this model")
    row = await check_chat("candidate", ScriptedModel("m", [refused]))
    assert not row.ok
    assert "BadRequestError" in row.detail and "cerebrus" in row.detail


async def test_the_judge_check_sends_the_schema_and_reads_the_verdict():
    model = ScriptedModel("m", say('{"grounded": true, "unsupported": []}'))
    row = await check_judge("judge", model)
    assert row.ok and "verdict grounded" in row.detail
    assert model.schemas[0]["name"] == "verdict"
    assert model.calls[0][1] == []  # the judge gets no tools

    wrong = await check_judge(
        "judge", ScriptedModel("m", say('{"grounded": false, "unsupported": ["x"]}'))
    )
    assert not wrong.ok
    silent = await check_judge("judge", ScriptedModel("m", say("Looks fine.")))
    assert not silent.ok and "JudgeError" in silent.detail


class FakeEmbedder:
    name = "fake"

    def __init__(self, related: float) -> None:
        self.related = related

    async def embed_query(self, _text):
        return np.array([1.0, 0.0], dtype=np.float32)

    async def embed_documents(self, texts):
        near = np.array([self.related, 1.0 - self.related], dtype=np.float32)
        return [near, np.array([0.0, 1.0], dtype=np.float32)][: len(texts)]


class FakeReranker:
    name = "fake"

    def __init__(self, scores) -> None:
        self.scores = scores

    async def score(self, _query, _documents):
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


async def test_retrieval_models_must_put_the_related_passage_first():
    good = await check_embedder("e", FakeEmbedder(related=0.9))
    assert good.ok and good.detail.startswith("2 dimensions")
    assert not (await check_embedder("e", FakeEmbedder(related=0.0))).ok

    assert (await check_reranker("r", FakeReranker([4.2, -6.0]))).ok
    assert not (await check_reranker("r", FakeReranker([-6.0, 4.2]))).ok
    refused = await check_reranker(
        "r", FakeReranker(HostedModelRejected("HTTP 404: no such model"))
    )
    assert not refused.ok and "no such model" in refused.detail
