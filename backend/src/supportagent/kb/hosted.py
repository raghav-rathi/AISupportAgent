"""Embedding and reranking through OpenRouter, behind the same interfaces as the local models.

The request and response shapes follow OpenRouter's API reference: `/embeddings` in OpenAI's
format, and `/rerank` in Cohere's, both under the same base URL as chat completions.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
import numpy as np

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class MissingKey(Exception):
    """A hosted model was asked for and no API key is configured."""


class HostedModelError(Exception):
    """A hosted model failed in a way that may pass: a timeout, a dropped connection, a 429, a
    5xx, or a reply that is not what the documentation shows."""


class HostedModelRejected(Exception):
    """A hosted model refused the request and will refuse it again: a bad key, a spend cap, a
    model OpenRouter does not have, a malformed request."""


async def _post(client: httpx.AsyncClient, url: str, key: str, body: dict[str, Any]) -> Any:
    try:
        response = await client.post(url, headers={"Authorization": f"Bearer {key}"}, json=body)
    except httpx.TransportError as exc:  # timeouts and connection failures
        raise HostedModelError(type(exc).__name__) from exc
    status = response.status_code
    if status in (408, 429) or status >= 500:
        raise HostedModelError(f"HTTP {status}")
    if status >= 400:
        # The provider's own words go into the error. They reach our log and the setup check,
        # never the customer: the tool guard reports a rejected call by its type alone.
        said = " ".join(response.text.split())[:200]
        raise HostedModelRejected(f"HTTP {status}: {said}" if said else f"HTTP {status}")
    try:
        return response.json()
    except ValueError as exc:  # a proxy's error page sent with status 200
        raise HostedModelError("the reply was not JSON") from exc


def _logit(probability: float) -> float:
    """Map a score from (0, 1) onto the whole real line.

    Cohere-style scores crowd together near 0 and 1. On the logit scale 0.99 and 0.999 are as far
    apart as 0.5 and 0.91, which gives a threshold room to sit between them.
    """
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


class HostedEmbedder:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = OPENROUTER_URL,
        query_instruction: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 16,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """`query_instruction` is put in front of queries only, for models trained that way.

        `dimensions` asks for shorter vectors from a model that can give them.
        """
        self.name = model
        self._key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        # The layout of the Qwen3-Embedding model card: no space after "Query:".
        self._query_prefix = f"Instruct: {query_instruction}\nQuery:" if query_instruction else ""
        self._dimensions = dimensions
        self._batch_size = batch_size
        # Measured through OpenRouter: 3 to 4 seconds for a handful of texts, and over 10 for 64.
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        request: dict[str, Any] = {"model": self.name, "input": texts}
        if self._dimensions:
            request["dimensions"] = self._dimensions
        body = await _post(self._client, self._url, self._key, request)
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            vectors = [np.asarray(row["embedding"], dtype=np.float32) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise HostedModelError("the reply held no embeddings") from exc
        if len(vectors) != len(texts):
            raise HostedModelError(f"{len(vectors)} embeddings for {len(texts)} texts")
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            vectors += await self._embed(texts[start : start + self._batch_size])
        return vectors

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self._embed([self._query_prefix + text]))[0]


class HostedReranker:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = OPENROUTER_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = model
        self._key = api_key
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def score(self, query: str, documents: list[str]) -> list[float]:
        """One score per document, in the order given, on the logit scale."""
        if not documents:
            return []
        body = await _post(
            self._client,
            self._url,
            self._key,
            {"model": self.name, "query": query, "documents": documents, "top_n": len(documents)},
        )
        try:
            by_index = {row["index"]: float(row["relevance_score"]) for row in body["results"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise HostedModelError("the reply held no scores") from exc
        if sorted(by_index) != list(range(len(documents))):
            raise HostedModelError(f"scores for {len(by_index)} of {len(documents)} documents")
        return [_logit(by_index[i]) for i in range(len(documents))]
