"""Embedding and reranking models behind two small interfaces.

Hosted and local models implement the same two protocols, so retrieval code and evals can swap them.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


class Embedder(Protocol):
    name: str

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    name: str

    async def score(self, query: str, documents: list[str]) -> list[float]:
        """One raw relevance score per document, higher is better. The scale differs per model."""
        ...


class FastEmbedEmbedder:
    """A small ONNX embedding model on CPU. The weights load on first use.

    Calls run one at a time. With lazy loading, two first calls in two threads both start loading
    the weights, and one of them reads a half-built model: a bake-off with three conversations at
    once lost most of its first candidate that way. A call takes milliseconds, so the lock costs
    little.
    """

    def __init__(self, model: str) -> None:
        self.name = model
        self._model = TextEmbedding(model_name=model, lazy_load=True)
        self._lock = threading.Lock()

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        def run() -> list[np.ndarray]:
            with self._lock:
                return [np.asarray(v, dtype=np.float32) for v in self._model.passage_embed(texts)]

        return await asyncio.to_thread(run)

    async def embed_query(self, text: str) -> np.ndarray:
        def run() -> np.ndarray:
            with self._lock:
                return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)

        return await asyncio.to_thread(run)


class FastEmbedReranker:
    """A small ONNX cross-encoder on CPU. Scores are raw logits. Calls run one at a time, as above."""

    def __init__(self, model: str) -> None:
        self.name = model
        self._model = TextCrossEncoder(model_name=model, lazy_load=True)
        self._lock = threading.Lock()

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        def run() -> list[float]:
            with self._lock:
                return [float(s) for s in self._model.rerank(query, documents)]

        return await asyncio.to_thread(run)
