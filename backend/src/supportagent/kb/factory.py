"""Turns a model name from the settings into an embedder or a reranker.

`openrouter:cohere/rerank-5-pro` is served by OpenRouter. A name without the prefix is a small
ONNX model that runs in this process.
"""

from __future__ import annotations

from supportagent.config import Settings
from supportagent.kb.hosted import HostedEmbedder, HostedReranker, MissingKey
from supportagent.kb.models import Embedder, FastEmbedEmbedder, FastEmbedReranker, Reranker

HOSTED = "openrouter:"


def _key(settings: Settings, name: str) -> str:
    if settings.openrouter_api_key is None:
        raise MissingKey(f"{name} needs OPENROUTER_API_KEY")
    return settings.openrouter_api_key.get_secret_value()


def make_embedder(name: str, settings: Settings) -> Embedder:
    if not name.startswith(HOSTED):
        return FastEmbedEmbedder(name)
    return HostedEmbedder(
        name.removeprefix(HOSTED),
        api_key=_key(settings, name),
        base_url=settings.openrouter_base_url,
        query_instruction=settings.embed_query_instruction,
        dimensions=settings.embed_dimensions,
    )


def make_reranker(name: str, settings: Settings) -> Reranker:
    if not name.startswith(HOSTED):
        return FastEmbedReranker(name)
    return HostedReranker(
        name.removeprefix(HOSTED),
        api_key=_key(settings, name),
        base_url=settings.openrouter_base_url,
    )
