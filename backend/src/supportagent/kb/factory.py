"""Turns a model name from the settings into an embedder or a reranker.

`gateway:cohere/rerank-v5-fast` is served by the AI Gateway. A name without the prefix is a small
ONNX model that runs in this process.
"""

from __future__ import annotations

from supportagent.config import Settings
from supportagent.kb.hosted import GatewayEmbedder, GatewayReranker, MissingKey
from supportagent.kb.models import Embedder, FastEmbedEmbedder, FastEmbedReranker, Reranker

GATEWAY = "gateway:"


def _key(settings: Settings, name: str) -> str:
    if settings.ai_gateway_api_key is None:
        raise MissingKey(f"{name} needs AI_GATEWAY_API_KEY")
    return settings.ai_gateway_api_key.get_secret_value()


def make_embedder(name: str, settings: Settings) -> Embedder:
    if not name.startswith(GATEWAY):
        return FastEmbedEmbedder(name)
    return GatewayEmbedder(
        name.removeprefix(GATEWAY),
        api_key=_key(settings, name),
        base_url=settings.ai_gateway_base_url,
        query_instruction=settings.embed_query_instruction,
        dimensions=settings.embed_dimensions,
    )


def make_reranker(name: str, settings: Settings) -> Reranker:
    if not name.startswith(GATEWAY):
        return FastEmbedReranker(name)
    return GatewayReranker(
        name.removeprefix(GATEWAY),
        api_key=_key(settings, name),
        base_url=settings.ai_gateway_base_url,
    )
