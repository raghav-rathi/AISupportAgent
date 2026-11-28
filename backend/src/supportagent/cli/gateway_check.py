"""Send one small request to every hosted model and say what came back. Exit code 1 on a failure.

    uv run python -m supportagent.cli.gateway_check ../evals/candidates.hosted.json \
        --embedder openrouter:qwen/qwen3-embedding-8b --reranker openrouter:cohere/rerank-5-pro

Run it once when a key arrives, before any eval. Until then every line it prints rests on
documentation alone: a model ID, a provider restriction, a reasoning setting, strict tool schemas,
a JSON schema on the judge's reply, the embedding endpoint, the rerank endpoint. A bake-off would
find the same mistakes, one candidate at a time and an hour in.

It starts with the health of every provider an OpenRouter model names, read from OpenRouter's
public listing without a key. A provider below 95% of attempts over the last 30 minutes fails the
check: do not start a bake-off then.

Without a candidates file it checks the models of the current settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_chunk_to_message,
)

from supportagent.agent.judge import VERDICT_SCHEMA, parse_verdict, request_messages
from supportagent.agent.prompts import system_prompt
from supportagent.agent.tools import TOOLS
from supportagent.config import ModelSettings, Settings, get_settings
from supportagent.evals.bakeoff import candidate_settings
from supportagent.kb.factory import make_embedder, make_reranker
from supportagent.kb.models import Embedder, Reranker
from supportagent.llm.client import ChatModel
from supportagent.runtime import build_model

_QUESTION = "How long do I have to ask for a refund on monthly billing?"
_RELEVANT = "Pro monthly invoices can be refunded within 14 days of the invoice date."
_UNRELATED = "The mobile app needs iOS 16 or later, or Android 11 or later."
_CREDENTIALS = "My account number is 10482917 and my PIN is 0000. Which plan are we on?"


@dataclass(frozen=True)
class Row:
    ok: bool
    kind: str  # chat | judge | embed | rerank
    name: str
    detail: str


def _error(exc: BaseException) -> str:
    """The error as the provider worded it, on one line. This is a setup tool, so it shows all."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


async def _reply(
    model: ChatModel,
    messages: list[BaseMessage],
    tools: list | None,
    json_schema: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[Any, float | None, float]:
    """One call straight to the model: the message, the time to its first token, the total time."""
    started = time.perf_counter()
    first: float | None = None
    merged = None
    async with asyncio.timeout(timeout_s):
        stream = (
            model.stream(messages, tools)
            if json_schema is None
            else model.stream(messages, tools, json_schema=json_schema)
        )
        async for chunk in stream:
            if first is None and (chunk.content or chunk.tool_call_chunks):
                first = (time.perf_counter() - started) * 1000
            merged = chunk if merged is None else merged + chunk
    if merged is None:
        raise ValueError("the stream ended without a single chunk")
    return message_chunk_to_message(merged), first, (time.perf_counter() - started) * 1000


def _usage(model: ChatModel, message: Any, first_ms: float | None, total_ms: float) -> str:
    usage = message.usage_metadata or {}
    tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    price = getattr(model, "price", None)
    cost = price.cost(tokens_in, tokens_out) if price else None
    served = (message.response_metadata or {}).get("model_name") or "unnamed"
    parts = [
        f"first token {first_ms:.0f} ms" if first_ms is not None else "no first token",
        f"total {total_ms:.0f} ms",
        f"{tokens_in} in and {tokens_out} out",
        f"${cost:.5f}" if cost is not None else "no price",
        f"served as {served}",
    ]
    return ", ".join(parts)


async def check_chat(name: str, model: ChatModel, timeout_s: float = 60.0) -> Row:
    """The agent's own kind of call: the system prompt, the seven tools, a message that needs one."""
    messages = [SystemMessage(system_prompt(None, "text")), HumanMessage(_CREDENTIALS)]
    try:
        message, first_ms, total_ms = await _reply(model, messages, TOOLS, None, timeout_s)
    except Exception as exc:
        return Row(False, "chat", name, _error(exc))
    usage = _usage(model, message, first_ms, total_ms)
    text = " ".join(str(message.text).split())
    if "<think>" in text:
        return Row(False, "chat", name, f"reasoning leaked into the reply text. {usage}")
    calls = [call["name"] for call in message.tool_calls]
    if calls != ["verify_identity"]:
        got = f"tool calls {calls}" if calls else f"text {text[:80]!r}"
        return Row(False, "chat", name, f"expected one verify_identity call, got {got}. {usage}")
    return Row(True, "chat", name, f"called verify_identity. {usage}")


async def check_judge(name: str, model: ChatModel, timeout_s: float = 60.0) -> Row:
    """The judge's kind of call: no tools, a JSON schema on the reply, a case that is grounded."""
    messages = request_messages(
        question=_QUESTION,
        answer="You have 14 days from the invoice date [1].",
        evidence=[
            f"tool: search_knowledge_base\n{json.dumps({'results': [{'n': 1, 'content': _RELEVANT}]})}"
        ],
    )
    try:
        message, first_ms, total_ms = await _reply(model, messages, None, VERDICT_SCHEMA, timeout_s)
        verdict = parse_verdict(str(message.text))
    except Exception as exc:
        return Row(False, "judge", name, _error(exc))
    usage = _usage(model, message, first_ms, total_ms)
    if not verdict.grounded:
        return Row(False, "judge", name, f"called a supported reply unsupported. {usage}")
    return Row(True, "judge", name, f"verdict grounded. {usage}")


async def check_embedder(name: str, embedder: Embedder) -> Row:
    started = time.perf_counter()
    try:
        query = await embedder.embed_query(_QUESTION)
        near, far = await embedder.embed_documents([_RELEVANT, _UNRELATED])
    except Exception as exc:
        return Row(False, "embed", name, _error(exc))
    ms = (time.perf_counter() - started) * 1000

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    close, distant = cosine(query, near), cosine(query, far)
    detail = (
        f"{len(query)} dimensions, related {close:.2f} and unrelated {distant:.2f}, {ms:.0f} ms"
    )
    return Row(close > distant, "embed", name, detail)


async def check_reranker(name: str, reranker: Reranker) -> Row:
    started = time.perf_counter()
    try:
        related, unrelated = await reranker.score(_QUESTION, [_RELEVANT, _UNRELATED])
    except Exception as exc:
        return Row(False, "rerank", name, _error(exc))
    ms = (time.perf_counter() - started) * 1000
    detail = f"related {related:.2f} and unrelated {unrelated:.2f}, {ms:.0f} ms"
    return Row(related > unrelated, "rerank", name, detail)


def _chat_specs(
    settings: Settings, spec: dict[str, Any] | None
) -> list[tuple[str, str, Settings, ModelSettings]]:
    """Every chat model to check, as (kind, label, the settings it runs under, its own settings)."""
    found: list[tuple[str, str, Settings, ModelSettings]] = []
    if spec is None:
        slots = (("chat", settings.llm), ("chat", settings.fallback), ("judge", settings.judge))
        return [(kind, model.model, settings, model) for kind, model in slots if model]
    judges: dict[str, ModelSettings] = {}
    for label, chosen in candidate_settings(settings, spec).items():
        found.append(("chat", label, chosen, chosen.llm))
        judges[chosen.judge.model] = chosen.judge
    return found + [("judge", model.model, settings, model) for model in judges.values()]


HEALTHY = 95.0  # percent of attempts over the last 30 minutes that a provider must reach


def _hosted_models(settings: Settings, specs: list[tuple[str, str, Settings, ModelSettings]]):
    """The (kind, label, model ID, named providers) of every model that goes through OpenRouter."""
    for kind, label, chosen, model in specs:
        if (model.base_url or chosen.openrouter_base_url) != settings.openrouter_base_url:
            continue
        provider = (model.extra_body or {}).get("provider") or {}
        yield kind, label, model.model, list(provider.get("only") or [])


async def check_health(
    client: httpx.AsyncClient, base_url: str, kind: str, label: str, model: str, only: list[str]
) -> list[Row]:
    """How each endpoint that may serve `model` has done lately, from OpenRouter's public listing.

    It needs no key. A bake-off should not start while a provider it names is failing: a candidate
    would be marked down for its provider's bad hour, and a judge that is down fails every turn it
    was meant to check. A provider slug such as `google-vertex` covers all its regions, but not
    its service tiers such as `flex` or `priority`, so those are left out.
    """
    name = f"{label} health"
    try:
        reply = await client.get(f"{base_url.rstrip('/')}/models/{model}/endpoints")
        reply.raise_for_status()
        endpoints = [e for e in reply.json()["data"]["endpoints"] if not e.get("service_tier")]
    except Exception as exc:
        return [Row(False, kind, name, _error(exc))]
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for entry in endpoints:
        by_slug.setdefault(str(entry.get("tag") or "").split("/")[0], []).append(entry)
    rows = []
    for provider in only or sorted(by_slug):
        if provider not in by_slug:
            rows.append(Row(False, kind, name, f"{provider} does not serve {model} today"))
            continue
        for entry in by_slug[provider]:
            recent, day = entry.get("uptime_last_30m"), entry.get("uptime_last_1d")
            latency = entry.get("latency_last_30m")
            median = latency.get("p50") if isinstance(latency, dict) else None
            detail = (
                f"{entry.get('tag')}: uptime {_percent(recent)} last 30 minutes and "
                f"{_percent(day)} last day, quantization {entry.get('quantization') or 'unknown'}, "
                f"latency median {_ms(median)}"
            )
            # No recent traffic is not a failure. A measured bad half hour is.
            rows.append(Row(recent is None or recent >= HEALTHY, kind, name, detail))
    return rows


def _percent(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.1f}%"


def _ms(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.0f} ms"


async def run(
    settings: Settings,
    spec: dict[str, Any] | None,
    embedders: list[str],
    rerankers: list[str],
    timeout_s: float = 60.0,
) -> list[Row]:
    rows: list[Row] = []
    specs = _chat_specs(settings, spec)
    async with httpx.AsyncClient(timeout=10.0) as client:
        for kind, label, model_id, only in _hosted_models(settings, specs):
            rows += await check_health(
                client, settings.openrouter_base_url, kind, label, model_id, only
            )
    for kind, label, chosen, model in specs:
        try:
            built = build_model(chosen, model)
        except Exception as exc:  # no key, most likely
            rows.append(Row(False, kind, label, _error(exc)))
            continue
        check = check_chat if kind == "chat" else check_judge
        rows.append(await check(label, built, timeout_s))
    for name in embedders:
        try:
            rows.append(await check_embedder(name, make_embedder(name, settings)))
        except Exception as exc:
            rows.append(Row(False, "embed", name, _error(exc)))
    for name in rerankers:
        try:
            rows.append(await check_reranker(name, make_reranker(name, settings)))
        except Exception as exc:
            rows.append(Row(False, "rerank", name, _error(exc)))
    return rows


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", type=Path, nargs="?", help="a bake-off candidates file")
    parser.add_argument("--embedder", action="append", default=[], help="repeatable")
    parser.add_argument("--reranker", action="append", default=[], help="repeatable")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per model call")
    args = parser.parse_args()
    spec = json.loads(args.candidates.read_text()) if args.candidates else None
    embedders = args.embedder or [settings.embed_model]
    rerankers = args.reranker or [settings.rerank_model]
    rows = asyncio.run(run(settings, spec, embedders, rerankers, args.timeout))
    for row in rows:
        print(f"{'ok  ' if row.ok else 'FAIL'} {row.kind:6} {row.name}: {row.detail}")
    return 0 if all(row.ok for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
