"""Measure retrieval and the scope gate on labeled question sets.

Run with:
    uv run python -m supportagent.evals.retrieval ../evals/retrieval_dev.jsonl --reranker NAME ...
    uv run python -m supportagent.evals.retrieval ../evals/retrieval_heldout.jsonl \
        --reranker openrouter:cohere/rerank-5-pro --embedder openrouter:qwen/qwen3-embedding-8b
    uv run --group lab python -m supportagent.evals.retrieval ../evals/retrieval_dev.jsonl \
        --teacher Qwen/Qwen3-Reranker-4B

A name that starts with "openrouter:" is served by OpenRouter and needs its key. Any other name
is a small ONNX model that runs on CPU. `--teacher` names a Qwen3-Reranker that runs on a GPU
and needs the `lab` dependency group. The knowledge base must have been ingested with the embedder
in use: `python -m supportagent.cli.ingest --embedder NAME`.

Each line of a set is a JSON object:
    {"q": question, "tier": plan or null, "kind": "in" | "near" | "far" | "plan",
     "expect": [{"doc": document id, "path": [headings]}]}
`in` is a question the help center answers. `far` is off topic. `near` is on a topic the help
center covers without holding the answer. `plan` is a question that only another plan's
passages answer, so the right move is to ask the customer to verify or to name the plan.
A relevance score can separate `in` from `far` and
cannot separate `in` from `near`, so the threshold targets are measured on `far` only. Near misses
are listed here and judged on the final reply by the scenario suite.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from supportagent import db
from supportagent.agent.context import Gate
from supportagent.config import get_settings
from supportagent.kb.factory import make_embedder, make_reranker
from supportagent.kb.models import Embedder, Reranker
from supportagent.kb.retrieval import Hit, Retriever, SearchResult
from supportagent.kb.teacher import SUPPORT_INSTRUCTION, QwenReranker

# The gate targets from the plan: off-topic questions let through, in-scope questions escalated.
MAX_FALSE_ACCEPT = 0.02
MAX_FALSE_ESCALATION = 0.05
_ACCEPTED = ("answer", "check")
_BANDS = ("answer", "check", "verify", "restricted", "escalate")


def _rank(hits: list[Hit], expect: list[dict]) -> int | None:
    """1-based position of the first hit from an expected section, or None."""
    wanted = {(e["doc"], tuple(e["path"])) for e in expect}
    for position, hit in enumerate(hits, start=1):
        if (hit.doc_id, hit.heading_path) in wanted:
            return position
    return None


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _auc(positive: list[float], negative: list[float]) -> float:
    """Chance that a random in-scope question outscores a random out-of-scope one."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "0/0"


def _score(result: SearchResult) -> float:
    return float("-inf") if result.top_score is None else result.top_score


def _band(gate: Gate, case: dict, result: SearchResult) -> str:
    return gate.decide(result.top_score, result.plan_score, verified=case["tier"] is not None)


def _spread(bands: list[str]) -> str:
    counts = Counter(bands)
    return ", ".join(f"{band} {counts[band]}" for band in _BANDS if counts[band]) or "none"


def roc_points(inside: list[float], far: list[float], near: list[float]) -> list[dict]:
    """One row per threshold: what share of each kind a gate that accepts from there would accept."""
    rows = []
    for threshold in sorted({s for s in [*inside, *far, *near] if s != float("-inf")}):
        rows.append(
            {
                "accept_from": round(threshold, 4),
                "in_scope_escalated": sum(s < threshold for s in inside) / len(inside),
                "far_accepted": sum(s >= threshold for s in far) / len(far) if far else 0.0,
                "near_accepted": sum(s >= threshold for s in near) / len(near) if near else 0.0,
            }
        )
    return rows


def report(
    name: str, rows: list[tuple[dict, SearchResult]], limit: int, gate: Gate, verbose: bool
) -> list[dict]:
    """Print one reranker's results and return its ROC points."""
    inside = [(case, result) for case, result in rows if case["kind"] == "in"]
    far = [(case, result) for case, result in rows if case["kind"] == "far"]
    near = [(case, result) for case, result in rows if case["kind"] == "near"]
    gated = [(case, result) for case, result in rows if case["kind"] == "plan"]
    print(f"\n=== {name}")

    ranks = [_rank(result.hits, case["expect"]) for case, result in inside]
    fused = [_rank(result.fused, case["expect"]) for case, result in inside]
    n = len(inside)
    depth = max(len(result.fused) for _, result in rows)
    print(f"candidates (fused top {depth}): {_rate(sum(r is not None for r in fused), n)}")
    for k in (1, 3, limit):
        print(f"recall@{k}: {_rate(sum(r is not None and r <= k for r in ranks), n)}")
    for (case, result), rank in zip(inside, ranks, strict=True):
        if rank is None:
            got = "; ".join(f"{h.doc_id}/{' > '.join(h.heading_path)}" for h in result.hits[:3])
            print(f"  miss [{case['tier']}] {case['q']!r} -> {got}")

    # The gate as configured: what each kind of question would get.
    print(f"gate low {gate.low}, high {gate.high}, plan gap {gate.gap}:")
    in_bands = [_band(gate, case, result) for case, result in inside]
    print(f"  in scope: {_spread(in_bands)}")
    escalated = [c for (c, _), band in zip(inside, in_bands, strict=True) if band == "escalate"]
    print(f"    escalated although answerable: {_rate(len(escalated), n)}")
    # An in-scope question whose section the customer may see should not be told to verify.
    misdirected = [
        (case, result, band)
        for (case, result), band, found in zip(inside, in_bands, fused, strict=True)
        if band in ("verify", "restricted") and found is not None
    ]
    print(f"    sent to verify or another plan although visible: {_rate(len(misdirected), n)}")
    held = sum(band == "check" for band in in_bands)
    print(f"    held for the check instead of streamed: {_rate(held, n)}")
    for case in escalated:
        print(f"    escalated [{case['tier']}] {case['q']!r}")
    for case, result, band in misdirected:
        print(
            f"    {band} [{case['tier']}] {case['q']!r} "
            f"(visible {_score(result):.2f}, other plans {result.plan_score:.2f})"
        )
    for label, group in (("off topic", far), ("near miss", near)):
        bands = [_band(gate, case, result) for case, result in group]
        accepted = [c for (c, _), band in zip(group, bands, strict=True) if band in _ACCEPTED]
        print(f"  {label}: {_spread(bands)}")
        print(f"    accepted by the threshold: {_rate(len(accepted), len(group))}")
        for case in accepted if verbose or label == "off topic" else []:
            print(f"    accepted [{case['tier']}] {case['q']!r}")
    if gated:
        bands = [_band(gate, case, result) for case, result in gated]
        right = sum(band in ("verify", "restricted") for band in bands)
        print(f"  another plan's topic: {_spread(bands)}")
        print(f"    told to verify or told the plan: {_rate(right, len(gated))}")
    if verbose:
        for case, result in rows:
            other = "none" if result.plan_score is None else f"{result.plan_score:.2f}"
            print(
                f"  {case['kind']:4} {_band(gate, case, result):10} {_score(result):7.2f} "
                f"other {other:>6} [{case['tier']}] {case['q']!r}"
            )

    # What a single threshold could do at best, from the scores alone.
    in_scores = [_score(result) for _, result in inside]
    far_scores = [_score(result) for _, result in far]
    near_scores = [_score(result) for _, result in near]
    for label, scores in (("off topic", far_scores), ("near miss", near_scores)):
        if scores:
            print(
                f"top score in scope: min {min(in_scores):.2f} median "
                f"{statistics.median(in_scores):.2f} | {label}: median "
                f"{statistics.median(scores):.2f} max {max(scores):.2f} | AUC "
                f"{_auc(in_scores, scores):.3f}"
            )
    if far_scores:
        # Lowest threshold that keeps off-topic accepts within target, and what it costs.
        allowed = int(MAX_FALSE_ACCEPT * len(far_scores))
        ordered = sorted(far_scores, reverse=True)
        low = ordered[allowed] if allowed < len(ordered) else float("-inf")
        cost = sum(score <= low for score in in_scores)
        print(
            f"accept above {low:.2f}: off-topic accepts <= {allowed}, in-scope escalated "
            f"{_rate(cost, n)}"
        )
        # Highest threshold that keeps in-scope escalations within target, and what it lets in.
        high = sorted(in_scores)[int(MAX_FALSE_ESCALATION * n)]
        print(
            f"accept from {high:.2f}: in-scope escalated <= {int(MAX_FALSE_ESCALATION * n)}, "
            f"off-topic accepts {_rate(sum(s >= high for s in far_scores), len(far_scores))}, "
            f"near-miss accepts {_rate(sum(s >= high for s in near_scores), len(near_scores))}"
        )

    for label, values in (
        ("embed", [r.embed_ms for _, r in rows]),
        ("search", [r.search_ms for _, r in rows]),
        ("rerank", [r.rerank_ms for _, r in rows]),
        ("total", [r.embed_ms + r.search_ms + r.rerank_ms for _, r in rows]),
    ):
        print(
            f"{label} ms: median {statistics.median(values):.1f} p95 {_percentile(values, 0.95):.1f}"
        )
    return [{"reranker": name, **point} for point in roc_points(in_scores, far_scores, near_scores)]


async def run(
    cases: list[dict],
    embedder: Embedder,
    rerankers: list[tuple[str, Reranker]],
    limit: int,
    gate: Gate,
    verbose: bool,
) -> list[dict]:
    pool = db.make_pool(get_settings().database_url, max_size=2)
    await pool.open()
    points: list[dict] = []
    try:
        for label, reranker in rerankers:
            retriever = Retriever(embedder, reranker)
            await retriever.search(pool, "warm up", None)  # load the weights before timing
            rows = [
                (case, await retriever.search(pool, case["q"], case["tier"], limit=limit))
                for case in cases
            ]
            points += report(f"{label} | embedder {embedder.name}", rows, limit, gate, verbose)
    finally:
        await pool.close()
    return points


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sets", type=Path, nargs="+", help="one or more JSONL question sets")
    parser.add_argument("--embedder", default=settings.embed_model)
    parser.add_argument("--reranker", action="append", default=[], help="repeatable")
    parser.add_argument("--teacher", action="append", default=[], help="Qwen3-Reranker on the GPU")
    parser.add_argument("--instruction", default=SUPPORT_INSTRUCTION, help="for the teacher")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--low", type=float, default=settings.gate_low)
    parser.add_argument("--high", type=float, default=settings.gate_high)
    parser.add_argument("--gap", type=float, default=settings.gate_gap)
    parser.add_argument("--roc", type=Path, help="write the ROC points to this CSV file")
    parser.add_argument("--verbose", action="store_true", help="list every question")
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for path in args.sets
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not args.reranker and not args.teacher:
        args.reranker = [settings.rerank_model]
    rerankers: list[tuple[str, Reranker]] = [
        (name, make_reranker(name, settings)) for name in args.reranker
    ]
    rerankers += [
        (f"{name} | {args.instruction}", QwenReranker(name, args.instruction))
        for name in args.teacher
    ]
    gate = Gate(low=args.low, high=args.high, gap=args.gap)
    points = asyncio.run(
        run(
            cases, make_embedder(args.embedder, settings), rerankers, args.limit, gate, args.verbose
        )
    )
    if args.roc and points:
        with args.roc.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        print(f"\nROC points written to {args.roc}")


if __name__ == "__main__":
    main()
