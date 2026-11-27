"""Measure the grounding judge on replies whose right verdict is known.

    uv run python -m supportagent.evals.judge              # every case, once
    uv run python -m supportagent.evals.judge --runs 3

Two error rates matter and they pull against each other. A miss lets an unsupported reply through.
A false alarm withdraws a correct reply and opens a ticket nobody needed. The judge is the one the
app itself would use: the model named by the JUDGE__ settings. It reads evidence in the shape the
app gives it: each tool result under the name of its tool.

The judge also says whether a grounded reply leaves a question about Fernhill open, which makes
the turn open a ticket. That has the same two error rates: a customer left with no answer and no
ticket, and a ticket on a question that was answered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

from supportagent import clock
from supportagent.agent.graph import who
from supportagent.agent.judge import Judge, JudgeError
from supportagent.agent.tools import CAPABILITIES
from supportagent.config import get_settings
from supportagent.evals.judge_cases import CASES, JudgeCase
from supportagent.evals.suite import wilson
from supportagent.kb.chunker import Chunk, load_kb
from supportagent.llm.errors import LLMUnavailable
from supportagent.runtime import NotConfigured, open_runtime


@dataclass(frozen=True)
class Ruling:
    case: JudgeCase
    verdict: bool | None  # None when the judge gave no usable verdict
    unsupported: tuple[str, ...]
    unresolved: bool = False  # the judge said the reply leaves a question open


def _sections(evidence: str, chunks: list[Chunk]) -> list[Chunk]:
    """The chunks of the one knowledge base section a `kb:` item names, in reading order."""
    wanted = evidence[3:].rstrip("/")
    # Compared as whole keys. Splitting on "/" would break a heading that contains one.
    found = [c for c in chunks if "/".join((c.doc_id, *c.heading_path)) == wanted]
    if not found:
        raise KeyError(f"no knowledge base section for {evidence!r}")
    if len({(c.doc_id, c.heading_path) for c in found}) > 1:
        raise KeyError(f"{evidence!r} names more than one section")
    return sorted(found, key=lambda c: c.ordinal)


def resolve(evidence: str, chunks: list[Chunk], first: int = 1) -> str:
    """Turn a case's evidence into what the judge reads in the app.

    `kb:<document id>/<heading>/<heading>` becomes the result of a search that returned that
    section, its passages numbered from `first`. `<tool name>: <JSON>` becomes that tool's result.
    `{today}` becomes the date.
    """
    if not evidence.startswith("kb:"):
        tool, _, result = evidence.partition(": ")
        return f"tool: {tool}\n{result.replace('{today}', clock.today().isoformat())}"
    results = [
        {"n": n, "source": " > ".join((c.doc_title, *c.heading_path)), "content": c.content}
        for n, c in enumerate(_sections(evidence, chunks), start=first)
    ]
    return "tool: search_knowledge_base\n" + json.dumps({"results": results})


def evidence_of(case: JudgeCase, chunks: list[Chunk]) -> list[str]:
    """A case's evidence as the app would hand it over: passage numbers run on from one search
    to the next, so that a number in the reply points at one passage."""
    texts, first = [], 1
    for item in case.evidence:
        texts.append(resolve(item, chunks, first))
        if item.startswith("kb:"):
            first += len(_sections(item, chunks))
    return texts


async def rule(judge: Judge, case: JudgeCase, chunks: list[Chunk]) -> Ruling:
    try:
        verdict = await judge.check(
            question=case.question,
            earlier=case.earlier,
            answer=case.answer,
            # As in the app, the evidence opens with what the assistant's tools can do and with
            # who the customer is.
            evidence=[
                CAPABILITIES,
                who(case.customer),
                *evidence_of(case, chunks),
            ],
            emit=lambda _event: None,
        )
    except (JudgeError, LLMUnavailable):
        return Ruling(case, None, ())
    return Ruling(case, verdict.grounded, tuple(verdict.unsupported), verdict.unresolved)


def report(rulings: list[Ruling]) -> None:
    grounded = [r for r in rulings if r.case.grounded]
    seeded = [r for r in rulings if not r.case.grounded]
    alarms = [r for r in grounded if r.verdict is False]
    misses = [r for r in seeded if r.verdict is True]
    silent = [r for r in rulings if r.verdict is None]

    for label, wrong, group in (
        ("false alarms on correct replies", alarms, grounded),
        ("misses on seeded errors", misses, seeded),
    ):
        if not group:
            print(f"{label}: no such cases in this run")
            continue
        low, high = wilson(len(wrong), len(group))
        rate = len(wrong) / len(group)
        print(f"{label}: {len(wrong)}/{len(group)} = {rate:.1%}, interval {low:.1%} to {high:.1%}")
    print(f"no verdict: {len(silent)}/{len(rulings)}")
    # Whether a question is left open only matters for a reply that stands.
    stood = [r for r in grounded if r.verdict is True]
    left_open = [r for r in stood if r.case.unresolved]
    settled = [r for r in stood if not r.case.unresolved]
    unnoticed = [r for r in left_open if not r.unresolved]
    needless = [r for r in settled if r.unresolved]
    for label, wrong, group in (
        ("open questions the judge did not flag", unnoticed, left_open),
        ("settled replies flagged as open", needless, settled),
    ):
        if group:
            low, high = wilson(len(wrong), len(group))
            rate = len(wrong) / len(group)
            print(
                f"{label}: {len(wrong)}/{len(group)} = {rate:.1%}, interval {low:.1%} to {high:.1%}"
            )
    for title, wrong in (
        ("FALSE ALARM", alarms),
        ("MISS", misses),
        ("NO VERDICT", silent),
        ("OPEN QUESTION NOT FLAGGED", unnoticed),
        ("SETTLED REPLY FLAGGED AS OPEN", needless),
    ):
        for ruling in wrong:
            case = ruling.case
            print(f"\n{title} {case.id} ({case.kind})\n  reply: {case.answer}")
            if ruling.unsupported:
                print(f"  judge said unsupported: {'; '.join(ruling.unsupported)}")


async def run(runs: int, concurrency: int, only: str | None = None) -> list[Ruling]:
    settings = get_settings()
    chunks = load_kb(settings.kb_dir)
    limit = asyncio.Semaphore(concurrency)
    cases = [case for case in CASES if not only or only in case.id or only in case.kind]
    async with open_runtime(settings) as rt:
        if rt.judge is None:
            raise NotConfigured("no judge model: set JUDGE__MODEL")
        judge = rt.judge

        async def one(case: JudgeCase) -> Ruling:
            async with limit:
                return await rule(judge, case, chunks)

        return await asyncio.gather(*(one(case) for case in cases for _ in range(runs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--only", help="only cases whose id or kind contains this text")
    args = parser.parse_args()
    report(asyncio.run(run(args.runs, args.concurrency, args.only)))


if __name__ == "__main__":
    main()
