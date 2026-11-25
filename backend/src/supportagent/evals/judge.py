"""Measure the grounding judge on replies whose right verdict is known.

    uv run python -m supportagent.evals.judge              # every case, once
    uv run python -m supportagent.evals.judge --runs 3

Two error rates matter and they pull against each other. A miss lets an unsupported reply through.
A false alarm withdraws a correct reply and opens a ticket nobody needed. The judge is the one the
app itself would use: the model named by the JUDGE__ settings. It reads evidence in the shape the
app gives it: each tool result under the name of its tool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

from supportagent import clock
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


def resolve(evidence: str, chunks: list[Chunk]) -> str:
    """Turn a case's evidence into what the judge reads in the app.

    `kb:<document id>/<heading>/<heading>` becomes the result of a search that returned that
    section. `<tool name>: <JSON>` becomes that tool's result. `{today}` becomes the date.
    """
    if not evidence.startswith("kb:"):
        tool, _, result = evidence.partition(": ")
        return f"tool: {tool}\n{result.replace('{today}', clock.today().isoformat())}"
    wanted = evidence[3:].rstrip("/")
    # Compared as whole keys. Splitting on "/" would break a heading that contains one.
    found = [c for c in chunks if "/".join((c.doc_id, *c.heading_path)) == wanted]
    if not found:
        raise KeyError(f"no knowledge base section for {evidence!r}")
    if len({(c.doc_id, c.heading_path) for c in found}) > 1:
        raise KeyError(f"{evidence!r} names more than one section")
    results = [
        {"n": n, "source": " > ".join((c.doc_title, *c.heading_path)), "content": c.content}
        for n, c in enumerate(sorted(found, key=lambda c: c.ordinal), start=1)
    ]
    return "tool: search_knowledge_base\n" + json.dumps({"results": results})


async def rule(judge: Judge, case: JudgeCase, chunks: list[Chunk]) -> Ruling:
    try:
        verdict = await judge.check(
            question=case.question,
            earlier=case.earlier,
            answer=case.answer,
            # As in the app, the evidence opens with what the assistant's tools can do.
            evidence=[CAPABILITIES, *(resolve(e, chunks) for e in case.evidence)],
            emit=lambda _event: None,
        )
    except (JudgeError, LLMUnavailable):
        return Ruling(case, None, ())
    return Ruling(case, verdict.grounded, tuple(verdict.unsupported))


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
    for title, wrong in (("FALSE ALARM", alarms), ("MISS", misses), ("NO VERDICT", silent)):
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
