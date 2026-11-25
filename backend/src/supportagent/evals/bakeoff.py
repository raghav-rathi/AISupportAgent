"""Compare candidate agent models on the scenario suite and pick one by a rule fixed in advance.

    uv run python -m supportagent.evals.bakeoff ../evals/candidates.standin.json

The file is a JSON object:

    {"judge": {...}, "alternate_judge": {...}, "candidates": [{"name": "...", ...}, ...]}

Each candidate has a `name` plus the fields of an LLM__ setting: `model`, and optionally `base_url`,
`api_key`, `extra_body`, `strict_tools`, `json_schema`, `input_price`, `output_price`. `judge` and
`alternate_judge` take the same fields without a name.

Every candidate runs alone and is graded by someone else. It gets no fallback model, because a
fallback that quietly took over would earn the candidate credit for another model's answers. Its
answers are checked by `judge`, or by `alternate_judge` when `judge` is the candidate's own model.
Nothing is taken from the JUDGE__ and FALLBACK__ settings of the environment. A file for a
development run on one local model may set `"allow_self_grading": true` and say so in its results.

Stage 1 runs the 15 screening scenarios once per candidate. Stage 2 runs the full suite three
times for the best three.

The rule: a candidate must pass at least 95% of stage 2. Among those, the fastest median turn with
three or more tool calls wins, measured on the runs that passed. Candidates within 5% of the
fastest count as tied, and a tie goes to the lower cost per turn of the candidate's own calls. If
nobody reaches 95%, nobody wins and the bar stays where it is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supportagent.config import ModelSettings, Settings, get_settings
from supportagent.evals import suite
from supportagent.evals.scenarios import SCENARIOS

PASS_BAR = 0.95
TIE_MARGIN = 0.05


@dataclass(frozen=True)
class Standing:
    name: str
    passed: int
    total: int
    chain_ms: float | None  # median turn with three or more tool calls, over the runs that passed
    cost_per_turn: float | None  # the candidate's own calls. None when its price is not known.

    @property
    def rate(self) -> float:
        return self.passed / self.total


def standing(name: str, results: list[suite.Result]) -> Standing:
    """Sum up a candidate's runs. A failed run's timings say nothing about a working answer:
    a run that gave up early would look fast, and one that looped would look slow."""
    good = [r for r in results if r.passed]
    chains = [ms for r in good for ms in r.chain_ms]
    turns = sum(len(r.turn_ms) for r in results)
    costs = [r.agent_cost_usd for r in results]
    priced = turns and all(cost is not None for cost in costs)
    return Standing(
        name=name,
        passed=len(good),
        total=len(results),
        chain_ms=statistics.median(chains) if chains else None,
        cost_per_turn=sum(costs) / turns if priced else None,
    )


def pick(standings: list[Standing]) -> Standing | None:
    """Apply the rule. None means no candidate reached the bar."""
    passing = [s for s in standings if s.rate >= PASS_BAR and s.chain_ms is not None]
    if not passing:
        return None
    fastest = min(s.chain_ms for s in passing)
    tied = [s for s in passing if s.chain_ms <= fastest * (1 + TIE_MARGIN)]

    def cost(s: Standing) -> float:
        return float("inf") if s.cost_per_turn is None else s.cost_per_turn

    return min(tied, key=lambda s: (cost(s), s.chain_ms))


def candidate_settings(base: Settings, spec: dict[str, Any]) -> dict[str, Settings]:
    """The settings each candidate runs under, by candidate name."""
    judge = ModelSettings(**spec["judge"])
    alternate = ModelSettings(**spec["alternate_judge"]) if spec.get("alternate_judge") else None
    chosen: dict[str, Settings] = {}
    for entry in spec["candidates"]:
        model = ModelSettings(**{k: v for k, v in entry.items() if k != "name"})
        grader = judge
        if (model.model, model.base_url) == (judge.model, judge.base_url):
            if alternate is not None:
                grader = alternate
            elif not spec.get("allow_self_grading"):
                raise ValueError(f"{entry['name']} would grade itself: give an alternate_judge")
        chosen[entry["name"]] = base.model_copy(
            update={"llm": model, "fallback": None, "judge": grader}
        )
    return chosen


def _table(title: str, standings: list[Standing], results: dict[str, list[suite.Result]]) -> None:
    print(f"\n{title}")
    for s in standings:
        low, high = suite.pass_interval(results[s.name])
        chain = "n/a" if s.chain_ms is None else f"{s.chain_ms:.0f} ms"
        cost = "cost unknown" if s.cost_per_turn is None else f"${s.cost_per_turn:.5f} per turn"
        print(
            f"  {s.name:40} {s.passed}/{s.total} = {s.rate:.1%} ({low:.1%} to {high:.1%}), "
            f"chain {chain}, {cost}"
        )


async def compare(spec: dict[str, Any], finalists: int, runs: int, concurrency: int) -> None:
    candidates = candidate_settings(get_settings(), spec)

    screening = [s for s in SCENARIOS if s.screen]
    first_runs = {
        name: await suite.run(screening, 1, concurrency, settings)
        for name, settings in candidates.items()
    }
    first = [standing(name, results) for name, results in first_runs.items()]
    _table("Stage 1: screening set, one run", first, first_runs)

    best = sorted(first, key=lambda s: (-s.rate, s.chain_ms or float("inf")))[:finalists]
    second_runs = {
        s.name: await suite.run(list(SCENARIOS), runs, concurrency, candidates[s.name])
        for s in best
    }
    second = [standing(name, results) for name, results in second_runs.items()]
    _table(f"Stage 2: full suite, {runs} runs", second, second_runs)
    for name, results in second_runs.items():
        for result in results:
            if not result.passed:
                print(f"\nFAIL {name}: {result.scenario} (session {result.session_id})")
                for failure in result.failures:
                    print(f"  - {failure}")

    winner = pick(second)
    if winner is None:
        print(f"\nNo candidate passed {PASS_BAR:.0%}. Tighten the agent and rerun.")
    else:
        print(f"\nWinner by the rule: {winner.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    spec = json.loads(args.candidates.read_text())
    asyncio.run(compare(spec, args.finalists, args.runs, args.concurrency))


if __name__ == "__main__":
    main()
