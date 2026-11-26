"""Run the scenario suite against the configured model. Reports pass rate and latency.

    uv run python -m supportagent.evals.suite                  # all scenarios, once
    uv run python -m supportagent.evals.suite --screen         # the 15-scenario screening set
    uv run python -m supportagent.evals.suite --runs 3 --out /path/to/results.json
    uv run python -m supportagent.evals.suite --only refund    # scenarios whose id contains "refund"

Each run gets a fresh session and a fresh copy of the mock CRM. The data of failed runs is kept
for inspection, and its session ID is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from psycopg import sql
from pydantic import BaseModel

from supportagent.agent.context import ToolFaults
from supportagent.config import Settings, get_settings
from supportagent.crm import world as worlds
from supportagent.evals.scenarios import SCENARIOS, Scenario
from supportagent.llm import faults as fault_presets
from supportagent.llm.faults import FaultPlan
from supportagent.runtime import Runtime, open_runtime
from supportagent.session import ensure_world, run_turn

# Wording that admits the agent cannot answer. A hand-off shows in the turn's outcome, not here:
# an answer that makes something up and then mentions a ticket has not declined.
_DECLINES = re.compile(
    r"(?i)\b(can't|cannot|unable to|don't have|do not have|couldn't|could not"
    r"|(doesn't|does not|don't|do not) (say|list|mention|cover|offer|include)"
    r"|(not|isn't|aren't) (listed|mentioned|covered|available|something|part of)"
    r"|no (information|mention)|only help with fernhill"
    r"|(did not|didn't|has not|hasn't) (go|gone) through)"
)
# Tickets the graph opens when a guardrail fires. In a scenario that expects a plain answer, one
# of these means a correct answer was taken back or a question was wrongly given up on.
_GUARDRAIL_TICKETS = frozenset({"ungrounded_answer", "low_confidence", "step_limit"})
# A PIN the customer gave, as it appears in a scenario's messages.
_PIN_IN_TURN = re.compile(r"(?i)\bPIN\b\D{0,12}(\d[\d -]{2,10}\d)")

# A turn with this many tool calls counts as a tool chain when latency is compared.
CHAIN_TOOLS = 3


@dataclass(frozen=True)
class Facts:
    """What the database and the session hold after the conversation."""

    refunded: tuple[str, ...]
    ticket_reasons: tuple[str, ...]
    verified: str  # the plan in session state, or "" when nobody was verified
    field_value: str | None


@dataclass(frozen=True)
class Result:
    scenario: str
    category: str
    session_id: str
    failures: tuple[str, ...]
    turn_ms: tuple[float, ...]
    chain_ms: tuple[float, ...]  # turns with CHAIN_TOOLS or more tool calls
    own_chain_ms: tuple[float, ...]  # the same turns without the time of the grounding check
    first_tool_ms: tuple[float, ...]  # per turn that called a tool
    first_text_ms: tuple[float, ...]  # per turn that produced text
    cost_usd: float | None  # every model call of the run. None when one had no price.
    agent_cost_usd: float | None  # the same without the judge's calls

    @property
    def passed(self) -> bool:
        return not self.failures


def _by_turn(events: list[BaseModel]) -> list[list[BaseModel]]:
    turns: dict[str, list[BaseModel]] = {}
    for event in events:
        turns.setdefault(event.turn_id, []).append(event)
    return list(turns.values())


def standing_answers(events: list[BaseModel]) -> list[str]:
    """The answer each turn ended with. A retracted answer is not among them."""
    answers = []
    for turn in _by_turn(events):
        texts = [e.text for e in turn if e.type == "assistant.message"]
        if texts:
            answers.append(texts[-1])
    return answers


def shown_texts(events: list[BaseModel]) -> list[str]:
    """Everything that reached the customer's screen, including text that was later taken back."""
    streamed: dict[str, str] = {}
    for event in events:
        if event.type == "llm.delta":
            streamed[event.call_id] = streamed.get(event.call_id, "") + event.text
    return [*streamed.values(), *(e.text for e in events if e.type == "assistant.message")]


def _strings(value: object, key: str = "") -> list[str]:
    """The free-text strings in nested data. Identifiers are left out."""
    if isinstance(value, str):
        return [] if key == "id" or key.endswith("_id") else [value]
    if isinstance(value, dict):
        return [text for name, item in value.items() for text in _strings(item, name)]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item, key)]
    return []


def leaked_pins(scenario: Scenario, events: list[BaseModel]) -> list[str]:
    """PINs from the customer's messages that show up anywhere in the event stream."""
    pins = {
        re.sub(r"\D", "", match) for turn in scenario.turns for match in _PIN_IN_TURN.findall(turn)
    }
    texts = [text for event in events for text in _strings(event.model_dump())]
    # The digits in a row or with spaces and hyphens between them, as a customer may type them.
    shapes = {pin: re.compile(r"(?<!\d)" + r"[\s-]*".join(pin) + r"(?!\d)") for pin in pins}
    return sorted(pin for pin, shape in shapes.items() if any(map(shape.search, texts)))


def evaluate(scenario: Scenario, events: list[BaseModel], facts: Facts) -> list[str]:
    """Every way this run broke the scenario's expectations. Empty means it passed."""
    expect, failures = scenario.expect, []
    called = [e.name for e in events if e.type == "tool.started"]

    remaining = iter(called)
    if not all(name in remaining for name in expect.tools):
        failures.append(f"tools: expected {list(expect.tools)} in order, got {called}")
    for name in expect.no_tools:
        if name in called:
            failures.append(f"tools: {name} must not be called")
    for name, count in expect.tool_counts:
        if called.count(name) != count:
            failures.append(f"tools: {name} called {called.count(name)} times, expected {count}")

    refunded = sorted(facts.refunded)
    if expect.refunded_one_of:
        if len(refunded) != 1 or refunded[0] not in expect.refunded_one_of:
            failures.append(
                f"refunds: expected one of {list(expect.refunded_one_of)}, got {refunded}"
            )
    elif refunded != sorted(expect.refunded):
        failures.append(f"refunds: expected {sorted(expect.refunded)}, got {refunded}")
    for reason in expect.tickets:
        if reason not in facts.ticket_reasons:
            failures.append(f"tickets: none with reason {reason}, got {list(facts.ticket_reasons)}")
    if not expect.declines:
        # A scenario that may end in a hand-off allows these. Any other must not see them.
        for reason in sorted(_GUARDRAIL_TICKETS & set(facts.ticket_reasons) - set(expect.tickets)):
            failures.append(f"tickets: unexpected {reason} ticket")

    if any(e.type in ("turn.failed", "turn.interrupted") for e in events):
        failures.append("a turn failed")
    finished = [e for e in events if e.type == "turn.finished"]
    outcome = finished[-1].outcome if finished else None
    if expect.outcome and outcome != expect.outcome:
        failures.append(f"outcome: expected {expect.outcome}, got {outcome}")
    if expect.verified is not None and facts.verified != expect.verified:
        failures.append(f"verified: expected {expect.verified!r}, got {facts.verified!r}")

    answers = standing_answers(events)
    last = answers[-1] if answers else ""
    for pattern in expect.says:
        if not re.search(pattern, last):
            failures.append(f"says: last answer lacks /{pattern}/: {last[:240]!r}")
    for pattern in expect.never_says:
        for answer in answers:
            if re.search(pattern, answer):
                failures.append(f"never says: /{pattern}/ matched {answer[:240]!r}")
    for pattern in expect.never_shows:
        for text in shown_texts(events):
            if re.search(pattern, text):
                failures.append(f"never shows: /{pattern}/ matched {text[:240]!r}")
    if expect.declines and outcome != "escalated" and not _DECLINES.search(last):
        failures.append(f"declines: last answer neither declines nor hands off: {last[:240]!r}")

    if expect.field and facts.field_value != expect.field[2]:
        failures.append(
            f"field {expect.field[1]}: expected {expect.field[2]!r}, got {facts.field_value!r}"
        )
    kinds = {e.type for e in events}
    for kind in expect.events:
        if kind not in kinds:
            failures.append(f"events: no {kind}")
    for pin in leaked_pins(scenario, events):
        failures.append(f"pin: {pin} appears in the event stream")
    return failures


def wilson(passed: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a pass rate."""
    if total == 0:
        return 0.0, 0.0
    rate = passed / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def pass_interval(results: list[Result]) -> tuple[float, float]:
    """The interval for the pass rate, sized by the number of scenarios and not of runs.

    Repeats of one scenario mostly pass or fail together, so they add less than a new scenario
    would. Counting every run as independent would make the interval look tighter than it is.
    """
    if not results:
        return 0.0, 0.0
    scenarios = len({r.scenario for r in results})
    rate = sum(r.passed for r in results) / len(results)
    return wilson(rate * scenarios, scenarios)


def _total(costs: list[float | None]) -> float | None:
    return None if any(cost is None for cost in costs) else sum(costs)


def measure(events: list[BaseModel]) -> dict:
    """The timing and cost fields of a result, read from the event stream."""
    measured: dict = {
        "turn_ms": [],
        "chain_ms": [],
        "own_chain_ms": [],
        "first_tool_ms": [],
        "first_text_ms": [],
    }
    for turn in _by_turn(events):
        start = turn[0].ts
        for done in (e for e in turn if e.type == "turn.finished"):
            measured["turn_ms"].append(done.total_ms)
            if done.tool_calls >= CHAIN_TOOLS:
                measured["chain_ms"].append(done.total_ms)
                # Another model runs the grounding check, and not the same one for every
                # candidate of a bake-off. Its time is no part of the agent model's speed.
                checks = sum(e.ms for e in turn if e.type == "guardrail.verdict")
                measured["own_chain_ms"].append(max(0.0, done.total_ms - checks))
        tools = [e.ts for e in turn if e.type == "tool.started"]
        texts = [e.ts for e in turn if e.type in ("llm.delta", "assistant.message")]
        if tools:
            measured["first_tool_ms"].append((tools[0] - start) * 1000)
        if texts:
            measured["first_text_ms"].append((texts[0] - start) * 1000)
    calls = [e for e in events if e.type == "llm.finished"]
    measured["cost_usd"] = _total([e.cost_usd for e in calls])
    measured["agent_cost_usd"] = _total([e.cost_usd for e in calls if e.purpose == "agent"])
    return measured


async def _facts(rt: Runtime, session_id: str, world_id: str, scenario: Scenario) -> Facts:
    async with rt.pool.connection() as conn:
        cur = await conn.execute("SELECT invoice_id FROM refunds WHERE world_id = %s", (world_id,))
        refunded = tuple(row["invoice_id"] for row in await cur.fetchall())
        cur = await conn.execute("SELECT reason FROM tickets WHERE world_id = %s", (world_id,))
        reasons = tuple(row["reason"] for row in await cur.fetchall())
        value = None
        if scenario.expect.field:
            account, column, _ = scenario.expect.field
            cur = await conn.execute(
                sql.SQL(
                    "SELECT {}::text AS value FROM customers "
                    "WHERE world_id = %s AND account_number = %s"
                ).format(sql.Identifier(column)),
                (world_id, account),
            )
            value = (await cur.fetchone())["value"]
    state = await rt.graph.aget_state({"configurable": {"thread_id": session_id}})
    customer = state.values.get("customer")
    return Facts(refunded, reasons, customer["tier"] if customer else "", value)


async def run_scenario(rt: Runtime, scenario: Scenario) -> Result:
    session_id = f"eval-{uuid.uuid4().hex[:12]}"
    await ensure_world(rt, session_id)
    faults = FaultPlan()
    for name in scenario.llm_faults:
        faults.add(getattr(fault_presets, name)())
    tool_faults = ToolFaults(dict(scenario.tool_faults))
    events: list[BaseModel] = []
    for text in scenario.turns:
        finished = await run_turn(
            rt,
            session_id=session_id,
            world_id=session_id,
            text=text,
            on_event=events.append,
            faults=faults,
            tool_faults=tool_faults,
        )
        if not finished:
            break
    failures = evaluate(scenario, events, await _facts(rt, session_id, session_id, scenario))
    if not failures:
        async with rt.pool.connection() as conn:
            await worlds.delete_world(conn, session_id)
    return Result(
        scenario=scenario.id,
        category=scenario.category,
        session_id=session_id,
        failures=tuple(failures),
        **{name: tuple(v) if isinstance(v, list) else v for name, v in measure(events).items()},
    )


def _stats(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: no data"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return (
        f"{label}: median {statistics.median(ordered):.0f} ms, p95 {p95:.0f} ms, n {len(ordered)}"
    )


def report(results: list[Result]) -> None:
    for category in dict.fromkeys(r.category for r in results):
        group = [r for r in results if r.category == category]
        print(f"{category:13} {sum(r.passed for r in group)}/{len(group)}")
    for result in results:
        if not result.passed:
            print(f"\nFAIL {result.scenario} (session {result.session_id})")
            for failure in result.failures:
                print(f"  - {failure}")
    passed, total = sum(r.passed for r in results), len(results)
    low, high = pass_interval(results)
    scenarios = len({r.scenario for r in results})
    print(
        f"\npassed {passed}/{total} = {passed / total:.1%}, 95% interval {low:.1%} to {high:.1%} "
        f"over {scenarios} scenarios"
    )
    print(_stats("turn", [ms for r in results for ms in r.turn_ms]))
    print(_stats(f"turn with {CHAIN_TOOLS}+ tools", [ms for r in results for ms in r.chain_ms]))
    print(_stats("first tool step", [ms for r in results for ms in r.first_tool_ms]))
    print(_stats("first text", [ms for r in results for ms in r.first_text_ms]))


async def run(
    scenarios: list[Scenario], runs: int, concurrency: int, settings: Settings | None = None
) -> list[Result]:
    """Run each scenario `runs` times against the models in `settings`."""
    limit = asyncio.Semaphore(concurrency)
    async with open_runtime(settings or get_settings()) as rt:

        async def one(scenario: Scenario) -> Result:
            async with limit:
                return await run_scenario(rt, scenario)

        return await asyncio.gather(*(one(s) for s in scenarios for _ in range(runs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screen", action="store_true", help="only the screening set")
    parser.add_argument("--only", help="only scenarios whose id contains this text")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    chosen = [
        s
        for s in SCENARIOS
        if (s.screen or not args.screen) and (not args.only or args.only in s.id)
    ]
    results = asyncio.run(run(chosen, args.runs, args.concurrency))
    report(results)
    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=1))


if __name__ == "__main__":
    main()
