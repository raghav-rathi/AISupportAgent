"""The selection rule is fixed before any candidate is measured. These tests pin it."""

from __future__ import annotations

import pytest

from supportagent.config import ModelSettings, Settings, get_settings
from supportagent.evals.bakeoff import Standing, candidate_settings, pick, standing
from supportagent.evals.suite import Result


def s(name, passed, chain_ms, cost=0.001, total=150):
    return Standing(name, passed, total, chain_ms, cost)


def test_a_faster_candidate_below_the_bar_does_not_win():
    assert pick([s("fast but wrong", 141, 900.0), s("slower and right", 143, 2400.0)]).name == (
        "slower and right"
    )


def test_among_those_that_pass_the_fastest_tool_chain_wins():
    assert pick([s("a", 150, 2400.0), s("b", 145, 1500.0), s("c", 148, 1900.0)]).name == "b"


def test_candidates_within_five_percent_are_tied_and_the_cheaper_one_wins():
    quick = s("quick and dear", 150, 1000.0, cost=0.004)
    close = s("close and cheap", 150, 1049.0, cost=0.001)
    behind = s("behind and cheapest", 150, 1100.0, cost=0.0001)
    assert pick([quick, close, behind]).name == "close and cheap"
    # A candidate whose price is not known cannot win a tie on price.
    unknown = s("close, price unknown", 150, 1010.0, cost=None)
    assert pick([quick, unknown]).name == "quick and dear"


def test_nobody_wins_when_nobody_reaches_the_bar():
    assert pick([s("a", 142, 900.0), s("b", 100, 500.0)]) is None


def result(failures, chain_ms, turns, agent_cost, total_cost=None):
    # The whole turn took a second longer than the candidate's own part: the judge's second.
    whole = tuple(ms + 1000.0 for ms in chain_ms)
    return Result(
        "x", "happy", "sid", failures, turns, whole, chain_ms, (), (), total_cost, agent_cost
    )


def test_a_standing_counts_passes_and_takes_speed_from_the_runs_that_passed():
    runs = [
        result((), (3000.0,), (3000.0,), 0.002),
        # This run failed after looping through its tools. Its long chain says nothing about
        # how fast the model answers correctly.
        result(("says: ...",), (9000.0,), (1000.0, 9000.0), 0.004),
        result((), (5000.0,), (800.0, 5000.0), 0.002),
    ]
    summary = standing("m", runs)
    assert (summary.passed, summary.total) == (2, 3)
    assert summary.chain_ms == 4000.0
    # Cost is per turn of every run, and it is the candidate's own calls, not the judge's.
    assert summary.cost_per_turn == pytest.approx(0.008 / 5)


def test_a_standing_has_no_cost_when_any_run_had_none():
    runs = [result((), (3000.0,), (3000.0,), 0.002), result((), (), (800.0,), None)]
    assert standing("m", runs).cost_per_turn is None


CANDIDATE = {"name": "qwen", "model": "alibaba/qwen3.8-27b", "input_price": 0.99}
GEMINI = {"model": "google/gemini-3.6-flash"}
HAIKU = {"model": "anthropic/claude-haiku-4.5"}


def test_a_candidate_runs_without_a_fallback_and_never_grades_itself():
    base = get_settings().model_copy(
        update={
            "fallback": ModelSettings(model="some/fallback"),
            "judge": ModelSettings(model="judge/from-the-environment"),
        }
    )
    spec = {
        "judge": GEMINI,
        "alternate_judge": HAIKU,
        "candidates": [CANDIDATE, {"name": "gemini", **GEMINI}],
    }
    chosen = candidate_settings(base, spec)
    assert chosen["qwen"].llm.model == "alibaba/qwen3.8-27b"
    assert chosen["qwen"].llm.input_price == 0.99
    assert all(settings.fallback is None for settings in chosen.values())
    assert chosen["qwen"].judge.model == "google/gemini-3.6-flash"
    assert chosen["gemini"].judge.model == "anthropic/claude-haiku-4.5"

    with pytest.raises(ValueError, match="grade itself"):
        candidate_settings(base, {"judge": GEMINI, "candidates": [{"name": "gemini", **GEMINI}]})
    # A development run on one local model may say that it accepts this.
    alone = {"judge": GEMINI, "allow_self_grading": True, "candidates": [{"name": "g", **GEMINI}]}
    assert candidate_settings(base, alone)["g"].judge.model == "google/gemini-3.6-flash"


def _limits(settings: Settings) -> tuple[float, ...]:
    return (
        settings.llm_first_token_timeout_s,
        settings.llm_stall_timeout_s,
        settings.llm_total_timeout_s,
        settings.llm_deadline_s,
    )


def test_a_candidate_runs_under_the_default_time_limits_whatever_the_environment_says():
    # A machine set up for a slow local model has long limits in its environment. A hosted
    # candidate measured under them would pass turns that time out for a customer.
    patient = get_settings().model_copy(
        update={
            "llm_first_token_timeout_s": 120.0,
            "llm_stall_timeout_s": 600.0,
            "llm_total_timeout_s": 240.0,
            "llm_deadline_s": 300.0,
        }
    )
    spec = {"judge": GEMINI, "candidates": [CANDIDATE]}
    defaults = _limits(Settings(database_url=patient.database_url, _env_file=None))
    assert _limits(candidate_settings(patient, spec)["qwen"]) == defaults != _limits(patient)

    # A file for a slow model says so itself, limit by limit.
    slow = candidate_settings(patient, {**spec, "limits": {"first_token_timeout_s": 120}})
    assert _limits(slow["qwen"]) == (120.0, *defaults[1:])
    with pytest.raises(ValueError, match="unknown limits"):
        candidate_settings(patient, {**spec, "limits": {"first_token_s": 8}})
