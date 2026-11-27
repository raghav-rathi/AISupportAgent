from __future__ import annotations

import json

import pytest

from supportagent import clock
from supportagent.config import get_settings
from supportagent.evals.judge import evidence_of, resolve
from supportagent.evals.judge_cases import CASES
from supportagent.kb.chunker import load_kb


@pytest.fixture(scope="module")
def chunks():
    return load_kb(get_settings().kb_dir)


def test_the_cases_are_balanced_and_every_piece_of_evidence_exists(chunks):
    assert len({case.id for case in CASES}) == len(CASES)
    assert sum(case.grounded for case in CASES) == sum(not case.grounded for case in CASES) == 26
    # Only a reply that stands can leave a question open.
    assert all(case.grounded for case in CASES if case.unresolved)
    assert sum(case.unresolved for case in CASES) == 4
    # Evidence only a verified customer can have comes with that customer.
    plans = {case.id: (case.customer or {}).get("tier") for case in CASES}
    assert (plans["faithful-limit"], plans["faithful-table"], plans["not-listed"]) == (
        "pro",
        "enterprise",
        None,
    )
    for case in CASES:
        for evidence in case.evidence:
            name, _, result = resolve(evidence, chunks).partition("\n")
            assert name.startswith("tool: ") and json.loads(result), case.id


def test_evidence_resolves_to_what_the_judge_reads_in_the_app(chunks):
    name, _, result = resolve("kb:refund-policy/Pro plan refunds", chunks).partition("\n")
    assert name == "tool: search_knowledge_base"
    passage = json.loads(result)["results"][0]
    assert (passage["n"], passage["source"]) == (1, "Refund policy > Pro plan refunds")
    assert "14 days from the invoice date" in passage["content"]

    today = clock.today().isoformat()
    assert resolve('issue_refund: {"on": "{today}"}', chunks) == (
        f'tool: issue_refund\n{{"on": "{today}"}}'
    )
    with pytest.raises(KeyError):
        resolve("kb:refund-policy/No such section", chunks)


def test_passage_numbers_run_on_from_one_piece_of_evidence_to_the_next(chunks):
    case = next(case for case in CASES if case.id == "two-passages")
    numbers = [
        passage["n"]
        for text in evidence_of(case, chunks)
        for passage in json.loads(text.partition("\n")[2])["results"]
    ]
    assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 2
