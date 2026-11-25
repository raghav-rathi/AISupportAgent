from __future__ import annotations

import json

import pytest

from supportagent import clock
from supportagent.config import get_settings
from supportagent.evals.judge import resolve
from supportagent.evals.judge_cases import CASES
from supportagent.kb.chunker import load_kb


@pytest.fixture(scope="module")
def chunks():
    return load_kb(get_settings().kb_dir)


def test_the_cases_are_balanced_and_every_piece_of_evidence_exists(chunks):
    assert len({case.id for case in CASES}) == len(CASES)
    assert sum(case.grounded for case in CASES) == sum(not case.grounded for case in CASES) == 22
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
