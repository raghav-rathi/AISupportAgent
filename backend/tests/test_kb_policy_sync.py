"""The documents the agent quotes and the rules the tools enforce must state the same numbers."""

from __future__ import annotations

import pytest

from supportagent.config import get_settings
from supportagent.crm import policy
from supportagent.crm.fixtures import CUSTOMERS
from supportagent.kb.chunker import load_kb


@pytest.fixture(scope="module")
def kb():
    return load_kb(get_settings().kb_dir)


def section(kb, doc_id: str, heading: str):
    chunks = [c for c in kb if c.doc_id == doc_id and heading in c.heading_path]
    assert chunks, f"no section {heading!r} in {doc_id}"
    return chunks


def text(chunks) -> str:
    return "\n".join(c.content for c in chunks)


def dollars(cents: int) -> str:
    return f"${cents // 100:,}"


def test_pro_refund_terms_match_policy(kb):
    pro = section(kb, "refund-policy", "Pro plan refunds")
    body = text(pro)
    assert f"| Monthly | {policy.REFUND_WINDOW_DAYS[('pro', 'monthly')]} days" in body
    assert f"| Annual | {policy.REFUND_WINDOW_DAYS[('pro', 'annual')]} days" in body
    assert f"up to {dollars(policy.AUTO_REFUND_CAP_CENTS['pro'])} on the spot" in body
    assert {c.tiers for c in pro} == {("pro",)}


def test_enterprise_refund_terms_match_policy(kb):
    enterprise = section(kb, "refund-policy", "Enterprise plan refunds")
    body = text(enterprise)
    assert f"within {policy.REFUND_WINDOW_DAYS[('enterprise', 'annual')]} days" in body
    assert f"up to {dollars(policy.AUTO_REFUND_CAP_CENTS['enterprise'])} on the spot" in body
    assert {c.tiers for c in enterprise} == {("enterprise",)}


def test_duplicate_charge_window_matches_policy(kb):
    body = text(section(kb, "refund-policy", "Duplicate charges"))
    assert f"up to {policy.DUPLICATE_CHARGE_WINDOW_DAYS} days" in body


def test_documented_account_changes_match_policy(kb):
    body = text(section(kb, "account-changes-and-cancellation", "What support can change in chat"))
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(bullets) == len(policy.UPDATABLE_FIELDS)


def test_storage_addon_prices_match_the_invoices(kb):
    body = text(section(kb, "storage-and-file-limits", "Enterprise"))
    invoices = {i.description: i.amount_cents for c in CUSTOMERS for i in c.invoices}
    assert (
        f"| 500 GB | {dollars(invoices['Additional storage, 500 GB, annual'])} per year |" in body
    )
    assert f"| 1 TB | {dollars(invoices['Additional storage, 1 TB, annual'])} per year |" in body


def test_pro_seat_prices_match_the_invoices(kb):
    body = text(section(kb, "plans-and-pricing", "Pro plan"))
    monthly = next(i for c in CUSTOMERS for i in c.invoices if i.id == "INV-30417")
    annual = next(i for c in CUSTOMERS for i in c.invoices if i.id == "INV-30388")
    assert f"{dollars(monthly.amount_cents // 8)} per seat per month" in body
    assert f"{dollars(annual.amount_cents // 12)} per seat per year" in body
