from __future__ import annotations

from datetime import date, timedelta

import pytest

from supportagent.crm.policy import evaluate_refund

TODAY = date(2026, 11, 24)


def decide(age_days: int = 5, **overrides):
    case = {
        "tier": "pro",
        "billing_cycle": "monthly",
        "account_status": "active",
        "invoice_status": "paid",
        "amount_cents": 9_600,
        "issued_on": TODAY - timedelta(days=age_days),
        "today": TODAY,
        "is_duplicate": False,
    }
    case.update(overrides)
    return evaluate_refund(**case)


@pytest.mark.parametrize(
    ("tier", "cycle", "age_days", "code"),
    [
        ("pro", "monthly", 14, "ok"),
        ("pro", "monthly", 15, "outside_window"),
        ("pro", "annual", 30, "ok"),
        ("pro", "annual", 31, "outside_window"),
        ("enterprise", "annual", 30, "ok"),
        ("enterprise", "annual", 31, "outside_window"),
    ],
)
def test_window_is_inclusive_and_depends_on_plan(tier, cycle, age_days, code):
    decision = decide(age_days, tier=tier, billing_cycle=cycle)
    assert decision.code == code
    assert decision.allowed is (code == "ok")
    assert decision.escalate is False


@pytest.mark.parametrize(
    ("tier", "amount_cents", "code"),
    [
        ("pro", 50_000, "ok"),
        ("pro", 50_001, "over_limit"),
        ("enterprise", 200_000, "ok"),
        ("enterprise", 200_001, "over_limit"),
    ],
)
def test_support_limit_is_inclusive_and_larger_refunds_go_to_a_human(tier, amount_cents, code):
    decision = decide(tier=tier, billing_cycle="annual", amount_cents=amount_cents)
    assert decision.code == code
    assert decision.escalate is (code == "over_limit")


def test_duplicate_charge_gets_ninety_days_but_keeps_the_limit():
    assert decide(90, is_duplicate=True).code == "ok"
    assert decide(91, is_duplicate=True).code == "outside_window"
    assert decide(20, is_duplicate=True, amount_cents=50_001).code == "over_limit"


@pytest.mark.parametrize(
    ("overrides", "code", "escalate"),
    [
        ({"account_status": "suspended"}, "account_suspended", True),
        ({"account_status": "past_due"}, "account_past_due", False),
        ({"account_status": "cancelled"}, "ok", False),
        ({"invoice_status": "open"}, "invoice_not_paid", False),
        ({"invoice_status": "refunded"}, "already_refunded", False),
        ({"tier": "free", "billing_cycle": None}, "free_plan", True),
    ],
)
def test_account_and_invoice_state_gate_the_refund(overrides, code, escalate):
    decision = decide(**overrides)
    assert decision.code == code
    assert decision.escalate is escalate


def test_a_suspended_account_is_refused_even_inside_window_and_limit():
    decision = decide(1, account_status="suspended", amount_cents=100)
    assert not decision.allowed


def test_the_billing_team_lifts_the_two_limits_that_send_a_case_to_it_and_no_other():
    # Above the support limit, and on a suspended account: both are refused with "needs the
    # billing team", and a person there may approve them.
    assert decide(amount_cents=144_000).code == "over_limit"
    assert decide(amount_cents=144_000, by_billing=True).allowed
    assert decide(account_status="suspended", by_billing=True).allowed
    # Nobody refunds outside the window, twice, or with a balance unpaid.
    assert decide(age_days=40, by_billing=True).code == "outside_window"
    assert decide(invoice_status="refunded", by_billing=True).code == "already_refunded"
    assert decide(account_status="past_due", by_billing=True).code == "account_past_due"
    assert decide(tier="free", by_billing=True).code == "free_plan"
