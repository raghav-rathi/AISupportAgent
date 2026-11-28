"""Business rules for refunds and account changes.

The knowledge base documents state the same numbers. tests/test_kb_policy_sync.py fails if they drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Tier = Literal["free", "pro", "enterprise"]
TIERS: tuple[Tier, ...] = ("free", "pro", "enterprise")

# Days after the invoice date in which a refund is allowed, inclusive.
REFUND_WINDOW_DAYS: dict[tuple[str, str], int] = {
    ("pro", "monthly"): 14,
    ("pro", "annual"): 30,
    ("enterprise", "monthly"): 30,
    ("enterprise", "annual"): 30,
}
DUPLICATE_CHARGE_WINDOW_DAYS = 90

# Largest invoice support may refund without the billing team, in cents, inclusive.
AUTO_REFUND_CAP_CENTS: dict[str, int] = {"pro": 50_000, "enterprise": 200_000}

MAX_VERIFICATION_ATTEMPTS = 3

# Fields support may change after identity verification.
UPDATABLE_FIELDS: tuple[str, ...] = (
    "billing_email",
    "contact_name",
    "contact_phone",
    "marketing_opt_in",
)

RefundCode = Literal[
    "ok",
    "free_plan",
    "invoice_not_found",
    "invoice_not_paid",
    "already_refunded",
    "account_suspended",
    "account_past_due",
    "outside_window",
    "over_limit",
]


@dataclass(frozen=True)
class RefundDecision:
    code: RefundCode
    allowed: bool
    escalate: bool  # True when policy sends this case to a human
    message: str


def usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _deny(code: RefundCode, message: str, *, escalate: bool = False) -> RefundDecision:
    return RefundDecision(code=code, allowed=False, escalate=escalate, message=message)


FREE_PLAN_REFUND = _deny(
    "free_plan",
    "Free plan workspaces are never charged, so there is nothing to refund. "
    "A charge on a Free workspace needs the billing team.",
    escalate=True,
)


def evaluate_refund(
    *,
    tier: str,
    billing_cycle: str | None,
    account_status: str,
    invoice_status: str,
    amount_cents: int,
    issued_on: date,
    today: date,
    is_duplicate: bool,
    by_billing: bool = False,
) -> RefundDecision:
    """Decide whether one invoice may be refunded in full.

    Without `by_billing` this is what support, and so the assistant, may do. The two refusals
    that send a case to the billing team are the billing team's to lift: a person there may
    refund above the support limit and on a suspended account. Nobody refunds outside the
    window, twice, or on an account with an unpaid balance.
    """
    if tier == "free":
        return FREE_PLAN_REFUND
    if invoice_status == "refunded":
        return _deny("already_refunded", "This invoice was already refunded.")
    if invoice_status != "paid":
        return _deny(
            "invoice_not_paid", "This invoice has not been paid, so there is nothing to refund."
        )
    if account_status == "suspended" and not by_billing:
        return _deny(
            "account_suspended",
            "The account is suspended. Refunds on suspended accounts need a review by the "
            "billing team.",
            escalate=True,
        )
    if account_status == "past_due":
        return _deny(
            "account_past_due",
            "The account has an unpaid balance. Refunds are available once the open invoice is paid.",
        )

    age_days = (today - issued_on).days
    plan = "Enterprise" if tier == "enterprise" else f"Pro {billing_cycle}"
    if is_duplicate:
        window = DUPLICATE_CHARGE_WINDOW_DAYS
        rule = f"Duplicate charges can be refunded within {window} days of the invoice date."
    else:
        window = REFUND_WINDOW_DAYS[(tier, billing_cycle or "monthly")]
        rule = f"{plan} invoices can be refunded within {window} days of the invoice date."
    if age_days > window:
        return _deny("outside_window", f"The invoice is {age_days} days old. {rule}")

    cap = AUTO_REFUND_CAP_CENTS[tier]
    if amount_cents > cap and not by_billing:
        return _deny(
            "over_limit",
            f"The invoice is for {usd(amount_cents)}. Support can refund up to {usd(cap)} per "
            f"invoice on the {tier.capitalize()} plan. Larger refunds need the billing team.",
            escalate=True,
        )
    return RefundDecision(code="ok", allowed=True, escalate=False, message="Refund approved.")
