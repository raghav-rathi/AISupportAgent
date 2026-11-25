"""The mock customers and invoices.

Dates are offsets from today, so every new world starts with invoices of the same age. Account numbers
and PINs are digits only, so they survive speech recognition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvoiceFixture:
    id: str
    amount_cents: int
    description: str
    days_ago: int  # issued this many days before today
    period_days: int  # length of the period the invoice covers
    status: str = "paid"
    subscription: bool = True  # False for one-off purchases and add-ons


@dataclass(frozen=True)
class CustomerFixture:
    account_number: str
    pin: str
    company_name: str
    contact_name: str
    contact_email: str
    contact_phone: str
    tier: str
    billing_cycle: str | None
    status: str
    seats: int
    customer_since_days_ago: int
    invoices: tuple[InvoiceFixture, ...] = ()


def _monthly(invoice_id: str, seats: int, days_ago: int, status: str = "paid") -> InvoiceFixture:
    return InvoiceFixture(
        invoice_id, seats * 1_200, f"Pro plan, {seats} seats, monthly", days_ago, 30, status
    )


def _license(invoice_id: str, seats: int, days_ago: int) -> InvoiceFixture:
    return InvoiceFixture(
        invoice_id, seats * 24_000, f"Enterprise license, {seats} seats, annual", days_ago, 365
    )


def _addon(invoice_id: str, cents: int, description: str, days_ago: int) -> InvoiceFixture:
    return InvoiceFixture(invoice_id, cents, description, days_ago, 365, subscription=False)


CUSTOMERS: tuple[CustomerFixture, ...] = (
    # Pro monthly. A recent invoice inside the 14-day window and two outside it.
    CustomerFixture(
        "10482917",
        "4721",
        "Larkspur Design Co",
        "Maya Okafor",
        "maya.okafor@larkspurdesign.example",
        "+1 415 555 0142",
        "pro",
        "monthly",
        "active",
        8,
        400,
        (_monthly("INV-30417", 8, 5), _monthly("INV-29880", 8, 35), _monthly("INV-29310", 8, 66)),
    ),
    # Pro annual. Inside the 30-day window but above the $500 support limit.
    CustomerFixture(
        "20573846",
        "9034",
        "Quillon Freight",
        "Daniel Reyes",
        "daniel.reyes@quillonfreight.example",
        "+1 312 555 0187",
        "pro",
        "annual",
        "active",
        12,
        375,
        (InvoiceFixture("INV-30388", 144_000, "Pro plan, 12 seats, annual", 10, 365),),
    ),
    # Pro monthly. Charged twice for the same period, 21 days ago.
    CustomerFixture(
        "31764025",
        "1586",
        "Brindle & Wye Accounting",
        "Priya Natarajan",
        "priya@brindlewye.example",
        "+1 617 555 0113",
        "pro",
        "monthly",
        "active",
        4,
        210,
        (_monthly("INV-30212", 4, 21), _monthly("INV-30213", 4, 21), _monthly("INV-29655", 4, 52)),
    ),
    # Pro monthly, past due. One open invoice.
    CustomerFixture(
        "42851730",
        "6672",
        "Tamsin Labs",
        "Oliver Brandt",
        "oliver.brandt@tamsinlabs.example",
        "+1 206 555 0165",
        "pro",
        "monthly",
        "past_due",
        15,
        300,
        (_monthly("INV-30401", 15, 9, "open"), _monthly("INV-29840", 15, 40)),
    ),
    # Enterprise. One add-on under the $2,000 limit, one purchase above it.
    CustomerFixture(
        "53920164",
        "2458",
        "Harrowgate Robotics",
        "Elena Sokolova",
        "elena.sokolova@harrowgate.example",
        "+1 646 555 0128",
        "enterprise",
        "annual",
        "active",
        240,
        820,
        (
            _license("INV-28001", 240, 120),
            _addon("INV-30390", 90_000, "Additional storage, 500 GB, annual", 12),
            _addon("INV-30402", 450_000, "Premium onboarding package", 8),
        ),
    ),
    # Enterprise, suspended. Nothing is refunded or changed in chat.
    CustomerFixture(
        "64037291",
        "8815",
        "Ostrander Health Systems",
        "Marcus Bell",
        "marcus.bell@ostranderhealth.example",
        "+1 713 555 0176",
        "enterprise",
        "annual",
        "suspended",
        500,
        640,
        (
            _license("INV-27110", 500, 200),
            _addon("INV-30377", 120_000, "Admin training workshop", 6),
        ),
    ),
    # Free. No invoices.
    CustomerFixture(
        "75148302",
        "3390",
        "Pellam Studio",
        "Jun Watanabe",
        "jun@pellamstudio.example",
        "+1 503 555 0119",
        "free",
        None,
        "active",
        3,
        95,
    ),
    # Pro monthly, cancelled three days after the last charge. Still inside the window.
    CustomerFixture(
        "86259413",
        "7046",
        "Corvid Analytics",
        "Sofia Marchetti",
        "sofia.marchetti@corvidanalytics.example",
        "+1 917 555 0154",
        "pro",
        "monthly",
        "cancelled",
        6,
        150,
        (_monthly("INV-30430", 6, 3),),
    ),
    # Enterprise. An add-on bought 45 days ago, outside the 30-day window.
    CustomerFixture(
        "97360524",
        "5127",
        "Wexcombe Logistics",
        "Aisha Rahman",
        "aisha.rahman@wexcombe.example",
        "+1 404 555 0191",
        "enterprise",
        "annual",
        "active",
        80,
        510,
        (
            _license("INV-27760", 80, 150),
            _addon("INV-29905", 180_000, "Additional storage, 1 TB, annual", 45),
        ),
    ),
    # Free. No invoices.
    CustomerFixture(
        "18471635",
        "6209",
        "Nettlefield Bakery",
        "Tom Hughes",
        "tom@nettlefieldbakery.example",
        "+1 802 555 0137",
        "free",
        None,
        "active",
        2,
        40,
    ),
    # Pro monthly. An invoice on the last day of the window.
    CustomerFixture(
        "29582746",
        "4483",
        "Sable Peak Outfitters",
        "Grace Lindqvist",
        "grace.lindqvist@sablepeak.example",
        "+1 720 555 0108",
        "pro",
        "monthly",
        "active",
        20,
        260,
        (_monthly("INV-30299", 20, 14), _monthly("INV-29720", 20, 44)),
    ),
    # Enterprise. A purchase exactly at the $2,000 limit.
    CustomerFixture(
        "30693857",
        "9951",
        "Dunmore Legal Group",
        "Henry Castellanos",
        "henry.castellanos@dunmorelegal.example",
        "+1 215 555 0149",
        "enterprise",
        "annual",
        "active",
        60,
        470,
        (
            _license("INV-27990", 60, 130),
            _addon("INV-30350", 200_000, "Onboarding package, small team", 15),
        ),
    ),
)
