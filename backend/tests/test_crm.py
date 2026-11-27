from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from supportagent.crm import repo
from supportagent.crm import world as worlds

MAYA = "10482917"  # Pro monthly, INV-30417 is 5 days old
DANIEL = "20573846"  # Pro annual, INV-30388
PRIYA = "31764025"  # Pro monthly, INV-30212 and INV-30213 are a duplicate pair, 21 days old
MARCUS = "64037291"  # Enterprise, suspended


async def refund(pool, world, account, invoice, today, call_id=None):
    async with pool.connection() as conn:
        return await repo.refund_invoice(
            conn,
            world_id=world,
            account_number=account,
            invoice_id=invoice,
            reason="changed_mind",
            tool_call_id=call_id,
            today=today,
        )


async def scalar(pool, query, *params):
    async with pool.connection() as conn:
        cur = await conn.execute(query, params)
        return next(iter((await cur.fetchone()).values()))


async def refund_count(pool, world):
    return await scalar(pool, "SELECT count(*) FROM refunds WHERE world_id = %s", world)


async def invoice_status(pool, world, invoice):
    return await scalar(
        pool, "SELECT status FROM invoices WHERE world_id = %s AND id = %s", world, invoice
    )


async def test_invoice_dates_follow_today(pool, world, today):
    async with pool.connection() as conn:
        invoices = await repo.list_invoices(conn, world, MAYA)
    assert [i.id for i in invoices] == ["INV-30417", "INV-29880", "INV-29310"]
    assert invoices[0].issued_on == today - timedelta(days=5)
    assert invoices[0].amount_cents == 9_600


async def test_verify_pin_accepts_only_the_right_pair(pool, world):
    async with pool.connection() as conn:
        assert (await repo.verify_pin(conn, world, MAYA, "4721")).tier == "pro"
        assert await repo.verify_pin(conn, world, MAYA, "0000") is None
        assert await repo.verify_pin(conn, world, "00000000", "4721") is None
        # Another customer's PIN must not open this account.
        assert await repo.verify_pin(conn, world, MAYA, "9034") is None


async def test_refund_marks_the_invoice_and_records_one_refund(pool, world, today):
    outcome = await refund(pool, world, MAYA, "INV-30417", today, "call-1")
    assert outcome.decision.code == "ok"
    assert outcome.amount_cents == 9_600
    assert outcome.refund_id.startswith("RF-")
    assert await invoice_status(pool, world, "INV-30417") == "refunded"
    assert await refund_count(pool, world) == 1


async def test_a_second_request_for_the_same_invoice_does_not_pay_twice(pool, world, today):
    first = await refund(pool, world, MAYA, "INV-30417", today, "call-1")
    second = await refund(pool, world, MAYA, "INV-30417", today, "call-2")
    assert second.decision.code == "already_refunded"
    assert second.refund_id == first.refund_id
    assert await refund_count(pool, world) == 1


async def test_a_retry_of_the_same_tool_call_reports_the_original_refund(pool, world, today):
    first = await refund(pool, world, MAYA, "INV-30417", today, "call-1")
    retry = await refund(pool, world, MAYA, "INV-30417", today, "call-1")
    assert retry.decision.code == "ok"
    assert retry.replayed is True
    assert retry.refund_id == first.refund_id
    assert await refund_count(pool, world) == 1


async def test_concurrent_refunds_pay_once(pool, world, today):
    outcomes = await asyncio.gather(
        *(refund(pool, world, MAYA, "INV-30417", today, f"call-{n}") for n in range(6))
    )
    codes = sorted(o.decision.code for o in outcomes)
    assert codes == ["already_refunded"] * 5 + ["ok"]
    assert await refund_count(pool, world) == 1


async def test_one_customer_cannot_refund_another_customers_invoice(pool, world, today):
    outcome = await refund(pool, world, MAYA, "INV-30388", today)
    assert outcome.decision.code == "invoice_not_found"
    assert await invoice_status(pool, world, "INV-30388") == "paid"
    assert await refund_count(pool, world) == 0


async def test_only_one_invoice_of_a_duplicate_pair_is_refunded(pool, world, today):
    # 21 days old: past the 14-day window, allowed only because the charge is a duplicate.
    first = await refund(pool, world, PRIYA, "INV-30213", today)
    assert first.decision.code == "ok"
    # Its twin is no longer a duplicate, so the normal window applies again.
    second = await refund(pool, world, PRIYA, "INV-30212", today)
    assert second.decision.code == "outside_window"
    assert await refund_count(pool, world) == 1


async def test_refused_refund_changes_nothing(pool, world, today):
    outcome = await refund(pool, world, DANIEL, "INV-30388", today)
    assert outcome.decision.code == "over_limit"
    assert outcome.decision.escalate is True
    assert outcome.refund_id is None
    assert await invoice_status(pool, world, "INV-30388") == "paid"
    assert await refund_count(pool, world) == 0


async def test_worlds_do_not_share_state(pool, world, today):
    other = f"test-{uuid.uuid4().hex[:12]}"
    async with pool.connection() as conn:
        await worlds.create_world(conn, other, today)
    try:
        await refund(pool, world, MAYA, "INV-30417", today)
        assert await invoice_status(pool, other, "INV-30417") == "paid"
        assert (await refund(pool, other, MAYA, "INV-30417", today)).decision.code == "ok"
    finally:
        async with pool.connection() as conn:
            await worlds.delete_world(conn, other)


async def test_deleting_a_world_removes_its_rows(pool, today):
    world_id = f"test-{uuid.uuid4().hex[:12]}"
    async with pool.connection() as conn:
        await worlds.create_world(conn, world_id, today)
    await refund(pool, world_id, MAYA, "INV-30417", today)
    async with pool.connection() as conn:
        await worlds.delete_world(conn, world_id)
    for table in ("customers", "invoices", "refunds"):
        assert (
            await scalar(pool, f"SELECT count(*) FROM {table} WHERE world_id = %s", world_id) == 0
        )


async def test_update_field_validates_changes_and_audits(pool, world):
    async with pool.connection() as conn:
        with pytest.raises(repo.FieldError):
            await repo.update_field(
                conn, world_id=world, account_number=MAYA, field="billing_email", value="nope"
            )
        with pytest.raises(repo.FieldError):
            await repo.update_field(
                conn, world_id=world, account_number=MAYA, field="tier", value="enterprise"
            )
        with pytest.raises(repo.FieldError):
            await repo.update_field(
                conn,
                world_id=world,
                account_number=MARCUS,
                field="billing_email",
                value="ap@ostranderhealth.example",
            )
        change = await repo.update_field(
            conn,
            world_id=world,
            account_number=MAYA,
            field="billing_email",
            value=" Accounts@LarkspurDesign.example ",
        )
        customer = await repo.get_customer(conn, world, MAYA)
    assert change.new_value == "accounts@larkspurdesign.example"
    assert customer.billing_email == "accounts@larkspurdesign.example"
    assert customer.tier == "pro"
    assert (
        await scalar(pool, "SELECT count(*) FROM account_changes WHERE world_id = %s", world) == 1
    )


async def test_purging_removes_unused_conversations_and_keeps_the_ones_in_use(pool, today):
    names = ("unused", "new", "old_but_active", "old_and_mid_turn", "old_with_a_dead_turn")
    ids = {name: f"test-{uuid.uuid4().hex[:12]}" for name in names}
    async with pool.connection() as conn:
        for world_id in ids.values():
            await worlds.create_world(conn, world_id, today)
            await conn.execute(
                "INSERT INTO sessions (id, world_id) VALUES (%s, %s)", (world_id, world_id)
            )
        old = [ids[name] for name in names if name != "new"]
        await conn.execute(
            "UPDATE worlds SET created_at = now() - interval '3 days' WHERE id = ANY(%s)", (old,)
        )
        await conn.execute(
            "UPDATE sessions SET last_active_at = now() - interval '3 days' WHERE id = ANY(%s)",
            (old,),
        )
        # Someone came back to this one an hour ago.
        await conn.execute(
            "UPDATE sessions SET last_active_at = now() - interval '1 hour' WHERE id = %s",
            (ids["old_but_active"],),
        )
        # A turn is running on this one, and its process renewed the claim a moment ago.
        await conn.execute(
            "UPDATE sessions SET turn_running = true, turn_heartbeat_at = now() WHERE id = %s",
            (ids["old_and_mid_turn"],),
        )
        # The process that ran this one died long ago and left its claim behind.
        await conn.execute(
            "UPDATE sessions SET turn_running = true, "
            "turn_heartbeat_at = now() - interval '2 days' WHERE id = %s",
            (ids["old_with_a_dead_turn"],),
        )
    try:
        async with pool.connection() as conn:
            assert await worlds.purge_worlds(conn, timedelta(days=2)) == 2
            cur = await conn.execute(
                "SELECT id FROM sessions WHERE id = ANY(%s)", (list(ids.values()),)
            )
            left = {row["id"] for row in await cur.fetchall()}
            assert left == {ids["new"], ids["old_but_active"], ids["old_and_mid_turn"]}
            cur = await conn.execute(
                "SELECT count(*) AS n FROM customers WHERE world_id = %s", (ids["unused"],)
            )
            assert (await cur.fetchone())["n"] == 0
    finally:
        async with pool.connection() as conn:
            for world_id in ids.values():
                await worlds.delete_world(conn, world_id)


async def test_purging_keeps_a_conversation_that_is_open_in_this_process(pool, today):
    ids = [f"test-{uuid.uuid4().hex[:12]}" for _ in range(2)]
    open_here, unused = ids
    async with pool.connection() as conn:
        for world_id in ids:
            await worlds.create_world(conn, world_id, today)
            await conn.execute(
                "INSERT INTO sessions (id, world_id) VALUES (%s, %s)", (world_id, world_id)
            )
        await conn.execute(
            "UPDATE worlds SET created_at = now() - interval '3 days' WHERE id = ANY(%s)", (ids,)
        )
        # Both were quiet for three days. One was resumed a moment ago and has not done anything.
        await conn.execute(
            "UPDATE sessions SET last_active_at = now() - interval '3 days' WHERE id = ANY(%s)",
            (ids,),
        )
    try:
        async with pool.connection() as conn:
            assert await worlds.purge_worlds(conn, timedelta(days=2), keep={open_here}) == 1
            cur = await conn.execute("SELECT id FROM sessions WHERE id = ANY(%s)", (ids,))
            assert {row["id"] for row in await cur.fetchall()} == {open_here}
    finally:
        async with pool.connection() as conn:
            for world_id in ids:
                await worlds.delete_world(conn, world_id)
