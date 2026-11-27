"""Worlds: private copies of the mock CRM, one per session or eval scenario."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, timedelta

from psycopg import AsyncConnection, sql

from supportagent.crm.fixtures import CUSTOMERS, CustomerFixture
from supportagent.crm.repo import hash_pin

# LangGraph's checkpoint tables hold the saved conversation of each session, keyed by the session
# ID as `thread_id`. They belong to the library and have no foreign key, so no cascade reaches
# them: a world that goes has to take its conversations along explicitly.
_THREAD_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


def _renews_on(customer: CustomerFixture, today: date) -> date | None:
    subscriptions = [i for i in customer.invoices if i.subscription]
    if customer.status == "cancelled" or not subscriptions:
        return None
    latest = min(subscriptions, key=lambda i: i.days_ago)
    return today - timedelta(days=latest.days_ago) + timedelta(days=latest.period_days)


async def create_world(conn: AsyncConnection, world_id: str, today: date) -> bool:
    """Insert a fresh copy of every fixture customer and invoice, dated relative to `today`.

    Returns False when the world already exists. Two callers may race to create the same world:
    the second one waits for the first to commit and then finds it complete.
    """
    customers = [
        (
            world_id,
            c.account_number,
            hash_pin(c.account_number, c.pin),
            c.company_name,
            c.contact_name,
            c.contact_email,
            c.contact_phone,
            c.contact_email,
            c.tier,
            c.billing_cycle,
            c.status,
            c.seats,
            today - timedelta(days=c.customer_since_days_ago),
            _renews_on(c, today),
        )
        for c in CUSTOMERS
    ]
    invoices = []
    for c in CUSTOMERS:
        for i in c.invoices:
            issued = today - timedelta(days=i.days_ago)
            invoices.append(
                (
                    world_id,
                    i.id,
                    c.account_number,
                    i.amount_cents,
                    i.description,
                    issued,
                    issued + timedelta(days=i.period_days),
                    issued,
                    i.status,
                )
            )
    async with conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO worlds (id) VALUES (%s) ON CONFLICT (id) DO NOTHING", (world_id,)
        )
        if cur.rowcount == 0:
            return False
        await cur.executemany(
            "INSERT INTO customers (world_id, account_number, pin_hash, company_name, "
            "contact_name, contact_email, contact_phone, billing_email, tier, billing_cycle, "
            "status, seats, customer_since, renews_on) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            customers,
        )
        await cur.executemany(
            "INSERT INTO invoices (world_id, id, account_number, amount_cents, description, "
            "period_start, period_end, issued_on, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            invoices,
        )
    return True


async def _has_thread_tables(conn: AsyncConnection) -> bool:
    """False in a database where no checkpointer has set up its tables."""
    cur = await conn.execute("SELECT to_regclass('checkpoints') IS NOT NULL AS present")
    return (await cur.fetchone())["present"]


async def _forget_threads(conn: AsyncConnection, where: sql.Composable, params: tuple) -> None:
    """Delete the saved conversations whose thread matches `where`."""
    if not await _has_thread_tables(conn):
        return
    for table in _THREAD_TABLES:
        await conn.execute(
            sql.SQL("DELETE FROM {} WHERE {}").format(sql.Identifier(table), where), params
        )


async def delete_world(conn: AsyncConnection, world_id: str) -> None:
    """Remove a world, everything in it, and the saved conversations of its sessions."""
    async with conn.transaction():
        await _forget_threads(
            conn,
            sql.SQL("thread_id IN (SELECT id FROM sessions WHERE world_id = %s)"),
            (world_id,),
        )
        await conn.execute("DELETE FROM worlds WHERE id = %s", (world_id,))


# A world is in use while one of its sessions did something lately, is running a turn that some
# process still renews, or is open in the process that purges. Such a world is never purged,
# however old it is.
_UNUSED = sql.SQL(
    "w.created_at < now() - %(age)s AND NOT EXISTS ("
    "SELECT 1 FROM sessions s WHERE s.world_id = w.id AND (s.last_active_at > now() - %(age)s "
    "OR (s.turn_running AND s.turn_heartbeat_at > now() - %(lease)s) OR s.id = ANY(%(keep)s)))"
)


async def purge_worlds(
    conn: AsyncConnection,
    older_than: timedelta,
    lease: timedelta = timedelta(seconds=45),
    *,
    keep: Collection[str] = (),
) -> int:
    """Delete worlds that are older than `older_than` and have not been used for as long.

    Age alone would delete a conversation in the middle of a turn: its events would then fail to
    store, and its saved state would be written back with no session to belong to. `keep` names
    sessions that are open in this process: one resumed after two quiet days has done nothing yet
    that marks it active, and it goes on to write into its world. Returns how many worlds went.
    """
    params = {"age": older_than, "lease": lease, "keep": list(keep)}
    async with conn.transaction():
        cur = await conn.execute(
            sql.SQL("SELECT w.id FROM worlds w WHERE {} FOR UPDATE OF w SKIP LOCKED").format(
                _UNUSED
            ),
            params,
        )
        gone = [row["id"] for row in await cur.fetchall()]
        if not gone:
            return 0
        await _forget_threads(
            conn,
            sql.SQL("thread_id IN (SELECT id FROM sessions WHERE world_id = ANY(%s))"),
            (gone,),
        )
        await conn.execute("DELETE FROM worlds WHERE id = ANY(%s)", (gone,))
        return len(gone)


async def forget_orphaned_threads(conn: AsyncConnection) -> int:
    """Delete saved conversations whose session no longer exists. Returns how many threads went.

    Every thread this app creates has a session row. A thread without one was left behind when its
    world was deleted by a version that did not clean up after itself.
    """
    if not await _has_thread_tables(conn):
        return 0
    orphaned = sql.SQL("thread_id NOT IN (SELECT id FROM sessions)")
    async with conn.transaction():
        cur = await conn.execute(
            sql.SQL("SELECT count(DISTINCT thread_id) AS n FROM checkpoints WHERE {}").format(
                orphaned
            )
        )
        count = (await cur.fetchone())["n"]
        await _forget_threads(conn, orphaned, ())
    return count
