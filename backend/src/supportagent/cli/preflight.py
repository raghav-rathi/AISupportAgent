"""Checks that this machine can run the backend. Exit code 1 on any failure.

Run with: uv run python -m supportagent.cli.preflight

It also prints the models as the settings loader put them together, which is the quickest way to
see a value that leaked in from the .env file or did not get cleared.
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlsplit

import httpx
import psycopg

from supportagent import db
from supportagent.config import ModelSettings, Settings, get_settings, same_model

HOSTS = {
    "OpenRouter": "https://openrouter.ai/api/v1/models",
    "Deepgram": "https://api.deepgram.com/v1/projects",
}


async def _check_db(dsn: str) -> tuple[bool, str]:
    try:
        async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=5) as conn:
            cur = await conn.execute(
                "SELECT current_setting('server_version'), "
                "(SELECT default_version FROM pg_available_extensions WHERE name = 'vector'), "
                "to_regclass('schema_migrations') IS NOT NULL"
            )
            version, vector, has_table = await cur.fetchone()
            done: set[str] = set()
            if has_table:
                cur = await conn.execute("SELECT name FROM schema_migrations")
                done = {row[0] for row in await cur.fetchall()}
    except psycopg.Error as exc:
        return False, f"Postgres: cannot connect ({str(exc).strip()})"
    if vector is None:
        return False, f"Postgres {version}: the pgvector extension is not available"
    pending = [name for name, _ in db.migration_files() if name not in done]
    return True, f"Postgres {version}, pgvector {vector}, {len(pending)} migrations pending"


async def _check_host(client: httpx.AsyncClient, name: str, url: str) -> tuple[bool, str]:
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        return False, f"{name}: unreachable ({type(exc).__name__})"
    # 401 and 403 still prove the host answers. Only a server error or silence fails.
    ms = response.elapsed.total_seconds() * 1000
    return response.status_code < 500, f"{name}: HTTP {response.status_code} in {ms:.0f} ms"


def _describe(settings: Settings, spec: ModelSettings) -> str:
    """One model as the app will call it. The key itself is never printed."""
    host = urlsplit(spec.base_url or settings.openrouter_base_url).netloc
    key = (
        "own key" if spec.api_key else "OpenRouter key" if settings.openrouter_api_key else "NO KEY"
    )
    priced = spec.input_price is not None and spec.output_price is not None
    price = f"${spec.input_price}/{spec.output_price} per M" if priced else "no price"
    options = [f"extra_body={spec.extra_body}"] if spec.extra_body else []
    options += ["strict tools"] if spec.strict_tools else []
    options += [] if spec.json_schema else ["no JSON schema"]
    return f"{spec.model} at {host}, {key}, {price}" + "".join(f", {o}" for o in options)


def model_report(settings: Settings) -> list[tuple[str, str]]:
    """Lines about the configured models, each with a level: ok, warn or FAIL."""
    lines: list[tuple[str, str]] = []
    slots = (("agent", settings.llm), ("fallback", settings.fallback), ("judge", settings.judge))
    for name, spec in slots:
        if spec is not None:
            lines.append(("ok", f"{name} model: {_describe(settings, spec)}"))
    if settings.llm is None:
        lines.append(("FAIL", "no agent model: set LLM__MODEL"))
    if settings.judge is None:
        level = "warn" if settings.allow_unchecked_answers else "FAIL"
        lines.append((level, "no grounding judge: answers would go out unchecked"))
    elif settings.llm and same_model(settings.judge, settings.llm, settings.openrouter_base_url):
        lines.append(("warn", "the judge is the agent's own model: fine for development only"))
    if settings.fallback is None:
        lines.append(("warn", "no fallback model: an outage of the agent model fails the turn"))
    if settings.fault_injection:
        lines.append(("warn", "fault injection is on: clients can simulate failures"))
    if not settings.allowed_origins:
        lines.append(("warn", "ALLOWED_ORIGINS is empty: no browser page can connect"))
    return lines


async def main() -> int:
    version = sys.version_info
    results = [(version >= (3, 12), f"Python {version.major}.{version.minor}.{version.micro}")]
    try:
        settings = get_settings()
    except Exception as exc:  # a missing or malformed setting
        print(f"FAIL settings: {exc}")
        return 1
    results.append(await _check_db(settings.database_url))
    async with httpx.AsyncClient(timeout=8) as client:
        results += await asyncio.gather(*(_check_host(client, n, u) for n, u in HOSTS.items()))
    for ok, message in results:
        print(("ok   " if ok else "FAIL ") + message)
    models = model_report(settings)
    for level, message in models:
        print(f"{level:4} {message}")
    keys = {
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "DEEPGRAM_API_KEY": settings.deepgram_api_key,
    }
    for name, value in keys.items():
        print(("ok   " if value else "warn ") + f"{name} {'set' if value else 'not set'}")
    failed = not all(ok for ok, _ in results) or any(level == "FAIL" for level, _ in models)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
