"""Talk to the agent from a terminal and watch every event.

Interactive:    uv run python -m supportagent.cli.chat
Scripted:       uv run python -m supportagent.cli.chat --say "first message" --say "second"
Resume:         uv run python -m supportagent.cli.chat --session ID

It runs the agent in this process. Each new session gets its own copy of the mock CRM.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from pydantic import BaseModel

from supportagent.config import get_settings
from supportagent.runtime import NotConfigured, Runtime, open_runtime
from supportagent.session import ensure_world, run_turn

_MARK = {"ok": "ok", "refused": "refused", "failed": "FAILED"}


class Printer:
    """Renders the event stream as lines of text."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self._mid_line = False
        self._streamed = False  # the current answer went out as deltas

    def _line(self, text: str) -> None:
        if self._mid_line:
            print()
            self._mid_line = False
        print(text)

    def __call__(self, event: BaseModel) -> None:
        e, kind = event, event.type
        if kind == "turn.started":
            self._streamed = False
        elif kind == "llm.delta":
            if not self._mid_line:
                print("  agent> ", end="")
            print(e.text, end="", flush=True)
            self._mid_line = self._streamed = True
        elif kind == "tool.started":
            self._line(f"  tool   {e.name} {e.args}")
        elif kind == "tool.error":
            self._line(f"  tool   {e.name} attempt failed: {e.message}")
        elif kind == "tool.retry":
            self._line(f"  tool   {e.name} retry {e.attempt} of {e.max_attempts}")
        elif kind == "tool.finished":
            self._line(
                f"  tool   {e.name} {_MARK[e.outcome]}: {e.summary} ({e.duration_ms:.0f} ms)"
            )
        elif kind == "retrieval.result":
            top = "none" if e.top_score is None else f"{e.top_score:.2f}"
            other = "" if e.plan_score is None else f", other plans {e.plan_score:.2f}"
            self._line(
                f"  gate   {e.band}: top score {top}{other}, thresholds {e.low} and {e.high}, "
                f"rerank {e.rerank_ms:.0f} ms"
            )
            if self.verbose:
                for hit in e.hits:
                    self._line(f"         [{hit.n}] {hit.score:6.2f} {hit.source}")
        elif kind == "llm.retry":
            self._line(
                f"  llm    {e.reason}: retry {e.attempt} of {e.max_attempts} in {e.wait_ms} ms"
            )
        elif kind == "llm.fallback":
            self._line(f"  llm    switched from {e.from_model} to {e.to_model} ({e.reason})")
        elif kind == "llm.reset":
            self._line(f"  llm    partial answer dropped ({e.reason})")
            self._streamed = False
        elif kind == "llm.finished" and self.verbose:
            ttft = "n/a" if e.ttft_ms is None else f"{e.ttft_ms:.0f}"
            self._line(
                f"  llm    {e.purpose} on {e.model}: first token {ttft} ms, total "
                f"{e.total_ms:.0f} ms, tokens {e.input_tokens} in and {e.output_tokens} out"
            )
        elif kind == "guardrail.verdict":
            detail = f": {'; '.join(e.unsupported)}" if e.unsupported else ""
            self._line(f"  check  {e.verdict}{' (answer was held)' if e.held else ''}{detail}")
        elif kind == "answer.retracted":
            self._line(f"  check  answer retracted: {e.reason}")
            self._streamed = False
        elif kind == "assistant.message" and not self._streamed:
            # Nothing of this answer has been printed: it was held for its check, or this is a
            # replay of stored events, which hold no deltas.
            self._line(f"  agent> {e.text}")
        elif kind == "escalation.created":
            t = e.ticket
            self._line(f"  ticket {t.id}: {t.reason}, {t.priority} priority. {t.summary}")
        elif kind == "session.state":
            c = e.customer
            who = f"verified {c.contact_name}, {c.company_name}, {c.tier}, {c.status}" if c else ""
            self._line(f"  state  {who or 'not verified'}")
        elif kind == "turn.failed":
            self._line(f"  turn   failed: {e.message}")
        elif kind == "turn.interrupted":
            self._line(f"  turn   interrupted after {e.total_ms:.0f} ms")
        elif kind == "turn.finished":
            cost = "cost unknown" if e.cost_usd is None else f"${e.cost_usd:.4f}"
            self._line(
                f"  turn   {e.outcome} in {e.total_ms:.0f} ms: {e.llm_calls} model calls, "
                f"{e.tool_calls} tool calls, {e.input_tokens} tokens in, {e.output_tokens} out, "
                f"{cost}"
            )


async def _converse(rt: Runtime, session_id: str, lines: list[str] | None, verbose: bool) -> None:
    world_id = f"cli-{session_id}"
    await ensure_world(rt, world_id)
    printer = Printer(verbose)
    print(f"session {session_id}")
    scripted = iter(lines) if lines is not None else None
    while True:
        if scripted is not None:
            text = next(scripted, None)
            if text is None:
                return
            print(f"\nyou> {text}")
        else:
            try:
                text = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except EOFError:
                return
            if text in {"", "/quit"}:
                return
        done = await run_turn(
            rt, session_id=session_id, world_id=world_id, text=text, on_event=printer
        )
        while not done and scripted is None:
            answer = await asyncio.to_thread(input, "  retry the turn? [y/N] ")
            if answer.strip().lower() != "y":
                break
            done = await run_turn(
                rt, session_id=session_id, world_id=world_id, text=None, on_event=printer
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--say", action="append", help="send this message, repeatable, then exit")
    parser.add_argument("--session", help="resume this session instead of starting a new one")
    parser.add_argument("--verbose", action="store_true", help="show passages and model timings")
    args = parser.parse_args()
    try:
        async with open_runtime(get_settings()) as rt:
            await _converse(rt, args.session or uuid.uuid4().hex, args.say, args.verbose)
    except NotConfigured as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
