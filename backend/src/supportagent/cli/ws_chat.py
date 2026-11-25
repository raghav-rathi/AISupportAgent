"""Talk to a running backend over its WebSocket and watch every event.

Interactive:    uv run python -m supportagent.cli.ws_chat
Scripted:       uv run python -m supportagent.cli.ws_chat --say "first message" --say "second"
Resume:         uv run python -m supportagent.cli.ws_chat --session ID
Break things:   uv run python -m supportagent.cli.ws_chat --fault llm_rate_limit --say "hello"

This is the same path a browser takes. `supportagent.cli.chat` runs the agent in-process instead.
The default address is 127.0.0.1 on SUPPORTAGENT_PORT, or 8787 when that is not set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from websockets.asyncio.client import ClientConnection, connect

from supportagent.cli.chat import Printer
from supportagent.events import EVENT_ADAPTER

_TURN_ENDS = {"turn.finished", "turn.failed", "turn.interrupted", "error"}
_SKIP_HISTORY = 2**53  # as `last_seq`: resume a session without replaying what it already said


async def _turn(ws: ClientConnection, printer: Printer, message: dict[str, str]) -> None:
    await ws.send(json.dumps(message))
    while True:
        event = EVENT_ADAPTER.validate_json(await ws.recv())
        if event.type == "error":
            print(f"  error  {event.code}: {event.message}")
        else:
            printer(event)
        if event.type in _TURN_ENDS:
            return


async def main() -> int:
    port = os.environ.get("SUPPORTAGENT_PORT", "8787")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=f"ws://127.0.0.1:{port}/ws")
    parser.add_argument("--say", action="append", help="send this message, repeatable, then exit")
    parser.add_argument("--session", help="resume this session instead of starting a new one")
    parser.add_argument("--fault", action="append", default=[], help="inject before the first turn")
    parser.add_argument(
        "--retry", action="store_true", help="after --say, retry a failed turn once"
    )
    parser.add_argument("--verbose", action="store_true", help="show passages and model timings")
    args = parser.parse_args()

    async with connect(args.url) as ws:
        hello = {"type": "hello", "session_id": args.session, "last_seq": _SKIP_HISTORY}
        await ws.send(json.dumps(hello))
        ready = EVENT_ADAPTER.validate_json(await ws.recv())
        if ready.type != "session.ready":
            print(f"the server did not open a session: {ready}", file=sys.stderr)
            return 1
        who = f", verified {ready.customer.contact_name}" if ready.customer else ""
        print(f"session {ready.session_id}{who}")
        for kind in args.fault:
            await ws.send(json.dumps({"type": "fault.inject", "kind": kind}))

        printer = Printer(args.verbose)
        if args.say is not None:
            for text in args.say:
                print(f"\nyou> {text}")
                await _turn(ws, printer, {"type": "user.message", "text": text})
            if args.retry:
                print("\nretry>")
                await _turn(ws, printer, {"type": "turn.retry"})
            return 0
        while True:
            try:
                text = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except EOFError:
                return 0
            if text in {"", "/quit"}:
                return 0
            await _turn(ws, printer, {"type": "user.message", "text": text})


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
