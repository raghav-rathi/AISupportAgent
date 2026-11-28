"""Tells the open desks of a workspace that one of its tickets is new or has changed.

In this process only, like the hubs. With several instances a desk would hear of a ticket its own
instance did not touch on its next load, not at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

# Changes a desk may be behind before the oldest are dropped. A dropped one costs a row that is
# stale until the desk is loaded again, and a desk that far behind is not being read.
_MAX_LAG = 500

Changed = tuple[str, str]  # the world of a ticket and its ID


class DeskBus:
    def __init__(self) -> None:
        self._desks: dict[str, set[asyncio.Queue[Changed]]] = {}

    def subscribe(self, workspace_id: str) -> tuple[asyncio.Queue[Changed], Callable[[], None]]:
        """A queue of the workspace's changes from now on, and the way to stop them."""
        queue: asyncio.Queue[Changed] = asyncio.Queue(maxsize=_MAX_LAG)
        desks = self._desks.setdefault(workspace_id, set())
        desks.add(queue)

        def stop() -> None:
            desks.discard(queue)
            if not desks and self._desks.get(workspace_id) is desks:
                del self._desks[workspace_id]

        return queue, stop

    def publish(self, workspace_id: str | None, world_id: str, ticket_id: str) -> None:
        for queue in self._desks.get(workspace_id or "", ()):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait((world_id, ticket_id))
