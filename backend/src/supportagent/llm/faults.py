"""Faults a tester can queue for the next model calls.

A fault raises inside the same code path as a real failure, so the retry, failover and UI states it
produces are the real ones. A fault names the kind of call it hits, so a fault meant for the agent
never lands on the grounding judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from supportagent.events import Purpose
from supportagent.llm.errors import Failure

Role = Literal["primary", "fallback"]
MAX_PENDING = 20  # a session cannot queue more faults than this


@dataclass(frozen=True)
class Fault:
    failure: Failure
    target: Role | Literal["any"] = "primary"
    after_chunks: int = 0  # 0 fails before the first token, n fails after n chunks went out
    delay_s: float = 0.0  # wait this long before failing, to look like a slow upstream
    purpose: Purpose = "agent"  # which calls it hits


def timeout(target: Role | Literal["any"] = "primary", delay_s: float = 0.0) -> Fault:
    return Fault(Failure("timeout", retryable=True), target, delay_s=delay_s)


def rate_limit(retry_after_s: float = 1.0, target: Role | Literal["any"] = "primary") -> Fault:
    return Fault(Failure("rate_limited", True, retry_after_s, "HTTP 429"), target)


def server_error(target: Role | Literal["any"] = "primary", after_chunks: int = 0) -> Fault:
    return Fault(Failure("server_error", True, detail="HTTP 500"), target, after_chunks)


def quota(target: Role | Literal["any"] = "primary", purpose: Purpose = "agent") -> Fault:
    return Fault(Failure("quota", retryable=False, detail="HTTP 429"), target, purpose=purpose)


@dataclass
class FaultPlan:
    """The faults queued for one session. Each model attempt takes at most one."""

    _queue: list[Fault] = field(default_factory=list)

    def add(self, fault: Fault, times: int = 1) -> int:
        """Queue the fault up to `times` times. Returns how many fit under the cap."""
        accepted = max(0, min(times, MAX_PENDING - len(self._queue)))
        self._queue.extend([fault] * accepted)
        return accepted

    def take(self, role: Role, purpose: Purpose = "agent") -> Fault | None:
        for index, fault in enumerate(self._queue):
            if fault.purpose == purpose and fault.target in (role, "any"):
                return self._queue.pop(index)
        return None

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)
