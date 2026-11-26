"""What a turn needs besides its state: connections, models, thresholds and per-session switches.

The context is passed to the graph on every turn and is never saved. Nodes and tools read it from
their runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from psycopg_pool import AsyncConnectionPool

from supportagent.agent.judge import Judge
from supportagent.events import Band, LlmFinished
from supportagent.kb.retrieval import Retriever
from supportagent.llm.client import ResilientLLM
from supportagent.llm.faults import FaultPlan
from supportagent.redact import Redactor


@dataclass(frozen=True)
class Gate:
    """Thresholds on the reranker's scores.

    Below `low` the knowledge base has no answer and the turn goes to a person. From `high` up the
    answer streams and is checked afterwards. In between, the answer waits for the grounding check.
    `gap` is how much better a passage written for another plan has to score than anything the
    customer may see before it changes what happens. The scale belongs to the reranker, so all
    three are calibrated per reranker.
    """

    low: float
    high: float
    gap: float = 3.0

    def band(self, score: float | None) -> Band:
        # Written so that a score that is not a number fails the comparison and escalates.
        if score is None or not score >= self.low:
            return "escalate"
        return "answer" if score >= self.high else "check"

    def decide(self, visible: float | None, other: float | None, *, verified: bool) -> Band:
        """The band for one search.

        `visible` is the best score among passages the customer may see, `other` the best among
        passages written for other plans. The other passage matters only when what the customer
        may see does not clearly answer and the other passage beats it by `gap`.

        When nothing the customer may see is relevant at all, any relevant passage for another
        plan decides: the visitor is asked to verify, the verified customer is told which plan
        covers the topic. `low` is set so that no off-topic question reaches it.

        When something visible is relevant but weak, the other passage has to beat it by `gap`.
        For a verified customer it must also be a confident match, because the reply then says
        their own plan does not cover a topic it partly covers.
        """
        band = self.band(visible)
        if band == "answer" or other is None:
            return band
        beats = band == "escalate" or other - visible >= self.gap
        if not beats:
            return band
        if verified:
            confident = other >= self.high or (band == "escalate" and other >= self.low)
            return "restricted" if confident else band
        if other >= self.high or (band == "escalate" and other >= self.low):
            return "verify"
        return band


class TransientToolError(Exception):
    """A tool failure worth retrying, such as a timeout."""


@dataclass
class ToolFaults:
    """Simulated tool failures for one session: tool name to the number of calls that will fail."""

    remaining: dict[str, int] = field(default_factory=dict)

    def check(self, name: str) -> None:
        if self.remaining.get(name, 0) > 0:
            self.remaining[name] -= 1
            raise TransientToolError(f"{name} timed out (simulated)")


@dataclass
class TurnUsage:
    """Running totals for one turn."""

    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    priced_usd: float = 0.0
    unpriced_calls: int = 0  # calls on a model that has no price configured

    def add(self, event: LlmFinished) -> None:
        self.llm_calls += 1
        self.input_tokens += event.input_tokens or 0
        self.output_tokens += event.output_tokens or 0
        if event.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.priced_usd += event.cost_usd

    @property
    def cost_usd(self) -> float | None:
        """The turn's cost, or None when part of it is unknown. A sum that leaves calls out would
        read as a real figure and be too low."""
        return None if self.unpriced_calls else self.priced_usd


@dataclass
class AgentContext:
    pool: AsyncConnectionPool
    retriever: Retriever
    llm: ResilientLLM
    judge: Judge | None
    gate: Gate
    world_id: str
    mode: Literal["text", "voice"] = "text"
    max_steps: int = 8
    tool_timeout_s: float = 8.0
    tool_attempts: int = 3
    tool_backoff_s: float = 0.2
    faults: FaultPlan = field(default_factory=FaultPlan)
    tool_faults: ToolFaults = field(default_factory=ToolFaults)
    usage: TurnUsage = field(default_factory=TurnUsage)
    redactor: Redactor = field(default_factory=Redactor)
