"""Token prices, so every model call can report what it cost.

A price belongs to one configured model, not to a model name: the same model can be configured
twice, on two providers, at two prices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Cost of one call in dollars, or None when the token counts are unknown."""
        if input_tokens is None or output_tokens is None:
            return None
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000
