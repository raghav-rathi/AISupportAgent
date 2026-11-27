"""What a ticket's transcript excerpt carries."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from supportagent.agent.tickets import transcript_excerpt
from supportagent.redact import Redactor

MESSAGES = [
    HumanMessage("I'm Maya at Harrowgate Studio. When does my invoice renew?"),
    AIMessage("Your Pro plan renews on the 3rd."),
    HumanMessage("Account 20573846, PIN 0000."),
    AIMessage("That PIN does not match."),
]


def test_an_excerpt_carries_the_last_lines_of_the_whole_conversation():
    excerpt = transcript_excerpt(MESSAGES, Redactor())
    assert "Harrowgate" in excerpt
    assert excerpt.count("\n") == 3


def test_an_excerpt_for_a_ticket_not_about_the_customer_starts_at_the_current_turn():
    excerpt = transcript_excerpt(MESSAGES, Redactor(), current_turn_only=True)
    assert "Harrowgate" not in excerpt
    assert "Pro plan" not in excerpt
    assert excerpt.splitlines()[0].startswith("Customer: Account ")
    assert excerpt.splitlines()[-1] == "Assistant: That PIN does not match."
