"""What a ticket's transcript excerpt carries, and which of the customer's words are the question."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from supportagent.agent.state import NO_LONGER_VERIFIED
from supportagent.agent.tickets import open_question, transcript_excerpt
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


def test_an_excerpt_starts_where_the_customer_verified_now_came_in():
    # Maya was verified, then Daniel verified in the same conversation. What the tools had
    # returned about Maya was taken out at that moment, and her lines are not Daniel's.
    messages = [
        HumanMessage("Account 10482917, PIN 4721. Which invoices do I have?"),
        ToolMessage(NO_LONGER_VERIFIED, name="verify_identity", tool_call_id="c1"),
        ToolMessage(NO_LONGER_VERIFIED, name="get_account", tool_call_id="c2"),
        AIMessage("Larkspur Design has INV-30417 for $96."),
        HumanMessage("Thanks."),
        AIMessage("You are welcome."),
        HumanMessage("Now account 20573846, PIN 9034. Refund INV-30388."),
        ToolMessage('{"verified": true}', name="verify_identity", tool_call_id="c3"),
        AIMessage("That invoice is above what I can refund."),
    ]
    excerpt = transcript_excerpt(messages, Redactor())
    assert "Larkspur" not in excerpt and "INV-30417" not in excerpt and "welcome" not in excerpt
    assert excerpt.splitlines()[0].startswith("Customer: Now account 20573846")
    assert excerpt.splitlines()[-1] == "Assistant: That invoice is above what I can refund."


QUESTION = "How fast do you respond when we report a P1 outage?"


@pytest.mark.parametrize(
    "reply",
    [
        "My account number is 53920164 and my PIN is 2458.",
        "53920164 2458",
        "Sure, it is 53920164. And the PIN is 2458.",
    ],
)
def test_a_reply_that_only_gives_credentials_leaves_the_question_before_it_open(reply):
    asked = [HumanMessage(QUESTION), AIMessage("Account number and PIN, please?")]
    assert open_question([*asked, HumanMessage(reply)], Redactor()) == QUESTION


def test_the_latest_message_is_the_question_when_it_asks_one():
    messages = [
        HumanMessage(QUESTION),
        HumanMessage("Account 53920164, PIN 2458, do you have a SOC 2 report we can get?"),
    ]
    assert open_question(messages, Redactor()) == "Do you have a SOC 2 report we can get?"
    assert open_question([HumanMessage("Thanks.")], Redactor()) == "Thanks."
    # Nobody asked anything: there is nothing to search for in the customer's words.
    assert open_question([HumanMessage("53920164 and PIN 2458")], Redactor()) == ""
