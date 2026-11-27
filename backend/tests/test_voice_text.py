"""Text on its way into speech and out of it, and PINs in what a recogniser really wrote."""

from __future__ import annotations

import pytest

from supportagent.redact import Redactor
from supportagent.voice.text import (
    asks_for_credentials,
    digit_count,
    gives_credentials,
    sounds_like_echo,
    speakable,
    tidy_digits,
    without_echo,
)

REPLY = "Refunds take five days. Is there anything else I can do for you?"

# What Deepgram Flux wrote on 2026-11-27 for synthetic callers who said an account number and a
# PIN, with the PIN that was said. The first three are whole turns in digit mode. The rest are
# the pieces the same speech falls into without it, as numerals and as words.
HEARD = [
    ("My account number is 1 0 4 8 2 9 1 7. My PIN is 4 7 2 1.", "4721"),
    ("2 0 5 7 3 8 4 6 9 0 3 4", "9034"),
    ("Sure, it is 5 3 9 2 0 1 6 4. And the PIN is 2 4. 5 8.", "2458"),
    ("My PIN is 4 7 2 1.", "4721"),
    ("9 0 3 4.", "9034"),
    ("My PIN is four seven two one.", "4721"),
    ("nine zero three four.", "9034"),
]


@pytest.mark.parametrize(("heard", "pin"), HEARD)
def test_a_pin_is_masked_in_what_the_recogniser_really_wrote(heard, pin):
    masked = Redactor().text(tidy_digits(heard), customer=True)  # a visitor, not yet verified
    assert pin not in masked.replace(" ", "")
    assert not any(word in masked for word in ("four seven", "nine zero"))
    assert "****" in masked


def test_an_account_number_read_out_digit_by_digit_comes_out_as_one_number():
    assert tidy_digits("It is 1 0 4 8 2 9 1 7.") == "It is 10482917."
    # Twelve digits in one breath are the account number and the PIN.
    assert tidy_digits("2 0 5 7 3 8 4 6 9 0 3 4") == "20573846 9034"
    assert Redactor().text(tidy_digits("2 0 5 7 3 8 4 6 9 0 3 4"), customer=True) == "20573846 ****"
    # Numbers that were not read out digit by digit are left alone.
    assert tidy_digits("I have 2 invoices from 2024, 96 dollars each") == (
        "I have 2 invoices from 2024, 96 dollars each"
    )
    assert digit_count("10482917 and then 4 7 2 1") == 12


def test_a_reply_is_made_fit_to_be_read_aloud():
    written = "**Pro** costs $12 per seat [1, 2].\n- Free is $0 [3]\n\nI will not repeat ****."
    assert speakable(written) == "Pro costs $12 per seat. Free is $0 I will not repeat your PIN."


def test_digit_mode_follows_what_is_asked_for_and_what_is_offered():
    assert asks_for_credentials("Please give me your 8-digit account number and your PIN.")
    assert not asks_for_credentials("You have 14 days to ask for a refund.")
    assert gives_credentials("Sure, my account number is")
    assert gives_credentials("okay the pin is")
    assert not gives_credentials("I pinned a card to the board")


def test_the_replys_own_words_are_an_echo_and_other_words_are_the_customer():
    assert sounds_like_echo("Refunds take five", REPLY) is True
    assert sounds_like_echo("refunds take five days is there anything", REPLY) is True
    assert sounds_like_echo("Wait", REPLY) is False
    assert sounds_like_echo("Wait, what about annual plans?", REPLY) is False
    # One or two words that the reply also has do not say who is speaking.
    assert sounds_like_echo("I", REPLY) is None
    assert sounds_like_echo("I can", REPLY) is None
    assert sounds_like_echo("I know", REPLY) is False
    # The reply's words in another order are not the reply.
    assert sounds_like_echo("days five take refunds", REPLY) is False


def test_what_the_customer_said_over_an_echo_is_what_is_left_after_the_replys_words():
    assert without_echo("Refunds take five wait, what about annual plans?", REPLY) == (
        "wait, what about annual plans?"
    )
    assert without_echo("Wait a moment", REPLY) == "Wait a moment"
