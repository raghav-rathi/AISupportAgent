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


# The same speech on its way: what the recogniser had written a moment before each of the
# transcripts above was whole, and where it stopped when a pause ended the turn in mid-number.
# Each is what the page would show at that moment. None may show a digit of the PIN, and where
# an account number is shown it is all eight digits of it or none.
ON_THE_WAY = [
    # Asked for them: every short number is expected to be part of one.
    ("1 0 4", True, "****"),
    ("1 0 4 8 2 9 1 7", True, "10482917"),
    ("1 0 4 8 2 9 1 7 4", True, "10482917 ****"),
    ("1 0 4 8 2 9 1 7 4 7 2", True, "10482917 ****"),
    ("4", True, "****"),
    ("4 7", True, "****"),
    ("And the PIN is 2 4.", True, "And the PIN is ****."),
    ("5 8.", True, "****."),
    ("The dentist three", True, "The dentist ****"),
    # Not asked: the words say what the number is, or there is nothing but a number.
    ("My PIN is 4", False, "My PIN is ****"),
    ("the PIN is four", False, "the PIN is ****"),
    ("10482917 4", False, "10482917 ****"),
    ("9", False, "****"),
    ("nine zero three", False, "****"),
    # And what is no part of either stays readable.
    ("I have 2 invoices from March", False, "I have 2 invoices from March"),
    ("Oh. One moment.", True, "Oh. One moment."),
]


@pytest.mark.parametrize(("heard", "asked", "shown"), ON_THE_WAY)
def test_no_digit_of_a_pin_is_shown_while_it_is_being_said(heard, asked, shown):
    redactor = Redactor()  # nobody is verified
    redactor.spoken, redactor.expecting = True, asked
    assert redactor.text(tidy_digits(heard), customer=True) == shown


def test_typed_numbers_are_masked_as_before_whatever_voice_does():
    typed = Redactor()
    assert typed.text("2", customer=True) == "2"
    assert typed.text("nine zero", customer=True) == "nine zero"
    assert typed.text("my PIN is 4721", customer=True) == "my PIN is ****"


def test_the_second_half_of_a_pin_the_session_knows_is_masked_as_a_message_of_its_own():
    redactor = Redactor(strict=False)  # verified by now, and voice is off
    redactor.learn("2458")
    assert redactor.text("58.", customer=True) == "****."
    assert redactor.text("24", customer=True) == "****"
    assert redactor.text("58 seats", customer=True) == "58 seats"
    assert redactor.text("58.", customer=False) == "58."  # the app's own text is not a message


def test_digit_mode_comes_on_for_a_request_and_not_for_a_mention():
    for asking in (
        "Please provide your 8-digit account number and your PIN to verify your identity.",
        "I can help with that. What is your account number?",
        "That PIN does not match. Please try again.",
        "To verify you, I need your account number and PIN.",
    ):
        assert asks_for_credentials(asking), asking
    for mentioning in (
        "You can reset your PIN in settings.",
        "Your account number and PIN check out. How can I help?",
        "You are on the Pro plan. Is there anything else?",
    ):
        assert not asks_for_credentials(mentioning), mentioning


def test_digit_mode_comes_on_when_the_customer_starts_to_give_a_number():
    for giving in ("My account number is", "my PIN is 4", "1 0 4 8", "one zero four", "okay 1048"):
        assert gives_credentials(tidy_digits(giving)), giving
    for other in (
        "the account number on my invoice is wrong",
        "I forgot my PIN.",
        "I have 3 invoices and 2 seats",
        "one zero",
    ):
        assert not gives_credentials(tidy_digits(other)), other
