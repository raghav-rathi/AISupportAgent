"""The redactor: what it masks, what it leaves alone, and how it handles events and messages."""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from supportagent.events import (
    LlmDelta,
    LlmFinished,
    LlmReset,
    LlmTrace,
    RetrievalResult,
    ToolStarted,
    TraceMessage,
    TurnStarted,
)
from supportagent.redact import (
    MASK,
    Redactor,
    digits_of,
    scrub_messages,
    session_key,
    spoken_digits,
)

# How people give a PIN in chat and in a voice transcript. An unverified session masks all of them.
PHRASINGS = [
    "Account 10482917, my PIN is 4721. Please refund my last invoice.",
    "10482917, 4721",
    "4721",
    "4 7 2 1",
    "4-7-2-1",
    "47-21",
    "PIN 47 21",
    "four seven two one",
    "forty-seven twenty-one",
    "my PIN's 4721",
    "PIN #4721",
    "PIN, 4721",
    "PIN of 4721",
    "4721 (my PIN)",
    "my passcode is 4721",
    "PIN: **4721**",
    "it's 10482917 and the code is 4721",
    "10482917 4721",
]


def shows_pin(text: str) -> bool:
    return "4721" in digits_of(text.replace("10482917", ""))


@pytest.mark.parametrize("text", PHRASINGS)
def test_an_unverified_customers_pin_is_masked_however_it_is_written(text):
    masked = Redactor().text(text, customer=True)
    assert not shows_pin(masked), masked
    assert ("10482917" in masked) == ("10482917" in text)  # the account number is not a secret


@pytest.mark.parametrize(
    "text",
    [
        "my PIN is 4721",
        "PIN: 4721",
        "the pin number is 4 7 2 1",
        "4721 is my PIN",
        '{"account_number": "10482917", "pin": "4721"}',
        "my pin is four seven two one",
    ],
)
def test_a_pin_next_to_the_word_is_masked_in_any_text(text):
    assert not shows_pin(Redactor(strict=False).text(text))


@pytest.mark.parametrize(
    "text",
    [
        "Please refund INV-30417, it was $1,440.00 on 2026-11-19.",
        "We have 1200 seats and pay 4800 dollars a year since March 2024.",
        "Call me on +1 415 555 0134 after 14:30.",
        "Ask the customer for their 8-digit account number and their PIN.",
        "Support can refund up to $2,000 per invoice within 30 days.",
        "I have one question about the two plans.",
    ],
)
def test_a_verified_customers_other_numbers_are_left_alone(text):
    assert Redactor(strict=False).text(text, customer=True) == text


@pytest.mark.parametrize(
    "text",
    [
        "Please refund INV-30417, it was $1,440.00 on 2026-11-19.",
        "My account is 10482917.",
        "What does Pro cost for 25 seats, and is the 99.9% uptime included?",
        "We signed up in 2024.",
    ],
)
def test_strict_mode_keeps_ids_amounts_dates_and_account_numbers(text):
    assert Redactor().text(text, customer=True) == text


def test_a_learned_pin_is_masked_everywhere_and_in_every_form():
    redactor = Redactor(strict=False)
    redactor.learn("4-7-2-1")
    for text in (
        "ok 4721",
        "4 7 2 1",
        "47-21",
        "it is four seven two one",
        "forty seven twenty-one",
    ):
        assert not shows_pin(redactor.text(text)), text
    # A longer number that merely contains the digits is something else.
    assert redactor.text("account 10472188") == "account 10472188"


def test_spoken_numbers_become_digits():
    assert spoken_digits("four seven two one") == "4721"
    assert spoken_digits("forty-seven twenty-one") == "4721"
    assert spoken_digits("four thousand seven hundred and twenty-one") == "4721"
    assert spoken_digits("double seven oh one") == "7701"
    assert digits_of("four seven two one") == "4721"
    assert digits_of("47-21") == "4721"


def test_nul_bytes_are_dropped():
    assert Redactor(strict=False).text("a\x00b") == "ab"


def test_a_tool_call_teaches_the_pin_before_the_rest_of_the_trace_is_masked():
    redactor = Redactor(strict=False)
    trace = LlmTrace(
        call_id="c1",
        purpose="agent",
        model="m",
        prompt=[
            TraceMessage(role="system", content="Ask for the account number and the PIN."),
            TraceMessage(role="user", content="sure: 10482917 then 4721"),
        ],
        tools=["verify_identity"],
        response=TraceMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_4721_x",
                    "name": "verify_identity",
                    "args": {"account_number": "10482917", "pin": "4721"},
                }
            ],
        ),
    )
    cleaned = redactor.event(trace)
    assert cleaned.prompt[1].content == f"sure: 10482917 then {MASK}"
    call = cleaned.response.tool_calls[0]
    assert call["args"] == {"account_number": "10482917", "pin": MASK}
    assert call["id"] == "call_4721_x"  # identifiers are never rewritten


def test_events_mask_the_customers_text_and_tool_arguments():
    redactor = Redactor()
    started = redactor.event(TurnStarted(mode="text", text="10482917 4721"))
    assert started.text == f"10482917 {MASK}"
    args = {"account_number": "10482917", "pin": "9034", "note": "said the PIN was 1586"}
    tool = redactor.event(ToolStarted(tool_call_id="t1", name="verify_identity", args=args))
    assert tool.args == {
        "account_number": "10482917",
        "pin": MASK,
        "note": f"said the PIN was {MASK}",
    }
    search = RetrievalResult(
        tool_call_id="t2",
        query="refund 4721",
        band="escalate",
        top_score=-9.0,
        plan_score=None,
        low=-4.0,
        high=2.0,
        hits=[],
        embed_ms=1.0,
        search_ms=1.0,
        rerank_ms=1.0,
    )
    assert redactor.event(search).query == f"refund {MASK}"


def test_scrubbing_replaces_only_the_messages_that_show_the_pin():
    redactor = Redactor(strict=False)
    redactor.learn("4721")
    question = HumanMessage("I was charged 1440 dollars twice.", id="h1")
    credentials = HumanMessage("Account 10482917 and it's 4721", id="h2")
    call = AIMessage(
        "",
        id="a1",
        tool_calls=[
            {
                "name": "verify_identity",
                "args": {"account_number": "10482917", "pin": "4721"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    replaced = scrub_messages([question, credentials, call], redactor)
    assert [m.id for m in replaced] == ["h2", "a1"]
    assert replaced[0].content == f"Account 10482917 and it's {MASK}"
    assert replaced[1].tool_calls[0]["args"] == {"account_number": "10482917", "pin": MASK}


# Forms an outside audit got past the redactor with. The slash, colon, asterisk and dash forms
# verify as 4721, so they have to be masked as 4721. The last three are how a speech-to-text
# service writes a PIN that was read out.
ODD_FORMS = [
    "4/7/2/1",
    "47/21",
    "47:21",
    "4\u20137\u20132\u20131",
    "4*7*2*1",
    "PIN: 4  7  2  1",
    "Four. Seven. Two. One.",
    "4. 7. 2. 1.",
    "four, seven, two, one",
]


@pytest.mark.parametrize("text", ODD_FORMS)
def test_a_learned_pin_is_masked_with_any_short_separator(text):
    redactor = Redactor(strict=False)
    redactor.learn("4721")
    assert not shows_pin(redactor.text(f"it is {text} ok", customer=True)), text


@pytest.mark.parametrize(
    "text",
    [
        "my pin is 47/21",
        "PIN 4/7/2/1",
        "pin: 47:21",
        "PIN: 4  7  2  1",
        "my PIN is Four. Seven. Two. One.",
        "the pin is nine o three four",
        "My PIN is 12345678",
    ],
)
def test_the_word_pin_masks_the_number_in_these_forms_too(text):
    masked = Redactor(strict=False).text(text, customer=True)
    assert not re.search(r"\d", masked) and "four" not in masked.lower(), masked


def test_a_pin_said_partly_in_numerals_and_partly_in_words_is_masked():
    fresh = Redactor(strict=False)
    assert fresh.text("My PIN is 47 twenty-one", customer=True) == f"My PIN is {MASK}"
    assert fresh.text("PIN: 4 seven 2 one", customer=True) == f"PIN: {MASK}"
    assert fresh.text("47 twenty-one is my PIN", customer=True) == f"{MASK} is my PIN"
    learned = Redactor(strict=False)
    learned.learn("4721")
    assert learned.text("it was 4 seven 2 one", customer=True) == f"it was {MASK}"
    assert learned.text("Your code is 47 twenty-one.") == f"Your code is {MASK}."
    # Numerals next to number words in an ordinary sentence are not a PIN, in the customer's
    # words or in the app's.
    for text in ("We were charged 100 one time.", "I have 3 two-year plans and 1 one-year plan."):
        assert learned.text(text, customer=True) == text
        assert Redactor().text(text, customer=True) == text
    written = "Refunds of $47, twenty-one in all."
    assert learned.text(written) == written


@pytest.mark.parametrize(
    "text",
    [
        "Account-PIN: 10482917-4721",
        "10482917/4721",
        "PIN: 10482917:4721",
        "10482917, 4721 \u2014 what plan am I on?",
        "Switch to 53920164, 2458",
    ],
)
def test_a_number_right_after_an_account_number_is_a_pin_even_once_verified(text):
    masked = Redactor(strict=False).text(text, customer=True)
    assert "4721" not in masked and "2458" not in masked, masked


def test_spoken_o_counts_as_zero():
    assert spoken_digits("nine o three four") == "9034"
    assert digits_of("Four. Seven. Two. One.") == "4721"
    assert digits_of(4721.0) == "4721"  # not "47210"


@pytest.mark.parametrize(
    "text",
    [
        "We have 1500 seats",
        "We got HTTP error 5003 on port 8080",
        "Our zip code is 94107",
        "Please refund Invoice 30417",
        "My account is 104 829 17",
        "Q1 2026 invoice",
        "The renewal is on 11/26 at 10:30",
    ],
)
def test_strict_mode_leaves_numbers_that_say_what_they_are(text):
    assert Redactor().text(text, customer=True) == text


def test_a_learned_pin_does_not_garble_amounts_ids_and_dates_the_app_wrote():
    redactor = Redactor(strict=False)
    for value in ("4721", "1440", "2026", "30417"):
        redactor.learn(value)
    written = "Invoice INV-30417 is $1,440.00, refunded on 2026-11-26. Your charge of $47.21 went through."
    assert redactor.text(written) == written
    # The same digits in the customer's own words are still a PIN.
    assert not shows_pin(redactor.text("ok it's 47.21", customer=True))
    assert redactor.text("Your code 4721 is on file.") == f"Your code {MASK} is on file."


def test_half_of_a_learned_pin_in_words_is_masked_as_a_message_of_its_own():
    redactor = Redactor(strict=False)  # verified by now
    redactor.learn("2458")
    assert redactor.text("five eight.", customer=True) == f"{MASK}."
    assert redactor.text("Twenty-four", customer=True) == MASK
    # One word is an answer more often than half a PIN, and words in a sentence are not a half.
    for text in ("Two.", "Fifty-eight seats"):
        assert redactor.text(text, customer=True) == text
    assert redactor.text("five eight.") == "five eight."  # the app's own text is not a message


def test_a_pin_under_another_argument_name_or_as_a_number_is_masked():
    redactor = Redactor(strict=False)
    as_number = redactor.event(
        ToolStarted(
            tool_call_id="t1",
            name="verify_identity",
            args={"account_number": "10482917", "pin": 4721.0},
        )
    )
    assert as_number.args["pin"] == MASK
    assert redactor.text("it was 4721") == f"it was {MASK}"  # and it was learned as 4721


def test_a_pin_given_to_verify_identity_under_another_name_is_masked_in_its_trace():
    redactor = Redactor(strict=False)
    trace = LlmTrace(
        call_id="c1",
        purpose="agent",
        model="m",
        prompt=[TraceMessage(role="user", content="Switch to 53920164 and 2458")],
        tools=["verify_identity"],
        response=TraceMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "t1",
                    "name": "verify_identity",
                    "args": {"account_number": "53920164", "code": "2458"},
                }
            ],
        ),
    )
    cleaned = redactor.event(trace)
    assert cleaned.prompt[0].content == f"Switch to 53920164 and {MASK}"
    assert cleaned.response.tool_calls[0]["args"] == {"account_number": "53920164", "code": MASK}
    # Another tool's arguments teach it nothing, and the account number is never the PIN.
    search = ToolStarted(tool_call_id="t2", name="search_knowledge_base", args={"query": "3390"})
    assert redactor.event(search) == search
    assert redactor.text("it was 3390") == "it was 3390"
    assert redactor.text("account 53920164") == "account 53920164"


def test_a_new_process_recognises_the_pins_an_earlier_one_hashed():
    key = session_key("server secret", "session-1")
    first = Redactor(strict=False, key=key)
    first.learn("4721")
    first.learn("9034")
    stored = first.hashes
    assert len(stored) == 2 and not any("4721" in h for h in stored)

    later = Redactor(strict=False, key=key, known=stored)
    assert later.text("Did 4721 work earlier?", customer=True) == f"Did {MASK} work earlier?"
    assert later.text("Your code 4 7 2 1 is on file.") == f"Your code {MASK} is on file."
    assert later.text("was it nine o three four", customer=True) == f"was it {MASK}"
    assert later.text("I have 1234 cards", customer=True) == "I have 1234 cards"
    # Another session's key, or another server's secret, does not recognise them.
    for other in (session_key("server secret", "session-2"), session_key("another", "session-1")):
        assert Redactor(strict=False, key=other, known=stored).text("4721") == "4721"


def test_a_new_process_masks_a_pin_wherever_the_first_one_would_have():
    key = session_key("server secret", "session-1")
    first = Redactor(strict=False, key=key)
    first.learn("4721")
    for text in (
        "Use 4|7|2|1 please",
        "Use 47--21 please",
        "Use 4(7)2(1) please",
        "Step 1: 47-21",
        "It was one zero four eight two nine one seven four seven two one",
    ):
        later = Redactor(strict=False, key=key, known=first.hashes)  # it has only the hash
        assert not shows_pin(later.text(text, customer=True)), text
    later = Redactor(strict=False, key=key, known=first.hashes)
    for text in ("Step 1: open Settings, step 2: pick 47-22.", "one zero four eight two nine one"):
        assert later.text(text, customer=True) == text


def texts(events):
    return "".join(e.text for e in events if isinstance(e, LlmDelta))


def done(call_id="c1"):
    return LlmFinished(
        call_id=call_id, purpose="agent", model="m", attempts=1, used_fallback=False,
        ttft_ms=1.0, total_ms=2.0, input_tokens=1, output_tokens=1, cost_usd=None,
    )  # fmt: skip


def test_a_pin_split_over_two_streamed_chunks_is_masked():
    redactor = Redactor(strict=False)
    redactor.learn("4721")
    out = []
    for piece in ("Confirmed, code 47", "21 worked", ". Your PIN is 9", "03", "4."):
        out += redactor.outgoing(LlmDelta(call_id="c1", text=piece))
    out += redactor.outgoing(done())
    assert texts(out) == f"Confirmed, code {MASK} worked. Your PIN is {MASK}."
    assert out[-1].type == "llm.finished"


def test_a_spoken_pin_split_over_streamed_chunks_is_masked():
    redactor = Redactor(strict=False)
    redactor.learn("4721")
    out = []
    for piece in ("Your code is ", "four ", "seven ", "two ", "one."):
        out += redactor.outgoing(LlmDelta(call_id="c1", text=piece))
    out += redactor.outgoing(done())
    assert texts(out) == f"Your code is {MASK}."
    out = []
    for piece in ("It is four se", "ven two o", "ne. Thanks"):
        out += redactor.outgoing(LlmDelta(call_id="c2", text=piece))
    assert texts(out) == f"It is {MASK}. Thanks"
    # Number words in an ordinary answer wait only for the chunk that shows where they end.
    out = []
    for piece in ("I have one", " question. Two", " plans fit."):
        out += redactor.outgoing(LlmDelta(call_id="c3", text=piece))
    assert texts(out) == "I have one question. Two plans fit."


def test_streamed_text_arrives_whole_and_in_order_when_nothing_is_secret():
    redactor = Redactor(strict=False)
    answer = (
        "Pro costs $12 per seat per month, or $120 per seat per year [1]. Refund window in days: 14"
    )
    out = []
    for start in range(0, len(answer), 7):
        out += redactor.outgoing(LlmDelta(call_id="c1", text=answer[start : start + 7]))
    assert texts(out) == answer[:-2]  # a number at the very end waits for what follows it
    out += redactor.outgoing(done())
    assert texts(out) == answer


def test_a_reset_drops_what_a_stream_was_holding_back():
    redactor = Redactor(strict=False)
    out = redactor.outgoing(LlmDelta(call_id="c1", text="Order 12"))
    out += redactor.outgoing(LlmReset(call_id="c1", reason="tool_call"))
    out += redactor.outgoing(LlmDelta(call_id="c1", text="Hello"))
    assert texts(out) == "Order Hello"


def test_scrubbing_reaches_block_content_and_malformed_tool_calls():
    redactor = Redactor(strict=False)
    redactor.learn("4721")
    blocks = HumanMessage(
        [{"type": "text", "text": "it's 4721"}, {"type": "image_url", "image_url": "x"}], id="h1"
    )
    broken = AIMessage(
        "",
        id="a1",
        invalid_tool_calls=[
            {
                "name": "verify_identity",
                "args": '{"account_number": "10482917", "pin": "4721"',
                "id": "c1",
                "error": "parse",
                "type": "invalid_tool_call",
            }
        ],
    )
    replaced = scrub_messages([blocks, broken], redactor)
    assert [m.id for m in replaced] == ["h1", "a1"]
    assert (
        replaced[0].content[0]["text"] == f"it's {MASK}"
        and replaced[0].content[1]["type"] == "image_url"
    )
    assert "4721" not in replaced[1].invalid_tool_calls[0]["args"]
