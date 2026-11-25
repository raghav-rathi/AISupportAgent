"""Keeps PINs out of everything the backend stores or sends.

A PIN may sit in one place only: the conversation the model reads, and only until verify_identity
has been called with it. Events, traces, tickets and tool-argument echoes all pass through a
Redactor first.

A Redactor masks three things:

- Every value the session has passed to verify_identity as a PIN, whether it shows up as digits,
  with spaces or hyphens between them, or spelled out in words. This rule is exact.
- A number next to a word such as "PIN" or "passcode".
- In strict mode, any number in the customer's own text that could be a PIN: 4 to 8 digits,
  however they are separated or spelled, unless it is an 8-digit account number or reads as an
  amount, a date or an ID. Strict mode is on while the session has no verified customer, which is
  when a PIN is expected and not yet known.

The second and third rules are guesses and can miss an odd phrasing. The first cannot, and it holds
from the moment the model asks for verification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel

from supportagent.events import LlmDelta

MASK = "****"

_UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}
_REPEATS = {"double": 2, "triple": 3}
_WORD = "|".join(sorted([*_UNITS, *_TEENS, *_TENS, *_SCALES, *_REPEATS], key=len, reverse=True))
# "four seven two one", "forty-seven twenty-one", "four thousand seven hundred and twenty-one".
_SPOKEN = re.compile(
    rf"\b(?:{_WORD})\b(?:(?:\s*[-,]\s*|\s+)(?:and\s+)?(?:{_WORD})\b)*", re.IGNORECASE
)
_DIGITS = re.compile(r"\d+")
# What may sit between the short groups of one number: "4 7 2 1", "47-21", "4.7.2.1", "4, 7, 2, 1".
_JOINTS = frozenset({" ", "-", ".", " - ", ", "})
_CUE = re.compile(
    r"\b(?:pins?|p\.i\.n\.?|passcode|pass\s?code|password|passphrase"
    r"|(?:security|access|secret|verification|support)\s+code)\b",
    re.IGNORECASE,
)
_YEAR_LEAD = re.compile(
    r"\b(?:in|since|of|year|during|until|by|from|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct"
    r"|nov|dec|january|february|march|april|june|july|august|september|october|november"
    r"|december)\.?,?\s$",
    re.IGNORECASE,
)
# Keys whose values identify something and are never free text.
_KEPT_KEYS = frozenset(
    {"type", "id", "model", "from_model", "to_model", "source", "name", "account_number", "role"}
)
# Fields that hold the customer's own words, or a search query built from them.
_CUSTOMER_KEYS = frozenset({"query"})
_CUSTOMER_FIELDS = {"turn.started": frozenset({"text"})}


def spoken_digits(run: str) -> str:
    """The digits a run of number words stands for: "forty-seven twenty-one" gives "4721"."""
    words = [w for w in re.split(r"[\s,-]+", run.lower()) if w and w != "and"]
    if any(w in _SCALES for w in words):
        total = current = 0
        for word in words:
            if word == "hundred":
                current = max(current, 1) * 100
            elif word == "thousand":
                total += max(current, 1) * 1000
                current = 0
            elif word not in _REPEATS:
                current += _UNITS.get(word, _TEENS.get(word, _TENS.get(word, 0)))
        return str(total + current)
    pieces: list[str] = []
    repeat = 1
    index = 0
    while index < len(words):
        word = words[index]
        index += 1
        if word in _REPEATS:
            repeat = _REPEATS[word]
            continue
        if word in _TENS:
            value = _TENS[word]
            if index < len(words) and _UNITS.get(words[index], 0) > 0:
                value += _UNITS[words[index]]
                index += 1
        else:
            value = _TEENS.get(word, _UNITS.get(word, 0))
        pieces.append(str(value) * repeat)
        repeat = 1
    return "".join(pieces)


def digits_of(value: object) -> str:
    """The digits in a value, written as numerals or spelled out."""
    text = str(value)
    digits = re.sub(r"\D", "", text)
    return digits or "".join(spoken_digits(m.group()) for m in _SPOKEN.finditer(text))


def _numbers(text: str) -> list[tuple[int, int, str]]:
    """Each number in the text as (start, end, digits).

    Short groups joined by a space, a hyphen or a period count as one number, which is how people
    type a PIN with separators: "4 7 2 1", "47-21".
    """
    groups = [m.span() for m in _DIGITS.finditer(text)]
    found = []
    index = 0
    while index < len(groups):
        start, end = groups[index]
        last = index
        if end - start <= 3:
            while (
                last + 1 < len(groups)
                and groups[last + 1][1] - groups[last + 1][0] <= 3
                and text[groups[last][1] : groups[last + 1][0]] in _JOINTS
            ):
                last += 1
        end = groups[last][1]
        found.append((start, end, re.sub(r"\D", "", text[start:end])))
        index = last + 1
    return found


def _near_cue(text: str, start: int, end: int) -> bool:
    """True when a word such as "PIN" stands shortly before or after text[start:end]."""
    before = text[max(0, start - 40) : start]
    for cue in _CUE.finditer(before):
        if _joins(before[cue.end() :], 24):
            return True
    after = text[end : end + 32]
    cue = _CUE.search(after)
    return cue is not None and _joins(after[: cue.start()], 16)


def _joins(gap: str, limit: int) -> bool:
    """True when the text between a cue word and a number keeps them in one phrase."""
    return len(gap) <= limit and not any(ch.isdigit() or ch in ".!?;\n" for ch in gap)


def _reads_as_something_else(text: str, start: int, end: int, digits: str) -> bool:
    """True when the number at text[start:end] is clearly not a PIN: an ID, an amount, a date."""
    before, after = text[:start], text[end:]
    if len(digits) == 8 and end - start == 8:
        return True  # an account number
    if before[-1:].isalpha() or before[-1:] == "_" or after[:1].isalpha() or after[:1] == "_":
        return True  # part of a longer token
    if re.search(r"[A-Za-z]{2,}-$", before):
        return True  # INV-30417, RF-7214CA07
    if re.search(r"[$€£]\s?$", before) or after[:1] == "%":
        return True
    if re.match(r"[.,]\d", after) or re.search(r"\d[.,]$", before):
        return True  # 1440.50 or 1,440
    if re.match(r"[-/:]\d", after) or re.search(r"\d[-/:]$", before):
        return True  # a date, a time, a range, part of a phone number
    return len(digits) == 4 and digits[:2] in ("19", "20") and bool(_YEAR_LEAD.search(before))


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    position = 0
    for start, end in sorted(spans):
        if start < position:  # overlaps the span before it
            position = max(position, end)
            continue
        pieces += [text[position:start], MASK]
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


class Redactor:
    """Masks PINs for one session. `learn` teaches it a PIN, `strict` turns the third rule on."""

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self._secrets: dict[str, re.Pattern[str]] = {}

    def learn(self, value: object) -> None:
        """Remember a value that was given as a PIN, right or wrong."""
        digits = digits_of(value)
        if len(digits) >= 3 and digits not in self._secrets:
            joined = r"[\s\-.,_]{0,3}".join(digits)
            self._secrets[digits] = re.compile(rf"(?<!\d){joined}(?!\d)")

    def text(self, text: str, *, customer: bool = False) -> str:
        """`text` with every PIN masked. `customer` marks the customer's own words."""
        if not text:
            return text
        text = text.replace("\x00", "")  # Postgres cannot store it, and nothing needs it
        strict = customer and self.strict
        spans = [m.span() for pattern in self._secrets.values() for m in pattern.finditer(text)]
        for start, end, digits in _numbers(text):
            if not 3 <= len(digits) <= 8 or _reads_as_something_else(text, start, end, digits):
                continue
            if _near_cue(text, start, end) or (strict and len(digits) >= 4):
                spans.append((start, end))
        for run in _SPOKEN.finditer(text):
            digits = spoken_digits(run.group())
            words = len(re.split(r"[\s,-]+", run.group()))
            if any(secret in digits for secret in self._secrets) or (
                len(digits) >= 3
                and words >= 2
                and (_near_cue(text, *run.span()) or (strict and len(digits) >= 4))
            ):
                spans.append(run.span())
        return _mask(text, spans)

    def data(self, value: Any, *, customer: bool = False, key: str = "") -> Any:
        """A copy of nested dicts and lists with every string redacted and every `pin` masked."""
        if isinstance(value, str):
            if key in _KEPT_KEYS or key.endswith("_id"):
                return value
            return self.text(value, customer=customer or key in _CUSTOMER_KEYS)
        if isinstance(value, dict):
            from_customer = value.get("role") == "user"  # a traced message
            return {
                k: MASK
                if k.lower() == "pin" and isinstance(v, str | int)
                else self.data(v, customer=customer or (from_customer and k == "content"), key=k)
                for k, v in value.items()
            }
        if isinstance(value, list | tuple):
            return [self.data(item, customer=customer, key=key) for item in value]
        return value

    def _harvest(self, value: Any) -> None:
        """Learn every `pin` in a tool call, so the same PIN is masked wherever else it shows."""
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() == "pin" and isinstance(item, str | int):
                    self.learn(item)
                else:
                    self._harvest(item)
        elif isinstance(value, list | tuple):
            for item in value:
                self._harvest(item)

    def event(self, event: BaseModel) -> BaseModel:
        """The event as it may be sent and stored."""
        if isinstance(event, LlmDelta):
            text = self.text(event.text)
            return event if text == event.text else event.model_copy(update={"text": text})
        dumped = event.model_dump()
        self._harvest(dumped)
        customer_fields = _CUSTOMER_FIELDS.get(dumped.get("type", ""), frozenset())
        cleaned = {
            key: self.data(value, customer=key in customer_fields, key=key)
            for key, value in dumped.items()
        }
        return event if cleaned == dumped else type(event).model_validate(cleaned)


def scrub_messages(messages: Sequence[AnyMessage], redactor: Redactor) -> list[AnyMessage]:
    """Copies of the stored messages that still show a PIN, with the PIN masked.

    Returned in a state update, each copy replaces its original by id. verify_identity calls this
    once it has used the PIN, which is the last moment the model needs it. Only the first two rules
    apply here: the model still needs the other numbers the customer wrote.
    """
    replaced: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            content = redactor.text(message.content)
            if content != message.content:
                replaced.append(message.model_copy(update={"content": content}))
        elif isinstance(message, AIMessage):
            update = {
                "tool_calls": redactor.data(message.tool_calls),
                "additional_kwargs": redactor.data(message.additional_kwargs),
            }
            if isinstance(message.content, str):
                update["content"] = redactor.text(message.content)
            if any(getattr(message, name) != value for name, value in update.items()):
                replaced.append(message.model_copy(update=update))
    return replaced
