"""Keeps PINs out of everything the backend stores or sends.

A PIN may sit in one place only: the conversation the model reads, and only until verify_identity
has been called with it. Events, traces, tickets, search queries and tool-argument echoes all pass
through a Redactor first.

A Redactor masks four things:

- Every value the session has passed to verify_identity as a PIN, right or wrong, however it is
  written: as digits, with anything short between the digits, or spelled out. In the customer's
  own words this rule has no exceptions. In text the app wrote itself, a match that is plainly an
  amount, a date or an ID is left alone, so a refund of $47.21 survives a PIN of 4721.
- A number whose keyed hash the session has stored. A process that picks a session up after a
  reload or a restart has never seen its PINs. The hashes tell it which numbers to mask again.
- A number next to a word such as "PIN" or "passcode", or right after an account number.
- In strict mode, any number in the customer's own text that could be a PIN: 4 to 7 digits,
  however they are separated or spelled, unless it reads as an amount, a date, an ID or a count of
  something. Strict mode is on while the session has no verified customer, which is when a PIN is
  expected and not yet known.

The last two rules are guesses and can miss an odd phrasing. The first cannot, and it holds from
the moment the model asks for verification.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel

from supportagent.events import LlmDelta

MASK = "****"

_UNITS = {
    "zero": 0,
    "oh": 0,
    "o": 0,
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
# "four seven two one", "forty-seven twenty-one", "four thousand seven hundred and twenty-one",
# "nine o three four", and digits read out one by one as speech-to-text writes them:
# "Four. Seven. Two. One."
_BETWEEN_WORDS = r"(?:\s*[-,./:;]\s*|\s+)(?:and\s+)?"
_SPOKEN = re.compile(rf"\b(?:{_WORD})\b(?:{_BETWEEN_WORDS}(?:{_WORD})\b)*", re.IGNORECASE)
_WORD_BREAK = re.compile(r"[\s,\-./:;]+")
_DIGITS = re.compile(r"\d+")
# What may sit between the short groups of one number: "4 7 2 1", "47-21", "4.7.2.1", "4, 7, 2, 1",
# "4/7/2/1", "47:21", "4*7*2*1", an en dash or an em dash, and up to three spaces.
_JOINT = re.compile(r"\s{1,3}|\s?[-.,_/:*\u2013\u2014]\s?")
# Between two digits of a PIN the session has learned: up to three characters that are neither a
# digit nor a letter. The value is known here, so anything short between its digits still gives it
# away.
_GAP = r"[^0-9A-Za-z]{0,3}"
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
# An account number and a short separator: what follows is the PIN. "10482917-4721",
# "10482917, 4721", "10482917 / 4721".
_AFTER_ACCOUNT = re.compile(r"(?<!\d)\d{8}\s{0,2}[-/:,;.|]?\s{0,2}$")
# A date, a time or a ratio written with a slash or a colon: 11/26, 10:30, 24/7, 11/26/2026.
_DATE_LIKE = re.compile(r"\d{1,2}([/:])\d{1,2}(?:\1\d{1,4})?")
# A word right after a number that says what the number counts.
_COUNTED = re.compile(
    r"\s?(?:seats?|users?|members?|people|licen[cs]es?|days?|weeks?|months?|years?|hours?"
    r"|minutes?|seconds?|[kmgt]b|projects?|boards?|cards?|tasks?|invoices?|tickets?|requests?"
    r"|calls?|dollars?|euros?|usd|eur|percent|times)\b",
    re.IGNORECASE,
)
# A word right before a number that says what the number is.
_LABELLED = re.compile(
    r"\b(?:error|status|http|port|zip|postal|postcode|version|build|room|suite|extension|ext"
    r"|order|case|reference|ref|ticket|invoice|line|step|page|q[1-4])"
    r"(?:\s(?:code|number|no\.?))?(?:\s(?:is|was|of))?\s?[:#=]?\s?$",
    re.IGNORECASE,
)
# A number at the end of text that is still streaming. It may be the start of a PIN.
_UNFINISHED_NUMBER = re.compile(r"\d[\d \t\-.,_/:*\u2013\u2014]{0,40}$")
# Keys whose values identify something and are never free text.
_KEPT_KEYS = frozenset(
    {"type", "id", "model", "from_model", "to_model", "source", "name", "account_number", "role"}
)
# Keys whose value is a secret whatever it looks like.
_SECRET_KEYS = frozenset({"pin", "pin_code", "passcode", "password"})
# Fields that hold the customer's own words, or a search query built from them.
_CUSTOMER_KEYS = frozenset({"query"})
_CUSTOMER_FIELDS = {"turn.started": frozenset({"text"})}


def session_key(secret: str | None, session_id: str) -> bytes:
    """The key under which one session's PIN hashes are kept.

    A PIN has four digits, so a plain hash of it can be reversed by trying them all. With a
    server secret in the key, the hashes in the database are useless to someone who has only the
    database.
    """
    return hmac.new((secret or "").encode(), session_id.encode(), hashlib.sha256).digest()


def spoken_digits(run: str) -> str:
    """The digits a run of number words stands for: "forty-seven twenty-one" gives "4721"."""
    words = [w for w in _WORD_BREAK.split(run.lower()) if w and w != "and"]
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
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # a model may send 4721.0 for 4721
    text = str(value)
    digits = re.sub(r"\D", "", text)
    return digits or "".join(spoken_digits(m.group()) for m in _SPOKEN.finditer(text))


def _numbers(text: str) -> list[tuple[int, int, str]]:
    """Each number in the text as (start, end, digits).

    Short groups with a short separator between them count as one number, which is how people
    type a PIN: "4 7 2 1", "47-21", "4/7/2/1".
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
                and _JOINT.fullmatch(text, groups[last][1], groups[last + 1][0])
            ):
                last += 1
        end = groups[last][1]
        found.append((start, end, re.sub(r"\D", "", text[start:end])))
        index = last + 1
    return found


def _joins(gap: str, limit: int) -> bool:
    """True when the text between a cue word and a number keeps them in one phrase."""
    return len(gap) <= limit and not any(ch.isdigit() or ch in ".!?;\n" for ch in gap)


def _cue_before(text: str, start: int) -> bool:
    """True when a word such as "PIN" stands shortly before text[start:]."""
    before = text[max(0, start - 40) : start]
    return any(_joins(before[cue.end() :], 24) for cue in _CUE.finditer(before))


def _cue_after(text: str, end: int) -> bool:
    """True when a word such as "PIN" stands shortly after text[:end]."""
    after = text[end : end + 32]
    cue = _CUE.search(after)
    return cue is not None and _joins(after[: cue.start()], 16)


def _reads_as_something_else(
    text: str, start: int, end: int, digits: str, *, cue_before: bool, cued: bool
) -> bool:
    """True when the number at text[start:end] is clearly not a PIN: an ID, an amount, a date."""
    before, after = text[:start], text[end:]
    if len(digits) == 8:
        return not cue_before  # an account number, unless the customer calls it a PIN
    if before[-1:].isalpha() or before[-1:] == "_" or after[:1].isalpha() or after[:1] == "_":
        return True  # part of a longer token
    if re.search(r"[A-Za-z]{2,}-$", before):
        return True  # INV-30417, RF-7214CA07
    if _AFTER_ACCOUNT.search(before):
        return False  # "10482917-4721" is an account number and its PIN, not a range
    if re.search(r"[$€£]\s?$", before) or after[:1] == "%":
        return True
    if re.match(r"[.,]\d", after) or re.search(r"\d[.,]$", before):
        return True  # 1440.50 or 1,440
    if re.match(r"[-/:]\d", after) or re.search(r"\d[-/:]$", before):
        return True  # a date, a time, a range, part of a phone number
    if not cued and _DATE_LIKE.fullmatch(text, start, end):
        return True  # 11/26 or 10:30, when nothing calls it a PIN
    return len(digits) == 4 and digits[:2] in ("19", "20") and bool(_YEAR_LEAD.search(before))


def _counts_something(text: str, start: int, end: int) -> bool:
    """True when the words around a number say what it is: "1500 seats", "error 5003"."""
    return bool(_COUNTED.match(text, end) or _LABELLED.search(text[:start]))


def _plainly_not_a_pin(text: str, start: int, end: int) -> bool:
    """True when a learned PIN's digits turn up in app-written text as an amount, an ID or a date.

    A tool result holds amounts, invoice numbers and dates. One of them can carry the same digits
    as a PIN by chance, and masking it would garble the trace without hiding anything.
    """
    before, after = text[:start], text[end:]
    if re.search(r"[$€£]\s?$", before) or after[:1] == "%":
        return True  # $47.21
    if re.search(r"[A-Za-z]{2,}-$", before):
        return True  # INV-4721
    if re.match(r"-\d{2}-\d{2}(?!\d)", after) or re.search(r"(?<!\d)\d{4}-(?:\d{2}-)?$", before):
        return True  # part of 2026-11-26
    return False


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


def _is_secret(key: str, value: object) -> bool:
    return (
        key.lower() in _SECRET_KEYS
        and isinstance(value, str | int | float)
        and not isinstance(value, bool)
    )


@dataclass
class _Stream:
    """The text of one model call as it streams: what the model wrote and what has gone out."""

    raw: str = ""
    sent: str = ""


class Redactor:
    """Masks PINs for one session. `learn` teaches it a PIN, `strict` turns the guessing rule on.

    `key` and `known` carry what an earlier process learned: `known` holds keyed hashes of this
    session's PINs, and a number whose hash is among them is learned again on sight.
    """

    def __init__(self, *, strict: bool = True, key: bytes = b"", known: Iterable[str] = ()) -> None:
        self.strict = strict
        self._key = key
        self._known: set[str] = set(known)
        self._secrets: dict[str, re.Pattern[str]] = {}
        self._streams: dict[str, _Stream] = {}

    @property
    def hashes(self) -> list[str]:
        """Keyed hashes of every PIN this session has used, for the next process to recognise."""
        return sorted(self._known)

    def _hash(self, digits: str) -> str:
        return hmac.new(self._key, digits.encode(), hashlib.sha256).hexdigest()[:32]

    def _remember(self, digits: str) -> None:
        joined = _GAP.join(digits)
        self._secrets[digits] = re.compile(rf"(?<!\d){joined}(?!\d)")
        self._known.add(self._hash(digits))

    def learn(self, value: object) -> None:
        """Remember a value that was given as a PIN, right or wrong."""
        digits = digits_of(value)
        if 3 <= len(digits) <= 12 and digits not in self._secrets:
            self._remember(digits)

    def _recognise(self, candidates: Iterable[str]) -> None:
        """Learn again the PINs an earlier process stored hashes of."""
        if len(self._known) == len(self._secrets):
            return  # every stored hash already has its value
        for digits in candidates:
            if 3 <= len(digits) <= 12 and digits not in self._secrets:
                if self._hash(digits) in self._known:
                    self._remember(digits)

    def text(
        self, text: str, *, customer: bool = False, guess: bool = True, known_only: bool = False
    ) -> str:
        """`text` with every PIN masked.

        `customer` marks the customer's own words. `guess` false keeps the strict rule out, for
        text the model still has to read the other numbers of. `known_only` keeps every rule
        out that works by the words around a number, and masks only PINs that were passed to
        verify_identity: for the message of a turn that has just ended, whose PIN the model may
        need in the next one.
        """
        if not text:
            return text
        text = text.replace("\x00", "")  # Postgres cannot store it, and nothing needs it
        strict = customer and guess and self.strict
        numbers = _numbers(text)
        runs = [
            (m.start(), m.end(), spoken_digits(m.group()), m.group())
            for m in _SPOKEN.finditer(text)
        ]
        self._recognise([digits for _, _, digits in numbers] + [digits for _, _, digits, _ in runs])

        spans: list[tuple[int, int]] = []
        for pattern in self._secrets.values():
            for match in pattern.finditer(text):
                if customer or not _plainly_not_a_pin(text, match.start(), match.end()):
                    spans.append(match.span())
        for start, end, digits in numbers:
            if known_only or not 3 <= len(digits) <= 8:
                continue
            cue_before, cue_after = _cue_before(text, start), _cue_after(text, end)
            paired = len(digits) != 8 and bool(_AFTER_ACCOUNT.search(text[:start]))
            cued = cue_before or cue_after or paired
            if _reads_as_something_else(text, start, end, digits, cue_before=cue_before, cued=cued):
                continue
            if cued or (
                strict and 4 <= len(digits) <= 7 and not _counts_something(text, start, end)
            ):
                spans.append((start, end))
        for start, end, digits, words in runs:
            count = len([w for w in _WORD_BREAK.split(words) if w])
            if any(secret in digits for secret in self._secrets) or (
                not known_only
                and len(digits) >= 3
                and count >= 2
                and (
                    _cue_before(text, start)
                    or _cue_after(text, end)
                    or (strict and len(digits) >= 4)
                )
            ):
                spans.append((start, end))
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
                if _is_secret(k, v)
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
                if _is_secret(key, item):
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

    def outgoing(self, event: BaseModel) -> list[BaseModel]:
        """What to send for one event of a live turn, in order. May be nothing.

        Streamed text is masked as a whole and not chunk by chunk, so a PIN that the model splits
        over two chunks is still caught: a number at the end of the text so far is held back until
        the next chunk shows where it ends. The rest of a call's text goes out before the event
        that finishes the call.
        """
        if isinstance(event, LlmDelta):
            stream = self._streams.setdefault(event.call_id, _Stream())
            stream.raw += event.text
            unfinished = _UNFINISHED_NUMBER.search(stream.raw)
            text = self._release(stream, unfinished.start() if unfinished else len(stream.raw))
            return [event.model_copy(update={"text": text})] if text else []
        before: list[BaseModel] = []
        stream = self._streams.get(getattr(event, "call_id", ""))
        if stream is not None and event.type in ("llm.finished", "llm.reset"):
            del self._streams[event.call_id]
            if event.type == "llm.finished":
                rest = self._release(stream, len(stream.raw))
                if rest:
                    before.append(LlmDelta(call_id=event.call_id, text=rest))
        return [*before, self.event(event)]

    def _release(self, stream: _Stream, upto: int) -> str:
        """The part of a stream's text up to `upto` that has not gone out yet, masked."""
        masked = self.text(stream.raw[:upto])
        if masked.startswith(stream.sent):
            new = masked[len(stream.sent) :]
        else:
            # Later text changed how earlier text is masked, and what went out cannot be taken
            # back. Send what differs from there on. The final message replaces all of it.
            new = masked[len(os.path.commonprefix([masked, stream.sent])) :]
        stream.sent = masked
        return new


def _scrubbed_content(content: Any, redactor: Redactor, **how: bool) -> Any:
    """A message's content with PINs masked, whether it is a string or a list of blocks."""

    def clean(text: str) -> str:
        return redactor.text(text, **how)

    if isinstance(content, str):
        return clean(content)
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, str):
                block = clean(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                block = {**block, "text": clean(block["text"])}
            blocks.append(block)
        return blocks
    return content


def scrub_messages(
    messages: Sequence[AnyMessage], redactor: Redactor, *, turn_start: int | None = None
) -> list[AnyMessage]:
    """Copies of the stored messages that still show a PIN, with the PIN masked.

    Returned in a state update, each copy replaces its original by id. It is called at two
    moments, and the customer's messages are treated differently at each. The strict rule stays
    out of both where the model still has to read the other numbers.

    verify_identity calls it once it has used the PIN, which is the last moment the model needs
    it. `turn_start` is None: PINs passed to the tool are masked, and so is every number the
    words around it call a PIN.

    A turn calls it when it ends, with the index of the message that opened it, because a model
    can repeat a PIN into a later tool call or into its answer. In that message only PINs that
    were passed to the tool are masked: one the customer typed and the model has not used yet may
    be what the next turn verifies with. In the customer's earlier messages every rule applies,
    so a PIN that was never used is gone from the saved conversation one turn later.
    """
    replaced: list[AnyMessage] = []
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            if turn_start is None:
                how = {"customer": True, "guess": False}
            elif index < turn_start:
                how = {"customer": True, "guess": True}
            else:
                how = {"customer": True, "known_only": True}
            content = _scrubbed_content(message.content, redactor, **how)
            if content != message.content:
                replaced.append(message.model_copy(update={"content": content}))
        elif isinstance(message, AIMessage):
            update = {
                "content": _scrubbed_content(message.content, redactor, customer=False),
                "tool_calls": redactor.data(message.tool_calls),
                "invalid_tool_calls": redactor.data(message.invalid_tool_calls),
                "additional_kwargs": redactor.data(message.additional_kwargs),
            }
            if any(getattr(message, name) != value for name, value in update.items()):
                replaced.append(message.model_copy(update=update))
    return replaced
