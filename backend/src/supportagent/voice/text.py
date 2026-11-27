"""Text on its way into speech and out of it.

What a voice may read aloud, how many digits a transcript holds, whether the customer is giving
or being asked for credentials, and whether what the recogniser heard is the assistant's own
voice coming back through the microphone.
"""

from __future__ import annotations

import re

from supportagent.redact import MASK

_CITATION = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}|>)\s*", re.MULTILINE)
_EMPHASIS = re.compile(r"[*`]+|__")
_WORD = re.compile(r"[a-z0-9']+")
# The assistant asks for them, or the customer starts to give them.
_ASKS = re.compile(r"(?i)\baccount number\b|\bPIN\b")
_GIVES = re.compile(r"(?i)\baccount number\b|\b(?:my|the|our) pin\b|\bpin (?:is|number|code)\b")
_READ_OUT = re.compile(r"(?<!\d)\d(?: \d)+(?!\d)")  # single digits with single spaces between


def speakable(text: str) -> str:
    """A reply as a voice should say it: no citation numbers, no markup, one line.

    A masked PIN has no business in a reply. If one gets this far it is named, not spelled out
    as four asterisks.
    """
    text = text.replace(MASK, "your PIN")
    text = _CITATION.sub("", text)
    text = _BULLET.sub("", text)
    text = _HEADING.sub("", text)
    text = _EMPHASIS.sub("", text)
    return " ".join(text.split())


def tidy_digits(transcript: str) -> str:
    """Write a number that was read out digit by digit as one number.

    The recogniser returns "1 0 4 8 2 9 1 7" for an account number read aloud. Twelve digits
    in a row are an account number and a PIN said in one breath, and come out as the two.
    """

    def join(match: re.Match[str]) -> str:
        digits = match.group().replace(" ", "")
        return f"{digits[:8]} {digits[8:]}" if len(digits) == 12 else digits

    return _READ_OUT.sub(join, transcript)


def without_echo(heard: str, speaking: str) -> str:
    """What is left of `heard` after the words the assistant was saying at the time.

    For a customer who spoke while the microphone was picking up the reply: the recogniser
    hands over both as one sentence, the reply's words first.
    """
    said = words(speaking)
    tokens = heard.split()
    position = kept = 0
    for token in tokens:
        key = "".join(_WORD.findall(token.lower()))
        try:
            position = said.index(key, position) + 1
        except ValueError:
            break
        kept += 1
    return " ".join(tokens[kept:])


def digit_count(transcript: str) -> int:
    """How many digits a transcript holds, written as numerals."""
    return sum(ch.isdigit() for ch in transcript)


def asks_for_credentials(reply: str) -> bool:
    return bool(_ASKS.search(reply))


def gives_credentials(transcript: str) -> bool:
    return bool(_GIVES.search(transcript))


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sounds_like_echo(heard: str, speaking: str) -> bool | None:
    """Whether the recogniser is hearing the assistant's own voice.

    True when what it heard are words the assistant is saying, in the order it says them. False
    when they are not. None while one or two words fit and more are needed to tell: "I" and
    "the" are in almost every reply, and a customer who says "I know" must not be taken for an
    echo.
    """
    said, got = words(speaking), words(heard)
    if not got:
        return None
    position = matched = 0
    for word in got:
        try:
            position = said.index(word, position) + 1
        except ValueError:
            continue
        matched += 1
    share = matched / len(got)
    if len(got) < 3:
        return None if share == 1.0 else False
    return share >= 0.8
