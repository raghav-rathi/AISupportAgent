"""Text on its way into speech and out of it.

What a voice may read aloud, how many digits a transcript holds, whether the customer is giving
or being asked for credentials, and whether what the recogniser heard is the assistant's own
voice coming back through the microphone.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from supportagent.redact import MASK

_CITATION = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}|>)\s*")
_EMPHASIS = re.compile(r"[*`]+|__|~~")
_UNDERLINED = re.compile(r"(?<![A-Za-z0-9])_([^_\n]+)_(?![A-Za-z0-9])")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_TABLE_RULE = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*")
_ENDS = ".!?:;,"
_WORD = re.compile(r"[a-z0-9']+")
# The assistant asks for them: a sentence that names one and reads as a request, or says that
# one did not match and is wanted again. A reply that only mentions a PIN, "you can reset your
# PIN in settings", is not one.
_NAMED = r"\b(?:account number|PIN)\b"
_ASKS = re.compile(
    r"(?i)\b(?:please|provide|share|give|tell|say|read|enter|need|what(?:'s| is| are))\b[^.?!]*"
    + _NAMED
    + r"|"
    + _NAMED
    + r"[^.?!]*\?|"
    + _NAMED
    + r"[^?]*\b(?:again|once more)\b"
)
# The customer starts to give them: one is named, and a digit follows or is about to.
_GIVES = re.compile(
    r"(?i)\b(?:account number|pin)\b(?:\s+(?:number|code))?(?:\s+(?:is|it's|its|was))?[\s,:]*(?:\d|$)"
)
# Nothing but numbers, three digits or more of them, with the little words people put between:
# someone reading a number out, whether or not anyone asked for it.
_NUMBER_WORD = r"zero|oh|one|two|three|four|five|six|seven|eight|nine"
_READING = re.compile(
    rf"(?i)\W*(?:(?:\d+|{_NUMBER_WORD}|and|then|it's|its|it|is|um|uh|okay|ok|sure|so)\b\W*)+"
)
_A_DIGIT = re.compile(rf"(?i)\d|\b(?:{_NUMBER_WORD})\b")
# One word that is a whole answer. After words of the reply itself in one transcript, a single
# other word is more often the recogniser's mistake than the customer.
_SHORT_ANSWERS = frozenset(
    {"yes", "no", "yeah", "yep", "nope", "okay", "ok", "stop", "wait", "thanks", "sure", "right"}
)
_READ_OUT = re.compile(r"(?<!\d)\d(?: \d)+(?!\d)")  # single digits with single spaces between
# A digit that sizes a number and is no part of one: "my 8 digit account number".
_SIZE_OF_A_NUMBER = re.compile(r"(?i)\b\d+[- ]?(?:digit|digits|character|characters)\b")
# Words that are numbers, or go with them, when the recogniser spells a number out.
_NUMBER_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty "
    "ninety hundred thousand million dollar dollars cent cents percent".split()
)
# What people say to get a word in. Never taken for the assistant's own voice.
_INTERRUPTS = _SHORT_ANSWERS | {
    "cancel",
    "hello",
    "hey",
    "sorry",
    "pardon",
    "excuse",
    "hold",
    "what",
}
_JUST_SAID = 12  # words: about four seconds of speech
_NUMBER = "#"  # stands for any number, see `_content`


def speakable(text: str) -> str:
    """A reply as a voice should say it: no citation numbers, no markup, one line.

    A masked PIN has no business in a reply. If one gets this far it is named, not spelled out
    as four asterisks.
    """
    text = text.replace(MASK, "your PIN")
    text = _LINK.sub(r"\1", text)  # a link is said as its label, not as its address
    text = _CITATION.sub("", text)
    said = []
    for line in text.splitlines():
        if _TABLE_RULE.fullmatch(line) and "-" in line:
            continue  # the rule under a table's head
        row = line.lstrip().startswith("|")
        # A list item, a heading or a table row is a sentence of its own. Run together
        # with the next one, the voice reads them in one breath.
        apart = row or _BULLET.match(line) is not None or _HEADING.match(line) is not None
        line = _HEADING.sub("", _BULLET.sub("", line))
        if row:
            line = ", ".join(cell.strip() for cell in line.split("|") if cell.strip())
        line = _UNDERLINED.sub(r"\1", _EMPHASIS.sub("", line)).strip()
        if apart and line and line[-1] not in _ENDS:
            line += "."
        said.append(line)
    return " ".join(" ".join(said).split())


def tidy_digits(transcript: str) -> str:
    """Write a number that was read out digit by digit as one number.

    The recogniser returns "1 0 4 8 2 9 1 7" for an account number read aloud. More than eight
    digits in a row are an account number and a PIN said in one breath, and come out as the
    two. That holds while they are still coming: nine digits are an account number and the
    first digit of a PIN, and written as one number that digit would be shown.
    """

    def join(match: re.Match[str]) -> str:
        digits = match.group().replace(" ", "")
        return f"{digits[:8]} {digits[8:]}" if len(digits) > 8 else digits

    return _READ_OUT.sub(join, transcript)


def _content(text: str) -> list[str]:
    """The words of `text` as they are compared between what was said and what was heard.

    A number is one word, "#", however it is spelled and however long it is. The reply has
    "$500" and "14 days", and the recogniser writes what it hears of them as "five hundred
    dollars" and "fourteen days", or as digits, depending on how it is set at the moment. A
    digit that only sizes a number, "8-digit", is left out: it is not a number anyone reads out.
    """
    out: list[str] = []
    for word in words(_SIZE_OF_A_NUMBER.sub(" ", text)):
        if word.isdigit() or word in _NUMBER_WORDS:
            if not out or out[-1] != _NUMBER:
                out.append(_NUMBER)
        else:
            out.append(word)
    return out


def _runs(said: list[str], got: list[str]) -> list[tuple[int, int]]:
    """Where `got` repeats two or more words in a row of `said`: (start in `got`, length), in order."""
    matcher = SequenceMatcher(None, said, got, autojunk=False)
    return [(block.b, block.size) for block in matcher.get_matching_blocks() if block.size >= 2]


def _within(few: list[str], many: list[str]) -> bool:
    """Whether `few` stand side by side somewhere in `many`."""
    size = len(few)
    return any(many[i : i + size] == few for i in range(len(many) - size + 1))


def without_echo(heard: str, played: str) -> str:
    """What is left of `heard` after the words the assistant had just said.

    For a customer who spoke while the microphone was picking up the reply: the recogniser
    hands over both as one sentence, the reply's words first. Those are taken off only where
    they are plainly the reply: three or more of its words in a row, at or near the start,
    and what follows of the same kind with at most a misheard word or two between. A sentence
    of the customer's own that begins with two words the reply also has, "the account number
    is", keeps them.
    """
    tokens = heard.split()
    got: list[str] = []
    token_of: list[int] = []
    for index, token in enumerate(tokens):
        for word in _content(token):
            if word == _NUMBER and got and got[-1] == _NUMBER:
                token_of[-1] = index  # a number said in several words ends with its last
            else:
                got.append(word)
                token_of.append(index)
    runs = _runs(_content(played), got)
    if not runs or runs[0][0] > 2 or runs[0][1] < 3:
        return heard
    end = runs[0][0] + runs[0][1]
    for start, size in runs[1:]:
        if start - end > 2:
            break
        end = start + size
    return " ".join(tokens[token_of[end - 1] + 1 :])


def digit_count(transcript: str) -> int:
    """How many digits of a number a transcript holds, written as numerals.

    A digit that only says how long a number is does not count: in "my 8 digit account number
    is" nothing of the number has been said yet.
    """
    return sum(ch.isdigit() for ch in _SIZE_OF_A_NUMBER.sub("", transcript))


def asks_for_credentials(reply: str) -> bool:
    """Whether a reply asks the customer for the account number or the PIN."""
    return bool(_ASKS.search(reply))


def is_speech(rest: str) -> bool:
    """Whether what is left after the reply's own words is the customer saying something."""
    said = words(rest)
    return len(said) >= 2 or (len(said) == 1 and said[0] in _SHORT_ANSWERS)


def gives_credentials(transcript: str) -> bool:
    """Whether the customer has started to give an account number or a PIN: they name one and
    go on to say it, or they are reading out digits and nothing else."""
    if _GIVES.search(transcript):
        return True
    return len(_A_DIGIT.findall(transcript)) >= 3 and _READING.fullmatch(transcript) is not None


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sounds_like_echo(heard: str, played: str) -> bool | None:
    """Whether the recogniser is hearing the assistant's own voice.

    `played` is what the assistant has said so far, as far as the customer's player has got.
    Words it has not played yet cannot come back through the microphone, and a customer who
    happens to use them is not an echo.

    True when what was heard repeats the reply in runs of words: most of it, or a good part
    with one run of three or more, which is what the recogniser makes of the voice when it
    mishears a word here and there. False when it does not, and at once when it opens with a
    word people interrupt with. None while there are only one or two words that the reply also
    has, side by side: "thank you" and "the plan" are in many replies, and more is needed to
    tell. A number counts as a number, whatever it is, see `_content`.
    """
    said, got = _content(played), _content(heard)
    if not got:
        return None
    runs = _runs(said, got)
    if got[0] in _INTERRUPTS and not (runs and runs[0][0] == 0):
        return False
    if len(got) < 3:
        return None if _within(got, said) else False
    share = sum(size for _, size in runs) / len(got)
    longest = max((size for _, size in runs), default=0)
    return share >= 0.8 or (longest >= 3 and share >= 0.6)


def is_short_echo(heard: str, played: str) -> bool:
    """Whether a turn of one or two words was the assistant's own voice.

    For a turn that ended before there were words enough to tell. It is taken for the echo
    when its words are among the last few the assistant said, side by side, and none of them
    is a word people interrupt with: a "stop" that is passed over because the reply also said
    "stop" would leave the customer with no way in.
    """
    got = _content(heard)
    if not got or len(got) > 2 or any(word in _INTERRUPTS for word in got):
        return False
    if all(word == _NUMBER for word in got):
        return False  # a number and nothing else: more likely the PIN than the reply
    return _within(got, _content(played)[-_JUST_SAID:])
