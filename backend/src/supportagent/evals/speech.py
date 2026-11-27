"""Speech measurements against the live service.

    uv run python -m supportagent.evals.speech make DIR     synthesise the test clips into DIR
    uv run python -m supportagent.evals.speech wer DIR      domain sentences, with and without keyterms
    uv run python -m supportagent.evals.speech digits DIR   account numbers with pauses, both modes
    uv run python -m supportagent.evals.speech pins DIR     the same clips: is the PIN masked in what
                                                            the recogniser wrote, however it wrote it
    uv run python -m supportagent.evals.speech heard A.wav  what a recording says, for comparing a
                                                            cut-off reply with what was stored

All of it needs DEEPGRAM_API_KEY and costs a few cents of its credit.

The clips are synthetic: Deepgram's voices, read back to Deepgram's recogniser, with a little
noise in place of a room. That is easier than a person on a laptop microphone, so an error rate
measured here is a floor and not an estimate. The pauses inside the numbers are the harder kind:
each part was synthesised as a sentence of its own, so the voice drops at every pause the way a
speaker does who has finished.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import re
import wave
from pathlib import Path

import httpx

from supportagent.config import get_settings
from supportagent.redact import Redactor, spoken_digits
from supportagent.voice.deepgram import FluxListener, SpeechConfig, SpeechError
from supportagent.voice.session import _DIGITS, _USUAL
from supportagent.voice.text import digit_count, tidy_digits

_RATE = 16_000
_FRAME = 2_560  # 80 ms
_VOICES = [
    "aura-2-thalia-en",
    "aura-2-apollo-en",
    "aura-2-helena-en",
    "aura-2-arcas-en",
    "aura-2-luna-en",
    "aura-2-orion-en",
]

DOMAIN = [
    "Hi, I need help with a refund.",
    "How long do I have to ask for a refund on a monthly plan?",
    "What is my refund limit?",
    "Can you check the status of my last invoice?",
    "I was charged twice for the same invoice.",
    "Does Fernhill support single sign-on on the Pro plan?",
    "How fast do you respond to an outage on the Enterprise plan?",
    "I forgot my PIN. How do I reset it?",
    "Please update the billing email on my account.",
    "I want to talk to a person.",
    "What is the difference between the Free plan and the Pro plan?",
    "Can I connect Fernhill to Microsoft Teams?",
    "My account is suspended. What do I need to do?",
    "How many seats are included in my subscription?",
    "I would like to cancel my annual contract.",
    "Is there a discount for nonprofits?",
    "Where can I download my invoices?",
    "Can I get a refund for the storage add-on?",
    "What was your uptime last month?",
    "Should I give you my account number and my PIN now?",
]

# What a scripted caller says in a conversation. The first ten get an answer without any tool.
DIALOGUE = {
    "hello": "Hi, I need help with a refund.",
    "hear": "Hello, can you hear me?",
    "what": "What can you help me with?",
    "email": "I want to change my billing email.",
    "thanks": "Thanks, that is all for now.",
    "check": "Can you check my account?",
    "twice": "I think I was charged twice.",
    "second": "Okay, give me a second.",
    "need": "What do you need from me?",
    "morning": "Good morning.",
    "creds": "My account number is 1 0 4 8 2 9 1 7, and my PIN is 4 7 2 1.",
    "window": "How long do I have to ask for a refund on a monthly plan?",
    "limit": "What is my refund limit?",
    "wait": "Wait, stop. I have another question.",
    "story": "Please explain how refunds work on the Pro plan, with all the details.",
}

_ACCOUNTS = [
    ("10482917", "4721"),
    ("20573846", "9034"),
    ("53920164", "2458"),
    ("75148302", "3390"),
    ("64037291", "8815"),
    ("42851730", "6672"),
]


def _spaced(digits: str) -> str:
    return " ".join(digits)


def digit_clips(seed: int = 7) -> list[dict]:
    """Twenty ways to say an account number and a PIN, with pauses of 0.7 to 2.0 s inside."""
    rng = random.Random(seed)
    numbers = list(_ACCOUNTS)
    while len(numbers) < 20:
        numbers.append((f"{rng.randrange(10**7, 10**8)}", f"{rng.randrange(10**3, 10**4)}"))
    clips = []
    for index, (account, pin) in enumerate(numbers):
        first, second = _spaced(account[:4]), _spaced(account[4:])

        def pause() -> float:
            return round(rng.uniform(0.7, 2.0), 1)

        shapes = [
            [f"My account number is {first}.", pause(), f"{second}.", pause(),
             f"My PIN is {_spaced(pin)}."],
            [f"{first}.", pause(), f"{second}.", pause(), f"{_spaced(pin)}."],
            [f"Sure. It is {first}.", pause(), f"{second}.", pause(),
             f"And the PIN is {_spaced(pin[:2])}.", pause(), f"{_spaced(pin[2:])}."],
            [f"The account number is {_spaced(account)}.", pause(), f"The PIN is {_spaced(pin)}."],
        ]  # fmt: skip
        clips.append(
            {
                "name": f"digits_{index:02}",
                "parts": shapes[index % len(shapes)],
                "digits": account + pin,
                "voice": _VOICES[index % len(_VOICES)],
            }
        )
    return clips


def _key() -> str:
    key = get_settings().deepgram_api_key
    if key is None:
        raise SystemExit("DEEPGRAM_API_KEY is not set")
    return key.get_secret_value()


async def _synthesise(http: httpx.AsyncClient, text: str, voice: str) -> bytes:
    """Raw mono PCM16 at 16 kHz for one sentence."""
    response = await http.post(
        "https://api.deepgram.com/v1/speak",
        params={"model": voice, "encoding": "linear16", "sample_rate": _RATE, "container": "none"},
        json={"text": text},
    )
    response.raise_for_status()
    return response.content


def _room(seconds: float, rng: random.Random) -> bytes:
    """A little noise for a pause: a real microphone is never at zero."""
    count = int(seconds * _RATE)
    return b"".join(rng.randrange(-24, 25).to_bytes(2, "little", signed=True) for _ in range(count))


def _write(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(_RATE)
        out.writeframes(pcm)


async def make(folder: Path) -> None:
    rng = random.Random(11)
    manifest: dict[str, dict] = {}
    headers = {"Authorization": f"Token {_key()}"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as http:
        jobs = [
            (f"domain_{i:02}", [text], _VOICES[i % len(_VOICES)], {"text": text})
            for i, text in enumerate(DOMAIN)
        ]
        jobs += [
            (name, [text], _VOICES[i % 2], {"text": text})
            for i, (name, text) in enumerate(DIALOGUE.items())
        ]
        jobs += [
            (clip["name"], clip["parts"], clip["voice"], {"digits": clip["digits"]})
            for clip in digit_clips()
        ]
        for name, parts, voice, facts in jobs:
            pcm = _room(0.3, rng)
            for part in parts:
                if isinstance(part, float):
                    pcm += _room(part, rng)
                else:
                    pcm += await _synthesise(http, part, voice)
            _write(folder / f"{name}.wav", pcm + _room(0.2, rng))
            manifest[name] = {"voice": voice, "parts": parts, **facts}
            print(f"{name}: {len(pcm) / 2 / _RATE:.1f} s, {voice}")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


async def _listen(
    pcm: bytes, config: SpeechConfig, settings: dict | None, tail_s: float = 6.0
) -> list[dict]:
    """Play a clip to the recogniser in real time. Returns each turn it ended, with its time."""
    listener = FluxListener(config)
    await listener.open()
    if settings:
        await listener.configure(**settings)
    ended: list[dict] = []

    async def read() -> None:
        try:
            async for turn in listener.turns():
                if turn.event == "EndOfTurn":
                    ended.append(
                        {"text": turn.transcript, "at_s": turn.window_end, "by": turn.trigger}
                    )
        except SpeechError:
            pass

    reader = asyncio.create_task(read())
    audio = pcm + bytes(int(tail_s * _RATE) * 2)
    loop = asyncio.get_running_loop()
    due = loop.time()
    for start in range(0, len(audio), _FRAME):
        await listener.send(audio[start : start + _FRAME])
        due += 0.08
        await asyncio.sleep(max(0.0, due - loop.time()))
        if ended and ended[-1]["at_s"] > len(pcm) / 2 / _RATE:
            break  # the turn after the last word is over
    await listener.close()
    await reader
    return ended


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


def _errors(reference: list[str], heard: list[str]) -> int:
    """Word edits between the two: the Levenshtein distance."""
    row = list(range(len(heard) + 1))
    for i, word in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(heard, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1]


async def _in_batches(jobs: list, size: int = 5) -> list:
    results = []
    for start in range(0, len(jobs), size):
        results += await asyncio.gather(*jobs[start : start + size])
    return results


async def wer(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    names = sorted(name for name in manifest if name.startswith("domain_"))
    base = SpeechConfig.from_settings(get_settings())
    for label, config in (
        ("with keyterms", base),
        ("without keyterms", SpeechConfig(api_key=base.api_key, keyterms=())),
    ):
        runs = await _in_batches([_listen(_pcm(folder / f"{n}.wav"), config, None) for n in names])
        wrong = total = 0
        for name, ended in zip(names, runs, strict=True):
            reference = _words(manifest[name]["text"])
            heard = _words(" ".join(turn["text"] for turn in ended))
            errors = _errors(reference, heard)
            wrong, total = wrong + errors, total + len(reference)
            if errors:
                print(f"  {name}: {manifest[name]['text']!r} heard as {[t['text'] for t in ended]}")
        print(f"{label}: {wrong} of {total} words wrong, {wrong / total:.1%}")


async def digits(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    names = sorted(name for name in manifest if name.startswith("digits_"))
    config = SpeechConfig.from_settings(get_settings())
    for label, settings in (("digit mode", _DIGITS), ("usual mode", _USUAL | {"numerals": True})):
        jobs = [_listen(_pcm(folder / f"{n}.wav"), config, settings) for n in names]
        cut = right = 0
        for name, ended in zip(names, await _in_batches(jobs), strict=True):
            wanted = manifest[name]["digits"]
            got = "".join(c for c in tidy_digits(ended[0]["text"]) if c.isdigit()) if ended else ""
            whole = len(ended) == 1
            cut += not whole
            right += whole and got == wanted
            if not whole or got != wanted:
                turns = [(t["text"], t["by"]) for t in ended]
                print(f"  {name}: wanted {wanted}, got {got!r} in {len(ended)} turn(s): {turns}")
            elif digit_count(ended[0]["text"]) != len(wanted):
                print(f"  {name}: {ended[0]['text']!r}")
        print(f"{label}: {cut} of {len(names)} cut into pieces, {right} of {len(names)} exact")


def _still_shows(pin: str, account: str, masked: str) -> bool:
    """Whether the PIN can be read in a masked transcript: as digits, spaced out, or in words."""
    runs = re.findall(r"\d+", spoken_digits(re.sub(r"(?<=\d)[ .,-]+(?=\d)", "", masked)))
    return any(pin in run for run in runs if run != account)


async def pins(folder: Path) -> list[dict]:
    """Every transcript of the digit clips through the redactor, as an unverified visitor's words."""
    manifest = json.loads((folder / "manifest.json").read_text())
    names = sorted(name for name in manifest if name.startswith("digits_"))
    config = SpeechConfig.from_settings(get_settings())
    kept = []
    for label, settings in (
        ("digit mode", _DIGITS),
        ("usual mode, numerals", _USUAL | {"numerals": True}),
        ("usual mode, words", _USUAL),
    ):
        jobs = [_listen(_pcm(folder / f"{n}.wav"), config, settings) for n in names]
        shown = total = 0
        for name, ended in zip(names, await _in_batches(jobs), strict=True):
            account, pin = manifest[name]["digits"][:8], manifest[name]["digits"][8:]
            redactor = Redactor()  # a new visitor each time, not verified
            for turn in ended:
                masked = redactor.text(tidy_digits(turn["text"]), customer=True)
                total += 1
                leak = _still_shows(pin, account, masked)
                shown += leak
                kept.append(
                    {
                        "mode": label,
                        "clip": name,
                        "pin": pin,
                        "heard": turn["text"],
                        "masked": masked,
                        "shows": leak,
                    }
                )
                if leak:
                    print(f"  {name}: {turn['text']!r} -> {masked!r}")
        print(f"{label}: the PIN can be read in {shown} of {total} transcripts")
    return kept


async def heard(paths: list[Path]) -> None:
    """Print what each recording says, by Deepgram's recogniser for recorded audio."""
    headers = {"Authorization": f"Token {_key()}", "Content-Type": "audio/wav"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as http:
        for path in paths:
            body = io.BytesIO(path.read_bytes()).getvalue()
            response = await http.post(
                "https://api.deepgram.com/v1/listen",
                params={"model": "nova-3", "smart_format": "false"},
                content=body,
            )
            response.raise_for_status()
            alternative = response.json()["results"]["channels"][0]["alternatives"][0]
            print(f"{path.name}: {alternative['transcript']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["make", "wer", "digits", "pins", "heard"])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.what == "make":
        args.paths[0].mkdir(parents=True, exist_ok=True)
    if args.what == "heard":
        asyncio.run(heard(args.paths))
    elif args.what == "pins":
        kept = asyncio.run(pins(args.paths[0]))
        if len(args.paths) > 1:  # every transcript and what became of it, as JSON
            args.paths[1].write_text(json.dumps(kept, indent=1))
    else:
        asyncio.run({"make": make, "wer": wer, "digits": digits}[args.what](args.paths[0]))


if __name__ == "__main__":
    main()
