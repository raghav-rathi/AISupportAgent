"""Markdown chunker built on the heading tree.

Rules:

- A chunk never crosses a heading, because sections can be visible to different plans.
- A chunk never starts or ends inside a fenced code block. A code block larger than the size limit
  gets a chunk of its own. A fence may be indented by up to three spaces, as inside a list item, and
  is closed by a fence of the same character that is at least as long. Inside a code block nothing
  is a heading, a comment or a plan marker.
- A table or a numbered procedure is never split. Items separated by blank lines are still one list.
- A bulleted list is split only when it alone is larger than the size limit, and then only between
  its top-level items. An item, and a code block inside it, is never cut.
- A paragraph is split between sentences, and only when it alone is larger than the size limit.
- A lead-in line that ends in a colon stays with the list, table or code block it introduces.
- Each chunk carries its heading path and the plans that may see it.
- A table is kept as Markdown for display and spelled out row by row for search. Rerankers score
  "Billing cycle: Monthly; Refund window: 14 days." higher than the same row between pipes.
- IDs depend on the document and the heading path, not on the text, so an edited section keeps its
  ID and only its hash changes. IDs are unique across the knowledge base, even when a heading
  contains `/`.
- HTML comments never reach a chunk. A comment may span lines, and no line inside it is a heading.
  The text around a comment on its line is kept. A comment that is never closed is an error.

Document format: front matter with `id`, `title` and `tiers`, then Markdown under one `# ` title.

Plan markers. A section and everything below it is narrowed to some plans with a marker:

    <!-- tiers: pro, enterprise -->

- The marker is a line on its own: `<!--`, the word `tiers` in lower case, a colon, plan names
  separated by commas, `-->`. Plan names are the lower case names in `TIERS`, at least one.
- It sits under a heading of level 2 or deeper, before any content of that section. Blank lines and
  other one-line comments may come between the heading and the marker. A comment that spans lines
  counts as content.
- It may only narrow the plans of the parent section. A section has at most one marker.
- The document title takes its plans from the front matter. A marker under it or before it is an
  error.

Markers fail closed. Every HTML comment outside a code block whose text contains `tier`, in any case
and inside any word (`plan_tiers`, `mytiers`), is either a valid marker in a valid place or a
`ChunkerError` that names the file and line. A misspelled or misplaced marker never leaves its
section open to every plan.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from supportagent.crm.policy import TIERS

TARGET_CHARS = 1_200
MAX_CHARS = 2_000

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TIERS_MARK = re.compile(r"<!--\s*tiers:\s*([^<>]*?)\s*-->")
_MENTIONS_TIER = re.compile(r"tier", re.IGNORECASE)
# A backtick fence has no backtick after its opening run, so "```x```" on a line is not a fence.
_FENCE = re.compile(r"^ {0,3}(`{3,}(?=[^`]*$)|~{3,})")
_ORDERED = re.compile(r"^\d+[.)]\s+")
_BULLET = re.compile(r"^[-*+]\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TABLE_RULE = re.compile(r":?-+:?")
_CELL_BREAK = re.compile(r"(?<!\\)\|")  # a pipe written as `\|` belongs to its cell

_LISTS = ("procedure", "bullets")

# Block kinds as parsed, mapped to the kinds a chunk reports.
_PUBLIC_KIND = {
    "prose": "prose",
    "bullets": "prose",
    "code": "prose",
    "procedure": "procedure",
    "table": "table",
}

# How a unit led by each block kind may be split when it alone is above the size limit.
_SPLIT = {
    "prose": "sentences",
    "bullets": "items",
    "code": "never",
    "procedure": "never",
    "table": "never",
}


class ChunkerError(ValueError):
    """A document the chunker cannot turn into chunks without guessing."""


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    heading_path: tuple[str, ...]  # headings below the document title
    ordinal: int  # position inside its section
    tiers: tuple[str, ...]
    kind: str  # prose | procedure | table | mixed
    content: str  # the Markdown shown to the model and cited to the user
    embed_text: str  # heading path plus search text: what is embedded, indexed and reranked
    content_hash: str


@dataclass(frozen=True)
class _Block:
    kind: str  # prose | bullets | procedure | table | code
    text: str


@dataclass(frozen=True)
class _Unit:
    """Text that must stay in one chunk, in its display form and its search form."""

    text: str
    search: str
    kind: str
    split: str = "never"  # never | sentences | items, used only above the size limit


@dataclass(frozen=True)
class _Section:
    path: tuple[str, ...]
    tiers: tuple[str, ...]
    blocks: list[_Block]
    line: int  # line of the heading that opens the section


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("/", "\\/").replace("|", "\\|")


def _chunk_id(doc_id: str, path: tuple[str, ...], ordinal: int) -> str:
    """Hash a key that tells every document, heading path and position apart.

    A backslash, `/` or `|` inside a name is escaped, so ("A/B",) and ("A", "B") get different keys.
    A name without those characters gives the same key, and so the same ID, as it always has.
    """
    path_key = "/".join(_escape(name) for name in path)
    return _digest(f"{_escape(doc_id)}|{path_key}|{ordinal}")


def _parse_tiers(value: str, where: str) -> tuple[str, ...]:
    names = {name.strip() for name in value.split(",") if name.strip()}
    if not names or names - set(TIERS):
        raise ChunkerError(f"{where}: bad tiers {value!r}, expected some of {', '.join(TIERS)}")
    return tuple(tier for tier in TIERS if tier in names)


def _front_matter(text: str, source: str) -> tuple[dict[str, str], list[str], int]:
    """Return the front matter, the body lines and the line number of the first body line."""
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ChunkerError(f"{source}: missing front matter")
    end = lines.index("---", 1)
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    missing = [key for key in ("id", "title", "tiers") if not meta.get(key)]
    if missing:
        raise ChunkerError(f"{source}: front matter lacks {', '.join(missing)}")
    return meta, lines[end + 1 :], end + 2


def _closes(line: str, fence: str) -> bool:
    """True when `line` closes a code block opened with `fence`.

    The closing fence repeats the opening character at least as often, after at most three spaces.
    """
    mark = line.rstrip()
    run = mark.lstrip(" ")
    return len(mark) - len(run) <= 3 and len(run) >= len(fence) and run == fence[0] * len(run)


def _split_comments(line: str, open_comment: bool) -> tuple[str, list[str], bool]:
    """Split `line` into the text outside HTML comments and the text of each comment.

    `open_comment` says whether a comment is open at the start of the line. The last value says
    whether one is still open at its end.
    """
    visible: list[str] = []
    texts: list[str] = []
    pos = 0
    while True:
        if open_comment:
            end = line.find("-->", pos)
            if end < 0:
                texts.append(line[pos:])
                break
            texts.append(line[pos:end])
            pos, open_comment = end + 3, False
            continue
        start = line.find("<!--", pos)
        if start < 0:
            visible.append(line[pos:])
            break
        visible.append(line[pos:start])
        # `<!-->` and `<!--->` are whole comments, so the close may overlap the opener.
        end = line.find("-->", start + 2)
        if end < 0:
            texts.append(line[start + 4 :])
            open_comment = True
            break
        texts.append(line[start + 4 : end])
        pos = end + 3
    return "".join(visible), texts, open_comment


def _line_kind(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("|"):
        return "table"
    if _ORDERED.match(stripped):
        return "procedure"
    if _BULLET.match(stripped):
        return "bullets"
    return "prose"


def _blocks(lines: list[str]) -> list[_Block]:
    """Group the lines of one section into paragraphs, lists, tables and code fences.

    Blank lines between the items of a list keep the list in one block. An indented code fence
    inside a list belongs to the list.
    """
    blocks: list[_Block] = []
    buffer: list[str] = []
    kind = "prose"
    fence: str | None = None
    gap = False  # a blank line came after the last line of the open list

    def flush() -> None:
        nonlocal gap
        if buffer:
            blocks.append(_Block(kind, "\n".join(buffer)))
            buffer.clear()
        gap = False

    for line in lines:
        if fence:
            buffer.append(line)
            if _closes(line, fence):
                fence = None
                if kind == "code":
                    flush()
            continue
        opener = _FENCE.match(line)
        if opener and buffer and kind in _LISTS and line[0] == " ":
            if gap:
                buffer.append("")
                gap = False
            fence = opener.group(1)
            buffer.append(line)
            continue
        if opener:
            flush()
            kind, fence = "code", opener.group(1)
            buffer.append(line)
            continue
        if not line.strip():
            if kind in _LISTS:
                gap = True
            else:
                flush()
            continue
        line_kind = _line_kind(line)
        if buffer and kind in _LISTS and (line_kind == kind or line[0] in " \t"):
            if gap:
                buffer.append("")
                gap = False
            buffer.append(line)
        elif buffer and line_kind == kind:
            buffer.append(line)
        else:
            flush()
            kind = line_kind
            buffer.append(line)
    flush()
    return blocks


def _sections(
    lines: list[str], first_line: int, doc_tiers: tuple[str, ...], source: str
) -> list[_Section]:
    """Split the body at headings, drop HTML comments and give each section its plans.

    `first_line` is the line number of `lines[0]` in the file, for error messages.
    """
    sections: list[_Section] = []
    stack: list[tuple[int, str, tuple[str, ...]]] = []  # (level, title, tiers) of open headings
    body: list[str] = []
    path: tuple[str, ...] = ()
    tiers = doc_tiers
    level = 0  # level of the heading that opened the current section, 0 before the title
    opened_at = first_line
    fence: str | None = None
    fence_at = 0
    in_comment = False  # inside an HTML comment that spans lines
    comment_at = 0  # line where that comment opened
    preamble = True  # from a heading until the first content line of its section
    marked = False  # the current section has a plan marker

    def flush() -> None:
        blocks = _blocks(body)
        if blocks:
            sections.append(_Section(path, tiers, blocks, opened_at))
        body.clear()

    for number, line in enumerate(lines, start=first_line):
        where = f"{source}:{number}"
        if fence:
            body.append(line)
            if _closes(line, fence):
                fence = None
            continue
        opener = None if in_comment else _FENCE.match(line)
        if opener:
            fence, fence_at, preamble = opener.group(1), number, False
            body.append(line)
            continue

        was_open = in_comment
        mark = None if was_open else _TIERS_MARK.fullmatch(line.strip())
        visible, texts, in_comment = _split_comments(line, was_open)
        if in_comment and (not was_open or "-->" in line):
            comment_at = number
        if not mark and any(_MENTIONS_TIER.search(text) for text in texts):
            raise ChunkerError(
                f"{where}: this comment mentions tiers but is not a plan marker. Write the marker"
                " as `<!-- tiers: pro, enterprise -->` on a line of its own, or reword the comment."
            )
        if texts:
            # Keep the text around the comments, indented as the line was, so a list stays a list.
            indent = "" if was_open else line[: len(line) - len(line.lstrip())]
            visible = f"{indent}{visible.strip()}" if visible.strip() else ""
            heading_before = not was_open and _HEADING.match(line)
            if bool(heading_before) != bool(_HEADING.match(visible)) or _FENCE.match(visible):
                raise ChunkerError(
                    f"{where}: an HTML comment makes it unclear whether this line is a heading or"
                    " a code fence. Put the comment on a line of its own."
                )

        heading = _HEADING.match(visible)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            opened_at, preamble, marked = number, True, False
            if level == 1:
                # The document title leads every embed text already, so it stays out of the path.
                path, tiers = (), doc_tiers
                continue
            parent_tiers = stack[-1][2] if stack else doc_tiers
            stack.append((level, title, parent_tiers))
            path, tiers = tuple(s[1] for s in stack), parent_tiers
            continue

        if mark:
            if level < 2:
                raise ChunkerError(
                    f"{where}: a plan marker must sit under a heading of level 2 or deeper. The"
                    " plans of the whole document come from `tiers` in the front matter."
                )
            heading_level, title, parent_tiers = stack[-1]
            if not preamble:
                raise ChunkerError(
                    f"{where}: the plan marker of section {title!r} must come right after its"
                    " heading, before any content."
                )
            if marked:
                raise ChunkerError(f"{where}: section {title!r} has a second plan marker.")
            tiers = _parse_tiers(mark.group(1), where)
            if not set(tiers) <= set(parent_tiers):
                raise ChunkerError(
                    f"{where}: section {title!r} widens its parent's tiers {parent_tiers}"
                )
            stack[-1] = (heading_level, title, tiers)
            marked = True
            continue
        if was_open or in_comment:
            preamble = False
        if texts and not visible:
            continue
        if visible.strip():
            preamble = False
        body.append(visible)

    if fence:
        raise ChunkerError(f"{source}:{fence_at}: this code block is never closed.")
    if in_comment:
        raise ChunkerError(f"{source}:{comment_at}: this HTML comment is never closed.")
    flush()
    return sections


def _cells(row: str) -> list[str]:
    """Split a table row into cells. An empty first cell stays a cell and `\\|` stays text."""
    row = row.strip().removeprefix("|")
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _CELL_BREAK.split(row)]


def _spell_out_table(table: str) -> str:
    """Turn each table row into a sentence of `header: cell` pairs.

    When the first header cell is empty, the first column holds row labels, and the label leads
    the sentence: "Price. Free: $0; Pro: $12 per seat per month."

    Only the line right after the header can be the rule line, so a data row of dashes is kept.
    """
    rows = [_cells(line) for line in table.splitlines()]
    if len(rows) > 1 and all(_TABLE_RULE.fullmatch(cell) for cell in rows[1]):
        del rows[1]
    if len(rows) < 2:
        return table
    header, body = rows[0], rows[1:]
    sentences = []
    for row in body:
        pairs = [f"{name}: {cell}" for name, cell in zip(header, row, strict=False) if name]
        label = "" if header[0] else f"{row[0]}. "
        sentences.append(f"{label}{'; '.join(pairs)}.")
    return "\n".join(sentences)


def _units(blocks: list[_Block]) -> list[_Unit]:
    """Join each lead-in to the block it introduces, and give tables their search form."""

    def search(block: _Block) -> str:
        return _spell_out_table(block.text) if block.kind == "table" else block.text

    units: list[_Unit] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        follower = blocks[i + 1] if i + 1 < len(blocks) else None
        if (
            block.kind == "prose"
            and block.text.endswith(":")
            and follower
            and follower.kind != "prose"
        ):
            units.append(
                _Unit(
                    f"{block.text}\n\n{follower.text}",
                    f"{block.text}\n\n{search(follower)}",
                    _PUBLIC_KIND[follower.kind],
                    _SPLIT[follower.kind],
                )
            )
            i += 2
        else:
            units.append(
                _Unit(block.text, search(block), _PUBLIC_KIND[block.kind], _SPLIT[block.kind])
            )
            i += 1
    return units


def _split_prose(text: str, target: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > target:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _split_items(text: str, target: int) -> list[str]:
    """Split a bulleted list between its top-level items into parts of about `target` characters.

    The top-level items are the bullets with the least indentation, so an indented list splits too.
    Lines before the first item, such as a lead-in, stay with the first item. An item is never cut,
    and a line inside a code block is never taken for an item.
    """
    lines = text.split("\n")
    bullets: list[tuple[int, int]] = []  # (line, indentation) of each bullet outside code blocks
    fence: str | None = None
    for i, line in enumerate(lines):
        if fence:
            if _closes(line, fence):
                fence = None
            continue
        opener = _FENCE.match(line)
        if opener:
            fence = opener.group(1)
            continue
        stripped = line.lstrip()
        if _BULLET.match(stripped):
            bullets.append((i, len(line) - len(stripped)))
    top = min((indent for _, indent in bullets), default=0)
    items = [i for i, indent in bullets if indent == top]
    bounds = [0, *items[1:], len(lines)]
    parts: list[str] = []
    start = 0
    for cut, end in pairwise(bounds[1:]):
        if len("\n".join(lines[start:end]).rstrip()) > target:
            parts.append("\n".join(lines[start:cut]).rstrip())
            start = cut
    parts.append("\n".join(lines[start:]).rstrip())
    return parts


def _pack(units: list[_Unit], target: int, limit: int) -> list[_Unit]:
    """Fill chunks up to `target` characters.

    Only a unit above `limit` is ever split, and only where its kind allows: a paragraph between
    sentences, a bulleted list between items. Any other unit stays whole, alone in its chunk when
    it is above `target`, so no chunk starts or ends inside a code block, table or procedure.
    """
    pieces: list[_Unit] = []
    for unit in units:
        if len(unit.text) <= limit or unit.split == "never":
            pieces.append(unit)
        elif unit.split == "sentences":
            pieces += [_Unit(part, part, unit.kind) for part in _split_prose(unit.text, target)]
        else:
            pieces += [_Unit(part, part, unit.kind) for part in _split_items(unit.text, target)]

    chunks: list[_Unit] = []
    group: list[_Unit] = []

    def flush() -> None:
        kinds = {unit.kind for unit in group}
        chunks.append(
            _Unit(
                "\n\n".join(unit.text for unit in group),
                "\n\n".join(unit.search for unit in group),
                kinds.pop() if len(kinds) == 1 else "mixed",
            )
        )
        group.clear()

    size = 0
    for piece in pieces:
        if group and size + 2 + len(piece.text) > target:
            flush()
            size = 0
        size += len(piece.text) + (2 if group else 0)
        group.append(piece)
    if group:
        flush()
    return chunks


def chunk_document(
    text: str,
    source: str = "<memory>",
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[Chunk]:
    """Split one knowledge base document into chunks. Raise `ChunkerError` rather than guess."""
    meta, body, first_line = _front_matter(text, source)
    doc_id, title = meta["id"], meta["title"]
    doc_tiers = _parse_tiers(meta["tiers"], f"{source}: front matter")
    chunks: list[Chunk] = []
    seen: dict[tuple[str, ...], int] = {}  # heading path -> line of its heading
    for section in _sections(body, first_line, doc_tiers, source):
        if section.path in seen:
            raise ChunkerError(
                f"{source}:{section.line}: two sections share the heading path {section.path},"
                f" the first one starts at line {seen[section.path]}"
            )
        seen[section.path] = section.line
        packed = _pack(_units(section.blocks), target_chars, max_chars)
        for ordinal, unit in enumerate(packed):
            embed_text = " > ".join((title, *section.path)) + "\n\n" + unit.search
            chunks.append(
                Chunk(
                    id=_chunk_id(doc_id, section.path, ordinal),
                    doc_id=doc_id,
                    doc_title=title,
                    heading_path=section.path,
                    ordinal=ordinal,
                    tiers=section.tiers,
                    kind=unit.kind,
                    content=unit.text,
                    embed_text=embed_text,
                    content_hash=_digest(f"{embed_text}|{unit.text}|{','.join(section.tiers)}"),
                )
            )
    return chunks


def load_kb(folder: Path) -> list[Chunk]:
    """Chunk every Markdown file in `folder`, in file name order.

    Every chunk ID is unique across the folder. A clash raises `ChunkerError`.
    """
    chunks: list[Chunk] = []
    doc_ids: dict[str, str] = {}
    by_id: dict[str, tuple[str, Chunk]] = {}  # chunk id -> (file name, chunk)
    for path in sorted(folder.glob("*.md")):
        doc_chunks = chunk_document(path.read_text(encoding="utf-8"), path.name)
        if not doc_chunks:
            raise ChunkerError(f"{path.name}: no content")
        doc_id = doc_chunks[0].doc_id
        if doc_id in doc_ids:
            raise ChunkerError(f"{path.name}: id {doc_id!r} is already used by {doc_ids[doc_id]}")
        doc_ids[doc_id] = path.name
        for chunk in doc_chunks:
            if chunk.id in by_id:
                other_file, other = by_id[chunk.id]
                raise ChunkerError(
                    f"{path.name}: chunk {chunk.ordinal} of {chunk.heading_path} gets the id"
                    f" {chunk.id}, already used by chunk {other.ordinal} of {other.heading_path}"
                    f" in {other_file}"
                )
            by_id[chunk.id] = (path.name, chunk)
        chunks += doc_chunks
    return chunks
