from __future__ import annotations

import hashlib

import pytest

from supportagent.config import get_settings
from supportagent.kb.chunker import MAX_CHARS, ChunkerError, chunk_document, load_kb

DOC = """---
id: sample
title: Sample
tiers: free, pro, enterprise
---

# Sample

Intro paragraph.

## Limits

To reset the thing:

1. Open settings.
2. Select reset.
3. Confirm.

| Plan | Limit |
|---|---|
| Free | 1 |
| Pro | 2 |

## Paid
<!-- tiers: pro, enterprise -->

Paid text.

### Enterprise only
<!-- tiers: enterprise -->

Enterprise text.

### Inherits

Inherited text.
"""

TABLE = "| Plan | Limit |\n|---|---|\n| Free | 1 |\n| Pro | 2 |"
PROCEDURE = "To reset the thing:\n\n1. Open settings.\n2. Select reset.\n3. Confirm."

# The marker sits on line 12 of SECRET.
SECRET = """---
id: secret
title: Secret
tiers: free, pro, enterprise
---

# Secret

Tiers differ by plan.

## Enterprise secrets
<!-- tiers: enterprise -->

Discount formula: 40% off.
"""
MARK = "<!-- tiers: enterprise -->"


def by_path(chunks):
    return {c.heading_path: c for c in chunks}


def section(chunks, *path):
    return [c for c in chunks if c.heading_path == path]


def test_heading_path_leads_the_embed_text():
    chunk = by_path(chunk_document(DOC))[("Paid", "Enterprise only")]
    assert chunk.embed_text == "Sample > Paid > Enterprise only\n\nEnterprise text."
    assert chunk.content == "Enterprise text."


def test_tiers_inherit_from_the_parent_and_narrow_on_a_marker():
    chunks = by_path(chunk_document(DOC))
    assert chunks[()].tiers == ("free", "pro", "enterprise")
    assert chunks[("Paid",)].tiers == ("pro", "enterprise")
    assert chunks[("Paid", "Enterprise only")].tiers == ("enterprise",)
    assert chunks[("Paid", "Inherits")].tiers == ("pro", "enterprise")
    # A section after the narrowed one goes back to the document's tiers.
    assert chunks[("Limits",)].tiers == ("free", "pro", "enterprise")


def test_a_section_cannot_widen_its_parents_tiers():
    widened = DOC.replace("<!-- tiers: enterprise -->", "<!-- tiers: free -->")
    with pytest.raises(ValueError, match="widens"):
        chunk_document(widened)


@pytest.mark.parametrize(
    ("old", "new", "line"),
    [
        (MARK, "<!-- tier: enterprise -->", 12),
        (MARK, "<!-- Tiers: enterprise -->", 12),
        (MARK, "<!-- TIERS: enterprise -->", 12),
        (MARK, "<!-- tiers enterprise -->", 12),
        (MARK, "<!-- tiers: Enterprise -->", 12),
        (MARK, "<!-- tiers: platinum -->", 12),
        (MARK, "<!-- tiers: -->", 12),
        (MARK, "<!-- note --> <!-- tiers: enterprise -->", 12),
        (MARK, "<!--\ntiers: enterprise\n-->", 13),
        (MARK, f"{MARK}\n{MARK}", 13),
        (MARK, f"Early text.\n{MARK}", 13),
        ("# Secret\n", f"# Secret\n{MARK}\n", 8),
        ("---\n\n# Secret", f"---\n{MARK}\n# Secret", 6),
    ],
)
def test_a_plan_marker_that_cannot_be_applied_fails_loudly(old, new, line):
    doc = SECRET.replace(old, new)
    assert doc != SECRET
    with pytest.raises(ChunkerError, match=rf"^secret\.md:{line}: "):
        chunk_document(doc, "secret.md")


def test_valid_plan_markers_apply_and_code_is_not_a_marker():
    after_comment = SECRET.replace(MARK, f"<!-- author note: updated Q4 -->\n{MARK}")
    chunks = by_path(chunk_document(after_comment))
    assert chunks[("Enterprise secrets",)].tiers == ("enterprise",)
    assert chunks[()].content == "Tiers differ by plan."

    code = "```html\n<!-- tier: enterprise -->\n```"
    chunks = by_path(chunk_document(SECRET.replace("Tiers differ by plan.", code)))
    assert chunks[()].content == code
    assert issubclass(ChunkerError, ValueError)


def test_tables_and_procedures_survive_a_tiny_target():
    contents = [c.content for c in chunk_document(DOC, target_chars=30, max_chars=60)]
    assert TABLE in contents
    assert PROCEDURE in contents  # the lead-in stayed with its list
    for content in contents:
        if "|" in content:
            assert content == TABLE


def test_a_chunk_never_starts_or_ends_inside_a_code_block():
    code = "\n".join(f"line_{i} = 'This is sentence {i}. It has punctuation.'" for i in range(60))
    script = f"Run this script:\n\n```python\n{code}\n```"
    assert len(script) > MAX_CHARS
    doc = DOC.replace("Paid text.", f"Paid text.\n\n{script}\n\nAfter the script.")
    paid = section(chunk_document(doc), "Paid")
    assert [c.content for c in paid] == ["Paid text.", script, "After the script."]

    unclosed = DOC.replace("Paid text.", "```\nPaid text.")
    with pytest.raises(ChunkerError, match="never closed"):
        chunk_document(unclosed)


def test_a_table_is_markdown_for_display_and_sentences_for_search():
    table = next(c for c in chunk_document(DOC, target_chars=30, max_chars=60) if c.kind == "table")
    assert table.content == TABLE
    assert table.embed_text == "Sample > Limits\n\nPlan: Free; Limit: 1.\nPlan: Pro; Limit: 2."


def test_a_table_with_row_labels_leads_each_sentence_with_the_label():
    doc = DOC.replace(
        TABLE, "| | Free | Pro |\n|---|---|---|\n| Price | $0 | $12 |\n| Seats | 5 | 100 |"
    )
    limits = by_path(chunk_document(doc))[("Limits",)]
    assert limits.embed_text.endswith("Price. Free: $0; Pro: $12.\nSeats. Free: 5; Pro: 100.")
    assert "| Price | $0 | $12 |" in limits.content


@pytest.mark.parametrize(
    ("table", "spelled"),
    [
        ("|| Free | Pro |\n|---|---|---|\n| Price | $0 | $12 |", "Price. Free: $0; Pro: $12."),
        (
            "| Feature | Free | Pro |\n|---|---|---|\n| - | - | - |\n| Audit | - | 30d |",
            "Feature: -; Free: -; Pro: -.\nFeature: Audit; Free: -; Pro: 30d.",
        ),
        ("| Filter | Syntax |\n|---|---|\n| OR | `a \\| b` |", "Filter: OR; Syntax: `a | b`."),
    ],
)
def test_table_cells_stay_in_their_columns(table, spelled):
    doc = f"---\nid: t\ntitle: T\ntiers: free\n---\n\n# T\n\n## Table\n\n{table}\n"
    (chunk,) = chunk_document(doc)
    assert chunk.content == table
    assert chunk.embed_text == f"T > Table\n\n{spelled}"


def test_a_loose_list_stays_together_and_splits_only_between_items():
    steps = (
        "1. First step does something detailed.\n\n"
        "2. Second step continues the procedure.\n\n"
        "3. Third step finishes the procedure."
    )
    doc = DOC.replace("Paid text.", f"Paid text.\n\n{steps}")
    paid = section(chunk_document(doc, target_chars=50, max_chars=80), "Paid")
    assert [(c.content, c.kind) for c in paid] == [("Paid text.", "prose"), (steps, "procedure")]

    items = [f"- Item {n} has a few words of text." for n in range(6)]
    bullets = "Remember:\n\n" + "\n\n".join(items)
    doc = DOC.replace("Paid text.", bullets)
    chunks = chunk_document(doc, target_chars=80, max_chars=120)
    parts = [c.content for c in section(chunks, "Paid")]
    assert parts == [
        f"Remember:\n\n{items[0]}\n\n{items[1]}",
        f"{items[2]}\n\n{items[3]}",
        f"{items[4]}\n\n{items[5]}",
    ]


def test_kind_names_what_the_chunk_holds():
    small = {c.content: c.kind for c in chunk_document(DOC, target_chars=30, max_chars=60)}
    assert small[TABLE] == "table"
    assert small[PROCEDURE] == "procedure"
    assert small["Paid text."] == "prose"
    assert by_path(chunk_document(DOC))[("Limits",)].kind == "mixed"


def test_ids_survive_an_edit_and_the_hash_tracks_it():
    before = by_path(chunk_document(DOC))
    after = by_path(chunk_document(DOC.replace("Paid text.", "Paid text, revised.")))
    assert {p: c.id for p, c in before.items()} == {p: c.id for p, c in after.items()}
    changed = {p for p in before if before[p].content_hash != after[p].content_hash}
    assert changed == {("Paid",)}


def test_a_tier_change_alone_changes_the_hash():
    narrowed = DOC.replace("<!-- tiers: pro, enterprise -->", "<!-- tiers: pro -->").replace(
        "<!-- tiers: enterprise -->", "<!-- tiers: pro -->"
    )
    before, after = by_path(chunk_document(DOC)), by_path(chunk_document(narrowed))
    assert before[("Paid",)].content == after[("Paid",)].content
    assert before[("Paid",)].content_hash != after[("Paid",)].content_hash


def test_a_long_paragraph_splits_between_sentences():
    sentences = [f"Sentence number {n} has a few words in it." for n in range(40)]
    doc = DOC.replace("Intro paragraph.", " ".join(sentences))
    chunks = chunk_document(doc, target_chars=200, max_chars=300)
    parts = [c.content for c in chunks if not c.heading_path]
    assert len(parts) > 5
    assert all(len(p) <= 200 and p.endswith(".") for p in parts)
    assert " ".join(parts) == " ".join(sentences)


def test_two_sections_with_the_same_path_are_rejected():
    with pytest.raises(ValueError, match="share the heading path"):
        chunk_document(DOC + "\n## Limits\n\nAgain.\n")


def test_a_slash_in_a_heading_does_not_clash_with_a_nested_path():
    doc = """---
id: slash
title: Slash
tiers: free, pro, enterprise
---

# Slash

## CI/CD

Pipeline docs.

## CI

Intro.

### CD

Nested docs.

## 24/7 Support

Around the clock.
"""
    chunks = chunk_document(doc)
    paths = [c.heading_path for c in chunks]
    assert paths == [("CI/CD",), ("CI",), ("CI", "CD"), ("24/7 Support",)]
    assert len({c.id for c in chunks}) == len(chunks)
    # A path without `/`, `|` or a backslash keeps the ID it always had.
    chunk = by_path(chunk_document(DOC))[("Paid", "Enterprise only")]
    assert chunk.id == hashlib.sha1(b"sample|Paid/Enterprise only|0").hexdigest()[:16]


def test_load_kb_rejects_two_chunks_with_one_id(tmp_path, monkeypatch):
    for doc_id in ("first", "second"):
        (tmp_path / f"{doc_id}.md").write_text(DOC.replace("id: sample", f"id: {doc_id}"))
    # Leave the document out of the key, so the same heading path clashes across documents.
    monkeypatch.setattr(
        "supportagent.kb.chunker._chunk_id",
        lambda doc_id, path, ordinal: f"{'/'.join(path)}|{ordinal}",
    )
    with pytest.raises(ChunkerError, match=r"^second\.md: .* already used .* in first\.md$"):
        load_kb(tmp_path)


def test_headings_inside_code_fences_are_not_headings():
    doc = DOC.replace("Intro paragraph.", "```\n# not a heading\n\n## nor this\n```")
    intro = by_path(chunk_document(doc))[()]
    assert intro.content == "```\n# not a heading\n\n## nor this\n```"


def test_an_indented_fence_keeps_its_section_and_its_plans():
    doc = (
        "---\nid: d4\ntitle: D4\ntiers: free, pro, enterprise\n---\n\n# D4\n\n"
        "## Enterprise Proc\n<!-- tiers: enterprise -->\n\n"
        "1. Run this:\n\n   ```bash\n# shell comment\necho ok\n   ```\n\n2. Done.\n"
    )
    chunks = chunk_document(doc, "d4.md")
    assert [(c.heading_path, c.tiers, c.content) for c in chunks] == [
        (
            ("Enterprise Proc",),
            ("enterprise",),
            "1. Run this:\n\n   ```bash\n# shell comment\necho ok\n   ```\n\n2. Done.",
        ),
    ]

    # A shorter fence does not close a longer one.
    tildes = "  ~~~~\n## not a heading\n~~~\n ~~~~~"
    intro = by_path(chunk_document(DOC.replace("Intro paragraph.", tildes)))[()]
    assert intro.content == tildes


@pytest.mark.parametrize(
    "marker",
    [
        "<!-- plan_tiers: enterprise -->",
        "<!-- tier_list: enterprise -->",
        "<!-- mytiers: enterprise -->",
    ],
)
def test_a_misspelled_plan_marker_fails_loudly(marker):
    doc = (
        "---\nid: d2\ntitle: D2\ntiers: free, pro, enterprise\n---\n\n# D2\n\n## Secret\n"
        f"{marker}\n\nSecret Enterprise info.\n"
    )
    with pytest.raises(ChunkerError, match=r"^d2\.md:10: "):
        chunk_document(doc, "d2.md")


def test_text_between_two_comments_on_one_line_is_kept():
    doc = (
        "---\nid: d1\ntitle: D1\ntiers: free\n---\n\n# D1\n\n## Sec\n\n"
        "<!-- a --> CRITICAL POLICY TEXT MUST NOT DISAPPEAR <!-- b -->\n"
    )
    chunks = chunk_document(doc, "d1.md")
    assert [(c.heading_path, c.content) for c in chunks] == [
        (("Sec",), "CRITICAL POLICY TEXT MUST NOT DISAPPEAR"),
    ]
    assert "<!--" not in chunks[0].embed_text


def test_a_comment_over_several_lines_vanishes_with_the_heading_inside_it():
    doc = (
        "---\nid: d3\ntitle: D3\ntiers: free\n---\n\n# D3\n\n## Real\n\n"
        "<!--\n## Draft Secret Heading\nInternal author note.\n-->\n\nVisible text.\n"
    )
    chunks = chunk_document(doc, "d3.md")
    assert [(c.heading_path, c.content) for c in chunks] == [(("Real",), "Visible text.")]

    draft = (
        "<!-- note 1 --> Support can approve refunds up to $500 within 14 days. <!-- note 2 -->\n\n"
        "<!--\nINTERNAL DRAFT DO NOT PUBLISH: raising Pro price to $29 next quarter\n-->\n\n"
        "Keep this line."
    )
    paid = section(chunk_document(DOC.replace("Paid text.", draft)), "Paid")
    assert [c.content for c in paid] == [
        "Support can approve refunds up to $500 within 14 days.\n\nKeep this line.",
    ]


def test_a_comment_over_several_lines_that_mentions_tiers_fails_loudly():
    note = "<!--\nDraft: move this section to the Enterprise tier next quarter.\n-->"
    doc = SECRET.replace("Discount formula: 40% off.", f"Discount formula: 40% off.\n\n{note}")
    with pytest.raises(ChunkerError, match=r"^secret\.md:17: "):
        chunk_document(doc, "secret.md")


def test_a_comment_that_is_never_closed_fails_loudly():
    doc = SECRET.replace("Discount formula", "<!-- draft\nDiscount formula")
    with pytest.raises(ChunkerError, match=r"^secret\.md:14: this HTML comment is never closed"):
        chunk_document(doc, "secret.md")


def test_a_long_indented_list_splits_between_items():
    items = "\n".join(
        f"  - Indented bullet item {i} with enough text to easily exceed the size limit."
        for i in range(40)
    )
    doc = f"---\nid: d5\ntitle: D5\ntiers: free\n---\n\n# D5\n\n## Bullets\n\n{items}\n"
    chunks = chunk_document(doc, "d5.md")
    assert len(items) > MAX_CHARS
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHARS for c in chunks)
    assert all(c.content.startswith("  - Indented bullet item") for c in chunks)
    assert "\n".join(c.content for c in chunks) == items


def test_a_code_block_inside_a_long_list_item_is_not_split():
    config = "  ```yaml\n- name: not an item\n\n- name: nor this\n  ```"
    items = [
        f"  - Item {n} has enough words to make the whole list longer than the limit."
        for n in range(40)
    ]
    items[20] = f"  - Item 20 uses this config:\n\n{config}"
    bullets = "\n".join(items)
    doc = f"---\nid: d6\ntitle: D6\ntiers: free\n---\n\n# D6\n\n## Bullets\n\n{bullets}\n"
    chunks = chunk_document(doc, "d6.md")
    assert len(bullets) > MAX_CHARS
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHARS for c in chunks)
    assert any(items[20] in c.content for c in chunks)
    assert all(c.content.count("```") % 2 == 0 for c in chunks)
    assert "\n".join(c.content for c in chunks) == bullets


def test_the_real_knowledge_base_loads():
    chunks = load_kb(get_settings().kb_dir)
    assert len({c.doc_id for c in chunks}) == 16
    assert len({c.id for c in chunks}) == len(chunks)
    oversized_prose = [c for c in chunks if c.kind == "prose" and len(c.content) > MAX_CHARS]
    assert oversized_prose == []
