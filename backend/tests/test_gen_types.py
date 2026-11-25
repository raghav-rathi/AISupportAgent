from __future__ import annotations

from supportagent.cli.gen_types import DEFAULT_PATH, render


def test_the_checked_in_typescript_matches_the_models():
    assert DEFAULT_PATH.read_text() == render(), "run: python -m supportagent.cli.gen_types"


def test_types_are_rendered_the_way_the_wire_carries_them():
    text = render()
    # A literal becomes the discriminator, a nullable field keeps its null, a list keeps its type.
    assert '  type: "tool.finished";' in text
    assert '  outcome: "ok" | "refused" | "failed";' in text
    assert "  top_score: number | null;" in text
    assert "  hits: RetrievalHit[];" in text
    assert "  args: Record<string, unknown>;" in text
    # The server always sends every field. A client may leave out the ones with defaults.
    assert "  seq: number;" in text
    assert "  session_id?: string | null;" in text
    assert "  | ToolFinished" in text and "  | UserMessage" in text
