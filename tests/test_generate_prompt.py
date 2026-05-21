"""Pure-function tests for prompt construction."""

from __future__ import annotations

from boardgames_rag.generate import NO_ANSWER, build_prompt, format_context
from boardgames_rag.retrieve import RetrievedChunk


def _chunk(
    pid: str,
    text: str,
    *,
    source: str = "catan.pdf",
    heading: str | None = "Trading",
    **payload_extra: object,
) -> RetrievedChunk:
    payload: dict[str, object] = {"text": text, "source_file": source}
    if heading is not None:
        payload["heading"] = heading
    payload.update(payload_extra)
    return RetrievedChunk(point_id=pid, score=1.0, payload=payload)


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_single_chunk_rendered(self):
        ctx = format_context([_chunk("a", "Trade 2:1 at a port.")])
        assert "[1]" in ctx
        assert "catan.pdf" in ctx
        assert "Trading" in ctx
        assert "Trade 2:1 at a port." in ctx

    def test_multiple_chunks_numbered_in_order(self):
        chunks = [_chunk("a", "first"), _chunk("b", "second"), _chunk("c", "third")]
        ctx = format_context(chunks)
        assert ctx.index("[1]") < ctx.index("[2]") < ctx.index("[3]")

    def test_empty_list_returns_empty_string(self):
        assert format_context([]) == ""

    def test_missing_source_falls_back_to_unknown(self):
        chunk = RetrievedChunk(point_id="a", score=1.0, payload={"text": "body"})
        assert "unknown" in format_context([chunk])

    def test_parent_heading_used_when_heading_absent(self):
        chunk = RetrievedChunk(
            point_id="a",
            score=1.0,
            payload={"text": "b", "source_file": "x.pdf", "parent_heading": "Setup"},
        )
        assert "Setup" in format_context([chunk])

    def test_missing_heading_renders_dash(self):
        chunk = RetrievedChunk(
            point_id="a", score=1.0, payload={"text": "b", "source_file": "x.pdf"}
        )
        assert "—" in format_context([chunk])


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_returns_system_then_user_message(self):
        msgs = build_prompt("How do I trade?", [_chunk("a", "trade text")])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_instructs_grounded_refusal(self):
        msgs = build_prompt("q", [_chunk("a", "x")])
        assert NO_ANSWER in msgs[0]["content"]

    def test_user_message_contains_question(self):
        msgs = build_prompt("How do I win Azul?", [_chunk("a", "scoring")])
        assert "How do I win Azul?" in msgs[1]["content"]

    def test_user_message_contains_context_text(self):
        msgs = build_prompt("q", [_chunk("a", "UNIQUE_CONTEXT_MARKER")])
        assert "UNIQUE_CONTEXT_MARKER" in msgs[1]["content"]

    def test_empty_chunks_yields_no_passages_marker(self):
        msgs = build_prompt("q", [])
        assert "no relevant passages found" in msgs[1]["content"]
