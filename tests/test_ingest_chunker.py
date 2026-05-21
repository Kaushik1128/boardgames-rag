"""Tests for the heading-aware chunker and document parsers."""

from __future__ import annotations

import itertools

import pytest

from boardgames_rag.ingest import (
    Section,
    chunk_section_text,
    load_file_sections,
    pack_sections,
    parse_markdown_into_sections,
    parse_mtg_text_into_sections,
)

# ---------------------------------------------------------------------------
# chunk_section_text — single-section sliding window
# ---------------------------------------------------------------------------


class TestChunkSectionText:
    def test_short_text_produces_one_chunk(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "short.txt").read_text(encoding="utf-8")
        chunks = chunk_section_text(
            text,
            heading="Intro",
            target_tokens=128,
            overlap_tokens=20,
            tokenizer=tokenizer,
        )
        assert len(chunks) == 1
        assert chunks[0].heading == "Intro"
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_no_heading_section_produces_chunk_with_none_heading(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "no_headings.txt").read_text(encoding="utf-8")
        chunks = chunk_section_text(
            text,
            heading=None,
            target_tokens=512,
            overlap_tokens=50,
            tokenizer=tokenizer,
        )
        assert len(chunks) == 1
        assert chunks[0].heading is None

    def test_oversized_text_splits_within_target_tokens(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        target = 128
        overlap = 32
        chunks = chunk_section_text(
            text,
            heading="Big Paragraph",
            target_tokens=target,
            overlap_tokens=overlap,
            tokenizer=tokenizer,
        )
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(tokenizer.encode(chunk.text)) <= target
        assert all(c.heading == "Big Paragraph" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_oversized_has_token_overlap_between_neighbors(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        target = 128
        overlap = 32
        chunks = chunk_section_text(
            text,
            heading="X",
            target_tokens=target,
            overlap_tokens=overlap,
            tokenizer=tokenizer,
        )
        assert len(chunks) >= 2
        # Allow small drift from BPE-boundary effects on re-encoding the
        # decoded slice. Most ASCII English round-trips cleanly.
        min_shared = overlap - 4
        for a, b in itertools.pairwise(chunks):
            a_tokens = tokenizer.encode(a.text)
            b_tokens = tokenizer.encode(b.text)
            shared = 0
            for k in range(overlap, min_shared - 1, -1):
                if a_tokens[-k:] == b_tokens[:k]:
                    shared = k
                    break
            assert shared >= min_shared, (
                f"Only {shared} overlapping tokens between adjacent chunks; "
                f"expected at least {min_shared}"
            )

    def test_first_and_last_chunks_anchor_full_text(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        chunks = chunk_section_text(
            text,
            heading="X",
            target_tokens=128,
            overlap_tokens=32,
            tokenizer=tokenizer,
        )
        assert text.lstrip()[:40] in chunks[0].text
        assert text.rstrip()[-40:] in chunks[-1].text

    def test_invalid_overlap_raises(self, tokenizer):
        with pytest.raises(ValueError):
            chunk_section_text(
                "hello",
                heading=None,
                target_tokens=50,
                overlap_tokens=50,
                tokenizer=tokenizer,
            )

    def test_starting_index_offsets_chunk_indices(self, tokenizer, fixtures_dir):
        text = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        chunks = chunk_section_text(
            text,
            heading="X",
            target_tokens=128,
            overlap_tokens=32,
            tokenizer=tokenizer,
            starting_index=7,
        )
        assert chunks[0].chunk_index == 7
        assert chunks[-1].chunk_index == 7 + len(chunks) - 1


# ---------------------------------------------------------------------------
# pack_sections — sibling packing & parent boundaries
# ---------------------------------------------------------------------------


class TestPackSections:
    def test_siblings_with_same_parent_pack_together(self, tokenizer):
        sections = [
            Section(heading="A.1", parent_heading="A", text="alpha " * 10),
            Section(heading="A.2", parent_heading="A", text="beta " * 10),
            Section(heading="A.3", parent_heading="A", text="gamma " * 10),
        ]
        chunks = pack_sections(sections, target_tokens=512, overlap_tokens=50, tokenizer=tokenizer)
        assert len(chunks) == 1
        assert chunks[0].heading == "A.1"
        assert chunks[0].parent_heading == "A"
        assert "alpha" in chunks[0].text
        assert "beta" in chunks[0].text
        assert "gamma" in chunks[0].text

    def test_parent_boundary_forces_new_chunk(self, tokenizer):
        sections = [
            Section(heading="A.1", parent_heading="A", text="alpha " * 10),
            Section(heading="B.1", parent_heading="B", text="bravo " * 10),
        ]
        chunks = pack_sections(sections, target_tokens=512, overlap_tokens=50, tokenizer=tokenizer)
        assert len(chunks) == 2
        assert chunks[0].parent_heading == "A"
        assert chunks[1].parent_heading == "B"
        assert [c.chunk_index for c in chunks] == [0, 1]

    def test_oversized_single_section_splits_with_overlap(self, tokenizer, fixtures_dir):
        big = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        sections = [Section(heading="Huge", parent_heading="Root", text=big)]
        chunks = pack_sections(sections, target_tokens=128, overlap_tokens=32, tokenizer=tokenizer)
        assert len(chunks) >= 2
        assert all(c.heading == "Huge" for c in chunks)
        assert all(c.parent_heading == "Root" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_overflow_within_same_parent_starts_new_chunk(self, tokenizer):
        # Two sections under the same parent that together exceed target.
        # Drive target off measured token counts so the test isn't sensitive
        # to BPE quirks of any particular word.
        s1_text = " ".join(["alpha"] * 30)
        s2_text = " ".join(["alpha"] * 30)
        t1 = len(tokenizer.encode(s1_text))
        t2 = len(tokenizer.encode(s2_text))
        target = t1 + t2 - 5  # each fits alone; combined overflows
        assert t1 < target and t2 < target and t1 + t2 > target
        chunks = pack_sections(
            [
                Section(heading="P.1", parent_heading="P", text=s1_text),
                Section(heading="P.2", parent_heading="P", text=s2_text),
            ],
            target_tokens=target,
            overlap_tokens=10,
            tokenizer=tokenizer,
        )
        assert len(chunks) == 2
        assert all(c.parent_heading == "P" for c in chunks)
        assert chunks[0].heading == "P.1"
        assert chunks[1].heading == "P.2"

    def test_empty_input_returns_empty(self, tokenizer):
        assert pack_sections([], target_tokens=512, overlap_tokens=50, tokenizer=tokenizer) == []

    def test_packed_chunk_index_continues_after_oversized_split(self, tokenizer, fixtures_dir):
        big = (fixtures_dir / "oversized_paragraph.txt").read_text(encoding="utf-8")
        sections = [
            Section(heading="A.1", parent_heading="A", text="alpha " * 5),
            Section(heading="Huge", parent_heading="A", text=big),
            Section(heading="A.3", parent_heading="A", text="gamma " * 5),
        ]
        chunks = pack_sections(sections, target_tokens=128, overlap_tokens=32, tokenizer=tokenizer)
        # Indices remain dense and monotonically increasing.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# parse_markdown_into_sections
# ---------------------------------------------------------------------------


class TestParseMarkdown:
    def test_basic_three_section_doc(self):
        md = "# Title\n\nIntro.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
        sections = parse_markdown_into_sections(md)
        assert [s.heading for s in sections] == ["Title", "Section A", "Section B"]
        assert sections[0].parent_heading is None
        assert sections[1].parent_heading == "Title"
        assert sections[2].parent_heading == "Title"

    def test_pre_heading_content_kept_as_no_heading_section(self):
        md = "Preamble text without a heading.\n\n# H\n\nBody."
        sections = parse_markdown_into_sections(md)
        assert sections[0].heading is None
        assert "Preamble" in sections[0].text
        assert sections[1].heading == "H"

    def test_deeper_nesting_tracks_parents(self):
        md = "# A\nbody a\n## A.1\nbody a.1\n### A.1.1\nbody a.1.1\n## A.2\nbody a.2\n"
        sections = parse_markdown_into_sections(md)
        assert [s.heading for s in sections] == ["A", "A.1", "A.1.1", "A.2"]
        assert [s.parent_heading for s in sections] == [None, "A", "A.1", "A"]

    def test_empty_document_returns_empty(self):
        assert parse_markdown_into_sections("") == []

    def test_strips_markdown_emphasis_from_headings(self):
        # pymupdf4llm renders bold PDF headings as "# **HEADING**".
        md = "# **CITIES**\n\nbody one\n\n## *Setup*\n\nbody two\n"
        sections = parse_markdown_into_sections(md)
        headings = [s.heading for s in sections]
        assert "CITIES" in headings
        assert "Setup" in headings
        assert all("*" not in (h or "") for h in headings)


# ---------------------------------------------------------------------------
# parse_mtg_text_into_sections
# ---------------------------------------------------------------------------


class TestParseMtg:
    def test_subsection_used_as_heading_with_top_as_parent(self):
        text = (
            "100. General\n"
            "\n"
            "100.1. These Magic rules apply to any game with two or more players.\n"
            "\n"
            "100.2. There are different mode rules.\n"
            "\n"
            "200. Game Concepts\n"
            "\n"
            "200.1. There are many rules.\n"
        )
        sections = parse_mtg_text_into_sections(text)
        headings = [s.heading for s in sections]
        parents = [s.parent_heading for s in sections]
        assert "100.1" in headings
        assert "100.2" in headings
        assert "200.1" in headings
        # parent for 100.x is "100. General"
        assert parents[headings.index("100.1")] == "100. General"
        assert parents[headings.index("200.1")] == "200. Game Concepts"

    def test_sub_subsection_lines_stay_inside_parent_subsection(self):
        text = (
            "100. General\n"
            "\n"
            "100.1. Main subsection.\n"
            "100.1a. Detailed sub-sub.\n"
            "100.1b. Another sub-sub.\n"
        )
        sections = parse_mtg_text_into_sections(text)
        body = next(s for s in sections if s.heading == "100.1").text
        assert "100.1a" in body
        assert "100.1b" in body


# ---------------------------------------------------------------------------
# load_file_sections — extension dispatch
# ---------------------------------------------------------------------------


class TestLoadFileSections:
    def test_unknown_extension_returns_none(self, tmp_path):
        p = tmp_path / "rules.docx"
        p.write_text("anything")
        assert load_file_sections(p) is None

    def test_txt_dispatches_to_mtg_parser(self, tmp_path):
        p = tmp_path / "mtg.txt"
        p.write_text("100. General\n\n100.1. Test content.\n", encoding="utf-8")
        sections = load_file_sections(p)
        assert sections is not None
        assert any(s.heading == "100.1" for s in sections)
