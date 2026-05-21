"""Tests for shared utilities — hashing, stable IDs, and stdio setup."""

from __future__ import annotations

import sys
import uuid

from boardgames_rag.utils import _force_utf8_stdio, content_hash, stable_chunk_id


class TestContentHash:
    def test_is_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_differs_for_different_input(self):
        assert content_hash("alpha") != content_hash("bravo")

    def test_is_sha256_hex(self):
        digest = content_hash("x")
        assert len(digest) == 64
        int(digest, 16)  # parses as hex


class TestStableChunkId:
    def test_is_deterministic(self):
        a = stable_chunk_id("rules.pdf", 0, "chunk text")
        b = stable_chunk_id("rules.pdf", 0, "chunk text")
        assert a == b

    def test_changes_when_text_changes(self):
        a = stable_chunk_id("rules.pdf", 0, "text one")
        b = stable_chunk_id("rules.pdf", 0, "text two")
        assert a != b

    def test_changes_when_index_changes(self):
        a = stable_chunk_id("rules.pdf", 0, "text")
        b = stable_chunk_id("rules.pdf", 1, "text")
        assert a != b

    def test_changes_when_source_changes(self):
        a = stable_chunk_id("catan.pdf", 0, "text")
        b = stable_chunk_id("azul.pdf", 0, "text")
        assert a != b

    def test_is_a_valid_uuid_string(self):
        uuid.UUID(stable_chunk_id("rules.pdf", 0, "text"))  # raises if invalid


class TestForceUtf8Stdio:
    def test_reconfigures_both_streams_to_utf8(self, monkeypatch):
        calls: list[dict] = []

        class FakeStream:
            def reconfigure(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(sys, "stdout", FakeStream())
        monkeypatch.setattr(sys, "stderr", FakeStream())
        _force_utf8_stdio()
        assert calls == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]

    def test_tolerates_stream_without_reconfigure(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", object())
        monkeypatch.setattr(sys, "stderr", object())
        _force_utf8_stdio()  # must not raise

    def test_tolerates_reconfigure_failure(self, monkeypatch):
        class FailingStream:
            def reconfigure(self, **kwargs):
                raise ValueError("buffered data")

        monkeypatch.setattr(sys, "stdout", FailingStream())
        monkeypatch.setattr(sys, "stderr", FailingStream())
        _force_utf8_stdio()  # must not raise
