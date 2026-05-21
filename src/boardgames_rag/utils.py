"""Shared utilities: deterministic chunk IDs and rich-formatted logging."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import sys
import uuid

from rich.logging import RichHandler

# Project-specific UUID namespace (deterministic — derived from a fixed string).
INGEST_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "boardgames-rag.ingest")


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 (best-effort).

    Rulebook text and CLI output contain non-ASCII characters (em dashes,
    arrows, accented letters). Windows consoles default to cp1252, which
    raises UnicodeEncodeError on those; UTF-8 encodes all of them. Wrapped
    streams (e.g. pytest/Click capture buffers) may lack ``reconfigure`` or
    refuse it — both cases are tolerated.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging through rich and force UTF-8 console output."""
    _force_utf8_stdio()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
        force=True,
    )


def content_hash(text: str) -> str:
    """Hex SHA-256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    """Deterministic UUIDv5 for a chunk.

    Same ``(source_file, chunk_index, text)`` always yields the same UUID — so
    re-ingesting unchanged input is a no-op. Editing chunk text yields a new
    UUID, preserving the "upsert by content hash" semantics.
    """
    seed = f"{source_file}|{chunk_index}|{content_hash(text)}"
    return str(uuid.uuid5(INGEST_NAMESPACE, seed))
