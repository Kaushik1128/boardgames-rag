"""Shared utilities: deterministic chunk IDs and rich-formatted logging."""

from __future__ import annotations

import hashlib
import logging
import uuid

from rich.logging import RichHandler

# Project-specific UUID namespace (deterministic — derived from a fixed string).
INGEST_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "boardgames-rag.ingest")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to render through rich."""
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
