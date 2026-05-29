"""Ingestion pipeline: PDF + plain-text → heading-aware chunks → embeddings → Qdrant.

CLI:

.. code-block:: bash

    python -m boardgames_rag.ingest --source-dir ./data/raw --collection boardgames
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import tiktoken
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from boardgames_rag.config import Settings, load_settings
from boardgames_rag.utils import setup_logging, stable_chunk_id

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)
console = Console()


__all__ = [
    "Chunk",
    "Embedder",
    "QdrantStore",
    "Section",
    "chunk_section_text",
    "ingest_directory",
    "load_file_sections",
    "main",
    "pack_sections",
    "parse_markdown_into_sections",
    "parse_mtg_text_into_sections",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A heading-keyed slice of a document, before packing/splitting."""

    heading: str | None
    parent_heading: str | None
    text: str
    page: int | None = None


@dataclass
class Chunk:
    """A retrieval-sized unit derived from one or more Sections."""

    heading: str | None
    parent_heading: str | None
    text: str
    chunk_index: int
    page: int | None = None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_section_text(
    text: str,
    *,
    heading: str | None,
    parent_heading: str | None = None,
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: tiktoken.Encoding,
    starting_index: int = 0,
    page: int | None = None,
) -> list[Chunk]:
    """Sliding-window chunk a single section's text with token-level overlap."""
    if target_tokens <= overlap_tokens:
        raise ValueError(
            f"target_tokens ({target_tokens}) must exceed overlap_tokens ({overlap_tokens})"
        )

    tokens = tokenizer.encode(text)
    if len(tokens) <= target_tokens:
        return [
            Chunk(
                heading=heading,
                parent_heading=parent_heading,
                text=text,
                chunk_index=starting_index,
                page=page,
            )
        ]

    chunks: list[Chunk] = []
    step = target_tokens - overlap_tokens
    i = 0
    idx = starting_index
    while True:
        end = min(i + target_tokens, len(tokens))
        sub_text = tokenizer.decode(tokens[i:end])
        chunks.append(
            Chunk(
                heading=heading,
                parent_heading=parent_heading,
                text=sub_text,
                chunk_index=idx,
                page=page,
            )
        )
        idx += 1
        if end >= len(tokens):
            break
        i += step
    return chunks


def pack_sections(
    sections: list[Section],
    *,
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[Chunk]:
    """Pack sibling sections (same parent_heading) up to ``target_tokens``.

    A single section larger than ``target_tokens`` is split with sliding-window
    overlap. Packed multi-section chunks themselves do not overlap — overlap
    only applies inside an oversized split.
    """
    chunks: list[Chunk] = []
    next_idx = 0
    buf: list[Section] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, next_idx
        if not buf:
            return
        first = buf[0]
        merged_text = "\n\n".join(s.text for s in buf if s.text)
        chunks.append(
            Chunk(
                heading=first.heading,
                parent_heading=first.parent_heading,
                text=merged_text,
                chunk_index=next_idx,
                page=first.page,
            )
        )
        next_idx += 1
        buf = []
        buf_tokens = 0

    for section in sections:
        section_tokens = len(tokenizer.encode(section.text))

        if section_tokens > target_tokens:
            flush()
            sub_chunks = chunk_section_text(
                section.text,
                heading=section.heading,
                parent_heading=section.parent_heading,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tokenizer,
                starting_index=next_idx,
                page=section.page,
            )
            chunks.extend(sub_chunks)
            next_idx += len(sub_chunks)
            continue

        parent_changed = bool(buf) and buf[0].parent_heading != section.parent_heading
        would_overflow = buf_tokens + section_tokens > target_tokens
        if parent_changed or (would_overflow and buf):
            flush()

        buf.append(section)
        buf_tokens += section_tokens

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Inline markdown styling markers that pymupdf4llm leaves embedded in heading
# text: bold/italic (`*`, `_`), code (backtick), strikethrough (`~`). E.g. it
# renders styled PDF headings as "# **CITIES**" or "# ~~LOAN ACTION~~".
_MD_EMPHASIS_RE = re.compile(r"[*_~`]+")


def _clean_heading(raw: str) -> str:
    """Strip markdown emphasis markers from heading text.

    Without this, literal asterisks leak into a chunk's ``heading`` metadata
    (``**CITIES**`` instead of ``CITIES``). Heading text is not part of the
    chunk's stable ID, so re-ingesting after this change updates payloads in
    place without creating new points.
    """
    return _MD_EMPHASIS_RE.sub("", raw).strip()


def parse_markdown_into_sections(md: str) -> list[Section]:
    """Parse markdown text into a flat list of Sections.

    Detects ATX-style ``#``..``######`` headings. ``parent_heading`` for each
    section is the nearest higher-level heading still in scope at parse time.
    """
    sections: list[Section] = []
    headings_by_level: dict[int, str] = {}
    current_heading: str | None = None
    current_parent: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if body or current_heading is not None:
            sections.append(
                Section(heading=current_heading, parent_heading=current_parent, text=body)
            )
        buf = []

    for line in md.splitlines():
        m = _MD_HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            new_heading = _clean_heading(m.group(2))
            for lvl in [lv for lv in headings_by_level if lv >= level]:
                del headings_by_level[lvl]
            higher = [lv for lv in headings_by_level if lv < level]
            current_parent = headings_by_level[max(higher)] if higher else None
            current_heading = new_heading
            headings_by_level[level] = new_heading
        else:
            buf.append(line)
    flush()
    return sections


_MTG_TOP_RE = re.compile(r"^(\d{3})\.\s+(.+?)\s*$")
_MTG_SUB_RE = re.compile(r"^(\d{3}\.\d+)\.\s+(.*?)\s*$")


def parse_mtg_text_into_sections(text: str) -> list[Section]:
    """Parse the MTG comprehensive rules .txt into Sections.

    Subsection numbers (e.g. ``100.1``) become the section ``heading``; the
    top-section line (``100. General``) becomes ``parent_heading``. Lines like
    ``100.1a.`` are deeper sub-subsections and stay as body content under
    their parent subsection.
    """
    sections: list[Section] = []
    current_top: str | None = None
    current_sub: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if body or current_sub:
            sections.append(Section(heading=current_sub, parent_heading=current_top, text=body))
        buf = []

    for line in text.splitlines():
        # Test sub before top — "100.1." starts with the same prefix as "100.".
        if (m := _MTG_SUB_RE.match(line)) is not None:
            flush()
            current_sub = m.group(1)
            buf.append(line)
        elif (m := _MTG_TOP_RE.match(line)) is not None:
            flush()
            current_top = f"{m.group(1)}. {m.group(2)}".strip()
            current_sub = None
        else:
            buf.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def _load_pdf_sections(path: Path) -> list[Section]:
    import pymupdf4llm  # lazy: keeps test imports light

    page_dicts = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    sections: list[Section] = []
    for page_data in page_dicts:
        page_md = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page")
        for section in parse_markdown_into_sections(page_md):
            section.page = page_num
            sections.append(section)
    return sections


def _load_txt_sections(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8")
    return parse_mtg_text_into_sections(text)


def load_file_sections(path: Path) -> list[Section] | None:
    """Dispatch on file extension. Returns None for unsupported types."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_sections(path)
    if suffix == ".txt":
        return _load_txt_sections(path)
    return None


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------


class Embedder:
    """Wraps sentence-transformers for batched encoding."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        logger.info("Loading embedding model %s on %s ...", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def build_qdrant_client(qdrant_config: Any) -> Any:
    """Construct a QdrantClient honoring the path-or-url config toggle.

    ``qdrant_config.path`` (embedded, on-disk; used by the HF Spaces deploy)
    wins over ``qdrant_config.url`` (server, used in local dev). The path
    directory is created on demand; the URL mode assumes a server is up.
    """
    from qdrant_client import QdrantClient  # lazy

    if qdrant_config.path:
        Path(qdrant_config.path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=qdrant_config.path)
    return QdrantClient(url=qdrant_config.url)


class QdrantStore:
    """Thin Qdrant wrapper: connect, ensure collection, upsert."""

    def __init__(self, qdrant_config: Any, dim: int) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        self.client = build_qdrant_client(qdrant_config)
        self.collection = qdrant_config.collection
        distance = qdrant_config.distance
        if not self.client.collection_exists(self.collection):
            logger.info(
                "Creating Qdrant collection %r (dim=%d, distance=%s)",
                self.collection,
                dim,
                distance,
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance[distance.upper()],
                ),
            )

    def upsert(self, points: list) -> None:
        self.client.upsert(collection_name=self.collection, points=points, wait=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def ingest_directory(
    source_dir: Path,
    collection: str,
    settings: Settings,
) -> int:
    """Run the full ingestion pipeline. Returns total chunks upserted.

    Side effects per run:

    * Upserts dense vectors + payloads into the Qdrant collection.
    * Writes a BM25 index pickle to ``settings.retrieval.bm25.index_dir /
      f"{collection}.pkl"`` (the source of truth for week-2 lexical retrieval).
    """
    from qdrant_client.http.models import PointStruct  # lazy

    # Lazy import to avoid a module-level cycle: retrieve.py imports Embedder
    # from this module inside its CLI builder.
    from boardgames_rag.retrieve import BM25Index, tokenize

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    candidate_files = sorted(p for p in source_dir.iterdir() if p.is_file())
    if not candidate_files:
        logger.warning("No files in %s — nothing to ingest.", source_dir)
        return 0

    tokenizer = tiktoken.get_encoding(settings.chunking.tokenizer)
    embedder = Embedder(
        model_name=settings.embedding.model_name,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )
    # Honor an explicit --collection override by swapping it into a copy of
    # the qdrant config before handing it to the store. Avoids mutating the
    # caller's settings object.
    qdrant_config = settings.qdrant.model_copy(update={"collection": collection})
    store = QdrantStore(qdrant_config=qdrant_config, dim=embedder.dim)

    # Pass 1: parse + chunk every supported file (cheap; no embedding yet).
    plan: list[tuple[Path, list[Chunk]]] = []
    for path in candidate_files:
        sections = load_file_sections(path)
        if sections is None:
            logger.warning("Skipped unsupported file: %s", path.name)
            continue
        chunks = pack_sections(
            sections,
            target_tokens=settings.chunking.target_tokens,
            overlap_tokens=settings.chunking.overlap_tokens,
            tokenizer=tokenizer,
        )
        plan.append((path, chunks))
        logger.info("Chunked %s → %d chunks", path.name, len(chunks))

    total_chunks = sum(len(c) for _, c in plan)
    if total_chunks == 0:
        logger.warning("No chunks produced — nothing to upsert.")
        return 0

    # Pass 2: embed + upsert with a progress bar.
    progress_cols = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    # Parallel arrays gathered during pass 2 → fed to the BM25 index in pass 3.
    bm25_point_ids: list[str] = []
    bm25_tokenized: list[list[str]] = []

    with Progress(*progress_cols, console=console) as progress:
        task = progress.add_task("Embedding & upserting", total=total_chunks)
        for path, chunks in plan:
            for batch in batched(chunks, settings.embedding.batch_size):
                vectors = embedder.embed([c.text for c in batch])
                points: list = []
                for i, chunk in enumerate(batch):
                    pid = stable_chunk_id(path.name, chunk.chunk_index, chunk.text)
                    points.append(
                        PointStruct(
                            id=pid,
                            vector=vectors[i].tolist(),
                            payload={
                                "source_file": path.name,
                                "page": chunk.page,
                                "chunk_index": chunk.chunk_index,
                                "heading": chunk.heading,
                                "parent_heading": chunk.parent_heading,
                                "text": chunk.text,
                            },
                        )
                    )
                    bm25_point_ids.append(pid)
                    bm25_tokenized.append(tokenize(chunk.text))
                store.upsert(points)
                progress.advance(task, len(batch))

    # Pass 3: build & persist the BM25 index alongside the dense vectors.
    bm25_index = BM25Index.build(
        bm25_point_ids,
        bm25_tokenized,
        k1=settings.retrieval.bm25.k1,
        b=settings.retrieval.bm25.b,
    )
    index_path = settings.retrieval.bm25.index_dir / f"{collection}.pkl"
    bm25_index.save(index_path)
    logger.info("BM25 index → %s (%d points)", index_path, len(bm25_point_ids))

    logger.info("Ingestion complete: %d chunks into %r", total_chunks, collection)
    return total_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    source_dir: Annotated[
        Path | None,
        typer.Option("--source-dir", help="Directory of rulebook PDFs and .txt files."),
    ] = None,
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Qdrant collection name."),
    ] = None,
    qdrant_path: Annotated[
        str | None,
        typer.Option(
            "--qdrant-path",
            help=(
                "Write to an embedded on-disk Qdrant instance at this path instead "
                "of the configured server URL. Used to build the index baked into "
                "the deploy image."
            ),
        ),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = Path("config.yaml"),
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG / INFO / WARNING / ERROR."),
    ] = "INFO",
) -> None:
    """Ingest board game rulebooks into Qdrant."""
    setup_logging(level=getattr(logging, log_level.upper(), logging.INFO))
    settings = load_settings(config_path)
    if source_dir is not None:
        settings.ingest.source_dir = source_dir
    if collection is not None:
        settings.qdrant.collection = collection
    if qdrant_path is not None:
        settings.qdrant.path = qdrant_path

    total = ingest_directory(
        source_dir=settings.ingest.source_dir,
        collection=settings.qdrant.collection,
        settings=settings,
    )
    typer.echo(f"Ingested {total} chunks into collection {settings.qdrant.collection!r}.")


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
