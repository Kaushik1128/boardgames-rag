"""Hybrid retrieval over board game rulebooks.

Combines:
- **BM25** (rank-bm25, pickled index on disk)
- **Dense** (Qdrant + bge-small embeddings)

Fused via Reciprocal Rank Fusion (default) or weighted score fusion.

CLI:

.. code-block:: bash

    python -m boardgames_rag.retrieve --query "How do I take a turn in Catan?" \
        --collection boardgames --k 10
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

import numpy as np
import typer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from boardgames_rag.config import Settings, load_settings
from boardgames_rag.utils import setup_logging

logger = logging.getLogger(__name__)
console = Console()


__all__ = [
    "BM25Index",
    "BM25Retriever",
    "CrossEncoderProtocol",
    "DenseRetriever",
    "HybridRetriever",
    "IndexOutOfSyncError",
    "Reranker",
    "RetrievedChunk",
    "build_hybrid_retriever",
    "main",
    "reciprocal_rank_fusion",
    "rerank_chunks",
    "tokenize",
    "weighted_score_fusion",
]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Captures alphanumeric tokens, optionally joined by . or - (preserves
# identifiers like "100.6b" or "two-player"). Apostrophes kept inline so
# "card's" survives. Colons / semicolons / brackets are split on.
_TOKEN_RE = re.compile(r"[a-z0-9']+(?:[.\-][a-z0-9']+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase + regex tokenize for BM25. Preserves identifiers like ``100.6b``."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A retrieval result with its fused score and per-retriever provenance."""

    point_id: str
    score: float
    payload: dict[str, Any]
    debug: dict[str, Any] = field(default_factory=dict)


class IndexOutOfSyncError(RuntimeError):
    """Raised when the BM25 pickle and the Qdrant collection disagree."""


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


@dataclass
class _BM25State:
    """Picklable state for BM25Index. Internal; do not touch directly."""

    point_ids: list[str]
    tokenized_corpus: list[list[str]]
    corpus_size: int
    point_id_hash: str
    bm25: BM25Okapi
    version: str = "1"


def _hash_point_ids(point_ids: list[str]) -> str:
    """sha256 over the sorted, newline-joined point_id list."""
    joined = "\n".join(sorted(point_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


class BM25Index:
    """Wraps a rank-bm25 model + a parallel point_id list + a corpus signature.

    The signature ``(corpus_size, point_id_hash)`` lets HybridRetriever detect
    when the pickle has drifted from the live Qdrant collection.
    """

    def __init__(self, state: _BM25State) -> None:
        self._state = state

    @classmethod
    def build(
        cls,
        point_ids: list[str],
        tokenized_corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        if len(point_ids) != len(tokenized_corpus):
            raise ValueError(
                f"point_ids ({len(point_ids)}) and tokenized_corpus "
                f"({len(tokenized_corpus)}) must have the same length"
            )
        if not point_ids:
            raise ValueError("cannot build a BM25Index on an empty corpus")
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        state = _BM25State(
            point_ids=list(point_ids),
            tokenized_corpus=[list(toks) for toks in tokenized_corpus],
            corpus_size=len(point_ids),
            point_id_hash=_hash_point_ids(point_ids),
            bm25=bm25,
        )
        return cls(state)

    @property
    def signature(self) -> tuple[int, str]:
        return (self._state.corpus_size, self._state.point_id_hash)

    @property
    def point_ids(self) -> list[str]:
        return list(self._state.point_ids)

    def search(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        """Top-``k`` (point_id, score) pairs, descending. Drops zero scores."""
        if k <= 0 or not query_tokens:
            return []
        scores = self._state.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        results: list[tuple[str, float]] = []
        for idx in top_idx:
            s = float(scores[idx])
            if s <= 0:
                continue
            results.append((self._state.point_ids[int(idx)], s))
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            state = pickle.load(f)
        if not isinstance(state, _BM25State):
            raise ValueError(f"Not a BM25Index pickle: {path}")
        return cls(state)


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> Any: ...

    @property
    def dim(self) -> int: ...


class QdrantClientProtocol(Protocol):
    def query_points(self, *args: Any, **kwargs: Any) -> Any: ...

    def scroll(self, *args: Any, **kwargs: Any) -> Any: ...

    def retrieve(self, *args: Any, **kwargs: Any) -> Any: ...


class BM25Retriever:
    """Thin search wrapper around a loaded BM25Index."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.index.search(tokenize(query), k)


class DenseRetriever:
    """Embeds the query and runs a dense vector search against Qdrant."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        client: QdrantClientProtocol,
        collection: str,
    ) -> None:
        self.embedder = embedder
        self.client = client
        self.collection = collection

    def search(self, query: str, k: int) -> list[tuple[str, float, dict[str, Any]]]:
        if k <= 0:
            return []
        vec = self.embedder.embed([query])[0]
        vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        result = self.client.query_points(
            collection_name=self.collection,
            query=vec_list,
            limit=k,
            with_payload=True,
        )
        # qdrant-client returns a QueryResponse with .points in modern versions;
        # older versions returned the list directly. Handle both.
        points = getattr(result, "points", result)
        out: list[tuple[str, float, dict[str, Any]]] = []
        for p in points:
            out.append((str(p.id), float(p.score), dict(p.payload or {})))
        return out


# ---------------------------------------------------------------------------
# Fusion (pure functions)
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """RRF: ``score = sum_retrievers 1 / (k + rank)``.

    Returns ``{point_id: rrf_score}`` unsorted; caller sorts.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_score_fusion(
    rankings_with_scores: list[list[tuple[str, float]]],
    weights: list[float],
) -> dict[str, float]:
    """Min-max normalize each retriever's scores, then weighted sum.

    A retriever with weight ``0`` is ignored. A retriever whose scores are all
    identical contributes a constant ``1.0`` for every result it returns.
    """
    if len(rankings_with_scores) != len(weights):
        raise ValueError(
            f"rankings ({len(rankings_with_scores)}) and weights "
            f"({len(weights)}) must have the same length"
        )
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    fused: dict[str, float] = {}
    for ranking, w in zip(rankings_with_scores, weights, strict=False):
        if not ranking or w == 0:
            continue
        scores = [s for _, s in ranking]
        smin, smax = min(scores), max(scores)
        span = smax - smin
        for pid, s in ranking:
            norm = 1.0 if span == 0 else (s - smin) / span
            fused[pid] = fused.get(pid, 0.0) + w * norm
    return fused


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Runs dense + BM25, fuses rankings, hydrates payloads."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        client: QdrantClientProtocol,
        collection: str,
        *,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        weighted_alpha: float = 0.5,
        k_per_retriever: int = 20,
    ) -> None:
        if fusion_method not in {"rrf", "weighted"}:
            raise ValueError(f"unknown fusion_method: {fusion_method!r}")
        if not 0.0 <= weighted_alpha <= 1.0:
            raise ValueError(f"weighted_alpha must be in [0, 1], got {weighted_alpha}")
        self.dense = dense
        self.bm25 = bm25
        self.client = client
        self.collection = collection
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.weighted_alpha = weighted_alpha
        self.k_per_retriever = k_per_retriever

    def verify_consistency(self) -> None:
        """Raise IndexOutOfSyncError if BM25 disagrees with Qdrant.

        Cheap signature check: counts must match, and the sha256 of the sorted
        point_id lists must match. Mismatch usually means ingestion was re-run
        but the BM25 index wasn't (or vice versa).
        """
        bm25_size, bm25_hash = self.bm25.index.signature
        qdrant_ids: list[str] = []
        offset: Any = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            records, offset = result if isinstance(result, tuple) else (result, None)
            for r in records:
                qdrant_ids.append(str(r.id))
            if offset is None:
                break

        qdrant_hash = _hash_point_ids(qdrant_ids)
        if len(qdrant_ids) != bm25_size or qdrant_hash != bm25_hash:
            raise IndexOutOfSyncError(
                f"BM25 index out of sync with Qdrant collection "
                f"{self.collection!r}: BM25 has {bm25_size} points, "
                f"Qdrant has {len(qdrant_ids)}. Re-run ingestion: "
                f"python -m boardgames_rag.ingest --collection {self.collection}"
            )

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return up to ``top_k`` fused results with debug provenance attached."""
        dense_results = self.dense.search(query, self.k_per_retriever)
        bm25_results = self.bm25.search(query, self.k_per_retriever)

        dense_rank = {pid: r for r, (pid, _, _) in enumerate(dense_results, start=1)}
        dense_score = {pid: s for pid, s, _ in dense_results}
        dense_payload = {pid: p for pid, _, p in dense_results}
        bm25_rank = {pid: r for r, (pid, _) in enumerate(bm25_results, start=1)}
        bm25_score = {pid: s for pid, s in bm25_results}

        if self.fusion_method == "rrf":
            fused_scores = reciprocal_rank_fusion(
                [
                    [pid for pid, _, _ in dense_results],
                    [pid for pid, _ in bm25_results],
                ],
                k=self.rrf_k,
            )
        else:
            fused_scores = weighted_score_fusion(
                [
                    [(pid, s) for pid, s, _ in dense_results],
                    bm25_results,
                ],
                weights=[self.weighted_alpha, 1.0 - self.weighted_alpha],
            )

        top_ids = sorted(fused_scores.keys(), key=lambda p: fused_scores[p], reverse=True)[:top_k]

        # Hydrate payloads for BM25-only hits (dense already returned payloads).
        missing = [pid for pid in top_ids if pid not in dense_payload]
        hydrated: dict[str, dict[str, Any]] = dict(dense_payload)
        if missing:
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=missing,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                hydrated[str(r.id)] = dict(r.payload or {})

        return [
            RetrievedChunk(
                point_id=pid,
                score=float(fused_scores[pid]),
                payload=hydrated.get(pid, {}),
                debug={
                    "dense_rank": dense_rank.get(pid),
                    "dense_score": dense_score.get(pid),
                    "bm25_rank": bm25_rank.get(pid),
                    "bm25_score": bm25_score.get(pid),
                    "fusion_method": self.fusion_method,
                },
            )
            for pid in top_ids
        ]


# ---------------------------------------------------------------------------
# Reranking (cross-encoder)
# ---------------------------------------------------------------------------


class CrossEncoderProtocol(Protocol):
    """Minimal surface of sentence-transformers' CrossEncoder used here."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


def rerank_chunks(
    model: CrossEncoderProtocol,
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Re-score chunks with a cross-encoder, sort descending, keep ``top_k``.

    The cross-encoder score becomes each chunk's primary ``score``. The prior
    fusion score and the new rerank rank are recorded in ``debug`` so the full
    retrieve → fuse → rerank provenance survives.
    """
    if not chunks or top_k <= 0:
        return []
    pairs = [(query, c.payload.get("text", "")) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(chunks, scores, strict=True),
        key=lambda cs: float(cs[1]),
        reverse=True,
    )

    out: list[RetrievedChunk] = []
    for rank, (chunk, score) in enumerate(ranked[:top_k], start=1):
        debug = dict(chunk.debug)
        debug["fusion_score"] = chunk.score
        debug["rerank_score"] = float(score)
        debug["rerank_rank"] = rank
        out.append(
            RetrievedChunk(
                point_id=chunk.point_id,
                score=float(score),
                payload=chunk.payload,
                debug=debug,
            )
        )
    return out


class Reranker:
    """Cross-encoder reranker backed by sentence-transformers.

    A cross-encoder reads ``(query, chunk_text)`` jointly and outputs one
    relevance score — far more accurate than bi-encoder dense retrieval, but
    too slow to run corpus-wide, so it only re-scores retrieved candidates.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import CrossEncoder  # lazy

        logger.info("Loading reranker model %s on %s ...", model_name, device)
        self.model: CrossEncoderProtocol = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return rerank_chunks(self.model, query, chunks, top_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_results_table(chunks: list[RetrievedChunk]) -> Table:
    table = Table(title=f"Top {len(chunks)} results", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("D rank", justify="right", no_wrap=True)
    table.add_column("B rank", justify="right", no_wrap=True)
    table.add_column("Source", max_width=22)
    table.add_column("Heading", max_width=24)
    table.add_column("Snippet")
    for i, chunk in enumerate(chunks, start=1):
        snippet = (chunk.payload.get("text") or "")[:160].replace("\n", " ")
        table.add_row(
            str(i),
            f"{chunk.score:.4f}",
            str(chunk.debug.get("dense_rank") or "—"),
            str(chunk.debug.get("bm25_rank") or "—"),
            str(chunk.payload.get("source_file") or "—"),
            str(chunk.payload.get("heading") or "—"),
            snippet,
        )
    return table


def build_hybrid_retriever(
    settings: Settings,
    collection: str,
) -> HybridRetriever:
    """Construct a HybridRetriever with the real Embedder + Qdrant client."""
    from qdrant_client import QdrantClient  # lazy

    from boardgames_rag.ingest import Embedder  # lazy, avoids module cycle

    embedder = Embedder(
        model_name=settings.embedding.model_name,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )
    client = QdrantClient(url=settings.qdrant.url)
    dense = DenseRetriever(embedder=embedder, client=client, collection=collection)

    index_path = settings.retrieval.bm25.index_dir / f"{collection}.pkl"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No BM25 index at {index_path}. Run ingestion first: "
            f"python -m boardgames_rag.ingest --collection {collection}"
        )
    bm25 = BM25Retriever(BM25Index.load(index_path))

    return HybridRetriever(
        dense=dense,
        bm25=bm25,
        client=client,
        collection=collection,
        fusion_method=settings.retrieval.fusion.method,
        rrf_k=settings.retrieval.fusion.rrf_k,
        weighted_alpha=settings.retrieval.fusion.weighted_alpha,
        k_per_retriever=settings.retrieval.k_per_retriever,
    )


def main(
    query: Annotated[str, typer.Option("--query", "-q", help="Natural-language query.")],
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Qdrant collection name."),
    ] = None,
    k: Annotated[
        int | None,
        typer.Option("--k", help="Number of results to return."),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = Path("config.yaml"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print results as JSON instead of a table."),
    ] = False,
    skip_consistency_check: Annotated[
        bool,
        typer.Option("--skip-consistency-check", help="Don't verify BM25 vs Qdrant."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG / INFO / WARNING / ERROR."),
    ] = "WARNING",
) -> None:
    """Query the hybrid retriever (dense + BM25)."""
    setup_logging(level=getattr(logging, log_level.upper(), logging.WARNING))
    settings = load_settings(config_path)
    if collection is None:
        collection = settings.qdrant.collection
    top_k = k or settings.retrieval.top_k

    hybrid = build_hybrid_retriever(settings, collection)
    if not skip_consistency_check:
        hybrid.verify_consistency()

    chunks = hybrid.search(query, top_k=top_k)

    if json_output:
        out = [
            {
                "rank": i,
                "point_id": c.point_id,
                "score": c.score,
                "payload": c.payload,
                "debug": c.debug,
            }
            for i, c in enumerate(chunks, start=1)
        ]
        console.print_json(json.dumps(out, default=str))
    else:
        console.print(_format_results_table(chunks))


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
