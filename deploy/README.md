---
title: boardgames-rag
emoji: 🎲
colorFrom: emerald
colorTo: slate
sdk: docker
app_port: 7860
short_description: Agentic RAG over 10 board game rulebooks
pinned: false
license: mit
---

# boardgames-rag

Ask a natural-language rules question about ten board games — Catan, Ticket
to Ride, Carcassonne, Azul, 7 Wonders, Wingspan, Scythe, Terraforming Mars,
Brass: Birmingham, and *Magic: The Gathering* — and watch an agentic RAG
pipeline answer it with cited rulebook passages, live.

This Space is the deploy artifact for an end-to-end 7-week project that
combines hybrid retrieval (BM25 + dense vectors, fused with Reciprocal
Rank Fusion), a cross-encoder reranker, and a LangGraph agent that
plans, retrieves, critiques, and answers — all measured with Ragas.

**[Source on GitHub →](https://github.com/Kaushik1128/boardgames-rag)**

## Stack

- **Generator:** Groq · `llama-3.3-70b-versatile` (free tier)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local)
- **Reranker:** `BAAI/bge-reranker-base` (local cross-encoder)
- **Vector store:** Qdrant in embedded on-disk mode
- **Lexical:** rank-bm25 with a tokenizer matched to the chunking
- **Agent:** LangGraph state machine — plan → retrieve → critique → generate

## Rulebook content

Rule excerpts shown as sources are © their respective publishers and
are surfaced here for educational demonstration. If you represent a
rights-holder and would like content removed, please open an issue on the
[GitHub repository](https://github.com/Kaushik1128/boardgames-rag/issues)
and the affected rulebook will be removed from the corpus.
