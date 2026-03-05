"""
KSelect RAG demo — run this to test search + RAG query against your own data.

Usage:
    python main.py   # search only (no key needed)
    python main.py   # full RAG if ANTHROPIC_API_KEY is set in .env

By default it indexes a small inline dataset so you can run it immediately.
Point DATA_PATH at your own folder of .txt/.pdf/.md files to use real data.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

# ── 0. Optional: point at your own data ──────────────────────────────────────
# Set DATA_PATH to a folder with your docs, or leave None to use inline sample.
DATA_PATH: str | None = None  # e.g. "/Users/you/docs"

# ── 1. Build a sample dataset if no real path is given ───────────────────────
def _make_sample_docs() -> str:
    """Write a few text files into a temp dir so the demo works out of the box."""
    tmp = tempfile.mkdtemp(prefix="kselect_demo_")
    docs = {
        "python_basics.txt": (
            "Python is a high-level, interpreted programming language known for its "
            "readability and simplicity. It supports multiple programming paradigms "
            "including procedural, object-oriented, and functional programming. "
            "Python was created by Guido van Rossum and first released in 1991. "
            "It has a large standard library and a vibrant ecosystem of third-party packages."
        ),
        "machine_learning.txt": (
            "Machine learning is a subset of artificial intelligence that allows systems "
            "to learn and improve from experience without being explicitly programmed. "
            "Common algorithms include linear regression, decision trees, random forests, "
            "support vector machines, and neural networks. Deep learning, a subset of "
            "machine learning, uses artificial neural networks with many layers to model "
            "complex patterns in data."
        ),
        "rag_overview.txt": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information "
            "retrieval with language model generation. Instead of relying solely on a "
            "pre-trained model's parametric knowledge, RAG retrieves relevant documents "
            "from an external knowledge base and feeds them as context to the LLM. "
            "This grounds the model's responses in actual data, reducing hallucinations "
            "and enabling up-to-date answers without retraining the model."
        ),
        "vector_search.txt": (
            "Vector search (semantic search) represents text as dense floating-point "
            "embeddings in a high-dimensional space. Documents with similar meaning are "
            "close together in this space. FAISS (Facebook AI Similarity Search) is a "
            "popular library for efficient vector similarity search. Hybrid search "
            "combines dense vector search with sparse BM25 keyword search for better "
            "recall across different query types."
        ),
        "kselect_info.txt": (
            "KSelect is a Python SDK for building RAG pipelines and vector search systems. "
            "It supports multiple ingestion formats (folders, CSV, JSON, JSONL), "
            "hybrid retrieval with FAISS + BM25, multiple reranking strategies "
            "(fast, cross-encoder, ColBERT, hybrid MMR), semantic caching, "
            "multi-tenant isolation, and Claude-backed LLM query answering. "
            "The public API is: KSelect.from_folder(), .search(), .query(), .save(), .load()."
        ),
    }
    for name, content in docs.items():
        (Path(tmp) / name).write_text(content)
    print(f"[demo] Created sample docs in: {tmp}")
    return tmp


# ── 2. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    data_path = DATA_PATH or _make_sample_docs()

    print("\n" + "="*60)
    print("KSelect RAG Demo")
    print("="*60)
    print(f"Indexing: {data_path}")
    print("This may take 30–90s on first run (model download + embedding).\n")

    from kselect import KSelect

    ks = KSelect.from_folder(
        data_path,
        chunk_size=200,
        chunk_overlap=30,
        bm25=True,
    )

    print(f"Index built: {ks.index_size()} chunks\n")

    # ── 3. Search (no LLM) ────────────────────────────────────────────────────
    queries = [
        "What is RAG and how does it work?",
        "How does vector search work?",
        "What programming language is good for beginners?",
    ]

    print("-"*60)
    print("SEARCH RESULTS (no LLM)")
    print("-"*60)

    for q in queries:
        print(f"\nQuery: {q!r}")
        result = ks.search(q, k=3, fast=True)
        for hit in result.hits:
            snippet = hit.snippet.replace("\n", " ")[:120]
            print(f"  [{hit.rank}] score={hit.score:.4f} | {snippet}...")

    # ── 4. RAG query with Claude ──────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n" + "="*60)
        print("Skipping RAG query — add ANTHROPIC_API_KEY to .env to enable LLM answers.")
        print("="*60)
        return

    print("\n" + "-"*60)
    print("RAG QUERY RESULTS (Claude)")
    print("-"*60)

    from kselect.llm.anthropic_client import AnthropicClient
    from kselect.models.config import KSelectConfig, LLMConfig

    cfg = KSelectConfig()
    cfg.llm = LLMConfig(model="claude-sonnet-4-6", max_tokens=512)

    ks_rag = KSelect.from_folder(
        data_path,
        config=cfg,
        chunk_size=200,
        chunk_overlap=30,
        bm25=True,
    )
    llm = AnthropicClient(api_key=api_key, model="claude-sonnet-4-6")
    ks_rag._llm = llm
    ks_rag._assembler._llm = llm

    rag_queries = [
        "Explain RAG and why it reduces hallucinations.",
        "What reranking strategies does KSelect support?",
    ]

    for q in rag_queries:
        print(f"\nQuestion: {q!r}")
        result = ks_rag.query(q, k=5, fast=True)
        print(f"Answer:   {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Sources used: {result.chunks_in_context} chunks")
        for src in result.sources[:2]:
            snippet = src.snippet.replace("\n", " ")[:100]
            print(f"  - [{src.doc_id}] {snippet}...")

    # ── 5. Save + reload ──────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("Save / Load demo")
    save_path = "/tmp/kselect_saved_state"
    ks.save(save_path)
    print(f"Saved to: {save_path}")

    ks2 = KSelect.load(save_path)
    r = ks2.search("what is machine learning", k=2, fast=True)
    print(f"Loaded index, search returned {len(r.hits)} hits")
    print("Done!")


if __name__ == "__main__":
    main()
