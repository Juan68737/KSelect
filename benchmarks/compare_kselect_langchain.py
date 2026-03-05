"""
BEIR / SciFact benchmark — KSelect vs vanilla LangChain RAG
============================================================
Loads the SciFact test split (300 queries, ~5k corpus docs) and evaluates
retrieval quality for both systems using standard IR metrics.

Usage:
    cd /path/to/KSelect/KSelect
    PYTHONPATH=. .venv/bin/python benchmarks/compare_kselect_langchain.py

Runtime: ~3-5 min (embedding ~5k docs once, shared + reranking 300 queries)

Index parameters (tuned for SciFact corpus size ~5k docs)
----------------------------------------------------------
KSelect FAISS:
  - IndexType.FLAT (IndexFlatIP) — exact search, best recall for <50k docs.
    IVF requires nlist training vectors; with only 5k docs, IVF offers no
    speed benefit and risks recall loss from bad cluster assignment.
  - nlist=128, nprobe=32 — kept in config but irrelevant for FLAT index.
  - BM25 enabled (k1=1.2, b=0.75) — scientific text has technical terms
    that benefit from exact keyword matching ("BRCA1", "mTOR pathway").
  - Fusion: RRF with rrf_k=20 — low k amplifies BM25 exact-match boost.
  - No cross-encoder (CPU too slow for 300q benchmark — ~1h runtime).
  - RRF top-30 candidates truncated to K=10 directly.

LangChain + Chroma (vanilla baseline):
  - HuggingFaceEmbeddings → Chroma.from_documents → retriever.
  - No reranking, no BM25, no custom config. Standard tutorial-style RAG.
"""
from __future__ import annotations

import sys
import os
import time
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Model constants ────────────────────────────────────────────────────────────

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"   # same for both systems — fair comparison
K            = 10                          # final results returned
FUSION_TOP_N = 30                          # RRF fusion candidates
MAX_CORPUS   = 2000                        # cap corpus size — full 5k crashes low-RAM Macs
EMBED_BATCH  = 32                          # small batch — safe on MPS/CPU


# ── BEIR data loading ─────────────────────────────────────────────────────────

def load_scifact(data_dir: str = "datasets/scifact"):
    """Download SciFact via BEIR if not cached, return (corpus, queries, qrels)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    if not Path(data_dir).exists():
        print("Downloading SciFact dataset …")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        out_dir = str(Path(data_dir).parent)
        data_path = util.download_and_unzip(url, out_dir)
        data_dir = data_path

    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    print(f"SciFact loaded — corpus: {len(corpus):,} docs, "
          f"queries: {len(queries):,}, qrels: {sum(len(v) for v in qrels.values())} judgements")
    return corpus, queries, qrels


# ── Embedding helper ───────────────────────────────────────────────────────────

_embed_model_cache = {}

def get_embed_model():
    if EMBED_MODEL not in _embed_model_cache:
        from sentence_transformers import SentenceTransformer
        # CPU only — MPS allocates too much unified memory and crashes Mac
        device = "cpu"
        print(f"  embedding device: {device}")
        _embed_model_cache[EMBED_MODEL] = SentenceTransformer(EMBED_MODEL, device=device)
    return _embed_model_cache[EMBED_MODEL]

def embed(texts: list[str], batch_size: int = EMBED_BATCH) -> np.ndarray:
    model = get_embed_model()
    vecs = model.encode(texts, batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    return np.array(vecs, dtype="float32")


# ── IR metrics ─────────────────────────────────────────────────────────────────

def recall_at_k(results: dict[str, list[str]], qrels: dict, k: int) -> float:
    """Fraction of relevant docs retrieved in top-k, averaged over queries."""
    scores = []
    for qid, ranked in results.items():
        relevant = set(qrels.get(qid, {}).keys())
        if not relevant:
            continue
        hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
        scores.append(hits / len(relevant))
    return float(np.mean(scores)) if scores else 0.0

def ndcg_at_k(results: dict[str, list[str]], qrels: dict, k: int) -> float:
    """nDCG@k averaged over queries. Uses binary relevance (rel ≥ 1)."""
    scores = []
    for qid, ranked in results.items():
        relevant = set(qrels.get(qid, {}).keys())
        if not relevant:
            continue
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, doc_id in enumerate(ranked[:k])
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def mrr(results: dict[str, list[str]], qrels: dict) -> float:
    """Mean Reciprocal Rank (first relevant hit)."""
    scores = []
    for qid, ranked in results.items():
        relevant = set(qrels.get(qid, {}).keys())
        if not relevant:
            continue
        rr = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


# ── KSelect retriever ──────────────────────────────────────────────────────────

def build_kselect(corpus: dict, doc_ids: list, texts: list, vecs: np.ndarray) -> "KSelectRetriever":
    """Build KSelect index. Receives precomputed embeddings — no re-embedding."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from kselect.models.chunk import Chunk, ChunkMetadata
    from kselect.models.config import (
        KSelectConfig, IndexConfig, IndexType, BM25Config,
        FusionConfig, RankingConfig, EmbeddingConfig,
    )
    from kselect.index.faiss_index import FAISSIndex
    from kselect.index.bm25_index import BM25Index
    from kselect.index.manager import IndexManager
    from kselect.backends.local import LocalBackend

    chunks = [
        Chunk(
            id=doc_id,
            text=text,
            embedding=vec.tolist(),
            metadata=ChunkMetadata(
                source_file="scifact",
                chunk_index=i,
                char_start=0,
                char_end=len(text),
                token_count=len(text.split()),
            ),
        )
        for i, (doc_id, text, vec) in enumerate(zip(doc_ids, texts, vecs))
    ]

    cfg = KSelectConfig()
    cfg.embedding  = EmbeddingConfig(model=EMBED_MODEL)
    # FLAT index — exact IP search, best recall for corpus size <50k.
    # IVF adds no speed benefit here and risks recall loss.
    cfg.index      = IndexConfig(type=IndexType.FLAT, nlist=128)
    cfg.bm25       = BM25Config(enabled=True, k1=1.2, b=0.75)
    cfg.fusion     = FusionConfig(mode="rrf", rrf_k=20)
    cfg.ranking    = RankingConfig(
        mode="fast",   # no cross-encoder — benchmark needs speed, not production quality
        mmr_lambda=0.7,
        k=K,
    )

    state_dir = tempfile.mkdtemp(prefix="ks_scifact_")
    mgr = IndexManager(FAISSIndex(), BM25Index(), LocalBackend(state_dir), cfg)
    mgr.build(chunks)
    print(f"  index built — {mgr.index_size():,} vectors")

    return KSelectRetriever(mgr, cfg)


class KSelectRetriever:
    def __init__(self, mgr, cfg):
        from kselect.retrieval.engine import RetrievalEngine
        self._mgr = mgr
        self._cfg = cfg
        self._engine = RetrievalEngine(mgr, cfg)

    def retrieve(self, query: str) -> list[str]:
        from kselect.retrieval.fusion import rrf

        q_emb = self._engine.embed_query(query)
        faiss_res = self._mgr.search_faiss(q_emb, FUSION_TOP_N)
        bm25_res  = self._mgr.search_bm25(query, FUSION_TOP_N)
        candidates = rrf(faiss_res, bm25_res, k=self._cfg.fusion.rrf_k, top_n=FUSION_TOP_N)

        return [cid for cid, _ in candidates[:K]]


def run_kselect(retriever: "KSelectRetriever", queries: dict) -> dict[str, list[str]]:
    print("KSelect — evaluating queries …")
    t0 = time.perf_counter()
    results = {}
    items = list(queries.items())
    for i, (qid, qtext) in enumerate(items, 1):
        results[qid] = retriever.retrieve(qtext)
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} queries …", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  done — {len(queries):,} queries in {elapsed:.1f}s ({elapsed/len(queries)*1000:.0f}ms/q)")
    return results


# ── LangChain retriever ────────────────────────────────────────────────────────

def build_langchain(doc_ids: list, texts: list, vecs: np.ndarray):
    """
    Standard LangChain RAG — Chroma vector store, no reranking, no BM25.
    Reuses precomputed embeddings to avoid a second full encoding pass.
    """
    import chromadb
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("LangChain (Chroma) — building index from precomputed vectors …")
    t0 = time.perf_counter()

    client = chromadb.EphemeralClient()
    collection = client.create_collection("scifact", metadata={"hnsw:space": "cosine"})
    # Add in batches of 500 — Chroma has a per-call limit
    batch = 500
    for i in range(0, len(doc_ids), batch):
        collection.add(
            ids=doc_ids[i:i+batch],
            embeddings=vecs[i:i+batch].tolist(),
            documents=texts[i:i+batch],
            metadatas=[{"doc_id": d} for d in doc_ids[i:i+batch]],
        )

    embeddings_obj = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        client=client,
        collection_name="scifact",
        embedding_function=embeddings_obj,
    )
    print(f"  indexed {len(doc_ids):,} docs in {time.perf_counter()-t0:.1f}s")
    return vectorstore.as_retriever(search_kwargs={"k": K})


def run_langchain(retriever, queries: dict) -> dict[str, list[str]]:
    print("LangChain (Chroma) — evaluating queries …")
    t0 = time.perf_counter()
    results = {}
    items = list(queries.items())
    for i, (qid, qtext) in enumerate(items, 1):
        docs = retriever.invoke(qtext)
        results[qid] = [d.metadata["doc_id"] for d in docs]
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} queries …", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  done — {len(queries):,} queries in {elapsed:.1f}s ({elapsed/len(queries)*1000:.0f}ms/q)")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print()
    print("| Method    | Recall@10 | nDCG@10 | MRR  |")
    print("|-----------|-----------|---------|------|")
    for r in rows:
        print(
            f"| {r['method']:<9} "
            f"| {r['recall']:.3f}     "
            f"| {r['ndcg']:.3f}   "
            f"| {r['mrr']:.3f} |"
        )
    print()


def main():
    corpus, queries, qrels = load_scifact()

    # Cap corpus to avoid OOM on low-RAM Macs
    doc_ids_all = list(corpus.keys())
    if len(doc_ids_all) > MAX_CORPUS:
        print(f"  capping corpus {len(doc_ids_all):,} → {MAX_CORPUS:,} docs")
        doc_ids_all = doc_ids_all[:MAX_CORPUS]
        corpus = {d: corpus[d] for d in doc_ids_all}

    # Embed corpus ONCE — shared by both KSelect and LangChain.
    print(f"Embedding {len(corpus):,} docs (once, shared) …")
    t0 = time.perf_counter()
    doc_ids = list(corpus.keys())
    texts   = [corpus[d]["title"] + " " + corpus[d]["text"] for d in doc_ids]
    vecs    = embed(texts)
    print(f"  done in {time.perf_counter()-t0:.1f}s — {vecs.nbytes / 1e6:.0f} MB")

    # ── KSelect ──
    ks_retriever  = build_kselect(corpus, doc_ids, texts, vecs)
    ks_results    = run_kselect(ks_retriever, queries)

    # ── LangChain + Chroma (standard vanilla RAG) ──
    lc_retriever  = build_langchain(doc_ids, texts, vecs)
    lc_results    = run_langchain(lc_retriever, queries)

    # ── Metrics ──
    rows = [
        {
            "method":  "KSelect",
            "recall":  recall_at_k(ks_results, qrels, K),
            "ndcg":    ndcg_at_k(ks_results, qrels, K),
            "mrr":     mrr(ks_results, qrels),
        },
        {
            "method":  "LC+Chroma",
            "recall":  recall_at_k(lc_results, qrels, K),
            "ndcg":    ndcg_at_k(lc_results, qrels, K),
            "mrr":     mrr(lc_results, qrels),
        },
    ]

    print_table(rows)

    # Delta summary
    ks, lc = rows[0], rows[1]
    print("Delta (KSelect − LangChain):")
    print(f"  Recall@10 : {ks['recall']-lc['recall']:+.3f}")
    print(f"  nDCG@10   : {ks['ndcg']-lc['ndcg']:+.3f}")
    print(f"  MRR       : {ks['mrr']-lc['mrr']:+.3f}")


if __name__ == "__main__":
    main()
