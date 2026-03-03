# KSelect — SPEC Part 4 of 5: Index, Retrieval, and Ranking

**Read order:** After SPEC-03. This covers `FAISSIndex`, `BM25Index`, `IndexManager`, `RetrievalEngine`, fusion algorithms, and the full ranking layer including `CrossEncoderReranker`, `ColBERTReranker`, and MMR diversification.

---

## 8. Index Layer

### 8.1 `FAISSIndex` — `index/faiss_index.py`

```python
import faiss
import numpy as np
from kselect.models.chunk import Chunk
from kselect.models.config import IndexConfig, IndexType


class FAISSIndex:
    """
    Wraps a faiss.Index. Owns the positional id→chunk_id mapping.

    Internal state:
      self._index: faiss.Index              # the FAISS index object
      self._id_map: list[str]               # position i → chunk_id (append-only)
      self._chunk_id_to_pos: dict[str, int] # chunk_id → position i (for deletion lookup)
      self._original_size: int              # size at last full build (for drift calc)

    Index construction by IndexType:
      FLAT:      faiss.IndexFlatIP
                 (inner product on L2-normalized vectors == cosine similarity)
      IVF_PQ128: faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
                 quantizer = faiss.IndexFlatL2(dim)
                 Requires .train() before .add()
      HNSW_SQ:   faiss.IndexHNSWFlat(dim, hnsw_m) wrapped in faiss.IndexIDMap2
      VLQ_ADC:   faiss.IndexIVFPQ with VLQ subquantizer configuration
                 (treat as IVF_PQ128 for Phase 3; implement VLQ specifics in Phase 9)
      FCVI:      VLQ_ADC base + pre-filtering hook
                 (stub in Phase 3; implement fully in Phase 9)

    ALL index types normalize input vectors to unit length before add/search.
    Use faiss.normalize_L2(vectors) before every add() and search() call.
    """

    def build(self, chunks: list[Chunk], config: IndexConfig) -> None:
        """
        Build index from scratch from a list of embedded Chunks.
        Chunks must have embedding populated (not None).
        Raises KSelectIndexError if any chunk has embedding=None.
        For IVF types: train on all embeddings before adding.
        """

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, score) sorted by score descending.
        Score is inner product (cosine similarity after normalization), range [-1, 1].
        If k > index size, returns all available results without error.
        """

    def add(self, chunks: list[Chunk]) -> None:
        """
        Incremental add of new chunks to an existing index.
        Appends to self._id_map and self._chunk_id_to_pos.

        Drift check: after appending, compute drift ratio:
          drift = (self.size - self._original_size) / self._original_size
        If drift > 0.20, set self._needs_reindex = True.
        Does NOT rebuild — IndexManager handles that.

        For IVF types: faiss does not support true incremental add without
        retraining. Use faiss.index_cpu_to_gpu / direct add for now;
        log a WARNING that recall may degrade above 20% drift.
        """

    def save(self, path: str) -> None:
        """
        Write to {path}/faiss/index.faiss (faiss.write_index)
        Write to {path}/faiss/id_map.json (json.dump of self._id_map)
        Write to {path}/faiss/meta.json (original_size, index_type string)
        """

    def load(self, path: str) -> None:
        """Read all three files from {path}/faiss/"""

    @property
    def size(self) -> int:
        """Total number of vectors in the index."""

    @property
    def needs_reindex(self) -> bool:
        """True if drift has exceeded 20% threshold."""
```

---

### 8.2 `BM25Index` — `index/bm25_index.py`

```python
import bm25s
from kselect.models.chunk import Chunk
from kselect.models.config import BM25Config


class BM25Index:
    """
    Wraps bm25s.BM25.
    Maintains a parallel id_map list aligned with the BM25 corpus.

    Internal state:
      self._bm25: bm25s.BM25
      self._id_map: list[str]   # position i → chunk_id (aligned with BM25 corpus)

    Serialization:
      {path}/bm25/model.pkl   → pickle.dump(self._bm25)
      {path}/bm25/id_map.json → json.dump(self._id_map)
    """

    def build(self, chunks: list[Chunk], config: BM25Config) -> None:
        """
        Tokenize chunk texts and index with bm25s.
        bm25s.BM25(k1=config.k1, b=config.b)
        corpus = [chunk.text for chunk in chunks]
        tokenized = bm25s.tokenize(corpus)
        self._bm25.index(tokenized)
        self._id_map = [chunk.id for chunk in chunks]
        """

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """
        Tokenize query, run BM25 retrieval.
        Returns list of (chunk_id, bm25_score) sorted by score descending.
        BM25 scores are NOT normalized to [0,1] — they are raw BM25 scores.
        Fusion layer handles normalization before combining with FAISS scores.
        If k > index size, return all results.
        """

    def add(self, chunks: list[Chunk]) -> None:
        """
        bm25s supports incremental indexing via repeated .index() calls.
        Append new chunk texts to corpus; reindex.
        Append new chunk ids to self._id_map.
        BM25 does NOT degrade on incremental add — no drift check needed.
        """

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

    @property
    def size(self) -> int: ...
```

---

### 8.3 `IndexManager` — `index/manager.py`

```python
import os
import uuid
import threading
from kselect.index.faiss_index import FAISSIndex
from kselect.index.bm25_index import BM25Index
from kselect.backends.base import VectorBackend
from kselect.models.chunk import Chunk
from kselect.models.config import KSelectConfig


class IndexManager:
    """
    Single owner of FAISSIndex + BM25Index.
    All code outside index/ interacts with IndexManager only — never with
    FAISSIndex or BM25Index directly.

    Responsibilities:
    - Build both indexes from a list[Chunk]
    - Unified atomic save/load
    - add_chunks() flow for incremental ingestion
    - Background reindex when drift exceeds threshold
    - Expose index_drift(), recall_estimate(), index_size()
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        backend: VectorBackend,
        config: KSelectConfig,
    ): ...

    def build(self, chunks: list[Chunk]) -> None:
        """Build both indexes from the full chunk list. Blocks until complete."""

    def search_faiss(
        self,
        embedding: "np.ndarray",
        k: int,
    ) -> list[tuple[str, float]]:
        """Delegate to FAISSIndex.search(). Thread-safe (read lock)."""

    def search_bm25(
        self,
        query: str,
        k: int,
    ) -> list[tuple[str, float]]:
        """Delegate to BM25Index.search(). Thread-safe (read lock)."""

    def get_chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """
        Return {chunk_id: text} for the given ids.
        Used by rerankers to fetch text for scoring.
        Looks up from in-memory chunk store (dict[str, Chunk] loaded at build/load time).
        """

    def get_chunk_embeddings(self, chunk_ids: list[str]) -> dict[str, "np.ndarray"]:
        """Return {chunk_id: embedding} for MMR diversification."""

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Incremental add flow (called by KSelect.add_doc / add_folder):

        1. Assert all chunks have embeddings populated.
        2. Acquire write lock.
        3. FAISSIndex.add(chunks)
        4. BM25Index.add(chunks)
        5. backend.upsert_chunks(chunks)   ← no-op for LocalBackend
        6. Append to in-memory chunk store
        7. Append chunks to chunks.jsonl on disk (atomic line append)
        8. Release write lock.
        9. If FAISSIndex.needs_reindex: schedule _background_reindex() in thread.
        """

    def save(self, path: str) -> None:
        """
        Atomic save sequence:
        1. tmp_dir = path + "/.tmp_ks_save_" + uuid4().hex
        2. os.makedirs(tmp_dir)
        3. Write to tmp_dir: faiss/, bm25/, chunks.jsonl, config.json, version.txt
        4. os.replace(tmp_dir, path)   ← atomic rename on POSIX; best-effort on Windows

        version.txt contains: "0.1.0"
        config.json contains: KSelectConfig.model_dump_json()
        """

    def load(self, path: str) -> None:
        """
        Load sequence:
        1. Read version.txt. Check major version compatibility.
           Mismatch → raise KSelectVersionError.
        2. Load KSelectConfig from config.json.
        3. FAISSIndex.load(path), BM25Index.load(path).
        4. Read chunks.jsonl into in-memory chunk store.
        """

    def _background_reindex(self) -> None:
        """
        Runs in a daemon thread. Rebuilds FAISSIndex from full chunk store.
        Acquires write lock during swap. Logs start/end at INFO level.
        Resets FAISSIndex._original_size and _needs_reindex after completion.
        """

    def index_drift(self) -> float:
        """(current_size - original_size) / original_size. 0.0 if no adds since build."""

    def recall_estimate(self) -> float:
        """
        Heuristic: 1.0 at drift=0.0, degrades linearly to 0.90 at drift=0.20,
        then degrades faster. Not empirically calibrated — clearly documented as estimate.
        """

    def index_size(self) -> int:
        """Total vectors in FAISSIndex."""
```

---

## 9. Retrieval Layer

### 9.1 Fusion — `retrieval/fusion.py`

```python
def rrf(
    faiss_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke, Buettcher — SIGIR 2009).

    Algorithm:
      For each document d appearing in either list:
        score(d) = Σ 1 / (k + rank(d, list_i))
        where rank is 1-indexed position in the sorted list.
        If d does not appear in a list, it contributes 0 from that list.

    Inputs:
      faiss_results: list of (chunk_id, score) sorted by score desc.
      bm25_results:  list of (chunk_id, bm25_score) sorted by score desc.
      k: smoothing constant. Standard value is 60. Lower k = more aggressive
         rank boosting for top results. Do not tune unless benchmarked.
      top_n: truncate output. Default: max(len(faiss_results), len(bm25_results)).

    Returns: list of (chunk_id, rrf_score) sorted by rrf_score desc.
    Guarantees: no duplicate chunk_ids in output.
    """


def weighted_fusion(
    faiss_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """
    Weighted linear combination after min-max score normalization.

    Normalization per list:
      normalized_score = (score - min_score) / (max_score - min_score)
      If max == min: all scores = 1.0 (degenerate case, single result).

    Combined score = dense_weight * dense_norm + bm25_weight * bm25_norm.

    Raises ValueError if dense_weight + bm25_weight != 1.0 (tolerance 1e-6).
    Returns: list of (chunk_id, combined_score) sorted desc.
    """
```

### 9.2 `RetrievalEngine` — `retrieval/engine.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from kselect.index.manager import IndexManager
from kselect.models.config import KSelectConfig, FusionMode
from kselect.models.trace import QueryTrace
from kselect.retrieval.fusion import rrf, weighted_fusion


class RetrievalEngine:
    """
    Runs FAISS and BM25 searches in parallel, fuses results, returns candidates.
    Does NOT rank — that is entirely RankingEngine's responsibility.
    """

    OVER_FETCH_FACTOR: int = 3
    # Always retrieve k * OVER_FETCH_FACTOR candidates from each source.
    # The reranker receives these and returns final top-k.
    # This is the standard retrieve-then-rerank over-fetch pattern.

    def __init__(self, index_manager: IndexManager, config: KSelectConfig): ...

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int,
        fusion_mode: FusionMode,
        filters: dict | None = None,
        trace: QueryTrace | None = None,
    ) -> list[tuple[str, float]]:
        """
        Returns top k * OVER_FETCH_FACTOR candidates as (chunk_id, score).

        Flow:
        1. fetch_k = k * OVER_FETCH_FACTOR

        2. If fusion_mode == DENSE:
             faiss_results = index_manager.search_faiss(query_embedding, fetch_k)
             bm25_results = []

           Else (RRF or WEIGHTED):
             Run in parallel via ThreadPoolExecutor(max_workers=2):
               faiss_results = index_manager.search_faiss(query_embedding, fetch_k)
               bm25_results  = index_manager.search_bm25(query, fetch_k)

        3. Apply metadata filters (post-retrieval):
             Filter both result lists to only chunk_ids whose metadata
             matches all key-value pairs in filters.
             Note: FCVI index applies filters pre-retrieval (Phase 9).
             For all other index types, filtering is post-retrieval here.
             This means over-fetching is extra important when filters are active.

        4. Fuse:
             If RRF:      results = rrf(faiss_results, bm25_results, k=config.fusion.rrf_k)
             If WEIGHTED: results = weighted_fusion(faiss_results, bm25_results, ...)
             If DENSE:    results = faiss_results

        5. Truncate to fetch_k. Return.

        6. Populate trace fields if trace is not None:
             trace.faiss_candidates = len(faiss_results)
             trace.bm25_candidates  = len(bm25_results)
             trace.after_fusion     = len(results)
             trace.retrieval_latency_ms = elapsed
        """

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query string using the same embedding model as the index.
        L2-normalize the result before returning.
        Caches the embedding model at class level (load once, reuse across calls).
        """
```

---

## 10. Ranking Layer

### 10.1 `CrossEncoderReranker` — `ranking/cross_encoder.py`

```python
from sentence_transformers import CrossEncoder
from kselect.models.chunk import Chunk


class CrossEncoderReranker:
    """
    Wraps sentence_transformers.CrossEncoder.
    Model is loaded once at __init__ and reused across all queries.
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - 22M parameters, fast, strong on MSMARCO, good general baseline.

    Batching: score all (query, chunk_text) pairs in one forward pass.
    Batch size limited by available GPU/CPU memory. Default: all candidates at once.
    Log a warning if candidate count exceeds 200 (cross-encoder bottleneck territory).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"): ...

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],    # (chunk_id, fusion_score)
        chunk_store: dict[str, Chunk],            # chunk_id → Chunk (for text)
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        1. Build input pairs: [(query, chunk.text) for each candidate]
        2. CrossEncoder.predict(pairs) → scores array
        3. Sort candidates by cross-encoder score descending
        4. Return top_k as list of (chunk_id, cross_encoder_score)

        If a chunk_id is in candidates but not in chunk_store,
        skip it and log a WARNING. Do not raise.
        """
```

### 10.2 `ColBERTReranker` — `ranking/colbert.py`

```python
class ColBERTReranker:
    """
    Late interaction reranker using token-level embeddings.
    Uses colbert-ai library (installed via kselect[rerank]).

    If colbert-ai is not installed: log a WARNING and fall back to
    CrossEncoderReranker automatically. Do not raise.

    Late interaction (MaxSim):
      For each document d:
        score(q, d) = Σ_i max_j sim(q_i, d_j)
      where q_i = query token embeddings, d_j = document token embeddings.

    Returns same interface as CrossEncoderReranker:
      list[tuple[str, float]] (chunk_id, score) sorted desc.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"): ...

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        chunk_store: dict[str, Chunk],
        top_k: int,
    ) -> list[tuple[str, float]]: ...
```

### 10.3 MMR — `ranking/mmr.py`

```python
import numpy as np


def mmr_diversify(
    candidates: list[tuple[str, float]],       # (chunk_id, score) post-rerank, sorted desc
    chunk_embeddings: dict[str, np.ndarray],    # chunk_id → L2-normalized embedding
    top_k: int,
    lambda_: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    Iteratively selects the next chunk that maximizes:
      MMR(d) = lambda_ * relevance(d) - (1 - lambda_) * max_sim(d, selected)

    where:
      relevance(d) = rerank score (from input candidates, normalized to [0,1])
      max_sim(d, selected) = max cosine similarity between d and any already-selected chunk

    Parameters:
      lambda_ = 1.0 → pure relevance ranking (MMR = no-op)
      lambda_ = 0.0 → pure diversity (max dissimilarity)
      lambda_ = 0.5 → balanced (default)

    Complexity: O(top_k * n) where n = len(candidates). Fine for top_k ≤ 100.

    Returns: list of (chunk_id, mmr_score) length top_k, in selection order.
    If a chunk_id has no entry in chunk_embeddings, assign max_sim = 0.0
    (treat as maximally diverse from selected set). Log WARNING.
    """
```

### 10.4 Ranking Dispatch Table

The ranking mode selected by the caller determines which components run. This table is authoritative — implement search() logic exactly as specified here.

```
RankingMode.HYBRID
  Stage 1: RetrievalEngine.retrieve()     (FAISS + BM25 + RRF)
  Stage 2: CrossEncoderReranker.rerank()
  Stage 3: mmr_diversify()
  Returns: top-k after MMR

RankingMode.COLBERT
  Stage 1: RetrievalEngine.retrieve()     (FAISS + BM25 + RRF)
  Stage 2: ColBERTReranker.rerank()
  Stage 3: (skip MMR)
  Returns: top-k after ColBERT

RankingMode.FAST
  Stage 1: RetrievalEngine.retrieve()     (FAISS + BM25 + RRF)
  Stage 2: (skip reranking)
  Stage 3: (skip MMR)
  Returns: top-k from fusion directly

RankingMode.CROSS
  Stage 1: (skip RetrievalEngine entirely)
           Retrieve ALL chunks from index (up to 50K limit — raise KSelectIndexError above)
  Stage 2: CrossEncoderReranker.rerank() on all chunks
  Stage 3: (skip MMR)
  Returns: top-k from cross-encoder

RankingMode.NONE
  Stage 1: RetrievalEngine.retrieve()     (FAISS only, no BM25)
  Stage 2: (skip reranking)
  Stage 3: (skip MMR)
  Returns: raw FAISS top-k
```

**Important:** `hybrid=True` on `KSelect.search()` means `RankingMode.HYBRID` (cross-encoder + MMR). It does **not** control BM25 fusion. BM25 fusion is controlled by `config.bm25.enabled`. These are independent axes.

---

### Phase 3–5 test targets

```python
# tests/unit/test_fusion.py

def test_rrf_no_duplicates():
    """RRF output contains no duplicate chunk_ids."""

def test_rrf_sorted_desc():
    """RRF output is sorted by rrf_score descending."""

def test_rrf_doc_in_one_list_only():
    """Doc appearing in only one input list still appears in output with correct score."""

def test_weighted_fusion_sum_to_one():
    """weighted_fusion raises ValueError when weights don't sum to 1.0."""

def test_mmr_lambda_one():
    """MMR with lambda_=1.0 returns same order as input (no reranking)."""

def test_mmr_lambda_zero():
    """MMR with lambda_=0.0 selects maximally diverse chunks."""

# tests/unit/test_index.py

def test_faiss_build_search_roundtrip():
    """Build index from 100 chunks, search returns expected chunk_id at rank 1."""

def test_faiss_save_load():
    """Save and reload FAISSIndex; search results identical before and after."""

def test_faiss_drift_threshold():
    """add() sets needs_reindex=True after 20% of original size is added."""

def test_bm25_exact_term_retrieval():
    """BM25 ranks chunk with exact query term at position 0."""

# tests/integration/test_search.py

def test_search_fast_mode_returns_search_result():
    """search() with fast=True returns SearchResult with correct types."""

def test_search_hybrid_scores_differ_from_fast():
    """HYBRID mode scores differ from FAST mode scores on same query."""

def test_search_filters_applied():
    """Metadata filter excludes chunks with non-matching metadata values."""
```

---

*Continue to SPEC-05-core-class-infra.md*
