# KSelect — SPEC Part 5 of 5: Core Class, Infrastructure, and Implementation Order

**Read order:** Final file. Covers the main `KSelect` public class, `SemanticCache`, `ContextAssembler`, `MultiTenantKSelect`, observability, persistence layout, YAML config reference, error taxonomy, naming disambiguation, and the full implementation order.

---

## 11. Context Layer

### `ContextAssembler` — `context/assembler.py`

```python
import tiktoken
from kselect.models.hit import Hit
from kselect.models.config import ContextConfig, ContextStrategy


class ContextAssembler:
    """
    Takes ranked hits and constructs the final context window for the LLM.
    Called only by query(), never by search().

    Token counting uses tiktoken cl100k_base encoding.
    Results are cached per chunk_id to avoid re-encoding the same text.
    """

    def assemble(
        self,
        hits: list[Hit],
        config: ContextConfig,
    ) -> tuple[list[Hit], int]:
        """
        Returns (final_chunks, total_token_count).

        SCORE_ORDER:
          Take hits in score order. Drop trailing hits once token budget exhausted.

        LOST_IN_MIDDLE:
          Reorder hits to counteract the lost-in-the-middle attention degradation
          (Liu et al., 2023). Place most relevant chunk at position 0 and
          second-most-relevant at position -1. Interleave remaining chunks:
            [rank_1, rank_3, rank_5, ..., rank_6, rank_4, rank_2]
          Then apply token budget truncation from the middle outward.

        TRUNCATE:
          Drop lowest-scored hits (from the tail) until within token budget.
          Identical to SCORE_ORDER but explicit about the truncation policy.

        SUMMARIZE_OVERFLOW:
          Keep as many top-ranked hits as fit in 75% of max_context_tokens.
          Compress remaining hits into a summary paragraph using a short LLM call
          (max_tokens=256). Append summary as final context item.
          Requires LLMClient to be wired in. If not available, falls back to TRUNCATE.
        """

    def _count_tokens(self, text: str) -> int:
        """
        tiktoken.get_encoding("cl100k_base").encode(text).
        Cached in self._token_cache: dict[str, int] keyed by chunk_id.
        Cache is per-instance — cleared on assembler construction.
        """
```

---

## 12. Cache Layer

### `SemanticCache` — `cache/semantic_cache.py`

```python
import time
import faiss
import numpy as np
from collections import deque
from pydantic import BaseModel
from kselect.models.hit import SearchResult
from kselect.models.answer import QueryResult
from kselect.models.cache import CacheStats


class SemanticCache:
    """
    Embedding-based semantic query cache.

    Architecture:
      - Backed by a small faiss.IndexFlatIP (separate from the document index).
      - This index stores query embeddings, NOT document embeddings.
      - Cache keys = query embeddings (vectors).
      - Cache values = CacheEntry objects stored in self._entries: dict[int, CacheEntry]
        where int is the FAISS positional index.

    TTL: enforced lazily at read time. No background cleanup thread.
    LRU eviction: tracked via collections.deque of FAISS positions.
      On max_size reached: evict the position at the front of the deque.
      FAISS does not support deletion — eviction is logical (mark position as invalid).
      Rebuild the FAISS index from scratch when >50% of positions are evicted.

    Thread safety: protect with threading.RLock() for all public methods.
    """

    class CacheEntry(BaseModel):
        query: str
        query_embedding: list[float]
        result: SearchResult | QueryResult
        created_at: float           # time.time() at insertion
        hit_count: int = 0

    def __init__(self, config: "CacheConfig", cross_encoder: "CrossEncoderReranker | None" = None):
        """
        cross_encoder: used for borderline verification.
        If None and verify_borderline=True: skip verification, treat as cache miss.
        """

    def get(
        self,
        query_embedding: np.ndarray,
        query: str,
        config: "CacheConfig",
    ) -> SearchResult | QueryResult | None:
        """
        1. Search FAISS index for nearest neighbor to query_embedding.
           If index is empty, return None immediately.

        2. If similarity >= config.similarity_threshold:
             Check TTL: if (now - entry.created_at) > config.ttl_seconds → miss (evict entry).
             Else: increment entry.hit_count, update stats, return entry.result.

        3. If config.verify_borderline and config.borderline_lower <= similarity < config.similarity_threshold:
             Run CrossEncoder.predict([(query, entry.query)]).
             If cross_encoder_score > 0.85 → treat as cache hit (return entry.result).
             Else → miss.

        4. Otherwise: return None (cache miss).
        """

    def set(
        self,
        query_embedding: np.ndarray,
        query: str,
        result: SearchResult | QueryResult,
    ) -> None:
        """
        Add new entry to cache.
        If at max_size: evict LRU entry (front of deque).
        Add embedding to FAISS index.
        Store CacheEntry in self._entries.
        Append FAISS position to deque.
        """

    def stats(self) -> CacheStats:
        """
        Compute from accumulated counters:
          hit_rate = hits / (hits + misses)
          false_positive_rate = false_positives / (hits + false_positives)
          llm_calls_saved = hits  (each hit skips one LLM call)
          cost_saved_usd = llm_calls_saved * estimated_cost_per_call
            (estimated_cost_per_call = 0.001 USD default; not configurable via public API)
        """
```

---

## 13. Main KSelect Class

### `kselect/kselect.py`

This is the only module users import. All internal classes are private. Never expose `FAISSIndex`, `BM25Index`, `RetrievalEngine`, etc. in the public API.

```python
from __future__ import annotations
from typing import Any
from kselect.models.config import (
    KSelectConfig, RankingMode, FusionMode, ContextStrategy
)
from kselect.models.hit import SearchResult
from kselect.models.answer import QueryResult
from kselect.models.cache import CacheStats


class KSelect:
    """
    Public entry point. Do not call __init__ directly.
    Use classmethods: from_folder, from_csv, from_json, from_jsonl,
                      from_backend, from_yaml, load.
    """

    def __init__(
        self,
        config: KSelectConfig,
        _index_manager: "IndexManager",        # internal; not part of public API
        _cache: "SemanticCache | None" = None,
        _llm: "LLMClient | None" = None,
        _metrics: "KSelectMetrics | None" = None,
    ):
        """Private. Use classmethods."""

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_folder(
        cls,
        path: str,
        *,
        config: KSelectConfig | None = None,
        index_type: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking: str | None = None,
        hybrid_fcvi: bool | None = None,
        bm25: bool | None = None,
        metadata_fields: list[str] | None = None,
        batch_size: int | None = None,
        extract_tables: bool | None = None,
        max_docs: int | None = None,
        max_memory_gb: float | None = None,
    ) -> "KSelect":
        """
        Build from a folder of documents.

        Convenience kwargs (index_type, embedding_model, etc.) override the
        corresponding fields in config (or the default config if config=None).
        This allows one-liner usage: KSelect.from_folder("docs/", chunk_size=256)
        without constructing a KSelectConfig manually.

        Internal flow:
        1. Build or clone config; apply any provided kwargs.
        2. Instantiate LocalBackend(tmp state dir).
        3. FolderLoader(path, ...) → IngestionPipeline.run() → list[Chunk]
        4. IndexManager.build(chunks)
        5. Return KSelect(config, index_manager)
        """

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        text_col: str,
        metadata: list[str] | None = None,
        vector_col: str | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        """
        vector_col: if provided, read embeddings from this column.
        IngestionPipeline skips embedding step when vector_col is present.
        Raises KSelectIngestionError if vector_col values are not parseable as list[float].
        """

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        text_key: str,
        metadata: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect": ...

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        *,
        text_key: str,
        metadata: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect": ...

    @classmethod
    def from_backend(
        cls,
        uri: str,
        *,
        text_col: str = "content",
        metadata_cols: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        """
        Wrap an existing vector database. Does NOT re-embed existing vectors.

        Flow:
        1. parse_backend_uri(uri) → VectorBackend
        2. backend.get_all_chunks() → list[Chunk] (streamed in batches)
        3. IndexManager.build(chunks)  ← builds FAISS and BM25 from existing embeddings
        4. Return KSelect(config, index_manager)

        For pgvector: streams in batches of 10,000 via server-side cursor.
        Progress logged at INFO every 100,000 chunks.
        """

    @classmethod
    def from_yaml(cls, path: str) -> "KSelect":
        """
        Load KSelectConfig from YAML.
        If config.backend is set: call from_backend(config.backend.uri, config=config).
        Otherwise: raise KSelectConfigError("from_yaml requires backend.uri to be set.
                   For local folders, use from_folder() directly.")
        """

    @classmethod
    def load(cls, path: str) -> "KSelect":
        """
        Load previously saved KSelect state.
        Instantiate IndexManager.load(path).
        Instantiate SemanticCache if cache/ directory exists in path.
        """

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Delegate to self._index_manager.save(path). Atomic write."""

    # ── Incremental ingestion ─────────────────────────────────────────────────

    def add_doc(self, path: str, metadata: dict[str, Any] | None = None) -> int:
        """
        Add a single document. Returns number of new chunks added.

        Flow:
        1. FolderLoader([path], extract_tables=config.chunking.extract_tables)
        2. Apply metadata override if provided (merged into loader-extracted metadata).
        3. IngestionPipeline.run(loader, config, existing_chunk_ids=self._chunk_ids())
        4. If len(chunks) == 0: log WARNING "No new chunks from {path}"; return 0.
        5. self._index_manager.add_chunks(chunks)
        6. Return len(chunks)
        """

    def add_folder(self, path: str, metadata: dict[str, Any] | None = None) -> int:
        """Add all documents in folder. Returns total new chunks added."""

    # ── Search (no LLM) ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        hybrid: bool = False,
        colbert: bool = False,
        fast: bool = False,
        cross: bool = False,
        none: bool = False,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        trace: bool | None = None,
    ) -> SearchResult:
        """
        Retrieve and rank. Does NOT call LLM.

        Ranking mode resolution (strictly in this priority order):
          1. If exactly one of [hybrid, colbert, fast, cross, none] is True → use that mode.
          2. If config.ranking.mode is set → use it.
          3. Default: RankingMode.HYBRID.
          4. If more than one bool flag is True → raise KSelectConfigError.

        Fusion mode resolution:
          1. If fusion kwarg is provided → parse as FusionMode string.
          2. Else use config.fusion.mode.

        SemanticCache check:
          1. Embed query.
          2. If cache enabled: cache.get(embedding, query, config.cache)
             If hit: return cached SearchResult (update hit count).
          3. Run retrieval + ranking pipeline.
          4. If cache enabled: cache.set(embedding, query, result).
          5. Return result.

        trace flag resolution:
          trace kwarg → config.observability.trace → False (default).
        """

    async def search_async(self, query: str, **kwargs) -> SearchResult:
        """
        Async variant. Runs search() in a thread via anyio.to_thread.run_sync.
        Safe to use in FastAPI route handlers.
        """

    # ── Query (search + LLM) ──────────────────────────────────────────────────

    def query(
        self,
        query: str,
        *,
        k: int | None = None,
        hybrid: bool = False,
        colbert: bool = False,
        fast: bool = False,
        cross: bool = False,
        none: bool = False,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        context_strategy: str | None = None,
        max_context_tokens: int | None = None,
        return_context: bool = False,
        model: str | None = None,
        trace: bool | None = None,
    ) -> QueryResult:
        """
        Full RAG pipeline: search() → ContextAssembler → LLMClient.

        SemanticCache wraps the full pipeline:
          - Cache hit: return cached QueryResult immediately (skip all stages).
          - Cache miss: run full pipeline, then cache.set(embedding, query, result).

        Flow:
        1. Embed query.
        2. Check cache (if enabled).
        3. Run search() internally (reuses embedding, skips re-embed).
        4. ContextAssembler.assemble(hits, context_config).
        5. LLMClient.generate(query, context_chunks, max_tokens).
        6. Build QueryResult from all collected data.
        7. Write to cache (if enabled).
        8. Return QueryResult.

        model kwarg overrides config.llm.model for this call only.
        context_strategy kwarg overrides config.context.strategy for this call only.
        max_context_tokens kwarg overrides config.context.max_context_tokens.
        """

    async def query_async(self, query: str, **kwargs) -> QueryResult:
        """Async variant. LLMClient.generate() is awaited directly (it is already async)."""

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def index_size(self) -> int:
        """Total vectors in the FAISS index."""

    def index_drift(self) -> float:
        """Fraction of index that has been added since last full build. 0.0–1.0."""

    def recall_estimate(self) -> float:
        """Heuristic recall estimate based on drift. 1.0 at drift=0, degrades above 0.20."""

    def cache_stats(self) -> CacheStats:
        """Returns CacheStats. All zeros if cache is disabled."""

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataset: list[dict],
        strategies: list[dict] | None = None,
    ) -> list["EvalMetrics"]:
        """
        dataset: list of {"query": str, "gold_docs": list[str]}
          gold_docs: list of source_file values that are relevant for this query.

        strategies: list of config override dicts to compare.
          Each dict is applied as config overrides (same syntax as from_folder kwargs).
          If None: evaluate current config only.

        Returns one EvalMetrics per strategy.
        """
```

---

## 14. Multi-Tenant Extension

### `kselect/multi_tenant.py`

```python
from kselect.kselect import KSelect
from kselect.models.config import KSelectConfig
from kselect.models.hit import SearchResult
from kselect.models.answer import QueryResult


class MultiTenantKSelect:
    """
    Manages N independent KSelect instances, one per tenant.

    The embedding model is shared across all tenants (loaded once, reused).
    This is the primary memory optimization — embedding models are 200MB–2GB.
    Each tenant has its own FAISSIndex, BM25Index, and SemanticCache.

    query() and search() require a tenant: str argument.
    Unknown tenant raises KSelectTenantError.
    """

    @classmethod
    def from_yaml(cls, path: str) -> "MultiTenantKSelect":
        """
        YAML format (see Section 17 for full reference):
          tenants:
            acme:
              backend_uri: "pgvector://prod-db/acme_docs"
              index_type: "faiss_vlq_adc"
            globex:
              backend_uri: "pgvector://prod-db/globex_docs"

        Each tenant entry is merged with global config defaults.
        Tenants are loaded sequentially at startup (not lazy).
        """

    def search(self, tenant: str, query: str, **kwargs) -> SearchResult:
        """Delegates to self._tenants[tenant].search(query, **kwargs)."""

    def query(self, tenant: str, query: str, **kwargs) -> QueryResult:
        """Delegates to self._tenants[tenant].query(query, **kwargs)."""

    def add_tenant(self, tenant_id: str, config: KSelectConfig) -> None:
        """
        Hot-add a new tenant. Builds indexes from config.backend_uri.
        Raises KSelectTenantError if tenant_id already exists.
        Thread-safe: acquires write lock before adding to self._tenants.
        """

    def remove_tenant(self, tenant_id: str) -> None:
        """
        Remove tenant from routing table. Does NOT delete underlying data.
        Raises KSelectTenantError if tenant_id not found.
        """

    def tenants(self) -> list[str]:
        """Return sorted list of active tenant IDs."""
```

---

## 15. Observability

### `observability/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge


class KSelectMetrics:
    """
    All Prometheus metrics use the 'kselect_' prefix.
    Instantiate once and pass to KSelect via constructor.
    If not passed, KSelect uses a NullMetrics no-op emitter —
    prometheus_client is never imported unless KSelectMetrics is used.
    """

    query_total = Counter(
        "kselect_query_total",
        "Total queries processed",
        ["mode", "cache_hit"],        # labels: mode=hybrid|fast|..., cache_hit=true|false
    )
    query_latency_ms = Histogram(
        "kselect_query_latency_ms",
        "End-to-end query latency in milliseconds",
        buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
    )
    retrieval_latency_ms = Histogram(
        "kselect_retrieval_latency_ms",
        "FAISS + BM25 retrieval latency",
        buckets=[1, 2, 5, 10, 25, 50, 100, 250],
    )
    rerank_latency_ms = Histogram(
        "kselect_rerank_latency_ms",
        "Cross-encoder / ColBERT reranker latency",
        buckets=[5, 10, 25, 50, 100, 250, 500],
    )
    cache_hit_rate  = Gauge("kselect_cache_hit_rate", "Rolling 1000-query cache hit rate")
    confidence_p50  = Gauge("kselect_confidence_p50", "Rolling p50 answer confidence")
    confidence_p95  = Gauge("kselect_confidence_p95", "Rolling p95 answer confidence")
    index_size      = Gauge("kselect_index_size", "Total vectors in FAISS index")
    index_drift     = Gauge("kselect_index_drift", "Index drift ratio (0.0–1.0)")
```

### `observability/logging.py`

```python
"""
Structured logging via structlog.

Configuration:
  KSELECT_LOG_FORMAT env var:
    "json"    → structlog JSONRenderer (production)
    "console" → structlog ConsoleRenderer with colors (development, default)

Every search() and query() call emits one structured log event at INFO level:
{
    "event":              "search" | "query",
    "query":              "<truncated at 200 chars>",
    "mode":               "hybrid" | "fast" | ...,
    "fusion":             "rrf" | "weighted" | "dense",
    "cache_hit":          true | false,
    "latency_ms":         42.3,
    "confidence":         0.91,          # query() only
    "chunks_retrieved":   60,
    "chunks_in_context":  18,            # query() only
    "index_drift":        0.03
}

Errors are logged at ERROR level with exc_info=True.
No PII is logged. query text is truncated to 200 chars.
"""
```

---

## 16. Serialization and Persistence

### Saved state directory layout

```
kselect_state/
├── version.txt              # "0.1.0" — checked on load
├── config.json              # KSelectConfig.model_dump_json()
├── faiss/
│   ├── index.faiss          # faiss.write_index() output
│   ├── id_map.json          # list[str]: position i → chunk_id
│   └── meta.json            # {"original_size": N, "index_type": "faiss_vlq_adc"}
├── bm25/
│   ├── model.pkl            # pickle.dump(bm25s.BM25 object)
│   └── id_map.json          # list[str]: position i → chunk_id
├── chunks.jsonl             # one Chunk.model_dump_json() per line
│                            # source of truth for chunk text and metadata
└── cache/                   # present only if cache was enabled and populated
    ├── index.faiss          # SemanticCache FAISS index
    └── entries.jsonl        # one CacheEntry.model_dump_json() per line
```

### Version compatibility rules

`KSelect.load()` reads `version.txt` and applies these rules:

| Situation | Action |
|---|---|
| Major version mismatch (e.g., saved 0.x, loading with 1.x) | Raise `KSelectVersionError` |
| Minor version mismatch | Log WARNING, attempt load |
| Patch version difference | Compatible, load silently |
| `version.txt` missing | Raise `KSelectVersionError` (corrupted state) |

### Atomicity guarantee

`IndexManager.save()` writes to a temp directory first, then uses `os.replace()` for an atomic rename. On POSIX systems, this is guaranteed atomic. On Windows, `os.replace()` is best-effort atomic. The state directory is never partially written.

---

## 17. Configuration (YAML) Full Reference

```yaml
# kselect.yaml — all fields optional; defaults shown

backend:
  uri: "pgvector://prod-db/legal_cases"   # omit for local-only builds
  text_col: "content"
  metadata_cols: ["client_id", "jurisdiction", "case_number"]

index:
  type: "faiss_vlq_adc"      # faiss_flat | faiss_ivf_pq128 | faiss_hnsw_sq | faiss_fcvi | faiss_vlq_adc
  hybrid_fcvi: true
  nlist: 1024
  m: 64
  hnsw_m: 32

chunking:
  strategy: "sliding_window" # sliding_window | sentence | semantic | paragraph
  chunk_size: 512
  chunk_overlap: 64
  semantic_threshold: 0.75
  min_chunk_length: 50
  remove_duplicates: false
  extract_tables: false

embedding:
  model: "nomic-embed-text-v1.5"
  api: null                   # null (local) | "openai"
  api_key: "${EMBEDDING_API_KEY}"
  batch_size: 256
  normalize: true

bm25:
  enabled: true
  k1: 1.5
  b: 0.75

fusion:
  mode: "rrf"                 # rrf | weighted | dense
  rrf_k: 60
  bm25_weight: 0.3
  dense_weight: 0.7

ranking:
  mode: "hybrid"              # hybrid | colbert | fast | cross | none
  k: 20
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  colbert_model: "colbert-ir/colbertv2.0"
  mmr_lambda: 0.5

context:
  strategy: "lost_in_middle"  # score_order | lost_in_middle | truncate | summarize_overflow
  max_context_tokens: 4096
  return_context: false

cache:
  enabled: false
  similarity_threshold: 0.97
  verify_borderline: true
  borderline_lower: 0.93
  ttl_seconds: 3600
  max_size: 10000

llm:
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  base_url: null              # set for OpenAI-compatible endpoints
  temperature: 0.0
  max_tokens: 1024

observability:
  log_queries: false
  metrics_port: null          # set to 9090 to start Prometheus HTTP server
  trace: false
```

---

## 18. Error Taxonomy

All KSelect exceptions inherit from `KSelectError`. Never raise raw `ValueError`, `RuntimeError`, or `Exception` across module boundaries — always wrap in a domain exception.

```python
# kselect/exceptions.py

class KSelectError(Exception):
    """Base exception for all KSelect errors."""


class KSelectConfigError(KSelectError):
    """
    Invalid configuration detected at construction time.
    Examples: unknown backend scheme, missing required field,
    multiple ranking mode flags set to True simultaneously.
    """


class KSelectIndexError(KSelectError):
    """
    Index build, search, or save/load failure.
    Examples: embedding dimension mismatch, FAISS OOM, corrupted index file.
    """


class KSelectBackendError(KSelectError):
    """
    Backend connection or query failure.
    Wraps the original upstream exception as __cause__.
    Message must include: backend type, operation attempted, original error string.
    """
    def __init__(self, message: str, cause: Exception):
        super().__init__(message)
        self.__cause__ = cause


class KSelectIngestionError(KSelectError):
    """
    File parsing or chunking failure.
    Message must include the file path.
    Raised by loaders and IngestionPipeline.
    """


class KSelectEmbeddingError(KSelectError):
    """
    Embedding model load or inference failure.
    Examples: model not found, CUDA OOM during batch embed.
    """


class KSelectRankingError(KSelectError):
    """
    Reranker model load or inference failure.
    Examples: cross-encoder model not found, colbert-ai not installed.
    """


class KSelectLLMError(KSelectError):
    """
    LLM API call failure.
    Message must include HTTP status code when available.
    Raised by LLMClient.generate().
    """


class KSelectTenantError(KSelectError):
    """
    Unknown or misconfigured tenant in MultiTenantKSelect.
    Examples: tenant_id not in self._tenants, duplicate tenant on add_tenant().
    """


class KSelectVersionError(KSelectError):
    """
    Incompatible saved state version detected during load().
    Message must include: saved version, current version.
    """
```

---

## 19. Naming Disambiguation

This section is authoritative. When the README and this spec conflict on naming, **this spec wins for all internal implementation**.

| README term | Internal (spec) name | Notes |
|---|---|---|
| `RAG` class | `KSelect` class | One class. It is `KSelect`. There is no `RAG` class. |
| `rag.search()` | `KSelect.search()` | Returns `SearchResult` |
| `rag.answer()` | `KSelect.query()` | Returns `QueryResult` |
| `result.text` | `result.answer` | LLM output field is `.answer` |
| `smart_rank=True` | `hybrid=True` | `smart_rank` is retired. Do not implement it. |
| `hybrid=True` in ranking section | `RankingMode.HYBRID` | Means cross-encoder + MMR |
| `hybrid_fcvi=True` in index section | `config.index.hybrid_fcvi = True` | FCVI index option; separate from ranking |
| BM25 "hybrid" fusion | `FusionMode.RRF` | BM25+dense fusion is a retrieval-layer concern |
| `index_type="faiss_vlq_adc"` | `config.index.type = IndexType.VLQ_ADC` | String values accepted as convenience in `from_folder()` kwargs |

**Critical distinction — `hybrid` is overloaded:**
- `KSelect.search(hybrid=True)` → `RankingMode.HYBRID` (cross-encoder + MMR applied after retrieval)
- `config.bm25.enabled = True` → BM25+dense fusion active during retrieval
- These are orthogonal. `hybrid=True` on the query does NOT turn on BM25 fusion.
  BM25 fusion is always on when `config.bm25.enabled = True` (default: `True`).
  To disable BM25 per-query: pass `fusion="dense"`.

---

## 20. Implementation Order

Build in this sequence. **Do not skip phases.** Each phase ends with a passing test suite before the next phase begins.

---

**Phase 1 — Data models and config (no logic, no I/O)**

Files: `models/`, `exceptions.py`

```
models/chunk.py       Chunk, ChunkMetadata
models/hit.py         Hit, SearchResult
models/answer.py      QueryResult, Source
models/trace.py       QueryTrace
models/cache.py       CacheStats
models/config.py      all enums, all sub-configs, KSelectConfig
exceptions.py         full error taxonomy
```

Test target: `KSelectConfig.from_yaml()` round-trips cleanly. All validators pass on valid input. All validators raise on invalid input.

---

**Phase 2 — Ingestion (local files only)**

Files: `backends/local.py`, `ingestion/`

```
backends/local.py
ingestion/loaders.py     FolderLoader (PDF + DOCX + TXT), CSVLoader, JSONLoader
ingestion/chunking.py    sliding_window + sentence first; semantic + paragraph after Phase 5
ingestion/pipeline.py
```

Test target: Ingest a folder of mixed files. Assert `list[Chunk]` returned. Assert all chunks have `embedding` populated. Assert metadata fields present.

---

**Phase 3 — Index (FLAT and IVF_PQ128 only)**

Files: `index/faiss_index.py`, `index/bm25_index.py`, `index/manager.py`

```
index/faiss_index.py    FLAT and IVF_PQ128; stub VLQ_ADC and FCVI
index/bm25_index.py
index/manager.py        build, save, load, index_size only
```

Test target: Build from 1,000 chunks. Save. Load. Assert sizes identical. Assert search returns sorted results.

---

**Phase 4 — Retrieval and fusion**

Files: `retrieval/fusion.py`, `retrieval/engine.py`

```
retrieval/fusion.py     rrf(), weighted_fusion()
retrieval/engine.py     FAST mode (no BM25); add BM25 parallel search afterward
```

Test target: RRF output sorted, no duplicates. FAST mode returns `list[tuple[str, float]]`. BM25 parallel search completes without race condition.

---

**Phase 5 — Ranking and `KSelect.search()`**

Files: `ranking/cross_encoder.py`, `ranking/mmr.py`, `kselect/kselect.py` (search only)

```
ranking/cross_encoder.py
ranking/mmr.py
kselect.py               from_folder() + search() only; stub query()
```

Test target: `search()` returns `SearchResult`. HYBRID mode scores differ from FAST mode. Filters correctly exclude non-matching chunks. RankingMode dispatch table matches Section 10.4 exactly.

---

**Phase 6 — LLM, context, and `KSelect.query()`**

Files: `llm/base.py`, `llm/openai_client.py`, `context/assembler.py`, `kselect.py` (query)

```
llm/base.py
llm/openai_client.py
context/assembler.py     SCORE_ORDER and LOST_IN_MIDDLE first; SUMMARIZE_OVERFLOW last
kselect.py               query() completed
```

Test target: Mock LLM client. `query()` returns `QueryResult` with all fields populated. `chunks_in_context` ≤ `chunks_retrieved`. `context_tokens` ≤ `max_context_tokens`.

---

**Phase 7 — Semantic cache**

Files: `cache/semantic_cache.py`, wire into `KSelect.search()` and `KSelect.query()`

Test target: Identical query → cache hit on second call. Paraphrase query at threshold → cache hit. Clearly different query → cache miss. TTL expiry → cache miss after TTL.

---

**Phase 8 — Remote backends**

Files: `backends/pgvector.py`, `backends/pinecone.py`, `backends/chromadb.py`, `backends/factory.py`

Test target: Integration test against local Postgres + pgvector extension. `from_backend("pgvector://...")` builds correct index. `add_doc()` writes to both FAISS and the DB.

---

**Phase 9 — Advanced features**

Files: `index/faiss_index.py` (VLQ_ADC, FCVI), `ranking/colbert.py`, `ingestion/chunking.py` (semantic strategy)

Test target: VLQ_ADC search results comparable to IVF_PQ128. ColBERT falls back to cross-encoder when colbert-ai not installed.

---

**Phase 10 — Observability, multi-tenant, and eval**

Files: `observability/`, `multi_tenant.py`, `eval/`

Test target: Prometheus metrics emitted after 10 queries. MultiTenantKSelect routes correctly. `evaluate()` returns `EvalMetrics` with `recall_at_10` populated.

---

*End of SPEC — all 5 parts.*
