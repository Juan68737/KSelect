# KSelect — SPEC Part 2 of 5: Data Models and Configuration

**Read order:** Read after SPEC-01. Implement everything in this file first — no logic, only types. Every other module imports from `models/`. Nothing in `models/` imports from anywhere else in the codebase.

**Build this entire file before writing any other module.**

---

## 4. Data Models (Pydantic)

All data flowing between layers is typed with Pydantic v2 models. No `dict` passing across module boundaries.

---

### 4.1 `models/chunk.py`

```python
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Arbitrary user-defined metadata attached to a chunk at ingestion time.
    All values must be JSON-serializable (str, int, float, bool, None).
    """
    model_config = {"extra": "allow"}

    source_file: str
    chunk_index: int                        # position within parent document
    char_start: int
    char_end: int
    token_count: int
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """
    The atomic unit produced by IngestionPipeline and consumed by IndexManager.
    Each Chunk maps 1:1 to one FAISS vector and one BM25 document.
    """
    id: str                                 # uuid4, assigned at ingestion
    text: str                               # raw text of this chunk
    embedding: list[float] | None = None    # populated after embed(); None before
    metadata: ChunkMetadata
```

---

### 4.2 `models/hit.py`

```python
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Hit(BaseModel):
    """
    One ranked result returned by search().
    """
    chunk_id: str
    doc_id: str                             # source_file from ChunkMetadata
    score: float                            # final score after all ranking stages
    snippet: str                            # chunk.text (or truncated)
    metadata: dict[str, Any]               # passthrough of ChunkMetadata.extra
    rank: int                               # 1-indexed position in result list

    # Diagnostic fields — populated when trace=True
    faiss_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class SearchResult(BaseModel):
    """
    Return type of KSelect.search().
    """
    hits: list[Hit]
    total_hits: int
    query: str
    trace: "QueryTrace | None" = None
```

---

### 4.3 `models/answer.py`

```python
from __future__ import annotations
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    One cited source in a QueryResult.
    """
    chunk_id: str
    doc_id: str
    snippet: str
    score: float
    metadata: dict


class QueryResult(BaseModel):
    """
    Return type of KSelect.query(). Extends SearchResult with LLM output.
    """
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source]
    query: str

    # Context window diagnostics
    chunks_retrieved: int
    chunks_in_context: int
    chunks_dropped: int
    context_tokens: int
    max_context_tokens: int

    trace: "QueryTrace | None" = None
```

---

### 4.4 `models/trace.py`

```python
from __future__ import annotations
from pydantic import BaseModel


class QueryTrace(BaseModel):
    """
    Per-query timing and diagnostic data. Only populated when trace=True.
    """
    cache_hit: bool = False
    cache_similarity: float | None = None

    retrieval_latency_ms: float = 0.0       # FAISS + BM25 parallel search
    fusion_latency_ms: float = 0.0          # RRF or weighted fusion
    rerank_latency_ms: float = 0.0          # cross-encoder / ColBERT
    mmr_latency_ms: float = 0.0
    context_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    faiss_candidates: int = 0
    bm25_candidates: int = 0
    after_fusion: int = 0
    after_rerank: int = 0
    after_mmr: int = 0
    chunks_in_context: int = 0

    embedding_model: str = ""
    index_type: str = ""
    ranking_mode: str = ""
    fusion_mode: str = ""
```

---

### 4.5 `models/cache.py`

```python
from __future__ import annotations
from pydantic import BaseModel


class CacheStats(BaseModel):
    hit_rate: float
    false_positive_rate: float
    size: int
    llm_calls_saved: int
    cost_saved_usd: float
    qps_saved: float
```

---

### 4.6 `models/config.py`

```python
from __future__ import annotations
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class IndexType(StrEnum):
    FLAT         = "faiss_flat"
    IVF_PQ128    = "faiss_ivf_pq128"
    HNSW_SQ      = "faiss_hnsw_sq"
    FCVI         = "faiss_fcvi"
    VLQ_ADC      = "faiss_vlq_adc"


class ChunkingStrategy(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    SENTENCE       = "sentence"
    SEMANTIC       = "semantic"
    PARAGRAPH      = "paragraph"


class RankingMode(StrEnum):
    HYBRID  = "hybrid"       # FAISS + BM25 + cross-encoder + MMR
    COLBERT = "colbert"      # FAISS + BM25 + ColBERT late interaction
    FAST    = "fast"         # FAISS + BM25, no reranking
    CROSS   = "cross"        # cross-encoder only, no ANN pre-filter
    NONE    = "none"         # raw FAISS scores only


class FusionMode(StrEnum):
    RRF      = "rrf"
    WEIGHTED = "weighted"
    DENSE    = "dense"       # disable BM25 entirely


class ContextStrategy(StrEnum):
    SCORE_ORDER        = "score_order"
    LOST_IN_MIDDLE     = "lost_in_middle"
    TRUNCATE           = "truncate"
    SUMMARIZE_OVERFLOW = "summarize_overflow"


# ── Sub-configs ────────────────────────────────────────────────────────────────

class BackendConfig(BaseModel):
    uri: str
    text_col: str = "content"
    metadata_cols: list[str] = Field(default_factory=list)


class IndexConfig(BaseModel):
    type: IndexType = IndexType.VLQ_ADC
    hybrid_fcvi: bool = True
    nlist: int = 1024               # IVF nlist parameter
    m: int = 64                     # PQ subquantizers
    hnsw_m: int = 32                # HNSW M parameter


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW
    chunk_size: int = Field(default=512, ge=32, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    semantic_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_chunk_length: int = 50
    remove_duplicates: bool = False
    extract_tables: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "nomic-embed-text-v1.5"
    api: str | None = None              # "openai" | None (local)
    api_key: str | None = None
    batch_size: int = 256
    normalize: bool = True
    dim: int | None = None              # inferred from model if None


class BM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


class FusionConfig(BaseModel):
    mode: FusionMode = FusionMode.RRF
    rrf_k: int = 60                 # RRF smoothing constant (Cormack et al. 2009)
    bm25_weight: float = 0.3        # only used for WEIGHTED mode
    dense_weight: float = 0.7


class RankingConfig(BaseModel):
    mode: RankingMode = RankingMode.HYBRID
    k: int = Field(default=20, ge=1, le=1000)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    colbert_model: str = "colbert-ir/colbertv2.0"
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)  # 1.0=pure relevance, 0.0=pure diversity


class ContextConfig(BaseModel):
    strategy: ContextStrategy = ContextStrategy.LOST_IN_MIDDLE
    max_context_tokens: int = 4096
    return_context: bool = False


class CacheConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = Field(default=0.97, ge=0.0, le=1.0)
    verify_borderline: bool = True
    borderline_lower: float = 0.93  # [borderline_lower, threshold] triggers verification pass
    ttl_seconds: int = 3600
    max_size: int = 10_000


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None     # for OpenAI-compatible endpoints
    temperature: float = 0.0
    max_tokens: int = 1024


class ObservabilityConfig(BaseModel):
    log_queries: bool = False
    metrics_port: int | None = None  # None = don't start Prometheus server
    trace: bool = False              # global default; overridable per-call


# ── Root config ───────────────────────────────────────────────────────────────

class KSelectConfig(BaseModel):
    """
    Full configuration for a KSelect instance.
    Constructed programmatically or loaded via from_yaml().
    All sub-configs have safe defaults — only override what you need.
    """
    backend: BackendConfig | None = None    # None = local FAISS only
    index: IndexConfig = Field(default_factory=IndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "KSelectConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = _expand_env(raw)
        return cls.model_validate(raw)


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} patterns in string values."""
    import os, re
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    return obj
```

---

### Phase 1 test target

```python
# tests/unit/test_config.py
def test_config_round_trip(tmp_path):
    """Config serializes to YAML and reloads with identical values."""
    cfg = KSelectConfig()
    yaml_path = tmp_path / "kselect.yaml"
    import yaml
    yaml_path.write_text(yaml.dump(cfg.model_dump()))
    reloaded = KSelectConfig.from_yaml(str(yaml_path))
    assert reloaded == cfg

def test_chunk_overlap_validator():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=128, chunk_overlap=128)

def test_env_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    yaml_path = tmp_path / "kselect.yaml"
    yaml_path.write_text("llm:\n  api_key: \"${OPENAI_API_KEY}\"\n")
    cfg = KSelectConfig.from_yaml(str(yaml_path))
    assert cfg.llm.api_key == "sk-test"
```

---

*Continue to SPEC-03-backends-ingestion.md*
