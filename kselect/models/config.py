from __future__ import annotations

import os
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class IndexType(StrEnum):
    FLAT = "faiss_flat"
    IVF_PQ128 = "faiss_ivf_pq128"
    HNSW_SQ = "faiss_hnsw_sq"
    FCVI = "faiss_fcvi"
    VLQ_ADC = "faiss_vlq_adc"


class ChunkingStrategy(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"


class RankingMode(StrEnum):
    HYBRID = "hybrid"  # FAISS + BM25 + cross-encoder + MMR
    COLBERT = "colbert"  # FAISS + BM25 + ColBERT late interaction
    FAST = "fast"  # FAISS + BM25, no reranking
    CROSS = "cross"  # cross-encoder only, no ANN pre-filter
    NONE = "none"  # raw FAISS scores only


class FusionMode(StrEnum):
    RRF = "rrf"
    WEIGHTED = "weighted"
    DENSE = "dense"  # disable BM25 entirely


class ContextStrategy(StrEnum):
    SCORE_ORDER = "score_order"
    LOST_IN_MIDDLE = "lost_in_middle"
    TRUNCATE = "truncate"
    SUMMARIZE_OVERFLOW = "summarize_overflow"


# ── Sub-configs ────────────────────────────────────────────────────────────────


class BackendConfig(BaseModel):
    uri: str
    text_col: str = "content"
    metadata_cols: list[str] = Field(default_factory=list)


class IndexConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    type: IndexType = IndexType.VLQ_ADC
    hybrid_fcvi: bool = True
    nlist: int = 1024  # IVF nlist parameter
    m: int = 64  # PQ subquantizers
    hnsw_m: int = 32  # HNSW M parameter


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW
    chunk_size: int = Field(default=512, ge=32, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    semantic_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_chunk_length: int = 50
    remove_duplicates: bool = False
    extract_tables: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: Any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-large-en-v1.5"
    api: str | None = None  # "openai" | None (local)
    api_key: str | None = None
    batch_size: int = 256
    normalize: bool = True
    dim: int | None = None  # inferred from model if None


class BM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


class FusionConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    mode: FusionMode = FusionMode.RRF
    rrf_k: int = 60  # RRF smoothing constant (Cormack et al. 2009)
    bm25_weight: float = 0.3  # only used for WEIGHTED mode
    dense_weight: float = 0.7


class RankingConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    mode: RankingMode = RankingMode.HYBRID
    k: int = Field(default=20, ge=1, le=1000)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    colbert_model: str = "colbert-ir/colbertv2.0"
    mmr_lambda: float = Field(
        default=0.5, ge=0.0, le=1.0
    )  # 1.0=pure relevance, 0.0=pure diversity


class ContextConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

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
    base_url: str | None = None  # for OpenAI-compatible endpoints
    temperature: float = 0.0
    max_tokens: int = 1024


class ObservabilityConfig(BaseModel):
    log_queries: bool = False
    metrics_port: int | None = None  # None = don't start Prometheus server
    trace: bool = False  # global default; overridable per-call


# ── Root config ───────────────────────────────────────────────────────────────


class KSelectConfig(BaseModel):
    """
    Full configuration for a KSelect instance.
    Constructed programmatically or loaded via from_yaml().
    All sub-configs have safe defaults — only override what you need.
    """

    backend: BackendConfig | None = None  # None = local FAISS only
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
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    return obj


# ── YAML representers ─────────────────────────────────────────────────────────
# PyYAML's Dumper emits Python-tagged output for Enum subclasses (via
# __reduce_ex__) even though StrEnum IS a str.  Register representers so that
# yaml.dump(cfg.model_dump()) always produces safe, tag-free YAML.

try:
    import yaml as _yaml

    def _strenum_representer(dumper: _yaml.Dumper, data: StrEnum) -> _yaml.Node:
        return dumper.represent_str(str(data))

    for _cls in (IndexType, ChunkingStrategy, RankingMode, FusionMode, ContextStrategy):
        _yaml.add_representer(_cls, _strenum_representer)
        _yaml.add_representer(_cls, _strenum_representer, Dumper=_yaml.SafeDumper)

    del _yaml, _strenum_representer, _cls
except ImportError:
    pass  # yaml is a required dep; ImportError only in unusual test isolation
