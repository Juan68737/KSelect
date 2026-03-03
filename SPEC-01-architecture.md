# KSelect — SPEC Part 1 of 5: Architecture, Dependencies, Project Layout

**Read order:** Start here. This file gives you the full system picture, the dependency pinlist, and the directory structure before you write a single line of code.

**All 5 parts:**
- `SPEC-01-architecture.md` ← you are here
- `SPEC-02-models-config.md` — Pydantic models and full config schema
- `SPEC-03-backends-ingestion.md` — VectorBackend ABC, all backends, ingestion pipeline, chunking
- `SPEC-04-index-retrieval-ranking.md` — FAISSIndex, BM25Index, IndexManager, RetrievalEngine, RankingEngine
- `SPEC-05-core-class-infra.md` — KSelect main class, cache, context, multi-tenant, observability, persistence, errors, naming, implementation order

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        KSelect                              │
│                    (public entry point)                     │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │IngestionPipeline│  from_folder / from_csv / from_json / from_jsonl
       │  + Chunker     │  → List[Chunk]
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  IndexManager  │  owns both indexes; handles save/load/add_doc
       │                │
       │  ┌───────────┐ │
       │  │ FAISSIndex│ │  dense ANN (VLQ-ADC / FCVI / HNSW / IVF-PQ / Flat)
       │  └───────────┘ │
       │  ┌───────────┐ │
       │  │ BM25Index │ │  sparse keyword (bm25s)
       │  └───────────┘ │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ RetrievalEngine│  runs FAISS + BM25 in parallel, fuses via RRF
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  RankingEngine │  cross-encoder → ColBERT → MMR (mode-dependent)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ContextAssembler│  lost-in-middle reorder, truncation, overflow
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │   LLMClient    │  pluggable; optional — search() stops before here
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  SemanticCache │  wraps entire pipeline; checked first, written last
       └────────────────┘

Orthogonal components (injected via constructor):
  VectorBackend   — abstract storage interface (local / pgvector / pinecone / chroma)
  MetricsEmitter  — Prometheus + structured logging
```

**Key rule:** `search()` exits after `RankingEngine`. `query()` continues through `ContextAssembler` and `LLMClient`. `SemanticCache` wraps both — it intercepts before any stage and writes after `LLMClient` (for `query()`) or after `RankingEngine` (for `search()`).

---

## 2. Dependency Pinlist

```toml
# pyproject.toml — required dependencies
[project]
name = "kselect"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Core
    "pydantic>=2.7,<3.0",
    "numpy>=1.26,<2.0",

    # Vector index
    "faiss-cpu>=1.8.0",          # swap for faiss-gpu in [gpu] extra

    # Sparse retrieval
    "bm25s>=0.2.0",              # pure-Python BM25, no Java dep

    # Embeddings
    "sentence-transformers>=3.0,<4.0",
    "torch>=2.2,<3.0",           # pulled by sentence-transformers
    "transformers>=4.40,<5.0",

    # Chunking
    "nltk>=3.8",                 # sentence tokenizer
    "tiktoken>=0.7",             # token counting

    # Document parsing
    "pypdf>=4.0",
    "python-docx>=1.1",
    "unstructured>=0.13",        # tables, OCR dispatch

    # HTTP / async
    "httpx>=0.27",
    "anyio>=4.0",

    # Config
    "pyyaml>=6.0",
    "python-dotenv>=1.0",

    # Observability
    "structlog>=24.0",
    "prometheus-client>=0.20",
]

[project.optional-dependencies]
gpu = [
    "faiss-gpu>=1.8.0",
]
rerank = [
    # cross-encoder already in sentence-transformers; ColBERT via:
    "colbert-ai>=0.2.19",
]
pgvector = [
    "psycopg[binary]>=3.1",      # psycopg3, async-capable
    "pgvector>=0.3",
]
pinecone = [
    "pinecone-client>=4.0",
]
chromadb = [
    "chromadb>=0.5",
]
beir = [
    "beir>=2.0",
    "datasets>=2.19",
]
eval = [
    "pandas>=2.2",
    "scipy>=1.13",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
    "hypothesis>=6.100",
]
```

---

## 3. Project Layout

```
kselect/
├── kselect/
│   ├── __init__.py                  # exports KSelect, MultiTenantKSelect
│   ├── py.typed                     # PEP 561 marker
│   ├── exceptions.py                # full error taxonomy (see SPEC-05)
│   │
│   ├── models/                      # Pydantic data models — no logic
│   │   ├── __init__.py
│   │   ├── chunk.py                 # Chunk, ChunkMetadata
│   │   ├── hit.py                   # Hit, SearchResult
│   │   ├── answer.py                # QueryResult, Source
│   │   ├── trace.py                 # QueryTrace
│   │   ├── config.py                # KSelectConfig and all sub-configs
│   │   └── cache.py                 # CacheEntry, CacheStats
│   │
│   ├── backends/                    # Storage abstraction
│   │   ├── __init__.py
│   │   ├── base.py                  # VectorBackend ABC
│   │   ├── local.py                 # LocalBackend
│   │   ├── pgvector.py              # PGVectorBackend
│   │   ├── pinecone.py              # PineconeBackend
│   │   ├── chromadb.py              # ChromaDBBackend
│   │   └── factory.py               # parse_backend_uri() → VectorBackend
│   │
│   ├── ingestion/                   # Document → Chunk pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py               # FileLoader, CSVLoader, JSONLoader
│   │   ├── chunking.py              # Chunker, all strategies
│   │   └── pipeline.py              # IngestionPipeline: orchestrates loaders + chunker
│   │
│   ├── index/                       # Index lifecycle
│   │   ├── __init__.py
│   │   ├── faiss_index.py           # FAISSIndex: build/search/save/load/add
│   │   ├── bm25_index.py            # BM25Index: build/search/save/load/add
│   │   └── manager.py               # IndexManager: owns both, unified save/load
│   │
│   ├── retrieval/                   # Retrieval + fusion
│   │   ├── __init__.py
│   │   ├── engine.py                # RetrievalEngine: parallel search + RRF
│   │   └── fusion.py                # rrf(), weighted_fusion()
│   │
│   ├── ranking/                     # Post-retrieval ranking
│   │   ├── __init__.py
│   │   ├── cross_encoder.py         # CrossEncoderReranker
│   │   ├── colbert.py               # ColBERTReranker
│   │   └── mmr.py                   # mmr_diversify()
│   │
│   ├── context/                     # Context window assembly
│   │   ├── __init__.py
│   │   └── assembler.py             # ContextAssembler: reorder, truncate, summarize
│   │
│   ├── cache/                       # Semantic cache
│   │   ├── __init__.py
│   │   └── semantic_cache.py        # SemanticCache: FAISS-backed query index
│   │
│   ├── llm/                         # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── base.py                  # LLMClient ABC
│   │   └── openai_client.py         # OpenAIClient (default)
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logging.py               # structlog setup
│   │   └── metrics.py               # KSelectMetrics (Prometheus)
│   │
│   ├── multi_tenant.py              # MultiTenantKSelect
│   ├── kselect.py                   # KSelect: main public class
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py               # recall_at_k, mrr, ndcg
│       ├── beir.py                  # BEIR runner
│       ├── ablation.py              # component ablation
│       └── latency.py               # latency benchmark
│
├── examples/
│   ├── law_firm/
│   ├── ecommerce/
│   ├── support/
│   └── financial/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── SPEC-01-architecture.md
├── SPEC-02-models-config.md
├── SPEC-03-backends-ingestion.md
├── SPEC-04-index-retrieval-ranking.md
├── SPEC-05-core-class-infra.md
├── README.md
└── CONTRIBUTING.md
```

---

*Continue to SPEC-02-models-config.md*
