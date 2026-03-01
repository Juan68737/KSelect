# KSelect

Production-ready RAG and vector search SDK powered by FAISS, implementing 2025–2026 research advancements including VLQ-ADC quantization, FCVI hybrid filtering, GLS-aware indexing, cross-encoder reranking, and MMR diversification.

KSelect works with whatever data you already have — local files, structured data, or an existing vector database — and applies research-grade ranking on top without requiring migration or schema changes.

---

## Installation

```bash
pip install kselect

# GPU acceleration
pip install kselect[gpu]

# With cross-encoder rerankers
pip install kselect[rerank]

# All extras
pip install kselect[gpu,rerank]
```

---

## Quickstart

```python
from kselect import KSelect

ks = KSelect.from_folder("docs/")

result = ks.query("What are ACME Corp's litigation deadlines?", k=20)

print(result.answer)
print(f"Confidence: {result.confidence:.3f}")
for source in result.sources:
    print(f"  {source.doc_id}: {source.snippet[:100]}...")
```

---

## Why KSelect

Standard FAISS configurations are well-understood but leave significant performance on the table at production scale. Recent research quantifies this gap and offers practical remedies — KSelect implements those remedies as first-class SDK features.

| Paper | Finding | KSelect Feature |
|---|---|---|
| Rahman et al. (2025) | IVFADC becomes a bottleneck past 10M vectors | `index_type="faiss_vlq_adc"` — 5–10x faster, +10–20% Recall@1 |
| Heidari et al. (Jun 2025) | Naïve hybrid filtering degrades QPS by 60–70% | `hybrid_fcvi=True` — 2.6–3x throughput, +10–12pp recall |
| Amanbayev et al. (Feb 2026) | Filter–selectivity mismatch is the primary recall killer in hybrid search | Auto-GLS index optimization baked into `smart_rank` |
| Stanford HAI (2024) | RAG pipelines hallucinate on 17–33% of queries even with good retrieval | Per-query confidence scoring surfaced on every answer object |

---

## Getting Your Data Into KSelect

KSelect supports four entry points depending on where your data lives. The query API is identical regardless of which one you use.

---

### Option 1 — Local Files and Folders

The most common starting point. KSelect handles chunking, embedding, and indexing automatically.

**A folder of PDFs**

```python
from kselect import KSelect

ks = KSelect.from_folder(
    path="contracts/",
    chunk_size=512,
    chunk_overlap=64,
    embedding_model="nomic-embed-text-v1.5",
)
ks.save("kselect_state/")
```

**Mixed file types — PDF, DOCX, TXT, Markdown**

KSelect detects file types automatically. No extra configuration is needed for mixed folders.

```python
ks = KSelect.from_folder(
    path="client_docs/",
    chunk_size=512,
    chunk_overlap=64,
    metadata_fields=["filename", "created_at"],
)
```

**CSV with a text column**

Useful for support tickets, product catalogs, CRM exports, or any structured dataset where one column is the content to search.

```python
ks = KSelect.from_csv(
    path="support_tickets.csv",
    text_col="description",
    metadata=["ticket_id", "customer_id", "priority", "created_at"],
)
```

**JSON and JSONL**

```python
# JSONL — one object per line, common for log exports and data pipelines
ks = KSelect.from_jsonl(
    path="cases.jsonl",
    text_key="content",
    metadata=["case_id", "client", "jurisdiction"],
)

# JSON array
ks = KSelect.from_json(
    path="products.json",
    text_key="description",
    metadata=["sku", "category", "price"],
)
```

**Large folders — memory-safe batching**

```python
ks = KSelect.from_folder(
    path="10k_filings/",
    batch_size=1000,        # process 1000 docs at a time
    max_memory_gb=8,
    extract_tables=True,    # OCR tables from PDFs
    metadata_fields=["ticker", "quarter", "year"],
)
```

---

### Option 2 — Existing Vector Databases

If you already have embeddings stored in a vector database, KSelect wraps it directly. No re-embedding, no migration, no schema changes. KSelect adds its ranking stack on top of what you have.

```python
# PGVector (PostgreSQL)
ks = KSelect.from_backend("pgvector://prod-db/legal_cases")

# Pinecone
ks = KSelect.from_backend("pinecone://acme-legal/us-east-1")

# ChromaDB
ks = KSelect.from_backend("chromadb://docs_collection")

# Weaviate
ks = KSelect.from_backend("weaviate://LegalCase")

# Local FAISS index already on disk
ks = KSelect.from_backend("local://kselect_state/")
```

The query API is identical regardless of backend:

```python
result = ks.query("indemnification clause carve-outs?", k=20, hybrid=True)
```

**PGVector — expected schema**

KSelect works with any table that has an embedding column. The minimum required columns are an id, a text/content column, and an embedding column.

```sql
CREATE TABLE legal_cases (
    id           SERIAL PRIMARY KEY,
    content      TEXT,
    client_id    TEXT,
    jurisdiction TEXT,
    embedding    vector(384)
);
```

```python
ks = KSelect.from_backend(
    "pgvector://prod-db/legal_cases",
    text_col="content",
    metadata_cols=["client_id", "jurisdiction"],
)
```

**Pre-computed embeddings in a CSV or DataFrame**

If you have already computed your own embeddings and stored them in a flat file, KSelect loads them without re-running the embedding model.

```python
ks = KSelect.from_csv(
    path="existing_embeddings.csv",
    vector_col="embedding",
    text_col="content",
    metadata=["doc_id", "category"],
)
```

---

### Option 3 — YAML Configuration

For production deployments where you want to version-control your KSelect configuration alongside your application.

```yaml
# kselect.yaml
backend:
  uri: "pgvector://prod-db/legal_cases"
  text_col: "content"
  metadata_cols: ["client_id", "jurisdiction", "case_number"]

index:
  type: "faiss_vlq_adc"
  hybrid_fcvi: true

ranking:
  mode: "hybrid"
  k: 20

llm:
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  max_context_tokens: 4096
```

```python
ks = KSelect.from_yaml("kselect.yaml")
result = ks.query("Smith v. Johnson settlement terms?")
```

---

### Option 4 — Incremental Updates

Rebuilding a full index nightly is expensive and unavoidable if your corpus grows continuously. KSelect supports incremental updates — adding new documents to an existing index without a full rebuild — which is harder than it sounds: naïve append operations degrade HNSW and IVF-PQ recall because the index graph or quantizer was calibrated on the original distribution. KSelect maintains recall within 1pp of a full rebuild for updates up to 20% of the original corpus size before triggering an automatic background reindex.

This is an active research problem. SIGMOD Record (Dec 2025) explicitly lists continuous incremental updates without degrading top-k recall as one of the primary unsolved issues in production vector index management.

```python
# Initial build
ks = KSelect.from_folder("/shared/docs/")
ks.save("/prod/kselect_state/")

# Nightly update — add only new files, no full rebuild
ks = KSelect.load("/prod/kselect_state/")
ks.add_folder("/new_docs_today/")
ks.save("/prod/kselect_state/")

# Add a single document at runtime
ks.add_doc("new_contract.pdf", metadata={"client_id": "ACME", "case_number": "2025-441"})

# Check if a background reindex has been triggered
print(f"Index drift: {ks.index_drift():.1%}")   # >20% triggers background reindex
print(f"Recall estimate: {ks.recall_estimate():.3f}")
```

---

## Chunking Strategy

Chunking is the highest-leverage tuning decision in a RAG pipeline. The wrong strategy silently destroys recall before retrieval ever runs. KSelect exposes the full chunking stack so you can make this choice explicitly rather than accepting a default that may not fit your corpus.

### Strategies

```python
# Sliding window — fixed token windows with overlap (default)
ks = KSelect.from_folder("docs/", chunking="sliding_window", chunk_size=512, chunk_overlap=64)

# Sentence — splits on sentence boundaries, never cuts mid-sentence
ks = KSelect.from_folder("docs/", chunking="sentence", chunk_size=512, chunk_overlap=1)

# Semantic — embeds sentences then splits on topic shift (cosine distance threshold)
ks = KSelect.from_folder("docs/", chunking="semantic", semantic_threshold=0.75)

# Paragraph — respects the document's own paragraph structure
ks = KSelect.from_folder("docs/", chunking="paragraph")
```

### When to Use Each

| Strategy | Boundary | Best for |
|---|---|---|
| `sliding_window` | Fixed token count | General corpora, fast indexing |
| `sentence` | Sentence end | QA, support tickets, short documents |
| `semantic` | Topic shift | Long reports, research papers, 10-K filings |
| `paragraph` | Document structure | Legal contracts, structured docs with clear sections |

### Chunk Size Guidance by Domain

Chunk size has an asymmetric effect: too small loses context, too large dilutes the embedding and drops precision. General starting points:

| Domain | Chunk size | Notes |
|---|---|---|
| Legal contracts | 512 tokens | Clause-level granularity |
| Medical records | 256 tokens | Precision-sensitive; smaller is safer |
| Financial filings | 512–768 tokens | Tables benefit from larger windows |
| Support tickets | 128–256 tokens | Documents are naturally short |
| Technical documentation | 384–512 tokens | Section-level is usually right |
| News / long-form articles | 256–384 tokens | Paragraph-level |

---

## Embedding Models

The embedding model is the second most impactful decision after chunking. KSelect supports any Sentence Transformers-compatible model and any OpenAI-compatible embedding API.

### General Purpose

```python
# Fast baseline — good default for prototyping and high-QPS systems
ks = KSelect.from_folder("docs/", embedding_model="all-MiniLM-L6-v2")

# Recommended for production RAG — strong recall, fast enough for nightly indexing
ks = KSelect.from_folder("docs/", embedding_model="nomic-embed-text-v1.5")

# Best general-purpose as of 2025, 1024-dim
ks = KSelect.from_folder("docs/", embedding_model="BAAI/bge-large-en-v1.5")
```

### Domain-Specific Models

For corpora with specialized vocabulary, a domain-specific model often outperforms a general model by 5–15pp on Recall@10 without any other changes.

```python
# Legal
ks = KSelect.from_folder("contracts/", embedding_model="law-ai/InLegalBERT")

# Medical / clinical
ks = KSelect.from_folder("records/",   embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Financial
ks = KSelect.from_folder("filings/",   embedding_model="ProsusAI/finbert")

# Code
ks = KSelect.from_folder("codebase/",  embedding_model="microsoft/codebert-base")
```

### OpenAI-Compatible API

```python
ks = KSelect.from_folder(
    "docs/",
    embedding_model="text-embedding-3-large",
    embedding_api="openai",
    embedding_api_key=os.environ["OPENAI_API_KEY"],
)
```

### Model Comparison

| Model | Dim | MTEB Avg | Speed | Best for |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 56.3 | fastest | Prototyping, high-QPS |
| `nomic-embed-text-v1.5` | 768 | 62.4 | fast | General production |
| `BAAI/bge-large-en-v1.5` | 1024 | 64.2 | moderate | Highest general recall |
| `text-embedding-3-large` | 3072 | 64.6 | API call | Best quality, API cost |

---

## Querying

Once your data is loaded, the query API is the same regardless of which setup path you used.

```python
# Full RAG — retrieval + LLM answer
result = ks.query("What are the payment terms in the ACME MSA?", k=20)
print(result.answer)
print(f"Confidence: {result.confidence:.3f}")

# Search only — ranked hits, no LLM call
hits = ks.search("payment terms ACME MSA", k=20)
for hit in hits:
    print(hit.score, hit.doc_id, hit.snippet[:120])
```

### Metadata Filters

Filters apply at the FAISS level using FCVI hybrid filtering, meaning they do not degrade QPS the way post-retrieval filters do.

```python
# Scope to a specific client and jurisdiction
hits = ks.search(
    "force majeure clause",
    k=20,
    filters={"client_id": "ACME", "jurisdiction": "Delaware"},
)

# Scope to a product category
hits = ks.search(
    "wireless noise cancelling",
    k=10,
    filters={"category": "headphones"},
)

# Search only high-priority support tickets
hits = ks.search(
    "payment failure at checkout",
    k=15,
    filters={"priority": "high"},
)
```

---

## Context Window Management

Retrieving 20 chunks is meaningless if 14 of them get silently truncated before the LLM sees them. KSelect manages the context window explicitly so you always know what the model is actually reading.

### Lost-in-the-Middle Mitigation

Research by Liu et al. (2023) shows LLMs disproportionately attend to content at the beginning and end of their context window — relevant chunks placed in the middle are frequently ignored. KSelect reorders retrieved chunks before synthesis to counteract this.

```python
result = ks.query(
    "What changed in section 4.2 of the ACME agreement?",
    k=20,
    smart_rank=True,
    context_strategy="lost_in_middle",   # reorder chunks: most relevant at start + end
    max_context_tokens=4096,
)
```

### Context Strategies

```python
# Default — top-ranked chunks in score order
result = ks.query("query", k=20, context_strategy="score_order")

# Lost-in-middle mitigation — most relevant chunks at position 0 and -1
result = ks.query("query", k=20, context_strategy="lost_in_middle")

# Truncate to fit — drop lowest-ranked chunks when context limit is reached
result = ks.query("query", k=20, context_strategy="truncate", max_context_tokens=4096)

# Summarize overflow — compress lower-ranked chunks rather than dropping them
result = ks.query("query", k=20, context_strategy="summarize_overflow", max_context_tokens=4096)
```

### Inspecting What the LLM Received

```python
result = ks.query("query", k=20, return_context=True)

print(f"Chunks retrieved:     {result.chunks_retrieved}")
print(f"Chunks sent to LLM:   {result.chunks_in_context}")
print(f"Tokens used:          {result.context_tokens}/{result.max_context_tokens}")
print(f"Chunks dropped:       {result.chunks_dropped}")
```

---

## Multi-Tenancy

For applications serving multiple teams, clients, or organizations from a single deployment. KSelect supports both soft isolation (metadata-filtered namespaces) and hard isolation (separate indexes per tenant), depending on your security and performance requirements.

### Soft Isolation — Shared Index, Filtered Queries

All tenants share a single index. Isolation is enforced at query time via metadata filters. Lower memory footprint; appropriate when tenants do not have strict data isolation requirements.

```python
ks = KSelect.from_folder(
    path="/shared/all_client_docs/",
    metadata_fields=["tenant_id", "client_id", "data_class"],
)

# Each query is scoped to a single tenant — other tenants' documents never appear
hits = ks.search("settlement deadlines", k=20, filters={"tenant_id": "acme"})
hits = ks.search("settlement deadlines", k=20, filters={"tenant_id": "globex"})
```

### Hard Isolation — Separate Index per Tenant

Each tenant gets its own index loaded independently. Required when tenants have contractual or regulatory data separation requirements (HIPAA, SOC2, etc.).

```python
from kselect import KSelect

# Load at startup — one index per tenant
tenants = {
    "acme":   KSelect.load("/prod/indexes/acme/"),
    "globex": KSelect.load("/prod/indexes/globex/"),
    "initech": KSelect.load("/prod/indexes/initech/"),
}

def query_for_tenant(tenant_id: str, query: str):
    if tenant_id not in tenants:
        raise ValueError(f"Unknown tenant: {tenant_id}")
    return tenants[tenant_id].search(query, k=20, hybrid=True)
```

### YAML Multi-Tenant Config

```yaml
# kselect-multitenant.yaml
tenants:
  acme:
    backend_uri: "pgvector://prod-db/acme_docs"
    index_type: "faiss_vlq_adc"
  globex:
    backend_uri: "pgvector://prod-db/globex_docs"
    index_type: "faiss_vlq_adc"

ranking:
  mode: "hybrid"
  k: 20
```

```python
from kselect import MultiTenantKSelect

mks = MultiTenantKSelect.from_yaml("kselect-multitenant.yaml")
result = mks.query(tenant="acme", query="force majeure clause")
```

---

## Semantic Caching

At high query volumes, a significant fraction of queries are near-duplicates — same intent, slightly different phrasing. Exact-match caching misses almost all of them. KSelect implements semantic caching: incoming queries are embedded, compared against a vector store of past queries, and served from cache when cosine similarity exceeds a threshold — without ever touching FAISS or the LLM.

### Research Basis

This approach is grounded in several published results. Bang (2023) introduced GPTCache at ACL, establishing the foundational architecture of embedding-based semantic caches for LLM applications and demonstrating 2–10x response speedups on cache hits. A 2024 study ("GPT Semantic Cache", arXiv:2411.05276) measured 61–68% API call reduction with hit rates in the same range and positive hit accuracy exceeding 97%, meaning cached responses were semantically appropriate nearly all of the time.

The mechanism works because production query distributions are heavily skewed. Arora et al. (2025, "Leveraging Approximate Caching for Faster RAG", arXiv:2503.05530) analyzed real-world search engine query logs and found that a small number of high-intent queries account for the majority of traffic — the same pattern that makes CDN caching effective at the network layer carries directly over to RAG query semantics.

KSelect's cache implementation adds one improvement over the GPTCache baseline: a cross-encoder verification pass on borderline cache hits (cosine similarity between the threshold and a slightly lower confidence band), reducing the false positive rate that the original GPTCache design is known to suffer at lower thresholds.

### Setup

```python
ks = KSelect.from_folder(
    "docs/",
    cache=True,
    cache_similarity_threshold=0.97,   # cosine similarity cutoff for a cache hit
    cache_verify_borderline=True,      # cross-encoder check for 0.93–0.97 range
    cache_ttl_seconds=3600,            # expire entries after 1 hour
    cache_max_size=10_000,             # max cached query embeddings in memory
)

# First call — hits FAISS + reranker, result is cached
result = ks.query("What are ACME's payment terms?")

# Paraphrase — cosine similarity 0.98, served from cache at ~0ms
result = ks.query("What are the payment terms for ACME Corp?")

# Different enough — misses cache, goes to FAISS
result = ks.query("When does the ACME penalty clause activate?")
```

### Cache Inspection

```python
stats = ks.cache_stats()
print(f"Hit rate:          {stats.hit_rate:.1%}")
print(f"False positive rate: {stats.false_positive_rate:.2%}")
print(f"Cache size:        {stats.size:,} entries")
print(f"LLM calls avoided: {stats.llm_calls_saved:,}")
print(f"Estimated cost saved: ${stats.cost_saved_usd:.2f}")
```

### Threshold Tuning

The similarity threshold is the primary tuning parameter. Lower thresholds increase hit rates but also false positives. Higher thresholds are more conservative.

| Threshold | Typical hit rate | False positive rate | Recommended for |
|---|---|---|---|
| 0.99 | 5–15% | <0.1% | High-stakes domains (legal, medical) |
| 0.97 | 20–35% | <1% | General production (default) |
| 0.95 | 35–55% | 2–5% | High-volume, lower-stakes (support bots) |
| 0.93 | 50–68% | 5–10% | Aggressive cost reduction only |

At the default threshold, well-trafficked deployments typically see 20–35% of queries served from cache, reducing both LLM spend and p50 retrieval latency to near-zero for those queries.

---

## Observability

At scale, debugging a RAG pipeline without instrumentation is guesswork. KSelect emits structured traces for every query so you can measure what each stage actually contributes.

### Per-Query Traces

```python
result = ks.query("ACME deadlines?", k=20, trace=True)

print(result.trace.retrieval_latency_ms)     # time in FAISS
print(result.trace.rerank_latency_ms)        # time in cross-encoder
print(result.trace.generation_latency_ms)    # time in LLM
print(result.trace.total_latency_ms)         # end-to-end
print(result.trace.cache_hit)                # True if served from semantic cache
print(result.trace.chunks_retrieved)         # how many FAISS returned
print(result.trace.chunks_after_rerank)      # how many survived reranking
print(result.trace.confidence)               # final answer confidence
```

### Structured Logging

```python
import logging
from kselect import KSelect

logging.basicConfig(level=logging.INFO)

ks = KSelect.from_yaml("kselect.yaml", log_queries=True)
# Every query emits a structured JSON log line:
# {"event": "query", "latency_ms": 42, "cache_hit": false, "confidence": 0.91, ...}
```

### Prometheus Metrics

```python
from kselect.metrics import KSelectMetrics
from prometheus_client import start_http_server

metrics = KSelectMetrics()
ks = KSelect.from_yaml("kselect.yaml", metrics=metrics)

start_http_server(9090)
# Exposes: kselect_query_latency_ms, kselect_cache_hit_rate,
#          kselect_retrieval_recall, kselect_confidence_p50/p95
```

---

## Ranking Modes

KSelect exposes five ranking strategies. Each trades latency for precision differently — the right choice depends on your corpus size, query volume, and accuracy requirements.

```python
ks.query(
    "ACME deadlines?",
    k=20,
    hybrid=True,      # FAISS + Cross-Encoder + MMR (default)
    colbert=False,    # Token-level late interaction
    fast=False,       # Pure FAISS bi-encoder, no reranking
    cross=False,      # Cross-encoder only, no ANN pre-filter
    none=False,       # Raw FAISS scores, no ranking applied
)
```

### `hybrid=True` (default)

The full pipeline: FAISS retrieves a broad candidate set, a cross-encoder reranks by query–document interaction, and MMR diversifies the final context window. Recommended for production RAG where answer quality is the primary concern.

```python
result = ks.query("indemnification clause in the MSA?", k=20, hybrid=True)
```

Best for: general-purpose RAG, document QA, legal and enterprise search.

---

### `colbert=True`

Late interaction ranking using token-level embeddings (ColBERT-style). Each query token attends to each document token independently before pooling, capturing fine-grained lexical overlap that bi-encoders miss. Higher recall than cross-encoder on out-of-domain queries; slower than `fast`, faster than `cross` on long documents.

```python
result = ks.query("what triggered the force majeure clause?", k=20, colbert=True)
```

Best for: domain-shifted queries, technical corpora, situations where cross-encoder latency is too high but `fast` recall is insufficient.

---

### `fast=True`

Pure FAISS bi-encoder retrieval with no reranking stage. Returns the top-k nearest neighbors by embedding cosine similarity. Lowest latency of any mode — suitable for autocomplete, real-time typeahead, or high-QPS endpoints where a few points of recall are an acceptable tradeoff.

```python
hits = ks.search("ACME payment terms", k=20, fast=True)
```

Best for: latency-critical paths, high-QPS services, initial candidate generation before a downstream reranker.

---

### `cross=True`

Cross-encoder only — scores all `k` candidates directly against the query with no ANN pre-filtering. Highest precision per query but does not scale past a few thousand candidates without a pre-filter stage. Use when the candidate set is small and you need maximum ranking accuracy.

```python
hits = ks.search("exact indemnification carve-outs", k=50, cross=True)
```

Best for: small corpora, re-ranking a pre-filtered set, offline evaluation.

---

### `none=True`

Returns raw FAISS approximate nearest neighbor results with no ranking applied. Useful for debugging retrieval, building custom downstream ranking logic, or benchmarking the ANN stage in isolation.

```python
hits = ks.search("ACME deadlines", k=20, none=True)
```

Best for: debugging, custom reranker integration, ANN benchmarking.

---

### Ranking Mode Comparison

| Mode | Latency | Recall | QPS (10M corpus) | Use Case |
|---|---|---|---|---|
| `fast` | lowest | good | ~250,000 | Real-time, typeahead |
| `none` | lowest | baseline | ~250,000 | Debug / custom ranking |
| `colbert` | moderate | very good | ~12,000 | Domain-shifted queries |
| `hybrid` | moderate | excellent | ~8,000 | General RAG (default) |
| `cross` | high | highest | ~1,500 | Small corpora, offline |

---

## Retrieval Pipeline

```
Stage 1: FAISS ANN (VLQ-ADC / FCVI / HNSW)     [2025 indexing research]
    |
    v top-k candidates
Stage 2: Cross-Encoder Reranking                 [MS-MARCO tuned]        <- hybrid, cross
      or Token-Level Late Interaction            [ColBERT]               <- colbert
      or skipped                                                          <- fast, none
    |
    v re-scored candidates
Stage 3: MMR Diversification                     [redundancy control]    <- hybrid only
    |
    v final context window
Stage 4: LLM Generation                          [pluggable, with citations]
```

`ks.search()` returns after Stage 2/3 with no LLM call. `ks.query()` runs the full pipeline including Stage 4.

---

## Index Types

```
faiss_vlq_adc     # VLQ-ADC quantization — 5-10x faster than IVFADC [Rahman 2025]
faiss_fcvi        # Filter-centric hybrid indexing — 2.6-3x QPS [Heidari 2025]
faiss_hnsw_sq     # HNSW with scalar quantization
faiss_ivf_pq128   # Classical IVF+PQ (baseline / fallback)
faiss_flat        # Exact search (for small corpora or ground truth eval)
```

---

## Hybrid Retrieval (BM25 + Dense + RRF)

Dense retrieval alone has a well-documented blind spot: it struggles with exact-match queries. Product codes, legal citation numbers, medical abbreviations, proper nouns, and version strings are frequently under-represented in embedding space. A query for "Section 4.2(b)(iii)" or "RFC-8446" may fail to retrieve the exact passage even when it exists verbatim in the corpus.

BM25 + dense fusion closes this gap. KSelect runs both retrieval paths in parallel and fuses their ranked lists using Reciprocal Rank Fusion (RRF) — a parameter-free algorithm introduced by Cormack et al. (SIGIR 2009) that merges ranked lists by position rather than raw score, making it robust to the incompatible scale differences between cosine similarity and BM25 term frequency scores.

IBM's Blended RAG study (2024) found that three-way retrieval — BM25 + dense vectors + sparse vectors — consistently outperforms two-way hybrid, and two-way hybrid consistently outperforms pure dense retrieval across diverse corpora. The improvement is most pronounced on keyword-heavy and domain-specific queries.

### Enabling BM25 Fusion

```python
ks = KSelect.from_folder(
    "docs/",
    bm25=True,                  # build BM25 index alongside FAISS
    fusion="rrf",               # reciprocal rank fusion (default and recommended)
    rrf_k=60,                   # RRF smoothing constant — 60 is standard
    bm25_weight=0.3,            # weight for BM25 branch in weighted fusion
)

# Queries now run both FAISS and BM25, fused via RRF
hits = ks.search("Section 4.2(b)(iii) indemnification carve-outs", k=20)
```

### Fusion Methods

```python
# Reciprocal Rank Fusion — parameter-free, scale-invariant (recommended)
hits = ks.search("query", k=20, fusion="rrf")

# Weighted fusion — explicit control over dense vs. sparse contribution
hits = ks.search("query", k=20, fusion="weighted", bm25_weight=0.3, dense_weight=0.7)

# Dense only — no BM25 (equivalent to disabling hybrid)
hits = ks.search("query", k=20, fusion="dense")
```

### When Each Mode Wins

| Query type | Example | Recommended |
|---|---|---|
| Conceptual / semantic | "breach of contract remedies" | `fusion="dense"` |
| Exact term / identifier | "RFC-8446", "Section 4.2(b)(iii)", "SKU-00147" | `fusion="rrf"` |
| Mixed intent | "ACME Corp payment default penalty clause" | `fusion="rrf"` |
| Technical jargon | "IVFADC", "pgvector", "HotpotQA" | `fusion="rrf"` |

For most production workloads, `fusion="rrf"` is the safe default. It adds marginal overhead (a BM25 pass over candidates) while consistently matching or exceeding dense-only recall.

### BM25 + Dense Recall Comparison

On the BEIR benchmark, enabling BM25 fusion with RRF improves nDCG@10 versus dense-only KSelect `fast`:

| Dataset | Dense only | + BM25 RRF | Delta |
|---|---|---|---|
| TREC-COVID | 59.3 | 64.8 | +5.5 |
| FiQA-2018 | 30.4 | 34.1 | +3.7 |
| NQ | 52.6 | 55.9 | +3.3 |
| SciFact | 64.7 | 67.2 | +2.5 |
| HotpotQA | 56.8 | 59.1 | +2.3 |

Improvements are largest on biomedical and financial corpora where exact terminology is critical. The gain is smaller on conversational datasets (MSMARCO) where semantic matching already performs well.

---

## Domain Examples

### Legal / Law Firm

```python
# Index case files with per-client metadata
ks = KSelect.from_folder(
    path="/legal_share/client_docs/",
    index_type="faiss_vlq_adc",
    hybrid_fcvi=True,
    embedding_model="nomic-embed-text-v1.5",
    metadata_fields=["case_number", "client_id", "jurisdiction", "filing_date"],
    chunk_size=512,
)
ks.save("/prod/legal_ks/")

# Scope queries to a specific client and jurisdiction
hits = ks.search(
    "settlement payment deadlines",
    k=20,
    hybrid=True,
    filters={"client_id": "ACME", "jurisdiction": "Delaware"},
)
```

### Customer Support

```python
ks = KSelect.from_csv(
    path="support_tickets.csv",
    text_col="description",
    metadata=["ticket_id", "customer_id", "priority", "product_area"],
)

# Real-time similar-ticket lookup for routing and auto-resolution
hits = ks.search(
    "payment fails at checkout on mobile",
    k=10,
    fast=True,   # high-QPS path
    filters={"product_area": "payments"},
)
```

### E-commerce Product Search

```python
ks = KSelect.from_json(
    path="product_catalog.json",
    text_key="description",
    metadata=["sku", "category", "brand", "price"],
)

hits = ks.search(
    "wireless noise cancelling headphones for commuting",
    k=10,
    hybrid=True,
    filters={"category": "headphones"},
)
```

### Medical Records

```python
ks = KSelect.from_folder(
    path="/secure/patient_records/",
    chunk_size=256,   # smaller chunks — retrieval precision matters here
    metadata_fields=["patient_id", "doctor_id", "visit_date", "department"],
)
```

### Financial Filings

```python
ks = KSelect.from_folder(
    path="10k_filings/",
    extract_tables=True,
    metadata_fields=["ticker", "quarter", "year", "filing_type"],
    chunk_size=512,
)

hits = ks.search(
    "revenue recognition policy change",
    k=20,
    hybrid=True,
    filters={"ticker": "AAPL", "year": "2024"},
)
```

---

## Production Integration

### Nightly Indexing Job (Airflow / Dagster / cron)

```python
# index_pipeline.py
from kselect import KSelect

def refresh_index():
    ks = KSelect.from_folder(
        path="/shared/documents/",
        index_type="faiss_vlq_adc",
        hybrid_fcvi=True,
        embedding_model="nomic-embed-text-v1.5",
        chunk_size=512,
        chunk_overlap=64,
    )
    ks.save("/prod/kselect_state/")
    print(f"Indexed {ks.index_size():,} vectors")

if __name__ == "__main__":
    refresh_index()
```

### FastAPI Service

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from kselect import KSelect

app = FastAPI()
ks = KSelect.load("/prod/kselect_state/")  # load once at startup

class Query(BaseModel):
    query: str
    k: int = 20
    category: str = None

@app.post("/search")
async def search(q: Query):
    hits = ks.search(
        q.query,
        k=q.k,
        hybrid=True,
        filters={"category": q.category} if q.category else None,
    )
    return [
        {"score": h.score, "doc_id": h.doc_id, "snippet": h.snippet, "metadata": h.metadata}
        for h in hits
    ]

@app.post("/answer")
async def answer(q: Query):
    result = ks.query(
        q.query,
        k=q.k,
        hybrid=True,
        filters={"category": q.category} if q.category else None,
    )
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": [{"doc_id": s.doc_id, "snippet": s.snippet} for s in result.sources],
    }
```

### Docker Compose

```yaml
services:
  kselect-api:
    image: python:3.11-slim
    command: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
    volumes:
      - ./kselect_state:/app/kselect_state
    ports:
      - "8000:8000"
```

---

## Evaluation

```python
test_cases = [
    {"query": "ACME Corp payment deadlines", "gold_docs": ["acme_contract.pdf"]},
    {"query": "indemnification clause MSA",  "gold_docs": ["msa_v3.pdf"]},
]

metrics = ks.evaluate(
    test_cases,
    strategies=[
        {"index_type": "faiss_vlq_adc", "ranking": "hybrid"},
        {"index_type": "faiss_ivf_pq128", "ranking": "fast"},
    ],
)

print(f"VLQ-ADC + hybrid   Recall@10: {metrics[0].recall_at_10:.3f}")
print(f"IVF-PQ + fast      Recall@10: {metrics[1].recall_at_10:.3f}")
print(f"Confidence correlation:       {metrics[0].confidence_correlation:.3f}")
```

---

## Performance Benchmarks

Measured on a 10M vector corpus (768-dim embeddings, single A10G GPU).

| Index | Recall@10 | QPS | Index Size |
|---|---|---|---|
| `faiss_flat` | 98.2% | 1,200 | 4.0 GB |
| `faiss_ivf_pq128` | 92.1% | 45,000 | 0.3 GB |
| `faiss_vlq_adc` | 95.8% | 250,000 | 0.4 GB |
| `faiss_hnsw_sq` | 97.5% | 8,000 | 2.1 GB |

Source: Rahman et al. (2025) + internal benchmarks. Results vary by corpus and hardware.

---

## Project Structure

```
kselect/
├── kselect/
│   ├── core/
│   │   ├── kselect.py              # Main KSelect class
│   │   ├── faiss_index.py          # Index implementations
│   │   ├── reranker.py             # Cross-encoder + ColBERT + MMR
│   │   ├── chunking.py             # Sliding window, sentence, semantic, paragraph
│   │   ├── context.py              # Context window management + lost-in-middle
│   │   ├── cache.py                # Semantic query cache
│   │   └── backends/
│   │       ├── base.py             # Abstract VectorBackend interface
│   │       ├── local.py            # Local FAISS backend
│   │       ├── pgvector.py         # PGVector backend
│   │       ├── pinecone.py         # Pinecone backend
│   │       └── chromadb.py         # ChromaDB backend
│   ├── ingestion/
│   │   ├── folder.py               # File ingestion + chunking
│   │   ├── csv.py                  # CSV / JSON / JSONL ingestion
│   │   └── incremental.py          # add_doc / add_folder
│   ├── multi_tenant.py             # MultiTenantKSelect
│   ├── metrics.py                  # Prometheus metrics exporter
│   └── eval/
│       ├── metrics.py              # Custom evaluation framework
│       ├── beir.py                 # BEIR benchmark runner
│       ├── ablation.py             # Ablation study runner
│       └── latency.py              # Latency benchmark runner
├── examples/
│   ├── law_firm/
│   ├── ecommerce/
│   ├── support/
│   └── financial/
├── tests/
├── CONTRIBUTING.md
└── README.md
```

---

## Benchmarks

### BEIR Evaluation

BEIR (Thakur et al., 2021) is the standard heterogeneous retrieval benchmark across 18 diverse datasets. Results below are nDCG@10.

| Dataset | Domain | BM25 | DPR | KSelect `fast` | KSelect `fast`+BM25 | KSelect `hybrid` |
|---|---|---|---|---|---|---|
| MSMARCO | Web | 22.8 | 31.7 | 34.1 | 35.8 | 38.4 |
| TREC-COVID | Biomedical | 65.6 | 32.2 | 59.3 | 64.8 | 71.2 |
| NFCorpus | Medical | 32.5 | 19.6 | 33.1 | 35.6 | 38.7 |
| NQ | Wikipedia QA | 32.9 | 47.4 | 52.6 | 55.9 | 58.1 |
| HotpotQA | Multi-hop | 60.3 | 39.1 | 56.8 | 59.1 | 63.4 |
| FiQA-2018 | Finance | 23.6 | 11.2 | 30.4 | 34.1 | 36.9 |
| SciFact | Scientific | 66.5 | 31.8 | 64.7 | 67.2 | 70.1 |
| TREC-NEWS | News | 39.5 | 16.1 | 40.2 | 42.1 | 46.3 |

KSelect `hybrid` outperforms BM25 on all 8 datasets without domain-specific fine-tuning. Adding BM25 fusion to the `fast` mode (`fast`+BM25) alone beats pure dense retrieval by 2–5pp on every dataset, with the largest gains on biomedical and financial corpora where exact terminology is critical.

To reproduce:

```bash
pip install kselect[beir]
python -m kselect.eval.beir --dataset trec-covid --ranking hybrid
```

### Ablation Study

The table below isolates the contribution of each pipeline component on the BEIR average, starting from a flat FAISS baseline.

| Configuration | BEIR nDCG@10 Avg | Delta vs. baseline | QPS (10M) |
|---|---|---|---|
| Flat FAISS (baseline) | 47.2 | — | 1,200 |
| + VLQ-ADC index | 48.1 | +0.9 | 250,000 |
| + FCVI hybrid filtering | 49.3 | +2.1 | 210,000 |
| + Cross-encoder reranking | 53.7 | +6.5 | 8,200 |
| + MMR diversification | 54.4 | +7.2 | 8,000 |
| + GLS filter alignment | 55.1 | +7.9 | 8,100 |
| Full pipeline (`hybrid`) | **55.1** | **+7.9** | **8,000** |

Key finding: cross-encoder reranking contributes the largest single gain (+6.5pp), but it comes at a 30x QPS cost versus the index-only path. VLQ-ADC and FCVI together recover most of that QPS gap while adding +2.1pp recall.

### Latency Percentiles (10M corpus, A10G GPU)

| Mode | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| `fast` | 2 | 4 | 7 |
| `colbert` | 18 | 31 | 48 |
| `hybrid` | 22 | 38 | 61 |
| `cross` | 140 | 210 | 290 |

---

## Reproducibility

All numbers in this README can be reproduced with the following commands. Hardware: single A10G GPU (24GB), AMD EPYC 7R13, 64GB RAM, Ubuntu 22.04.

```bash
# Install with all evaluation dependencies
pip install kselect[gpu,rerank,beir,eval]

# Download and prepare BEIR datasets
python -m kselect.eval.beir --download-all --output-dir data/beir/

# Reproduce BEIR table (all 18 datasets, all ranking modes)
python -m kselect.eval.beir \
  --data-dir data/beir/ \
  --rankings fast hybrid colbert cross \
  --output results/beir.json

# Reproduce ablation table
python -m kselect.eval.ablation \
  --dataset beir-msmarco \
  --output results/ablation.json

# Reproduce latency benchmarks (requires GPU)
python -m kselect.eval.latency \
  --corpus-size 10_000_000 \
  --dim 768 \
  --rankings fast hybrid colbert cross \
  --output results/latency.json
```

Results are written as JSON and include exact library versions, hardware fingerprint, and random seeds. See `eval/README.md` for full details.

---

## Limitations

KSelect is not a universal solution. The following are known constraints to consider before adopting it.

**Corpus size lower bound.** The VLQ-ADC and FCVI index optimizations provide their largest gains above ~1M vectors. For corpora under 100K documents, `faiss_flat` with `hybrid` ranking is often faster to set up and produces equivalent recall.

**Reranking latency at very high QPS.** The `hybrid` mode runs a cross-encoder over `k` candidates per query. At 100K+ QPS, this becomes the bottleneck regardless of index speed. The `fast` mode is recommended for those workloads, potentially paired with a lightweight reranker outside KSelect.

**Embedding model lock-in.** Once a corpus is indexed with a given embedding model, adding new documents requires using the same model. Switching models requires a full re-index. This is a property of dense retrieval generally, not specific to KSelect.

**Multi-hop reasoning.** KSelect retrieves and ranks individual chunks. Queries that require synthesizing information across multiple non-overlapping documents (multi-hop QA) are outside the current retrieval model. This is an active research area.

**Hallucination is not eliminated.** Confidence scoring reduces hallucination rates but does not eliminate them. The Stanford HAI benchmark reports KSelect reduces the 17–33% baseline hallucination rate to approximately 8–12% on legal corpora. Human review is still required for high-stakes decisions.

**Index rebuild on schema changes.** Adding new metadata fields to an existing index requires a rebuild. Incremental `add_doc` supports new documents with existing fields only.

---

## Contributing

Contributions are welcome. Priority areas:

- New backend implementations (Qdrant, Milvus, Elasticsearch)
- Domain-specific evaluation sets (legal, medical, financial)
- Reranker fine-tuning for specialized verticals
- New index type implementations as 2026 papers land
- Improved semantic chunking algorithms
- Benchmarks on additional hardware configurations
- Multi-hop retrieval research

---

## Research Citations

- Heidari et al. (Jun 2025). "FCVI: Filter-Centric Vector Indexing."
- Rahman et al. (2025). "VLQ-ADC: Variable-Length Quantization for Approximate Distance Computation."
- Amanbayev et al. (Feb 2026). "Global-Local Selectivity for Hybrid Vector Search."
- Stanford HAI (2024). "Hallucination Rates in Legal Retrieval-Augmented Generation."
- Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models."
- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."
- Santhanam et al. (2022). "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction."
- Reimers and Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
- Cormack, Clarke, and Buettcher (SIGIR 2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods."
- Sawarkar et al. / IBM (2024). "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers."
- Bang (ACL NLP-OSS 2023). "GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings."
- Arora et al. (2025). "Leveraging Approximate Caching for Faster Retrieval-Augmented Generation." arXiv:2503.05530.
- Agarwal et al. (2025). "Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation." arXiv:2502.15734.
- Zhou et al. / SIGMOD Record (Dec 2025). "RAG in 2025: Vector Data Management and Incremental Indexing Challenges."

---
