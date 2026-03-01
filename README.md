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

For pipelines that continuously ingest new documents without rebuilding the full index from scratch.

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
```

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
│   │   ├── kselect.py         # Main KSelect class
│   │   ├── faiss_index.py     # Index implementations
│   │   ├── reranker.py        # Cross-encoder + ColBERT + MMR
│   │   └── backends/
│   │       ├── base.py        # Abstract VectorBackend interface
│   │       ├── local.py       # Local FAISS backend
│   │       ├── pgvector.py    # PGVector backend
│   │       ├── pinecone.py    # Pinecone backend
│   │       └── chromadb.py    # ChromaDB backend
│   ├── ingestion/
│   │   ├── folder.py          # File ingestion + chunking
│   │   ├── csv.py             # CSV / JSON / JSONL ingestion
│   │   └── incremental.py     # add_doc / add_folder
│   └── eval/
│       └── metrics.py         # Evaluation framework
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

## Contributing

Contributions are welcome. Priority areas:

- New backend implementations (Qdrant, Milvus, Elasticsearch)
- Domain-specific evaluation sets (legal, medical, financial)
- Reranker fine-tuning for specialized verticals
- New index type implementations as 2026 papers land
- Benchmarks on additional hardware configurations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Research Citations

- Heidari et al. (Jun 2025). "FCVI: Filter-Centric Vector Indexing."
- Rahman et al. (2025). "VLQ-ADC: Variable-Length Quantization for Approximate Distance Computation."
- Amanbayev et al. (Feb 2026). "Global-Local Selectivity for Hybrid Vector Search."
- Stanford HAI (2024). "Hallucination Rates in Legal Retrieval-Augmented Generation."

---

## License

MIT — see [LICENSE](LICENSE).
