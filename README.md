# KSelect

Open-source hybrid RAG and vector search library. Combines FAISS dense search with BM25 keyword search, fuses results with Reciprocal Rank Fusion (RRF), and optionally reranks with a cross-encoder. Beats vanilla cosine-similarity RAG out of the box.

---

## Install

```bash
git clone https://github.com/your-repo/KSelect.git
cd KSelect
pip install -r requirements.txt
```

**Mac:** CPU mode only. Do not use MPS with large corpora — it will crash.
**Windows / Linux:** Full GPU support via CUDA.
**Use Hiper Gator** Only for University of Florida Students

---

## Quickstart

```python
from kselect import KSelect

# Build index from a folder of files
ks = KSelect.from_folder("docs/")

# Search — returns ranked chunks
result = ks.search("What are the treatment options for BRCA1 mutations?", k=10)
for chunk in result.chunks:
    print(chunk.text)

# RAG query — LLM answer with sources (requires LLM API key in config)
result = ks.query("What are the treatment options for BRCA1 mutations?", k=10)
print(result.answer)
print(result.confidence)
```

---

## Load from different sources

```python
ks = KSelect.from_folder("docs/")                              # .txt, .pdf files
ks = KSelect.from_csv("data.csv", text_column="body")         # CSV
ks = KSelect.from_json("data.json", text_key="content")       # JSON
ks = KSelect.from_jsonl("data.jsonl", text_key="content")     # JSONL

# Save and reload — no re-embedding needed
ks.save("my_index/")
ks = KSelect.load("my_index/")
```

---

## Configuration guide

All settings are optional — defaults work out of the box. Only change what you need.

```python
from kselect import KSelect
from kselect.models.config import (
    KSelectConfig, IndexConfig, IndexType,
    BM25Config, FusionConfig, FusionMode,
    RankingConfig, RankingMode,
    EmbeddingConfig, ChunkingConfig, ChunkingStrategy,
    ContextConfig, ContextStrategy,
    CacheConfig, LLMConfig,
)

cfg = KSelectConfig()
ks = KSelect.from_folder("docs/", config=cfg)
```

Or load from a YAML file:

```python
cfg = KSelectConfig.from_yaml("config.yaml")
```

---

### Index — `IndexConfig`

Controls how vectors are stored and searched in FAISS.

```python
cfg.index = IndexConfig(type=IndexType.FLAT)
```

| Option        | Type        | Default   | Description                                              |
| ------------- | ----------- | --------- | -------------------------------------------------------- |
| `type`        | `IndexType` | `VLQ_ADC` | Index algorithm (see table below)                        |
| `hybrid_fcvi` | bool        | `True`    | Enable FCVI hybrid filtering (faster filtered queries)   |
| `nlist`       | int         | `1024`    | Number of IVF clusters (only used with IVF_PQ128)        |
| `m`           | int         | `64`      | PQ subquantizers (only used with IVF_PQ128)              |
| `hnsw_m`      | int         | `32`      | HNSW graph connections per node (only used with HNSW_SQ) |

**Index types:**

| `IndexType` | Best for                  | Notes                                                                                                          |
| ----------- | ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `FLAT`      | <50k docs, best accuracy  | Exact search, no approximation. Slowest at scale but most accurate. Use this for benchmarks and small corpora. |
| `IVF_PQ128` | 100k–10M docs             | Approximate. Clusters docs into `nlist` groups, searches `nprobe` of them. Faster, small recall loss.          |
| `HNSW_SQ`   | Low-latency production    | Graph-based ANN. Fast queries, higher RAM usage.                                                               |
| `FCVI`      | Filtered queries at scale | Optimized for metadata-filtered search. Avoids the 60–70% QPS drop of naive filtering.                         |
| `VLQ_ADC`   | Large scale (>10M docs)   | Quantized vectors, 5–10x faster than IVF at scale with better recall.                                          |

**Mac recommendation:** Use `IndexType.FLAT` — it's the most accurate and the corpus is small enough that speed doesn't matter.

---

### Chunking — `ChunkingConfig`

Controls how documents are split before indexing.

```python
cfg.chunking = ChunkingConfig(
    strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=512,
    chunk_overlap=64,
)
```

| Option               | Type               | Default          | Description                               |
| -------------------- | ------------------ | ---------------- | ----------------------------------------- |
| `strategy`           | `ChunkingStrategy` | `SLIDING_WINDOW` | How to split text (see below)             |
| `chunk_size`         | int                | `512`            | Max tokens per chunk (32–4096)            |
| `chunk_overlap`      | int                | `64`             | Overlap between consecutive chunks        |
| `semantic_threshold` | float              | `0.75`           | Similarity cutoff for `SEMANTIC` strategy |
| `min_chunk_length`   | int                | `50`             | Discard chunks shorter than this          |
| `remove_duplicates`  | bool               | `False`          | Drop near-duplicate chunks                |

**Chunking strategies:**

| `ChunkingStrategy` | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `SLIDING_WINDOW`   | Fixed-size windows with overlap. Fast, predictable. Good default.              |
| `SENTENCE`         | Split on sentence boundaries. Better for QA.                                   |
| `PARAGRAPH`        | Split on paragraph breaks. Good for structured docs.                           |
| `SEMANTIC`         | Split where embedding similarity drops below threshold. Best quality, slowest. |

**Recommended chunk sizes:**

- General documents: `chunk_size=512, chunk_overlap=64`
- Short QA / FAQs: `chunk_size=256, chunk_overlap=32`
- Long technical docs: `chunk_size=1024, chunk_overlap=128`

---

### Embedding — `EmbeddingConfig`

The model used to convert text into vectors.

```python
cfg.embedding = EmbeddingConfig(model="BAAI/bge-small-en-v1.5", batch_size=32)
```

| Option       | Type | Default                  | Description                                              |
| ------------ | ---- | ------------------------ | -------------------------------------------------------- |
| `model`      | str  | `BAAI/bge-large-en-v1.5` | HuggingFace model name                                   |
| `batch_size` | int  | `256`                    | Docs encoded per batch. Lower if you hit OOM.            |
| `normalize`  | bool | `True`                   | L2-normalize vectors (required for cosine/IP similarity) |
| `api`        | str  | `None`                   | Set to `"openai"` to use OpenAI embeddings instead       |
| `api_key`    | str  | `None`                   | API key if using `api="openai"`                          |

**Model recommendations:**

| Model                    | Speed  | Quality | Use when               |
| ------------------------ | ------ | ------- | ---------------------- |
| `BAAI/bge-small-en-v1.5` | Fast   | Good    | Mac CPU, quick testing |
| `BAAI/bge-base-en-v1.5`  | Medium | Better  | Balanced Mac/Windows   |
| `BAAI/bge-large-en-v1.5` | Slow   | Best    | Windows/Linux with GPU |

**Mac safe batch size:** `32`. Windows GPU: `128–256`.

---

### BM25 — `BM25Config`

Keyword search that runs alongside FAISS. Results are merged via fusion.

```python
cfg.bm25 = BM25Config(enabled=True, k1=1.2, b=0.75)
```

| Option    | Type  | Default | Description                                                                        |
| --------- | ----- | ------- | ---------------------------------------------------------------------------------- |
| `enabled` | bool  | `True`  | Enable BM25 alongside FAISS                                                        |
| `k1`      | float | `1.5`   | Term frequency saturation. Higher = more weight on repeated terms. Range: 1.0–2.0. |
| `b`       | float | `0.75`  | Document length normalization. 0 = no normalization, 1 = full.                     |

**When BM25 helps most:** Technical docs with exact terms (gene names, product codes, legal citations). For conversational text, you can disable it.

---

### Fusion — `FusionConfig`

How FAISS and BM25 results are merged into a single ranked list.

```python
cfg.fusion = FusionConfig(mode="rrf", rrf_k=20)
```

| Option         | Type         | Default | Description                                                            |
| -------------- | ------------ | ------- | ---------------------------------------------------------------------- |
| `mode`         | `FusionMode` | `RRF`   | Merge strategy (see below)                                             |
| `rrf_k`        | int          | `60`    | RRF smoothing constant. Lower = stronger boost for top-ranked results. |
| `bm25_weight`  | float        | `0.3`   | BM25 weight in `WEIGHTED` mode                                         |
| `dense_weight` | float        | `0.7`   | FAISS weight in `WEIGHTED` mode                                        |

**Fusion modes:**

| `FusionMode` | Description                                                             |
| ------------ | ----------------------------------------------------------------------- |
| `RRF`        | Reciprocal Rank Fusion. Combines ranks, not scores. Robust default.     |
| `WEIGHTED`   | Weighted sum of normalized scores. Tune `bm25_weight` / `dense_weight`. |
| `DENSE`      | Disables BM25 entirely — pure FAISS vector search.                      |

**`rrf_k` tuning:**

- `rrf_k=20` — stronger boost for exact BM25 keyword matches (good for scientific/technical text)
- `rrf_k=60` — balanced default
- `rrf_k=100` — smoother merge, closer to pure dense retrieval

---

### Ranking — `RankingConfig`

What happens to the fused candidates before returning results.

```python
cfg.ranking = RankingConfig(mode="hybrid", mmr_lambda=0.7, k=10)
```

| Option                | Type          | Default                                | Description                                                           |
| --------------------- | ------------- | -------------------------------------- | --------------------------------------------------------------------- |
| `mode`                | `RankingMode` | `HYBRID`                               | Reranking pipeline (see below)                                        |
| `k`                   | int           | `20`                                   | Number of final results to return                                     |
| `cross_encoder_model` | str           | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model used in `HYBRID` and `CROSS` modes                              |
| `colbert_model`       | str           | `colbert-ir/colbertv2.0`               | Model used in `COLBERT` mode                                          |
| `mmr_lambda`          | float         | `0.5`                                  | MMR diversity weight. `1.0` = pure relevance, `0.0` = pure diversity. |

**Ranking modes:**

| `RankingMode` | Pipeline                              | Speed       | Quality                                  |
| ------------- | ------------------------------------- | ----------- | ---------------------------------------- |
| `FAST`        | FAISS + BM25 + RRF only               | Very fast   | Good — use this on Mac                   |
| `HYBRID`      | + cross-encoder reranking + MMR       | Slow on CPU | Best quality                             |
| `COLBERT`     | + ColBERT late interaction            | Medium      | High quality, lighter than cross-encoder |
| `CROSS`       | Cross-encoder only, no ANN pre-filter | Slowest     | High quality on small sets               |
| `NONE`        | Raw FAISS scores, no BM25             | Fastest     | Baseline only                            |

**`mmr_lambda` tuning:**

- `0.9` — almost all relevance, minimal diversity (good for focused factual QA)
- `0.7` — balanced (good default for most use cases)
- `0.5` — equal relevance + diversity (good for exploratory search)
- `0.3` — prioritize diversity (good for broad topic coverage)

---

### Context — `ContextConfig`

How retrieved chunks are assembled into the LLM prompt.

```python
cfg.context = ContextConfig(
    strategy=ContextStrategy.LOST_IN_MIDDLE,
    max_context_tokens=4096,
)
```

| Option               | Type              | Default          | Description                                |
| -------------------- | ----------------- | ---------------- | ------------------------------------------ |
| `strategy`           | `ContextStrategy` | `LOST_IN_MIDDLE` | Chunk ordering in prompt                   |
| `max_context_tokens` | int               | `4096`           | Max tokens in context window               |
| `return_context`     | bool              | `False`          | Include assembled context in `QueryResult` |

**Context strategies:**

| `ContextStrategy`    | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `SCORE_ORDER`        | Highest-scored chunks first. Simple and fast.                                     |
| `LOST_IN_MIDDLE`     | Best chunks at start and end, weaker ones in middle. Exploits LLM attention bias. |
| `TRUNCATE`           | Hard-cut at `max_context_tokens`.                                                 |
| `SUMMARIZE_OVERFLOW` | Summarize chunks that don't fit instead of dropping them.                         |

---

### Cache — `CacheConfig`

Semantic cache — returns cached answers for near-duplicate queries.

```python
cfg.cache = CacheConfig(enabled=True, similarity_threshold=0.97, ttl_seconds=3600)
```

| Option                 | Type  | Default | Description                                          |
| ---------------------- | ----- | ------- | ---------------------------------------------------- |
| `enabled`              | bool  | `False` | Enable semantic cache                                |
| `similarity_threshold` | float | `0.97`  | Cosine similarity above which a query is a cache hit |
| `ttl_seconds`          | int   | `3600`  | How long cached results live (seconds)               |
| `max_size`             | int   | `10000` | Max number of cached queries                         |

---

### LLM — `LLMConfig`

Only needed if you use `ks.query()`.

```python
cfg.llm = LLMConfig(model="gpt-4o-mini", api_key="sk-...", temperature=0.0)
```

| Option        | Type  | Default       | Description                                              |
| ------------- | ----- | ------------- | -------------------------------------------------------- |
| `model`       | str   | `gpt-4o-mini` | Model name                                               |
| `api_key`     | str   | `None`        | API key                                                  |
| `base_url`    | str   | `None`        | Custom endpoint (for OpenAI-compatible APIs like Ollama) |
| `temperature` | float | `0.0`         | LLM temperature. `0.0` = deterministic.                  |
| `max_tokens`  | int   | `1024`        | Max tokens in LLM response                               |

---

## Recommended configs by use case

**Mac development (fast, safe):**

```python
cfg.embedding = EmbeddingConfig(model="BAAI/bge-small-en-v1.5", batch_size=32)
cfg.index     = IndexConfig(type=IndexType.FLAT)
cfg.ranking   = RankingConfig(mode="fast", k=10)
```

**Windows/Linux production (best quality):**

```python
cfg.embedding = EmbeddingConfig(model="BAAI/bge-large-en-v1.5", batch_size=128)
cfg.index     = IndexConfig(type=IndexType.FLAT)   # or VLQ_ADC for >100k docs
cfg.bm25      = BM25Config(enabled=True, k1=1.2, b=0.75)
cfg.fusion    = FusionConfig(mode="rrf", rrf_k=20)
cfg.ranking   = RankingConfig(mode="hybrid", mmr_lambda=0.7, k=10)
```

**Large corpus (>100k docs):**

```python
cfg.index   = IndexConfig(type=IndexType.VLQ_ADC, nlist=1024)
cfg.ranking = RankingConfig(mode="hybrid", k=10)
```

---

## Benchmark results (SciFact, 2,000 docs)

Mac CPU, `bge-small-en-v1.5`, `FAST` ranking mode (no cross-encoder):

| Method                       | Recall@10 | nDCG@10 | MRR    |
| ---------------------------- | --------- | ------- | ------ |
| KSelect (BM25 + FAISS + RRF) | 0.419     | 0.366   | 0.362  |
| LangChain + Chroma (vanilla) | 0.399     | 0.353   | 0.350  |
| Delta                        | +0.021    | +0.013  | +0.011 |

With full corpus (5,183 docs) + cross-encoder reranking on GPU, the gap widens further.

Run it yourself:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/compare_kselect_langchain.py
```

---

## Multi-tenant

```python
from kselect import MultiTenantKSelect

mt = MultiTenantKSelect()
mt.create_tenant("customer_a")
mt.get("customer_a").from_folder("docs/customer_a/")

result = mt.get("customer_a").search("query")
```

---

## License

MIT
