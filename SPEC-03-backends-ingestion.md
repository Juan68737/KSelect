# KSelect — SPEC Part 3 of 5: Backends and Ingestion

**Read order:** After SPEC-02. This covers the `VectorBackend` abstraction, all concrete backend implementations, the URI factory, document loaders, chunking strategies, and the ingestion pipeline that produces `list[Chunk]`.

---

## 5. Core Abstractions

### 5.1 `VectorBackend` ABC — `backends/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from kselect.models.chunk import Chunk


class VectorBackend(ABC):
    """
    Abstract interface over any vector store.
    All backends must implement these five methods.
    IndexManager calls these; it does not call backend-specific methods directly.
    """

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        """
        Return all chunks with embeddings populated.
        Called during initial index build from an existing backend.
        For large backends, implementations MUST stream in batches of 10,000.
        Never load the full table into memory.
        """

    @abstractmethod
    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Return specific chunks by chunk_id. Used during reranking for text lookup."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """
        Write chunks to the backend. Called by add_doc() / add_folder().
        For LocalBackend: no-op (FAISS is the source of truth).
        For remote backends: write vectors + metadata to the database.
        """

    @abstractmethod
    def delete_chunks(self, ids: list[str]) -> None:
        """Delete chunks by chunk_id. Used for document removal."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks stored."""
```

### 5.2 `LLMClient` ABC — `llm/base.py`

```python
from abc import ABC, abstractmethod
from kselect.models.hit import Hit


class LLMClient(ABC):

    @abstractmethod
    async def generate(
        self,
        query: str,
        context_chunks: list[Hit],
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """
        Generate an answer given a query and retrieved context.
        Returns (answer_text, confidence_score).
        Confidence derived from token logprobs when available;
        otherwise estimated from cross-encoder scores of context chunks.
        """
```

---

## 6. Backend Layer

### 6.1 `LocalBackend` — `backends/local.py`

The local backend is a passthrough. FAISS and BM25 indexes are the source of truth for search. `get_all_chunks()` reads from the serialized `chunks.jsonl` file in the state directory. `upsert_chunks()` appends to the same file. No network calls.

```python
class LocalBackend(VectorBackend):
    """
    Wraps a local directory containing serialized chunks.
    Used when KSelect is initialized via from_folder() / from_csv() / etc.
    The chunk store is a JSONL file: one Chunk JSON per line.
    """
    def __init__(self, state_dir: str): ...

    # get_all_chunks: read chunks.jsonl line by line, yield Chunk
    # upsert_chunks: append new Chunk lines to chunks.jsonl
    # get_chunks_by_ids: load full JSONL into id→Chunk dict, return requested ids
    # delete_chunks: rewrite JSONL excluding deleted ids
    # count: wc -l equivalent on chunks.jsonl
```

### 6.2 `PGVectorBackend` — `backends/pgvector.py`

Uses `psycopg3` (the `psycopg` package, NOT `psycopg2`). Sync connection pool for indexing operations; async connection pool for serving. `get_all_chunks()` streams in batches of 10,000 via server-side cursors.

```python
class PGVectorBackend(VectorBackend):
    def __init__(self, dsn: str, table: str, text_col: str, metadata_cols: list[str]): ...

    @classmethod
    def from_uri(
        cls,
        uri: str,
        text_col: str = "content",
        metadata_cols: list[str] | None = None,
    ) -> "PGVectorBackend":
        """
        Parse "pgvector://host/dbname?table=legal_cases"
        Reconstructs DSN as: postgresql://host/dbname
        Table name extracted from query string.
        """
```

**Schema contract.** KSelect requires only:
- A primary key column (any name, any int/uuid type)
- A text content column (name set via `text_col`, default `"content"`)
- An embedding column of type `vector(N)` where N must match the embedding model dim

KSelect **never writes DDL**. It never alters the table schema. If the schema does not match, raise `KSelectBackendError` with a clear message.

### 6.3 `PineconeBackend` — `backends/pinecone.py`

```python
class PineconeBackend(VectorBackend):
    """
    Wraps pinecone-client v4+.
    get_all_chunks: uses list() + fetch() in batches of 1000 (Pinecone API limit).
    upsert_chunks: uses upsert() in batches of 100 vectors.
    Metadata stored as Pinecone metadata dict alongside the vector.
    """

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> "PineconeBackend":
        """
        Parse "pinecone://index-name/namespace"
        API key read from PINECONE_API_KEY env var.
        """
```

### 6.4 `ChromaDBBackend` — `backends/chromadb.py`

```python
class ChromaDBBackend(VectorBackend):
    """
    Wraps chromadb v0.5+.
    Client mode: http (if uri contains host) or embedded (local path).
    Collection name extracted from uri path component.
    """

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> "ChromaDBBackend":
        """
        Parse "chromadb://collection_name"           → embedded, collection=collection_name
        Parse "chromadb://host:8000/collection_name" → http client
        """
```

### 6.5 URI Factory — `backends/factory.py`

```python
URI_SCHEME_MAP: dict[str, type[VectorBackend]] = {
    "local":    LocalBackend,
    "pgvector": PGVectorBackend,
    "pinecone": PineconeBackend,
    "chromadb": ChromaDBBackend,
}


def parse_backend_uri(uri: str, **kwargs) -> VectorBackend:
    """
    Dispatch URI to the correct backend class.

    Examples:
      "pgvector://prod-db/legal_cases?table=legal_cases" → PGVectorBackend(...)
      "local://kselect_state/"                           → LocalBackend(...)
      "pinecone://acme-index/us-east-1"                  → PineconeBackend(...)

    Raises KSelectConfigError for unknown schemes.
    Raises KSelectConfigError if required optional dependency is not installed
    (e.g. psycopg not installed when using pgvector://).
    """
    scheme = uri.split("://")[0]
    if scheme not in URI_SCHEME_MAP:
        raise KSelectConfigError(
            f"Unknown backend scheme: {scheme!r}. "
            f"Valid schemes: {list(URI_SCHEME_MAP)}"
        )
    return URI_SCHEME_MAP[scheme].from_uri(uri, **kwargs)
```

---

## 7. Ingestion Layer

### 7.1 Loaders — `ingestion/loaders.py`

Each loader returns `list[tuple[str, dict]]` — raw document text paired with a flat metadata dict. Chunking happens downstream in `Chunker`. Loaders do not chunk.

```python
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[tuple[str, dict]]:
        """Returns list of (raw_text, metadata_dict) pairs. One pair per document."""


class FolderLoader(BaseLoader):
    """
    Recursively walks a directory tree. Dispatches by file extension:
      .pdf   → pypdf.PdfReader (page text concatenated)
      .docx  → python-docx Document (paragraph text concatenated)
      .txt   → plain utf-8 read
      .md    → plain read (preserve markdown syntax — do not strip)

    extract_tables=True: routes .pdf through unstructured.partition_pdf()
    for table extraction before text extraction. Slower but captures tabular data.

    Skips files:
      - Under min_file_size_bytes (default: 64 bytes)
      - Over max_file_size_mb (default: 50 MB)
      - With extensions not in the supported list above
      Logs a warning for each skipped file at DEBUG level.

    max_docs: if set, stop after loading this many documents (for testing).
    """
    def __init__(
        self,
        path: str,
        extract_tables: bool = False,
        min_file_size_bytes: int = 64,
        max_file_size_mb: float = 50.0,
        max_docs: int | None = None,
    ): ...


class CSVLoader(BaseLoader):
    """
    text_col is required — raises KSelectIngestionError if column not found.
    All other columns not in metadata list are ignored.
    metadata: list of column names to include as metadata. If None, includes all columns.
    vector_col: if provided, parse this column as a list[float] embedding.
                IngestionPipeline will skip re-embedding when vector_col is present.
    """
    def __init__(
        self,
        path: str,
        text_col: str,
        metadata: list[str] | None = None,
        vector_col: str | None = None,
    ): ...


class JSONLoader(BaseLoader):
    """
    Handles both:
    - JSON array: top-level list of objects
    - JSONL: one JSON object per line (auto-detected by attempting json.loads on first line)
    text_key: required. The key whose value is the document text.
    metadata: list of keys to include as metadata.
    """
    def __init__(
        self,
        path: str,
        text_key: str,
        metadata: list[str] | None = None,
    ): ...
```

### 7.2 Chunker — `ingestion/chunking.py`

```python
class Chunker:
    """
    Stateless transformer: (text, metadata_dict) → list[Chunk].
    Strategy selected via ChunkingConfig.strategy.

    All strategies enforce min_chunk_length: chunks shorter than this threshold
    are merged with the previous chunk rather than discarded or kept as-is.

    Each Chunk receives a uuid4 id assigned here. chunk_index is the 0-based
    position within the parent document.
    """

    def chunk(
        self,
        text: str,
        metadata: dict,
        config: "ChunkingConfig",
    ) -> list[Chunk]:
        """Dispatch to the appropriate private strategy method."""

    def _sliding_window(self, text: str, size: int, overlap: int) -> list[str]:
        """
        Token-based sliding window using tiktoken with cl100k_base encoding.
        Encodes full text → splits token list → decodes each window.
        overlap is in tokens, not characters.
        """

    def _sentence(self, text: str, size: int, overlap_sentences: int) -> list[str]:
        """
        NLTK punkt_tab sentence tokenizer.
        Packs sentences greedily until the token budget (size) is full.
        overlap_sentences: number of sentences from the end of the previous
        chunk to prepend to the next chunk.
        """

    def _semantic(self, text: str, threshold: float) -> list[str]:
        """
        Embedding-based semantic chunking.
        1. Tokenize into sentences with NLTK.
        2. Embed each sentence with the configured embedding model (same model as index).
        3. Compute cosine similarity between each adjacent sentence pair.
        4. Insert a chunk boundary where similarity < threshold (topic shift).
        5. Merge resulting chunks smaller than min_chunk_length.

        WARNING: O(n_sentences) embedding calls. Expensive without GPU.
        Log a warning if called on text with >500 sentences.
        """

    def _paragraph(self, text: str) -> list[str]:
        """
        Split on double newline (\\n\\n).
        Normalize whitespace within each paragraph.
        Merge paragraphs shorter than min_chunk_length with the previous paragraph.
        """
```

### 7.3 IngestionPipeline — `ingestion/pipeline.py`

```python
class IngestionPipeline:
    """
    Orchestrates the full document → Chunk pipeline:
      BaseLoader.load() → Chunker.chunk() → Embedder.embed() → list[Chunk]

    Processing is batched: embed in groups of EmbeddingConfig.batch_size.
    Logs progress at INFO level every 1,000 chunks processed.

    remove_duplicates=True: compute SHA256(chunk.text) for each chunk.
    Skip chunks whose hash matches any chunk already in existing_chunk_ids.
    This is a text-level dedup, not a semantic dedup.
    """

    def run(
        self,
        loader: BaseLoader,
        config: "KSelectConfig",
        existing_chunk_ids: set[str] | None = None,
    ) -> list[Chunk]:
        """
        existing_chunk_ids: set of chunk ids already in the index.
        Used by add_doc() / add_folder() for incremental runs to skip already-indexed content.
        Pass None for initial full builds.

        Returns list[Chunk] with embedding populated on every Chunk.
        Raises KSelectIngestionError if any document fails to parse,
        wrapping the original exception with the file path in the message.
        """
```

---

### Phase 2 test targets

```python
# tests/unit/test_ingestion.py

def test_folder_loader_pdf(tmp_path):
    """FolderLoader parses a PDF and returns non-empty text."""

def test_csv_loader_missing_col():
    """CSVLoader raises KSelectIngestionError when text_col not found."""

def test_chunker_sliding_window():
    """Chunks are within chunk_size token budget. No chunk exceeds size."""

def test_chunker_min_length_merge():
    """Chunks below min_chunk_length are merged with previous chunk, not dropped."""

def test_chunker_overlap():
    """Adjacent chunks share the expected number of overlapping tokens."""

def test_ingestion_pipeline_dedup():
    """Duplicate documents (same text hash) are skipped when existing_chunk_ids provided."""

def test_ingestion_pipeline_embedding_shape():
    """All returned Chunks have embedding populated with correct dimensionality."""
```

---

*Continue to SPEC-04-index-retrieval-ranking.md*
