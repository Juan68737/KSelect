"""
KSelect error taxonomy (SPEC-05 §18).

Hierarchy
---------
KSelectError                        — base for all library errors
├── ConfigError                     — bad / missing configuration
│   └── EnvVarMissingError          — required ${VAR} not set
├── BackendError                    — storage backend failures
│   ├── BackendConnectionError      — cannot connect to backend
│   ├── BackendWriteError           — write / upsert failed
│   └── BackendReadError            — read / query failed
├── IndexError                      — FAISS / BM25 index issues
│   ├── IndexNotBuiltError          — search called before build/load
│   ├── IndexSaveError              — cannot persist index to disk
│   └── IndexLoadError              — cannot restore index from disk
├── IngestionError                  — document loading / chunking failures
│   ├── LoaderError                 — file cannot be read or parsed
│   └── ChunkingError               — chunking strategy failure
├── EmbeddingError                  — embedding model failures
│   ├── EmbeddingModelNotFoundError — model weights not available
│   └── EmbeddingDimensionMismatch  — dim mismatch between index and model
├── RetrievalError                  — retrieval-stage failures
│   └── EmptyIndexError             — search on an empty index
├── RankingError                    — reranking failures
│   ├── CrossEncoderError           — cross-encoder reranker failed
│   └── ColBERTError                — ColBERT reranker failed
├── ContextError                    — context assembly failures
│   └── ContextOverflowError        — context exceeds max tokens with no fallback
├── CacheError                      — semantic cache failures
├── LLMError                        — LLM client failures
│   ├── LLMConnectionError          — cannot reach LLM endpoint
│   ├── LLMRateLimitError           — rate limited by LLM provider
│   ├── LLMTimeoutError             — LLM call timed out
│   └── LLMResponseError            — unexpected / malformed LLM response
├── TenantError                     — multi-tenant isolation failures
│   ├── TenantNotFoundError         — unknown tenant_id
│   └── TenantQuotaExceededError    — tenant quota exceeded
└── EvalError                       — evaluation / benchmarking failures
"""


# ── Base ──────────────────────────────────────────────────────────────────────


class KSelectError(Exception):
    """Base class for all KSelect errors."""


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigError(KSelectError):
    """Raised for bad or missing configuration."""


class EnvVarMissingError(ConfigError):
    """Raised when a required ${VAR} environment variable is not set."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f"Required environment variable '{var_name}' is not set.")
        self.var_name = var_name


# ── Backend ───────────────────────────────────────────────────────────────────


class BackendError(KSelectError):
    """Raised for storage backend failures."""


class BackendConnectionError(BackendError):
    """Raised when a connection to the backend cannot be established."""


class BackendWriteError(BackendError):
    """Raised when a write or upsert operation to the backend fails."""


class BackendReadError(BackendError):
    """Raised when a read or query operation from the backend fails."""


# ── Index ─────────────────────────────────────────────────────────────────────


class IndexError(KSelectError):
    """Raised for FAISS / BM25 index issues."""


class IndexNotBuiltError(IndexError):
    """Raised when search is called before the index has been built or loaded."""


class IndexSaveError(IndexError):
    """Raised when the index cannot be persisted to disk."""


class IndexLoadError(IndexError):
    """Raised when the index cannot be restored from disk."""


# ── Ingestion ─────────────────────────────────────────────────────────────────


class IngestionError(KSelectError):
    """Raised for document loading or chunking failures."""


class LoaderError(IngestionError):
    """Raised when a file cannot be read or parsed by a loader."""


class ChunkingError(IngestionError):
    """Raised when a chunking strategy fails to produce valid chunks."""


# ── Embedding ─────────────────────────────────────────────────────────────────


class EmbeddingError(KSelectError):
    """Raised for embedding model failures."""


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the requested embedding model weights are not available."""


class EmbeddingDimensionMismatch(EmbeddingError):
    """Raised when the embedding dimension does not match the existing index."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            f"Embedding dimension mismatch: index expects {expected}, model produces {got}."
        )
        self.expected = expected
        self.got = got


# ── Retrieval ─────────────────────────────────────────────────────────────────


class RetrievalError(KSelectError):
    """Raised for retrieval-stage failures."""


class EmptyIndexError(RetrievalError):
    """Raised when search is called on an empty index."""


# ── Ranking ───────────────────────────────────────────────────────────────────


class RankingError(KSelectError):
    """Raised for post-retrieval ranking failures."""


class CrossEncoderError(RankingError):
    """Raised when the cross-encoder reranker encounters an error."""


class ColBERTError(RankingError):
    """Raised when the ColBERT reranker encounters an error."""


# ── Context ───────────────────────────────────────────────────────────────────


class ContextError(KSelectError):
    """Raised for context assembly failures."""


class ContextOverflowError(ContextError):
    """Raised when context exceeds max_context_tokens and no fallback is configured."""


# ── Cache ─────────────────────────────────────────────────────────────────────


class CacheError(KSelectError):
    """Raised for semantic cache failures."""


# ── LLM ──────────────────────────────────────────────────────────────────────


class LLMError(KSelectError):
    """Raised for LLM client failures."""


class LLMConnectionError(LLMError):
    """Raised when the LLM endpoint cannot be reached."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits the request."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM call exceeds the configured timeout."""


class LLMResponseError(LLMError):
    """Raised when the LLM returns an unexpected or malformed response."""


# ── Multi-tenant ──────────────────────────────────────────────────────────────


class TenantError(KSelectError):
    """Raised for multi-tenant isolation failures."""


class TenantNotFoundError(TenantError):
    """Raised when the requested tenant_id is not registered."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"Tenant '{tenant_id}' not found.")
        self.tenant_id = tenant_id


class TenantQuotaExceededError(TenantError):
    """Raised when a tenant exceeds its configured resource quota."""


# ── Eval ─────────────────────────────────────────────────────────────────────


class EvalError(KSelectError):
    """Raised for evaluation or benchmarking failures."""

# ── SPEC-05 canonical name aliases (§18) ─────────────────────────────────────
# These names are used in the public API and documentation.
# The underlying classes are the same objects.

KSelectConfigError = ConfigError
KSelectIndexError = IndexError
KSelectBackendError = BackendError
KSelectIngestionError = IngestionError
KSelectEmbeddingError = EmbeddingError
KSelectRankingError = RankingError
KSelectLLMError = LLMError
KSelectTenantError = TenantError
KSelectVersionError = IndexLoadError  # raised on version mismatch during load()
