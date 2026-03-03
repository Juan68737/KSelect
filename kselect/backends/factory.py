from __future__ import annotations

from kselect.backends.base import VectorBackend
from kselect.exceptions import ConfigError


def parse_backend_uri(uri: str, **kwargs: object) -> VectorBackend:
    """
    Dispatch URI to the correct backend class.

    Examples:
      "pgvector://prod-db/legal_cases?table=legal_cases" → PGVectorBackend(...)
      "local://kselect_state/"                           → LocalBackend(...)
      "pinecone://acme-index/us-east-1"                  → PineconeBackend(...)
      "chromadb://collection_name"                       → ChromaDBBackend(...)

    Raises ConfigError for unknown schemes.
    Raises ConfigError if required optional dependency is not installed.
    """
    if "://" not in uri:
        raise ConfigError(f"Invalid backend URI (missing scheme): {uri!r}")

    scheme = uri.split("://")[0]

    # Lazy imports keep optional deps from blowing up on import
    if scheme == "local":
        from kselect.backends.local import LocalBackend
        return LocalBackend.from_uri(uri, **kwargs)

    if scheme == "pgvector":
        from kselect.backends.pgvector import PGVectorBackend
        return PGVectorBackend.from_uri(uri, **kwargs)

    if scheme == "pinecone":
        from kselect.backends.pinecone import PineconeBackend
        return PineconeBackend.from_uri(uri, **kwargs)

    if scheme == "chromadb":
        from kselect.backends.chromadb import ChromaDBBackend
        return ChromaDBBackend.from_uri(uri, **kwargs)

    raise ConfigError(
        f"Unknown backend scheme: {scheme!r}. "
        f"Valid schemes: ['local', 'pgvector', 'pinecone', 'chromadb']"
    )
