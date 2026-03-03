from __future__ import annotations

import threading
from typing import Any

from kselect.exceptions import TenantNotFoundError, TenantError
from kselect.kselect import KSelect
from kselect.models.answer import QueryResult
from kselect.models.config import KSelectConfig
from kselect.models.hit import SearchResult


class MultiTenantKSelect:
    """
    Manages N independent KSelect instances, one per tenant.

    The embedding model is shared across all tenants (loaded once, reused).
    Each tenant has its own FAISSIndex, BM25Index, and SemanticCache.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, KSelect] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_yaml(cls, path: str) -> "MultiTenantKSelect":
        """
        YAML format:
          tenants:
            acme:
              backend_uri: "pgvector://prod-db/acme_docs"
              index_type: "faiss_vlq_adc"
            globex:
              backend_uri: "pgvector://prod-db/globex_docs"
        """
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            raw = yaml.safe_load(f)

        mt = cls()
        global_defaults = {k: v for k, v in raw.items() if k != "tenants"}
        tenant_defs = raw.get("tenants", {})

        for tenant_id, tenant_cfg_raw in tenant_defs.items():
            merged = {**global_defaults, **tenant_cfg_raw}
            backend_uri = merged.pop("backend_uri", None)
            if not backend_uri:
                raise ValueError(f"Tenant '{tenant_id}' missing backend_uri in {path}")

            cfg = KSelectConfig.model_validate(merged) if merged else KSelectConfig()
            ks = KSelect.from_backend(backend_uri, config=cfg)
            mt._tenants[tenant_id] = ks

        return mt

    def search(self, tenant: str, query: str, **kwargs) -> SearchResult:
        return self._get_tenant(tenant).search(query, **kwargs)

    def query(self, tenant: str, query: str, **kwargs) -> QueryResult:
        return self._get_tenant(tenant).query(query, **kwargs)

    def add_tenant(self, tenant_id: str, config: KSelectConfig) -> None:
        with self._lock:
            if tenant_id in self._tenants:
                raise TenantError(f"Tenant '{tenant_id}' already exists.")
            if config.backend is None:
                raise TenantError(f"Tenant '{tenant_id}' config must have backend.uri set.")
            ks = KSelect.from_backend(config.backend.uri, config=config)
            self._tenants[tenant_id] = ks

    def remove_tenant(self, tenant_id: str) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(tenant_id)
            del self._tenants[tenant_id]

    def tenants(self) -> list[str]:
        with self._lock:
            return sorted(self._tenants.keys())

    def _get_tenant(self, tenant_id: str) -> KSelect:
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(tenant_id)
            return self._tenants[tenant_id]
