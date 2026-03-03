from __future__ import annotations

"""
Prometheus metrics for KSelect.
KSelectMetrics wraps prometheus_client counters/histograms/gauges.
If prometheus_client is not installed, NullMetrics is used transparently.
"""


class _NullMetrics:
    """No-op emitter — used when prometheus_client is not installed."""

    def observe_query(self, mode: str, cache_hit: bool, latency_ms: float) -> None:
        pass

    def observe_retrieval(self, latency_ms: float) -> None:
        pass

    def observe_rerank(self, latency_ms: float) -> None:
        pass

    def set_cache_hit_rate(self, rate: float) -> None:
        pass

    def set_confidence(self, p50: float, p95: float) -> None:
        pass

    def set_index_stats(self, size: int, drift: float) -> None:
        pass


class KSelectMetrics:
    """
    Prometheus metrics with 'kselect_' prefix.
    Instantiate once and pass to KSelect via the _metrics parameter.
    Falls back to NullMetrics if prometheus_client is not installed.
    """

    def __init__(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import-untyped]

            self._query_total = Counter(
                "kselect_query_total",
                "Total queries processed",
                ["mode", "cache_hit"],
            )
            self._query_latency = Histogram(
                "kselect_query_latency_ms",
                "End-to-end query latency in milliseconds",
                buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
            )
            self._retrieval_latency = Histogram(
                "kselect_retrieval_latency_ms",
                "FAISS + BM25 retrieval latency",
                buckets=[1, 2, 5, 10, 25, 50, 100, 250],
            )
            self._rerank_latency = Histogram(
                "kselect_rerank_latency_ms",
                "Cross-encoder / ColBERT reranker latency",
                buckets=[5, 10, 25, 50, 100, 250, 500],
            )
            self._cache_hit_rate = Gauge("kselect_cache_hit_rate", "Rolling 1000-query cache hit rate")
            self._confidence_p50 = Gauge("kselect_confidence_p50", "Rolling p50 answer confidence")
            self._confidence_p95 = Gauge("kselect_confidence_p95", "Rolling p95 answer confidence")
            self._index_size = Gauge("kselect_index_size", "Total vectors in FAISS index")
            self._index_drift = Gauge("kselect_index_drift", "Index drift ratio (0.0–1.0)")
            self._null = False
        except ImportError:
            self._null = True

    def observe_query(self, mode: str, cache_hit: bool, latency_ms: float) -> None:
        if self._null:
            return
        self._query_total.labels(mode=mode, cache_hit=str(cache_hit).lower()).inc()
        self._query_latency.observe(latency_ms)

    def observe_retrieval(self, latency_ms: float) -> None:
        if self._null:
            return
        self._retrieval_latency.observe(latency_ms)

    def observe_rerank(self, latency_ms: float) -> None:
        if self._null:
            return
        self._rerank_latency.observe(latency_ms)

    def set_cache_hit_rate(self, rate: float) -> None:
        if self._null:
            return
        self._cache_hit_rate.set(rate)

    def set_confidence(self, p50: float, p95: float) -> None:
        if self._null:
            return
        self._confidence_p50.set(p50)
        self._confidence_p95.set(p95)

    def set_index_stats(self, size: int, drift: float) -> None:
        if self._null:
            return
        self._index_size.set(size)
        self._index_drift.set(drift)


def null_metrics() -> _NullMetrics:
    return _NullMetrics()
