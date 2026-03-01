from __future__ import annotations

from pydantic import BaseModel


class CacheStats(BaseModel):
    hit_rate: float
    false_positive_rate: float
    size: int
    llm_calls_saved: int
    cost_saved_usd: float
    qps_saved: float
