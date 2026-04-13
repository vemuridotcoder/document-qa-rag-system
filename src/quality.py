"""Reusable retrieval and latency quality metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    mrr: float


def hit_rate_at_k(relevance_ranks: list[int | None], k: int) -> float:
    if not relevance_ranks:
        return 0.0
    hits = sum(1 for r in relevance_ranks if r is not None and r <= k)
    return hits / len(relevance_ranks)


def mean_reciprocal_rank(relevance_ranks: list[int | None]) -> float:
    if not relevance_ranks:
        return 0.0
    rr = [1.0 / r if r is not None and r > 0 else 0.0 for r in relevance_ranks]
    return sum(rr) / len(rr)


def compute_retrieval_metrics(
    relevance_ranks: list[int | None], k: int = 3
) -> RetrievalMetrics:
    return RetrievalMetrics(
        hit_at_k=hit_rate_at_k(relevance_ranks, k),
        mrr=mean_reciprocal_rank(relevance_ranks),
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    values = sorted(values)
    idx = (len(values) - 1) * (p / 100)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac
