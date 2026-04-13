"""Hybrid retrieval and reranking utilities for production RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RankedChunk:
    chunk: Any
    score: float


class HybridRetriever:
    """Reranks dense retrieval candidates with lexical signal + MMR diversification."""

    def __init__(self, lexical_weight: float = 0.25, mmr_lambda: float = 0.7):
        self.lexical_weight = lexical_weight
        self.mmr_lambda = mmr_lambda

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _lexical_score(self, query: str, text: str) -> float:
        q = self._tokenize(query)
        t = self._tokenize(text)
        if not q or not t:
            return 0.0
        return len(q & t) / max(len(q | t), 1)

    def _hybrid_score(self, query: str, item: Any) -> float:
        dense_similarity = 1.0 - float(item.distance)
        lexical_similarity = self._lexical_score(query, item.text)
        return (
            1 - self.lexical_weight
        ) * dense_similarity + self.lexical_weight * lexical_similarity

    def rerank(self, query: str, candidates: list[Any], top_k: int) -> list[Any]:
        if not candidates:
            return []

        ranked = [
            RankedChunk(chunk=c, score=self._hybrid_score(query, c)) for c in candidates
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)

        selected: list[RankedChunk] = []
        remaining = ranked.copy()
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_idx = 0
            best_mmr = float("-inf")
            for idx, cand in enumerate(remaining):
                max_redundancy = max(
                    self._lexical_score(cand.chunk.text, s.chunk.text) for s in selected
                )
                mmr = (
                    self.mmr_lambda * cand.score
                    - (1 - self.mmr_lambda) * max_redundancy
                )
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return [x.chunk for x in selected]
