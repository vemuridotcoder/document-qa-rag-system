import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import HybridRetriever


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    distance: float
    confidence: str
    metadata: dict


def _chunk(text: str, distance: float):
    return FakeChunk(
        chunk_id=text[:4],
        text=text,
        source="s",
        chunk_index=0,
        distance=distance,
        confidence="high",
        metadata={"filename": "x.txt"},
    )


def test_hybrid_rerank_prefers_lexical_match():
    retriever = HybridRetriever(lexical_weight=0.6, mmr_lambda=1.0)
    candidates = [
        _chunk("finance revenue report", 0.2),
        _chunk("education budget 2024", 0.25),
    ]
    ranked = retriever.rerank("education budget", candidates, top_k=1)
    assert ranked[0].text == "education budget 2024"
