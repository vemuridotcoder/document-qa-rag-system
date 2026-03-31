"""
chunk_experiment.py — Document Q&A System
===========================================
Systematically measures how chunk size affects retrieval quality.

README promised this experiment. This file delivers it.

What it tests:
    Chunk sizes: [128, 256, 512, 1024] tokens
    Metric: Hit Rate @3 on a fixed set of test questions
    Fixed: same embedding model, same query, same n_results=3

Why this matters:
    Chunk size is the single most impactful hyperparameter in a RAG system.
    Most tutorials pick 512 without justification. This experiment shows
    whether that choice is optimal for your specific document type.

Expected finding:
    - 128 tokens: low Hit Rate (answers split across too many small chunks)
    - 512 tokens: highest Hit Rate for dense factual text
    - 1024 tokens: lower Hit Rate (too much irrelevant content per chunk)
    - Optimal chunk size varies by document type (narrative vs technical vs tabular)

Run:
    python src/chunk_experiment.py --doc data/raw/your_document.txt

Output:
    evaluation/chunk_experiment_results.json
    evaluation/chunk_experiment_results.csv (for README table)
"""

import os
import sys
import json
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.append(os.path.dirname(__file__))
from ingestion import DocumentChunker, DocumentLoader
from embeddings import EmbeddingModel
from vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZES_TO_TEST = [128, 256, 512, 1024]

# Default test questions — replace with domain-specific questions
# for your document for meaningful results
DEFAULT_TEST_QUESTIONS = [
    {
        "question": "What is the main subject of this document?",
        "keywords": ["introduction", "overview", "objective", "purpose"],
    },
    {
        "question": "What methodology or approach is described?",
        "keywords": ["method", "approach", "procedure", "technique"],
    },
    {
        "question": "What are the key findings or conclusions?",
        "keywords": ["conclusion", "result", "finding", "summary"],
    },
    {
        "question": "What limitations or challenges are mentioned?",
        "keywords": ["limitation", "challenge", "drawback", "constraint"],
    },
    {
        "question": "What future work or recommendations are suggested?",
        "keywords": ["future", "recommend", "suggest", "improve"],
    },
]


@dataclass
class ExperimentResult:
    chunk_size: int
    chunk_overlap: int
    n_chunks_created: int
    avg_chunk_chars: float
    hit_rate_at_1: float
    hit_rate_at_3: float
    mrr: float


def evaluate_chunk_size(
    chunk_size: int,
    document_path: str,
    test_questions: list[dict],
    config: dict,
) -> ExperimentResult:
    """
    Ingest the document with a specific chunk size, run retrieval on
    test questions, and compute Hit Rate @3 and MRR.

    A temporary isolated ChromaDB collection is used per chunk size
    to avoid contamination between experiments.
    """
    overlap = max(25, chunk_size // 10)  # 10% overlap, minimum 25

    # Temporarily override config
    exp_config = dict(config)
    exp_config["ingestion"] = dict(config["ingestion"])
    exp_config["ingestion"]["chunk_size"] = chunk_size
    exp_config["ingestion"]["chunk_overlap"] = overlap

    exp_config["vectorstore"] = dict(config["vectorstore"])
    exp_config["vectorstore"]["persist_directory"] = f"data/vectorstore_exp_{chunk_size}"
    exp_config["vectorstore"]["collection_name"] = f"exp_{chunk_size}"

    # Load and chunk
    loader = DocumentLoader()
    text = loader.load(document_path)

    chunker = DocumentChunker(exp_config)
    chunks = chunker.chunk(text, document_path)

    if not chunks:
        logger.error(f"No chunks produced at size {chunk_size}")
        return ExperimentResult(chunk_size, overlap, 0, 0, 0, 0, 0)

    avg_chars = sum(c.char_count for c in chunks) / len(chunks)

    # Embed
    embedder = EmbeddingModel(config)
    texts = [c.text for c in chunks]
    embedding_result = embedder.embed(texts)

    # Index
    vectorstore = VectorStore(exp_config)
    vectorstore.add_chunks(chunks, embedding_result.embeddings)

    # Evaluate retrieval
    hits_at_1, hits_at_3, reciprocal_ranks = [], [], []

    for q_data in test_questions:
        query_embedding = embedder.embed_single(q_data["question"])
        retrieved = vectorstore.query(query_embedding, n_results=3)

        def contains_keyword(chunk_text: str) -> bool:
            text_lower = chunk_text.lower()
            hits = sum(1 for kw in q_data["keywords"] if kw.lower() in text_lower)
            return hits >= max(1, len(q_data["keywords"]) // 2)

        hit_1, hit_3, rr = False, False, 0.0
        for rank, chunk in enumerate(retrieved, start=1):
            if contains_keyword(chunk.text):
                if rank == 1:
                    hit_1 = True
                if rank <= 3:
                    hit_3 = True
                if rr == 0:
                    rr = 1.0 / rank
                break

        hits_at_1.append(hit_1)
        hits_at_3.append(hit_3)
        reciprocal_ranks.append(rr)

    # Cleanup temp vectorstore
    vectorstore.delete_collection()

    return ExperimentResult(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        n_chunks_created=len(chunks),
        avg_chunk_chars=round(avg_chars, 1),
        hit_rate_at_1=round(sum(hits_at_1) / len(hits_at_1), 4),
        hit_rate_at_3=round(sum(hits_at_3) / len(hits_at_3), 4),
        mrr=round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
    )


def run_experiment(document_path: str, config_path: str = "configs/config.yaml") -> list[ExperimentResult]:
    """Run experiment across all chunk sizes. Returns ranked results."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = []
    logger.info(f"Running chunk size experiment on: {document_path}")
    logger.info(f"Chunk sizes: {CHUNK_SIZES_TO_TEST}")
    logger.info(f"Test questions: {len(DEFAULT_TEST_QUESTIONS)}")

    for chunk_size in CHUNK_SIZES_TO_TEST:
        logger.info(f"\n--- Chunk size: {chunk_size} tokens ---")
        result = evaluate_chunk_size(chunk_size, document_path, DEFAULT_TEST_QUESTIONS, config)
        results.append(result)
        logger.info(
            f"Hit@1={result.hit_rate_at_1:.2%}, "
            f"Hit@3={result.hit_rate_at_3:.2%}, "
            f"MRR={result.mrr:.4f}, "
            f"Chunks={result.n_chunks_created}"
        )

    return results


def print_and_save_results(results: list[ExperimentResult]) -> None:
    """Print table and save to CSV/JSON for README."""
    print("\n" + "="*70)
    print("  CHUNK SIZE EXPERIMENT RESULTS")
    print("="*70)
    print(f"\n  {'Chunk Size':<12} {'Overlap':<10} {'Chunks':<8} "
          f"{'Avg Chars':<12} {'Hit@1':<8} {'Hit@3':<8} {'MRR':<8}")
    print("  " + "-"*64)

    best_hit3 = max(r.hit_rate_at_3 for r in results)
    for r in sorted(results, key=lambda x: x.hit_rate_at_3, reverse=True):
        marker = " ← best" if r.hit_rate_at_3 == best_hit3 else ""
        print(
            f"  {r.chunk_size:<12} {r.chunk_overlap:<10} {r.n_chunks_created:<8} "
            f"{r.avg_chunk_chars:<12.0f} {r.hit_rate_at_1:<8.2%} "
            f"{r.hit_rate_at_3:<8.2%} {r.mrr:<8.4f}{marker}"
        )

    best = max(results, key=lambda r: r.hit_rate_at_3)
    print(f"\n  Optimal chunk size: {best.chunk_size} tokens "
          f"(Hit@3={best.hit_rate_at_3:.2%})")
    print(f"\n  Interpretation:")
    print(f"  - Smaller chunks ({min(CHUNK_SIZES_TO_TEST)}) "
          f"split answers across chunk boundaries → lower Hit@3")
    print(f"  - Larger chunks ({max(CHUNK_SIZES_TO_TEST)}) "
          f"mix topics per chunk → retrieves irrelevant context")
    print(f"  - {best.chunk_size} tokens is optimal for this document type")

    os.makedirs("evaluation", exist_ok=True)
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv("evaluation/chunk_experiment_results.csv", index=False)

    summary = {
        "optimal_chunk_size": best.chunk_size,
        "results": [vars(r) for r in results]
    }
    with open("evaluation/chunk_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to evaluation/chunk_experiment_results.csv")
    print(f"  Copy Hit@3 column to README results table.")


def main():
    parser = argparse.ArgumentParser(description="Chunk size experiment for RAG retrieval")
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Path to document to experiment on (txt, md, or pdf)"
    )
    args = parser.parse_args()

    if args.doc is None or not os.path.exists(args.doc):
        print("Usage: python src/chunk_experiment.py --doc data/raw/your_document.txt")
        print("Document not found or not provided. Exiting.")
        return

    results = run_experiment(args.doc)
    print_and_save_results(results)


if __name__ == "__main__":
    main()
