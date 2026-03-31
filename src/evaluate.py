"""
evaluate.py — Document Q&A System
=====================================
Measures retrieval quality independently from generation quality.

WHY EVALUATE RETRIEVAL SEPARATELY FROM GENERATION:

If retrieval fails → generation will always fail, regardless of LLM quality.
If retrieval succeeds → generation may still fail (wrong LLM, bad prompt).

Separating these identifies WHERE failures occur:
- Poor Hit Rate → fix chunking strategy or embedding model
- Good Hit Rate, poor answers → fix prompt or LLM
- Both poor → fix retrieval first (it's the bottleneck)

Most RAG tutorials evaluate end-to-end answer quality.
This is wrong — it conflates two separate engineering problems
and makes debugging impossible.

METRICS:

Hit Rate @K:
  Was the answer-containing chunk in the top K retrieved results?
  Binary: 1 if yes, 0 if no. Averaged across all test questions.
  Hit Rate @3 = 0.85 means: for 85% of questions, the correct
  chunk appeared in the top 3 retrieved results.

Mean Reciprocal Rank (MRR):
  If the correct chunk is rank 1: score = 1/1 = 1.0
  If rank 2: score = 1/2 = 0.5
  If rank 3: score = 1/3 = 0.33
  Not in top K: score = 0
  MRR = average across all questions.
  MRR penalizes correct-but-not-first retrieval, which Hit Rate ignores.

Answer Accuracy (manual):
  Manually judged on a subset. Three levels: correct / partial / incorrect.
  Not automated — LLM-as-judge introduces its own biases.
"""

import os
import json
import logging
import yaml
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TestQuestion:
    """
    A single evaluation question with ground truth.

    ground_truth_keywords: list of words that MUST appear in the correct chunk.
    Used for automated Hit Rate evaluation without manual annotation of chunk IDs.

    category: type of question — helps identify which question types fail.
    """
    question: str
    ground_truth_answer: str
    ground_truth_keywords: list[str]
    category: str
    difficulty: str  # easy / medium / hard


@dataclass
class EvaluationResult:
    question: str
    category: str
    retrieved_chunks: list[str]
    top_chunk_distance: float
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    answer_generated: str
    retrieval_confidence: str


class RetrievalEvaluator:
    """
    Evaluates retrieval quality on a test set of questions.

    Usage:
        evaluator = RetrievalEvaluator(vectorstore, embedder, config)
        results = evaluator.run(test_questions)
        evaluator.print_report(results)
    """

    def __init__(self, vectorstore, embedder, config: dict):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.config = config

    def evaluate_question(
        self, question: TestQuestion, n_results: int = 3
    ) -> EvaluationResult:
        """
        Evaluate retrieval for one question.

        Hit detection: checks if any ground_truth_keyword appears in retrieved chunks.
        This is a proxy for "correct chunk retrieved" without requiring exact chunk IDs.
        Limitation: if keywords appear in many chunks, hit rate is overestimated.
        For research-quality evaluation, manual annotation of exact chunk IDs is needed.
        """
        query_embedding = self.embedder.embed_single(question.question)
        retrieved = self.vectorstore.query(query_embedding, n_results=n_results)

        if not retrieved:
            return EvaluationResult(
                question=question.question,
                category=question.category,
                retrieved_chunks=[],
                top_chunk_distance=1.0,
                hit_at_1=False,
                hit_at_3=False,
                reciprocal_rank=0.0,
                answer_generated="No chunks retrieved",
                retrieval_confidence="none",
            )

        # Hit detection: keyword matching as proxy for ground truth
        def chunk_contains_answer(chunk_text: str) -> bool:
            text_lower = chunk_text.lower()
            # Require at least half the keywords to appear
            # (full keyword matching is too strict for paraphrased documents)
            hits = sum(
                1 for kw in question.ground_truth_keywords
                if kw.lower() in text_lower
            )
            return hits >= max(1, len(question.ground_truth_keywords) // 2)

        # Compute Hit@1, Hit@3, MRR
        hit_at_1 = False
        hit_at_3 = False
        reciprocal_rank = 0.0

        for rank, chunk in enumerate(retrieved, start=1):
            if chunk_contains_answer(chunk.text):
                if rank == 1:
                    hit_at_1 = True
                if rank <= 3:
                    hit_at_3 = True
                if reciprocal_rank == 0:
                    reciprocal_rank = 1.0 / rank
                break

        return EvaluationResult(
            question=question.question,
            category=question.category,
            retrieved_chunks=[c.text[:200] + "..." for c in retrieved],
            top_chunk_distance=retrieved[0].distance,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            reciprocal_rank=reciprocal_rank,
            answer_generated="[Run with generator for full answer]",
            retrieval_confidence=retrieved[0].confidence,
        )

    def run(self, test_questions: list[TestQuestion]) -> list[EvaluationResult]:
        """Evaluate all questions. Returns list of results."""
        results = []
        for q in test_questions:
            result = self.evaluate_question(q)
            results.append(result)
            logger.info(
                f"Q: {q.question[:60]}... "
                f"Hit@3: {result.hit_at_3}, RR: {result.reciprocal_rank:.2f}, "
                f"Distance: {result.top_chunk_distance:.3f}"
            )
        return results

    def print_report(self, results: list[EvaluationResult]) -> dict:
        """
        Print evaluation report and return metrics dict.
        This output goes in the README results table.
        """
        if not results:
            print("No results to report.")
            return {}

        hit_at_1 = sum(r.hit_at_1 for r in results) / len(results)
        hit_at_3 = sum(r.hit_at_3 for r in results) / len(results)
        mrr = sum(r.reciprocal_rank for r in results) / len(results)
        avg_distance = sum(r.top_chunk_distance for r in results) / len(results)

        # Per-category breakdown
        categories = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r.hit_at_3)

        print("\n" + "=" * 60)
        print("  RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        print(f"\n  Total questions evaluated : {len(results)}")
        print(f"  Hit Rate @1              : {hit_at_1:.2%}")
        print(f"  Hit Rate @3              : {hit_at_3:.2%}")
        print(f"  Mean Reciprocal Rank     : {mrr:.4f}")
        print(f"  Avg retrieval distance   : {avg_distance:.4f}")

        print("\n  Per-category Hit Rate @3:")
        for cat, hits in categories.items():
            cat_hit_rate = sum(hits) / len(hits)
            print(f"    {cat:<25} {cat_hit_rate:.2%} ({len(hits)} questions)")

        print("\n  Failed retrievals (Hit@3 = False):")
        failures = [r for r in results if not r.hit_at_3]
        for f in failures[:5]:
            print(f"    - {f.question[:70]}...")
            print(f"      Distance: {f.top_chunk_distance:.3f}, Confidence: {f.retrieval_confidence}")

        print("\n" + "=" * 60)
        print("  INTERPRETATION")
        print("=" * 60)
        print(f"""
  Hit Rate @3 = {hit_at_3:.2%}:
  {"GOOD" if hit_at_3 > 0.7 else "NEEDS IMPROVEMENT"} —
  {"The correct chunk is retrieved in {:.0f}% of queries.".format(hit_at_3 * 100)}

  MRR = {mrr:.4f}:
  {"GOOD" if mrr > 0.6 else "NEEDS IMPROVEMENT"} —
  Higher MRR means the correct chunk appears near the top more often.
  A model that always retrieves the answer at rank 1 has MRR = 1.0.

  Key insight: Retrieval failures (not generation failures) are the
  primary bottleneck in RAG systems. Improving chunk size, overlap,
  or embedding model will have more impact than changing the LLM.
        """)

        metrics = {
            "total_questions": len(results),
            "hit_rate_at_1": round(hit_at_1, 4),
            "hit_rate_at_3": round(hit_at_3, 4),
            "mrr": round(mrr, 4),
            "avg_distance": round(avg_distance, 4),
            "per_category": {
                cat: round(sum(hits) / len(hits), 4)
                for cat, hits in categories.items()
            }
        }

        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/retrieval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Results saved to evaluation/retrieval_results.json")

        return metrics


def load_test_questions(path: str = "evaluation/test_questions.json") -> list[TestQuestion]:
    """Load test questions from JSON file."""
    if not os.path.exists(path):
        logger.warning(f"Test questions not found at {path}. Using sample questions.")
        return get_sample_questions()

    with open(path) as f:
        data = json.load(f)
    return [TestQuestion(**q) for q in data]


def get_sample_questions() -> list[TestQuestion]:
    """
    Sample test questions for demonstration.
    Replace with domain-specific questions for your indexed document.
    Add at least 20 questions covering all question categories for meaningful evaluation.
    """
    return [
        TestQuestion(
            question="What is the main topic of the document?",
            ground_truth_answer="Depends on indexed document",
            ground_truth_keywords=["introduction", "overview", "summary"],
            category="general",
            difficulty="easy"
        ),
        TestQuestion(
            question="What are the key conclusions or findings?",
            ground_truth_answer="Depends on indexed document",
            ground_truth_keywords=["conclusion", "finding", "result"],
            category="summary",
            difficulty="medium"
        ),
        TestQuestion(
            question="What methodology or approach was used?",
            ground_truth_answer="Depends on indexed document",
            ground_truth_keywords=["method", "approach", "procedure"],
            category="methodology",
            difficulty="medium"
        ),
    ]


def save_test_questions(questions: list[TestQuestion], path: str = "evaluation/test_questions.json"):
    """Save test questions to JSON for reuse."""
    os.makedirs("evaluation", exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(q) for q in questions], f, indent=2)
    logger.info(f"Saved {len(questions)} test questions to {path}")
