"""Latency benchmarking utility for RAG endpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from quality import percentile


def benchmark_http(base_url: str, questions: list[str], n_chunks: int = 3) -> dict:
    latencies = []
    errors = 0
    for q in questions:
        start = time.perf_counter()
        try:
            r = requests.post(f"{base_url}/ask", json={"question": q, "n_chunks": n_chunks}, timeout=120)
            r.raise_for_status()
        except Exception:
            errors += 1
        finally:
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "samples": len(latencies),
        "errors": errors,
        "error_rate": round(errors / max(len(latencies), 1), 4),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "min_ms": round(min(latencies), 2) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--questions-file", default="evaluation/benchmark_questions.json")
    parser.add_argument("--n-chunks", type=int, default=3)
    args = parser.parse_args()

    q_path = Path(args.questions_file)
    questions = json.loads(q_path.read_text()) if q_path.exists() else ["What is this document about?"]

    report = benchmark_http(args.base_url, questions, args.n_chunks)
    Path("evaluation").mkdir(exist_ok=True)
    out = Path("evaluation/latency_benchmark.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
