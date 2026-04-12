"""Regression gate for quality + latency metrics."""

from __future__ import annotations

import json
from pathlib import Path


def fail(msg: str):
    raise SystemExit(msg)


def _load_gates() -> dict:
    # Avoid heavyweight parsing deps in constrained envs.
    cfg = Path("configs/config.yaml")
    gates = {"min_hit_at_3": 0.7, "min_mrr": 0.55, "max_p95_ms": 2500, "max_p99_ms": 4000}
    if not cfg.exists():
        return gates

    current = None
    for raw in cfg.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current = line[:-1]
            continue
        if current == "quality_gates" and ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            try:
                gates[k] = float(v)
            except ValueError:
                pass
    return gates


def main():
    gates = _load_gates()
    retrieval_path = Path("evaluation/retrieval_results.json")
    latency_path = Path("evaluation/latency_benchmark.json")

    if not retrieval_path.exists() or not latency_path.exists():
        fail("Missing evaluation artifacts. Run evaluate.py and benchmark.py first.")

    retrieval = json.loads(retrieval_path.read_text())
    latency = json.loads(latency_path.read_text())

    if retrieval.get("hit_rate_at_3", 0) < gates.get("min_hit_at_3", 0.7):
        fail(f"Hit@3 regression: {retrieval.get('hit_rate_at_3')}")
    if retrieval.get("mrr", 0) < gates.get("min_mrr", 0.55):
        fail(f"MRR regression: {retrieval.get('mrr')}")
    if latency.get("p95_ms", 10**9) > gates.get("max_p95_ms", 2500):
        fail(f"Latency regression p95={latency.get('p95_ms')}")
    if latency.get("p99_ms", 10**9) > gates.get("max_p99_ms", 4000):
        fail(f"Latency regression p99={latency.get('p99_ms')}")

    print("Regression gate passed")


if __name__ == "__main__":
    main()
