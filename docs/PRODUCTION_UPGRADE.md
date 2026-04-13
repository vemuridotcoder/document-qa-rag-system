# Production Upgrade Guide

## Key production upgrades
- **Scalability**: Redis cache backend, Postgres analytics backend, async ingestion jobs with job polling.
- **Reliability**: deterministic extractive fallback when LLM generation fails or retrieval confidence is low.
- **Observability**: request IDs, Prometheus metrics, OpenTelemetry tracing bootstrap.
- **Evaluation**: reusable Hit@K/MRR metric helpers, benchmark tooling with P50/P95/P99.
- **Operational safety**: regression gate script to fail builds on quality/latency regressions.

## Measurable impact targets
- Hit@3 >= **0.70**
- MRR >= **0.55**
- P95 latency <= **2500 ms**
- P99 latency <= **4000 ms**
- Cache hit-rate improves as repeated-query traffic rises (observe `/health` + Prometheus counters).

## Run stack
```bash
docker compose up --build
```

## Run benchmark
```bash
python src/benchmark.py --base-url http://localhost:8001 --questions-file evaluation/benchmark_questions.json
```

## Run retrieval evaluation
```bash
python src/evaluate.py
```

## Run regression gate
```bash
python src/regression_gate.py
```

## Load testing
```bash
locust -f loadtest/locustfile.py --host=http://localhost:8001
```


## Before/after delta report
```bash
python src/results_report.py
```
