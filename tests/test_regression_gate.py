import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from regression_gate import main


def test_regression_gate_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("evaluation").mkdir()

    Path("configs/config.yaml").write_text(
        "quality_gates:\n  min_hit_at_3: 0.7\n  min_mrr: 0.55\n  max_p95_ms: 2500\n  max_p99_ms: 4000\n"
    )
    Path("evaluation/retrieval_results.json").write_text(json.dumps({"hit_rate_at_3": 0.8, "mrr": 0.6}))
    Path("evaluation/latency_benchmark.json").write_text(json.dumps({"p95_ms": 1000, "p99_ms": 1500}))

    main()
