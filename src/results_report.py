"""Generate before/after improvement report from evaluation snapshots."""

from __future__ import annotations

import json
from pathlib import Path


def pct_change(before: float, after: float, lower_is_better: bool = False) -> float:
    if before == 0:
        return 0.0
    raw = (after - before) / before * 100
    return -raw if lower_is_better else raw


def main():
    before = json.loads(Path("evaluation/results_before.json").read_text())
    after = json.loads(Path("evaluation/results_after.json").read_text())

    metrics = [
        ("hit_at_3", False),
        ("mrr", False),
        ("p50_ms", True),
        ("p95_ms", True),
        ("p99_ms", True),
        ("cache_hit_rate", False),
    ]

    print("Metric,Before,After,Improvement(%)")
    for m, lower_better in metrics:
        b = float(before[m])
        a = float(after[m])
        imp = pct_change(b, a, lower_is_better=lower_better)
        print(f"{m},{b},{a},{imp:.2f}")


if __name__ == "__main__":
    main()
