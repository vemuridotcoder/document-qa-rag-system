import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from quality import hit_rate_at_k, mean_reciprocal_rank, percentile


def test_hit_rate_at_k():
    ranks = [1, 2, None, 4]
    assert hit_rate_at_k(ranks, 3) == 0.5


def test_mean_reciprocal_rank():
    ranks = [1, 2, None]
    assert round(mean_reciprocal_rank(ranks), 4) == round((1 + 0.5 + 0) / 3, 4)


def test_percentile_regression():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == 30
    assert percentile(vals, 95) == 48
