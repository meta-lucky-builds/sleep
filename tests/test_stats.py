"""The decision-rule statistics, on data small enough to check by hand."""
from __future__ import annotations

import math

import pytest

from llmsleep.eval.metrics import Tally
from llmsleep.stats import (
    BootstrapResult,
    auroc,
    binomial_tail,
    holm,
    non_inferior,
    paired_cluster_bootstrap,
    seed_bootstrap,
    slope,
    supported_difference,
)


def _clusters(values):
    return {f"c{i}": [float(v), 1.0] for i, v in enumerate(values)}


def test_auroc_known_values_and_ties():
    assert auroc([1, 1, 0, 0], [3, 4, 1, 2]) == pytest.approx(1.0)
    assert auroc([1, 1, 0, 0], [1, 2, 3, 4]) == pytest.approx(0.0)
    assert auroc([1, 0], [5, 5]) == pytest.approx(0.5)
    # Pairs: (2>1) 1, (2=2) 0.5, (3>1) 1, (3>2) 1 of 4.
    assert auroc([1, 1, 0, 0], [2, 3, 1, 2]) == pytest.approx(3.5 / 4)
    assert auroc([1, 1], [1, 2]) is None
    assert auroc([], []) is None


def test_holm_worked_example():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # Monotone, not 0.04.


def test_holm_skips_missing_values_without_using_a_slot():
    adjusted = holm({"a": 0.01, "b": None, "c": float("nan")})
    assert adjusted == {"a": pytest.approx(0.01), "b": None, "c": None}


def test_bootstrap_of_identical_arms_is_exactly_zero():
    data = {s: _clusters([0, 1, 1, 0, 1]) for s in range(5)}
    result = paired_cluster_bootstrap(data, data, n_boot=200)
    assert result.estimate == 0.0 and result.ci_low == 0.0 and result.ci_high == 0.0
    assert result.n_seeds == 5 and result.n_clusters == 25


def test_bootstrap_of_all_right_versus_all_wrong():
    a = {s: _clusters([1] * 6) for s in range(5)}
    b = {s: _clusters([0] * 6) for s in range(5)}
    result = paired_cluster_bootstrap(a, b, n_boot=200)
    assert result.estimate == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(1.0) and result.ci_high == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1 / 200)


def test_bootstrap_is_paired_and_seeded():
    a = {0: _clusters([1, 0, 1, 0, 1, 1]), 1: _clusters([0, 0, 1, 1, 1, 0])}
    b = {0: _clusters([0, 0, 1, 0, 1, 0]), 1: _clusters([0, 0, 0, 1, 1, 0])}
    first = paired_cluster_bootstrap(a, b, n_boot=300, seed=4)
    assert first == paired_cluster_bootstrap(a, b, n_boot=300, seed=4)
    # Point: seed 0 is 4/6 - 2/6, seed 1 is 3/6 - 2/6.
    assert first.estimate == pytest.approx((2 / 6 + 1 / 6) / 2)
    # Every paired difference is non-negative, so no resample can go below zero.
    assert first.ci_low >= 0.0


def test_bootstrap_uses_only_common_seeds_and_clusters():
    a = {0: {"x": [1.0, 1.0], "y": [1.0, 1.0]}, 1: {"x": [1.0, 1.0]}}
    b = {0: {"x": [0.0, 1.0], "z": [0.0, 1.0]}, 2: {"x": [0.0, 1.0]}}
    result = paired_cluster_bootstrap(a, b, n_boot=50)
    assert result.n_seeds == 1 and result.n_clusters == 1 and result.estimate == pytest.approx(1.0)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap({1: {}}, {2: {}}, n_boot=10)


def test_ratio_estimator_reproduces_a_mean_of_kinds():
    """acq_implied: three kinds of different sizes must weigh equally."""
    tally = Tally()
    for i in range(10):
        tally.add(f"r{i}", 1.0, 1 / 10)  # Reverse, all right.
    for i in range(2):
        tally.add(f"a{i}", 0.0, 1 / 2)  # Alias, all wrong.
    assert tally.rate() == pytest.approx(0.5)


def test_decision_helpers_need_ci_and_margin():
    big = BootstrapResult(0.10, 0.04, 0.16, 0.01, 5, 50)
    small = BootstrapResult(0.02, 0.01, 0.03, 0.01, 5, 50)
    across = BootstrapResult(0.10, -0.01, 0.2, 0.2, 5, 50)
    lower = BootstrapResult(-0.10, -0.16, -0.04, 0.01, 5, 50)
    assert supported_difference(big, +1, 0.03)
    assert not supported_difference(small, +1, 0.03)  # CI excludes zero, but below the margin.
    assert not supported_difference(across, +1, 0.03)  # Big estimate, CI includes zero.
    assert not supported_difference(big, -1, 0.03)  # Wrong direction.
    assert supported_difference(lower, -1, 0.03)
    assert non_inferior(BootstrapResult(-0.01, -0.02, 0.0, 0.5, 5, 50), 0.03)
    assert not non_inferior(BootstrapResult(-0.01, -0.04, 0.02, 0.5, 5, 50), 0.03)
    assert not supported_difference(BootstrapResult(0.1, math.nan, math.nan, math.nan, 5, 50), +1, 0.03)


def test_binomial_tail_known_values():
    assert binomial_tail(0, 10, 0.5) == 1.0
    assert binomial_tail(10, 10, 0.5) == pytest.approx(1 / 1024)
    assert binomial_tail(1, 2, 0.5) == pytest.approx(0.75)
    # One lucky year in 16 probes against 1/50: nowhere near p < 0.01 (review H3).
    assert binomial_tail(1, 16, 1 / 50) > 0.2


def test_seed_bootstrap_and_slope():
    assert slope([0, 1, 2], [1, 3, 5]) == pytest.approx(2.0)
    assert math.isnan(slope([1], [1]))
    result = seed_bootstrap({0: 1.0, 1: 1.0, 2: 1.0}, n_boot=50)
    assert result.estimate == pytest.approx(1.0) and result.ci_low == pytest.approx(1.0)
    with pytest.raises(ValueError):
        seed_bootstrap({0: float("nan")}, n_boot=10)
