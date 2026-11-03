"""Paired hierarchical bootstrap, Holm correction, AUROC. Pure numpy (PREREG decision rule).

Per-seed data is a mapping cluster -> (numerator, denominator). A metric is the ratio of sums, which handles
plain accuracies (correct, count) and the mean-of-kinds `acq_implied` (weighted) with one estimator.

A caveat: with 5 seeds the seed level of a percentile bootstrap has few distinct resamples, and its intervals
tend to be too narrow. The PREREG rule specifies it, so it is what we use; the seed count is always reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

Clusters = Mapping[str, Sequence[float]]  # cluster -> (numerator, denominator)
BySeed = Mapping[int, Clusters]


@dataclass(frozen=True)
class BootstrapResult:
    """`estimate` is the mean over seeds of the within-seed difference (a - b)."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float  # Two-sided, from the bootstrap distribution's mass on each side of `null`.
    n_seeds: int
    n_clusters: int
    null: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    """Element-wise num/den, nan where the denominator is zero."""
    out = np.full(np.broadcast(numerators, denominators).shape, np.nan, dtype=np.float64)
    positive = denominators > 0
    out[positive] = numerators[positive] / denominators[positive]
    return out


def _percentile_result(
    point: float, draws: np.ndarray, ci: float, null: float, n_seeds: int, n_clusters: int
) -> BootstrapResult:
    valid = draws[np.isfinite(draws)]
    if valid.size == 0:
        return BootstrapResult(point, float("nan"), float("nan"), float("nan"), n_seeds, n_clusters, null)
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(valid, [tail, 1.0 - tail])
    below = float(np.mean(valid <= null))
    above = float(np.mean(valid >= null))
    p_value = min(1.0, max(2.0 * min(below, above), 1.0 / valid.size))
    return BootstrapResult(point, float(low), float(high), p_value, n_seeds, n_clusters, null)


def paired_cluster_bootstrap(
    a: BySeed,
    b: BySeed,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    null: float = 0.0,
) -> BootstrapResult:
    """CI of the mean paired difference: seeds resampled, and clusters resampled within each seed.

    Within a seed the same clusters are drawn for both arms, which is what makes it paired. Only seeds and
    clusters present in both arms enter. Each seed slot of a resample gets its own within-seed draw, so a
    seed drawn twice does not count as one replicate twice.
    """
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise ValueError("no seed is common to both arms")
    rng = np.random.default_rng(seed)
    points: list[float] = []
    per_seed: list[list[np.ndarray]] = []  # per_seed[s][slot] has shape (n_boot,).
    total_clusters = 0
    for s in seeds:
        clusters = sorted(set(a[s]) & set(b[s]))
        total_clusters += len(clusters)
        if not clusters:
            points.append(float("nan"))
            per_seed.append([np.full(n_boot, np.nan) for _ in seeds])
            continue
        num_a = np.array([a[s][c][0] for c in clusters], dtype=np.float64)
        den_a = np.array([a[s][c][1] for c in clusters], dtype=np.float64)
        num_b = np.array([b[s][c][0] for c in clusters], dtype=np.float64)
        den_b = np.array([b[s][c][1] for c in clusters], dtype=np.float64)
        points.append(float(_ratio(num_a.sum(keepdims=True), den_a.sum(keepdims=True))[0]
                            - _ratio(num_b.sum(keepdims=True), den_b.sum(keepdims=True))[0]))
        size = len(clusters)
        slots = []
        for _ in seeds:
            counts = rng.multinomial(size, np.full(size, 1.0 / size), size=n_boot).astype(np.float64)
            slots.append(_ratio(counts @ num_a, counts @ den_a) - _ratio(counts @ num_b, counts @ den_b))
        per_seed.append(slots)
    point = float(np.nanmean(points)) if np.isfinite(points).any() else float("nan")
    n = len(seeds)
    draws = np.empty(n_boot, dtype=np.float64)
    choice = rng.integers(0, n, size=(n_boot, n))
    for index in range(n_boot):
        values = [per_seed[choice[index, slot]][slot][index] for slot in range(n)]
        finite = [v for v in values if np.isfinite(v)]
        draws[index] = np.mean(finite) if finite else np.nan
    return _percentile_result(point, draws, ci, null, n, total_clusters)


def seed_bootstrap(
    values: Mapping[int, float], n_boot: int = 2000, ci: float = 0.95, seed: int = 0, null: float = 0.0
) -> BootstrapResult:
    """CI of a mean over seeds of one per-seed number, such as a slope of a gap over nights."""
    keys = sorted(k for k, v in values.items() if v is not None and np.isfinite(v))
    if not keys:
        raise ValueError("no finite per-seed value")
    data = np.array([values[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(n_boot, len(data)))].mean(axis=1)
    return _percentile_result(float(data.mean()), draws, ci, null, len(data), len(data))


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope; nan with fewer than two distinct x."""
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan")
    centred = x - x.mean()
    return float((centred * (y - y.mean())).sum() / (centred**2).sum())


def holm(p_values: Mapping[str, Optional[float]]) -> dict[str, Optional[float]]:
    """Holm step-down adjusted p-values. A missing or nan p-value stays None and uses no slot in the family."""
    usable = sorted(
        ((k, float(v)) for k, v in p_values.items() if v is not None and np.isfinite(v)),
        key=lambda item: (item[1], item[0]),
    )
    size = len(usable)
    out: dict[str, Optional[float]] = {k: None for k in p_values}
    running = 0.0
    for rank, (key, value) in enumerate(usable):
        running = max(running, min(1.0, (size - rank) * value))
        out[key] = running
    return out


def auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve: P(score of a positive > score of a negative), ties at one half.

    None when a class is empty.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    start = 0
    while start < s.size:
        stop = start
        while stop + 1 < s.size and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def supported_difference(result: BootstrapResult, direction: int, margin: float) -> bool:
    """PREREG: the CI excludes zero in the predicted direction, and the point estimate meets the margin."""
    if not np.isfinite(result.ci_low) or not np.isfinite(result.ci_high):
        return False
    if direction > 0:
        return result.ci_low > 0 and result.estimate >= margin
    return result.ci_high < 0 and -result.estimate >= margin


def non_inferior(result: BootstrapResult, margin: float) -> bool:
    """The whole CI of (a - b) lies above -margin: a is not worse than b by the margin or more."""
    return bool(np.isfinite(result.ci_low)) and result.ci_low > -margin
