"""Stage 1, second half: thresholds from a null distribution, and the partition of candidates.

The null is CHD between two distractors, neither of which relates to the query: how much the answer moves
when a context's content is swapped for equally irrelevant content. Thresholds are quantiles of it, in nats per
answer token, since a sum over tokens would make a long hallucination outrank a short confident answer. Pairs
have their own null, because two-episode contexts move answers differently from one-episode ones.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from ..config import SelectConfig, SignalConfig
from ..handle import ModelHandle
from ..memory import DistractorSampler, Episode, format_context, tokenize
from ..signals import SignalCache, chd_safe
from .queries import Candidate

_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have in is it its of on or that the their this to "
    "was were what when where which who whom whose why how with".split()
)


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of the query's content words that appear in `text`."""
    query_words = set(tokenize(query)) - _STOPWORDS
    if not query_words:
        return 0.0
    return len(query_words & set(tokenize(text))) / len(query_words)


@dataclass
class Thresholds:
    """In nats per answer token. `source`: "quantile"; "extremes" when the null is tiny; "none" when empty."""

    integrate: Optional[float]
    anchor: Optional[float]
    gate: Optional[float]
    source: str
    null_size: int


NO_THRESHOLDS = Thresholds(None, None, None, "none", 0)


def thresholds_from_null(values: Sequence[float], cfg: SelectConfig) -> Thresholds:
    """Quantiles when the null is large enough. A tiny null cannot support quantiles, so use its extremes,
    which select only what clearly exceeds or falls below everything seen. An empty null selects nothing."""
    size = len(values)
    if size == 0:
        return NO_THRESHOLDS
    null = torch.tensor(list(values), dtype=torch.float64)
    if size < cfg.min_null:
        low, high = float(null.min()), float(null.max())
        return Thresholds(high, low, low, "extremes", size)
    return Thresholds(
        float(torch.quantile(null, cfg.integrate_null_q)),
        float(torch.quantile(null, cfg.anchor_null_q)),
        float(torch.quantile(null, cfg.gate_null_q)),
        "quantile",
        size,
    )


def draw_distractor(
    candidate: Candidate, sampler: DistractorSampler, generator: torch.Generator, exclude: Sequence[str] = ()
) -> tuple[Optional[str], tuple[str, ...]]:
    """A context unrelated to the candidate, shaped like its own: two episodes for a pair. (text, ids used)."""
    count = 2 if candidate.is_pair else 1
    target = len(candidate.context) // count
    used = list(exclude) + list(candidate.episode_ids)
    picks: list[Episode] = []
    for _ in range(count):
        episode = sampler.sample(candidate.entities, target, generator, used)
        if episode is None:
            return None, ()
        picks.append(episode)
        used.append(episode.id)
    return format_context(picks), tuple(e.id for e in picks)


def null_distribution(
    handle: ModelHandle,
    candidates: Sequence[Candidate],
    sampler: DistractorSampler,
    signal_cfg: SignalConfig,
    n: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> tuple[list[float], bool]:
    """Per-token CHD(x; d1, d2) on a seeded subset of up to `n` candidates, under active banks. (values, exhausted)."""
    if not candidates:
        return [], False
    order = torch.randperm(len(candidates), generator=generator).tolist()
    items: list[tuple[str, Optional[str], Optional[str]]] = []
    for index in order[:n]:
        candidate = candidates[index]
        first, first_ids = draw_distractor(candidate, sampler, generator)
        if first is None:
            continue
        second, _ = draw_distractor(candidate, sampler, generator, first_ids)
        if second is None:
            continue
        items.append((candidate.query, first, second))
    values, exhausted = chd_safe(
        handle, items, signal_cfg.k, signal_cfg.max_new_tokens, generator, cache, signal_cfg.chunk
    )
    return [v for v in values if v is not None], exhausted


@dataclass
class Partition:
    """Where each candidate went. `discarded` counts by reason, and is always reported."""

    integrate: list[Candidate] = field(default_factory=list)
    anchor: list[Candidate] = field(default_factory=list)
    discarded: Counter = field(default_factory=Counter)

    def summary(self) -> dict:
        return {
            "integrate": len(self.integrate),
            "anchor": len(self.anchor),
            "discarded": dict(self.discarded),
        }


def partition(
    candidates: Sequence[Candidate],
    values: Sequence[Optional[float]],
    thresholds: Thresholds,
    pair_thresholds: Optional[Thresholds] = None,
) -> Partition:
    """Tonight's queries integrate on high CHD. Pool queries (which name nothing of tonight's) anchor on low CHD.

    A query about tonight's episode never anchors, whatever its CHD: a germline target there would pull
    against the hard labels for the same fact, and learning it would read as drift.
    """
    out = Partition()
    for candidate, value in zip(candidates, values):
        limits = pair_thresholds if candidate.is_pair and pair_thresholds is not None else thresholds
        if not candidate.context:
            out.discarded["no_context"] += 1
        elif candidate.distractor is None:
            out.discarded["no_distractor"] += 1
        elif value is None:
            out.discarded["unscored"] += 1
        elif limits.integrate is None or limits.anchor is None:
            out.discarded["no_null"] += 1
        elif candidate.pool:
            if value <= limits.anchor:
                out.anchor.append(candidate)
            else:
                out.discarded["pool_not_low"] += 1
        elif value > limits.integrate:
            out.integrate.append(candidate)
        else:
            out.discarded["low_or_mid_chd"] += 1
    return out


def uniform_partition(
    candidates: Sequence[Candidate],
    pool: Sequence[Candidate],
    n_anchor: int,
    generator: torch.Generator,
) -> Partition:
    """The partition without CHD: all usable queries integrate, and anchors are a uniform draw from the pool."""
    out = Partition()
    out.integrate = [c for c in candidates if c.context]
    if len(out.integrate) < len(candidates):
        out.discarded["no_context"] += len(candidates) - len(out.integrate)
    order = torch.randperm(len(pool), generator=generator).tolist()
    out.anchor = [pool[i] for i in order[:n_anchor]]
    if len(pool) > len(out.anchor):
        out.discarded["pool_not_drawn"] += len(pool) - len(out.anchor)
    return out


def touching_episodes(
    query: str, episodes: Sequence[Episode], lexical_min: float, limit: int = 2
) -> list[Episode]:
    """The episodes a query is about: it names an entity of theirs, or overlaps their text enough."""
    lowered = query.casefold()
    scored = []
    for episode in episodes:
        overlap = lexical_overlap(query, episode.text)
        names_entity = any(entity.casefold() in lowered for entity in episode.entities)
        if names_entity or overlap >= lexical_min:
            scored.append((not names_entity, -overlap, episode.id, episode))
    scored.sort(key=lambda row: row[:3])
    return [row[3] for row in scored[:limit]]
