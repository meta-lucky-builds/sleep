"""Stage 1, second half: thresholds from a null distribution, and the partition of candidates.

The null is CHD between two distractors, neither of which relates to the query: how much the answer moves
when a context's content is swapped for equally irrelevant content. Thresholds are quantiles of it.
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
    """In nats. `source`: "quantile"; "extremes" when the null is tiny; "none" when it is empty."""

    integrate: Optional[float]
    anchor: Optional[float]
    gate: Optional[float]
    source: str
    null_size: int


def thresholds_from_null(values: Sequence[float], cfg: SelectConfig) -> Thresholds:
    """Quantiles when the null is large enough. A tiny null cannot support quantiles, so use its extremes,
    which select only what clearly exceeds or falls below everything seen. An empty null selects nothing."""
    size = len(values)
    if size == 0:
        return Thresholds(None, None, None, "none", 0)
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


def null_distribution(
    handle: ModelHandle,
    candidates: Sequence[Candidate],
    sampler: DistractorSampler,
    select_cfg: SelectConfig,
    signal_cfg: SignalConfig,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> tuple[list[float], bool]:
    """CHD(x; d1, d2) on a seeded subset of candidates. Returns (values, exhausted). Under active banks."""
    if not candidates:
        return [], False
    order = torch.randperm(len(candidates), generator=generator).tolist()
    items: list[tuple[str, Optional[str], Optional[str]]] = []
    for index in order[: select_cfg.n_null]:
        c = candidates[index]
        first = sampler.sample(c.entities, len(c.context), generator, c.episode_ids)
        if first is None:
            continue
        second = sampler.sample(c.entities, len(first.text), generator, tuple(c.episode_ids) + (first.id,))
        if second is None:
            continue
        items.append((c.query, format_context([first]), format_context([second])))
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
    lexical_min: float,
) -> Partition:
    """High CHD: integrate. Low CHD and lexically near its context: anchor. Everything else: discard."""
    out = Partition()
    for candidate, value in zip(candidates, values):
        if not candidate.context:
            out.discarded["no_context"] += 1
        elif candidate.distractor is None:
            out.discarded["no_distractor"] += 1
        elif value is None:
            out.discarded["unscored"] += 1
        elif thresholds.integrate is None or thresholds.anchor is None:
            out.discarded["no_null"] += 1
        elif value > thresholds.integrate:
            out.integrate.append(candidate)
        elif value <= thresholds.anchor:
            if lexical_overlap(candidate.query, candidate.context) >= lexical_min:
                out.anchor.append(candidate)
            else:
                out.discarded["low_chd_far"] += 1
        else:
            out.discarded["mid_chd"] += 1
    return out


def uniform_partition(
    candidates: Sequence[Candidate],
    n_integrate: int,
    n_anchor: int,
    lexical_min: float,
    generator: torch.Generator,
) -> Partition:
    """The same partition without CHD. Anchors are drawn uniformly from the lexically near candidates, then
    integrate queries uniformly from what is left. Anchors go first because only near queries qualify, so
    both counts are met whenever the set allows. Any shortfall shows in the summary."""
    out = Partition()
    usable = [c for c in candidates if c.context]
    if len(usable) < len(candidates):
        out.discarded["no_context"] += len(candidates) - len(usable)
    shuffled = [usable[i] for i in torch.randperm(len(usable), generator=generator).tolist()]
    out.anchor = [c for c in shuffled if lexical_overlap(c.query, c.context) >= lexical_min][:n_anchor]
    chosen = {id(c) for c in out.anchor}
    out.integrate = [c for c in shuffled if id(c) not in chosen][:n_integrate]
    not_drawn = len(usable) - len(out.anchor) - len(out.integrate)
    if not_drawn:
        out.discarded["not_drawn"] += not_drawn
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
