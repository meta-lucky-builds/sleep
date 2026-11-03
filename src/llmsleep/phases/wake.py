"""Day: the write gate. An episode whose own queries barely depend on it looks known already: defer it.

Deferred, not dropped (A1.12): the episode is stored without active retrieval and is still replayed, so a false
gate decision costs retrieval and tonight's teacher targets, and never loses a real fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..handle import ModelHandle
from ..memory import Episode, EpisodicStore, format_context
from ..signals import SignalCache, chd_safe


@dataclass
class WakeResult:
    episode_id: str
    deferred: bool
    reason: str
    mean_chd: Optional[float] = None
    stored: bool = True  # Every real episode is stored now; kept so reports read the same across arms.


def observe(
    handle: ModelHandle,
    store: EpisodicStore,
    episode: Episode,
    probes: Sequence[tuple[str, str]],
    threshold: Optional[float],
    k: int,
    max_new_tokens: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
    chunk: int = 4,
    bypass: bool = False,
) -> WakeResult:
    """Store `episode`, deferring it if per-token CHD on `probes` (query, distractor context) is <= `threshold`.

    If the gate cannot judge (no probes, no threshold, no budget), the episode stays active. A superseding episode
    always does: low CHD on it cannot mean "known" when it contradicts what the model holds. Runs under active banks.
    """
    store.add(episode)
    if bypass:
        return WakeResult(episode.id, False, "bypass")
    if episode.supersedes:
        return WakeResult(episode.id, False, "supersession")
    if not probes or threshold is None:
        return WakeResult(episode.id, False, "unscored")
    context = format_context([episode])
    items = [(query, context, distractor) for query, distractor in probes]
    values, _ = chd_safe(handle, items, k, max_new_tokens, generator, cache, chunk)
    scored = [v for v in values if v is not None]
    if not scored:
        return WakeResult(episode.id, False, "unscored")
    mean_chd = sum(scored) / len(scored)
    if mean_chd <= threshold:
        store.defer(episode.id)
        return WakeResult(episode.id, True, "already_known", mean_chd)
    return WakeResult(episode.id, False, "novel", mean_chd)
