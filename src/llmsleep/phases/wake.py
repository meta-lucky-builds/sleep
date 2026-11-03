"""Day: the write gate. An episode whose own queries barely depend on it is already known; don't store it."""
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
    stored: bool
    reason: str
    mean_chd: Optional[float] = None


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
    """Store `episode` unless CHD on `probes` (query, distractor context) is at or below `threshold`.

    Fails open: if the gate cannot judge (no probes, no threshold, no budget), the episode is stored, since
    losing a real episode is worse than storing a known one. A superseding episode always passes: low CHD
    on it cannot mean "already known" when it contradicts what the model believes it knows from elsewhere.
    Runs under the active banks.
    """
    if bypass:
        store.add(episode)
        return WakeResult(episode.id, True, "bypass")
    if episode.supersedes:
        store.add(episode)
        return WakeResult(episode.id, True, "supersession")
    if not probes or threshold is None:
        store.add(episode)
        return WakeResult(episode.id, True, "unscored")
    context = format_context([episode])
    items = [(query, context, distractor) for query, distractor in probes]
    values, _ = chd_safe(handle, items, k, max_new_tokens, generator, cache, chunk)
    scored = [v for v in values if v is not None]
    if not scored:
        store.add(episode)
        return WakeResult(episode.id, True, "unscored")
    mean_chd = sum(scored) / len(scored)
    if mean_chd <= threshold:
        return WakeResult(episode.id, False, "already_known", mean_chd)
    store.add(episode)
    return WakeResult(episode.id, True, "novel", mean_chd)
