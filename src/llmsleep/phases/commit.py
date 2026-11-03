"""Stage 5: audit on held-out queries, accept or roll back, retirement.

The gap is CHD(x; e, d) on the current weights: it falls as the student absorbs the episode, because the
distractor-context answer stops differing. Each item has its own seed, so before and after use common random
numbers regardless of which items were measured, which cuts the noise in the paired comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import CommitConfig, make_generator
from ..handle import ModelHandle
from ..ledger import LedgerExhausted
from ..memory import EpisodicStore
from ..signals import chd
from .consolidate import TokenTargets, reduced_kl


@dataclass
class AuditItem:
    """A held-out query with its fixed contexts. JSON-able, kept in episode meta across nights."""

    query: str
    context: str
    distractor: str
    episode_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        record = asdict(self)
        record["episode_ids"] = list(self.episode_ids)
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "AuditItem":
        return cls(record["query"], record["context"], record["distractor"], tuple(record["episode_ids"]))


@dataclass
class Decision:
    """The night's verdict, with reasons. Gaps are means over paired items; `None` means not measured."""

    accepted: bool
    reasons: list[str] = field(default_factory=list)
    gap_before: Optional[float] = None
    gap_after: Optional[float] = None
    anchor_drift: Optional[float] = None
    paired_items: int = 0
    audit_exhausted: bool = False


def measure_gaps(
    handle: ModelHandle,
    items: Sequence[AuditItem],
    world_seed: int,
    night: int,
    k: int,
    max_new_tokens: int,
) -> tuple[list[Optional[float]], bool]:
    """Gap per item under the active banks. Returns (values, exhausted); unmeasured items are None."""
    values: list[Optional[float]] = [None] * len(items)
    for index, item in enumerate(items):
        generator = make_generator(world_seed, night, f"audit|{item.query}|{item.context}|{item.distractor}")
        try:
            result = chd(handle, [item.query], [item.context], [item.distractor], k, max_new_tokens, generator)
        except LedgerExhausted:
            return values, True
        values[index] = float(result.values[0])
    return values, False


def measure_drift(handle: ModelHandle, targets: Sequence[TokenTargets]) -> tuple[Optional[float], bool]:
    """Mean reduced KL per answer token, stored anchor target to current model. (None, _) if nothing measured."""
    total, count = 0.0, 0
    prompts = [t.prompt_ids for t in targets]
    conts = [t.cont_ids for t in targets]
    try:
        for rows, logits in handle.iter_answer_logits(prompts, conts):
            log_probs = F.log_softmax(logits, dim=-1)
            device = log_probs.device
            kl = reduced_kl(
                log_probs,
                torch.cat([targets[r].top_idx for r in rows]).to(device),
                torch.cat([targets[r].top_logp for r in rows]).to(device),
                torch.cat([targets[r].res_logp for r in rows]).to(device),
            )
            total += float(kl.sum())
            count += kl.numel()
    except LedgerExhausted:
        return (total / count if count else None), True
    return (total / count if count else None), False


def decide(
    gaps_before: Sequence[Optional[float]],
    gaps_after: Sequence[Optional[float]],
    drift: Optional[float],
    cfg: CommitConfig,
    transactional: bool,
    audit_exhausted: bool = False,
) -> Decision:
    """Pure decision rule. Compares only items measured both before and after.

    Fails closed: without a single paired item the night cannot show the gap fell, so it is rejected.
    A partial audit decides on what it has. `transactional=False` always accepts, keeping the findings.
    """
    pairs = [(b, a) for b, a in zip(gaps_before, gaps_after) if b is not None and a is not None]
    decision = Decision(accepted=True, anchor_drift=drift, paired_items=len(pairs), audit_exhausted=audit_exhausted)
    findings: list[str] = []
    if not pairs:
        findings.append("no_audit_data")
    else:
        decision.gap_before = sum(b for b, _ in pairs) / len(pairs)
        decision.gap_after = sum(a for _, a in pairs) / len(pairs)
        if not decision.gap_after < decision.gap_before - cfg.min_gap_drop:
            findings.append("gap_did_not_fall")
    if drift is not None and drift > cfg.max_anchor_drift:
        findings.append("anchor_drift")
    if transactional:
        decision.accepted = not findings
        decision.reasons = findings or ["passed"]
    else:
        decision.reasons = ["non_transactional"] + findings
    return decision


def update_episodes(
    store: EpisodicStore,
    items: Sequence[AuditItem],
    gaps: Sequence[Optional[float]],
    retire_threshold: Optional[float],
    count_toward_retirement: bool,
    retire_nights: int,
) -> list[str]:
    """Record each episode's gap on the weights now in place. Returns the ids retired tonight.

    An episode retires after `retire_nights` consecutive accepted nights with its gap at or below the
    threshold. A rejected night neither counts toward retirement nor resets the count.
    """
    by_episode: dict[str, list[float]] = {}
    for item, gap in zip(items, gaps):
        if gap is None:
            continue
        for episode_id in item.episode_ids:
            by_episode.setdefault(episode_id, []).append(gap)
    retired = []
    for episode_id, values in by_episode.items():
        if episode_id not in store:
            continue
        episode = store.get(episode_id)
        gap = sum(values) / len(values)
        episode.meta["last_gap"] = gap
        if not count_toward_retirement or retire_threshold is None:
            continue
        low_nights = episode.meta.get("low_gap_nights", 0) + 1 if gap <= retire_threshold else 0
        episode.meta["low_gap_nights"] = low_nights
        if low_nights >= retire_nights and episode.status == "active":
            store.retire(episode_id)
            retired.append(episode_id)
    return retired
