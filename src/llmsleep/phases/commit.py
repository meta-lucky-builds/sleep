"""Stage 5: audit on held-out queries, accept or roll back, retirement.

The gap is per-token CHD(x; e, d) on the current weights: it falls as the student absorbs the episode, because
the distractor-context answer stops differing. Each item has its own seed, so before and after use common random
numbers regardless of which items were measured, which cuts the noise in the paired comparison.

Three things keep the test honest (Amendment 1, A1.5 and A1.6):
- Items are flagged with an independent measurement, under another seed. Flagging on the "before" value itself
  would select noisy highs, and the mean would then fall by regression alone.
- The gap must fall by more than it falls on a null-control set: the same queries between two unrelated contexts,
  which tracks how much the night moved answers in general.
- Anchor drift is the increase over the night, on identical targets. An absolute level would include all earlier
  nights' deltas, and once over the limit no later night could pass.
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
    """The night's verdict, with reasons. Gaps are means over paired items; `None` means not measured.

    `anchor_drift` is the increase over the night, in nats per answer token.
    """

    accepted: bool
    reasons: list[str] = field(default_factory=list)
    gap_before: Optional[float] = None
    gap_after: Optional[float] = None
    control_drop: Optional[float] = None
    anchor_drift: Optional[float] = None
    anchor_drift_before: Optional[float] = None
    anchor_drift_after: Optional[float] = None
    paired_items: int = 0
    audit_exhausted: bool = False


def measure_gaps(
    handle: ModelHandle,
    items: Sequence[AuditItem],
    world_seed: int,
    night: int,
    k: int,
    max_new_tokens: int,
    purpose: str = "audit",
) -> tuple[list[Optional[float]], bool]:
    """Per-token gap per item under the active banks. (values, exhausted); unmeasured items are None.

    Call with another `purpose` for a measurement independent of the first.
    """
    values: list[Optional[float]] = [None] * len(items)
    for index, item in enumerate(items):
        generator = make_generator(world_seed, night, f"{purpose}|{item.query}|{item.context}|{item.distractor}")
        try:
            result = chd(handle, [item.query], [item.context], [item.distractor], k, max_new_tokens, generator)
        except LedgerExhausted:
            return values, True
        values[index] = float(result.per_token[0])
    return values, False


def drift_by_target(
    handle: ModelHandle, targets: Sequence[TokenTargets]
) -> tuple[list[Optional[tuple[float, int]]], bool]:
    """(sum of reduced KL, tokens) per target, under the active banks. Unmeasured targets are None."""
    out: list[Optional[tuple[float, int]]] = [None] * len(targets)
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
            cursor = 0
            for row in rows:
                length = len(targets[row].cont_ids)
                out[row] = (float(kl[cursor : cursor + length].sum()), length)
                cursor += length
    except LedgerExhausted:
        return out, True
    return out, False


def _mean_drift(values: Sequence[tuple[float, int]]) -> Optional[float]:
    tokens = sum(n for _, n in values)
    return sum(s for s, _ in values) / tokens if tokens else None


def measure_drift(handle: ModelHandle, targets: Sequence[TokenTargets]) -> tuple[Optional[float], bool]:
    """Mean reduced KL per answer token, stored target to current model. (None, _) if nothing was measured."""
    values, exhausted = drift_by_target(handle, targets)
    return _mean_drift([v for v in values if v is not None]), exhausted


def drift_change(
    before: Sequence[Optional[tuple[float, int]]], after: Sequence[Optional[tuple[float, int]]]
) -> tuple[Optional[float], Optional[float]]:
    """(mean before, mean after), over only the targets measured at both times."""
    paired = [(b, a) for b, a in zip(before, after) if b is not None and a is not None]
    return _mean_drift([b for b, _ in paired]), _mean_drift([a for _, a in paired])


def _pairs(before: Sequence[Optional[float]], after: Sequence[Optional[float]]) -> list[tuple[float, float]]:
    return [(b, a) for b, a in zip(before, after) if b is not None and a is not None]


def decide(
    gaps_before: Sequence[Optional[float]],
    gaps_after: Sequence[Optional[float]],
    drift: Optional[float],
    cfg: CommitConfig,
    transactional: bool,
    audit_exhausted: bool = False,
    control_before: Optional[Sequence[Optional[float]]] = None,
    control_after: Optional[Sequence[Optional[float]]] = None,
) -> Decision:
    """Pure decision rule. `drift` is the increase on identical targets. Compares only items measured twice.

    Fails closed: without a paired item the night cannot show the gap fell, so it is rejected. With a control set
    supplied, the same applies to it, and the gap must fall by `min_gap_drop` more than the control does.
    A partial audit decides on what it has. `transactional=False` always accepts, keeping the findings.
    """
    pairs = _pairs(gaps_before, gaps_after)
    decision = Decision(accepted=True, anchor_drift=drift, paired_items=len(pairs), audit_exhausted=audit_exhausted)
    findings: list[str] = []
    baseline = 0.0
    if control_before is not None and control_after is not None:
        control = _pairs(control_before, control_after)
        if control:
            baseline = sum(b - a for b, a in control) / len(control)
            decision.control_drop = baseline
        else:
            findings.append("no_control_data")
    if not pairs:
        findings.insert(0, "no_audit_data")
    else:
        decision.gap_before = sum(b for b, _ in pairs) / len(pairs)
        decision.gap_after = sum(a for _, a in pairs) / len(pairs)
        if not decision.gap_before - decision.gap_after > baseline + cfg.min_gap_drop:
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
