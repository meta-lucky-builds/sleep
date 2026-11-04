"""H8's measurement (A2.2): within fact, before and after a night, on identical queries. Logged; drives nothing.

What the nightly evaluation had before this: one gap per episode, after the night only, from generated queries, set
against correctness on another query. That cannot show within-fact change, and it has no control for regression to
the mean. So for a panel of facts this logs, under the committed weights, immediately before training and after the
commit decision:

- the gap on the fact's frozen direct probe, whose wording no arm trains on (invariant 6);
- the gap on held-out generated queries, which is what PREREG's H8 names and what a deployed audit would have;
- closed-book greedy correctness on that same direct probe, scored against the world's gold.

Three things keep it clean:

- The panel is a hash of (world seed, fact key), so it is the same in every arm and cannot be selected on a gap or
  a result. Flagging on the "before" value would make the mean fall by regression alone.
- Each gap keeps its seed before and after, and its distractor on every night, so a change is not a resampling.
- Control facts about people who are in no episode are measured the same way. Their change is what the night does
  to a fact nothing taught, including regression to the mean, so `experiments/analyze.py` reads taught facts against it.

All of it runs inside `ledger.exempt()` on its own generators and writes only `report.fact_audit` and one meta key that
nothing else reads, so a night's weights and budget do not depend on it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..config import NightConfig, make_generator
from ..eval.answers import score_answer
from ..eval.metrics import generate_answers
from ..memory import DistractorSampler, EpisodicStore, format_context
from .commit import AuditItem, measure_gaps
from .queries import Candidate

_QUERIES_META = "fact_audit_queries"
_PROBE_KINDS = ("direct", "supersede")


@dataclass(frozen=True)
class FactItem:
    """One fact on one night. `role` is tonight, older or control. Contexts are fixed for the night."""

    key: str
    role: str
    attr: str
    query: str
    gold: tuple[str, ...]
    stale: tuple[str, ...]
    context: str
    distractor: Optional[str]
    entities: tuple[str, ...]
    heldout: tuple[str, ...] = ()


def panel_keys(fact_keys: Sequence[str], world_seed: int, size: int) -> list[str]:
    """The followed facts: a function of the seed and the key, whatever the arm."""
    rank = lambda key: hashlib.sha256(f"{world_seed}:fact_audit:{key}".encode("utf-8")).digest()  # noqa: E731
    return sorted(fact_keys, key=rank)[: max(0, size)]


def _distractor(sampler: DistractorSampler, entities: Sequence[str], context: str, world_seed: int, key: str) -> Optional[str]:
    generator = make_generator(world_seed, -1, f"fact_audit:distractor:{key}")
    episode = sampler.sample(entities, len(context), generator)
    return format_context([episode]) if episode is not None else None


def plan_fact_audit(
    world: Any,
    store: EpisodicStore,
    night: int,
    world_seed: int,
    held_candidates: Sequence[Candidate],
    cfg: NightConfig,
) -> list[FactItem]:
    """Panel facts taught by now, then the controls. Empty for a world with no probes or controls to offer."""
    if not hasattr(world, "probes") or not hasattr(world, "control_facts"):
        return []
    audit = cfg.fact_audit
    sampler = DistractorSampler(world.distractor_pool(), cfg.select.distractor_length_tol)
    panel = set(panel_keys(world.fact_keys, world_seed, audit.panel))
    fresh: dict[str, list[str]] = {}
    for candidate in held_candidates:
        if not candidate.is_pair:
            fresh.setdefault(candidate.episode_ids[0], []).append(candidate.query)
    items: list[FactItem] = []
    for probe in world.probes:
        if probe.kind not in _PROBE_KINDS or probe.cluster not in panel or not probe.active(night):
            continue
        episode_id = probe.episode_ids[0]
        if episode_id not in store:
            continue
        meta = store.get(episode_id).meta
        if episode_id in fresh and _QUERIES_META not in meta:
            meta[_QUERIES_META] = fresh[episode_id][: audit.heldout_per_fact]
        episode = world.episode(episode_id)
        context = format_context([episode])
        items.append(
            FactItem(
                key=probe.cluster,
                role="tonight" if probe.valid_from == night else "older",
                attr=probe.vocab,
                query=probe.query,
                gold=probe.gold,
                stale=probe.stale,
                context=context,
                distractor=_distractor(sampler, episode.entities, context, world_seed, probe.id),
                entities=tuple(episode.entities),
                heldout=tuple(meta.get(_QUERIES_META, ())),
            )
        )
    for fact in world.control_facts(audit.controls):
        items.append(
            FactItem(
                key=fact.key,
                role="control",
                attr=fact.attr,
                query=fact.query,
                gold=fact.gold,
                stale=(),
                context=fact.text,
                distractor=_distractor(sampler, fact.entities, fact.text, world_seed, fact.key),
                entities=fact.entities,
            )
        )
    return items


def measure_fact_state(
    handle: Any, items: Sequence[FactItem], world: Any, world_seed: int, night: int, cfg: NightConfig
) -> list[dict]:
    """Gap and correctness per item under the active banks. The caller holds `ledger.exempt()` and the banks."""
    gap_items: list[AuditItem] = []
    owners: list[tuple[int, str]] = []
    for index, item in enumerate(items):
        if item.distractor is None:
            continue
        gap_items.append(AuditItem(item.query, item.context, item.distractor, ()))
        owners.append((index, "probe"))
        for query in item.heldout:
            gap_items.append(AuditItem(query, item.context, item.distractor, ()))
            owners.append((index, "heldout"))
    gaps, _ = measure_gaps(
        handle, gap_items, world_seed, night, cfg.signals.k, cfg.signals.max_new_tokens, "fact_audit"
    )
    probe_gap: dict[int, float] = {}
    heldout_gaps: dict[int, list[float]] = {}
    for (index, kind), gap in zip(owners, gaps):
        if gap is None:
            continue
        if kind == "probe":
            probe_gap[index] = gap
        else:
            heldout_gaps.setdefault(index, []).append(gap)
    requests = [(item.query, None) for item in items]
    answers = (
        generate_answers(
            handle,
            requests,
            cfg.fact_audit.answer_tokens,
            0.0,
            make_generator(world_seed, night, "fact_audit:answers"),
            answer_mode=True,
        )
        if requests
        else {}
    )
    out = []
    for index, (item, request) in enumerate(zip(items, requests)):
        text = answers[request][0]
        scored = score_answer(text, item.gold, item.stale, world.vocabulary(item.attr), item.query)
        spread = heldout_gaps.get(index)
        out.append(
            {
                "gap": probe_gap.get(index),
                "heldout": sum(spread) / len(spread) if spread else None,
                "heldout_n": len(spread) if spread else 0,
                "correct": bool(scored.correct),
                "answer": text.strip()[:80],
            }
        )
    return out


def _null_gaps(
    handle: Any, items: Sequence[FactItem], world: Any, world_seed: int, night: int, cfg: NightConfig
) -> list[float]:
    """Gaps between two contexts unrelated to the query, as retirement's threshold would have been built from."""
    sampler = DistractorSampler(world.distractor_pool(), cfg.select.distractor_length_tol)
    null_items: list[AuditItem] = []
    for item in items:
        if len(null_items) >= cfg.fact_audit.null_items:
            break
        if item.role == "control" or item.distractor is None:
            continue
        generator = make_generator(world_seed, night, f"fact_audit:null:{item.key}")
        first = sampler.sample(item.entities, len(item.context), generator)
        second = sampler.sample(item.entities, len(item.context), generator, [first.id]) if first is not None else None
        if first is not None and second is not None:
            null_items.append(AuditItem(item.query, format_context([first]), format_context([second]), ()))
    gaps, _ = measure_gaps(
        handle, null_items, world_seed, night, cfg.signals.k, cfg.signals.max_new_tokens, "fact_audit_null"
    )
    return [gap for gap in gaps if gap is not None]


def fact_audit_record(
    handle: Any,
    items: Sequence[FactItem],
    before: Sequence[dict],
    world: Any,
    world_seed: int,
    night: int,
    accepted: bool,
    cfg: NightConfig,
) -> dict:
    """The second measurement, joined with the first. JSON-able; read only by `experiments/analyze.py`."""
    after = measure_fact_state(handle, items, world, world_seed, night, cfg)
    rows = [
        {
            "key": item.key,
            "role": item.role,
            "attr": item.attr,
            "gap_before": first["gap"],
            "gap_after": second["gap"],
            "heldout_before": first["heldout"],
            "heldout_after": second["heldout"],
            "heldout_n": min(first["heldout_n"], second["heldout_n"]),
            "correct_before": first["correct"],
            "correct_after": second["correct"],
            "answer_before": first["answer"],
            "answer_after": second["answer"],
        }
        for item, first, second in zip(items, before, after)
    ]
    return {
        "items": rows,
        "null_gaps": _null_gaps(handle, items, world, world_seed, night, cfg),
        "accepted": accepted,
        "retire_nights": cfg.commit.retire_nights,
        "k": cfg.signals.k,
        "max_new_tokens": cfg.signals.max_new_tokens,
    }
