"""H6: open-book contradiction-following, consolidated facts versus matched never-consolidated ones.

Each pair puts the same claimed value in the same wording into the context: once about a person whose true value
the arm has consolidated, once about a person never taught. A pair counts only if the fact is consolidated in this
arm's store tonight, so the comparison is not diluted by rolled-back or gated facts. The claim is directional and
says nothing good or bad by itself: lower following for consolidated facts is what entrenchment would look like.
"""
from __future__ import annotations

from typing import Any

from ..config import make_generator
from ..memory import EpisodicStore
from .answers import score_answer
from .metrics import EvalConfig, Tally, generate_answers
from .probes import select_probes


def evaluate_entrenchment(
    handle: Any, store: EpisodicStore, world: Any, night: int, world_seed: int, cfg: EvalConfig
) -> tuple[dict, dict[str, dict]]:
    """The caller holds `ledger.exempt()` and the bank selection."""
    facts = [
        p
        for p in select_probes(world.probes, "entrench_fact", night, None, world_seed)
        if p.episode_ids[0] in store and store.get(p.episode_ids[0]).meta.get("consolidated_night") is not None
    ]
    if cfg.entrench_max is not None:
        facts = facts[: cfg.entrench_max]
    wanted = {p.cluster for p in facts}
    controls = [
        p for p in select_probes(world.probes, "entrench_control", night, None, world_seed) if p.cluster in wanted
    ]
    answers = generate_answers(
        handle,
        [(p.query, p.context) for p in facts + controls],
        cfg.max_new_tokens,
        cfg.temperature,
        make_generator(world_seed, night, "eval:entrench"),
    )
    follow_fact, follow_control, kept_old = Tally(), Tally(), Tally()
    for probe in facts:
        scored = score_answer(answers[(probe.query, probe.context)][0], probe.gold, probe.stale, world.vocabulary(probe.vocab))
        follow_fact.add(probe.cluster, float(scored.correct))
        kept_old.add(probe.cluster, float(scored.stale))
    for probe in controls:
        scored = score_answer(answers[(probe.query, probe.context)][0], probe.gold, probe.stale, world.vocabulary(probe.vocab))
        follow_control.add(probe.cluster, float(scored.correct))
    metrics = {
        "entrench_follow_fact": follow_fact.rate(),
        "entrench_follow_control": follow_control.rate(),
        "entrench_kept_old": kept_old.rate(),
        "entrench_pairs": len(follow_fact),
    }
    clusters = {"entrench_follow_fact": follow_fact.clusters, "entrench_follow_control": follow_control.clusters}
    return metrics, clusters
