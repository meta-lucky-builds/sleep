"""Every PREREG metric for one night, by free generation through ModelHandle, inside `ledger.exempt()`.

Evaluation uses greedy generation under the committed bank. The prompt wrapper is the handle's own, the same
one used for training and teachers. Only an arm flag (`open_book_eval`) changes anything between arms.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import torch

from ..arms import ArmConfig
from ..config import make_generator
from ..memory import EpisodicStore, format_context, make_retriever
from ..stats import auroc
from .answers import has_artefact, is_abstention, is_specific, names_vocabulary, score_answer
from .probes import Probe, select_probes

IMPLIED_KINDS = ("reverse", "alias", "two_hop")
# Probe kind -> reported metric. Each also gets `<metric>_permuted` (gold of another fact of the same type, so
# what a model scores by knowing only the answer's type) and `<metric>_lift`. PREREG A1.3: margins apply to lift.
METRIC_OF_KIND = {
    "direct": "acq_direct",
    "paraphrase": "acq_paraphrase",
    "reverse": "acq_reverse",
    "alias": "acq_alias",
    "two_hop": "acq_twohop",
    "supersede": "supersede",
    "preservation": "preservation",
}


@dataclass
class EvalConfig:
    max_new_tokens: int = 24
    batch_size: int = 32
    temperature: float = 0.0  # Greedy. Sampled answers would make accuracy noisier for no gain.
    max_per_kind: Optional[int] = 300  # Cap per probe kind per night, by a stored hash ranking.
    drift: bool = True
    drift_tokens: int = 16
    capability: bool = True
    entrench_nights: list[int] = field(default_factory=list)  # Nights with the (costly) contradiction probe.
    entrench_max: Optional[int] = 100


@contextmanager
def eval_batch_size(handle: Any, size: int) -> Iterator[None]:
    previous = handle.batch_size
    handle.batch_size = size
    try:
        yield
    finally:
        handle.batch_size = previous


def generate_answers(
    handle: Any,
    requests: Sequence[tuple[str, Optional[str]]],
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator,
    answer_mode: bool = False,
) -> dict[tuple[str, Optional[str]], tuple[str, int]]:
    """{(query, context): (decoded text, token count)}. Duplicates cost one generation.

    Nightly evaluation stays out of answer mode, so `style_length` sees what the model really writes, and
    scoring cuts to the first line anyway. E0 uses answer mode to save CPU time.
    """
    unique = list(dict.fromkeys(requests))
    prompts = [handle.format(query, context) for query, context in unique]
    extra = {"answer_mode": True} if answer_mode else {}
    samples = handle.sample(prompts, 1, max_new_tokens, temperature, generator, **extra)
    return {
        request: (handle.decode(drawn[0].token_ids), len(drawn[0].token_ids))
        for request, drawn in zip(unique, samples)
    }


class Tally:
    """Numerator and denominator per cluster, which is all the bootstrap needs."""

    def __init__(self) -> None:
        self.clusters: dict[str, list[float]] = {}

    def add(self, cluster: str, value: float, weight: float = 1.0) -> None:
        entry = self.clusters.setdefault(cluster, [0.0, 0.0])
        entry[0] += weight * value
        entry[1] += weight

    def rate(self) -> Optional[float]:
        den = sum(entry[1] for entry in self.clusters.values())
        return sum(entry[0] for entry in self.clusters.values()) / den if den > 0 else None

    def __len__(self) -> int:
        return len(self.clusters)


def _context_for(
    probe: Probe, arm: ArmConfig, store: EpisodicStore, retrieval_k: int
) -> Optional[str]:
    """Closed book, unless the arm answers from retrieval. Entrenchment probes carry a fixed context."""
    if probe.context is not None:
        return probe.context
    if not arm.open_book_eval:
        return None
    retriever = make_retriever(arm.retriever)
    gold_ids = [episode_id for episode_id in probe.episode_ids if episode_id in store]
    context = format_context(retriever.retrieve(probe.query, store, retrieval_k, gold_ids))
    return context or None


def evaluate_probes(
    handle: Any,
    store: EpisodicStore,
    arm: ArmConfig,
    world: Any,
    night: int,
    world_seed: int,
    cfg: EvalConfig,
    retrieval_k: int = 2,
) -> tuple[dict, dict[str, dict]]:
    """(metrics, per_cluster) for acquisition, supersession, preservation, false recall and style."""
    selected = {
        kind: select_probes(world.probes, kind, night, cfg.max_per_kind, world_seed)
        for kind in ("direct", "paraphrase", "reverse", "alias", "two_hop", "supersede", "preservation", "false_recall")
    }
    requests: list[tuple[str, Optional[str]]] = []
    context_of: dict[str, Optional[str]] = {}
    for probes in selected.values():
        for probe in probes:
            context_of[probe.id] = _context_for(probe, arm, store, retrieval_k)
            requests.append((probe.query, context_of[probe.id]))
    answers = generate_answers(
        handle, requests, cfg.max_new_tokens, cfg.temperature, make_generator(world_seed, night, "eval:answers")
    )

    def answer(probe: Probe) -> tuple[str, int]:
        return answers[(probe.query, context_of[probe.id])]

    tallies: dict[str, Tally] = {}

    def tally(name: str) -> Tally:
        return tallies.setdefault(name, Tally())

    correct_by_episode: dict[str, bool] = {}
    age_hits: dict[int, list[int]] = {}
    for kind, name in METRIC_OF_KIND.items():
        weight = 1.0 / len(selected[kind]) if selected[kind] else 0.0
        for probe in selected[kind]:
            text = answer(probe)[0]
            vocabulary = world.vocabulary(probe.vocab)
            scored = score_answer(text, probe.gold, probe.stale, vocabulary, probe.query)
            correct = float(scored.correct)
            control = float(score_answer(text, probe.permuted, (), vocabulary, probe.query).correct)
            tally(name).add(probe.cluster, correct)
            tally(f"{name}_permuted").add(probe.cluster, control)
            tally(f"{name}_lift").add(probe.cluster, correct - control)
            if kind in IMPLIED_KINDS:
                tally("acq_implied").add(probe.cluster, correct, weight)
                tally("acq_implied_permuted").add(probe.cluster, control, weight)
                tally("acq_implied_lift").add(probe.cluster, correct - control, weight)
            if kind in ("direct", "supersede"):
                correct_by_episode[probe.episode_ids[0]] = scored.correct
                bucket = age_hits.setdefault(night - probe.valid_from, [0, 0])
                bucket[0] += int(scored.correct)
                bucket[1] += 1
            if kind == "supersede":
                tally("stale").add(probe.cluster, float(scored.stale))
            if kind == "two_hop":
                tally("acq_twohop_same_night" if probe.same_night else "acq_twohop_cross_night").add(
                    probe.cluster, correct
                )
                # Reported apart, since the two chains have answer vocabularies of different sizes (review H5).
                tally(f"acq_twohop_{probe.vocab}").add(probe.cluster, correct)
                if probe.ripple:
                    tally("acq_twohop_ripple").add(probe.cluster, correct)
    for probe in selected["false_recall"]:
        text = answer(probe)[0]
        abstained = is_abstention(text)
        vocabulary = world.vocabulary(probe.vocab)
        bleed = names_vocabulary(text, vocabulary) and not abstained
        tally("taught_value_bleed").add(probe.cluster, float(bleed))
        tally("false_recall_any_specific").add(probe.cluster, float(is_specific(text)))
        tally("abstain").add(probe.cluster, float(abstained))
        neighbour = score_answer(text, probe.neighbour, (), vocabulary, probe.query).correct and not abstained
        tally("false_recall_neighbour").add(probe.cluster, float(neighbour))

    style_probes = [p for kind in ("direct", "reverse", "alias", "two_hop") for p in selected[kind]]
    lengths = [answer(p)[1] for p in style_probes]
    artefacts = [has_artefact(answer(p)[0]) for p in style_probes]

    metrics: dict[str, Any] = {name: t.rate() for name, t in tallies.items()}
    always = ("acq_implied", "stale", "taught_value_bleed", "false_recall_any_specific", "false_recall_neighbour", "abstain")
    for name in tuple(METRIC_OF_KIND.values()) + always:  # Present even with no probe tonight.
        metrics.setdefault(name, None)
    metrics.update(
        {
            "style_length": sum(lengths) / len(lengths) if lengths else None,
            "style_artefact": sum(artefacts) / len(artefacts) if artefacts else None,
            "counts": {kind: len(probes) for kind, probes in selected.items()},
            # Nights since the fact was taught -> [correct, count]: M6, whether compaction decays old facts.
            "acq_by_age": {str(age): age_hits[age] for age in sorted(age_hits)},
        }
    )
    metrics.update(_audit_metrics(store, correct_by_episode))
    # The control's own clusters are not stored: `_lift` already carries it, and results stay smaller.
    per_cluster = {name: t.clusters for name, t in tallies.items() if not name.endswith("_permuted")}
    return metrics, per_cluster


def _audit_metrics(store: EpisodicStore, correct: dict[str, bool]) -> dict:
    """H8: does an episode's last audit gap predict its closed-book correctness? And is retirement safe?"""
    pairs: list[list[float]] = []
    retired_total, retired_wrong = 0, 0
    for episode in store.all_real():
        if episode.id not in correct:
            continue
        if "last_gap" in episode.meta:
            pairs.append([float(episode.meta["last_gap"]), float(correct[episode.id])])
        if episode.status == "retired":
            retired_total += 1
            retired_wrong += int(not correct[episode.id])
    # Low gap should mean correct, so the score is the negated gap.
    value = auroc([int(c) for _, c in pairs], [-g for g, _ in pairs]) if pairs else None
    return {
        "audit_auroc": value,
        "audit_pairs": pairs,
        "retired": retired_total,
        "false_retirement": retired_wrong / retired_total if retired_total else None,
    }


def evaluate_night(
    handle: Any,
    store: EpisodicStore,
    arm: ArmConfig,
    world: Any,
    night: int,
    world_seed: int,
    cfg: EvalConfig,
    capability: Any = None,
    drift: Any = None,
    retrieval_k: int = 2,
) -> dict:
    """All metrics after night `night`. Spends only exempt tokens and never touches the weights."""
    from .capability import evaluate_capability
    from .drift import measure_generic_drift
    from .entrench import evaluate_entrenchment

    with handle.ledger.exempt(), handle.banks("committed"), eval_batch_size(handle, cfg.batch_size):
        metrics, per_cluster = evaluate_probes(handle, store, arm, world, night, world_seed, cfg, retrieval_k)
        if capability is not None:
            real, real_clusters = evaluate_capability(handle, capability, cfg, world_seed, night)
            metrics.update(real)
            per_cluster.update(real_clusters)
        if drift is not None:
            value, clusters = measure_generic_drift(handle, drift)
            metrics["drift_generic"] = value
            per_cluster["drift_generic"] = clusters
        if night in cfg.entrench_nights:
            entrenched, entrench_clusters = evaluate_entrenchment(
                handle, store, world, night, world_seed, cfg
            )
            metrics.update(entrenched)
            per_cluster.update(entrench_clusters)
    delta = getattr(handle, "delta", None)
    if delta is not None:
        stats = delta.stats()
        metrics["delta_norm"] = stats["delta_norm"]
        metrics["delta_rank"] = stats["delta_rank"]
    metrics["night"] = night
    metrics["per_cluster"] = per_cluster
    return metrics
