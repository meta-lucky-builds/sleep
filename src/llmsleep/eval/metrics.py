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
) -> dict[tuple[str, Optional[str]], tuple[str, int]]:
    """{(query, context): (decoded text, token count)}. Duplicates cost one generation."""
    unique = list(dict.fromkeys(requests))
    prompts = [handle.format(query, context) for query, context in unique]
    samples = handle.sample(prompts, 1, max_new_tokens, temperature, generator)
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
    for kind in ("direct", "paraphrase", "reverse", "alias", "two_hop", "supersede", "preservation"):
        weight = 1.0 / len(selected[kind]) if selected[kind] else 0.0
        for probe in selected[kind]:
            scored = score_answer(answer(probe)[0], probe.gold, probe.stale, world.vocabulary(probe.vocab))
            tally(kind).add(probe.cluster, float(scored.correct))
            if kind in IMPLIED_KINDS:
                tally("acq_implied").add(probe.cluster, float(scored.correct), weight)
            if kind in ("direct", "supersede"):
                correct_by_episode[probe.episode_ids[0]] = scored.correct
            if kind == "supersede":
                tally("stale").add(probe.cluster, float(scored.stale))
            if kind == "two_hop":
                tally("two_hop_same_night" if probe.same_night else "two_hop_cross_night").add(
                    probe.cluster, float(scored.correct)
                )
                if probe.ripple:
                    tally("two_hop_ripple").add(probe.cluster, float(scored.correct))
    for probe in selected["false_recall"]:
        text = answer(probe)[0]
        abstained = is_abstention(text)
        vocabulary = world.vocabulary(probe.vocab)
        tally("false_recall").add(probe.cluster, float(names_vocabulary(text, vocabulary) and not abstained))
        tally("false_recall_any_specific").add(probe.cluster, float(is_specific(text)))
        tally("abstain").add(probe.cluster, float(abstained))
        neighbour = score_answer(text, probe.neighbour, (), vocabulary).correct and not abstained
        tally("false_recall_neighbour").add(probe.cluster, float(neighbour))

    style_probes = [p for kind in ("direct", "reverse", "alias", "two_hop") for p in selected[kind]]
    lengths = [answer(p)[1] for p in style_probes]
    artefacts = [has_artefact(answer(p)[0]) for p in style_probes]

    metrics: dict[str, Any] = {
        "acq_direct": tally("direct").rate(),
        "acq_paraphrase": tally("paraphrase").rate(),
        "acq_reverse": tally("reverse").rate(),
        "acq_alias": tally("alias").rate(),
        "acq_two_hop": tally("two_hop").rate(),
        "acq_two_hop_same_night": tally("two_hop_same_night").rate(),
        "acq_two_hop_cross_night": tally("two_hop_cross_night").rate(),
        "two_hop_ripple": tally("two_hop_ripple").rate(),
        "acq_implied": tally("acq_implied").rate(),
        "supersede": tally("supersede").rate(),
        "stale": tally("stale").rate(),
        "preservation": tally("preservation").rate(),
        "false_recall": tally("false_recall").rate(),
        "false_recall_any_specific": tally("false_recall_any_specific").rate(),
        "false_recall_neighbour": tally("false_recall_neighbour").rate(),
        "abstain": tally("abstain").rate(),
        "style_length": sum(lengths) / len(lengths) if lengths else None,
        "style_artefact": sum(artefacts) / len(artefacts) if artefacts else None,
        "counts": {kind: len(probes) for kind, probes in selected.items()},
    }
    metrics.update(_audit_metrics(store, correct_by_episode))
    per_cluster = {name: t.clusters for name, t in tallies.items()}
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
