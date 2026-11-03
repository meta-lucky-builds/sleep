"""One night: (theta, M) -> (theta', M'), and the State that survives between nights.

Arms differ only through ArmConfig flags, read here (invariant 5). A night is atomic: run it, then call
`State.save`. If a process dies mid-night, resume from the last save; since every seed derives from
(world_seed, night, purpose), the re-run is identical (invariant 7).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn

from ..arms import ArmConfig
from ..config import NightConfig, make_generator
from ..delta import DeltaModel
from ..handle import ModelHandle, PromptFormat
from ..ledger import ComputeLedger
from ..memory import DistractorSampler, Episode, EpisodicStore, format_context, make_retriever
from ..signals import SignalCache, chd_safe
from .commit import AuditItem, Decision, decide, measure_drift, measure_gaps, update_episodes
from .compact import compact_and_repair, merge_only
from .consolidate import (
    HardExample,
    TargetSpec,
    TokenTargets,
    build_self_sampled_targets,
    build_targets,
    hard_examples,
    train_loop,
)
from .queries import Candidate, PairQueryGen, WorldLike, generate_singles, make_query_gen, normalize_query, split_heldout
from .select import (
    Partition,
    Thresholds,
    null_distribution,
    partition,
    thresholds_from_null,
    touching_episodes,
    uniform_partition,
)
from .wake import observe


class SleepPressure:
    """The trigger value: mean remaining gap over the active store, from the latest audit. No clock."""

    @staticmethod
    def compute(store: EpisodicStore) -> Optional[float]:
        superseded = store.superseded_ids()
        gaps = [
            e.meta["last_gap"] for e in store.active() if e.id not in superseded and "last_gap" in e.meta
        ]
        return sum(gaps) / len(gaps) if gaps else None


@dataclass
class State:
    """Everything that survives a night. `night` is the index of the next night to run."""

    handle: ModelHandle
    store: EpisodicStore
    world: WorldLike
    world_seed: int
    night: int = 0
    heldout: set[str] = field(default_factory=set)  # Normalised queries never to be trained on.

    @property
    def ledger(self) -> ComputeLedger:
        return self.handle.ledger

    @classmethod
    def create(
        cls,
        model: nn.Module,
        tokenizer: Any,
        world: WorldLike,
        world_seed: int,
        cfg: NightConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> "State":
        """Attach banks to a freshly loaded base model. Mutates `model` in place."""
        delta = DeltaModel(model, rank=cfg.model.rank, alpha=cfg.model.alpha)
        handle = ModelHandle(
            delta,
            tokenizer,
            device,
            ComputeLedger(),
            PromptFormat(cfg.model.prompt_template, cfg.model.empty_context),
            cfg.model.batch_size,
            cfg.model.fp16,
        )
        return cls(handle, EpisodicStore(), world, world_seed)

    def save(self, root: str) -> str:
        """Atomic checkpoint: write a temp directory, rename it, then swap the LATEST pointer last.

        A crash at any point leaves LATEST naming a complete checkpoint. Returns the checkpoint directory.
        """
        os.makedirs(root, exist_ok=True)
        name = f"night-{self.night:05d}"
        tmp = os.path.join(root, f".tmp-{name}")
        final = os.path.join(root, name)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        torch.save(self.handle.delta.state_dict(), os.path.join(tmp, "adapters.pt"))
        self.store.save(os.path.join(tmp, "store.json"))
        meta = {
            "night": self.night,
            "world_seed": self.world_seed,
            "heldout": sorted(self.heldout),
            "base_fingerprint": self.handle.delta.base_fingerprint(),
        }
        with open(os.path.join(tmp, "state.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=1)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)
        pointer = os.path.join(root, "LATEST")
        with open(f"{pointer}.tmp", "w", encoding="utf-8") as handle:
            handle.write(name)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(f"{pointer}.tmp", pointer)
        return final

    @classmethod
    def load(cls, root: str, handle: ModelHandle, world: WorldLike) -> "State":
        """Restore the latest checkpoint into `handle`, whose base must be the same (checked by hash)."""
        with open(os.path.join(root, "LATEST"), "r", encoding="utf-8") as stream:
            directory = os.path.join(root, stream.read().strip())
        with open(os.path.join(directory, "state.json"), "r", encoding="utf-8") as stream:
            meta = json.load(stream)
        if handle.delta.base_fingerprint() != meta["base_fingerprint"]:
            raise ValueError("checkpoint was written over a different base model")
        handle.delta.load_state_dict(torch.load(os.path.join(directory, "adapters.pt"), map_location="cpu", weights_only=True))
        return cls(
            handle,
            EpisodicStore.load(os.path.join(directory, "store.json")),
            world,
            meta["world_seed"],
            meta["night"],
            set(meta["heldout"]),
        )


@dataclass
class NightReport:
    """JSON-able record of a night. Every stage says what it did, including what it could not finish."""

    night: int
    arm: str
    episodes_new: int = 0
    wake: list[dict] = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    compaction: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    retired: list[str] = field(default_factory=list)
    pressure: Optional[float] = None
    delta: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def active_stage_fractions(arm: ArmConfig, cfg: NightConfig) -> dict[str, float]:
    """The ledger stages this arm uses, with their configured fractions."""
    active = {"train"}
    if arm.query_source != "none":
        active.add("synthesis")
    if arm.selection or arm.write_gate:
        active.add("signals")
    if arm.teacher_kl or arm.self_sampled_kl or arm.anchors != "none":
        active.add("teacher")
    if arm.compact and cfg.compact.repair_steps > 0:
        active.add("repair")
    if arm.audit:
        active.add("audit")
    fractions = {stage: f for stage, f in cfg.budget.fractions.items() if stage in active}
    if arm.query_source != "none" and not (arm.selection or arm.write_gate):
        # Same split as a CHD arm, with the signal allocation credited to extra training steps. Those steps
        # only happen if TrainConfig.max_steps is None, so that training runs until its tokens are spent.
        fractions["train"] += cfg.budget.fractions.get("signals", 0.0)
    return fractions


def _uniform_counts(cfg: NightConfig, night: int, side: str, candidates: Sequence[Candidate]) -> tuple[int, int, str]:
    """(n_integrate, n_anchor, source) for a uniform draw: matched to the paired CHD arm, else fractions."""
    matched = cfg.select.matched_selection
    record = matched.get(str(night)) if matched else None
    if record is not None:
        return int(record[side]["integrate"]), int(record[side]["anchor"]), "matched"
    usable = sum(1 for c in candidates if c.context)
    return (
        math.ceil(cfg.select.uniform_integrate_frac * usable),
        math.floor(cfg.select.uniform_anchor_frac * usable),
        "fraction",
    )


def anchor_teacher(
    query: str, arm: ArmConfig, consolidated: Sequence[Episode], lexical_min: float
) -> tuple[str, Optional[str]]:
    """(banks, context) of an anchor's target (THEORY stage 3).

    A query touching consolidated knowledge uses the committed model with those episodes in context.
    Anything else uses germline, which cannot compound across nights; `anchors="chain"` swaps in yesterday's
    model at exactly this point.
    """
    touching = touching_episodes(query, consolidated, lexical_min)
    if touching:
        return "committed", format_context(touching)
    return ("committed" if arm.anchors == "chain" else "germline"), None


def hard_pool_examples(
    arm: ArmConfig,
    store: EpisodicStore,
    handle: ModelHandle,
    new_ids: set[str],
    integrate: Sequence[Candidate],
    boost: int,
    heldout: set[str],
) -> list[HardExample]:
    """new: tonight's episodes. all_real: uniform replay. selected: as all_real, with episodes that
    have integration queries sampled `boost` times as often."""
    replayable = store.replayable()
    if arm.hard_pool == "none":
        return []
    if arm.hard_pool == "new":
        return hard_examples(handle, [e for e in replayable if e.id in new_ids], heldout)
    if arm.hard_pool == "all_real":
        return hard_examples(handle, replayable, heldout)
    boosted = {episode_id for c in integrate for episode_id in c.episode_ids}
    episodes: list[Episode] = []
    for episode in replayable:
        episodes.extend([episode] * (boost if episode.id in boosted else 1))
    return hard_examples(handle, episodes, heldout)


def _attach_context_and_distractor(
    candidates: Sequence[Candidate],
    store: EpisodicStore,
    arm: ArmConfig,
    sampler: DistractorSampler,
    retrieval_k: int,
    generator: torch.Generator,
) -> None:
    if not candidates:
        return  # Replay arms have no retriever and no candidates.
    retriever = make_retriever(arm.retriever)
    for candidate in candidates:
        episodes = retriever.retrieve(candidate.query, store, retrieval_k, candidate.episode_ids)
        candidate.context = format_context(episodes)
        if not candidate.context:
            candidate.distractor = None
            continue
        distractor = sampler.sample(candidate.entities, len(candidate.context), generator, candidate.episode_ids)
        candidate.distractor = format_context([distractor]) if distractor is not None else None


def _load_old_items(store: EpisodicStore, skip: set[tuple[str, str]]) -> list[AuditItem]:
    """Persisted held-out items of episodes still in play, each once."""
    superseded = store.superseded_ids()
    items: list[AuditItem] = []
    for episode in store.active():
        if episode.id in superseded:
            continue
        for record in episode.meta.get("audit_items", []):
            item = AuditItem.from_dict(record)
            if (item.query, item.context) not in skip:
                skip.add((item.query, item.context))
                items.append(item)
    return items


def run_night(state: State, cfg: NightConfig) -> NightReport:
    """Run night `state.night`, advance it, and report. The caller saves the state afterwards."""
    arm = cfg.arm
    handle, store, world, night = state.handle, state.store, state.world, state.night
    ledger, delta = handle.ledger, handle.delta

    def rng(purpose: str) -> torch.Generator:
        return make_generator(state.world_seed, night, purpose)

    new_episodes = list(world.episodes(night))
    report = NightReport(night=night, arm=arm.name, episodes_new=len(new_episodes))
    ledger.begin_night(cfg.budget.night_tokens, active_stage_fractions(arm, cfg) if arm.train else {})

    if not arm.train:
        # Weights untouched and nothing spent. Episodes still go in the store, for retrieval by day.
        for episode in new_episodes:
            store.add(episode)
        report.wake = [{"episode_id": e.id, "stored": True, "reason": "no_training_arm"} for e in new_episodes]
        return _finish(state, report)

    delta.reset_fresh(rng("fresh_init"))
    snapshot = delta.snapshot()
    cache = SignalCache()
    new_ids = {e.id for e in new_episodes}

    # ---- Stage 1a: candidates for tonight's episodes, split before anything can be selected or trained on.
    train_cands: list[Candidate] = []
    held_cands: list[Candidate] = []
    query_gen = None
    synthesis_exhausted = False
    with handle.banks("committed"):
        if arm.query_source != "none":
            query_gen = make_query_gen(arm.query_source, world, handle, cfg.queries)
            with ledger.stage("synthesis"):
                singles, synthesis_exhausted = generate_singles(query_gen, new_episodes, rng("query_singles"))
            train_cands, held_cands, held = split_heldout(
                singles, state.world_seed, cfg.commit.heldout_frac, state.heldout
            )
            state.heldout |= held

        # ---- Null distribution, then the write gate.
        pool = list(store.all_real()) + list(world.distractor_pool()) + new_episodes
        sampler = DistractorSampler(pool, cfg.select.distractor_length_tol)
        thresholds = Thresholds(None, None, None, "none", 0)
        signals_exhausted = False
        if arm.selection or arm.write_gate:
            with ledger.stage("signals"):
                null, signals_exhausted = null_distribution(
                    handle, train_cands, sampler, cfg.select, cfg.signals, rng("null"), cache
                )
                thresholds = thresholds_from_null(null, cfg.select)
                gate_rng = rng("wake")
                for episode in new_episodes:
                    probes = _gate_probes(episode, train_cands, sampler, gate_rng)
                    outcome = observe(
                        handle,
                        store,
                        episode,
                        probes,
                        thresholds.gate,
                        cfg.signals.k,
                        cfg.signals.max_new_tokens,
                        gate_rng,
                        cache,
                        cfg.signals.chunk,
                        bypass=not arm.write_gate,
                    )
                    report.wake.append(asdict(outcome))
        else:
            for episode in new_episodes:
                store.add(episode)
            report.wake = [{"episode_id": e.id, "stored": True, "reason": "bypass"} for e in new_episodes]

        stored_new = {e.id for e in new_episodes if e.id in store}
        gated = new_ids - stored_new
        train_cands = [c for c in train_cands if not set(c.episode_ids) & gated]
        held_cands = [c for c in held_cands if not set(c.episode_ids) & gated]

        # ---- Stage 1b: pair contexts need the store to hold tonight's episodes.
        pair_count = 0
        if arm.pair_contexts and query_gen is not None:
            pair_gen = PairQueryGen(query_gen, cfg.queries.max_pairs)
            with ledger.stage("synthesis"):
                pair_cands, pairs_exhausted = pair_gen.generate(store, stored_new, rng("query_pairs"))
            synthesis_exhausted = synthesis_exhausted or pairs_exhausted
            pair_count = len(pair_cands)
            pair_train, pair_held, held = split_heldout(
                pair_cands, state.world_seed, cfg.commit.heldout_frac, state.heldout
            )
            state.heldout |= held
            train_cands += pair_train
            held_cands += pair_held
        if "synthesis" in ledger.caps:
            ledger.close_stage("synthesis")
        # A query can become held out through a pair after its single was split. Held-out always wins.
        train_cands = [c for c in train_cands if normalize_query(c.query) not in state.heldout]
        report.synthesis = {
            "train_candidates": len(train_cands),
            "heldout_candidates": len(held_cands),
            "pair_candidates": pair_count,
            "exhausted": synthesis_exhausted,
        }

        all_cands = train_cands + held_cands
        _attach_context_and_distractor(all_cands, store, arm, sampler, cfg.retrieval_k, rng("distractors"))

        # ---- Audit baseline on the weights we would roll back to. Paid from the audit stage.
        items: list[AuditItem] = []
        gate_flags: list[bool] = []
        gaps_before: list[Optional[float]] = []
        audit_exhausted = False
        held_part = Partition()
        if arm.audit:
            auditable = [c for c in held_cands if c.context and c.distractor is not None]
            auditable = auditable[: cfg.commit.audit_max_items]
            new_items = [AuditItem(c.query, c.context, c.distractor, c.episode_ids) for c in auditable]
            seen = {(i.query, i.context) for i in new_items}
            old_items = _load_old_items(store, seen)[: max(0, cfg.commit.audit_max_items - len(new_items))]
            items = new_items + old_items
            with ledger.stage("audit"):
                gaps_before, audit_exhausted = measure_gaps(
                    handle, items, state.world_seed, night, cfg.signals.k, cfg.signals.max_new_tokens
                )
            new_gaps = gaps_before[: len(new_items)]
            if arm.selection:
                held_part = partition(auditable, new_gaps, thresholds, cfg.select.lexical_min)
            else:
                n_integrate, n_anchor, _ = _uniform_counts(cfg, night, "heldout", auditable)
                held_part = uniform_partition(
                    auditable, n_integrate, n_anchor, cfg.select.lexical_min, rng("uniform_heldout")
                )
            integrate_ids = {id(c) for c in held_part.integrate}
            gate_flags = [id(c) in integrate_ids for c in auditable] + [False] * len(old_items)
            for candidate, item in zip(auditable, new_items):
                if id(candidate) in integrate_ids:
                    # Kept for later nights' pressure and retirement. Stored once, on the first episode.
                    store.get(candidate.episode_ids[0]).meta.setdefault("audit_items", []).append(item.to_dict())

        # ---- Stage 1c: score and partition the training candidates.
        if arm.selection:
            with ledger.stage("signals"):
                scorable = [c for c in train_cands if c.context and c.distractor is not None]
                values, exhausted = chd_safe(
                    handle,
                    [(c.query, c.context, c.distractor) for c in scorable],
                    cfg.signals.k,
                    cfg.signals.max_new_tokens,
                    rng("select"),
                    cache,
                    cfg.signals.chunk,
                )
                signals_exhausted = signals_exhausted or exhausted
            value_by_id = {id(c): v for c, v in zip(scorable, values)}
            train_part = partition(
                train_cands, [value_by_id.get(id(c)) for c in train_cands], thresholds, cfg.select.lexical_min
            )
            count_source = "chd"
        else:
            n_integrate, n_anchor, count_source = _uniform_counts(cfg, night, "train", train_cands)
            train_part = uniform_partition(
                train_cands, n_integrate, n_anchor, cfg.select.lexical_min, rng("uniform_train")
            )
        if "signals" in ledger.caps:
            ledger.close_stage("signals")
        report.signals = {
            "null_size": thresholds.null_size,
            "threshold_source": thresholds.source,
            "integrate_threshold": thresholds.integrate,
            "anchor_threshold": thresholds.anchor,
            "gate_threshold": thresholds.gate,
            "exhausted": signals_exhausted,
        }
        report.selection = {
            "train": train_part.summary(),
            "heldout": held_part.summary(),
            "count_source": count_source,
        }

    # ---- Stage 2, first half: all teacher targets, frozen for the rest of the night.
    consolidated = [
        e for e in store.replayable() if e.meta.get("consolidated_night") is not None and e.id not in new_ids
    ]
    integrate_targets: list[TokenTargets] = []
    anchor_targets: list[TokenTargets] = []
    teacher_exhausted = False
    if "teacher" in ledger.caps:
        teacher_rng = rng("teacher")
        with ledger.stage("teacher"):
            if arm.teacher_kl:
                specs = [TargetSpec(c.query, c.context, "committed", "integrate") for c in train_part.integrate]
                integrate_targets, exhausted = build_targets(handle, specs, cfg.teacher, teacher_rng, cache)
                teacher_exhausted = teacher_exhausted or exhausted
            if arm.anchors != "none":
                specs = _anchor_specs(train_part.anchor, arm, consolidated, cfg.select.lexical_min)
                anchor_targets, exhausted = build_targets(handle, specs, cfg.teacher, teacher_rng, cache)
                teacher_exhausted = teacher_exhausted or exhausted
            if arm.self_sampled_kl:
                self_targets, exhausted = build_self_sampled_targets(handle, cfg.teacher, teacher_rng)
                anchor_targets += self_targets
                teacher_exhausted = teacher_exhausted or exhausted
        ledger.close_stage("teacher")
    report.teacher = {
        "integrate_sequences": len(integrate_targets),
        "anchor_sequences": len(anchor_targets),
        "exhausted": teacher_exhausted,
    }

    heldout_anchor_targets: list[TokenTargets] = []
    if arm.audit and arm.anchors != "none" and held_part.anchor:
        with ledger.stage("audit"):
            specs = _anchor_specs(held_part.anchor, arm, consolidated, cfg.select.lexical_min)
            heldout_anchor_targets, exhausted = build_targets(handle, specs, cfg.teacher, rng("audit_anchors"))
            audit_exhausted = audit_exhausted or exhausted

    # ---- Stage 2, second half: training. Fresh is the only thing that moves.
    hard = hard_pool_examples(
        arm, store, handle, stored_new, train_part.integrate, cfg.select.hard_boost, state.heldout
    )
    with ledger.stage("train"):
        train_result = train_loop(handle, hard, integrate_targets, anchor_targets, cfg.train, rng("train"))
    ledger.close_stage("train")
    report.train = train_result.summary()

    # ---- Stage 4: compaction, or the exact merge.
    if arm.compact:
        repair_active = "repair" in ledger.caps
        with ledger.stage("repair") if repair_active else nullcontext():
            report.compaction = compact_and_repair(
                handle,
                hard,
                integrate_targets,
                anchor_targets,
                cfg.compact,
                cfg.train,
                rng("repair_init"),
                rng("repair"),
                repair_active,
            )
        if repair_active:
            ledger.close_stage("repair")
    else:
        report.compaction = merge_only(handle)

    # ---- Stage 5: audit and commit.
    decision = Decision(accepted=True, reasons=["no_audit"])
    gaps_after: list[Optional[float]] = [None] * len(items)
    if arm.audit:
        with handle.banks("committed"), ledger.stage("audit"):
            gaps_after, exhausted = measure_gaps(
                handle, items, state.world_seed, night, cfg.signals.k, cfg.signals.max_new_tokens
            )
            audit_exhausted = audit_exhausted or exhausted
            drift, exhausted = measure_drift(handle, heldout_anchor_targets)
            audit_exhausted = audit_exhausted or exhausted
        decision = decide(
            [g for g, flag in zip(gaps_before, gate_flags) if flag],
            [g for g, flag in zip(gaps_after, gate_flags) if flag],
            drift,
            cfg.commit,
            arm.transactional,
            audit_exhausted,
        )
    if not decision.accepted:
        delta.restore(snapshot)
    else:
        # Everything that fed the hard term tonight. An episode already marked keeps its first night.
        trained = stored_new if arm.hard_pool == "new" else stored_new | {e.id for e in store.replayable()}
        for episode_id in sorted(trained):
            store.get(episode_id).meta.setdefault("consolidated_night", night)
    report.decision = asdict(decision)
    if arm.audit:
        report.retired = update_episodes(
            store,
            items,
            gaps_after if decision.accepted else gaps_before,
            thresholds.gate,
            decision.accepted,
            cfg.commit.retire_nights,
        )
    return _finish(state, report)


def _finish(state: State, report: NightReport) -> NightReport:
    report.pressure = SleepPressure.compute(state.store)
    if state.handle.delta is not None:
        stats = state.handle.delta.stats()
        stats.pop("layers")
        report.delta = stats
    report.ledger = state.ledger.summary()
    state.night += 1
    return report


def _gate_probes(
    episode: Episode, candidates: Sequence[Candidate], sampler: DistractorSampler, generator: torch.Generator
) -> list[tuple[str, str]]:
    """(query, distractor context) for the write gate, on this episode's training queries only."""
    probes = []
    for candidate in candidates:
        if candidate.episode_ids != (episode.id,):
            continue
        distractor = sampler.sample(candidate.entities, len(candidate.context), generator, candidate.episode_ids)
        if distractor is not None:
            probes.append((candidate.query, format_context([distractor])))
    return probes


def _anchor_specs(
    candidates: Sequence[Candidate], arm: ArmConfig, consolidated: Sequence[Episode], lexical_min: float
) -> list[TargetSpec]:
    specs = []
    for candidate in candidates:
        banks, context = anchor_teacher(candidate.query, arm, consolidated, lexical_min)
        specs.append(TargetSpec(candidate.query, context, banks, "anchor"))
    return specs
