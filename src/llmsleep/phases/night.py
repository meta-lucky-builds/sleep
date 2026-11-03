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
from .commit import AuditItem, Decision, decide, drift_by_target, drift_change, measure_gaps, update_episodes
from .compact import compact_and_repair, merge_only
from .consolidate import (
    HardExample,
    TargetSpec,
    TokenTargets,
    build_self_sampled_targets,
    build_targets,
    estimate_step_tokens,
    hard_examples,
    pair_examples,
    train_loop,
)
from .queries import (
    Candidate,
    PairQueryGen,
    WorldLike,
    build_anchor_pool,
    choose_pairs,
    generate_singles,
    make_query_gen,
    normalize_query,
    pair_stats,
    split_heldout,
)
from .select import (
    NO_THRESHOLDS,
    Partition,
    draw_distractor,
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
        eos_ids: Optional[Sequence[int]] = None,
    ) -> "State":
        """Attach banks to a freshly loaded base model. Mutates `model` in place.

        `eos_ids` replaces the tokenizer's single EOS as the set of tokens that stop sampling.
        """
        delta = DeltaModel(model, rank=cfg.model.rank, alpha=cfg.model.alpha)
        handle = ModelHandle(
            delta,
            tokenizer,
            device,
            ComputeLedger(),
            PromptFormat(cfg.model.prompt_template, cfg.model.empty_context, chat=cfg.model.chat),
            cfg.model.batch_size,
            cfg.model.fp16,
            eos_ids,
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
            "exempt_forward": self.ledger.exempt_forward,
            "exempt_backward": self.ledger.exempt_backward,
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
        handle.ledger.restore_exempt(meta["exempt_forward"], meta["exempt_backward"])
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
    episodes_trained: int = 0  # Episodes in at least one completed optimizer step, training or repair.
    episodes_marked_consolidated: int = 0  # Of those, how many got `consolidated_night` for the first time.
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


def _uniform_anchor_count(cfg: NightConfig, night: int, side: str, pool_size: int) -> tuple[int, str]:
    """(n_anchor, source) for a draw without CHD: matched to the paired CHD arm, else a fraction of the pool."""
    matched = cfg.select.matched_selection
    record = matched.get(str(night)) if matched else None
    if record is not None:
        return int(record[side]["anchor"]), "matched"
    return math.floor(cfg.select.uniform_anchor_frac * pool_size), "fraction"


def anchor_teacher(
    query: str, arm: ArmConfig, older: Sequence[Episode], lexical_min: float
) -> tuple[str, Optional[str]]:
    """(banks, context) of an anchor's target (THEORY stage 3). `older`: real episodes from before tonight.

    A query touching older knowledge uses the committed model with those episodes in context, so it does not pull
    against their hard labels. Anything else uses germline, which cannot compound across nights;
    `anchors="chain"` swaps in yesterday's model at exactly this point.
    """
    touching = touching_episodes(query, older, lexical_min)
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
    have integration queries sampled `boost` times as often. With `replay_pairs`, also the pairs chosen so far."""
    replayable = store.replayable()
    if arm.hard_pool == "none":
        return []
    if arm.hard_pool == "new":
        examples = hard_examples(handle, [e for e in replayable if e.id in new_ids], heldout)
    elif arm.hard_pool == "all_real":
        examples = hard_examples(handle, replayable, heldout)
    else:
        boosted = {episode_id for c in integrate for episode_id in c.episode_ids}
        episodes: list[Episode] = []
        for episode in replayable:
            episodes.extend([episode] * (boost if episode.id in boosted else 1))
        examples = hard_examples(handle, episodes, heldout)
    if arm.replay_pairs:
        current = {e.id for e in replayable}
        pairs = [
            (e, store.get(other))
            for e in replayable
            for other in e.meta.get("replay_pairs", [])
            if other in current  # A superseded partner would replay a stale value.
        ]
        examples += pair_examples(handle, pairs)
    return examples


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
        candidate.context_ids = tuple(e.id for e in episodes)
        if not candidate.context:
            candidate.distractor = None
            continue
        candidate.distractor, _ = draw_distractor(candidate, sampler, generator)


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


def _shuffled(items: Sequence, generator: torch.Generator) -> list:
    return [items[i] for i in torch.randperm(len(items), generator=generator).tolist()]


def _control_items(
    candidates: Sequence[Candidate], sampler: DistractorSampler, n: int, generator: torch.Generator
) -> list[AuditItem]:
    """The null control for the audit: held-out queries between two contexts unrelated to them."""
    items: list[AuditItem] = []
    for candidate in candidates[:n]:
        first, first_ids = draw_distractor(candidate, sampler, generator)
        if first is None:
            continue
        second, _ = draw_distractor(candidate, sampler, generator, first_ids)
        if second is not None:
            items.append(AuditItem(candidate.query, first, second, ()))
    return items


def _release_surplus(
    ledger: ComputeLedger,
    hard: Sequence[HardExample],
    integrate: Sequence[TokenTargets],
    anchor: Sequence[TokenTargets],
    cfg: NightConfig,
    audit_after_need: float,
) -> None:
    """Give training whatever repair and the after-audit will not need, so that no stage ends with unspent tokens."""
    if "repair" in ledger.caps:
        need = math.ceil(cfg.commit.after_margin * cfg.compact.repair_steps * estimate_step_tokens(hard, integrate, anchor, cfg.train))
        surplus = ledger.remaining("repair") - need
        if surplus > 0:
            ledger.transfer("repair", "train", surplus)
    if "audit" in ledger.caps:
        surplus = ledger.remaining("audit") - math.ceil(cfg.commit.after_margin * audit_after_need)
        if surplus > 0:
            ledger.transfer("audit", "train", surplus)


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
    k, max_new = cfg.signals.k, cfg.signals.max_new_tokens

    # ---- Stage 1a: candidates for tonight's episodes, split before anything can be selected or trained on.
    singles: list[Candidate] = []
    train_cands: list[Candidate] = []
    held_cands: list[Candidate] = []
    pool_train: list[Candidate] = []
    pool_held: list[Candidate] = []
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

        # ---- Null distribution, then the write gate, each within its share so that scoring keeps a reserve.
        sampler_pool = list(store.all_real()) + list(world.distractor_pool()) + new_episodes
        sampler = DistractorSampler(sampler_pool, cfg.select.distractor_length_tol)
        thresholds = pair_thresholds = NO_THRESHOLDS
        signals_exhausted = False
        signals_total = ledger.remaining("signals") if "signals" in ledger.caps else 0
        null_budget = signals_total * cfg.select.null_share / (2 if arm.pair_contexts else 1)
        if "signals" in ledger.caps:
            with ledger.stage("signals"):
                with ledger.limit(int(null_budget)):
                    null, signals_exhausted = null_distribution(
                        handle, train_cands, sampler, cfg.signals, cfg.select.n_null, rng("null"), cache
                    )
                thresholds = thresholds_from_null(null, cfg.select)
                gate_rng = rng("wake")
                with ledger.limit(int(signals_total * cfg.select.gate_share)):
                    for episode in new_episodes:
                        outcome = observe(
                            handle,
                            store,
                            episode,
                            _gate_probe(episode, train_cands, sampler, gate_rng),
                            thresholds.gate,
                            k,
                            max_new,
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

        deferred = {e.id for e in new_episodes if store.get(e.id).status == "deferred"}
        train_cands = [c for c in train_cands if not set(c.episode_ids) & deferred]
        held_cands = [c for c in held_cands if not set(c.episode_ids) & deferred]
        for candidate in singles:
            store.get(candidate.episode_ids[0]).meta.setdefault("queries", []).append(candidate.query)

        # ---- Stage 1b: pair contexts need the store to hold tonight's episodes. Then the anchor pool.
        pair_count = 0
        chosen_pairs: dict[str, int] = {}
        if arm.pair_contexts and query_gen is not None:
            pair_gen = PairQueryGen(query_gen, cfg.queries.max_pairs)
            with ledger.stage("synthesis"):
                pair_cands, pairs_exhausted = pair_gen.generate(store, new_ids, rng("query_pairs"))
            chosen_pairs = pair_gen.stats
            synthesis_exhausted = synthesis_exhausted or pairs_exhausted
            pair_count = len(pair_cands)
            pair_train, pair_held, held = split_heldout(
                pair_cands, state.world_seed, cfg.commit.heldout_frac, state.heldout
            )
            state.heldout |= held
            train_cands += pair_train
            held_cands += pair_held
            if arm.selection:
                with ledger.stage("signals"), ledger.limit(int(null_budget)):
                    pair_null, exhausted = null_distribution(
                        handle, pair_train, sampler, cfg.signals, cfg.select.n_null_pair, rng("null_pair"), cache
                    )
                    signals_exhausted = signals_exhausted or exhausted
                pair_thresholds = thresholds_from_null(pair_null, cfg.select)
        if arm.replay_pairs:
            pairs = [(a, b) for a, b in store.pairs_sharing_entity() if a.id in new_ids or b.id in new_ids]
            picked = choose_pairs(pairs, cfg.queries.max_pairs, rng("replay_pairs"))
            for a, b in picked:
                a.meta.setdefault("replay_pairs", []).append(b.id)
            chosen_pairs = pair_stats(pairs, picked)
        if arm.anchors != "none":
            with ledger.stage("synthesis"):
                pool, exhausted = build_anchor_pool(handle, store, world, new_ids, cfg.queries, rng("anchor_pool"))
            synthesis_exhausted = synthesis_exhausted or exhausted
            pool_train, pool_held, held = split_heldout(pool, state.world_seed, cfg.commit.heldout_frac, state.heldout)
            state.heldout |= held
        if "synthesis" in ledger.caps:
            ledger.close_stage("synthesis")
        # A query can become held out after its first split. Held-out always wins.
        train_cands = [c for c in train_cands if normalize_query(c.query) not in state.heldout]
        pool_train = [c for c in pool_train if normalize_query(c.query) not in state.heldout]
        tonight_queries = {normalize_query(c.query) for c in train_cands + held_cands}
        pool_train = [c for c in pool_train if normalize_query(c.query) not in tonight_queries]
        pool_held = [c for c in pool_held if normalize_query(c.query) not in tonight_queries]
        report.synthesis = {
            "train_candidates": len(train_cands),
            "heldout_candidates": len(held_cands),
            "pair_candidates": pair_count,
            "pairs": chosen_pairs,
            "anchor_pool_train": len(pool_train),
            "anchor_pool_heldout": len(pool_held),
            "exhausted": synthesis_exhausted,
        }

        _attach_context_and_distractor(train_cands + held_cands, store, arm, sampler, cfg.retrieval_k, rng("distractors"))
        distractor_rng = rng("pool_distractors")
        for candidate in pool_train + pool_held:
            candidate.distractor, _ = draw_distractor(candidate, sampler, distractor_rng)

        # ---- Audit, before training, on the weights we would roll back to. Paid from the audit stage, within a
        # share that keeps enough for the after pass.
        items: list[AuditItem] = []
        control: list[AuditItem] = []
        gate_count = 0
        gaps_before: list[Optional[float]] = []
        control_before: list[Optional[float]] = []
        drift_before: list = []
        heldout_anchor_targets: list[TokenTargets] = []
        audit_after_need = 0
        audit_exhausted = False
        held_part = Partition()
        older = [e for e in store.replayable() if e.id not in new_ids]
        if arm.audit:
            cap = cfg.commit.audit_max_items
            auditable = _shuffled([c for c in held_cands if c.context and c.distractor is not None], rng("audit_order"))
            all_old = _load_old_items(store, {(c.query, c.context) for c in auditable})
            auditable = auditable[: cap - min(len(all_old), int(cap * cfg.commit.old_item_share))]
            old_items = all_old[: cap - len(auditable)]
            anchor_cands = _shuffled(
                [c for c in pool_held if c.distractor is not None], rng("audit_anchor_order")
            )[: cfg.commit.audit_max_anchors]
            with ledger.stage("audit"), ledger.limit(int(ledger.remaining("audit") * cfg.commit.audit_pre_share)):
                if arm.selection:
                    # Flagging on an independent measurement, so that the before/after test has no selection bias.
                    flag_items = [AuditItem(c.query, c.context, c.distractor, c.episode_ids) for c in auditable + anchor_cands]
                    flags, audit_exhausted = measure_gaps(handle, flag_items, state.world_seed, night, k, max_new, "audit_flag")
                    held_part = partition(auditable + anchor_cands, flags, thresholds, pair_thresholds)
                else:
                    n_anchor, _ = _uniform_anchor_count(cfg, night, "heldout", len(anchor_cands))
                    held_part = uniform_partition(auditable, anchor_cands, n_anchor, rng("uniform_heldout"))
                flagged = [c for c in held_part.integrate if c.distractor is not None]
                new_items = [AuditItem(c.query, c.context, c.distractor, c.episode_ids) for c in flagged]
                for candidate, item in zip(flagged, new_items):
                    # Kept for later nights' pressure and retirement. Stored once, on the first episode.
                    store.get(candidate.episode_ids[0]).meta.setdefault("audit_items", []).append(item.to_dict())
                items = new_items + old_items
                gate_count = len(new_items)
                after_start = ledger.spent("audit")
                gaps_before, exhausted = measure_gaps(handle, items, state.world_seed, night, k, max_new)
                audit_exhausted = audit_exhausted or exhausted
                control = _control_items(auditable, sampler, cfg.commit.control_items, rng("audit_control"))
                control_before, exhausted = measure_gaps(handle, control, state.world_seed, night, k, max_new)
                audit_exhausted = audit_exhausted or exhausted
                if arm.anchors != "none" and held_part.anchor:
                    specs = _anchor_specs(held_part.anchor, arm, older, cfg.select.lexical_min)
                    heldout_anchor_targets, exhausted = build_targets(handle, specs, cfg.teacher, rng("audit_anchors"))
                    audit_exhausted = audit_exhausted or exhausted
                    drift_before, exhausted = drift_by_target(handle, heldout_anchor_targets)
                    audit_exhausted = audit_exhausted or exhausted
                audit_after_need = ledger.spent("audit") - after_start

        # ---- Stage 1c: score candidates in a seeded mixed order, so that a short budget falls evenly on singles,
        # pairs and anchors rather than on whatever came last.
        if arm.selection:
            with ledger.stage("signals"):
                scorable = _shuffled(
                    [c for c in train_cands + pool_train if c.context and c.distractor is not None], rng("select_order")
                )
                values, exhausted = chd_safe(
                    handle,
                    [(c.query, c.context, c.distractor) for c in scorable],
                    k,
                    max_new,
                    rng("select"),
                    cache,
                    cfg.signals.chunk,
                )
                signals_exhausted = signals_exhausted or exhausted
            value_by_id = {id(c): v for c, v in zip(scorable, values)}
            everything = train_cands + pool_train
            train_part = partition(
                everything, [value_by_id.get(id(c)) for c in everything], thresholds, pair_thresholds
            )
            count_source = "chd"
        else:
            n_anchor, count_source = _uniform_anchor_count(cfg, night, "train", len(pool_train))
            train_part = uniform_partition(train_cands, pool_train, n_anchor, rng("uniform_train"))
        if "signals" in ledger.caps:
            ledger.close_stage("signals")
        report.signals = {
            "null_size": thresholds.null_size,
            "pair_null_size": pair_thresholds.null_size,
            "threshold_source": thresholds.source,
            "pair_threshold_source": pair_thresholds.source,
            "integrate_threshold": thresholds.integrate,
            "anchor_threshold": thresholds.anchor,
            "gate_threshold": thresholds.gate,
            "gate_deferred": len(deferred),
            "gate_rate": len(deferred) / len(new_episodes) if new_episodes else 0.0,
            "exhausted": signals_exhausted,
        }
        total_candidates = len(train_cands) + len(pool_train)
        unscored = train_part.discarded["unscored"]
        report.selection = {
            "train": train_part.summary(),
            "heldout": held_part.summary(),
            "count_source": count_source,
            "unscored": unscored,
            "unscored_fraction": unscored / total_candidates if total_candidates else 0.0,
        }

    # ---- Stage 2, first half: all teacher targets, frozen for the rest of the night.
    integrate_targets: list[TokenTargets] = []
    anchor_targets: list[TokenTargets] = []
    teacher_exhausted = False
    if "teacher" in ledger.caps:
        teacher_rng = rng("teacher")
        with ledger.stage("teacher"):
            if arm.teacher_kl:
                specs = [
                    TargetSpec(c.query, c.context, "committed", "integrate", c.context_ids)
                    for c in train_part.integrate
                ]
                integrate_targets, exhausted = build_targets(handle, specs, cfg.teacher, teacher_rng, cache)
                teacher_exhausted = teacher_exhausted or exhausted
            if arm.anchors != "none":
                specs = _anchor_specs(train_part.anchor, arm, older, cfg.select.lexical_min)
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

    # ---- Stage 2, second half: training. Fresh is the only thing that moves.
    hard = hard_pool_examples(arm, store, handle, new_ids, train_part.integrate, cfg.select.hard_boost, state.heldout)
    _release_surplus(ledger, hard, integrate_targets, anchor_targets, cfg, audit_after_need)
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

    # ---- Stage 5: audit and commit, on the same items and targets as before training.
    decision = Decision(accepted=True, reasons=["no_audit"])
    gaps_after: list[Optional[float]] = [None] * len(items)
    if arm.audit:
        with handle.banks("committed"), ledger.stage("audit"):
            gaps_after, exhausted = measure_gaps(handle, items, state.world_seed, night, k, max_new)
            audit_exhausted = audit_exhausted or exhausted
            control_after, exhausted = measure_gaps(handle, control, state.world_seed, night, k, max_new)
            audit_exhausted = audit_exhausted or exhausted
            drift_after, exhausted = drift_by_target(handle, heldout_anchor_targets)
            audit_exhausted = audit_exhausted or exhausted
        mean_before, mean_after = drift_change(drift_before, drift_after)
        increase = mean_after - mean_before if mean_before is not None and mean_after is not None else None
        decision = decide(
            gaps_before[:gate_count],
            gaps_after[:gate_count],
            increase,
            cfg.commit,
            arm.transactional,
            audit_exhausted,
            control_before,
            control_after,
        )
        decision.anchor_drift_before, decision.anchor_drift_after = mean_before, mean_after
    if not decision.accepted:
        delta.restore(snapshot)
    trained = set(train_result.trained_episode_ids)
    trained.update((report.compaction.get("repair") or {}).get("trained_episode_ids", []))
    report.episodes_trained = len(trained)
    if decision.accepted:
        # Only what reached a completed optimizer step, not all that could have been sampled: when the budget
        # ends early, the rest was never trained. An episode already marked keeps its first night.
        for episode_id in sorted(trained):
            meta = store.get(episode_id).meta
            if "consolidated_night" not in meta:
                meta["consolidated_night"] = night
                report.episodes_marked_consolidated += 1
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


def _gate_probe(
    episode: Episode, candidates: Sequence[Candidate], sampler: DistractorSampler, generator: torch.Generator
) -> list[tuple[str, str]]:
    """One seeded (query, distractor context) for the write gate, from this episode's training queries."""
    own = [c for c in candidates if c.episode_ids == (episode.id,)]
    if not own:
        return []
    candidate = own[int(torch.randint(len(own), (1,), generator=generator))]
    distractor, _ = draw_distractor(candidate, sampler, generator)
    return [(candidate.query, distractor)] if distractor is not None else []


def _anchor_specs(
    candidates: Sequence[Candidate], arm: ArmConfig, older: Sequence[Episode], lexical_min: float
) -> list[TargetSpec]:
    specs = []
    for candidate in candidates:
        banks, context = anchor_teacher(candidate.query, arm, older, lexical_min)
        specs.append(TargetSpec(candidate.query, context, banks, "anchor"))
    return specs
