"""The night end to end, and the invariants. No assertion here is about learning quality."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tiny import TinyTokenizer, TinyWorld, fresh_gpt2_state, small_config, tiny_gpt2

import llmsleep
from llmsleep.arms import ARMS, ArmConfig, get_arm
from llmsleep.config import CommitConfig, NightConfig, TeacherConfig, TrainConfig, derive_seed, make_generator
from llmsleep.delta import COMMITTED, DeltaModel
from llmsleep.handle import ModelHandle
from llmsleep.ledger import ComputeLedger
from llmsleep.memory import Episode, EpisodicStore
from llmsleep.phases.commit import AuditItem, decide, measure_drift, measure_gaps, update_episodes
from llmsleep.phases.consolidate import (
    HardExample,
    TargetSpec,
    build_self_sampled_targets,
    build_targets,
    compute_loss,
    hard_examples,
    train_loop,
)
from llmsleep.phases.night import SleepPressure, State, active_stage_fractions, anchor_teacher, run_night
from llmsleep.phases.queries import (
    Candidate,
    ModelQueryGen,
    TemplateQueryGen,
    WorldLike,
    generate_singles,
    normalize_query,
    parse_questions,
    split_heldout,
)
from llmsleep.phases.select import uniform_partition
from llmsleep.phases.wake import observe

PREREG_ARMS = {
    "B0_frozen",
    "B1_rag",
    "B2_seq",
    "B3_replay",
    "B4_replay_kl",
    "S_full",
    "S_oracleq",
    "S_noselect",
    "S_nokl",
    "S_chain",
    "S_nocompact",
    "S_nopair",
    "S_nocommit",
    "S_bm25",
}
SRC = Path(llmsleep.__file__).resolve().parent


def _probe_logits(state: State) -> torch.Tensor:
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    return state.handle.net(input_ids=ids).logits.detach().clone()


def _select_all_config(arm_name: str = "S_oracleq") -> NightConfig:
    """Templates, no CHD filter: guarantees integrate targets and a real training path on a tiny model."""
    return small_config(replace(get_arm(arm_name), selection=False, write_gate=False))


# ---- arms and invariant 5 --------------------------------------------------------------------------


def test_every_prereg_arm_exists_and_is_named_consistently():
    assert set(ARMS) == PREREG_ARMS
    for name, arm in ARMS.items():
        assert arm.name == name


def test_arm_flags_match_the_prereg_table():
    assert not get_arm("B0_frozen").train and not get_arm("B1_rag").train
    assert get_arm("B1_rag").open_book_eval and not get_arm("B0_frozen").open_book_eval
    assert get_arm("B2_seq").hard_pool == "new"
    assert get_arm("B3_replay").hard_pool == "all_real" and not get_arm("B3_replay").self_sampled_kl
    assert get_arm("B4_replay_kl").self_sampled_kl and get_arm("B4_replay_kl").anchors == "none"
    full = get_arm("S_full")
    assert full.query_source == "model" and full.selection and full.teacher_kl and full.anchors == "germline"
    assert full.pair_contexts and full.compact and full.transactional and full.retriever == "oracle"
    assert get_arm("S_oracleq").query_source == "template"
    assert not get_arm("S_noselect").selection and get_arm("S_noselect").teacher_kl
    assert get_arm("S_nokl").selection and not get_arm("S_nokl").teacher_kl
    assert get_arm("S_chain").anchors == "chain"
    assert not get_arm("S_nocompact").compact
    assert not get_arm("S_nopair").pair_contexts
    assert not get_arm("S_nocommit").transactional and get_arm("S_nocommit").audit
    assert get_arm("S_bm25").retriever == "bm25"


def test_no_arm_names_outside_arms_py():
    pattern = re.compile(r"""(==|!=|\bin\b)\s*["'](B\d_|S_)""")
    names = re.compile("|".join(re.escape(n) for n in PREREG_ARMS))
    for path in SRC.rglob("*.py"):
        if path.name == "arms.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not pattern.search(text), f"arm branching in {path}"
        assert not names.search(text), f"arm name in {path}"


def test_invalid_arm_flags_are_rejected():
    with pytest.raises(ValueError):
        ArmConfig(name="x", query_source="wat")
    with pytest.raises(ValueError):
        ArmConfig(name="x", transactional=True)
    with pytest.raises(ValueError):
        ArmConfig(name="x", query_source="model")
    with pytest.raises(ValueError):
        ArmConfig(name="x", train=True)


def test_config_round_trips_and_hash_is_stable():
    cfg = small_config(get_arm("S_full"))
    again = NightConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.to_dict() == cfg.to_dict()
    assert again.config_hash() == cfg.config_hash()
    assert NightConfig.from_dict({"arm": "B3_replay"}).arm == get_arm("B3_replay")
    with pytest.raises(KeyError):
        NightConfig.from_dict({"arm": "B3_replay", "nope": 1})


def test_replay_arms_spend_their_whole_allowance_on_training():
    cfg = small_config(get_arm("B3_replay"))
    assert set(active_stage_fractions(cfg.arm, cfg)) == {"train"}
    full = small_config(get_arm("S_full"))
    assert set(active_stage_fractions(full.arm, full)) == {"synthesis", "signals", "teacher", "train", "repair", "audit"}
    nokl = small_config(get_arm("S_nokl"))
    assert "teacher" in active_stage_fractions(nokl.arm, nokl)  # Anchors still need targets.


# ---- seeds ----------------------------------------------------------------------------------------


def test_seeds_are_stable_and_separate_purposes():
    assert derive_seed(1, 2, "x") == derive_seed(1, 2, "x")
    assert len({derive_seed(1, 2, "x"), derive_seed(1, 2, "y"), derive_seed(1, 3, "x"), derive_seed(2, 2, "x")}) == 4
    before = torch.get_rng_state()
    make_generator(1, 2, "x").get_state()
    assert torch.equal(torch.get_rng_state(), before)


# ---- full nights over all arms -------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", sorted(PREREG_ARMS))
def test_every_arm_runs_two_nights_within_budget_and_keeps_base_weights(arm_name, tokenizer, world):
    cfg = small_config(get_arm(arm_name))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    fingerprint = state.handle.delta.base_fingerprint()
    for night in range(2):
        report = run_night(state, cfg)
        assert report.night == night and report.arm == arm_name
        assert report.ledger["spent"] <= cfg.budget.night_tokens
        json.dumps(report.to_dict())  # JSON-able, for results.jsonl.
    assert state.night == 2
    assert state.handle.delta.base_fingerprint() == fingerprint  # Invariant 1.
    assert state.handle.delta.active == (COMMITTED, "fresh")  # Selection restored.
    if not cfg.arm.write_gate:
        assert len(state.store) == 4


def test_full_and_noselect_allocate_the_same_ledger_total(tokenizer):
    tokens = 80_000
    full_cfg = small_config(get_arm("S_full"), night_tokens=tokens)
    noselect_cfg = small_config(get_arm("S_noselect"), night_tokens=tokens)
    full, noselect = ComputeLedger(), ComputeLedger()
    full.begin_night(tokens, active_stage_fractions(full_cfg.arm, full_cfg))
    noselect.begin_night(tokens, active_stage_fractions(noselect_cfg.arm, noselect_cfg))
    assert sum(full.caps.values()) == sum(noselect.caps.values()) == tokens
    # Same split, with the signal allocation credited to training. Slack of a few tokens is float flooring.
    assert "signals" not in noselect.caps
    for stage in ("synthesis", "teacher", "repair", "audit"):
        assert abs(full.caps[stage] - noselect.caps[stage]) <= 1
    assert abs(noselect.caps["train"] - (full.caps["train"] + full.caps["signals"])) <= 4

    # And on real nights: closing stages moves tokens around but never changes the total.
    for arm_name in ("S_full", "S_noselect"):
        cfg = small_config(replace(get_arm(arm_name), query_source="template"), night_tokens=tokens)
        report = run_night(fresh_gpt2_state(cfg, TinyWorld(), tokenizer), cfg)
        assert sum(report.ledger["caps"].values()) == tokens
        assert report.ledger["spent"] <= tokens


def test_noselect_credit_buys_more_training_tokens_when_steps_are_uncapped(tokenizer):
    tokens = 20_000
    spent = {}
    for arm_name in ("S_full", "S_noselect"):
        cfg = small_config(replace(get_arm(arm_name), query_source="template"), night_tokens=tokens)
        cfg.train = TrainConfig(lr=1e-3, hard_batch=2, kl_batch=2, anchor_batch=2)  # max_steps None.
        report = run_night(fresh_gpt2_state(cfg, TinyWorld(), tokenizer), cfg)
        spent[arm_name] = report.ledger["forward"]["train"] + report.ledger["backward"]["train"]
    assert spent["S_noselect"] > spent["S_full"]


def test_noselect_draws_integrate_and_anchors_at_matched_counts_without_chd(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    def forbidden(*args, **kwargs):
        raise AssertionError("no CHD may be computed for selection in this arm")

    monkeypatch.setattr(night_module, "chd_safe", forbidden)
    monkeypatch.setattr(night_module, "null_distribution", forbidden)
    cfg = small_config(replace(get_arm("S_noselect"), query_source="template"))
    cfg.select.matched_selection = {
        "0": {"train": {"integrate": 2, "anchor": 1}, "heldout": {"integrate": 1, "anchor": 0}}
    }
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.selection["count_source"] == "matched"
    assert report.selection["train"]["integrate"] == 2 and report.selection["train"]["anchor"] == 1
    assert report.selection["heldout"]["integrate"] == 1 and report.selection["heldout"]["anchor"] == 0
    assert report.teacher["integrate_sequences"] == 2  # samples=1 in the small config.
    assert report.teacher["anchor_sequences"] == 1


def test_noselect_without_matched_counts_says_it_fell_back(tokenizer, world):
    cfg = small_config(replace(get_arm("S_noselect"), query_source="template"))
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.selection["count_source"] == "fraction"
    assert report.selection["train"]["anchor"] > 0


def test_uniform_partition_is_near_only_for_anchors_disjoint_and_counts_everything():
    near = [Candidate(f"Who is Mira{i}?", ("e",), ("M",), context=f"Mira{i} keeps kites.") for i in range(4)]
    far = [Candidate(f"Name colour {i}?", ("e",), ("M",), context="Mira keeps kites.") for i in range(4)]
    empty = [Candidate("Who?", ("e",), ("M",))]
    out = uniform_partition(near + far + empty, 3, 2, 0.5, torch.Generator().manual_seed(1))
    assert len(out.integrate) == 3 and len(out.anchor) == 2
    assert all(c in near for c in out.anchor)
    assert not ({id(c) for c in out.integrate} & {id(c) for c in out.anchor})
    assert dict(out.discarded) == {"no_context": 1, "not_drawn": 3}
    # Deterministic, and a shortfall is visible rather than an error.
    again = uniform_partition(near + far + empty, 3, 2, 0.5, torch.Generator().manual_seed(1))
    assert [c.query for c in again.integrate] == [c.query for c in out.integrate]
    short = uniform_partition(far, 0, 2, 0.5, torch.Generator().manual_seed(1))
    assert short.anchor == [] and short.discarded["not_drawn"] == 4


def test_bm25_arm_takes_teacher_context_from_retrieval(tokenizer, world):
    cfg = small_config(replace(get_arm("S_bm25"), query_source="template", selection=False, write_gate=False))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.teacher["integrate_sequences"] > 0
    assert report.train["steps"] > 0


@pytest.mark.parametrize("arm_name", ["B0_frozen", "B1_rag"])
def test_no_training_arms_spend_nothing_and_leave_the_model_alone(arm_name, tokenizer, world):
    cfg = small_config(get_arm(arm_name))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    before = _probe_logits(state)
    report = run_night(state, cfg)
    assert report.ledger["spent"] == 0
    assert report.delta["delta_rank"] == 0
    assert torch.equal(_probe_logits(state), before)
    assert len(state.store) == 2  # Retrieval by day still needs the episodes.


def test_b2_trains_only_on_tonights_episodes_and_b3_on_all_real(tokenizer, world):
    cfg = small_config(get_arm("B2_seq"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    assert state.handle.delta.stats()["delta_rank"] > 0
    run_night(state, cfg)
    assert len(state.store) == 4
    assert state.store.get("n0-e0").meta["consolidated_night"] == 0
    assert state.store.get("n1-e0").meta["consolidated_night"] == 1


def test_b4_uses_germline_self_samples(tokenizer, world):
    cfg = small_config(get_arm("B4_replay_kl"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.teacher["anchor_sequences"] == cfg.teacher.self_samples
    assert report.teacher["integrate_sequences"] == 0
    assert report.ledger["forward"]["teacher"] > 0


def test_self_sampled_targets_match_germline_exactly(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    _set_nonzero_fresh(delta)
    delta.merge_fresh_into_committed()  # So germline and committed really differ.
    targets, exhausted = build_self_sampled_targets(
        handle, TeacherConfig(self_samples=2, self_max_new_tokens=3, top_k=4), torch.Generator().manual_seed(1)
    )
    assert not exhausted and targets and all(t.kind == "self" for t in targets)
    target = targets[0]
    assert target.prompt_ids == handle.start_ids()
    assert target.top_idx.shape == (len(target.cont_ids), 4)
    assert target.res_logp.shape == (len(target.cont_ids),)
    # Germline as student on the stored target: KL is zero. The committed model's is not.
    with handle.banks("germline"):
        zero, _ = measure_drift(handle, targets)
    with handle.banks("committed"):
        moved, _ = measure_drift(handle, targets)
    assert zero == pytest.approx(0.0, abs=1e-4)
    assert moved > 1e-4


def _set_nonzero_fresh(delta: DeltaModel, seed: int = 3, magnitude: float = 1.0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in delta.layers.values():
            layer.fresh_A.copy_(torch.randn(layer.fresh_A.shape, generator=generator) * magnitude)
            layer.fresh_B.copy_(torch.randn(layer.fresh_B.shape, generator=generator) * magnitude)


def test_s_full_night_with_templates_exercises_every_stage(tokenizer, world):
    cfg = _select_all_config()
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.synthesis["train_candidates"] > 0 and report.synthesis["heldout_candidates"] > 0
    assert report.synthesis["pair_candidates"] > 0  # Episodes of a night share Zor.
    assert report.teacher["integrate_sequences"] > 0
    assert report.train["steps"] > 0
    assert report.compaction["compacted"]
    assert report.decision["paired_items"] > 0
    assert set(report.ledger["forward"]) >= {"synthesis", "teacher", "train", "audit"}


def test_selection_arm_reports_all_candidates(tokenizer, world):
    cfg = small_config(get_arm("S_oracleq"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    train = report.selection["train"]
    assert report.signals["threshold_source"] in {"quantile", "extremes", "none"}
    assert train["integrate"] + train["anchor"] + sum(train["discarded"].values()) == report.synthesis["train_candidates"]
    assert report.wake and all("reason" in w for w in report.wake)


def test_compaction_bounds_rank_and_nocompact_does_not(tokenizer, world):
    bounded = _select_all_config()
    state = fresh_gpt2_state(bounded, world, tokenizer)
    bounded.commit = CommitConfig(min_gap_drop=-1e9, max_anchor_drift=1e9)  # Always accept.
    for _ in range(2):
        run_night(state, bounded)
    layers = state.handle.delta.stats()["layers"]
    assert max(entry["rank"] for entry in layers.values()) <= bounded.compact.max_rank

    free = small_config(get_arm("B3_replay"))
    other = fresh_gpt2_state(free, TinyWorld(), tokenizer)
    for _ in range(2):
        run_night(other, free)
    layers = other.handle.delta.stats()["layers"]
    # Two exact merges of rank-4 fresh banks, on layers wide enough to hold them.
    assert max(entry["rank"] for entry in layers.values()) > free.model.rank


# ---- invariant 2: teachers frozen; invariant 1: only fresh moves ------------------------------------


def test_training_moves_fresh_only_and_cannot_alter_banks_or_targets(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    _set_nonzero_fresh(delta, magnitude=0.1)
    delta.merge_fresh_into_committed()
    delta.reset_fresh(torch.Generator().manual_seed(1))
    fingerprint = delta.base_fingerprint()
    committed = {n: l.frozen[COMMITTED].clone() for n, l in delta.layers.items()}

    teacher = TeacherConfig(samples=1, max_new_tokens=3, top_k=4)
    specs = [TargetSpec("Who?", "ctx", "committed", "integrate")]
    targets, _ = build_targets(handle, specs, teacher, torch.Generator().manual_seed(2))
    assert all(p.grad is None for p in delta.fresh_parameters())  # No gradients from teacher passes.
    stored = [(t.top_idx.clone(), t.top_logp.clone(), t.res_logp.clone()) for t in targets]
    hard = hard_examples(handle, [Episode("e", "Mira keeps kites.", 0)])

    fresh_before = [p.detach().clone() for p in delta.fresh_parameters()]
    result = train_loop(
        handle, hard, targets, [], TrainConfig(lr=1e-2, hard_batch=1, kl_batch=1), torch.Generator().manual_seed(3), max_steps=3
    )
    assert result.steps == 3
    assert any(not torch.equal(a, b) for a, b in zip(fresh_before, delta.fresh_parameters()))
    assert delta.base_fingerprint() == fingerprint
    for name, layer in delta.layers.items():
        assert torch.equal(layer.frozen[COMMITTED].U, committed[name].U)
        assert torch.equal(layer.frozen[COMMITTED].S, committed[name].S)
        assert torch.equal(layer.frozen[COMMITTED].V, committed[name].V)
    for target, (idx, top, res) in zip(targets, stored):
        assert torch.equal(target.top_idx, idx) and torch.equal(target.top_logp, top) and torch.equal(target.res_logp, res)


def test_teacher_and_student_use_the_same_tokens_but_different_prompts(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    teacher = TeacherConfig(samples=2, max_new_tokens=3, top_k=4)
    targets, _ = build_targets(
        handle, [TargetSpec("Who?", "ctx", "committed", "integrate")], teacher, torch.Generator().manual_seed(2)
    )
    assert len(targets) == 2
    closed = handle.encode_prompt(handle.format("Who?", None))
    for target in targets:
        assert target.prompt_ids == closed  # The student is closed-book.
        assert target.top_idx.size(0) == len(target.cont_ids)


def test_compute_loss_lines_up_rows_with_answer_tokens(tokenizer):
    """With the teacher's own prompt as the student's, KL at the teacher's weights must be zero."""
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    teacher = TeacherConfig(samples=2, max_new_tokens=4, top_k=4)
    targets, _ = build_targets(
        handle, [TargetSpec("Who?", None, "germline", "anchor")], teacher, torch.Generator().manual_seed(4)
    )
    hard = [HardExample([2, 3], [4, 5]), HardExample([6], [7])]
    with handle.banks("germline"):
        loss, parts = compute_loss(handle, hard, [], targets, TrainConfig())
        # Worked out by hand from the same logits.
        by_hand = -handle.sequence_logprob([[2, 3], [6]], [[4, 5], [7]]).sum() / 3
    assert parts["anchor"] == pytest.approx(0.0, abs=1e-4)
    assert parts["hard"] == pytest.approx(float(by_hand), abs=1e-4)
    assert float(loss) == pytest.approx(parts["hard"] + parts["anchor"], abs=1e-4)


def test_training_runs_under_gradient_checkpointing(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    hard = hard_examples(handle, [Episode("e", "Mira keeps kites.", 0)])
    result = train_loop(
        handle,
        hard,
        [],
        [],
        TrainConfig(hard_batch=1, gradient_checkpointing=True, accum_steps=2),
        torch.Generator().manual_seed(1),
        max_steps=1,
    )
    assert result.steps == 1 and not handle.net.training


# ---- invariant 3: generated text never becomes a fact --------------------------------------------


def test_generated_queries_are_candidates_never_episodes(tokenizer, world):
    cfg = small_config(get_arm("S_full"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    world_ids = {e.id for e in world.episodes(0)}
    assert {e.id for e in state.store.all_real()} <= world_ids
    assert all(e.provenance == "real" for e in state.store.all_real())


def test_a_world_offering_generated_text_is_refused(tokenizer):
    class Bad(TinyWorld):
        def episodes(self, night):
            return [Episode("g", "Generated.", night, provenance="generated")]

    cfg = small_config(get_arm("B3_replay"))
    state = fresh_gpt2_state(cfg, Bad(), tokenizer)
    with pytest.raises(ValueError):
        run_night(state, cfg)


def test_parse_questions():
    text = "1. Who is Mira?\n- What does she keep?\nA statement.\nWhat?\n2) Who is Mira ?\nwho  is MIRA?\n"
    assert parse_questions(text) == ["Who is Mira?", "What does she keep?", "Who is Mira ?"]


def test_model_query_gen_is_charged_and_returns_strings(tokenizer, world):
    from llmsleep.config import QueryConfig

    ledger = ComputeLedger()
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu", ledger, batch_size=2)
    gen = ModelQueryGen(handle, QueryConfig(model_samples=2, model_max_new_tokens=3))
    queries = gen.generate(world.episodes(0)[:1], torch.Generator().manual_seed(0))
    assert all(isinstance(q, str) for q in queries)
    assert ledger.unmetered_forward > 0


def test_tiny_world_satisfies_the_protocol(world):
    assert isinstance(world, WorldLike)


def test_template_gen_uses_world_and_caps(world):
    gen = TemplateQueryGen(world, per_episode=2)
    episodes = world.episodes(0)
    assert len(gen.generate(episodes[:1], torch.Generator())) == 2
    assert len(gen.generate(episodes[:2], torch.Generator())) == 1
    candidates, exhausted = generate_singles(gen, episodes, torch.Generator())
    assert not exhausted and len(candidates) == 4
    assert candidates[0].context == episodes[0].text  # Own text until retrieval runs.


# ---- point 9: held-out split ----------------------------------------------------------------------


def _candidates(queries) -> list[Candidate]:
    return [Candidate(q, ("e",), ("X",)) for q in queries]


def test_split_is_disjoint_deterministic_and_order_independent():
    queries = [f"Question number {i}?" for i in range(100)]
    train, held, names = split_heldout(_candidates(queries), 5, 0.2)
    assert train and held
    assert not ({c.query for c in train} & {c.query for c in held})
    assert len(train) + len(held) == 100
    again = split_heldout(_candidates(reversed(queries)), 5, 0.2)
    assert again[2] == names
    # A query's side does not depend on what else is present.
    subset = split_heldout(_candidates(queries[:50]), 5, 0.2)
    assert {normalize_query(q) for q in queries[:50]} & names == subset[2]
    # Another world seed gives another split.
    assert split_heldout(_candidates(queries), 6, 0.2)[2] != names


def test_split_treats_normalised_duplicates_as_one_query():
    queries = [f"Question {i}?" for i in range(40)]
    doubled = _candidates(queries) + _candidates([q.upper() + "  " for q in queries])
    train, held, _ = split_heldout(doubled, 5, 0.3)
    train_norm = {normalize_query(c.query) for c in train}
    held_norm = {normalize_query(c.query) for c in held}
    assert not train_norm & held_norm


def test_split_on_tiny_sets_still_holds_one_out():
    for seed in range(20):  # Whatever the hash does, a two-query set must hold one out.
        train, held, _ = split_heldout(_candidates(["One?", "Two?"]), seed, 0.01)
        assert len(held) >= 1
    assert split_heldout(_candidates(["One?"]), 0, 0.0)[1] == []
    assert split_heldout([], 0, 0.5) == ([], [], set())


def test_known_heldout_stays_held_out():
    known = {normalize_query("Never train me?")}
    train, held, names = split_heldout(_candidates(["Never train me?"]), 0, 0.0, known)
    assert train == [] and len(held) == 1 and names == known


def test_hard_labels_skip_qa_forms_of_heldout_queries(tokenizer):
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu")
    episode = Episode("e", "Mira keeps kites.", 0, meta={"qa": [["Who keeps kites?", "Mira"], ["What?", "Kites"]]})
    assert len(hard_examples(handle, [episode])) == 3  # The text plus two QA forms.
    kept = hard_examples(handle, [episode], {normalize_query("WHO keeps  kites?")})
    assert len(kept) == 2
    assert all(handle.encode_prompt(handle.format("Who keeps kites?", None)) != ex.prompt_ids for ex in kept)


def test_a_night_never_trains_on_heldout_queries(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    seen: list[str] = []
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        seen.extend(spec.query for spec in specs if spec.kind == "integrate")
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    cfg = _select_all_config()
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    assert seen
    assert state.heldout
    assert not ({normalize_query(q) for q in seen} & state.heldout)


# ---- wake ----------------------------------------------------------------------------------------


def _wake_handle(tokenizer):
    return ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu", batch_size=4)


def test_write_gate_stores_or_drops_by_threshold(tokenizer):
    handle = _wake_handle(tokenizer)
    args = (2, 2, torch.Generator().manual_seed(1))
    episode = Episode("e", "Mira keeps kites.", 0)
    probes = [("Who?", "Tides rise at noon daily.")]
    store = EpisodicStore()
    assert not observe(handle, store, episode, probes, 1e9, *args).stored  # Anything is below this.
    assert "e" not in store
    result = observe(handle, store, episode, probes, -1e9, *args)
    assert result.stored and result.reason == "novel" and "e" in store


def test_write_gate_fails_open_and_bypasses(tokenizer):
    handle = _wake_handle(tokenizer)
    args = (2, 2, torch.Generator().manual_seed(1))
    probes = [("Who?", "Tides rise.")]
    assert observe(handle, EpisodicStore(), Episode("a", "t", 0), [], 1e9, *args).reason == "unscored"
    assert observe(handle, EpisodicStore(), Episode("b", "t", 0), probes, None, *args).reason == "unscored"
    assert observe(handle, EpisodicStore(), Episode("c", "t", 0), probes, 1e9, *args, bypass=True).reason == "bypass"
    superseding = Episode("d", "t", 0, supersedes=["a"])
    assert observe(handle, EpisodicStore(), superseding, probes, 1e9, *args).reason == "supersession"


def test_write_gate_with_no_budget_stores_rather_than_drops(tokenizer):
    ledger = ComputeLedger()
    ledger.begin_night(0, {"signals": 1.0})
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu", ledger)
    store = EpisodicStore()
    with ledger.stage("signals"):
        result = observe(
            handle, store, Episode("a", "t", 0), [("Who?", "x")], 1e9, 2, 2, torch.Generator().manual_seed(1)
        )
    assert result.stored and result.reason == "unscored"


# ---- anchors ---------------------------------------------------------------------------------------


def test_anchor_source_follows_flags_only():
    mira = Episode("m", "Mira keeps kites.", 0, entities=["Mira"])
    germline, chain = get_arm("S_full"), get_arm("S_chain")
    assert anchor_teacher("Name a colour?", germline, [mira], 0.5) == ("germline", None)
    assert anchor_teacher("Name a colour?", chain, [mira], 0.5) == ("committed", None)
    # Touching consolidated knowledge: the committed model with the episode in context, in both arms.
    assert anchor_teacher("Who is Mira?", germline, [mira], 0.5) == ("committed", "Mira keeps kites.")
    assert anchor_teacher("Who is Mira?", chain, [mira], 0.5) == ("committed", "Mira keeps kites.")


def test_germline_anchor_target_is_unchanged_by_committed_delta(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    cfg = TeacherConfig(samples=1, max_new_tokens=3, top_k=4)
    spec = [TargetSpec("Who?", None, "germline", "anchor")]
    before, _ = build_targets(handle, spec, cfg, torch.Generator().manual_seed(1))
    _set_nonzero_fresh(delta)
    delta.merge_fresh_into_committed()
    after, _ = build_targets(handle, spec, cfg, torch.Generator().manual_seed(1))
    assert before[0].cont_ids == after[0].cont_ids
    assert torch.equal(before[0].top_idx, after[0].top_idx) and torch.equal(before[0].top_logp, after[0].top_logp)


# ---- commit, rollback, retirement ------------------------------------------------------------------


def test_decide_is_a_pure_rule_with_reasons():
    cfg = CommitConfig(max_anchor_drift=0.5, min_gap_drop=0.1)
    assert decide([2.0], [1.0], 0.1, cfg, True).reasons == ["passed"]
    assert decide([2.0], [1.0], 0.1, cfg, True).accepted
    rejected = decide([2.0], [1.95], 0.1, cfg, True)
    assert not rejected.accepted and rejected.reasons == ["gap_did_not_fall"]
    assert decide([2.0], [1.0], 0.9, cfg, True).reasons == ["anchor_drift"]
    assert decide([], [], None, cfg, True).reasons == ["no_audit_data"]
    assert not decide([], [], None, cfg, True).accepted
    assert decide([None], [1.0], None, cfg, True).reasons == ["no_audit_data"]  # Unpaired items do not count.
    both = decide([2.0], [3.0], 0.9, cfg, True)
    assert both.reasons == ["gap_did_not_fall", "anchor_drift"]
    open_door = decide([2.0], [3.0], 0.9, cfg, False)
    assert open_door.accepted and "gap_did_not_fall" in open_door.reasons
    partial = decide([2.0, None], [1.0, 9.0], None, cfg, True, audit_exhausted=True)
    assert partial.accepted and partial.paired_items == 1 and partial.audit_exhausted


def test_rejected_night_rolls_back_to_exact_outputs(tokenizer, world):
    cfg = _select_all_config()
    cfg.commit = CommitConfig(min_gap_drop=1e9)  # No gap can fall this far.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    before = _probe_logits(state)
    state_before = state.handle.delta.state_dict()
    report = run_night(state, cfg)
    assert not report.decision["accepted"]
    assert report.train["steps"] > 0  # It really trained first.
    assert torch.equal(_probe_logits(state), before)
    with state.handle.banks("committed"):
        assert torch.equal(_probe_logits(state), before)
    for name, layer in state.handle.delta.layers.items():
        assert layer.frozen[COMMITTED].rank == state_before[name]["frozen"][COMMITTED]["S"].numel()
    assert report.delta["delta_rank"] == 0
    assert all(e.meta.get("consolidated_night") is None for e in state.store.all_real())
    assert len(state.store) == 2  # The real episodes are kept even so.


def test_rollback_after_an_earlier_accepted_night_returns_to_that_night(tokenizer, world):
    cfg = _select_all_config()
    cfg.commit = CommitConfig(min_gap_drop=-1e9, max_anchor_drift=1e9)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    with state.handle.banks("committed"):
        accepted = _probe_logits(state)
    assert state.handle.delta.stats()["delta_rank"] > 0
    cfg.commit = CommitConfig(min_gap_drop=1e9)
    report = run_night(state, cfg)
    assert not report.decision["accepted"]
    with state.handle.banks("committed"):
        assert torch.equal(_probe_logits(state), accepted)


def test_nocommit_accepts_what_commit_would_reject(tokenizer, world):
    cfg = small_config(replace(get_arm("S_nocommit"), selection=False, write_gate=False, query_source="template"))
    cfg.commit = CommitConfig(min_gap_drop=1e9)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.decision["accepted"]
    assert "gap_did_not_fall" in report.decision["reasons"]
    assert report.delta["delta_rank"] > 0


def test_common_random_numbers_make_gaps_repeatable(tokenizer):
    handle = _wake_handle(tokenizer)
    items = [AuditItem("Who?", "Mira keeps kites.", "Tides rise.", ("e",)), AuditItem("What?", "A", "B", ("e",))]
    first, _ = measure_gaps(handle, items, 1, 0, 2, 3)
    second, _ = measure_gaps(handle, list(reversed(items)), 1, 0, 2, 3)
    assert first == list(reversed(second))  # Per-item seeds, so order does not matter.


def test_audit_items_round_trip():
    item = AuditItem("q", "c", "d", ("a", "b"))
    assert AuditItem.from_dict(json.loads(json.dumps(item.to_dict()))) == item


def test_retirement_needs_k_consecutive_accepted_low_nights():
    store = EpisodicStore()
    store.add(Episode("e", "t", 0))
    items = [AuditItem("q", "c", "d", ("e",))]
    assert update_episodes(store, items, [0.1], 1.0, True, 2) == []
    assert store.get("e").meta["low_gap_nights"] == 1 and store.get("e").meta["last_gap"] == 0.1
    # A rejected night neither counts nor resets, though it does update the measured gap.
    assert update_episodes(store, items, [0.2], 1.0, False, 2) == []
    assert store.get("e").meta["low_gap_nights"] == 1 and store.get("e").meta["last_gap"] == 0.2
    # A high gap resets.
    assert update_episodes(store, items, [5.0], 1.0, True, 2) == []
    assert store.get("e").meta["low_gap_nights"] == 0
    update_episodes(store, items, [0.1], 1.0, True, 2)
    assert update_episodes(store, items, [0.1], 1.0, True, 2) == ["e"]
    assert store.get("e").status == "retired"
    # Without a threshold (no null), nothing retires.
    store.add(Episode("f", "t", 0))
    items = [AuditItem("q", "c", "d", ("f",))]
    assert update_episodes(store, items, [0.0], None, True, 1) == [] and store.get("f").status == "active"


def test_sleep_pressure_is_the_mean_gap_over_active_current_episodes():
    store = EpisodicStore()
    assert SleepPressure.compute(store) is None
    store.add(Episode("a", "t", 0, meta={"last_gap": 1.0}))
    store.add(Episode("b", "t", 0, meta={"last_gap": 3.0}))
    store.add(Episode("c", "t", 0))  # Not yet measured.
    assert SleepPressure.compute(store) == pytest.approx(2.0)
    store.retire("b")
    assert SleepPressure.compute(store) == pytest.approx(1.0)
    store.add(Episode("a2", "t", 1, supersedes=["a"]))
    assert SleepPressure.compute(store) is None


def test_old_audit_items_are_remeasured_on_later_nights(tokenizer, world):
    cfg = _select_all_config()
    cfg.commit = CommitConfig(min_gap_drop=-1e9, max_anchor_drift=1e9)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    assert any(e.meta.get("audit_items") for e in state.store.all_real())
    assert any("last_gap" in e.meta for e in state.store.all_real())
    report = run_night(state, cfg)
    assert report.pressure is not None


def test_drift_is_measured_on_heldout_anchors_and_can_reject(tokenizer):
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    targets, _ = build_targets(
        handle,
        [TargetSpec("Who?", None, "germline", "anchor")],
        TeacherConfig(samples=2, max_new_tokens=3, top_k=4),
        torch.Generator().manual_seed(1),
    )
    _set_nonzero_fresh(delta)
    delta.merge_fresh_into_committed()
    with handle.banks("committed"):
        drift, exhausted = measure_drift(handle, targets)
    assert drift > 0 and not exhausted
    assert decide([2.0], [1.0], drift, CommitConfig(max_anchor_drift=drift / 2), True).reasons == ["anchor_drift"]


# ---- invariant 4: budget ----------------------------------------------------------------------------


@pytest.mark.parametrize("night_tokens", [0, 50, 500, 3000])
@pytest.mark.parametrize("arm_name", ["S_oracleq", "S_full", "B3_replay", "B4_replay_kl"])
def test_exhaustion_stops_each_phase_and_leaves_a_partial_record(arm_name, night_tokens, tokenizer, world):
    cfg = small_config(get_arm(arm_name), night_tokens=night_tokens)
    cfg.train = TrainConfig(hard_batch=2, kl_batch=2, anchor_batch=2)  # Run until the tokens are gone.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.ledger["spent"] <= night_tokens
    if night_tokens <= 50:  # Less than one batch of any stage. Larger budgets may finish a planned loop exactly.
        assert report.ledger["exhausted"]
    assert state.night == 1
    json.dumps(report.to_dict())


def test_stage_caps_stop_one_stage_starving_the_rest(tokenizer, world):
    cfg = small_config(get_arm("S_oracleq"), night_tokens=4000)
    cfg.signals.k = 50  # Signals would eat everything if allowed.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    ledger = report.ledger
    assert ledger["forward"]["signals"] + ledger["backward"]["signals"] <= ledger["caps"]["signals"]
    assert "signals" in ledger["exhausted"]
    assert report.signals["exhausted"]
    assert report.train["steps"] > 0  # Training still got its allocation.


def test_training_counts_forward_and_backward_tokens(tokenizer, world):
    cfg = small_config(get_arm("B3_replay"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.ledger["forward"]["train"] == report.ledger["backward"]["train"] > 0


def test_evaluation_after_a_night_must_be_marked_exempt(tokenizer, world):
    cfg = small_config(get_arm("B2_seq"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    spent = state.ledger.spent()
    assert spent == report.ledger["spent"]
    with pytest.raises(RuntimeError):
        state.handle.sequence_logprob([[2, 3]], [[4]])
    with state.ledger.exempt():
        state.handle.sequence_logprob([[2, 3]], [[4]])
    assert state.ledger.spent() == spent and state.ledger.exempt_forward == 3


# ---- invariant 7: resumable and deterministic -------------------------------------------------------


def _adapters_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for name in a:
        if not (torch.equal(a[name]["A"], b[name]["A"]) and torch.equal(a[name]["B"], b[name]["B"])):
            return False
        if a[name]["frozen"].keys() != b[name]["frozen"].keys():
            return False
        for bank in a[name]["frozen"]:
            for key in "USV":
                if not torch.equal(a[name]["frozen"][bank][key], b[name]["frozen"][bank][key]):
                    return False
    return True


def test_same_seed_gives_identical_runs_regardless_of_global_rng(tokenizer):
    cfg = _select_all_config()

    def run(global_seed: int):
        torch.manual_seed(global_seed)
        state = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
        report = run_night(state, cfg)
        return state, report

    first, report_a = run(1)
    second, report_b = run(2)
    assert report_a.to_dict() == report_b.to_dict()
    assert _adapters_equal(first.handle.delta.state_dict(), second.handle.delta.state_dict())


def test_world_seed_changes_the_run(tokenizer):
    cfg = _select_all_config()
    a = fresh_gpt2_state(cfg, TinyWorld(), tokenizer, world_seed=1)
    b = fresh_gpt2_state(cfg, TinyWorld(), tokenizer, world_seed=2)
    run_night(a, cfg)
    run_night(b, cfg)
    assert not _adapters_equal(a.handle.delta.state_dict(), b.handle.delta.state_dict())


@pytest.mark.parametrize("arm_name", ["S_oracleq", "B4_replay_kl"])
def test_resume_after_save_equals_uninterrupted_run(arm_name, tokenizer, tmp_path):
    cfg = small_config(get_arm(arm_name))
    straight = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    run_night(straight, cfg)
    run_night(straight, cfg)

    first = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    run_night(first, cfg)
    first.save(str(tmp_path))

    handle = fresh_gpt2_state(cfg, TinyWorld(), tokenizer).handle  # A new process would rebuild from scratch.
    resumed = State.load(str(tmp_path), handle, TinyWorld())
    assert resumed.night == 1 and resumed.world_seed == first.world_seed
    assert resumed.heldout == first.heldout
    run_night(resumed, cfg)

    assert resumed.night == straight.night
    assert _adapters_equal(resumed.handle.delta.state_dict(), straight.handle.delta.state_dict())
    assert [e.__dict__ for e in resumed.store.all_real()] == [e.__dict__ for e in straight.store.all_real()]


def test_checkpoint_is_atomic_and_latest_points_at_a_complete_one(tokenizer, tmp_path):
    cfg = small_config(get_arm("B2_seq"))
    state = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    run_night(state, cfg)
    first = state.save(str(tmp_path))
    # A crash during a later save leaves its temp directory behind; LATEST still names the good one.
    (tmp_path / ".tmp-night-00002").mkdir()
    (tmp_path / ".tmp-night-00002" / "adapters.pt").write_text("partial")
    handle = fresh_gpt2_state(cfg, TinyWorld(), tokenizer).handle
    assert State.load(str(tmp_path), handle, TinyWorld()).night == 1
    assert (tmp_path / "LATEST").read_text() == Path(first).name
    assert not (tmp_path / "LATEST.tmp").exists()
    assert sorted(p.name for p in Path(first).iterdir()) == ["adapters.pt", "state.json", "store.json"]


def test_load_refuses_a_different_base_model(tmp_path):
    tok = TinyTokenizer()
    cfg = small_config(get_arm("B0_frozen"))
    state = fresh_gpt2_state(cfg, TinyWorld(), tok)
    state.save(str(tmp_path))
    other = State.create(tiny_gpt2(tok.vocab_size, seed=99), tok, TinyWorld(), 7, cfg)
    with pytest.raises(ValueError):
        State.load(str(tmp_path), other.handle, TinyWorld())
