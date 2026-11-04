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
from llmsleep.config import CommitConfig, NightConfig, QueryConfig, TeacherConfig, TrainConfig, derive_seed, make_generator
from llmsleep.delta import COMMITTED, DeltaModel
from llmsleep.handle import ModelHandle
from llmsleep.ledger import ComputeLedger, LedgerExhausted
from llmsleep.memory import DistractorSampler, Episode, EpisodicStore
from llmsleep.phases.commit import AuditItem, decide, drift_by_target, drift_change, measure_drift, measure_gaps, update_episodes
from llmsleep.phases.consolidate import (
    HardExample,
    TargetSpec,
    build_self_sampled_targets,
    build_targets,
    compute_loss,
    hard_examples,
    train_loop,
)
from llmsleep.phases.night import (
    SleepPressure,
    State,
    _gate_probe,
    active_stage_fractions,
    anchor_teacher,
    hard_pool_examples,
    run_night,
)
from llmsleep.phases.queries import (
    Candidate,
    build_anchor_pool,
    choose_pairs,
    is_chain,
    pair_stats,
    ModelQueryGen,
    PairQueryGen,
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
    "B3_pairs",
    "B4_replay_kl",
    "B3_derived",
    "B4q",
    "B4q_idk",
    "B4q_compact",
    "B4q_r8",
    "S_core",
    "S_coreq",
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


def _oracle_full() -> ArmConfig:
    """S_full's selection and audit paths over oracle templates, since a tiny model's own questions rarely parse."""
    return replace(get_arm("S_full"), query_source="template")


def _select_all_config() -> NightConfig:
    """Templates, no CHD filter: guarantees integrate targets and a real training path on a tiny model."""
    return small_config(replace(_oracle_full(), selection=False, write_gate=False))


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
    assert get_arm("B3_pairs").replay_pairs and not get_arm("B3_replay").replay_pairs
    assert get_arm("B3_pairs").query_source == "none" and get_arm("B3_pairs").anchors == "none"
    assert get_arm("B4_replay_kl").self_sampled_kl and get_arm("B4_replay_kl").anchors == "none"
    full = get_arm("S_full")
    assert full.query_source == "model" and full.selection and full.teacher_kl and full.anchors == "germline"
    assert full.pair_contexts and full.compact and full.transactional and full.retriever == "oracle"
    core = get_arm("S_core")
    assert core.query_source == "model" and core.teacher_kl and core.anchors == "germline" and core.pair_contexts
    assert core.compact and core.drift_commit and core.transactional and core.passive_audit
    assert not core.selection and not core.write_gate and not core.audit  # A2.1: nothing here reads CHD.
    assert core.hard_pool == get_arm("B3_replay").hard_pool and core.anchor_count == "fixed"
    assert get_arm("S_oracleq").query_source == "template"
    assert not get_arm("S_noselect").selection and get_arm("S_noselect").teacher_kl
    assert not get_arm("S_nokl").teacher_kl
    assert get_arm("S_chain").anchors == "chain"
    assert not get_arm("S_nocompact").compact
    assert not get_arm("S_nopair").pair_contexts
    assert not get_arm("S_nocommit").transactional and get_arm("S_nocommit").drift_commit
    assert get_arm("S_bm25").retriever == "bm25"


@pytest.mark.parametrize(
    "arm_name, changed",
    [
        ("S_oracleq", {"query_source": "template"}),
        ("S_nokl", {"teacher_kl": False}),
        ("S_chain", {"anchors": "chain"}),
        ("S_nocompact", {"compact": False}),
        ("S_nopair", {"pair_contexts": False}),
        ("S_nocommit", {"transactional": False}),
        ("S_bm25", {"retriever": "bm25"}),
    ],
)
def test_each_ablation_is_s_core_with_one_flag_changed(arm_name, changed):
    """So that a difference from S_core attributes to that flag. The exempt log costs time only, so S_core alone has it."""
    expected = replace(get_arm("S_core"), name=arm_name, passive_audit=False, **changed)
    assert get_arm(arm_name) == expected


def test_noselect_stays_paired_with_s_full():
    assert get_arm("S_noselect") == replace(get_arm("S_full"), name="S_noselect", selection=False, write_gate=False)


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
    with pytest.raises(ValueError):
        ArmConfig(name="x", anchor_count="wat")
    with pytest.raises(ValueError):
        replace(get_arm("S_full"), anchor_count="fixed")
    with pytest.raises(ValueError):
        replace(get_arm("S_core"), audit=True)
    with pytest.raises(ValueError):
        replace(get_arm("S_core"), anchors="none")
    with pytest.raises(ValueError):
        ArmConfig(name="x", passive_audit=True)


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


def test_noselect_spends_its_signal_credit_on_teaching_when_steps_are_uncapped(tokenizer):
    # A1.4: KL runs on all usable queries, so the credit may go to teacher targets as well as training.
    tokens = 20_000
    teaching = {}
    for arm_name in ("S_full", "S_noselect"):
        cfg = small_config(replace(get_arm(arm_name), query_source="template"), night_tokens=tokens)
        cfg.train = TrainConfig(lr=1e-3, hard_batch=2, kl_batch=2, anchor_batch=2)  # max_steps None.
        report = run_night(fresh_gpt2_state(cfg, TinyWorld(), tokenizer), cfg)
        teaching[arm_name] = sum(
            report.ledger[kind].get(stage, 0) for kind in ("forward", "backward") for stage in ("teacher", "train")
        )
    assert teaching["S_noselect"] > teaching["S_full"]


def test_noselect_trains_on_all_queries_and_draws_matched_anchors_without_chd(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    def forbidden(*args, **kwargs):
        raise AssertionError("no CHD may be computed for selection in this arm")

    monkeypatch.setattr(night_module, "chd_safe", forbidden)
    monkeypatch.setattr(night_module, "null_distribution", forbidden)
    cfg = small_config(replace(get_arm("S_noselect"), query_source="template"))
    cfg.select.matched_selection = {
        "1": {"train": {"integrate": 99, "anchor": 1}, "heldout": {"integrate": 99, "anchor": 0}}
    }
    state = fresh_gpt2_state(cfg, world, tokenizer)
    assert run_night(state, cfg).selection["count_source"] == "fraction"  # Night 0 is not in the matched record.
    report = run_night(state, cfg)
    assert report.selection["count_source"] == "matched"
    assert report.selection["train"]["anchor"] == 1 and report.selection["heldout"]["anchor"] == 0
    # A1.4: KL on all usable generated queries, however many the other arm integrated.
    assert report.selection["train"]["integrate"] == report.synthesis["train_candidates"] > 0
    assert report.teacher["integrate_sequences"] == report.synthesis["train_candidates"]  # samples=1 in the small config.
    assert report.teacher["anchor_sequences"] == 1


def test_uniform_partition_integrates_everything_usable_and_draws_anchors_from_the_pool():
    tonight = [Candidate(f"Who is Mira{i}?", ("e",), ("M",), context="Mira keeps kites.") for i in range(3)]
    empty = [Candidate("Who?", ("e",), ("M",))]
    pool = [Candidate(f"Who is Zed{i}?", ("o",), ("Z",), context="Zed sails.", pool=True) for i in range(4)]
    out = uniform_partition(tonight + empty, pool, 2, torch.Generator().manual_seed(1))
    assert out.integrate == tonight
    assert len(out.anchor) == 2 and all(c in pool for c in out.anchor)
    assert dict(out.discarded) == {"no_context": 1, "pool_not_drawn": 2}
    again = uniform_partition(tonight + empty, pool, 2, torch.Generator().manual_seed(1))
    assert [c.query for c in again.anchor] == [c.query for c in out.anchor]
    assert uniform_partition(tonight, [], 5, torch.Generator()).anchor == []


def _anchor_queries(monkeypatch) -> list:
    """Record every anchor spec a night builds, on its way to `build_targets`."""
    import llmsleep.phases.night as night_module

    seen: list = []
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        seen.append([spec for spec in specs if spec.kind == "anchor"])
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    return seen


@pytest.mark.parametrize("selection", [True, False])
def test_anchors_never_name_tonights_entities(selection, tokenizer, world, monkeypatch):
    seen = _anchor_queries(monkeypatch)
    arm = replace(_oracle_full(), selection=selection, write_gate=selection)
    cfg = small_config(arm)
    cfg.select.matched_selection = None if selection else {"1": {"train": {"anchor": 3}, "heldout": {"anchor": 1}}}
    cfg.select.anchor_null_q = 1.0  # Everything at or below the null's top: most of the pool, on a noise model.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    seen.clear()
    run_night(state, cfg)
    specs = [spec for batch in seen for spec in batch]
    tonight = [e for episode in world.episodes(1) for e in episode.entities]
    for spec in specs:
        assert not any(name.casefold() in spec.query.casefold() for name in tonight)
    if not selection:
        assert specs  # The draw above is guaranteed; the CHD path depends on the noise model.


def test_anchor_pool_excludes_tonight_and_uses_earlier_queries_and_older_entities():
    tok = TinyTokenizer()
    world = TinyWorld()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)
    store = EpisodicStore()
    store.add(Episode("old", "Zed sails.", 0, entities=["Zed"], meta={"queries": ["Who is Zed?", "Who sails with Ivo?"]}))
    store.add(Episode("linked", "Ivo sails.", 0, entities=["Ivo"], meta={"queries": ["Who is Ivo?"]}))
    store.add(Episode("gone", "Una sails.", 0, entities=["Una"], meta={"queries": ["Who is Una?"]}))
    store.add(Episode("una2", "Una rows.", 1, entities=["Una2"], supersedes=["gone"]))
    store.add(Episode("new", "Ivo rows.", 1, entities=["Ivo"], meta={"queries": ["Who rows?"]}))
    pool, exhausted = build_anchor_pool(
        handle, store, world, {"new"}, QueryConfig(model_samples=1, model_max_new_tokens=2), torch.Generator().manual_seed(1)
    )
    assert not exhausted
    queries = {c.query for c in pool}
    assert "Who is Zed?" in queries
    assert "Who sails with Ivo?" not in queries  # It names an entity of tonight's.
    assert "Who is Ivo?" not in queries  # Its episode is about one.
    assert "Who is Una?" not in queries  # Superseded.
    assert "Who rows?" not in queries  # Tonight's own.
    assert all(c.pool and c.context for c in pool)


def test_anchor_pool_generates_over_older_episodes_and_the_distractor_pool(monkeypatch):
    import llmsleep.phases.queries as queries_module

    sources: list[str] = []

    class Recording:
        def __init__(self, handle, cfg):
            pass

        def generate(self, episodes, generator):
            sources.append(episodes[0].id)
            return [f"About {episodes[0].id}?"]

    monkeypatch.setattr(queries_module, "ModelQueryGen", Recording)
    store = EpisodicStore()
    store.add(Episode("old", "Zed sails.", 0, entities=["Zed"]))
    store.add(Episode("new", "Ivo rows.", 1, entities=["Ivo"]))
    pool, _ = build_anchor_pool(None, store, TinyWorld(), {"new"}, QueryConfig(), torch.Generator().manual_seed(1))
    assert "old" in sources and "new" not in sources
    assert any(s.startswith("filler") for s in sources)
    assert any(c.episode_ids == ("old",) for c in pool)
    capped, _ = build_anchor_pool(
        None, store, TinyWorld(), {"new"}, QueryConfig(anchor_pool_max=2), torch.Generator().manual_seed(1)
    )
    assert len(capped) == 2


def test_drift_is_tested_as_the_nights_increase_not_the_cumulative_level(tokenizer, world):
    """C1: a big earlier delta must not freeze later nights. With lr 0 the increase is exactly zero."""
    arm = replace(_oracle_full(), selection=False, write_gate=False, compact=False)
    cfg = small_config(arm)
    cfg.commit = CommitConfig(min_gap_drop=-1e9, max_anchor_drift=1e-6)
    cfg.select.matched_selection = {"1": {"train": {"anchor": 1}, "heldout": {"anchor": 1}}}
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    _set_nonzero_fresh(state.handle.delta, magnitude=0.3)  # A large committed delta from "earlier nights".
    state.handle.delta.merge_fresh_into_committed()
    cfg.train = TrainConfig(lr=0.0, hard_batch=2, kl_batch=2, anchor_batch=2, max_steps=2)
    report = run_night(state, cfg)
    assert report.decision["anchor_drift_before"] > 1e-6  # An absolute test would have rejected this.
    assert report.decision["anchor_drift"] == pytest.approx(0.0, abs=1e-9)
    assert report.decision["accepted"]


def test_drift_change_compares_only_targets_measured_twice():
    before = [(2.0, 2), None, (9.0, 3)]
    after = [(4.0, 2), (1.0, 1), None]
    assert drift_change(before, after) == (1.0, 2.0)
    assert drift_change([], []) == (None, None)


def test_drift_by_target_splits_rows_by_length(tokenizer):
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu", batch_size=2)
    cfg = TeacherConfig(samples=3, max_new_tokens=4, top_k=4)
    targets, _ = build_targets(handle, [TargetSpec("Who?", None, "germline", "anchor")], cfg, torch.Generator().manual_seed(1))
    values, exhausted = drift_by_target(handle, targets)
    assert not exhausted and [n for _, n in values] == [len(t.cont_ids) for t in targets]
    mean, _ = measure_drift(handle, targets)
    assert mean == pytest.approx(sum(s for s, _ in values) / sum(n for _, n in values))


def test_control_must_fall_less_than_the_gap():
    cfg = CommitConfig(max_anchor_drift=1.0, min_gap_drop=0.0)
    assert decide([2.0], [1.0], None, cfg, True, False, [2.0], [1.5]).reasons == ["passed"]  # 1.0 > 0.5.
    failing = decide([2.0], [1.0], None, cfg, True, False, [2.0], [0.5])  # All answers moved: 1.0 < 1.5.
    assert failing.reasons == ["gap_did_not_fall"] and failing.control_drop == pytest.approx(1.5)
    assert decide([2.0], [1.0], None, cfg, True, False, [], []).reasons == ["no_control_data"]
    assert decide([2.0], [1.0], None, cfg, True, False, [None], [1.0]).reasons == ["no_control_data"]


def test_audit_flags_with_an_independent_seed_and_measures_a_control(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    calls: list = []
    original = night_module.measure_gaps

    def recording(handle, items, seed, night, k, m, purpose="audit"):
        calls.append((purpose, list(items)))
        return original(handle, items, seed, night, k, m, purpose)

    monkeypatch.setattr(night_module, "measure_gaps", recording)
    cfg = small_config(_oracle_full())
    run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert calls[0][0] == "audit_flag"
    audited = [items for purpose, items in calls if purpose == "audit"]
    assert len(audited) == 4  # Items and control, before and after.
    assert audited[0] == audited[2] and audited[1] == audited[3]  # Identical sets at both times.
    for item in audited[1]:  # The control: two contexts, neither about the query's entities.
        assert item.episode_ids == ()
        names = re.findall(r"Mira\d+", item.query)
        assert not any(name in item.context or name in item.distractor for name in names)


def test_flagging_and_before_measurements_use_different_samples(tokenizer):
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu", batch_size=4)
    items = [AuditItem("Who?", "Mira keeps kites.", "Tides rise.", ("e",))]
    first, _ = measure_gaps(handle, items, 1, 0, 4, 6)
    same, _ = measure_gaps(handle, items, 1, 0, 4, 6)
    other, _ = measure_gaps(handle, items, 1, 0, 4, 6, "audit_flag")
    assert first == same and first != other


def test_old_items_keep_their_slots_under_the_audit_cap(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    calls: list = []
    original = night_module.measure_gaps

    def recording(handle, items, seed, night, k, m, purpose="audit"):
        calls.append(list(items))
        return original(handle, items, seed, night, k, m, purpose)

    monkeypatch.setattr(night_module, "measure_gaps", recording)
    cfg = _select_all_config()
    cfg.commit = CommitConfig(min_gap_drop=-1e9, max_anchor_drift=1e9, heldout_frac=0.5, audit_max_items=2, old_item_share=0.5)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    calls.clear()
    run_night(state, cfg)
    assert len(calls[0]) <= 2
    assert any(item.episode_ids[0].startswith("n0") for item in calls[0])


def test_write_gate_probes_one_query_per_episode(world):
    episode = world.episodes(0)[0]
    candidates = [
        Candidate(q, (episode.id,), tuple(episode.entities), context=episode.text)
        for q in world.template_queries(episode)
    ]
    sampler = DistractorSampler(world.distractor_pool())
    assert len(_gate_probe(episode, candidates, sampler, torch.Generator().manual_seed(1))) == 1
    assert _gate_probe(world.episodes(0)[1], candidates, sampler, torch.Generator()) == []


def test_scoring_mixes_singles_and_pairs_and_reports_unscored(tokenizer, monkeypatch):
    import llmsleep.phases.night as night_module

    class ManyPairQueries(TinyWorld):
        def pair_template_queries(self, a, b):
            return [f"Together {i}, how many kites have {a.entities[0]} and {b.entities[0]}?" for i in range(8)]

    seen: list = []
    original = night_module.chd_safe

    def recording(handle, items, *args, **kwargs):
        seen.append(list(items))
        return original(handle, items, *args, **kwargs)

    monkeypatch.setattr(night_module, "chd_safe", recording)
    cfg = small_config(_oracle_full())
    cfg.queries.per_episode = 8
    report = run_night(fresh_gpt2_state(cfg, ManyPairQueries(), tokenizer), cfg)
    order = [query for query, _, _ in seen[0]]
    pair_positions = [i for i, q in enumerate(order) if q.startswith("Together")]
    assert 0 < len(pair_positions) < len(order)
    # Appended last, and a short budget would have fallen on all of them.
    assert pair_positions != list(range(len(order) - len(pair_positions), len(order)))
    assert report.selection["unscored"] == report.selection["train"]["discarded"].get("unscored", 0)
    assert 0.0 <= report.selection["unscored_fraction"] <= 1.0


def test_unused_stage_tokens_go_to_training(tokenizer, world):
    cfg = small_config(replace(_oracle_full(), selection=False, write_gate=False), night_tokens=40_000)
    cfg.train = TrainConfig(lr=1e-3, hard_batch=2, kl_batch=2, anchor_batch=2)  # max_steps None.
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.ledger["unspent"] == cfg.budget.night_tokens - report.ledger["spent"]
    assert report.ledger["unspent"] < 0.25 * cfg.budget.night_tokens


def test_b4_spends_its_whole_teacher_allocation_on_fresh_samples(tokenizer, world):
    cfg = small_config(get_arm("B4_replay_kl"), night_tokens=20_000)
    cfg.teacher.self_samples = None
    state = fresh_gpt2_state(cfg, world, tokenizer)
    first = run_night(state, cfg)
    assert first.teacher["anchor_sequences"] > 4 and not first.teacher["exhausted"]
    spent = first.ledger["forward"]["teacher"] + first.ledger["backward"]["teacher"]
    assert spent > 0.9 * (20_000 * cfg.budget.fractions["teacher"] / (cfg.budget.fractions["teacher"] + cfg.budget.fractions["train"]))


def test_self_sampling_without_a_budget_needs_a_count(tokenizer):
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu")
    with pytest.raises(ValueError):
        build_self_sampled_targets(handle, TeacherConfig(self_samples=None), torch.Generator())


def test_b3_pairs_replays_real_concatenated_pairs_under_the_cap_with_no_model_or_signal(tokenizer, world):
    cfg = small_config(get_arm("B3_pairs"))
    cfg.queries.max_pairs = 1
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.synthesis["pairs"]["chosen"] == 1
    assert set(report.ledger["forward"]) == {"train"}
    plain = fresh_gpt2_state(small_config(get_arm("B3_replay")), TinyWorld(), tokenizer)
    run_night(plain, small_config(get_arm("B3_replay")))
    arm = get_arm("B3_pairs")
    with_pairs = hard_pool_examples(arm, state.store, state.handle, set(), [], 2, set())
    without = hard_pool_examples(get_arm("B3_replay"), plain.store, plain.handle, set(), [], 2, set())
    assert len(with_pairs) == len(without) + 1
    a, b = world.episodes(0)
    assert tokenizer.decode(with_pairs[-1].cont_ids) == f"{a.text}\n{b.text}"
    assert with_pairs[-1].episode_ids == (a.id, b.id)


def test_a_superseded_partner_leaves_pair_replay(tokenizer, world):
    cfg = small_config(get_arm("B3_pairs"))
    state = fresh_gpt2_state(cfg, world, tokenizer)
    run_night(state, cfg)
    count = len(hard_pool_examples(cfg.arm, state.store, state.handle, set(), [], 2, set()))
    a, b = world.episodes(0)
    state.store.add(Episode("update", "New fact.", 1, supersedes=[b.id]))
    # One pair, b's three forms, gone; the update's single form, added.
    assert len(hard_pool_examples(cfg.arm, state.store, state.handle, set(), [], 2, set())) == count - 1 - 3 + 1


def test_write_gate_defers_rather_than_drops_and_reports_the_rate(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    original = night_module.thresholds_from_null
    monkeypatch.setattr(
        night_module,
        "thresholds_from_null",
        lambda values, select_cfg: replace(original(values, select_cfg), gate=1e9),
    )
    cfg = small_config(_oracle_full())
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert all(e.status == "deferred" for e in state.store.all_real()) and len(state.store) == 2
    assert report.signals["gate_rate"] == 1.0
    assert report.episodes_trained > 0  # Still replayed.
    assert report.synthesis["train_candidates"] == 0  # But no teacher targets tonight.
    assert all(w["deferred"] for w in report.wake)


# ---- mutations 1, 3 and 4 of the review ------------------------------------------------------------


def test_stored_targets_are_the_teachers_logits_under_its_context(tokenizer):
    """Mutation 1: `handle.format(spec.query, None)` for the teacher passes shape tests and teaches nothing."""
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu", batch_size=4)
    cfg = TeacherConfig(samples=1, max_new_tokens=3, top_k=4)
    spec = TargetSpec("Who keeps kites?", "Mira keeps nine kites.", "committed", "integrate")
    target = build_targets(handle, [spec], cfg, torch.Generator().manual_seed(1))[0][0]

    def log_probs(context):
        prompt = handle.encode_prompt(handle.format(spec.query, context))
        with handle.banks("committed"):
            _, logits = next(handle.iter_answer_logits([prompt], [target.cont_ids]))
        return torch.log_softmax(logits, dim=-1)

    with_context = log_probs(spec.teacher_context)
    assert torch.allclose(with_context.gather(-1, target.top_idx), target.top_logp, atol=1e-5)
    assert not torch.allclose(log_probs(None).gather(-1, target.top_idx), target.top_logp, atol=1e-4)
    assert target.prompt_ids == handle.encode_prompt(handle.format(spec.query, None))  # The student stays closed.


def _names(text: str) -> set[str]:
    return set(re.findall(r"Mira\d+", text))


def test_selection_scores_each_query_with_its_own_context_on_the_sampling_side(tokenizer, world, monkeypatch):
    """Mutation 3: `(query, distractor, context)` at the call site is the wrong KL, and night tests only count."""
    import llmsleep.phases.night as night_module

    seen: list = []
    original = night_module.chd_safe

    def recording(handle, items, *args, **kwargs):
        seen.extend(items)
        return original(handle, items, *args, **kwargs)

    monkeypatch.setattr(night_module, "chd_safe", recording)
    cfg = small_config(_oracle_full())
    run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    named = [item for item in seen if _names(item[0])]  # A model-generated pool query might name nobody.
    assert named
    for query, context, distractor in named:
        assert _names(query) & _names(context), (query, context)
        assert not _names(query) & _names(distractor), (query, distractor)


def test_audit_measures_each_item_with_its_own_context_on_the_sampling_side(tokenizer, world, monkeypatch):
    """Mutation 3, at `commit.py`: the positional call to `chd`."""
    import llmsleep.phases.commit as commit_module

    calls: list = []
    original = commit_module.chd

    def recording(handle, queries, ctx_e, ctx_d, *args, **kwargs):
        calls.append((queries[0], ctx_e[0], ctx_d[0]))
        return original(handle, queries, ctx_e, ctx_d, *args, **kwargs)

    monkeypatch.setattr(commit_module, "chd", recording)
    items = [AuditItem("Who is Mira7?", "Mira7 keeps kites.", "Tides rise.", ("e",))]
    measure_gaps(ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu"), items, 1, 0, 2, 2)
    assert calls == [("Who is Mira7?", "Mira7 keeps kites.", "Tides rise.")]


def test_the_student_trains_with_committed_and_fresh_active(tokenizer):
    """Mutation 4: `delta.banks("fresh")` would learn while ignoring earlier nights, then merge."""
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    _set_nonzero_fresh(delta, magnitude=0.1)
    delta.merge_fresh_into_committed()
    seen: list = []
    original = handle.forward_train

    def recording(batch):
        seen.append(delta.active)
        return original(batch)

    handle.forward_train = recording
    hard = hard_examples(handle, [Episode("e", "Mira keeps kites.", 0)])
    train_loop(handle, hard, [], [], TrainConfig(hard_batch=1), torch.Generator().manual_seed(1), max_steps=2)
    assert seen == [(COMMITTED, "fresh")] * 2


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
    cfg = small_config(_oracle_full())
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    train = report.selection["train"]
    assert report.signals["threshold_source"] in {"quantile", "extremes", "none"}
    total = report.synthesis["train_candidates"] + report.synthesis["anchor_pool_train"]
    assert train["integrate"] + train["anchor"] + sum(train["discarded"].values()) == total
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


def _pair_store(old: int, new: int) -> EpisodicStore:
    store = EpisodicStore()
    for i in range(old):
        store.add(Episode(f"o{i}", f"Old {i}.", 0, entities=["Hub"]))
    for i in range(new):
        store.add(Episode(f"n{i}", f"New {i}.", 1, entities=["Hub"]))
    return store


class _EchoGen:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def usable_pair(self, a, b) -> bool:
        return True

    def generate(self, episodes, generator):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise LedgerExhausted("synthesis", 1, 0)
        self.calls += 1
        return ["Q " + "+".join(e.id for e in episodes) + "?"]


def test_pair_choice_prefers_links_across_nights():
    gen = PairQueryGen(_EchoGen(), max_pairs=3)
    candidates, exhausted = gen.generate(_pair_store(4, 2), {"n0", "n1"}, torch.Generator().manual_seed(1))
    assert not exhausted and len(candidates) == 3
    for candidate in candidates:  # Eight cross-night pairs compete for three places; n0+n1 must not get one.
        assert sum(episode_id.startswith("o") for episode_id in candidate.episode_ids) == 1
    assert gen.stats == {
        "available": 9,
        "chains_available": 0,
        "cross_night_available": 8,
        "chosen": 3,
        "chains_chosen": 0,
        "cross_night_chosen": 3,
    }
    # Old episodes pair only through a new one, and same-night pairs fill the places that remain.
    gen = PairQueryGen(_EchoGen(), max_pairs=9)
    candidates, _ = gen.generate(_pair_store(4, 2), {"n0", "n1"}, torch.Generator().manual_seed(1))
    assert len(candidates) == 9 and candidates[-1].episode_ids == ("n0", "n1")


def test_chains_come_first_in_the_pair_budget_and_are_reported():
    store = EpisodicStore()
    # Hub is the value of a and the subject of b: a chain. c and d merely share a subject with b or a value with a.
    store.add(Episode("a", "A works at Hub.", 0, entities=["A", "Hub"]))
    store.add(Episode("b", "Hub is in Oslo.", 1, entities=["Hub", "Oslo"]))
    store.add(Episode("c", "Hub makes kites.", 0, entities=["Hub"]))
    store.add(Episode("d", "D works at Hub.", 1, entities=["D", "Hub"]))
    assert is_chain(store.get("a"), store.get("b")) and is_chain(store.get("b"), store.get("a"))
    assert not is_chain(store.get("a"), store.get("d"))  # Same value: no second hop.
    assert not is_chain(store.get("b"), store.get("c"))  # Same subject.
    pairs = store.pairs_sharing_entity()
    chosen = choose_pairs(pairs, 1, torch.Generator().manual_seed(1))
    assert is_chain(*chosen[0])
    stats = pair_stats(pairs, chosen)
    assert stats["chains_chosen"] == 1 and stats["available"] == len(pairs) > 1


def test_oracle_pairs_are_filtered_before_the_cap():
    class OnlyOne(TinyWorld):
        def pair_template_queries(self, a, b):
            return ["Two hops?"] if {a.id, b.id} == {"o3", "n0"} else []

    world = OnlyOne()
    gen = PairQueryGen(TemplateQueryGen(world, per_episode=4), max_pairs=1)
    candidates, _ = gen.generate(_pair_store(4, 1), {"n0"}, torch.Generator().manual_seed(1))
    assert [c.episode_ids for c in candidates] == [("n0", "o3")]  # No slot wasted on a pair with no query.
    assert gen.stats["available"] == 1


def test_pair_choice_samples_rather_than_truncates():
    def chosen(seed: int) -> list[tuple[str, ...]]:
        candidates, _ = PairQueryGen(_EchoGen(), max_pairs=2).generate(
            _pair_store(0, 6), {f"n{i}" for i in range(6)}, torch.Generator().manual_seed(seed)
        )
        return [c.episode_ids for c in candidates]

    assert chosen(3) == chosen(3)  # Seeded.
    seen = {tuple(chosen(seed)) for seed in range(200)}
    assert len(seen) > 1
    # Sorted truncation would always give the same two; sampling reaches the end of the sort as well.
    assert any(("n4", "n5") in picks for picks in seen)


def test_pair_exhaustion_keeps_the_cross_night_pairs_already_done():
    gen = PairQueryGen(_EchoGen(fail_after=1), max_pairs=5)
    candidates, exhausted = gen.generate(_pair_store(2, 2), {"n0", "n1"}, torch.Generator().manual_seed(2))
    assert exhausted and len(candidates) == 1
    assert sum(episode_id.startswith("o") for episode_id in candidates[0].episode_ids) == 1


def test_a_night_reports_its_pair_choice(tokenizer, world):
    cfg = _select_all_config()
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.synthesis["pairs"]["available"] >= report.synthesis["pairs"]["chosen"] > 0


def test_train_loop_records_only_episodes_in_completed_steps(tokenizer):
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu")
    hard = hard_examples(handle, [Episode(i, f"Text of {i}.", 0) for i in "abc"])
    result = train_loop(handle, hard, [], [], TrainConfig(hard_batch=1), torch.Generator().manual_seed(1), max_steps=1)
    assert result.steps == 1 and len(result.trained_episode_ids) == 1
    json.dumps(result.summary())
    assert result.summary()["trained_episode_ids"] == sorted(result.trained_episode_ids)


def test_a_step_dropped_mid_accumulation_teaches_nothing(tokenizer):
    ledger = ComputeLedger()
    ledger.begin_night(17, {"train": 1.0})
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu", ledger)
    # Each example: start token + 4 characters + EOS = 6 tokens, so 12 charged. The second micro-batch cannot pay.
    hard = hard_examples(handle, [Episode("a", "aaaa", 0), Episode("b", "bbbb", 0)])
    with ledger.stage("train"):
        result = train_loop(
            handle, hard, [], [], TrainConfig(hard_batch=1, accum_steps=2), torch.Generator().manual_seed(1)
        )
    assert result.exhausted and result.steps == 0 and result.trained_episode_ids == set()


def test_integrate_targets_credit_their_episodes_and_anchors_do_not(tokenizer):
    handle = ModelHandle(DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2), tokenizer, "cpu", batch_size=4)
    cfg = TeacherConfig(samples=1, max_new_tokens=2, top_k=4)
    generator = torch.Generator().manual_seed(1)
    integrate, _ = build_targets(handle, [TargetSpec("Who?", "ctx", "committed", "integrate", ("taught",))], cfg, generator)
    anchor, _ = build_targets(handle, [TargetSpec("What?", None, "germline", "anchor")], cfg, generator)
    assert integrate[0].episode_ids == ("taught",)
    result = train_loop(
        handle, [], integrate, anchor, TrainConfig(kl_batch=1, anchor_batch=1), generator, max_steps=1
    )
    assert result.trained_episode_ids == {"taught"}


def test_only_episodes_that_received_steps_are_marked_consolidated(tokenizer, world):
    cfg = small_config(get_arm("B3_replay"))
    cfg.train = TrainConfig(hard_batch=1, max_steps=1)  # One example, from one of the night's two episodes.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    marked = [e.id for e in state.store.all_real() if e.meta.get("consolidated_night") is not None]
    assert len(state.store) == 2 and len(marked) == 1
    assert report.episodes_trained == 1 and report.episodes_marked_consolidated == 1
    assert report.train["trained_episode_ids"] == marked


def test_no_budget_means_nothing_is_marked_consolidated(tokenizer, world):
    cfg = small_config(get_arm("B3_replay"), night_tokens=0)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.decision["accepted"] and report.train["exhausted"]
    assert report.episodes_trained == 0 and report.episodes_marked_consolidated == 0
    assert all("consolidated_night" not in e.meta for e in state.store.all_real())


def test_an_episode_keeps_the_night_it_was_first_consolidated(tokenizer, world):
    cfg = small_config(get_arm("B3_replay"))  # Four steps of two examples: every example, both episodes.
    state = fresh_gpt2_state(cfg, world, tokenizer)
    first = run_night(state, cfg)
    assert first.episodes_marked_consolidated == 2
    cfg.train = TrainConfig(hard_batch=2, max_steps=20)
    second = run_night(state, cfg)
    assert second.episodes_trained == 4 and second.episodes_marked_consolidated == 2
    assert state.store.get("n0-e0").meta["consolidated_night"] == 0
    assert state.store.get("n1-e0").meta["consolidated_night"] == 1


def test_create_passes_stop_tokens_to_the_handle(tokenizer, world):
    cfg = small_config(get_arm("B3_replay"))
    state = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, 7, cfg, "cpu", [tokenizer.eos_token_id, 9])
    assert state.handle.eos_ids == {tokenizer.eos_token_id, 9}
    assert fresh_gpt2_state(cfg, world, tokenizer).handle.eos_ids == {tokenizer.eos_token_id}


def test_exempt_counters_survive_a_checkpoint_so_reports_match_after_resume(tokenizer, tmp_path):
    cfg = small_config(get_arm("B2_seq"))

    def evaluate(state: State) -> None:
        with state.ledger.exempt():
            state.ledger.charge(forward=11, backward=2)

    straight = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    run_night(straight, cfg)
    evaluate(straight)
    expected = run_night(straight, cfg)
    assert expected.ledger["exempt_forward"] == 11 and expected.ledger["exempt_backward"] == 2

    first = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    run_night(first, cfg)
    evaluate(first)
    first.save(str(tmp_path))
    resumed = State.load(str(tmp_path), fresh_gpt2_state(cfg, TinyWorld(), tokenizer).handle, TinyWorld())
    assert resumed.ledger.exempt_forward == 11 and resumed.ledger.exempt_backward == 2
    assert run_night(resumed, cfg).to_dict() == expected.to_dict()


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


def test_write_gate_defers_or_keeps_active_by_threshold(tokenizer):
    handle = _wake_handle(tokenizer)
    args = (2, 2, torch.Generator().manual_seed(1))
    episode = Episode("e", "Mira keeps kites.", 0)
    probes = [("Who?", "Tides rise at noon daily.")]
    store = EpisodicStore()
    result = observe(handle, store, episode, probes, 1e9, *args)  # Anything is below this.
    assert result.deferred and result.stored and "e" in store
    assert store.get("e").status == "deferred" and "e" in [x.id for x in store.replayable()]
    store = EpisodicStore()
    result = observe(handle, store, episode, probes, -1e9, *args)
    assert not result.deferred and result.reason == "novel" and store.get("e").status == "active"


def test_write_gate_fails_open_and_bypasses(tokenizer):
    handle = _wake_handle(tokenizer)
    args = (2, 2, torch.Generator().manual_seed(1))
    probes = [("Who?", "Tides rise.")]
    assert observe(handle, EpisodicStore(), Episode("a", "t", 0), [], 1e9, *args).reason == "unscored"
    assert observe(handle, EpisodicStore(), Episode("b", "t", 0), probes, None, *args).reason == "unscored"
    assert observe(handle, EpisodicStore(), Episode("c", "t", 0), probes, 1e9, *args, bypass=True).reason == "bypass"
    superseding = Episode("d", "t", 0, supersedes=["a"])
    assert observe(handle, EpisodicStore(), superseding, probes, 1e9, *args).reason == "supersession"


def test_write_gate_with_no_budget_keeps_the_episode_active(tokenizer):
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
    arm = replace(_oracle_full(), selection=False, write_gate=False, transactional=False)
    cfg = small_config(arm)
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
    cfg = small_config(_oracle_full(), night_tokens=4000)
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


@pytest.mark.parametrize("arm_name", ["S_oracleq", "S_core", "B4_replay_kl"])
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
