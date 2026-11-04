"""S_core (A2.1) and H8's exempt log (A2.2), on tiny models and a small real world. No downloads."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from tiny import TinyWorld, fresh_gpt2_state, small_config, tiny_gpt2

from llmsleep.arms import get_arm
from llmsleep.config import CommitConfig, FactAuditConfig
from llmsleep.phases.commit import decide_drift
from llmsleep.phases.fact_audit import panel_keys, plan_fact_audit
from llmsleep.phases.night import State, active_stage_fractions, run_night
from llmsleep.phases.queries import Candidate
from llmsleep.worlds import SyntheticWorld, WorldConfig

CORE = get_arm("S_core")
ORACLE_CORE = replace(CORE, query_source="template")  # A tiny model's own questions rarely parse.


def _no_chd_allowed(monkeypatch) -> None:
    import llmsleep.phases.night as night_module

    def forbidden(*args, **kwargs):
        raise AssertionError("this arm may compute no CHD that a decision could read")

    monkeypatch.setattr(night_module, "chd_safe", forbidden)
    monkeypatch.setattr(night_module, "null_distribution", forbidden)
    monkeypatch.setattr(night_module, "measure_gaps", forbidden)


def _fixed_pool(monkeypatch, size: int = 24) -> None:
    """Anchors that do not depend on a tiny model happening to write a question."""
    import llmsleep.phases.night as night_module

    def pool(handle, store, world, new_ids, cfg, generator):
        return (
            [
                Candidate(f"Who is Zed{i}?", ("old",), (f"Zed{i}",), context="Zed sails.", context_ids=("old",), pool=True)
                for i in range(size)
            ],
            False,
        )

    monkeypatch.setattr(night_module, "build_anchor_pool", pool)


def _committed_logits(state: State) -> torch.Tensor:
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    with state.handle.banks("committed"):
        return state.handle.net(input_ids=ids).logits.detach().clone()


# ---- the arm ------------------------------------------------------------------------------------------------


def test_core_has_no_signal_stage_and_an_equal_total():
    cfg = small_config(CORE)
    fractions = active_stage_fractions(CORE, cfg)
    assert set(fractions) == {"synthesis", "teacher", "train", "repair", "audit"}
    full = small_config(get_arm("S_full"))
    full_fractions = active_stage_fractions(full.arm, full)
    assert sum(fractions.values()) == pytest.approx(sum(full_fractions.values()))
    # The signals credit goes to training, as in the arm that ablates selection.
    assert fractions["train"] == pytest.approx(full_fractions["train"] + full_fractions["signals"])


def test_core_night_computes_no_gap_and_teaches_all_usable_queries(tokenizer, world, monkeypatch):
    _no_chd_allowed(monkeypatch)
    _fixed_pool(monkeypatch)
    cfg = small_config(ORACLE_CORE)
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    state = fresh_gpt2_state(cfg, world, tokenizer)
    report = run_night(state, cfg)
    assert report.selection["train"]["integrate"] == report.synthesis["train_candidates"] > 0
    assert report.teacher["integrate_sequences"] == report.synthesis["train_candidates"]  # samples=1.
    assert all(w["reason"] == "bypass" for w in report.wake) and len(state.store) == 2
    assert report.decision["paired_items"] == 0 and report.decision["gap_before"] is None
    assert report.signals["gate_rate"] == 0.0
    assert report.pressure is None and report.retired == []
    assert not any("audit_items" in e.meta or "last_gap" in e.meta for e in state.store.all_real())


def test_anchor_count_is_a_fixed_fraction_whatever_another_arm_recorded(tokenizer, world, monkeypatch):
    _fixed_pool(monkeypatch, size=40)
    cfg = small_config(ORACLE_CORE)
    cfg.commit = CommitConfig(max_anchor_drift=1e9, audit_max_anchors=100)
    cfg.select.fixed_anchor_frac = 0.5
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.selection["count_source"] == "fixed_fraction"
    available = report.synthesis["anchor_pool_train"]
    assert available > 0 and report.selection["train"]["anchor"] == math.floor(0.5 * available)
    # Held-out anchors are only measured, and the drift test alone decides, so all of them are used.
    assert report.selection["heldout"]["anchor"] == report.synthesis["anchor_pool_heldout"] > 0
    assert report.teacher["anchor_sequences"] == report.selection["train"]["anchor"]  # samples=1.

    cfg.select.matched_selection = {"0": {"train": {"anchor": 1}, "heldout": {"anchor": 0}}}
    again = run_night(fresh_gpt2_state(cfg, TinyWorld(), tokenizer), cfg)
    assert again.selection["train"]["anchor"] == report.selection["train"]["anchor"]
    assert again.selection["count_source"] == "fixed_fraction"


def test_anchors_are_built_first_so_integrate_targets_cannot_starve_them(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    _fixed_pool(monkeypatch)
    order: list[str] = []
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        if specs and specs[0].kind == "integrate":
            order.append("integrate")
        elif specs and len(args) == 3:  # Only the training stage passes a cache; held-out anchors do not.
            order.append("anchor")
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    cfg = small_config(ORACLE_CORE)
    run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert order == ["anchor", "integrate"]


def test_pairs_are_not_what_a_short_teacher_budget_always_drops(tokenizer, monkeypatch):
    import llmsleep.phases.night as night_module

    class ManyPairs(TinyWorld):
        def pair_template_queries(self, a, b):
            return [f"Together {i}, how many kites have {a.entities[0]} and {b.entities[0]}?" for i in range(8)]

    seen: list[str] = []
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        seen.extend(spec.query for spec in specs if spec.kind == "integrate")
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    cfg = small_config(ORACLE_CORE)
    cfg.queries.per_episode = 8
    run_night(fresh_gpt2_state(cfg, ManyPairs(), tokenizer), cfg)
    positions = [i for i, query in enumerate(seen) if query.startswith("Together")]
    assert 0 < len(positions) < len(seen)
    assert positions != list(range(len(seen) - len(positions), len(seen)))


# ---- the commit ---------------------------------------------------------------------------------------------


def test_decide_drift_is_a_pure_rule_that_fails_closed():
    cfg = CommitConfig(max_anchor_drift=0.5)
    assert decide_drift(0.1, cfg, True).accepted and decide_drift(0.1, cfg, True).reasons == ["passed"]
    rejected = decide_drift(0.9, cfg, True)
    assert not rejected.accepted and rejected.reasons == ["anchor_drift"] and rejected.anchor_drift == 0.9
    silent = decide_drift(None, cfg, True)
    assert not silent.accepted and silent.reasons == ["no_drift_data"]
    assert decide_drift(0.9, cfg, False).accepted and decide_drift(0.9, cfg, False).reasons[1] == "anchor_drift"
    assert decide_drift(None, cfg, False).accepted


def test_core_commits_on_drift_alone_and_can_roll_back(tokenizer, world, monkeypatch):
    _fixed_pool(monkeypatch)
    cfg = small_config(ORACLE_CORE)
    cfg.commit = CommitConfig(max_anchor_drift=1e9, min_gap_drop=1e9)  # A gap test would reject everything.
    accepted = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert accepted.decision["accepted"] and accepted.decision["reasons"] == ["passed"]
    assert accepted.decision["drift_targets"] > 0 and accepted.decision["anchor_drift"] is not None
    assert accepted.delta["delta_rank"] > 0

    cfg.commit = CommitConfig(max_anchor_drift=-1.0)
    state = fresh_gpt2_state(cfg, TinyWorld(), tokenizer)
    before = _committed_logits(state)
    rejected = run_night(state, cfg)
    assert not rejected.decision["accepted"] and rejected.decision["reasons"] == ["anchor_drift"]
    assert rejected.train["steps"] > 0 and rejected.delta["delta_rank"] == 0
    assert torch.equal(_committed_logits(state), before)
    assert all(e.meta.get("consolidated_night") is None for e in state.store.all_real())


def test_the_drift_test_is_paid_from_the_audit_stage(tokenizer, world, monkeypatch):
    _fixed_pool(monkeypatch)
    cfg = small_config(ORACLE_CORE)
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.ledger["forward"]["audit"] > 0
    assert report.ledger["spent"] <= cfg.budget.night_tokens


def test_nocommit_keeps_the_finding_and_the_night(tokenizer, world, monkeypatch):
    _fixed_pool(monkeypatch)
    cfg = small_config(replace(get_arm("S_nocommit"), query_source="template"))
    cfg.commit = CommitConfig(max_anchor_drift=-1.0)
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.decision["accepted"] and report.decision["reasons"] == ["non_transactional", "anchor_drift"]
    assert report.delta["delta_rank"] > 0


# ---- H8's log -----------------------------------------------------------------------------------------------


def _world(nights: int = 2, seed: int = 11) -> SyntheticWorld:
    return SyntheticWorld(
        WorldConfig(
            n_persons=2, n_orgs=2, n_cities=2, n_regions=1, nights=nights, supersession_rate=0.0,
            n_distractors=12, n_control_persons=2, n_entrench=0,
        ),
        seed,
    )


def _log_config(arm=ORACLE_CORE):
    cfg = small_config(arm)
    cfg.fact_audit = FactAuditConfig(panel=100, controls=4, null_items=3, answer_tokens=4)
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    return cfg


def _state(cfg, world, tokenizer, seed: int = 11) -> State:
    return State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, seed, cfg)


def test_control_facts_are_never_taught_and_do_not_touch_the_world():
    world = _world()
    printed = world.fingerprint()
    controls = world.control_facts(6)
    assert world.fingerprint() == printed
    assert [c.key for c in controls] == [c.key for c in world.control_facts(6)]
    assert len(controls) == 6 and len({c.key for c in controls}) == 6
    names = {c.subject for c in controls}
    assert names <= set(world.control_persons)
    for night in range(world.nights):
        for episode in world.episodes(night):
            assert not any(name in episode.text for name in names)
            assert not names & set(episode.entities)
    for fact in controls:
        assert fact.subject in fact.text and fact.subject in fact.query and fact.value in fact.text
        assert fact.value in fact.gold or fact.gold[0] == fact.value
    assert world.control_facts(0) == []


def test_panel_is_a_function_of_seed_and_key_alone():
    keys = [f"P{i}.employer" for i in range(50)]
    assert panel_keys(keys, 1, 10) == panel_keys(list(reversed(keys)), 1, 10)
    assert panel_keys(keys, 1, 10) != panel_keys(keys, 2, 10)
    assert len(panel_keys(keys, 1, 10)) == 10 and panel_keys(keys, 1, 0) == []


def test_log_has_every_field_on_identical_queries_with_controls(tokenizer, monkeypatch):
    import llmsleep.phases.fact_audit as module

    _fixed_pool(monkeypatch)
    world = _world()
    cfg = _log_config()
    state = _state(cfg, world, tokenizer)
    seen: list = []
    original = module.measure_gaps

    def recording(handle, items, seed, night, k, m, purpose="audit"):
        seen.append((purpose, list(items), seed, night))
        return original(handle, items, seed, night, k, m, purpose)

    monkeypatch.setattr(module, "measure_gaps", recording)
    report = run_night(state, cfg)
    audit = report.fact_audit
    json.dumps(audit)
    rows = audit["items"]
    tonight = [r for r in rows if r["role"] == "tonight"]
    controls = [r for r in rows if r["role"] == "control"]
    assert len(tonight) == len(world.episodes(0)) and len(controls) == 4
    for row in rows:
        for name in ("gap_before", "gap_after", "correct_before", "correct_after", "answer_before", "answer_after"):
            assert row[name] is not None, name
    assert all(r["key"].startswith("control:") for r in controls)
    assert any(r["heldout_before"] is not None and r["heldout_after"] is not None for r in tonight)
    # The same queries in the same contexts, under the same seed, before and after.
    measured = [call for call in seen if call[0] == "fact_audit"]
    assert len(measured) == 2 and measured[0][1] == measured[1][1] and measured[0][3] == measured[1][3]
    assert audit["null_gaps"] and audit["accepted"] is True and audit["retire_nights"] == cfg.commit.retire_nights


def test_log_after_a_rejected_night_shows_nothing_moved(tokenizer, monkeypatch):
    _fixed_pool(monkeypatch)
    cfg = _log_config()
    cfg.commit = CommitConfig(max_anchor_drift=-1.0)
    report = run_night(_state(cfg, _world(), tokenizer), cfg)
    assert report.fact_audit["accepted"] is False
    for row in report.fact_audit["items"]:
        assert row["gap_after"] == pytest.approx(row["gap_before"], abs=1e-4)
        assert row["correct_after"] == row["correct_before"] and row["answer_after"] == row["answer_before"]


def test_facts_stay_in_the_log_as_older_on_later_nights_with_their_queries(tokenizer):
    world = _world()
    cfg = _log_config()
    state = _state(cfg, world, tokenizer)
    first = run_night(state, cfg)
    second = run_night(state, cfg)
    first_keys = {r["key"] for r in first.fact_audit["items"] if r["role"] == "tonight"}
    older = {r["key"] for r in second.fact_audit["items"] if r["role"] == "older"}
    assert first_keys == older
    tonight = {r["key"] for r in second.fact_audit["items"] if r["role"] == "tonight"}
    assert tonight and not tonight & older
    with_queries = {r["key"] for r in first.fact_audit["items"] if r["heldout_n"]}
    assert with_queries == {r["key"] for r in second.fact_audit["items"] if r["role"] == "older" and r["heldout_n"]}
    assert {r["key"] for r in first.fact_audit["items"] if r["role"] == "control"} == {
        r["key"] for r in second.fact_audit["items"] if r["role"] == "control"
    }


def test_the_log_spends_no_ledger_tokens_and_changes_nothing(tokenizer):
    """Drives nothing: with and without it the weights, the spend and every other part of the report are the same."""
    with_log = _log_config()
    without = replace(with_log, arm=replace(with_log.arm, passive_audit=False))
    world_a, world_b = _world(), _world()
    state_a, state_b = _state(with_log, world_a, tokenizer), _state(without, world_b, tokenizer)
    for _ in range(2):
        report_a, report_b = run_night(state_a, with_log).to_dict(), run_night(state_b, without).to_dict()
        assert report_a["fact_audit"]["tokens"] > 0 and report_b["fact_audit"] == {}
        assert report_a["ledger"]["spent"] == report_b["ledger"]["spent"]
        assert report_a["ledger"]["forward"] == report_b["ledger"]["forward"]
        assert report_a["ledger"]["exempt_forward"] > report_b["ledger"]["exempt_forward"]
        for part in ("synthesis", "selection", "teacher", "train", "decision", "compaction"):
            assert report_a[part] == report_b[part], part
    assert torch.equal(_committed_logits(state_a), _committed_logits(state_b))


def test_a_world_with_nothing_to_offer_gives_an_empty_log(tokenizer, world):
    cfg = small_config(ORACLE_CORE)
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.fact_audit["items"] == [] and report.fact_audit["null_gaps"] == []


def test_panel_of_zero_turns_the_log_off(tokenizer):
    world = _world()
    cfg = _log_config()
    cfg.fact_audit = FactAuditConfig(panel=0, controls=0, null_items=0)
    state = _state(cfg, world, tokenizer)
    assert plan_fact_audit(world, state.store, 0, 11, [], cfg) == []


def test_same_facts_in_every_arm(tokenizer):
    cfg = _log_config()
    cfg.fact_audit = FactAuditConfig(panel=5, controls=2, null_items=1)
    keys = []
    for arm in (ORACLE_CORE, replace(ORACLE_CORE, teacher_kl=False)):
        cfg.arm = arm
        report = run_night(_state(cfg, _world(), tokenizer), cfg)
        keys.append(sorted(r["key"] for r in report.fact_audit["items"]))
    assert keys[0] == keys[1]
