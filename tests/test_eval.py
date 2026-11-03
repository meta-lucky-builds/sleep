"""Answer matching and every metric, with a scripted handle whose answers are known. No network."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tiny import TinyTokenizer, fresh_gpt2_state, small_config

from llmsleep.arms import get_arm
from llmsleep.eval.answers import (
    first_answer,
    has_artefact,
    is_abstention,
    is_specific,
    names_vocabulary,
    normalize_answer,
    score_answer,
)
from llmsleep.eval.capability import (
    CapabilityConfig,
    CapabilityItem,
    build_capability,
    cache_path,
    evaluate_capability,
    usable_answers,
)
from llmsleep.eval.drift import build_drift_probe, load_generic_prompts, measure_generic_drift
from llmsleep.eval.metrics import EvalConfig, evaluate_night, evaluate_probes
from llmsleep.handle import Sample
from llmsleep.ledger import ComputeLedger
from llmsleep.memory import EpisodicStore
from llmsleep.phases.night import run_night
from llmsleep.worlds import SyntheticWorld, WorldConfig
from llmsleep.worlds.familiarity import check_unknown


class ScriptedHandle:
    """Answers by a function of (query, context). Generation is the only thing eval needs from a model."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.ledger = ComputeLedger()
        self.batch_size = 4
        self.delta = None
        self.calls = 0
        self.active = "unset"
        self.sampled_under: set = set()  # The bank selection in force at each generation.

    def format(self, query, context=None):
        return json.dumps([query, context])

    def decode(self, ids):
        return "".join(ids)

    @contextmanager
    def banks(self, spec):
        previous, self.active = self.active, spec
        try:
            yield
        finally:
            self.active = previous

    def sample(self, prompts, n, max_new_tokens, temperature=1.0, generator=None):
        assert generator is not None
        self.calls += len(prompts)
        self.sampled_under.add(self.active)
        out = []
        for prompt in prompts:
            query, context = json.loads(prompt)
            text = self.answer(query, context)
            out.append([Sample(list(text), 0.0, True) for _ in range(n)])
        return out


def world(**overrides) -> SyntheticWorld:
    settings = dict(
        n_persons=8, n_orgs=3, n_cities=3, n_regions=2, nights=4, supersession_rate=0.5,
        n_distractors=8, n_control_persons=3, n_entrench=6,
    )
    settings.update(overrides)
    return SyntheticWorld(WorldConfig(**settings), 21)


def full_store(w: SyntheticWorld, up_to: int) -> EpisodicStore:
    store = EpisodicStore()
    for night in range(up_to + 1):
        for episode in w.episodes(night):
            store.add(episode)
    return store


def oracle_for(w: SyntheticWorld, night: int):
    """Answers each active probe with its first gold form. Queries are unique among active answerable probes."""
    table = {p.query: p.gold[0] for p in w.probes if p.active(night) and p.gold and p.context is None}

    def answer(query, context):
        return table.get(query, "I don't know.")

    return answer


LAST = 3
CLOSED = get_arm("B0_frozen")
CFG = EvalConfig(max_per_kind=None, max_new_tokens=8)


# ---- answer matching ---------------------------------------------------------------------------------


def test_word_boundaries_not_substrings():
    assert score_answer("Paris", ["Paris"]).correct
    assert not score_answer("Parisian", ["Paris"]).correct
    assert not score_answer("Comparison", ["Paris"]).correct
    assert score_answer("It is in Paris, France.", ["Paris"]).correct


def test_permissive_about_case_punctuation_articles_accents():
    for text in ("the PARIS.", "Paris!", "  paris  ", "a Paris", "\"Paris\""):
        assert score_answer(text, ["Paris"]).correct, text
    assert score_answer("Sao Paulo", ["São Paulo"]).correct
    assert score_answer("The Harlow valve", ["Harlow valve"]).correct
    assert normalize_answer("The Quick, Brown-Fox!") == "quick brown fox"


def test_strict_about_entity_substitution():
    vocabulary = ["Dara Quilmont", "Quilmont", "Dara Quilmond"]
    assert score_answer("Dara Quilmont", ["Dara Quilmont", "Quilmont"], (), vocabulary).correct
    assert not score_answer("Dara Quilmond", ["Dara Quilmont", "Quilmont"], (), vocabulary).correct
    assert not score_answer("Dara", ["Dara Quilmont", "Quilmont"], (), vocabulary).correct


def test_longest_term_wins_and_gold_alias_counts():
    vocabulary = ["Velmora Works", "Velmora", "Orvane Mills", "Orvane"]
    assert score_answer("Velmora Works", ["Velmora Works", "Velmora"], (), vocabulary).correct
    assert score_answer("velmora", ["Velmora Works", "Velmora"], (), vocabulary).correct
    # "Port Elda" must not also count as a mention of "Elda".
    assert not score_answer("Port Elda", ["Elda"], (), ["Port Elda", "Elda"]).correct


def test_naming_two_entities_is_not_correct():
    vocabulary = ["Velmora", "Orvane"]
    assert not score_answer("Velmora or Orvane", ["Velmora"], (), vocabulary).correct
    both = score_answer("Velmora, formerly Orvane", ["Velmora"], ["Orvane"], vocabulary)
    assert not both.correct and not both.stale


def test_stale_is_separate_from_wrong():
    vocabulary = ["Velmora", "Orvane", "Tessel"]
    assert score_answer("Orvane", ["Velmora"], ["Orvane"], vocabulary).stale
    wrong = score_answer("Tessel", ["Velmora"], ["Orvane"], vocabulary)
    assert not wrong.correct and not wrong.stale and wrong.other
    assert score_answer("Velmora", ["Velmora"], ["Velmora"], vocabulary).correct  # Unchanged value.


def test_a_term_the_query_itself_names_is_no_competitor():
    """Review H2: a sentence answer that repeats the question's own nickname is still right."""
    vocabulary = ["Dara Quilmont", "Quilmont", "Bobo", "Ivo Stant", "Stant", "Lume"]
    gold = ["Dara Quilmont", "Quilmont"]
    query = "Give the full name of the person known informally as Bobo."
    sentence = "The person known informally as Bobo is Dara Quilmont."
    assert score_answer(sentence, gold, (), vocabulary, query).correct
    assert not score_answer(sentence, gold, (), vocabulary).correct  # Without the query it reads as two entities.
    assert not score_answer("Bobo", gold, (), vocabulary, query).correct  # Echoing the term is not an answer.
    assert not score_answer("Bobo is Ivo Stant", gold, (), vocabulary, query).correct


def test_first_answer_stops_at_the_run_on():
    assert first_answer(" Paris\nQuestion: What else?") == "Paris"
    assert first_answer("Paris Question: What?") == "Paris"
    assert first_answer("Paris. It is big. And old.") == "Paris."
    assert first_answer("Dr. Quilmont did. Then more.") == "Dr. Quilmont did."
    assert first_answer("\n\n  Lyon") == "Lyon"
    assert first_answer("") == ""
    assert not score_answer("Lyon\nParis", ["Paris"]).correct  # Later lines are not the answer.


def test_abstention_and_specificity():
    for text in ("I don't know.", "Unknown", "I do not know who that is.", "There is no such person", "No one"):
        assert is_abstention(text), text
        assert not is_specific(text), text
    for text in ("Velmora", "I think Velmora"):
        assert not is_abstention(text), text
        assert is_specific(text), text
    assert not is_specific("   ")
    assert not is_abstention("Nonesuch")  # Word boundary: "none" inside a name is not abstention.


def test_names_vocabulary_is_the_type_check():
    assert names_vocabulary("It is the oboe.", ["oboe", "tuba"])
    assert not names_vocabulary("It is the guitar.", ["oboe", "tuba"])
    assert not names_vocabulary("oboes", ["oboe"])


def test_artefacts():
    assert has_artefact("Based on the text, Velmora.")
    assert not has_artefact("Velmora.")


# ---- metrics, from a scripted model ----------------------------------------------------------------------


def test_an_oracle_gets_full_marks_on_acquisition_and_supersession():
    w = world()
    handle = ScriptedHandle(oracle_for(w, LAST))
    metrics, clusters = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    for name in ("acq_direct", "acq_reverse", "acq_alias", "acq_twohop", "acq_implied", "supersede", "preservation"):
        assert metrics[name] == pytest.approx(1.0), name
    assert metrics["stale"] == pytest.approx(0.0)
    assert metrics["counts"]["supersede"] > 0
    assert set(clusters["acq_direct"]) and all(entry[1] > 0 for entry in clusters["acq_direct"].values())


def test_evaluation_generates_under_the_committed_bank():
    """Mutation 2 of the review: under germline, every arm would score like the frozen floor."""
    w = world()
    handle = ScriptedHandle(lambda q, c: "?")
    cfg = EvalConfig(max_per_kind=2, max_new_tokens=4, entrench_nights=[LAST], entrench_max=2)
    store = full_store(w, LAST)
    for probe in w.probes:
        if probe.kind == "entrench_fact":
            store.get(probe.episode_ids[0]).meta["consolidated_night"] = 0
    items = [CapabilityItem("q1", "Q one?", ("alpha",), 1.0, "head")]
    evaluate_night(handle, store, CLOSED, w, LAST, w.seed, cfg, items)
    assert handle.sampled_under == {"committed"}


def test_lift_is_measured_against_another_facts_gold():
    """A1.3: an answer of the right type but the wrong fact scores on the control, not on the metric."""
    w = world()
    active = [p for p in w.probes if p.active(LAST) and p.permuted and p.context is None]
    assert {p.kind for p in active} >= {"direct", "reverse", "alias", "two_hop", "supersede"}
    table = {p.query: p.permuted[0] for p in active}
    handle = ScriptedHandle(lambda q, c: table.get(q, "?"))
    metrics, clusters = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    for name in ("acq_reverse", "acq_alias", "acq_twohop"):
        assert metrics[name] == 0.0, name
        assert metrics[f"{name}_permuted"] > 0.3, name
        assert metrics[f"{name}_lift"] < -0.3, name
    assert "acq_twohop_lift" in clusters and "acq_twohop_permuted" not in clusters

    oracle = ScriptedHandle(oracle_for(w, LAST))
    metrics, _ = evaluate_probes(oracle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["acq_direct_lift"] == pytest.approx(1.0) and metrics["acq_direct_permuted"] == 0.0


def test_chains_and_types_are_reported_apart():
    w = world()
    handle = ScriptedHandle(oracle_for(w, LAST))
    metrics, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["acq_twohop_hq_city"] == pytest.approx(1.0) and metrics["acq_twohop_region"] == pytest.approx(1.0)
    assert metrics["acq_reverse"] is not None and metrics["acq_alias"] is not None


def test_accuracy_is_reported_by_fact_age():
    w = world()
    handle = ScriptedHandle(oracle_for(w, LAST))
    metrics, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    by_age = metrics["acq_by_age"]
    assert "0" in by_age and str(LAST) in by_age
    assert sum(count for _, count in by_age.values()) == metrics["counts"]["direct"] + metrics["counts"]["supersede"]
    assert all(hits == count for hits, count in by_age.values())
    json.dumps(by_age)


def test_an_abstainer_scores_zero_with_no_bleed():
    w = world()
    handle = ScriptedHandle(lambda q, c: "I don't know.")
    metrics, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["acq_direct"] == 0.0 and metrics["acq_implied"] == 0.0
    assert metrics["taught_value_bleed"] == 0.0 and metrics["false_recall_any_specific"] == 0.0
    assert metrics["abstain"] == 1.0


def test_bleed_counts_a_typed_answer_and_not_an_untyped_one():
    w = world()
    by_query = {p.query: p for p in w.probes if p.kind == "false_recall"}

    def typed(query, context):
        probe = by_query.get(query)
        return w.vocabulary(probe.vocab)[0] if probe else "?"

    metrics, _ = evaluate_probes(ScriptedHandle(typed), full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["taught_value_bleed"] == 1.0 and metrics["abstain"] == 0.0

    metrics, _ = evaluate_probes(
        ScriptedHandle(lambda q, c: "Zzyzzx Qqqq"), full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG
    )
    assert metrics["taught_value_bleed"] == 0.0  # Specific, but not of the attribute's type.
    assert metrics["false_recall_any_specific"] == 1.0

    def neighbour(query, context):
        probe = by_query.get(query)
        return probe.neighbour[0] if probe else "?"

    metrics, _ = evaluate_probes(ScriptedHandle(neighbour), full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["false_recall_neighbour"] == 1.0  # Answered with the look-alike's taught value.


def test_stale_answers_register_as_stale_not_as_supersede():
    w = world()
    stale_table = {p.query: p.stale[0] for p in w.probes if p.kind == "supersede" and p.active(LAST)}
    handle = ScriptedHandle(lambda q, c: stale_table.get(q, "?"))
    metrics, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["supersede"] == 0.0 and metrics["stale"] == 1.0


def test_only_taught_facts_are_probed_on_early_nights():
    w = world()
    handle = ScriptedHandle(lambda q, c: "?")
    early, _ = evaluate_probes(handle, full_store(w, 0), CLOSED, w, 0, w.seed, CFG)
    late, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert 0 < early["counts"]["direct"] < late["counts"]["direct"]
    assert early["counts"]["supersede"] == 0
    assert early["counts"]["false_recall"] == late["counts"]["false_recall"]


def test_acq_implied_is_the_mean_of_its_three_kinds():
    w = world()
    table = {p.query: p.gold[0] for p in w.probes if p.kind == "reverse" and p.active(LAST)}
    handle = ScriptedHandle(lambda q, c: table.get(q, "?"))
    metrics, clusters = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    assert metrics["acq_reverse"] == 1.0 and metrics["acq_alias"] == 0.0 and metrics["acq_twohop"] == 0.0
    assert metrics["acq_implied"] == pytest.approx(1 / 3)
    num = sum(entry[0] for entry in clusters["acq_implied"].values())
    den = sum(entry[1] for entry in clusters["acq_implied"].values())
    assert num / den == pytest.approx(1 / 3)


def test_open_book_arm_gets_gold_context_and_closed_book_gets_none():
    w = world()
    seen = []

    def record(query, context):
        seen.append(context)
        return "?"

    store = full_store(w, LAST)
    evaluate_probes(ScriptedHandle(record), store, CLOSED, w, LAST, w.seed, CFG)
    assert all(context is None for context in seen)
    seen.clear()
    evaluate_probes(ScriptedHandle(record), store, get_arm("B1_rag"), w, LAST, w.seed, CFG)
    assert any(context for context in seen)


def test_superseded_gold_context_is_the_newest_in_open_book():
    w = world()
    store = full_store(w, LAST)
    probe = next(p for p in w.probes if p.kind == "supersede" and p.active(LAST))
    contexts = {}

    def record(query, context):
        contexts[query] = context
        return "?"

    evaluate_probes(ScriptedHandle(record), store, get_arm("B1_rag"), w, LAST, w.seed, CFG)
    assert contexts[probe.query].startswith("[Most recent]")
    assert w.episode(probe.episode_ids[0]).text in contexts[probe.query]


def test_evaluation_is_batched_by_unique_prompt():
    w = world()
    handle = ScriptedHandle(lambda q, c: "?")
    metrics, _ = evaluate_probes(handle, full_store(w, LAST), CLOSED, w, LAST, w.seed, CFG)
    total = sum(metrics["counts"].values())
    assert handle.calls < total  # Preservation probes share their prompt with direct ones.


def test_audit_metric_reads_gaps_and_retirements():
    w = world()
    store = full_store(w, LAST)
    direct = [p for p in w.probes if p.kind == "direct" and p.active(LAST)]
    answers = {p.query: p.gold[0] for p in direct[:2]}
    store.get(direct[0].episode_ids[0]).meta["last_gap"] = 0.1
    store.get(direct[1].episode_ids[0]).meta["last_gap"] = 0.2
    store.get(direct[2].episode_ids[0]).meta["last_gap"] = 3.0  # Wrong, and a high gap.
    store.retire(direct[0].episode_ids[0])
    store.retire(direct[2].episode_ids[0])
    metrics, _ = evaluate_probes(ScriptedHandle(lambda q, c: answers.get(q, "?")), store, CLOSED, w, LAST, w.seed, CFG)
    assert metrics["audit_auroc"] == pytest.approx(1.0)
    assert metrics["retired"] == 2 and metrics["false_retirement"] == pytest.approx(0.5)


# ---- capability ----------------------------------------------------------------------------------------


QA = [
    {"question": "Q one?", "answers": ["alpha"]},
    {"question": "Q two?", "answers": ["beta", "bee"]},
    {"question": "Q three?", "answers": ["gamma"]},
    {"question": "Q four?", "answers": ["delta"]},
]


def _qa_handle(flaky_counter=None):
    known = {"Q one?": "alpha", "Q two?": "bee", "Q four?": "delta"}

    class Handle(ScriptedHandle):
        def sample(self, prompts, n, max_new_tokens, temperature=1.0, generator=None):
            self.calls += len(prompts)
            out = []
            for prompt in prompts:
                query, _ = json.loads(prompt)
                answer = known.get(query, "wrong")
                rows = []
                for i in range(n):
                    # At temperature 1, "Q four?" is right only half the time: the tail.
                    text = "wrong" if temperature > 0 and query == "Q four?" and i % 2 else answer
                    rows.append(Sample(list(text), 0.0, True))
                out.append(rows)
            return out

    return Handle(lambda q, c: "")


def test_capability_filters_splits_and_caches(tmp_path):
    cfg = CapabilityConfig(n_candidates=10, n_keep=10, k=4)
    path = cache_path(str(tmp_path), "base", cfg, QA)
    handle = _qa_handle()
    items = build_capability(handle, cfg, path, QA)
    assert {item.question for item in items} == {"Q one?", "Q two?", "Q four?"}  # Base-wrong is filtered out.
    by_question = {item.question: item for item in items}
    assert by_question["Q four?"].consistency == 0.5 and by_question["Q one?"].consistency == 1.0
    assert by_question["Q four?"].split == "tail" and by_question["Q one?"].split == "head"
    assert Path(path).exists()

    calls = handle.calls
    again = build_capability(handle, cfg, path, QA)  # From the cache: no model call, identical result.
    assert handle.calls == calls and again == items


def test_capability_cache_needs_no_items_and_never_downloads_when_cached(tmp_path, monkeypatch):
    cfg = CapabilityConfig(n_candidates=10, n_keep=10, k=2)
    path = cache_path(str(tmp_path), "base", cfg)
    build_capability(_qa_handle(), cfg, path, QA)

    def boom(*args, **kwargs):
        raise AssertionError("tried to download")

    monkeypatch.setattr("llmsleep.eval.capability.load_trivia_qa", boom)
    assert build_capability(_qa_handle(), cfg, path)


def test_short_aliases_cannot_give_false_credit():
    assert usable_answers(["United States", "US", "U.S.", "America"]) == ["United States", "America"]
    assert usable_answers(["Au", "gold"]) == ["Au", "gold"]  # The main answer stays, however short.
    assert usable_answers([]) == []


def test_cache_path_differs_by_model_and_settings(tmp_path):
    a = cache_path(str(tmp_path), "base-a", CapabilityConfig())
    assert a != cache_path(str(tmp_path), "base-b", CapabilityConfig())
    assert a != cache_path(str(tmp_path), "base-a", CapabilityConfig(k=3))
    assert a != cache_path(str(tmp_path), "base-a", CapabilityConfig(), QA)


def test_ret_real_reported_as_change_and_split():
    items = [
        CapabilityItem("q1", "Q one?", ("alpha",), 1.0, "head"),
        CapabilityItem("q2", "Q two?", ("beta",), 0.5, "tail"),
    ]
    handle = ScriptedHandle(lambda q, c: "alpha" if q == "Q one?" else "wrong")
    metrics, clusters = evaluate_capability(handle, items, CFG, 1, 0)
    assert metrics["ret_real"] == 0.5 and metrics["ret_real_change"] == -0.5
    assert metrics["ret_real_head"] == 1.0 and metrics["ret_real_tail"] == 0.0
    assert set(clusters["ret_real"]) == {"q1", "q2"}


# ---- real tiny model: exemption, banks, drift, entrenchment ------------------------------------------------


def _tiny_world() -> SyntheticWorld:
    return world(n_persons=2, n_orgs=2, n_cities=2, n_regions=1, nights=2, n_entrench=2, n_control_persons=2)


def test_generic_prompts_file_is_fixed_and_unique():
    prompts = load_generic_prompts()
    assert len(prompts) >= 50 and len(set(prompts)) == len(prompts)


def test_evaluation_is_exempt_and_leaves_the_model_alone():
    tok = TinyTokenizer()
    w = _tiny_world()
    cfg = small_config(get_arm("B2_seq"))
    state = fresh_gpt2_state(cfg, w, tok)
    run_night(state, cfg)
    spent, adapters = state.ledger.spent(), state.handle.delta.state_dict()
    exempt_before = state.ledger.exempt_forward
    eval_cfg = EvalConfig(max_new_tokens=2, max_per_kind=3, batch_size=4, entrench_nights=[0], entrench_max=2)
    drift = build_drift_probe(state.handle, 2, prompts=["The sky is", "Water is"])
    metrics = evaluate_night(state.handle, state.store, cfg.arm, w, 0, state.world_seed, eval_cfg, [], drift)
    assert state.ledger.spent() == spent and state.ledger.exempt_forward > exempt_before
    assert state.handle.batch_size == cfg.model.batch_size  # Restored.
    after = state.handle.delta.state_dict()
    for name in adapters:
        assert torch.equal(adapters[name]["B"], after[name]["B"])
    assert metrics["delta_rank"] == state.handle.delta.stats()["delta_rank"] > 0
    assert metrics["drift_generic"] is not None and metrics["drift_generic"] > 0
    assert metrics["night"] == 0
    assert metrics["entrench_pairs"] >= 0
    json.dumps(metrics)


def test_drift_is_zero_for_an_untouched_model_and_positive_after_a_change():
    tok = TinyTokenizer()
    w = _tiny_world()
    cfg = small_config(get_arm("B0_frozen"))
    state = fresh_gpt2_state(cfg, w, tok)
    handle = state.handle
    probe = build_drift_probe(handle, 3, prompts=["The sky is", "Water is", "A map"])
    with handle.ledger.exempt():
        value, clusters = measure_generic_drift(handle, probe)
    assert value == pytest.approx(0.0, abs=1e-5) and len(clusters) == 3
    with torch.no_grad():
        generator = torch.Generator().manual_seed(1)
        for layer in handle.delta.layers.values():
            layer.fresh_B.copy_(torch.randn(layer.fresh_B.shape, generator=generator))
    handle.delta.merge_fresh_into_committed()
    with handle.ledger.exempt():
        moved, _ = measure_generic_drift(handle, probe)
    assert moved > 1e-3


def test_entrenchment_only_counts_consolidated_facts():
    w = world()
    store = full_store(w, LAST)
    cfg = EvalConfig(max_new_tokens=4, max_per_kind=1, entrench_nights=[LAST], entrench_max=None, drift=False)
    facts = [p for p in w.probes if p.kind == "entrench_fact"]
    # Consolidate just the first; its pair alone should count.
    store.get(facts[0].episode_ids[0]).meta["consolidated_night"] = 0
    claimed = facts[0].gold[0]
    handle = ScriptedHandle(lambda q, c: claimed if c else "?")
    metrics = evaluate_night(handle, store, CLOSED, w, LAST, w.seed, cfg)
    assert metrics["entrench_pairs"] == 1
    assert metrics["entrench_follow_fact"] == 1.0 and metrics["entrench_follow_control"] == 1.0
    assert list(metrics["per_cluster"]["entrench_follow_fact"]) == [facts[0].cluster]


def test_entrenchment_measures_loyalty_to_the_old_value():
    w = world()
    store = full_store(w, LAST)
    cfg = EvalConfig(max_new_tokens=4, max_per_kind=1, entrench_nights=[LAST], entrench_max=None, drift=False)
    # Keyed with the context: a control person may appear in several pairs with one attribute.
    facts = {(p.query, p.context): p for p in w.probes if p.kind == "entrench_fact"}
    controls = {(p.query, p.context): p for p in w.probes if p.kind == "entrench_control"}
    for probe in facts.values():
        store.get(probe.episode_ids[0]).meta["consolidated_night"] = 0

    def answer(query, context):
        if (query, context) in facts:
            return facts[(query, context)].stale[0]  # Ignores the context: entrenched.
        if (query, context) in controls:
            return controls[(query, context)].gold[0]
        return "?"

    metrics = evaluate_night(ScriptedHandle(answer), store, CLOSED, w, LAST, w.seed, cfg)
    assert metrics["entrench_follow_fact"] == 0.0 and metrics["entrench_kept_old"] == 1.0
    assert metrics["entrench_follow_control"] == 1.0


# ---- familiarity ---------------------------------------------------------------------------------------


def test_check_unknown_rejects_a_knowing_model_and_lists_who_to_replace():
    w = world()
    clean = check_unknown(ScriptedHandle(lambda q, c: "I don't know."), w)
    assert clean["rejected"] == [] and clean["replace"] == []

    table = {p.query: p.gold[0] for p in w.probes if p.kind == "direct"}
    knowing = check_unknown(ScriptedHandle(lambda q, c: table.get(q, "?")), w)
    assert "birth_year" in knowing["rejected"] and knowing["replace"]
    assert all(name in {e.name for e in w.entities.values()} for name in knowing["replace"])


def test_one_lucky_year_does_not_reject():
    """Review H3: one hit in a closed vocabulary is what chance gives."""
    w = world()
    lucky = next(p for p in w.probes if p.kind == "direct" and p.vocab == "birth_year")
    result = check_unknown(ScriptedHandle(lambda q, c: lucky.gold[0] if q == lucky.query else "?"), w)
    assert result["attributes"]["birth_year"]["hits"] == 1
    assert result["rejected"] == []
