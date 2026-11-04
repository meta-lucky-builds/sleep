"""E1b's arms (A3.2, A4.1) and the budget at its sizes, on stub and tiny models and small real worlds. No downloads."""
from __future__ import annotations

import inspect
import re
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from tiny import TinyTokenizer, TinyWorld, _Output, fresh_gpt2_state, small_config, tiny_gpt2

import llmsleep
from llmsleep.arms import ArmConfig, get_arm
from llmsleep.config import CommitConfig, TeacherConfig
from llmsleep.delta import DeltaModel
from llmsleep.eval.answers import is_abstention
from llmsleep.eval.capability import CapabilityItem, question_overlap
from llmsleep.eval.probes import select_probes
from llmsleep.handle import ModelHandle, PromptFormat
from llmsleep.ledger import ComputeLedger, LedgerExhausted
from llmsleep.memory import EpisodicStore
from llmsleep.phases import consolidate
from llmsleep.phases.commit import measure_drift
from llmsleep.phases.consolidate import (
    IDK_ANSWER,
    build_question_frame_targets,
    derived_examples,
    frame_question,
    negative_examples,
    scramble_name,
)
from llmsleep.phases.night import State, _synthesis_reserve, active_stage_fractions, run_night
from llmsleep.phases.queries import Candidate, PairQueryGen, TemplateQueryGen
from llmsleep.worlds import SyntheticWorld, WorldConfig
from llmsleep.worlds.names import edit_distance
from llmsleep.worlds.templates import TRAIN_DERIVED_TEMPLATES, iter_train_templates, load_eval_templates

SRC = Path(llmsleep.__file__).resolve().parent
QUESTION = "Who wrote Hamlet?"
ANSWER = "Shakespeare"


class ScriptedQA(nn.Module):
    """After `Question:` it writes one question, and after `Answer:` one answer, nearly surely. Nothing else is in it."""

    def __init__(self, tokenizer: TinyTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def _next(self, text: str) -> str:
        scripts = ((text.rfind("Question:"), "Question:", f" {QUESTION}\n"), (text.rfind("Answer:"), "Answer:", f" {ANSWER}\n"))
        position, marker, script = max(scripts, key=lambda row: row[0])
        tail = text[position + len(marker) :] if position >= 0 else None
        if tail is None or len(tail) >= len(script) or not script.startswith(tail):
            return "\n"
        return script[len(tail)]

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch, width = input_ids.shape
        logits = torch.full((batch, width, self.tokenizer.vocab_size), -30.0)
        for row in range(batch):
            real = []
            for column in range(width):
                if attention_mask is None or attention_mask[row, column]:
                    real.append(int(input_ids[row, column]))
                    token = self.tokenizer(self._next(self.tokenizer.decode(real)))["input_ids"][0]
                    logits[row, column, token] = 30.0
        return _Output(logits)


def _scripted_handle(tokenizer: TinyTokenizer) -> ModelHandle:
    return ModelHandle(ScriptedQA(tokenizer), tokenizer, "cpu", batch_size=8)


def _world(nights: int = 2, seed: int = 11, **overrides) -> SyntheticWorld:
    settings = dict(
        n_persons=4, n_orgs=2, n_cities=2, n_regions=2, nights=nights, supersession_rate=0.0,
        n_distractors=12, n_control_persons=2, n_entrench=0,
    )
    settings.update(overrides)
    return SyntheticWorld(WorldConfig(**settings), seed)


def _fixed_pool(monkeypatch, size: int = 12) -> None:
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


def _accept_any_line(monkeypatch) -> None:
    """A tiny model does not write questions, so this stands in for one that does."""
    monkeypatch.setattr(consolidate, "frame_question", lambda text, finished: "Who is it now?")


# ---- the frame ----------------------------------------------------------------------------------------------


def test_the_opening_is_the_closed_book_prompt_up_to_the_question():
    fmt = PromptFormat()
    assert fmt.question_open() == "Context: (none)\nQuestion:"
    assert fmt.render("Who?").startswith(fmt.question_open() + " ")


def test_the_frame_is_refused_on_a_chat_template(tokenizer):
    handle = ModelHandle(ScriptedQA(tokenizer), tokenizer, "cpu", prompt_format=PromptFormat(chat=True))
    with pytest.raises(ValueError):
        handle.question_open()


def test_frame_samples_are_the_models_question_and_answer_in_the_evals_exact_form(tokenizer):
    handle = _scripted_handle(tokenizer)
    cfg = TeacherConfig(self_samples=3, self_max_new_tokens=40, max_new_tokens=20, top_k=4)
    targets, exhausted, stats = build_question_frame_targets(handle, cfg, torch.Generator().manual_seed(1))
    assert not exhausted and len(targets) == 3 and stats["sequences"] == 3 and stats["questions"] == [QUESTION] * 3
    for target in targets:
        assert target.kind == "self_frame"
        assert tokenizer.decode(target.prompt_ids) == "Context: (none)\nQuestion:"
        whole = tokenizer.decode(target.prompt_ids + target.cont_ids)
        assert whole == handle.format(QUESTION, None) + f" {ANSWER}\n"
        assert target.top_idx.shape == (len(target.cont_ids), 4)


def test_only_a_whole_line_counts_as_a_question():
    assert frame_question(" Who wrote it?\n", True) == "Who wrote it?"
    assert frame_question(" Name the city.\n", True) == "Name the city."  # The eval's own forms look like this.
    assert frame_question(" Who wrote it?", False) is None  # Cut off.
    assert frame_question("Who?\n", True) is None
    assert frame_question(" and then it", True) is None
    assert frame_question("\n", True) is None


def test_a_model_that_never_writes_a_question_cannot_loop_forever(tokenizer):
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu", batch_size=4)
    cfg = TeacherConfig(self_samples=2, self_max_new_tokens=2, top_k=4)
    targets, _, stats = build_question_frame_targets(handle, cfg, torch.Generator().manual_seed(1))
    assert stats["attempted"] <= 20 * cfg.self_samples + 4
    assert len(targets) <= 2


def test_frame_targets_are_germlines_and_every_token_is_scored(tokenizer, monkeypatch):
    _accept_any_line(monkeypatch)
    delta = DeltaModel(tiny_gpt2(tokenizer.vocab_size), rank=2)
    handle = ModelHandle(delta, tokenizer, "cpu", batch_size=4)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for layer in delta.layers.values():
            layer.fresh_A.copy_(torch.randn(layer.fresh_A.shape, generator=generator))
            layer.fresh_B.copy_(torch.randn(layer.fresh_B.shape, generator=generator))
    delta.merge_fresh_into_committed()  # So germline and committed really differ.
    cfg = TeacherConfig(self_samples=2, self_max_new_tokens=2, max_new_tokens=3, top_k=4)
    targets, _, _ = build_question_frame_targets(handle, cfg, torch.Generator().manual_seed(1))
    assert targets
    with handle.banks("germline"):
        zero, _ = measure_drift(handle, targets)
    with handle.banks("committed"):
        moved, _ = measure_drift(handle, targets)
    assert zero == pytest.approx(0.0, abs=1e-4)
    assert moved > 1e-4


def test_an_unmetered_ledger_needs_a_count(tokenizer):
    with pytest.raises(ValueError):
        build_question_frame_targets(_scripted_handle(tokenizer), TeacherConfig(self_samples=None), torch.Generator())


def test_draws_fit_what_is_left_so_little_is_lost_at_the_end(tokenizer):
    handle = _scripted_handle(tokenizer)
    ledger = handle.ledger
    ledger.begin_night(20000, {"teacher": 1.0})
    cfg = TeacherConfig(self_samples=None, self_max_new_tokens=40, max_new_tokens=20, top_k=4)
    with ledger.stage("teacher"):
        targets, exhausted, _ = build_question_frame_targets(handle, cfg, torch.Generator().manual_seed(1))
    assert targets and not exhausted  # Running out is how a count of None ends.
    assert ledger.spent("teacher") <= 20000
    per_sequence = len(handle.encode_prompt(handle.format(QUESTION, None))) * 3
    assert ledger.remaining("teacher") < 8 * per_sequence  # Less than one draw of the batch size was left unspent.


# ---- nothing of the real QA set can reach the samples ----------------------------------------------------------


def test_real_questions_have_no_way_into_frame_samples(tokenizer, monkeypatch):
    # By construction: the builder's inputs, the night's and the state's.
    assert list(inspect.signature(build_question_frame_targets).parameters) == ["handle", "cfg", "generator"]
    assert list(inspect.signature(run_night).parameters) == ["state", "cfg"]
    assert not {f.name for f in fields(State)} & {"capability", "qa_items", "real_items"}
    for name in ("consolidate.py", "night.py", "queries.py"):
        text = (SRC / "phases" / name).read_text(encoding="utf-8")
        assert "capability" not in text and "real_facts" not in text and "trivia" not in text.casefold(), name

    # In behaviour: every prompt the model is given is the opening, or a line it wrote itself put in the frame.
    handle = _scripted_handle(tokenizer)
    canary = "What is the canary of Zanzibar?"
    prompts: list[str] = []
    original = handle.encode_prompt

    def spy(text):
        prompts.append(text)
        return original(text)

    monkeypatch.setattr(handle, "encode_prompt", spy)
    cfg = TeacherConfig(self_samples=2, self_max_new_tokens=40, max_new_tokens=20, top_k=4)
    targets, _, _ = build_question_frame_targets(handle, cfg, torch.Generator().manual_seed(1))
    assert targets
    assert set(prompts) == {"Context: (none)\nQuestion:", handle.format(QUESTION, None)}
    assert all(canary not in tokenizer.decode(t.prompt_ids + t.cont_ids) for t in targets)


@pytest.mark.parametrize("arm_name", ["B4q", "S_coreq"])
def test_a_night_never_touches_the_real_qa_code(arm_name, tokenizer, world, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a night reached the real QA set")

    for name in ("load_trivia_qa", "build_capability", "evaluate_capability"):
        monkeypatch.setattr(f"llmsleep.eval.capability.{name}", forbidden)
    _accept_any_line(monkeypatch)
    cfg = small_config(get_arm(arm_name))
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.teacher["frame"]["sequences"] > 0


def test_overlap_with_the_real_set_is_counted_afterwards_word_for_word():
    items = [CapabilityItem("q1", "Who wrote Hamlet?", ("x",), 1.0, "head"), CapabilityItem("q2", "What is it?", ("y",), 1.0, "tail")]
    assert question_overlap(["who  wrote hamlet?", "Who wrote Hamlet?", "Something else?"], items) == 1
    assert question_overlap([], items) == 0


# ---- the arms that use them ---------------------------------------------------------------------------------


def test_frame_arms_differ_from_their_parents_by_flags_only():
    assert replace(get_arm("B4q"), name="B4_replay_kl", self_sample_frame=False) == get_arm("B4_replay_kl")
    plain = replace(get_arm("S_coreq"), name="S_core", self_sampled_kl=False, self_sample_frame=False, passive_audit=True)
    assert plain == get_arm("S_core")
    assert not get_arm("S_coreq").passive_audit and not get_arm("S_oracleq").passive_audit  # A4.3.
    with pytest.raises(ValueError):
        ArmConfig(name="x", self_sample_frame=True)


def test_b4q_spends_its_teacher_stage_on_frame_samples_and_trains_on_them(tokenizer, world, monkeypatch):
    _accept_any_line(monkeypatch)
    cfg = small_config(get_arm("B4q"))
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.teacher["anchor_sequences"] == cfg.teacher.self_samples == report.teacher["frame"]["sequences"]
    assert report.teacher["integrate_sequences"] == 0
    assert report.train["loss_anchor"] != 0.0


def test_s_coreq_pays_from_the_teacher_stage_within_its_share_and_hard_labels_keep_their_cap(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    _accept_any_line(monkeypatch)
    _fixed_pool(monkeypatch)
    rooms: list = []
    train_caps: list = []
    original_frame, original_train = night_module.build_question_frame_targets, night_module.train_loop

    def frame_spy(handle, cfg, generator):
        rooms.append((handle.ledger.room(), handle.ledger.remaining()))
        return original_frame(handle, cfg, generator)

    def train_spy(handle, *args, **kwargs):
        train_caps.append(handle.ledger.remaining())
        return original_train(handle, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_question_frame_targets", frame_spy)
    monkeypatch.setattr(night_module, "train_loop", train_spy)
    arm = replace(get_arm("S_coreq"), query_source="template")
    core = replace(arm, name="S_core", self_sampled_kl=False, self_sample_frame=False)
    cfg = small_config(arm)
    cfg.teacher = replace(cfg.teacher, self_samples=None, frame_share=0.33)
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    fractions = active_stage_fractions(arm, cfg)
    assert fractions == active_stage_fractions(core, cfg)  # The same allowance, split the same way.
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    room, stage = rooms[0]
    assert room == int(stage * 0.33)
    frame = report.teacher["frame"]
    assert 0 < frame["tokens"] <= room
    assert frame["sequences"] > 0
    assert report.teacher["integrate_sequences"] > 0  # What is left still teaches.
    assert report.train["steps"] > 0 and report.train["loss_hard"] > 0
    # Nothing was taken from hard labels: training has at least its own share of the night, whatever the frame spent.
    assert train_caps[0] >= int(cfg.budget.night_tokens * fractions["train"] / sum(fractions.values()))


def test_query_anchors_can_be_held_to_a_share_so_integrate_targets_survive(tokenizer, world, monkeypatch):
    import llmsleep.phases.night as night_module

    _fixed_pool(monkeypatch, size=200)
    arm = replace(get_arm("S_core"), query_source="template", passive_audit=False)
    cfg = small_config(arm, night_tokens=30_000)
    cfg.teacher = replace(cfg.teacher, query_anchor_share=0.25)
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    calls: list = []
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        if specs and specs[0].kind == "anchor" and len(args) == 3:  # Only the training stage passes a cache.
            calls.append((handle.ledger.room(), handle.ledger.remaining()))
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    room, stage = calls[0]
    assert room == int(stage * 0.25)
    assert report.teacher["integrate_sequences"] > 0


# ---- the synthesis reserve ----------------------------------------------------------------------------------


def test_the_reserve_keeps_later_parts_their_share_and_passes_on_what_is_unused():
    cfg = small_config(get_arm("S_core"))
    cfg.queries.synthesis_shares = {"singles": 0.5, "pairs": 0.3, "anchors": 0.2}
    ledger = ComputeLedger()
    ledger.begin_night(1000, {"synthesis": 1.0})
    with ledger.stage("synthesis"):
        with _synthesis_reserve(ledger, cfg, 1000, "singles"):
            ledger.charge(forward=100)  # Singles leave 400 of theirs.
            assert ledger.room() == 400
        with _synthesis_reserve(ledger, cfg, 1000, "singles", "pairs"):
            assert ledger.room() == 700  # Their own 300, and what singles left.
            ledger.charge(forward=700)
            with pytest.raises(LedgerExhausted):
                ledger.charge(forward=1)
        assert ledger.room() == 200  # The anchors' share is still there.


def test_without_shares_there_is_no_reserve():
    cfg = small_config(get_arm("S_core"))
    ledger = ComputeLedger()
    ledger.begin_night(1000, {"synthesis": 1.0})
    with ledger.stage("synthesis"), _synthesis_reserve(ledger, cfg, 1000, "singles"):
        assert ledger.room() == 1000


# ---- derived facts ------------------------------------------------------------------------------------------


def _expected_derived(world: SyntheticWorld, night: int) -> set[tuple[str, str, str]]:
    """Gold, worked out from the stream as a reader would: go through the episodes and keep each key's last value."""
    latest: dict = {}
    for n in range(night + 1):
        for episode in world.episodes(n):
            fact = world.fact_of_episode(episode.id)
            latest[fact.key] = fact
    alias_attrs = load_eval_templates()["alias"]
    out = set()
    for key, fact in latest.items():
        if fact.attr in alias_attrs and f"{fact.subject}.nickname" in latest:
            out.add(("alias", key, fact.value))
        for first, second in (("employer", "hq_city"), ("hometown", "region")):
            if fact.attr == first and f"{fact.value_id}.{second}" in latest:
                out.add(("two_hop", key, latest[f"{fact.value_id}.{second}"].value))
    return out


@pytest.mark.parametrize("seed", [11, 12])
def test_derived_facts_match_gold_worked_out_apart_from_the_builder(seed):
    world = _world(nights=4, seed=seed, n_persons=8, n_orgs=3, n_cities=3, supersession_rate=1.0)
    for night in range(world.nights):
        derived = world.derived_facts(night)
        assert {(d.kind, d.key, d.value) for d in derived} == _expected_derived(world, night)
        assert len(derived) == len({(d.kind, d.key) for d in derived})


def test_derived_facts_are_exactly_what_the_probes_ask_on_that_night():
    """Probes are built by other code, so this is a second, independent reading of what is true when."""
    world = _world(nights=4, n_persons=8, n_orgs=3, n_cities=3, supersession_rate=1.0)
    for night in range(world.nights):
        derived = {(d.kind, d.key): d for d in world.derived_facts(night)}
        probes = {
            (p.kind, p.cluster): p for kind in ("alias", "two_hop") for p in select_probes(world.probes, kind, night, None, world.seed)
        }
        assert set(derived) == set(probes)
        for key, fact in derived.items():
            assert fact.value in probes[key].gold


def test_a_derived_fact_waits_for_both_facts_and_drops_a_superseded_value():
    world = _world(nights=4, n_persons=8, n_orgs=3, n_cities=3, supersession_rate=1.0)
    changed = 0
    for night in range(world.nights):
        for fact in world.derived_facts(night):
            for episode_id in fact.episode_ids:
                assert world.fact_of_episode(episode_id).night <= night
            versions = world.versions(fact.key)
            if fact.kind == "alias" and len(versions) > 1:
                stale = [v.value for v in versions if v.night <= night][:-1]
                changed += bool(stale)
                for value in stale:
                    assert not any(value in text for text in fact.statements)
                    assert all(answer != value for _, answer in fact.qa)
    assert changed > 0


def test_derived_forms_are_training_forms_and_never_a_probe():
    world = _world(nights=2)
    forms = list(iter_train_templates())
    assert all(s in forms for attr in TRAIN_DERIVED_TEMPLATES["alias"].values() for s in attr["statements"] + attr["qa"])
    assert all(s in forms for statements in TRAIN_DERIVED_TEMPLATES["two_hop"].values() for s in statements)
    probes = {p.query for p in world.probes}
    for fact in world.derived_facts(1):
        assert not probes & {q for q, _ in fact.qa}
        assert not probes & set(fact.statements)


def test_calling_it_changes_neither_the_world_nor_its_fingerprint():
    world = _world(nights=2)
    printed = world.fingerprint()
    assert world.derived_facts(1) == world.derived_facts(1)
    assert world.fingerprint() == printed == _world(nights=2).fingerprint()


def test_b3_derived_trains_on_them_and_keeps_them_out_of_the_store(tokenizer):
    world = _world(nights=1)
    cfg = small_config(get_arm("B3_derived"))
    state = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, 11, cfg)
    report = run_night(state, cfg)
    assert report.train_extra["derived_facts"] == len(world.derived_facts(0)) > 0
    assert len(state.store) == len(world.episodes(0))


def test_derived_examples_are_plain_text_and_closed_book_qa(tokenizer):
    world = _world(nights=1)
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu")
    fact = world.derived_facts(0)[0]
    examples = derived_examples(handle, [fact])
    assert len(examples) == len(fact.statements) + len(fact.qa)
    assert all(example.episode_ids == fact.episode_ids for example in examples)
    assert tokenizer.decode(examples[0].cont_ids) == fact.statements[0]
    assert tokenizer.decode(examples[-1].prompt_ids) == handle.format(fact.qa[-1][0], None)
    assert fact.value in tokenizer.decode(examples[-1].cont_ids)


# ---- oracle queries reach composition ---------------------------------------------------------------------------


def test_oracle_pairs_cover_every_alias_and_two_hop_fact():
    world = _world(nights=1)
    store = EpisodicStore()
    for episode in world.episodes(0):
        store.add(episode)
    gen = PairQueryGen(TemplateQueryGen(world, 4), 1000)
    candidates, exhausted = gen.generate(store, {e.id for e in world.episodes(0)}, torch.Generator().manual_seed(1))
    assert not exhausted
    by_query = {c.query: c for c in candidates}
    derived = world.derived_facts(0)
    assert {d.kind for d in derived} == {"alias", "two_hop"}
    for fact in derived:
        for query, _ in fact.qa:
            assert by_query[query].is_pair and set(by_query[query].episode_ids) == set(fact.episode_ids)
    probes = {p.query for p in world.probes}
    assert not probes & set(by_query)


def test_alias_queries_need_the_nickname_and_the_fact_of_one_person():
    world = _world(nights=1)
    person, other = world.persons[0].id, world.persons[1].id
    nickname = world.episode(world.versions(f"{person}.nickname")[0].episode_id)
    employer = world.episode(world.versions(f"{person}.employer")[0].episode_id)
    elsewhere = world.episode(world.versions(f"{other}.employer")[0].episode_id)
    queries = world.pair_template_queries(nickname, employer)
    assert queries and queries == world.pair_template_queries(employer, nickname)
    assert all(world.versions(f"{person}.nickname")[0].value in q for q in queries)
    assert world.pair_template_queries(nickname, elsewhere) == []


def test_s_oracleq_teaches_alias_and_two_hop_queries_with_both_facts_in_context(tokenizer, monkeypatch):
    import llmsleep.phases.night as night_module

    _fixed_pool(monkeypatch)
    world = _world(nights=1)
    cfg = small_config(get_arm("S_oracleq"))
    cfg.queries.max_pairs = 1000
    cfg.queries.per_episode = 4
    cfg.commit = CommitConfig(max_anchor_drift=1e9)
    specs_seen: dict = {}
    original = night_module.build_targets

    def recording(handle, specs, *args, **kwargs):
        specs_seen.update({spec.query: spec for spec in specs if spec.kind == "integrate"})
        return original(handle, specs, *args, **kwargs)

    monkeypatch.setattr(night_module, "build_targets", recording)
    state = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, 11, cfg)
    run_night(state, cfg)
    kinds = set()
    for fact in world.derived_facts(0):
        for query, _ in fact.qa:
            spec = specs_seen.get(query)
            if spec is not None:
                kinds.add(fact.kind)
                for episode_id in fact.episode_ids:
                    assert world.episode(episode_id).text in spec.teacher_context
    assert kinds == {"alias", "two_hop"}


# ---- "I don't know" on scrambled names ----------------------------------------------------------------------------


def test_the_answer_is_one_the_eval_counts_as_abstaining():
    assert is_abstention(IDK_ANSWER)


def test_a_scramble_is_two_edits_from_its_source_and_keeps_first_letters():
    generator = torch.Generator().manual_seed(1)
    for name in ("Bara Tolvane", "Kelmont Foundry"):
        for _ in range(30):
            scrambled = scramble_name(name, generator)
            assert scrambled != name and edit_distance(scrambled, name) == 2
            assert [w[0] for w in scrambled.split()] == [w[0] for w in name.split()]
    assert scramble_name("Aaa", generator) is None


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_negatives_never_overlap_the_worlds_look_alikes_or_any_name(seed, tokenizer):
    world = _world(nights=4, seed=seed, n_persons=20, n_orgs=6, n_cities=4, n_control_persons=6)
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu")
    episodes = [e for night in range(world.nights) for e in world.episodes(night)]
    negatives = negative_examples(handle, episodes, 400, torch.Generator().manual_seed(seed))
    assert len(negatives) > 100
    assert world.lookalikes
    forbidden = (
        [entity.name for entity in world.entities.values()]
        + [lookalike.name for lookalike in world.lookalikes]
        + list(world.control_persons)
    )
    patterns = [re.compile(rf"\b{re.escape(name)}\b") for name in forbidden]
    for example in negatives:
        text = tokenizer.decode(example.prompt_ids)
        assert not any(pattern.search(text) for pattern in patterns), text
        assert is_abstention(tokenizer.decode(example.cont_ids))
        assert example.episode_ids == ()  # They teach no episode.


def test_b4q_idk_adds_its_share_to_the_hard_pool_on_real_names(tokenizer):
    world = _world(nights=2)
    cfg = small_config(get_arm("B4q_idk"))
    state = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, 11, cfg)
    report = run_night(state, cfg)
    assert report.train_extra["idk_negatives"] > 0
    assert report.ledger["spent"] <= cfg.budget.night_tokens


def test_a_world_without_full_names_gives_no_negatives(tokenizer, world):
    handle = ModelHandle(tiny_gpt2(tokenizer.vocab_size), tokenizer, "cpu")
    assert negative_examples(handle, world.episodes(0), 10, torch.Generator()) == []


# ---- rank and compaction -----------------------------------------------------------------------------------------


def test_r8_changes_the_rank_and_keeps_the_scale(tokenizer):
    plain = small_config(get_arm("B4q"))
    small = small_config(get_arm("B4q_r8"))
    plain.model.rank, plain.model.alpha = 32, 32.0
    small.model.rank, small.model.alpha = 32, 32.0
    a = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, TinyWorld(), 7, plain).handle.delta
    b = State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, TinyWorld(), 7, small).handle.delta
    layer_a, layer_b = next(iter(a.layers.values())), next(iter(b.layers.values()))
    assert layer_a.fresh_A.shape[1] == 32 and layer_b.fresh_A.shape[1] == 8
    assert a.rank == 32 and b.rank == 8 and a.scale == b.scale == 1.0


def test_b4q_compact_compacts_and_repairs(tokenizer, world):
    cfg = small_config(get_arm("B4q_compact"))
    report = run_night(fresh_gpt2_state(cfg, world, tokenizer), cfg)
    assert report.compaction["compacted"] and report.compaction["repair"] is not None
    assert "repair" in report.ledger["forward"]


def test_a_big_worlds_nicknames_fall_back_to_long_names_rather_than_fail():
    from llmsleep.worlds.names import NameForge, SeededRng

    forge = NameForge(SeededRng(1), min_distance=3)
    words = [forge.token(short=True, attempts=50) for _ in range(300)]
    assert len(set(words)) == 300
    assert forge._short_spent
    limit = 2
    assert all(edit_distance(a.casefold(), b.casefold(), limit) > limit for i, a in enumerate(words) for b in words[:i])
