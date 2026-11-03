"""SyntheticWorld: invariant 6, name collisions, determinism, stream structure, the night's protocol."""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tiny import TinyTokenizer, fresh_gpt2_state, small_config

from llmsleep.arms import get_arm
from llmsleep.eval.answers import contains_term, normalize_tokens
from llmsleep.eval.probes import select_probes
from llmsleep.phases.night import run_night
from llmsleep.phases.queries import WorldLike
from llmsleep.worlds import SyntheticWorld, TemplateLeakError, WorldConfig, check_disjoint, edit_distance
from llmsleep.worlds.names import BLOCKLIST, NameForge, SeededRng
from llmsleep.worlds.templates import (
    TRAIN_TEMPLATES,
    iter_eval_templates,
    iter_train_templates,
    load_eval_templates,
    ngrams,
    template_tokens,
)

ROOT = Path(__file__).resolve().parents[1]


def small_world(seed: int = 11, **overrides) -> SyntheticWorld:
    settings = dict(
        n_persons=8, n_orgs=3, n_cities=3, n_regions=2, nights=4, supersession_rate=0.5,
        n_distractors=12, n_control_persons=3, n_entrench=6,
    )
    settings.update(overrides)
    return SyntheticWorld(WorldConfig(**settings), seed)


def all_episodes(world: SyntheticWorld):
    return [e for night in range(world.nights) for e in world.episodes(night)]


def all_training_text(world: SyntheticWorld) -> list[str]:
    out = []
    for episode in all_episodes(world):
        out.append(episode.text)
        out.extend(episode.meta["statements"])
        out.extend(q for q, _ in episode.meta["qa"])
    return out


# ---- invariant 6 ------------------------------------------------------------------------------------


def test_shipped_templates_are_disjoint_with_no_shared_4gram():
    train = list(iter_train_templates())
    evals = list(iter_eval_templates(load_eval_templates()))
    assert train and evals
    train_grams = set().union(*(ngrams(template_tokens(t)) for t in train))
    eval_grams = set().union(*(ngrams(template_tokens(t)) for t in evals))
    assert not train_grams & eval_grams
    assert not {" ".join(template_tokens(t)) for t in train} & {" ".join(template_tokens(t)) for t in evals}


def test_slots_are_excluded_from_grams():
    assert template_tokens("Who is {subject}'s employer at {value}?") == ["who", "is", "employer", "at"]


def test_check_catches_identical_form_and_shared_4gram():
    with pytest.raises(TemplateLeakError):
        check_disjoint(["Where does {subject} work?"], ["Where does {subject} work?"])
    with pytest.raises(TemplateLeakError):  # Long shared prefix, with the slot in a different place.
        check_disjoint(["Tell me about the job of {subject}."], ["Tell me about the {subject} job, please."])
    check_disjoint(["Tell me about {subject}."], ["Tell me which job {subject} holds."])


def test_world_construction_refuses_a_leaking_eval_template():
    leaky = copy.deepcopy(load_eval_templates())
    leaky["direct"]["employer"] = [TRAIN_TEMPLATES["employer"]["qa"][0]]
    with pytest.raises(TemplateLeakError):
        SyntheticWorld(WorldConfig(n_persons=2, n_orgs=2, n_cities=2, nights=2), 1, eval_templates=leaky)
    shared = copy.deepcopy(load_eval_templates())
    shared["direct"]["employer"] = ["Now {subject} has a full-time job somewhere; where?"]
    with pytest.raises(TemplateLeakError):
        SyntheticWorld(WorldConfig(n_persons=2, n_orgs=2, n_cities=2, nights=2), 1, eval_templates=shared)


def test_no_probe_query_appears_in_training_data():
    world = small_world()
    text = "\n".join(all_training_text(world)).casefold()
    for probe in world.probes:
        assert probe.query.casefold() not in text


def test_oracle_queries_never_equal_probes():
    world = small_world()
    queries = {p.query for p in world.probes}
    for episode in all_episodes(world):
        assert not queries & set(world.template_queries(episode))


# ---- names -------------------------------------------------------------------------------------------


def test_edit_distance_values_and_limit():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abcdef", "uvwxyz", limit=2) == 3
    assert edit_distance("a", "abcdefgh", limit=2) == 3


def test_forge_keeps_distance_and_skips_blocklist():
    forge = NameForge(SeededRng(3), min_distance=3)
    tokens = [forge.token() for _ in range(40)] + [forge.token(short=True) for _ in range(20)]
    lowered = [t.casefold() for t in tokens]
    assert len(set(lowered)) == 60 and not set(lowered) & BLOCKLIST
    for i, a in enumerate(lowered):
        for b in lowered[i + 1 :]:
            assert edit_distance(a, b) >= 3


def test_lookalikes_are_near_and_never_taught():
    world = small_world()
    assert {x.kind for x in world.lookalikes} == {"person", "org"}
    text = "\n".join(all_training_text(world) + [e.text for e in world.distractor_pool()])
    taught = {e.name for e in world.entities.values()}
    for lookalike in world.lookalikes:
        target = world.entities[lookalike.target].name
        assert 1 <= edit_distance(lookalike.name, target) <= 2
        assert lookalike.name not in taught
        assert lookalike.name not in text


def test_every_pair_of_taught_people_is_far_apart():
    world = small_world()
    names = [p.name.casefold() for p in world.persons]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert edit_distance(a, b) >= 3


def test_control_people_are_never_taught():
    world = small_world()
    text = "\n".join(all_training_text(world))
    assert world.control_persons
    for name in world.control_persons:
        assert name not in text


# ---- determinism --------------------------------------------------------------------------------------


def test_same_seed_same_world_other_seed_other_world():
    assert small_world(5).fingerprint() == small_world(5).fingerprint()
    assert small_world(5).fingerprint() != small_world(6).fingerprint()


def test_world_is_independent_of_hash_randomisation():
    script = (
        "from llmsleep.worlds import SyntheticWorld, WorldConfig\n"
        "w = SyntheticWorld(WorldConfig(n_persons=4, n_orgs=2, n_cities=2, nights=3), 9)\n"
        "print(w.fingerprint())\n"
    )
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1] and outputs[0]


def test_seeded_rng_is_pure_in_its_seed():
    a, b = SeededRng(1), SeededRng(1)
    assert [a.below(100) for _ in range(20)] == [b.below(100) for _ in range(20)]
    assert sorted(SeededRng(2).shuffled(range(10))) == list(range(10))


# ---- the stream ---------------------------------------------------------------------------------------


def test_world_satisfies_the_protocol_and_episodes_are_real():
    world = small_world()
    assert isinstance(world, WorldLike)
    episodes = all_episodes(world)
    assert all(e.provenance == "real" for e in episodes)
    assert len({e.id for e in episodes}) == len(episodes)
    assert world.episodes(99) == []


def test_every_initial_fact_appears_and_load_is_even():
    world = small_world(supersession_rate=0.0)
    expected = 6 * 8 + 2 * 3 + 3
    counts = [len(world.episodes(n)) for n in range(4)]
    assert sum(counts) == expected and max(counts) - min(counts) <= 1


def test_training_forms_have_paraphrases_reversals_and_qa():
    world = small_world()
    for episode in all_episodes(world):
        fact = world.fact_of_episode(episode.id)
        statements = episode.meta["statements"]
        assert len(statements) >= 2
        assert episode.text.replace("Update: ", "") not in statements
        assert episode.meta["qa"] and all(len(pair) == 2 for pair in episode.meta["qa"])
        reversal = TRAIN_TEMPLATES[fact.attr]["reversals"][0]
        subject = world.entities[fact.subject].name
        assert reversal.format(subject=subject, value=fact.value) in statements


def test_a_statement_never_contains_a_second_linked_fact():
    """Co-location is what pair contexts test: hard-label data must not supply it."""
    world = small_world()
    for episode in all_episodes(world):
        fact = world.fact_of_episode(episode.id)
        if fact.attr not in ("employer", "hometown"):
            continue
        second_attr = "hq_city" if fact.attr == "employer" else "region"
        second = world.versions(f"{fact.value_id}.{second_attr}")[0]
        for text in [episode.text] + episode.meta["statements"]:
            assert not contains_term(normalize_tokens(text), normalize_tokens(second.value))


def test_supersession_links_to_the_earlier_version_of_the_same_attribute():
    world = small_world()
    changed = [e for e in all_episodes(world) if e.supersedes]
    assert changed
    for episode in changed:
        new, old = world.fact_of_episode(episode.id), world.fact_of_episode(episode.supersedes[0])
        assert new.key == old.key and old.night < new.night and new.value != old.value
        assert new.attr in ("employer", "hometown", "instrument")


def test_supersession_rate_is_respected_and_zero_means_none():
    assert not [e for e in all_episodes(small_world(supersession_rate=0.0)) if e.supersedes]
    many = [e for e in all_episodes(small_world(supersession_rate=1.0)) if e.supersedes]
    few = [e for e in all_episodes(small_world(supersession_rate=0.25)) if e.supersedes]
    assert len(few) < len(many)


def test_named_entities_only_so_values_do_not_make_false_pairs():
    world = small_world()
    for episode in all_episodes(world):
        fact = world.fact_of_episode(episode.id)
        if fact.attr in ("instrument", "birth_year", "product"):
            assert fact.value not in episode.entities


def test_pair_templates_only_for_chain_links_in_either_order():
    world = small_world()
    person = world.persons[0].id
    first = world.versions(f"{person}.employer")[0]
    second = world.versions(f"{first.value_id}.hq_city")[0]
    a, b = world.episode(first.episode_id), world.episode(second.episode_id)
    assert world.pair_template_queries(a, b) and world.pair_template_queries(a, b) == world.pair_template_queries(b, a)
    unrelated = world.episode(world.versions(f"{person}.instrument")[0].episode_id)
    assert world.pair_template_queries(a, unrelated) == []


# ---- distractors --------------------------------------------------------------------------------------


def test_distractors_are_never_in_the_stream_and_contain_no_stream_answer():
    world = small_world()
    ids = {e.id for e in all_episodes(world)}
    pool = world.distractor_pool()
    assert len(pool) == 12
    forms = set()
    for episode in all_episodes(world):
        forms.update(world.answer_forms(world.fact_of_episode(episode.id)))
    for episode in pool:
        assert episode.id not in ids
        tokens = normalize_tokens(episode.text)
        for form in forms:
            assert not contains_term(tokens, normalize_tokens(form))


# ---- probes -------------------------------------------------------------------------------------------


def test_two_hop_has_same_night_and_cross_night_chains():
    world = small_world(n_persons=30, n_orgs=6, n_cities=6, nights=5)
    flags = {p.same_night for p in world.probes if p.kind == "two_hop"}
    assert flags == {True, False}


def test_two_hop_available_only_once_both_facts_have_arrived():
    world = small_world(n_persons=20)
    for probe in world.probes:
        if probe.kind == "two_hop":
            nights = [world.fact_of_episode(i).night for i in probe.episode_ids]
            assert probe.valid_from == max(nights)
            assert not probe.active(probe.valid_from - 1)


def test_superseded_fact_changes_kind_at_the_change_with_stale_recorded():
    world = small_world()
    changed = next(e for e in all_episodes(world) if e.supersedes)
    fact = world.fact_of_episode(changed.id)
    old = world.fact_of_episode(changed.supersedes[0])
    direct = next(p for p in world.probes if p.id == f"direct:{old.id}")
    supersede = next(p for p in world.probes if p.id == f"supersede:{fact.id}")
    assert direct.active(fact.night - 1) and not direct.active(fact.night)
    assert supersede.active(fact.night) and not supersede.active(fact.night - 1)
    assert set(world.answer_forms(old)) <= set(supersede.stale)
    assert set(world.answer_forms(fact)) <= set(supersede.gold)
    assert supersede.query == direct.query


def test_probes_cover_every_required_kind():
    kinds = {p.kind for p in small_world().probes}
    assert kinds >= {"direct", "paraphrase", "reverse", "alias", "two_hop", "supersede", "preservation", "false_recall"}
    assert kinds >= {"entrench_fact", "entrench_control"}


def test_gold_includes_aliases():
    world = small_world()
    org_fact = next(world.versions(key)[0] for key in world.fact_keys if key.endswith(".employer"))
    assert len(world.answer_forms(org_fact)) == 2  # "Velmora Works", "Velmora".
    reverse = next(p for p in world.probes if p.kind == "reverse" and "nickname" in p.id)
    subject = world.fact_of_episode(reverse.episode_ids[0])
    # The nickname reverse probe names the nickname, so echoing it back must not count.
    assert subject.value not in reverse.gold
    assert world.entities[subject.subject].name in reverse.gold


def test_false_recall_pool_is_fixed_from_night_zero():
    world = small_world()
    pool = [p for p in world.probes if p.kind == "false_recall"]
    assert pool and all(p.valid_from == 0 and p.valid_to is None and p.gold == () and p.neighbour for p in pool)


def test_entrench_pairs_match_on_value_attribute_and_wording():
    world = small_world()
    facts = {p.cluster: p for p in world.probes if p.kind == "entrench_fact"}
    controls = {p.cluster: p for p in world.probes if p.kind == "entrench_control"}
    assert facts and set(facts) == set(controls)
    for cluster, fact in facts.items():
        control = controls[cluster]
        assert fact.gold == control.gold and fact.vocab == control.vocab
        assert not set(fact.gold) & set(fact.stale)
        subject = world.entities[world.fact_of_episode(fact.episode_ids[0]).subject].name
        assert subject in fact.context
        assert control.episode_ids == ()
        # Same wording: replacing each subject leaves the same string.
        control_subject = next(name for name in world.control_persons if name in control.context)
        assert fact.context.replace(subject, "X") == control.context.replace(control_subject, "X")


def test_probe_selection_is_stable_capped_and_shared():
    world = small_world(n_persons=20)
    first = select_probes(world.probes, "direct", 3, 10, world.seed)
    assert len(first) == 10
    assert first == select_probes(world.probes, "direct", 3, 10, world.seed)
    assert {p.id for p in first} != {p.id for p in select_probes(world.probes, "direct", 3, 10, world.seed + 1)}
    assert len(select_probes(world.probes, "direct", 3, None, world.seed)) >= 10


# ---- end to end with the night --------------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", ["S_oracleq", "B3_replay"])
def test_a_real_night_runs_on_this_world(arm_name):
    tok = TinyTokenizer()
    world = small_world(n_persons=2, n_orgs=2, n_cities=2, n_regions=1, nights=2, n_distractors=8, n_entrench=2)
    cfg = small_config(get_arm(arm_name))
    state = fresh_gpt2_state(cfg, world, tok)
    fingerprint = state.handle.delta.base_fingerprint()
    report = run_night(state, cfg)
    assert report.ledger["spent"] <= cfg.budget.night_tokens
    assert state.handle.delta.base_fingerprint() == fingerprint
    if arm_name == "B3_replay":
        assert report.train["steps"] > 0 and len(state.store) == len(world.episodes(0))
