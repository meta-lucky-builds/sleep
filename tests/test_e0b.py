"""E0b: the candidates against closed forms, the label rule, the winner rule and the confirm guard. No downloads."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny import EOS, BigramStub, ContextCategoricalStub, TinyTokenizer, tiny_gpt2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_e0b  # noqa: E402

from llmsleep.config import make_generator  # noqa: E402
from llmsleep.eval.query_labels import EXCLUDED, POSITIVE, THIRD, fact_terms, label_for_episode, label_question  # noqa: E402
from llmsleep.handle import ModelHandle, PromptFormat, Sample  # noqa: E402
from llmsleep.signals import (  # noqa: E402
    CANDIDATES,
    CandidateSpec,
    DistractorMemo,
    _mix_result,
    chd,
    chd_mix,
    content_words,
    first_answer_token_entropy,
    grounding_rate,
    log_mean_exp,
    modal_agreement,
    score_candidates,
)
from llmsleep.stats import auroc, bootstrap_auroc, pearson  # noqa: E402
from llmsleep.worlds import SyntheticWorld, WorldConfig  # noqa: E402

ALPHABET = "abcd"
ID = {ch: i + 2 for i, ch in enumerate(ALPHABET)}
VOCAB = len(ALPHABET) + 2
# Index order: pad, eos, a, b, c, d.
LAW_E = torch.tensor([0.02, 0.20, 0.40, 0.18, 0.10, 0.10], dtype=torch.float64)
LAW_B = torch.tensor([0.03, 0.10, 0.17, 0.40, 0.20, 0.10], dtype=torch.float64)
LAW_C = torch.tensor([0.05, 0.25, 0.10, 0.10, 0.10, 0.40], dtype=torch.float64)


def _stub_handle() -> ModelHandle:
    """The context's single character is the row's first token, which fixes an iid token law: a=e, b and c distractors."""
    table = torch.zeros(VOCAB, VOCAB, dtype=torch.float32)
    table[ID["a"]] = LAW_E.log().float()
    table[ID["b"]] = LAW_B.log().float()
    table[ID["c"]] = LAW_C.log().float()
    fmt = PromptFormat(template="{context}{query}", empty_context="c")
    return ModelHandle(ContextCategoricalStub(table), TinyTokenizer(ALPHABET), "cpu", prompt_format=fmt, batch_size=64)


def _exact_mix_kl(law_e: torch.Tensor, laws: list[torch.Tensor], m: int) -> float:
    """By enumerating every answer of at most m tokens, stopping at EOS. The mixture is over whole answers."""
    total = 0.0

    def walk(log_e: float, log_ds: list[float], depth: int) -> None:
        nonlocal total
        for token in range(VOCAB):
            next_e = log_e + float(law_e[token].log())
            next_ds = [log_d + float(law[token].log()) for log_d, law in zip(log_ds, laws)]
            if token == EOS or depth + 1 == m:
                mixture = math.log(sum(math.exp(x) for x in next_ds) / len(next_ds))
                total += math.exp(next_e) * (next_e - mixture)
            else:
                walk(next_e, next_ds, depth + 1)

    walk(0.0, [0.0] * len(laws), 0)
    return total


# ---- CHD_mix ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3])
def test_chd_mix_matches_a_closed_form_with_eos_and_truncation(m):
    result = chd_mix(_stub_handle(), "d", "a", ["b", "c"], 4000, m, torch.Generator().manual_seed(31))
    expected = _exact_mix_kl(LAW_E, [LAW_B, LAW_C], m)
    assert abs(float(result.values[0]) - expected) < 5 * float(result.stderr[0]) + 1e-6
    assert expected > 0


def test_chd_mix_at_one_token_is_the_kl_to_the_mean_law():
    mean_law = (LAW_B + LAW_C) / 2
    closed = float((LAW_E * (LAW_E / mean_law).log()).sum())
    assert _exact_mix_kl(LAW_E, [LAW_B, LAW_C], 1) == pytest.approx(closed, abs=1e-9)


def test_a_mixture_is_not_the_token_by_token_mixture():
    """Over two tokens, the mean of products is not the product of means. The estimator must use the former."""
    per_token_mean = (LAW_B + LAW_C) / 2
    token_wise = _exact_mix_kl(LAW_E, [per_token_mean], 2)
    assert abs(_exact_mix_kl(LAW_E, [LAW_B, LAW_C], 2) - token_wise) > 1e-3


def test_chd_mix_with_one_distractor_is_chd():
    handle = _stub_handle()
    mixed = chd_mix(handle, "d", "a", ["b"], 8, 3, torch.Generator().manual_seed(32))
    plain = chd(handle, ["d"], ["a"], ["b"], 8, 3, torch.Generator().manual_seed(32))
    assert float(mixed.values[0]) == pytest.approx(float(plain.values[0]), abs=1e-5)
    assert float(mixed.per_token[0]) == pytest.approx(float(plain.per_token[0]), abs=1e-5)


def test_chd_mix_is_zero_when_every_distractor_has_the_law_of_e():
    table = torch.zeros(VOCAB, VOCAB, dtype=torch.float32)
    for row in ("a", "b", "c"):
        table[ID[row]] = LAW_E.log().float()
    fmt = PromptFormat(template="{context}{query}", empty_context="c")
    handle = ModelHandle(ContextCategoricalStub(table), TinyTokenizer(ALPHABET), "cpu", prompt_format=fmt, batch_size=8)
    result = chd_mix(handle, "d", "a", ["b", "c"], 16, 4, torch.Generator().manual_seed(33))
    assert float(result.values[0]) == pytest.approx(0.0, abs=1e-5)


def test_log_mean_exp_survives_where_exp_underflows():
    values = torch.tensor([-5000.0, -5001.0], dtype=torch.float64)
    assert float(values.exp().mean()) == 0.0
    expected = -5000.0 + math.log((1 + math.exp(-1)) / 2)
    assert float(log_mean_exp(values)) == pytest.approx(expected, abs=1e-9)
    samples = [Sample([2, 3], -10.0, True)]
    result = _mix_result(samples, [[-9000.0], [-9001.0]])
    value = -10.0 - (-9000.0 + math.log((1 + math.exp(-1)) / 2))
    assert float(result.values[0]) == pytest.approx(value, abs=1e-9)
    assert float(result.per_token[0]) == pytest.approx(value / 2, abs=1e-9)
    assert math.isnan(float(result.stderr[0]))


def test_one_terrible_distractor_cannot_inflate_the_mixture_by_more_than_log_m():
    samples = [Sample([2], -1.0, True), Sample([3], -2.0, True)]
    best = [-1.5, -2.5]
    awful = [-900.0, -900.0]
    against_best = float(_mix_result(samples, [best]).values[0])
    assert float(_mix_result(samples, [best, awful]).values[0]) == pytest.approx(against_best + math.log(2), abs=1e-9)


# ---- first answer token --------------------------------------------------------------------------------------


def _entropy(law: torch.Tensor) -> float:
    return float(-(law * law.log()).sum())


def _bigram_handle() -> tuple[ModelHandle, dict[str, int]]:
    tok = TinyTokenizer("ab \n")
    ids = {"a": 2, "b": 3, " ": 4, "\n": 5}
    table = torch.full((tok.vocab_size, tok.vocab_size), 1e-6).log()
    table[ids["b"], :] = torch.tensor([0.01, 0.01, 0.02, 0.02, 0.93, 0.01]).log()  # After b: mostly a space.
    table[ids[" "], :] = torch.tensor([0.01, 0.01, 0.60, 0.20, 0.17, 0.01]).log()  # After a space: a or b.
    table[ids["a"], :] = torch.tensor([0.01, 0.01, 0.02, 0.02, 0.01, 0.93]).log()  # After a: end of answer.
    fmt = PromptFormat("{context}{query}")
    return ModelHandle(BigramStub(table), tok, "cpu", prompt_format=fmt, batch_size=4), ids


def test_first_answer_token_skips_a_lone_space_and_renormalises():
    handle, ids = _bigram_handle()
    expected = _entropy(torch.tensor([0.75, 0.25], dtype=torch.float64))
    assert first_answer_token_entropy(handle, [ids["b"]], 2) == pytest.approx(expected, abs=1e-5)
    assert handle.ledger.unmetered_forward == 1 + 2  # A pass over the prompt, then over the prompt and the space.


def test_first_answer_token_takes_the_raw_distribution_when_it_may_not_skip():
    handle, ids = _bigram_handle()
    # Only a and b have letters: equal masses here, and the space's mass is dropped.
    assert first_answer_token_entropy(handle, [ids["b"]], 0) == pytest.approx(math.log(2), abs=1e-5)
    assert handle.ledger.unmetered_forward == 1


def test_first_answer_token_does_not_walk_past_the_end_of_the_answer():
    handle, ids = _bigram_handle()
    assert first_answer_token_entropy(handle, [ids["a"]], 2) == pytest.approx(math.log(2), abs=1e-5)
    assert handle.ledger.unmetered_forward == 1


def test_content_mask_is_letters_and_digits_only():
    tok = TinyTokenizer("a1 ?\n")
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=2)
    mask = handle.content_token_mask(tok.vocab_size).tolist()
    assert mask == [False, False, True, True, False, False, False]


def test_entropy_drop_is_exact_on_a_stub():
    handle = _stub_handle()
    scores = score_candidates(handle, "d", "unused", "a", ["b"], ["entropy_drop"], CandidateSpec(k=2, max_new_tokens=2, mix_k=1, mix_m=1), lambda p: torch.Generator().manual_seed(1))
    content_e = LAW_E[2:] / LAW_E[2:].sum()
    content_d = LAW_B[2:] / LAW_B[2:].sum()
    assert scores.values["entropy_drop"] == pytest.approx(_entropy(content_d) - _entropy(content_e), abs=1e-5)
    assert scores.costs["entropy_drop"] == 4  # Two tokens on each side, and no sampling.


# ---- grounding and agreement ----------------------------------------------------------------------------------


def test_content_words_are_case_folded_whole_words_without_stopwords():
    assert content_words("The Oboe was played in 1975, by Mira's friend.") == {"oboe", "played", "1975", "mira", "friend"}


def test_grounding_matches_on_word_boundaries_only():
    target = content_words("Dara plays the Paris") - content_words("Where is Dara?")
    assert target == {"plays", "paris"}
    assert grounding_rate(["PARIS!", "Parisian", "nothing"], target) == pytest.approx(1 / 3)
    assert grounding_rate(["Paris"], set()) == 0.0
    assert grounding_rate([], target) == 0.0


def test_agreement_is_on_normalised_first_answers():
    assert modal_agreement(["The oboe.", "oboe", "A lute", "oboe\nQuestion: and?"]) == pytest.approx(3 / 4)
    assert modal_agreement(["x", "y"]) == pytest.approx(0.5)
    assert modal_agreement([]) == 0.0


def test_grounding_and_agreement_use_the_same_episode_words_on_both_sides(monkeypatch):
    """On the distractor side the question is whether its answers produced the episode's words, not its own."""
    tok = TinyTokenizer()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)
    scripted = {"e": ["oboe", "oboe"], "d": ["oboe", "tide"]}

    def sample(prompts, n, *args, **kwargs):
        side = "e" if "EPISODE" in tok.decode(prompts[0]) else "d"
        return [[Sample(tok(text, add_special_tokens=False)["input_ids"], -1.0, True) for text in scripted[side][:n]]]

    monkeypatch.setattr(handle, "sample", sample)
    spec = CandidateSpec(k=2, max_new_tokens=2, mix_k=1, mix_m=1)
    scores = score_candidates(
        handle, "Q?", "Dara plays the oboe", "EPISODE", ["tides rise"], ["grounding", "agreement"], spec,
        lambda p: torch.Generator().manual_seed(1),
    )
    assert scores.values["grounding"] == pytest.approx(1.0 - 0.5)  # "tide" is the distractor's word, not counted.
    assert scores.values["agreement"] == pytest.approx(1.0 - 0.5)


# ---- candidates together and alone ----------------------------------------------------------------------------


def _tiny_handle() -> ModelHandle:
    tok = TinyTokenizer()
    return ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)


SPEC = CandidateSpec(k=4, max_new_tokens=5, mix_k=2, mix_m=3, max_skip=2)
DISTRACTORS = ["Tides rise at dawn.", "A ferry runs weekly.", "Rain falls often."]


def _generator_for(purpose: str) -> torch.Generator:
    return make_generator(1, 0, purpose)


@pytest.mark.parametrize("name", CANDIDATES)
def test_a_candidate_gets_the_same_value_and_cost_alone_as_in_company(name):
    """So a shared run cannot flatter any candidate, and the confirm run computes what tuning computed."""
    query, text = "Who keeps kites?", "Mira keeps kites."
    together = score_candidates(_tiny_handle(), query, text, text, DISTRACTORS, CANDIDATES, SPEC, _generator_for)
    handle = _tiny_handle()
    alone = score_candidates(handle, query, text, text, DISTRACTORS, [name], SPEC, _generator_for)
    assert set(alone.values) == {name} and set(alone.costs) == {name}
    assert alone.values[name] == pytest.approx(together.values[name], abs=1e-4)
    assert alone.costs[name] == together.costs[name] == handle.ledger.unmetered_forward


def test_a_shared_run_really_pays_less_than_the_sum_of_its_parts():
    handle = _tiny_handle()
    text = "Mira keeps kites."
    scores = score_candidates(handle, "Who keeps kites?", text, text, DISTRACTORS, CANDIDATES, SPEC, _generator_for)
    assert handle.ledger.unmetered_forward < sum(scores.costs.values())


def test_chd_mix_costs_more_per_distractor():
    text = "Mira keeps kites."
    costs = []
    for size in (1, 3):
        spec = CandidateSpec(k=4, max_new_tokens=5, mix_k=2, mix_m=size)
        scores = score_candidates(_tiny_handle(), "Who keeps kites?", text, text, DISTRACTORS, ["chd_mix"], spec, _generator_for)
        costs.append(scores.costs["chd_mix"])
    assert costs[1] > costs[0]


def test_the_incumbent_is_literally_chd_on_the_same_samples():
    handle = _tiny_handle()
    text = "Mira keeps kites."
    samples: list = []
    original = handle.sample

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        samples.extend(out[0])
        return out

    handle.sample = recording
    scores = score_candidates(handle, "Who keeps kites?", text, text, DISTRACTORS, ["chd"], SPEC, _generator_for)
    handle.sample = original
    assert len(samples) == SPEC.k
    prompt = handle.encode_prompt(handle.format("Who keeps kites?", DISTRACTORS[0]))
    logp = handle.sequence_logprob([prompt] * SPEC.k, [s.token_ids for s in samples]).tolist()
    expected = sum((s.logprob - lp) / len(s.token_ids) for s, lp in zip(samples, logp)) / SPEC.k
    assert scores.values["chd"] == pytest.approx(expected, abs=1e-4)


def test_a_memo_reuses_the_distractor_side_but_still_charges_it_to_the_candidate():
    query, text = "Who keeps kites?", "Mira keeps kites."
    memo = DistractorMemo()
    handle = _tiny_handle()
    first = score_candidates(handle, query, text, text, DISTRACTORS, ["agreement", "entropy_drop"], SPEC, _generator_for, memo)
    spent = handle.ledger.unmetered_forward
    other = "Lima has hills."
    second = score_candidates(handle, query, other, other, DISTRACTORS, ["agreement", "entropy_drop"], SPEC, _generator_for, memo)
    paid = handle.ledger.unmetered_forward - spent
    fresh = _tiny_handle()
    alone = score_candidates(fresh, query, other, other, DISTRACTORS, ["agreement", "entropy_drop"], SPEC, _generator_for)
    assert second.costs == alone.costs and second.values == pytest.approx(alone.values, abs=1e-4)
    assert paid < fresh.ledger.unmetered_forward
    assert first.costs["agreement"] > 0


def test_unknown_candidates_and_too_few_distractors_are_refused():
    handle = _tiny_handle()
    with pytest.raises(ValueError):
        score_candidates(handle, "q", "e", "e", DISTRACTORS, ["nope"], SPEC, _generator_for)
    with pytest.raises(ValueError):
        score_candidates(handle, "q", "e", "e", DISTRACTORS[:1], ["chd_mix"], SPEC, _generator_for)
    with pytest.raises(ValueError):
        CandidateSpec(k=2, mix_k=3)


# ---- the label rule ---------------------------------------------------------------------------------------------


NAME = ["Dara Quilmont", "Quilmont", "Bobo", "Dara"]


@pytest.mark.parametrize(
    "query, attr, value, cls, reason",
    [
        ("Where does Dara Quilmont work?", "employer", ["Foo Labs", "Foo"], POSITIVE, "asks_value"),
        ("Which company does Dara work for?", "employer", ["Foo Labs", "Foo"], POSITIVE, "asks_value"),
        ("What is Quilmont's job?", "employer", ["Foo Labs", "Foo"], POSITIVE, "asks_value"),
        ("WHAT INSTRUMENT DOES DARA PLAY?", "instrument", ["oboe"], POSITIVE, "asks_value"),
        ("What musical instrument does Dara play?", "instrument", ["oboe"], POSITIVE, "asks_value"),
        ("What year was Dara born?", "birth_year", ["1975"], POSITIVE, "asks_value"),
        ("Where did Dara grow up?", "hometown", ["Zorb"], POSITIVE, "asks_value"),
        ("Where is Dara from?", "hometown", ["Zorb"], POSITIVE, "asks_value"),
        # The answer is in the question: neither class. Includes reversals, which are answerable.
        ("Does Dara work at Foo Labs?", "employer", ["Foo Labs", "Foo"], EXCLUDED, "contains_value"),
        ("Who works at Foo?", "employer", ["Foo Labs", "Foo"], EXCLUDED, "contains_value"),
        ("Who plays the OBOE?", "instrument", ["oboe"], EXCLUDED, "contains_value"),
        ("Was Dara born in 1975?", "birth_year", ["1975"], EXCLUDED, "contains_value"),
        # A word boundary: this is not the value.
        ("What does Dara play on the oboes?", "instrument", ["oboe"], EXCLUDED, "extra_terms"),
        # No subject: with the episode in context this may be answerable, so it is not a clean third class.
        ("Where does he work?", "employer", ["Foo Labs"], EXCLUDED, "same_topic_no_subject"),
        ("What is his job?", "employer", ["Foo Labs"], EXCLUDED, "same_topic_no_subject"),
        ("Does Dara work?", "employer", ["Foo Labs"], EXCLUDED, "yes_no"),
        ("How long has Dara worked?", "employer", ["Foo Labs"], EXCLUDED, "wrong_interrogative"),
        ("When did Dara start playing?", "instrument", ["oboe"], EXCLUDED, "wrong_interrogative"),
        ("Where was Dara born?", "birth_year", ["1975"], EXCLUDED, "wrong_interrogative"),
        ("Who is the boss at Dara's company?", "employer", ["Foo Labs"], EXCLUDED, "extra_terms"),
        ("What is the salary of Dara's manager?", "employer", ["Foo Labs"], EXCLUDED, "extra_terms"),
        ("Where does Dara work and what does Dara play?", "employer", ["Foo Labs"], EXCLUDED, "several_topics"),
        ("What does Dara do for a living?", "employer", ["Foo Labs"], EXCLUDED, "related_topic"),
        ("How old is Dara?", "birth_year", ["1975"], EXCLUDED, "related_topic"),
        ("Who is Dara?", "employer", ["Foo Labs"], EXCLUDED, "generic"),
        ("What does the text say about Dara?", "employer", ["Foo Labs"], EXCLUDED, "about_the_text"),
        # The episode does not answer these.
        ("What instrument does Dara play?", "employer", ["Foo Labs"], THIRD, "sibling_attribute"),
        ("When was Dara born?", "instrument", ["oboe"], THIRD, "sibling_attribute"),
        ("Where was Dara born?", "hometown", ["Zorb"], THIRD, "sibling_attribute"),
        ("What is Dara's favourite colour?", "employer", ["Foo Labs"], THIRD, "off_topic"),
        ("What is the capital of France?", "instrument", ["oboe"], THIRD, "off_topic"),
    ],
)
def test_label_rule(query, attr, value, cls, reason):
    label = label_question(query, attr, NAME, value)
    assert (label.cls, label.reason) == (cls, reason)


def test_where_only_names_a_location_attribute_for_a_subject_that_has_one():
    org = ["Foo Labs", "Foo"]
    assert label_question("Where is Foo Labs?", "hq_city", org, ["Zorb"]).cls == POSITIVE
    assert label_question("Where is Foo based?", "product", org, ["brass hinges", "brass", "hinges"]).cls == THIRD
    assert label_question("Where does Foo make things?", "product", org, ["brass"]).reason == "several_topics"
    assert label_question("Where is Zorb?", "region", ["Zorb"], ["Quil Valley", "Quil"]).cls == POSITIVE


def test_an_empty_question_is_excluded():
    assert label_question("?", "employer", NAME, ["x"]).reason == "empty"


def _labelled_world(seed: int = 8) -> SyntheticWorld:
    return SyntheticWorld(
        WorldConfig(
            n_persons=3, n_orgs=3, n_cities=3, n_regions=2, nights=1, supersession_rate=0.0, n_distractors=12,
            n_control_persons=1, n_entrench=0,
        ),
        seed,
    )


def test_every_forward_template_is_positive_and_every_reverse_one_contains_the_value():
    """The rule's sanity on what the world itself calls a question about this fact, in every attribute."""
    world = _labelled_world()
    attrs = set()
    for episode in world.episodes(0):
        fact = world.fact_of_episode(episode.id)
        attrs.add(fact.attr)
        forward = [q for q, a in episode.meta["qa"] if a == fact.value]
        reverse = [q for q, a in episode.meta["qa"] if a != fact.value]
        assert forward
        for query in forward:
            label = label_for_episode(world, query, episode.id)
            assert (label.cls, label.reason) == (POSITIVE, "asks_value"), (query, fact.attr)
        for query in reverse:
            assert label_for_episode(world, query, episode.id).reason == "contains_value", query
    assert attrs == {"employer", "hometown", "instrument", "birth_year", "invention", "nickname", "hq_city", "product", "region"}


def test_a_siblings_template_is_third_class_or_excluded_never_positive():
    world = _labelled_world()
    by_subject: dict = {}
    for episode in world.episodes(0):
        by_subject.setdefault(world.fact_of_episode(episode.id).subject, []).append(episode)
    thirds = 0
    for episodes in by_subject.values():
        for episode in episodes:
            for sibling in episodes:
                if sibling.id == episode.id:
                    continue
                label = label_for_episode(world, world.template_queries(sibling)[0], episode.id)
                assert label.cls != POSITIVE
                thirds += int(label.cls == THIRD)
    assert thirds > 0


def test_fact_terms_keep_a_nickname_out_of_the_subject_and_split_literal_values():
    world = _labelled_world()
    for episode in world.episodes(0):
        fact = world.fact_of_episode(episode.id)
        subject, value = fact_terms(world, fact)
        assert fact.value in value
        if fact.attr == "nickname":
            assert fact.value not in subject and world.entities[fact.subject].name in subject
        if fact.attr == "product":
            assert fact.value.split()[0] in value
        if world.entities[fact.subject].kind == "person":
            assert world.entities[fact.subject].name.split()[0] in subject


# ---- metrics and the winner -----------------------------------------------------------------------------------


def _entry(pos_neg, pos_third, cost, n=100):
    def part(value):
        return {"value": value, "n_first": n, "n_second": n, "ci_queries": None, "ci_episodes": None}

    return {"positive_vs_negative": part(pos_neg), "positive_vs_third": part(pos_third), "cost_per_query": cost}


def test_winner_is_the_highest_minimum_of_the_two_aurocs():
    metrics = {"chd": _entry(0.6, 0.6, 100.0), "good_at_one": _entry(0.99, 0.5, 100.0), "balanced": _entry(0.8, 0.75, 100.0)}
    winner, table = run_e0b.select_winner(metrics, "chd", 1.5, 30)
    assert winner == "balanced" and table["balanced"]["score"] == 0.75 and table["good_at_one"]["score"] == 0.5


def test_winner_must_cost_at_most_the_cap_times_the_incumbent():
    metrics = {"chd": _entry(0.6, 0.6, 100.0), "dear": _entry(0.95, 0.95, 151.0), "fair": _entry(0.7, 0.7, 150.0)}
    winner, table = run_e0b.select_winner(metrics, "chd", 1.5, 30)
    assert winner == "fair" and not table["dear"]["eligible"] and table["dear"]["cost_ratio"] == pytest.approx(1.51)


def test_the_incumbent_can_win_and_ties_go_to_the_cheaper_then_the_name():
    assert run_e0b.select_winner({"chd": _entry(0.7, 0.7, 100.0), "x": _entry(0.6, 0.9, 50.0)}, "chd", 1.5, 30)[0] == "chd"
    tied = {"chd": _entry(0.5, 0.5, 100.0), "b": _entry(0.7, 0.7, 90.0), "a": _entry(0.7, 0.7, 95.0), "c": _entry(0.7, 0.7, 90.0)}
    assert run_e0b.select_winner(tied, "chd", 1.5, 30)[0] == "b"


def test_no_winner_without_enough_data():
    metrics = {"chd": _entry(0.9, 0.9, 100.0, n=5)}
    winner, table = run_e0b.select_winner(metrics, "chd", 1.5, 30)
    assert winner is None and not table["chd"]["enough_data"]


def test_bootstrap_auroc_holds_the_estimate_and_keeps_units_together():
    positive = [float(i) for i in range(30)]
    negative = [i - 5.0 for i in range(30)]
    units = [f"u{i}" for i in range(30)]
    low, high = bootstrap_auroc(positive, negative, units, units, 300, seed=1)
    value = auroc([1] * 30 + [0] * 30, positive + negative)
    assert low <= value <= high
    assert bootstrap_auroc([], [1.0], [], ["a"], 10) is None
    # Perfect separation within every resample.
    assert bootstrap_auroc([5.0, 6.0], [1.0, 2.0], ["a", "b"], ["a", "b"], 50, seed=2) == (1.0, 1.0)


def test_clustered_intervals_are_wider_when_rows_of_one_unit_agree():
    rng = np.random.default_rng(3)
    effects = rng.normal(size=20)
    positive = [float(e + 0.3 + 0.01 * j) for e in effects for j in range(10)]
    negative = [float(x) for x in rng.normal(-0.3, 1.0, size=200)]
    per_row = [f"r{i}" for i in range(200)]
    per_group = [f"g{i // 10}" for i in range(200)]
    other = [f"n{i // 10}" for i in range(200)]
    other_rows = [f"nr{i}" for i in range(200)]
    narrow = bootstrap_auroc(positive, negative, per_row, other_rows, 400, seed=4)
    wide = bootstrap_auroc(positive, negative, per_group, other, 400, seed=4)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_pearson_handles_constant_input():
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 1.0], [2.0, 3.0]) is None and pearson([1.0], [1.0]) is None


def test_confirm_verdict_is_a2_3_as_written():
    assert run_e0b.confirm_verdict(_entry(0.85, 0.75, 1.0), 30)["passed"] is True
    assert run_e0b.confirm_verdict(_entry(0.84, 0.99, 1.0), 30)["passed"] is False
    assert run_e0b.confirm_verdict(_entry(0.99, 0.74, 1.0), 30)["passed"] is False
    short = run_e0b.confirm_verdict(_entry(0.99, 0.99, 1.0, n=5), 30)
    assert short["passed"] is False and short["enough_data"] is False
    assert "abandoned" in short["action"]


# ---- contexts ---------------------------------------------------------------------------------------------------


def test_unrelated_episode_shares_nothing_with_the_query_or_the_episode():
    world = _labelled_world()
    pool = world.episodes(0)
    episode = pool[0]
    query = f"Who is {episode.entities[0]}?"
    generator = torch.Generator().manual_seed(1)
    drawn = {run_e0b.unrelated_episode(query, episode, pool, generator).id for _ in range(30)}
    assert len(drawn) > 1 and episode.id not in drawn
    own = {x.casefold() for x in episode.entities}
    for other in pool:
        if other.id in drawn:
            assert not own & {x.casefold() for x in other.entities}
    assert run_e0b.unrelated_episode(query, episode, [episode], generator) is None


def test_distractors_differ_and_avoid_the_entities():
    world = _labelled_world()
    sampler = run_e0b.DistractorSampler(world.distractor_pool(), 0.5)
    out = run_e0b.draw_distractors(sampler, ["Nobody"], 30, 4, torch.Generator().manual_seed(1))
    assert len(set(out)) == 4
    assert run_e0b.draw_distractors(sampler, ["Nobody"], 30, 99, torch.Generator().manual_seed(1)) is None


# ---- the runs ---------------------------------------------------------------------------------------------------


def _config() -> dict:
    return {
        "experiment": "E0b",
        "seed_role": "dev",
        "model_id": "tiny/base",
        "world_seeds": [8, 9],
        "world": {
            "n_persons": 3, "n_orgs": 3, "n_cities": 3, "n_regions": 2, "nights": 1, "supersession_rate": 0.0,
            "n_distractors": 12, "n_control_persons": 1, "n_entrench": 0,
        },
        "batch_size": 8,
        "k": 2,
        "chd_tokens": 3,
        "chd_mix_k": 1,
        "chd_mix_m": 2,
        "max_skip": 1,
        "queries": {"per_episode": 3, "model_samples": 1, "model_max_new_tokens": 4, "temperature": 0.8},
        "max_episodes": 8,
        "quota": {"positive": 4, "third": 3},
        "template_quota": {"positive": 2, "third": 2},
        "samples_per_class": 2,
        "n_boot": 20,
        "cost_cap": 100.0,
        "min_class_n": 1,
    }


def factory(model_id, fp16):
    tok = TinyTokenizer()
    return tiny_gpt2(tok.vocab_size), tok


class ScriptedQueries:
    """Stands in for the model: for each episode a positive, a third-class question, a generic and a reversal."""

    def __init__(self, handle, cfg) -> None:
        self.handle = handle

    def generate(self, episodes, generator):
        episode = episodes[0]
        subject = episode.entities[0]
        questions = [episode.meta["qa"][0][0], f"What is the favourite colour of {subject}?", f"Who is {subject}?"]
        if len(episode.meta["qa"]) > 1:
            questions.append(episode.meta["qa"][-1][0])
        return questions


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(run_e0b, "ModelQueryGen", ScriptedQueries)


@pytest.fixture
def tuned(tmp_path, scripted):
    config = _config()
    return config, run_e0b.run_tune(config, tmp_path, factory, "cpu"), tmp_path


def test_tune_reports_every_candidate_every_class_and_its_labels(tuned):
    config, result, out = tuned
    stored = json.loads((out / "e0b_tune.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(result))
    assert result["world_seed"] == 8 and result["mode"] == "tune"
    assert set(result["metrics"]) == set(CANDIDATES) == set(result["template_metrics"])
    classes = {row["cls"] for row in result["rows"]}
    assert classes == {"positive", "negative", "third"}
    for entry in result["metrics"].values():
        assert entry["positive_vs_negative"]["n_first"] == 4 == entry["positive_vs_negative"]["n_second"]
        assert entry["positive_vs_third"]["n_second"] == 3
        assert entry["cost_per_query"] > 0
        assert entry["positive_vs_negative"]["ci_queries"] is not None
    assert result["metrics"]["entropy_drop"]["cost_per_query"] < result["metrics"]["chd"]["cost_per_query"]
    labels = result["labels"]
    assert labels["by_class"]["positive"] >= 4 and labels["by_class"]["third"] >= 3
    assert labels["by_reason"]["excluded:generic"] >= 1 and labels["by_reason"]["excluded:contains_value"] >= 1
    # Samples of every class and reason, to read by eye.
    assert set(labels["samples"]) == set(labels["by_reason"])
    for samples in labels["samples"].values():
        assert 1 <= len(samples) <= 2 and {"query", "episode", "attr"} == set(samples[0])
    assert result["template_labels"]["by_class"] == {"positive": 2, "third": 2}
    assert {row["cls"] for row in result["template_rows"]} == {"positive", "negative", "third"}


def test_a_negative_is_the_same_query_under_another_episode(tuned):
    _, result, _ = tuned
    positives = {r["query"]: r for r in result["rows"] if r["cls"] == "positive"}
    negatives = [r for r in result["rows"] if r["cls"] == "negative"]
    assert {r["query"] for r in negatives} == set(positives)
    for row in negatives:
        assert row["context"] != positives[row["query"]]["context"]
        assert row["episode_id"] == positives[row["query"]]["episode_id"]  # The unit stays with the question.


def test_tune_writes_the_winner_its_rule_chose(tuned):
    config, result, out = tuned
    expected, _ = run_e0b.select_winner(result["metrics"], "chd", config["cost_cap"], config["min_class_n"])
    assert result["winner"] == expected in CANDIDATES
    assert (out / "winner.txt").read_text(encoding="utf-8").strip() == expected
    assert "min(positive_vs_negative, positive_vs_third)" in result["rule"]


def test_the_run_stops_generating_when_its_quotas_are_met(tuned):
    _, result, _ = tuned
    assert result["generation"]["episodes_tried"] < 8


def test_confirm_runs_only_the_winner_on_the_other_seed(tuned, tmp_path_factory, monkeypatch):
    config, result, out = tuned
    winner = result["winner"]
    names: list = []
    original = run_e0b.score_candidates

    def recording(handle, query, e_text, ctx_e, distractors, wanted, *args, **kwargs):
        names.append(tuple(wanted))
        return original(handle, query, e_text, ctx_e, distractors, wanted, *args, **kwargs)

    monkeypatch.setattr(run_e0b, "score_candidates", recording)
    target = tmp_path_factory.mktemp("confirm")
    confirmed = run_e0b.run_confirm(winner, out / "e0b_tune.json", config, target, factory, "cpu")
    assert names and set(names) == {(winner,)}
    assert confirmed["world_seed"] == 9 and set(confirmed["metrics"]) == {winner}
    assert confirmed["world_fingerprint"] != result["world_fingerprint"]
    assert isinstance(confirmed["verdict"]["passed"], bool)
    assert confirmed["verdict"]["cost_ratio_to_tuning_incumbent"] > 0
    assert (target / "e0b_confirm.json").exists()
    with pytest.raises(run_e0b.ConfirmRefused):  # A2.3: tested once.
        run_e0b.run_confirm(winner, out / "e0b_tune.json", config, target, factory, "cpu")


def test_confirm_refuses_any_name_but_the_winner_before_loading_a_model(tuned, tmp_path_factory):
    config, result, out = tuned

    def must_not_load(model_id, fp16):
        raise AssertionError("a refused confirm run loaded a model")

    other = next(name for name in CANDIDATES if name != result["winner"])
    with pytest.raises(run_e0b.ConfirmRefused, match="not the tuning winner"):
        run_e0b.run_confirm(other, out / "e0b_tune.json", config, tmp_path_factory.mktemp("c"), must_not_load, "cpu")


def test_confirm_refuses_a_changed_config_a_wrong_seed_or_no_winner():
    config = _config()
    good = {"mode": "tune", "world_seed": 8, "config_hash": run_e0b.config_hash(config), "winner": "chd"}
    run_e0b.check_confirm("chd", good, config)
    changed = {**config, "k": 99}
    with pytest.raises(run_e0b.ConfirmRefused, match="config"):
        run_e0b.check_confirm("chd", good, changed)
    with pytest.raises(run_e0b.ConfirmRefused, match="seed"):
        run_e0b.check_confirm("chd", {**good, "world_seed": 9}, config)
    with pytest.raises(run_e0b.ConfirmRefused, match="no winner"):
        run_e0b.check_confirm("chd", {**good, "winner": None}, config)
    with pytest.raises(run_e0b.ConfirmRefused, match="not a tuning run"):
        run_e0b.check_confirm("chd", {**good, "mode": "confirm"}, config)


def test_an_instruct_model_is_refused():
    config = {**_config(), "model_id": "Qwen/Qwen2.5-0.5B-Instruct"}
    with pytest.raises(ValueError):
        run_e0b._load(config, "cpu", factory)


def test_the_real_config_is_seed_8_then_9_on_base_at_deployed_settings():
    config = json.loads((ROOT / "configs" / "e0b.json").read_text(encoding="utf-8"))
    deployed = json.loads((ROOT / "configs" / "e0.json").read_text(encoding="utf-8"))["deployed"]
    assert config["world_seeds"] == [8, 9] and config["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert config["k"] == deployed["k"] and config["chd_tokens"] == deployed["chd_tokens"]
    assert config["queries"]["per_episode"] == deployed["per_episode"]
    assert config["chd_mix_m"] == 4 and config["quota"]["positive"] == 150 and config["cost_cap"] == 1.5
    world = SyntheticWorld(WorldConfig(**config["world"]), 8)
    assert len(world.episodes(0)) >= config["max_episodes"]
    assert world.config.n_distractors >= config["chd_mix_m"]
    # CHD_mix must be able to meet the cap: see configs/README.md for the arithmetic.
    assert config["chd_mix_k"] < config["k"]


def test_the_workflow_runs_both_modes_with_the_committed_winner():
    text = (ROOT / ".github" / "workflows" / "e0.yml").read_text(encoding="utf-8")
    for tag in ("e0b-tune-*", "e0b-confirm-*"):
        assert tag in text
    assert "results/e0b/winner.txt" in text and "--confirm" in text and "--tune-output" in text
    assert "run_e0b.py --config configs/e0b.json" in text
    assert "e0b-${{ github.ref_name }}" in text  # The artifact.
    job = text.split("  e0b:")[1]
    assert "matrix" not in job  # One model, the one in the config.
