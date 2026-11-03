"""CHD against closed forms, reduced KL math, and null-relative thresholds."""
from __future__ import annotations

import pytest
import torch

from tiny import EOS, BigramStub, ContextCategoricalStub, TinyTokenizer, tiny_gpt2

from llmsleep.config import SelectConfig, SignalConfig
from llmsleep.handle import ModelHandle, PromptFormat
from llmsleep.ledger import ComputeLedger
from llmsleep.memory import DistractorSampler, Episode
from llmsleep.phases.consolidate import reduce_distribution, reduced_kl, residual_log_mass
from llmsleep.phases.queries import Candidate
from llmsleep.phases.select import (
    Thresholds,
    lexical_overlap,
    null_distribution,
    partition,
    thresholds_from_null,
    touching_episodes,
)
from llmsleep.signals import SignalCache, chd, chd_safe, raw_kl_nocontext

ALPHABET = "abcd"
ID = {ch: i + 2 for i, ch in enumerate(ALPHABET)}  # a=2, b=3, c=4, d=5; pad=0, eos=1
VOCAB = len(ALPHABET) + 2


def _laws() -> tuple[torch.Tensor, torch.Tensor]:
    """Two known token laws. Index order: pad, eos, a, b, c, d."""
    q_e = torch.tensor([0.02, 0.20, 0.40, 0.18, 0.10, 0.10], dtype=torch.float64)
    q_d = torch.tensor([0.03, 0.10, 0.17, 0.40, 0.20, 0.10], dtype=torch.float64)
    return q_e, q_d


def _stub_handle(ledger: ComputeLedger | None = None) -> ModelHandle:
    """The context's single character is the row's first token, which fixes the token law."""
    q_e, q_d = _laws()
    table = torch.zeros(VOCAB, VOCAB, dtype=torch.float32)
    table[ID["a"]] = q_e.log().float()  # Context "a" is e.
    table[ID["b"]] = q_d.log().float()  # Context "b" is d.
    table[ID["c"]] = q_e.log().float()
    fmt = PromptFormat(template="{context}{query}", empty_context="c")
    return ModelHandle(
        ContextCategoricalStub(table), TinyTokenizer(ALPHABET), "cpu", ledger, prompt_format=fmt, batch_size=64
    )


def _token_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * (p / q).log()).sum())


def _expected_prefix_kl(m: int) -> float:
    """With iid tokens and stopping at EOS or m tokens: E[min(L, m)] * KL per token.

    By the chain rule the sequence KL is the sum over t of P(still running at t) * KL(q_e || q_d), since
    the two conditionals are the same token laws at every step. P(still running at t) = (1 - e)^(t-1).
    """
    q_e, q_d = _laws()
    e = float(q_e[EOS])
    return sum((1 - e) ** t for t in range(m)) * _token_kl(q_e, q_d)


@pytest.mark.parametrize("m", [1, 3, 60])
def test_chd_matches_closed_form_with_eos_and_truncation(m):
    handle = _stub_handle()
    k = 4000
    result = chd(handle, ["d"], ["a"], ["b"], k, m, torch.Generator().manual_seed(11))
    expected = _expected_prefix_kl(m)
    assert abs(float(result.values[0]) - expected) < 5 * float(result.stderr[0]) + 1e-6


def test_truncated_samples_mean_a_prefix_kl_which_grows_with_m():
    assert _expected_prefix_kl(1) < _expected_prefix_kl(3) < _expected_prefix_kl(60)
    handle = _stub_handle()
    samples = chd(handle, ["d"], ["a"], ["b"], 200, 3, torch.Generator().manual_seed(12)).samples[0]
    lengths = {len(s.token_ids) for s in samples}
    assert max(lengths) == 3
    for s in samples:
        if s.finished:
            assert s.token_ids[-1] == EOS  # EOS counted on both sides.
        else:
            assert len(s.token_ids) == 3 and EOS not in s.token_ids


def test_chd_is_zero_for_identical_laws_and_exact_per_sample():
    handle = _stub_handle()
    result = chd(handle, ["d"], ["a"], ["a"], 16, 4, torch.Generator().manual_seed(13))
    assert float(result.values[0]) == pytest.approx(0.0, abs=1e-5)
    # Contexts "a" and "c" have the same law through different tokens: content, not wrapper, decides.
    result = chd(handle, ["d"], ["a"], ["c"], 16, 4, torch.Generator().manual_seed(14))
    assert float(result.values[0]) == pytest.approx(0.0, abs=1e-5)


def test_chd_is_near_zero_with_same_context_on_a_real_model():
    tok = TinyTokenizer()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)
    result = chd(handle, ["q?"], ["same"], ["same"], 4, 4, torch.Generator().manual_seed(15))
    assert abs(float(result.values[0])) < 1e-3


def test_both_log_probs_cover_the_same_tokens():
    handle = _stub_handle()
    q_e, q_d = _laws()
    result = chd(handle, ["d"], ["a"], ["b"], 8, 3, torch.Generator().manual_seed(16))
    ratios = []
    for sample in result.samples[0]:
        lp_e = float(sum(q_e[t].log() for t in sample.token_ids))
        lp_d = float(sum(q_d[t].log() for t in sample.token_ids))
        assert sample.logprob == pytest.approx(lp_e, abs=1e-4)
        ratios.append(lp_e - lp_d)
    assert float(result.values[0]) == pytest.approx(sum(ratios) / len(ratios), abs=1e-4)


def test_per_token_variant_divides_each_sample_by_its_length():
    handle = _stub_handle()
    q_e, q_d = _laws()
    result = chd(handle, ["d"], ["a"], ["b"], 8, 3, torch.Generator().manual_seed(17))
    per_token = []
    for sample in result.samples[0]:
        ratio = float(sum((q_e[t] / q_d[t]).log() for t in sample.token_ids))
        per_token.append(ratio / len(sample.token_ids))
    assert float(result.per_token[0]) == pytest.approx(sum(per_token) / len(per_token), abs=1e-4)


def test_raw_kl_uses_the_closed_condition():
    handle = _stub_handle()
    # Closed renders as "c" + "d", and c has the same law as e, so the raw KL here is zero.
    result = raw_kl_nocontext(handle, ["d"], ["a"], 8, 3, torch.Generator().manual_seed(18))
    assert float(result.values[0]) == pytest.approx(0.0, abs=1e-5)


def test_cache_means_budget_is_not_paid_twice():
    ledger = ComputeLedger()
    handle = _stub_handle(ledger)
    cache = SignalCache()
    generator = torch.Generator().manual_seed(19)
    first = chd(handle, ["d"], ["a"], ["b"], 4, 2, generator, cache)
    spent = ledger.unmetered_forward
    assert spent > 0
    state = generator.get_state()
    second = chd(handle, ["d"], ["a"], ["b"], 4, 2, generator, cache)
    assert ledger.unmetered_forward == spent
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(first.values, second.values)
    # A new distractor reuses the samples and pays only for the d-side scoring.
    third = chd(handle, ["d"], ["a"], ["c"], 4, 2, generator, cache)
    assert third.samples[0] == first.samples[0]
    assert torch.equal(generator.get_state(), state)
    assert ledger.unmetered_forward > spent


def test_cache_is_keyed_on_banks():
    from llmsleep.delta import DeltaModel

    tok = TinyTokenizer()
    delta = DeltaModel(tiny_gpt2(tok.vocab_size), rank=2)
    handle = ModelHandle(delta, tok, "cpu", batch_size=4)
    cache = SignalCache()
    generator = torch.Generator().manual_seed(20)
    with handle.banks("committed"):
        chd(handle, ["q"], ["e"], ["d"], 2, 2, generator, cache)
        state = generator.get_state()
    delta.bump_version()  # The weights changed, so old samples must not be reused.
    with handle.banks("committed"):
        chd(handle, ["q"], ["e"], ["d"], 2, 2, generator, cache)
    assert not torch.equal(generator.get_state(), state)


def test_chd_safe_keeps_finished_chunks_on_exhaustion():
    ledger = ComputeLedger()
    handle = _stub_handle(ledger)
    items = [("d", "a", "b")] * 6
    # Measure one chunk's cost on an unmetered ledger first.
    probe_ledger = ComputeLedger()
    probe = _stub_handle(probe_ledger)
    chd_safe(probe, items[:2], 2, 2, torch.Generator().manual_seed(21), chunk=2)
    cost = probe_ledger.unmetered_forward
    ledger.begin_night(cost, {"signals": 1.0})
    with ledger.stage("signals"):
        values, exhausted = chd_safe(handle, items, 2, 2, torch.Generator().manual_seed(21), chunk=2)
    assert exhausted
    assert values[0] is not None and values[1] is not None
    assert values[2:] == [None] * 4


# ---- reduced KL for stored targets ----------------------------------------------------------------


def _distribution(seed: int, vocab: int = 12) -> torch.Tensor:
    return torch.log_softmax(torch.randn(vocab, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_reduced_kl_equals_the_coarsened_kl_computed_by_hand():
    p, q = _distribution(1).unsqueeze(0), _distribution(2).unsqueeze(0)
    idx, top, res = reduce_distribution(p, 4)
    got = float(reduced_kl(q, idx, top, res))
    keep = idx[0].tolist()
    rest = [i for i in range(12) if i not in keep]
    pe, qe = p[0].exp(), q[0].exp()
    by_hand = sum(float(pe[i] * (pe[i] / qe[i]).log()) for i in keep)
    p_tail, q_tail = float(pe[rest].sum()), float(qe[rest].sum())
    by_hand += p_tail * float(torch.tensor(p_tail / q_tail).log())
    assert got == pytest.approx(by_hand, abs=1e-5)


def test_reduced_kl_is_zero_at_the_teacher_and_at_most_the_full_kl():
    p, q = _distribution(3).unsqueeze(0), _distribution(4).unsqueeze(0)
    idx, top, res = reduce_distribution(p, 3)
    assert float(reduced_kl(p, idx, top, res)) == pytest.approx(0.0, abs=1e-5)
    full = float((p.exp() * (p - q)).sum())
    reduced = float(reduced_kl(q, idx, top, res))
    assert 0.0 <= reduced <= full + 1e-6


def test_reduced_kl_ignores_only_the_shape_of_the_tail():
    p = _distribution(5).unsqueeze(0)
    idx, top, res = reduce_distribution(p, 3)
    # Same head and tail total, but all the tail's mass moved onto one tail token.
    rest = [i for i in range(12) if i not in idx[0].tolist()]
    q = p.exp().clone()
    tail_mass = q[0, rest].sum()
    q[0, rest] = 1e-6
    q[0, rest[0]] = tail_mass - 1e-6 * (len(rest) - 1)
    assert float(reduced_kl(q.log(), idx, top, res)) == pytest.approx(0.0, abs=1e-4)
    # Moving mass from head to tail is penalised, unlike a sampled-token estimate.
    moved = p.exp().clone()
    moved[0, idx[0, 0]] *= 0.1
    moved = moved / moved.sum()
    assert float(reduced_kl(moved.log(), idx, top, res)) > 1e-3


def test_reduced_kl_gradient_is_finite_with_a_vanishing_tail():
    logits = torch.full((1, 5), -200.0)
    logits[0, 0] = 200.0  # All the teacher's mass on one token: the residual underflows.
    p = torch.log_softmax(logits, dim=-1)
    idx, top, res = reduce_distribution(p, 2)
    student = torch.zeros(1, 5, requires_grad=True)
    loss = reduced_kl(torch.log_softmax(student, dim=-1), idx, top, res).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()


def test_top_k_at_vocab_size_still_leaves_a_tail():
    p = _distribution(6, vocab=4).unsqueeze(0)
    idx, top, res = reduce_distribution(p, 99)
    assert idx.shape == (1, 3)
    assert float(torch.logsumexp(torch.cat([top, res.unsqueeze(-1)], dim=-1), dim=-1)) == pytest.approx(0.0, abs=1e-5)


def test_residual_log_mass_is_accurate_when_the_tail_is_tiny():
    p = torch.log_softmax(torch.tensor([[30.0, 0.0, 0.0, 0.0]]), dim=-1)
    idx = torch.tensor([[0]])
    # 1 - exp(top) would give 0 in float32. The tail is really 3 e^-30.
    expected = torch.tensor(3.0).log() - 30.0
    assert float(residual_log_mass(p, idx)) == pytest.approx(float(expected), abs=1e-3)


# ---- thresholds and partition ---------------------------------------------------------------------


def _candidate(query: str = "How many kites has Mira?", context: str = "Mira has kites.", distractor: str | None = "x") -> Candidate:
    return Candidate(query, ("e",), ("Mira",), context=context, distractor=distractor)


def test_thresholds_are_relative_to_the_null():
    cfg = SelectConfig(integrate_null_q=0.9, anchor_null_q=0.5, gate_null_q=0.5, min_null=4)
    null = [float(i) for i in range(11)]
    t = thresholds_from_null(null, cfg)
    assert t.source == "quantile" and t.null_size == 11
    assert t.integrate == pytest.approx(9.0) and t.anchor == pytest.approx(5.0) and t.gate == pytest.approx(5.0)
    shifted = thresholds_from_null([v + 100 for v in null], cfg)
    assert shifted.integrate == pytest.approx(109.0)


def test_tiny_null_falls_back_to_extremes_and_empty_null_selects_nothing():
    cfg = SelectConfig(min_null=8)
    t = thresholds_from_null([1.0, 2.0, 3.0], cfg)
    assert t.source == "extremes" and t.integrate == 3.0 and t.anchor == 1.0 and t.gate == 1.0
    empty = thresholds_from_null([], cfg)
    assert empty.source == "none" and empty.integrate is None
    out = partition([_candidate()], [5.0], empty)
    assert out.integrate == [] and out.anchor == []
    assert out.discarded["no_null"] == 1


def _pool_candidate() -> Candidate:
    return Candidate("Who is Zed?", ("old",), ("Zed",), context="Zed sails.", distractor="x", pool=True)


def test_partition_routes_each_case_and_counts_every_discard():
    t = Thresholds(integrate=2.0, anchor=1.0, gate=1.0, source="quantile", null_size=20)
    anchor = _pool_candidate()
    cases = [
        (_candidate(), 3.0),  # integrate
        (anchor, 0.5),  # anchor: from the pool, and low
        (_pool_candidate(), 1.5),  # pool, not low
        (_pool_candidate(), 3.0),  # pool, high: pool queries never integrate
        (_candidate(), 0.5),  # tonight's query, low: never an anchor
        (_candidate(), 1.5),  # between thresholds
        (_candidate(context=""), 3.0),  # no context
        (_candidate(distractor=None), 3.0),  # no distractor
        (_candidate(), None),  # budget ran out
    ]
    out = partition([c for c, _ in cases], [v for _, v in cases], t)
    assert len(out.integrate) == 1 and out.anchor == [anchor]
    assert dict(out.discarded) == {
        "pool_not_low": 2,
        "low_or_mid_chd": 2,
        "no_context": 1,
        "no_distractor": 1,
        "unscored": 1,
    }
    summary = out.summary()
    assert summary["integrate"] + summary["anchor"] + sum(summary["discarded"].values()) == len(cases)


def test_pairs_use_their_own_null():
    singles = Thresholds(integrate=5.0, anchor=1.0, gate=1.0, source="quantile", null_size=20)
    pairs = Thresholds(integrate=2.0, anchor=1.0, gate=1.0, source="quantile", null_size=20)
    pair = Candidate("Q?", ("a", "b"), ("X",), is_pair=True, context="a\nb", distractor="x")
    single = _candidate()
    out = partition([pair, single], [3.0, 3.0], singles, pairs)
    assert out.integrate == [pair] and out.discarded["low_or_mid_chd"] == 1


def test_partition_of_nothing_is_fine():
    out = partition([], [], Thresholds(1.0, 0.0, 0.0, "quantile", 9))
    assert out.summary() == {"integrate": 0, "anchor": 0, "discarded": {}}


def test_null_distribution_compares_two_different_unrelated_distractors(monkeypatch):
    import llmsleep.phases.select as select_module

    seen = []
    original = select_module.chd_safe

    def recording(handle, items, *args, **kwargs):
        seen.extend(items)
        return original(handle, items, *args, **kwargs)

    monkeypatch.setattr(select_module, "chd_safe", recording)
    handle = _stub_handle()
    cand = Candidate("d", ("e",), ("Mira",), context="a")
    pool = [Episode("x", "a", 0, entities=["X"]), Episode("y", "b", 0, entities=["Y"]), Episode("m", "a", 0, entities=["Mira"])]
    values, exhausted = null_distribution(
        handle, [cand], DistractorSampler(pool), SignalConfig(k=4, max_new_tokens=2), 1, torch.Generator().manual_seed(22)
    )
    assert len(values) == 1 and not exhausted
    assert {seen[0][1], seen[0][2]} == {"a", "b"}  # With second = first the null would be zero.


def test_pair_null_uses_two_episode_contexts(monkeypatch):
    import llmsleep.phases.select as select_module

    seen = []
    monkeypatch.setattr(select_module, "chd_safe", lambda h, items, *a, **k: (seen.extend(items), ([0.0], False))[1])
    pool = [Episode(str(i), "ab"[i % 2], 0, entities=[f"E{i}"]) for i in range(4)]
    pair = Candidate("d", ("p", "q"), ("Mira",), is_pair=True, context="a\nb")
    null_distribution(
        _stub_handle(), [pair], DistractorSampler(pool), SignalConfig(k=1, max_new_tokens=1), 1, torch.Generator().manual_seed(1)
    )
    assert seen[0][1].count("\n") == 1 and seen[0][2].count("\n") == 1


def test_null_distribution_with_no_distractors_or_candidates_is_empty():
    handle = _stub_handle()
    sampler = DistractorSampler([Episode("m", "a", 0, entities=["Mira"])])
    sig = SignalConfig(k=2, max_new_tokens=2)
    cand = Candidate("d", ("e",), ("Mira",), context="a")
    assert null_distribution(handle, [cand], sampler, sig, 2, torch.Generator()) == ([], False)
    assert null_distribution(handle, [], sampler, sig, 2, torch.Generator()) == ([], False)


def test_chd_safe_reports_per_token_by_default():
    handle = _stub_handle()
    items = [("d", "a", "b")]
    per_token, _ = chd_safe(handle, items, 8, 3, torch.Generator().manual_seed(23))
    expected = chd(handle, ["d"], ["a"], ["b"], 8, 3, torch.Generator().manual_seed(23))
    assert per_token[0] == pytest.approx(float(expected.per_token[0]))
    total, _ = chd_safe(handle, items, 8, 3, torch.Generator().manual_seed(23), per_token=False)
    assert total[0] == pytest.approx(float(expected.values[0]))


def test_chd_stops_at_a_newline_on_both_sides():
    """A1.9: only the answer counts. After the newline a raw model runs on, and that must not enter the signal."""
    tok = TinyTokenizer("ab\n")
    newline = 4
    table = torch.full((tok.vocab_size, tok.vocab_size), -50.0)
    table[:, newline] = 50.0  # Everything leads straight to a newline.
    table[newline, :] = -50.0
    table[newline, 2] = 50.0
    handle = ModelHandle(BigramStub(table), tok, "cpu", prompt_format=PromptFormat("{context}{query}"), batch_size=4)
    result = chd(handle, ["a"], ["a"], ["b"], 4, 5, torch.Generator().manual_seed(1))
    for sample in result.samples[0]:
        assert sample.token_ids == [newline] and sample.finished


def test_lexical_overlap_and_touching_episodes():
    assert lexical_overlap("What is the capital?", "the capital of it") == 1.0
    assert lexical_overlap("the of", "anything") == 0.0
    a = Episode("a", "Mira keeps kites.", 0, entities=["Mira"])
    b = Episode("b", "Foxes keep Mira away from kites.", 0, entities=["Fox"])
    c = Episode("c", "Lima has hills.", 0, entities=["Lima"])
    # Naming an entity ranks ahead of mere overlap; c matches neither.
    assert [e.id for e in touching_episodes("Who is Mira?", [c, b, a], 0.9)] == ["a", "b"]
    assert touching_episodes("Name a colour?", [a, b, c], 0.5) == []
