"""Token alignment under both padding sides, sampling, ledger counts and stops."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from tiny import EOS, PAD, BigramStub, TinyTokenizer, tiny_gpt2, tiny_qwen2

from llmsleep.delta import DeltaModel
from llmsleep.handle import (
    ModelHandle,
    PromptFormat,
    answer_token_mask,
    build_batch,
    continuation_logprobs,
    gather_answer_logits,
    gather_answer_targets,
    positions_from_mask,
)
from llmsleep.ledger import ComputeLedger, LedgerExhausted

# Hand-checked case: row 0 is prompt [5, 6] + answer [7, 8]; row 1 is prompt [5] + answer [9].
PROMPTS = [[5, 6], [5]]
CONTS = [[7, 8], [9]]


def _table(vocab: int = 10) -> torch.Tensor:
    """A non-symmetric bigram table with distinct entries."""
    return torch.arange(vocab * vocab, dtype=torch.float32).reshape(vocab, vocab).remainder(7.0) * 0.5


def _bigram_lp(table: torch.Tensor, previous: int, token: int) -> float:
    return float(torch.log_softmax(table[previous], dim=-1)[token])


def test_right_padding_layout_is_as_worked_by_hand():
    batch = build_batch(PROMPTS, CONTS, PAD, "right")
    assert batch.input_ids.tolist() == [[5, 6, 7, 8], [5, 9, PAD, PAD]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.answer_mask.tolist() == [[False, False, True, True], [False, True, False, False]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3], [0, 1, 1, 1]]


def test_left_padding_layout_is_as_worked_by_hand():
    batch = build_batch(PROMPTS, CONTS, PAD, "left")
    assert batch.input_ids.tolist() == [[5, 6, 7, 8], [PAD, PAD, 5, 9]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert batch.answer_mask.tolist() == [[False, False, True, True], [False, False, False, True]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]


def test_gather_selects_the_logit_before_each_answer_token():
    # Each logit row is tagged by its (batch, time) so we can see which ones came out.
    for side, expected in [("right", [(0, 1), (0, 2), (1, 0)]), ("left", [(0, 1), (0, 2), (1, 2)])]:
        batch = build_batch(PROMPTS, CONTS, PAD, side)
        logits = torch.zeros(2, 4, 3)
        for b in range(2):
            for t in range(4):
                logits[b, t, 0] = b
                logits[b, t, 1] = t
        picked = gather_answer_logits(logits, batch.answer_mask)
        assert [(int(r[0]), int(r[1])) for r in picked] == expected
        assert gather_answer_targets(batch.input_ids, batch.answer_mask).tolist() == [7, 8, 9]


@pytest.mark.parametrize("side", ["left", "right"])
def test_continuation_logprob_hand_checked(side):
    table = _table()
    batch = build_batch(PROMPTS, CONTS, PAD, side)
    logits = table[batch.input_ids]
    got = continuation_logprobs(logits, batch.input_ids, batch.answer_mask)
    # log p(7|6) + log p(8|7), then log p(9|5): the prompt's other tokens contribute nothing.
    row0 = _bigram_lp(table, 6, 7) + _bigram_lp(table, 7, 8)
    row1 = _bigram_lp(table, 5, 9)
    assert got.tolist() == pytest.approx([row0, row1], abs=1e-5)


def test_continuation_logprob_ignores_minus_inf_at_unselected_positions():
    table = _table()
    table[:, PAD] = float("-inf")  # A pad target would give -inf, and -inf * 0 would be NaN.
    batch = build_batch(PROMPTS, CONTS, PAD, "right")
    got = continuation_logprobs(table[batch.input_ids], batch.input_ids, batch.answer_mask)
    assert torch.isfinite(got).all()


def test_empty_prompt_is_rejected_since_no_logit_precedes_it():
    with pytest.raises(ValueError):
        answer_token_mask([0], [2], "right")


def test_mask_helpers_agree():
    batch = build_batch(PROMPTS, CONTS, PAD, "left")
    assert torch.equal(answer_token_mask([2, 1], [2, 1], "left"), batch.answer_mask)
    assert torch.equal(positions_from_mask(batch.attention_mask), batch.position_ids)


def _stub_handle(table: torch.Tensor, ledger: ComputeLedger | None = None) -> ModelHandle:
    return ModelHandle(BigramStub(table), TinyTokenizer("abcdefgh"), "cpu", ledger, batch_size=2)


def test_sequence_logprob_batches_and_charges_without_padding():
    table = _table()
    ledger = ComputeLedger()
    handle = _stub_handle(table, ledger)
    got = handle.sequence_logprob(PROMPTS, CONTS)
    assert got.tolist() == pytest.approx(
        [_bigram_lp(table, 6, 7) + _bigram_lp(table, 7, 8), _bigram_lp(table, 5, 9)], abs=1e-5
    )
    assert ledger.unmetered_forward == 6  # 4 + 2 real tokens; the 2 pads are free.
    assert ledger.unmetered_backward == 0


@pytest.mark.parametrize("builder", [tiny_gpt2, tiny_qwen2])
def test_padding_does_not_change_scores_on_real_architectures(builder):
    tok = TinyTokenizer()
    handle = ModelHandle(builder(tok.vocab_size), tok, "cpu", batch_size=4)
    long_prompt = list(range(2, 12))
    short_prompt = [3, 4]
    alone = handle.sequence_logprob([short_prompt], [[5, 6]])
    together = handle.sequence_logprob([long_prompt, short_prompt], [[7, 8, 9], [5, 6]])
    assert float(together[1]) == pytest.approx(float(alone[0]), abs=1e-4)
    # The other padding side, assembled by hand, gives the same number.
    batch = build_batch([long_prompt, short_prompt], [[7, 8, 9], [5, 6]], handle.pad_id, "left")
    with torch.no_grad():
        logits = handle._forward(batch.input_ids, batch.attention_mask, batch.position_ids).logits
    left = continuation_logprobs(logits, batch.input_ids, batch.answer_mask)
    assert left.tolist() == pytest.approx(together.tolist(), abs=1e-4)


def test_sample_needs_an_explicit_generator():
    handle = _stub_handle(_table())
    with pytest.raises(ValueError):
        handle.sample([[2]], 1, 2)


@pytest.mark.parametrize("builder", [tiny_gpt2, tiny_qwen2])
def test_sampling_is_deterministic_and_ignores_global_rng(builder):
    tok = TinyTokenizer()
    handle = ModelHandle(builder(tok.vocab_size), tok, "cpu", batch_size=3)

    def draw(global_seed: int):
        torch.manual_seed(global_seed)
        before = torch.get_rng_state()
        samples = handle.sample(["ab", "cdefg"], 2, 5, 1.0, torch.Generator().manual_seed(9))
        assert torch.equal(torch.get_rng_state(), before)
        return samples

    assert draw(1) == draw(2)


@pytest.mark.parametrize("builder", [tiny_gpt2, tiny_qwen2])
def test_cache_and_no_cache_paths_agree(builder):
    tok = TinyTokenizer()
    cached = ModelHandle(builder(tok.vocab_size), tok, "cpu", batch_size=4)
    uncached = ModelHandle(builder(tok.vocab_size), tok, "cpu", batch_size=4)
    uncached._has_cache = False
    a = cached.sample(["ab", "cdefg"], 2, 6, 1.0, torch.Generator().manual_seed(3))
    b = uncached.sample(["ab", "cdefg"], 2, 6, 1.0, torch.Generator().manual_seed(3))
    for row_a, row_b in zip(a, b):
        for sa, sb in zip(row_a, row_b):
            assert sa.token_ids == sb.token_ids
            assert sa.logprob == pytest.approx(sb.logprob, abs=1e-3)


def test_sample_logprob_equals_sequence_logprob():
    tok = TinyTokenizer()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)
    samples = handle.sample(["ab", "cdefg"], 2, 5, 1.0, torch.Generator().manual_seed(4))
    prompts, conts, recorded = [], [], []
    for prompt, row in zip(["ab", "cdefg"], samples):
        for sample in row:
            prompts.append(prompt)
            conts.append(sample.token_ids)
            recorded.append(sample.logprob)
    assert handle.sequence_logprob(prompts, conts).tolist() == pytest.approx(recorded, abs=1e-3)


def test_temperature_changes_draws_but_recorded_logprob_is_untempered():
    tok = TinyTokenizer()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", batch_size=4)
    sample = handle.sample(["ab"], 1, 3, 0.5, torch.Generator().manual_seed(5))[0][0]
    assert handle.sequence_logprob(["ab"], [sample.token_ids]).item() == pytest.approx(sample.logprob, abs=1e-3)
    greedy = handle.sample(["ab"], 1, 3, 0, torch.Generator().manual_seed(5))[0][0]
    assert len(greedy.token_ids) == 3 or greedy.finished


def _eos_table(vocab: int, stop_from: int) -> torch.Tensor:
    """Greedy-like: after `stop_from`, EOS is certain. After anything else, `stop_from` is certain."""
    table = torch.full((vocab, vocab), -50.0)
    table[:, stop_from] = 50.0
    table[stop_from, :] = -50.0
    table[stop_from, EOS] = 50.0
    return table


def test_eos_is_included_finished_is_set_and_truncation_is_marked():
    vocab = TinyTokenizer("abcdefgh").vocab_size
    stop = 3
    handle = ModelHandle(BigramStub(_eos_table(vocab, stop)), TinyTokenizer("abcdefgh"), "cpu", batch_size=2)
    out = handle.sample([[2]], 1, 5, 1.0, torch.Generator().manual_seed(0))[0][0]
    assert out.token_ids == [stop, EOS] and out.finished
    assert out.logprob == pytest.approx(0.0, abs=1e-3)
    short = handle.sample([[2]], 1, 1, 1.0, torch.Generator().manual_seed(0))[0][0]
    assert short.token_ids == [stop] and not short.finished


def test_sampling_charges_real_tokens_only_no_cache():
    vocab = TinyTokenizer("abcdefgh").vocab_size
    stop = 3
    ledger = ComputeLedger()
    handle = ModelHandle(BigramStub(_eos_table(vocab, stop)), TinyTokenizer("abcdefgh"), "cpu", ledger, batch_size=2)
    # Row 0: prompt [2, 2] draws 3 then EOS. Row 1: prompt [3] draws EOS at once, finishing at step 0.
    handle.sample([[2, 2], [3]], 1, 4, 1.0, torch.Generator().manual_seed(0))
    # First pass: 2 + 1 = 3 real tokens. Row 1 finishes, so its EOS is masked, and the re-fed sequences hold
    # 3 (row 0) + 1 (row 1) real tokens. That pass finishes row 0 and the loop ends without another forward.
    assert ledger.unmetered_forward == 3 + 4


def test_sampling_charges_one_token_per_unfinished_row_with_cache():
    tok = TinyTokenizer()
    ledger = ComputeLedger()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", ledger, batch_size=2)
    samples = handle.sample([[2, 3, 4], [5]], 1, 3, 1.0, torch.Generator().manual_seed(1))
    rows = [row[0] for row in samples]

    def running_after(step: int) -> int:
        """Rows that still need another forward once `step` tokens have been drawn."""
        return sum(1 for s in rows if len(s.token_ids) > step or (len(s.token_ids) == step and not s.finished))

    # The prompts, then one token per unfinished row after each of the first two draws. None after the last.
    assert ledger.unmetered_forward == 4 + running_after(1) + running_after(2)


class _FalsyCache:
    """Like an empty DynamicCache: an object that is falsy, so code must not test it for truth."""

    def __bool__(self) -> bool:
        return False


class _CachingStub(torch.nn.Module):
    """A bigram model that returns a cache, and demands exactly that object back with one new token."""

    def __init__(self, table: torch.Tensor, make_cache) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(table.clone(), requires_grad=False)
        self.make_cache = make_cache
        self.expected = None
        self.cached_calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None):
        if self.expected is not None:
            assert past_key_values is self.expected
            assert input_ids.size(1) == 1
            self.cached_calls += 1
        self.expected = self.make_cache() if use_cache else None
        return SimpleNamespace(logits=self.table[input_ids], past_key_values=self.expected)


@pytest.mark.parametrize("make_cache", [lambda: (("k", "v"),), _FalsyCache], ids=["legacy_tuple", "cache_object"])
def test_cache_is_passed_back_opaquely_in_any_form(make_cache):
    table = _table()
    stub = _CachingStub(table, make_cache)
    cached = ModelHandle(stub, TinyTokenizer("abcdefgh"), "cpu", batch_size=2)
    plain = ModelHandle(BigramStub(table), TinyTokenizer("abcdefgh"), "cpu", batch_size=2)
    a = cached.sample([[2, 3], [4]], 1, 4, 1.0, torch.Generator().manual_seed(2))
    b = plain.sample([[2, 3], [4]], 1, 4, 1.0, torch.Generator().manual_seed(2))
    assert stub.cached_calls > 0
    assert [s[0].token_ids for s in a] == [s[0].token_ids for s in b]


def test_model_that_declines_to_return_a_cache_falls_back_to_refeeding():
    stub = _CachingStub(_table(), lambda: None)
    handle = ModelHandle(stub, TinyTokenizer("abcdefgh"), "cpu", batch_size=2)
    out = handle.sample([[2, 3]], 1, 3, 1.0, torch.Generator().manual_seed(2))
    assert stub.cached_calls == 0 and out[0][0].token_ids


def test_cache_support_is_detected_through_kwargs_only_signatures():
    class KwargsOnly(torch.nn.Module):
        def __init__(self, with_config: bool) -> None:
            super().__init__()
            self.table = torch.nn.Parameter(_table(), requires_grad=False)
            if with_config:
                self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.table[input_ids], past_key_values=None)

    tok = TinyTokenizer("abcdefgh")
    assert ModelHandle(KwargsOnly(True), tok, "cpu")._has_cache
    assert not ModelHandle(KwargsOnly(False), tok, "cpu")._has_cache
    assert ModelHandle(KwargsOnly(True), tok, "cpu").sample([[2]], 1, 2, 1.0, torch.Generator())[0][0].token_ids


def test_forward_train_charges_both_directions():
    tok = TinyTokenizer()
    ledger = ComputeLedger()
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", ledger)
    batch = build_batch(PROMPTS, CONTS, handle.pad_id, "right")
    handle.forward_train(batch)
    assert ledger.unmetered_forward == 6 and ledger.unmetered_backward == 6


def test_format_uses_one_wrapper_for_all_conditions():
    fmt = PromptFormat("[{context}] {query} =")
    assert fmt.render("q", "ctx") == "[ctx] q ="
    assert fmt.render("q", None) == "[(none)] q ="
    tok = TinyTokenizer()
    handle = ModelHandle(BigramStub(_table(tok.vocab_size)), tok, "cpu", prompt_format=fmt)
    assert handle.format("q", "e").replace("e", "d") == handle.format("q", "d")


def test_encode_answer_prefixes_a_space_and_appends_eos():
    tok = TinyTokenizer()
    handle = ModelHandle(BigramStub(_table(tok.vocab_size)), tok, "cpu")
    ids = handle.encode_answer(" Paris ")
    assert tok.decode(ids) == " Paris"
    assert ids[-1] == EOS


def test_training_mode_keeps_dropout_off_and_restores_eval():
    tok = TinyTokenizer()
    model = tiny_gpt2(tok.vocab_size)
    handle = ModelHandle(model, tok, "cpu")
    with handle.training_mode():
        assert model.training
        assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert not model.training


@pytest.mark.parametrize("builder", [tiny_gpt2, tiny_qwen2])
def test_gradient_checkpointing_gives_the_same_gradients(builder):
    """Banks are fixed while training, so the recompute during backward sees the same function."""
    tok = TinyTokenizer()

    def gradients(checkpointing: bool):
        delta = DeltaModel(builder(tok.vocab_size), rank=2, seed=5)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(6)
            for layer in delta.layers.values():
                layer.fresh_B.copy_(torch.randn(layer.fresh_B.shape, generator=generator) * 0.01)
        handle = ModelHandle(delta, tok, "cpu")
        batch = build_batch(PROMPTS, CONTS, handle.pad_id, "right")
        with delta.banks("night"), handle.training_mode(checkpointing):
            logits = handle.forward_train(batch)
            answer = gather_answer_logits(logits, batch.answer_mask)
            loss = torch.nn.functional.cross_entropy(answer, gather_answer_targets(batch.input_ids, batch.answer_mask))
            loss.backward()
        return [p.grad.clone() for p in delta.fresh_parameters()]

    plain, checkpointed = gradients(False), gradients(True)
    assert any(g.abs().sum() > 0 for g in plain)
    for a, b in zip(plain, checkpointed):
        assert torch.allclose(a, b, atol=1e-5)


# ---- ledger ---------------------------------------------------------------------------------------


def test_ledger_caps_sum_to_allowance_and_stop_before_overrun():
    ledger = ComputeLedger()
    ledger.begin_night(100, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert ledger.caps == {"a": 50, "b": 30, "c": 20}
    with ledger.stage("b"):
        ledger.charge(forward=20, backward=10)
        with pytest.raises(LedgerExhausted) as info:
            ledger.charge(forward=1)
    assert info.value.stage == "b" and info.value.remaining == 0
    assert ledger.spent("b") == 30  # The failed charge was not recorded.
    assert "b" in ledger.exhausted
    with ledger.stage("a"):  # One stage's exhaustion does not touch another's allowance.
        ledger.charge(forward=50)


def test_ledger_renormalises_inactive_stages_and_rounding_is_exact():
    ledger = ComputeLedger()
    ledger.begin_night(101, {"train": 0.45, "audit": 0.10})
    assert sum(ledger.caps.values()) == 101
    assert ledger.caps["train"] > 4 * ledger.caps["audit"]


def test_ledger_unused_tokens_roll_forward_to_open_stages_only():
    ledger = ComputeLedger()
    ledger.begin_night(100, {"a": 0.5, "b": 0.25, "c": 0.25})
    with ledger.stage("a"):
        ledger.charge(forward=10)
    ledger.close_stage("a")
    assert ledger.caps == {"a": 10, "b": 45, "c": 45}
    assert sum(ledger.caps.values()) == 100
    with ledger.stage("a"), pytest.raises(LedgerExhausted):
        ledger.charge(forward=1)
    ledger.close_stage("a")  # Idempotent.
    ledger.close_stage("b")
    assert ledger.caps == {"a": 10, "b": 0, "c": 90}


def test_ledger_metered_charge_needs_a_stage_and_inactive_stages_raise():
    ledger = ComputeLedger()
    ledger.begin_night(10, {"train": 1.0})
    with pytest.raises(RuntimeError):
        ledger.charge(forward=1)
    with pytest.raises(KeyError):
        with ledger.stage("audit"):
            pass


def test_ledger_evaluation_is_exempt_and_tracked_separately():
    ledger = ComputeLedger()
    ledger.begin_night(1, {"train": 1.0})
    with ledger.stage("train"), ledger.exempt():
        ledger.charge(forward=1000, backward=5)
    assert ledger.spent() == 0
    assert ledger.exempt_forward == 1000 and ledger.exempt_backward == 5
    assert ledger.summary()["exempt_forward"] == 1000


def test_handle_stops_on_exhaustion_before_doing_the_work():
    tok = TinyTokenizer()
    ledger = ComputeLedger()
    ledger.begin_night(3, {"signals": 1.0})
    handle = ModelHandle(tiny_gpt2(tok.vocab_size), tok, "cpu", ledger)
    with ledger.stage("signals"):
        with pytest.raises(LedgerExhausted):
            handle.sequence_logprob([[2, 3, 4]], [[5]])  # 4 tokens > 3.
    assert ledger.spent() == 0
    with ledger.stage("signals"):
        assert math.isfinite(float(handle.sequence_logprob([[2, 3]], [[5]])[0]))
    assert ledger.spent() == 3
