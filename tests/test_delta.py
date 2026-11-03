"""Low-rank banks: math against dense SVD, selection, rollback, and base immutability."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from tiny import tiny_gpt2, tiny_qwen2

from llmsleep.delta import COMMITTED, FRESH, DeltaModel, svd_of_product


class _Net(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.lm_head = nn.Linear(out_features, 3)

    def forward(self, x):
        return self.lm_head(self.fc(x))


def _net(seed: int = 0, in_features: int = 6, out_features: int = 5) -> _Net:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _Net(in_features, out_features)


def _randomise_fresh(delta: DeltaModel, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in delta.layers.values():
            layer.fresh_A.copy_(torch.randn(layer.fresh_A.shape, generator=generator))
            layer.fresh_B.copy_(torch.randn(layer.fresh_B.shape, generator=generator))


def _x(in_features: int = 6, batch: int = 4, seed: int = 2) -> torch.Tensor:
    return torch.randn(batch, in_features, generator=torch.Generator().manual_seed(seed))


def test_attach_targets_only_linears_excluding_head_and_embeddings():
    gpt2 = tiny_gpt2(20)
    delta = DeltaModel(gpt2, rank=2)
    names = set(delta.layers)
    assert "transformer.h.0.attn.c_attn" in names  # A Conv1D.
    assert "transformer.h.1.mlp.c_fc" in names
    assert not any("lm_head" in n or "wte" in n or "wpe" in n for n in names)
    qwen = tiny_qwen2(20)
    names = set(DeltaModel(qwen, rank=2).layers)
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.1.mlp.down_proj" in names
    assert not any("lm_head" in n or "embed_tokens" in n for n in names)


def test_conv1d_dims_are_in_then_out():
    gpt2 = tiny_gpt2(20)
    delta = DeltaModel(gpt2, rank=2)
    layer = delta.layers["transformer.h.0.attn.c_attn"]
    assert layer.in_features == 16
    assert layer.out_features == 48  # c_attn projects to 3 * n_embd.
    assert layer.fresh_A.shape == (16, 2)
    assert layer.fresh_B.shape == (48, 2)


@pytest.mark.parametrize("builder", [tiny_gpt2, tiny_qwen2])
def test_germline_and_zero_fresh_equal_the_untouched_model_exactly(builder):
    ids = torch.tensor([[3, 4, 5, 6]])
    reference = builder(20)(input_ids=ids).logits.detach()
    delta = DeltaModel(builder(20), rank=2)
    with delta.banks("germline"):
        assert torch.equal(delta.model(input_ids=ids).logits, reference)
    with delta.banks("night"):  # B = 0, so the fresh term is exactly zero.
        assert torch.equal(delta.model(input_ids=ids).logits, reference)


def test_forward_matches_dense_delta_including_scale():
    net = _net()
    delta = DeltaModel(net, rank=3, alpha=6.0)  # scale = 2
    layer = delta.layers["fc"]
    assert layer.scale == 2.0
    _randomise_fresh(delta)
    x = _x()
    base = layer.base(x).detach()
    with delta.banks("night"):
        out = layer(x).detach()
    dense = 2.0 * (layer.fresh_B @ layer.fresh_A.t()).detach()  # (out, in)
    assert torch.allclose(out, base + x @ dense.t(), atol=1e-5)


def test_svd_of_product_matches_dense_svd():
    generator = torch.Generator().manual_seed(3)
    left, right = torch.randn(9, 4, generator=generator), torch.randn(7, 4, generator=generator)
    factors = svd_of_product(left, right)
    dense = left @ right.t()
    expected = torch.linalg.svdvals(dense)[:4]
    assert factors.rank == 4
    assert torch.allclose(factors.S, expected, atol=1e-4)
    assert torch.allclose((factors.U * factors.S) @ factors.V.t(), dense, atol=1e-4)
    assert torch.allclose(factors.U.t() @ factors.U, torch.eye(4), atol=1e-4)
    assert torch.allclose(factors.V.t() @ factors.V, torch.eye(4), atol=1e-4)


def test_svd_of_product_handles_rank_deficiency_and_more_columns_than_rows():
    generator = torch.Generator().manual_seed(4)
    # k = 6 columns on a 3-row left factor: reduced QR gives a rectangular core.
    left, right = torch.randn(3, 6, generator=generator), torch.randn(8, 6, generator=generator)
    factors = svd_of_product(left, right)
    assert factors.rank == 3
    assert torch.allclose((factors.U * factors.S) @ factors.V.t(), left @ right.t(), atol=1e-4)
    # A duplicated column has true rank 1, and the numerically zero values are dropped.
    column = torch.randn(64, 1, generator=generator)
    factors = svd_of_product(torch.cat([column, column], dim=1), torch.cat([column, column], dim=1))
    assert factors.rank == 1


def test_svd_of_product_rank_zero_and_all_zero():
    assert svd_of_product(torch.zeros(5, 0), torch.zeros(4, 0)).rank == 0
    factors = svd_of_product(torch.zeros(5, 3), torch.zeros(4, 3))
    assert factors.rank == 0
    assert factors.U.shape == (5, 0) and factors.V.shape == (4, 0) and factors.S.shape == (0,)


def test_merge_equals_sum_of_banks_and_absorbs_scale():
    delta = DeltaModel(_net(), rank=3, alpha=6.0)
    layer = delta.layers["fc"]
    x = _x()
    _randomise_fresh(delta, seed=5)
    first = layer.dense_delta((FRESH,)).clone()
    delta.merge_fresh_into_committed()
    assert torch.allclose(layer.dense_delta((COMMITTED,)), first, atol=1e-4)
    assert not layer.fresh_B.any()

    _randomise_fresh(delta, seed=6)
    second = layer.dense_delta((FRESH,)).clone()
    with delta.banks("night"):
        before = layer(x).detach()
    delta.merge_fresh_into_committed()
    with delta.banks("committed"):
        after = layer(x).detach()
    assert torch.allclose(layer.dense_delta((COMMITTED,)), first + second, atol=1e-4)
    assert torch.allclose(after, before, atol=1e-4)


def test_merge_zero_fresh_on_rank_zero_stays_rank_zero_and_keeps_committed_bits():
    delta = DeltaModel(_net(), rank=3)
    delta.merge_fresh_into_committed()
    assert delta.layers["fc"].frozen[COMMITTED].rank == 0
    _randomise_fresh(delta)
    delta.merge_fresh_into_committed()
    committed = delta.layers["fc"].frozen[COMMITTED].clone()
    delta.merge_fresh_into_committed()  # Fresh is zero now.
    after = delta.layers["fc"].frozen[COMMITTED]
    assert torch.equal(after.U, committed.U) and torch.equal(after.S, committed.S)


def test_compact_equals_dense_soft_threshold():
    delta = DeltaModel(_net(in_features=8, out_features=7), rank=5)
    layer = delta.layers["fc"]
    _randomise_fresh(delta, seed=7)
    dense = layer.dense_delta((FRESH,)).clone()
    threshold = 0.3
    delta.compact(threshold)

    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    shrunk = (s - threshold * s[0]).clamp(min=0)
    keep = shrunk > 0
    expected = (u[:, keep] * shrunk[keep]) @ vh[keep]
    factors = layer.frozen[COMMITTED]
    assert factors.rank == int(keep.sum())
    assert torch.allclose(factors.S, shrunk[keep], atol=1e-4)
    assert torch.allclose((factors.U * factors.S) @ factors.V.t(), expected, atol=1e-4)
    assert not layer.fresh_B.any()


def test_compact_respects_max_rank_with_plain_truncation():
    delta = DeltaModel(_net(in_features=8, out_features=7), rank=5)
    layer = delta.layers["fc"]
    _randomise_fresh(delta, seed=8)
    dense = layer.dense_delta((FRESH,)).clone()
    stats = delta.compact(0.0, max_rank=2)
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    expected = (u[:, :2] * s[:2]) @ vh[:2]
    assert layer.frozen[COMMITTED].rank == 2
    assert stats["rank_after"] == 2 and stats["rank_before"] == 5
    assert torch.allclose(layer.dense_delta((COMMITTED,)), expected, atol=1e-4)


def test_compact_threshold_of_one_empties_the_bank():
    delta = DeltaModel(_net(), rank=3)
    _randomise_fresh(delta)
    delta.compact(1.0)
    assert delta.layers["fc"].frozen[COMMITTED].rank == 0


def test_bank_selection_nests_and_restores_on_exception():
    delta = DeltaModel(_net(), rank=2)
    assert delta.active == (COMMITTED, FRESH)
    with delta.banks("germline"):
        assert delta.active == ()
        with delta.banks("committed"):
            assert delta.active == (COMMITTED,)
            with pytest.raises(RuntimeError):
                with delta.banks("night"):
                    raise RuntimeError("boom")
            assert delta.active == (COMMITTED,)
        assert delta.active == ()
    assert delta.active == (COMMITTED, FRESH)


def test_bank_lists_are_canonical_and_unknown_names_raise():
    delta = DeltaModel(_net(), rank=2)
    with delta.banks([FRESH, COMMITTED]):
        assert delta.active == (COMMITTED, FRESH)
    with pytest.raises(KeyError):
        with delta.banks("nope"):
            pass


def test_named_frozen_bank_keeps_old_function():
    delta = DeltaModel(_net(), rank=3)
    layer = delta.layers["fc"]
    _randomise_fresh(delta, seed=9)
    x = _x()
    with delta.banks("night"):
        before = layer(x).detach()
    delta.freeze_bank("pre_compaction")
    delta.compact(0.9)
    with delta.banks("pre_compaction"):
        assert torch.allclose(layer(x).detach(), before, atol=1e-4)
    delta.drop_bank("pre_compaction")
    with pytest.raises(KeyError):
        with delta.banks("pre_compaction"):
            pass


def test_rollback_restores_exact_outputs():
    gpt2 = tiny_gpt2(20)
    delta = DeltaModel(gpt2, rank=2)
    _randomise_fresh(delta, seed=10)
    delta.merge_fresh_into_committed()
    _randomise_fresh(delta, seed=11)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    expected = gpt2(input_ids=ids).logits.detach().clone()
    snapshot = delta.snapshot()

    _randomise_fresh(delta, seed=12)
    delta.compact(0.2, max_rank=1)
    assert not torch.equal(gpt2(input_ids=ids).logits, expected)

    delta.restore(snapshot)
    assert torch.equal(gpt2(input_ids=ids).logits, expected)
    # The snapshot stays valid after a restore.
    delta.compact(0.2)
    delta.restore(snapshot)
    assert torch.equal(gpt2(input_ids=ids).logits, expected)


def test_only_fresh_is_trainable_and_base_hash_survives_merge_and_compact():
    gpt2 = tiny_gpt2(20)
    delta = DeltaModel(gpt2, rank=2)
    delta.assert_only_fresh_trainable()
    fresh = {id(p) for p in delta.fresh_parameters()}
    for parameter in gpt2.parameters():
        assert parameter.requires_grad == (id(parameter) in fresh)
    fingerprint = delta.base_fingerprint()
    _randomise_fresh(delta, seed=13)
    assert delta.base_fingerprint() == fingerprint  # Fresh factors are not part of the hash.
    delta.compact(0.1, max_rank=1)
    assert delta.base_fingerprint() == fingerprint
    with torch.no_grad():
        next(p for p in gpt2.parameters() if id(p) not in fresh).add_(1.0)
    assert delta.base_fingerprint() != fingerprint


def test_factors_stay_float32_when_the_model_changes_dtype():
    delta = DeltaModel(_net(), rank=2)
    _randomise_fresh(delta, seed=14)
    delta.merge_fresh_into_committed()
    delta.model.half()
    layer = delta.layers["fc"]
    assert layer.base.weight.dtype == torch.float16
    assert layer.fresh_A.dtype == torch.float32 and layer.fresh_B.dtype == torch.float32
    bank = layer.frozen[COMMITTED]
    assert bank.U.dtype == bank.S.dtype == bank.V.dtype == torch.float32
    assert layer.fresh_A.requires_grad


def test_state_dict_round_trip_is_exact_and_changes_cache_key():
    delta = DeltaModel(_net(), rank=2)
    _randomise_fresh(delta, seed=15)
    delta.merge_fresh_into_committed()
    _randomise_fresh(delta, seed=16)
    state = delta.state_dict()
    other = DeltaModel(_net(), rank=2)
    key_before = other.banks_key()
    other.load_state_dict(state)
    assert other.banks_key() != key_before
    a, b = delta.layers["fc"], other.layers["fc"]
    assert torch.equal(a.fresh_A, b.fresh_A) and torch.equal(a.fresh_B, b.fresh_B)
    assert torch.equal(a.frozen[COMMITTED].U, b.frozen[COMMITTED].U)
    assert torch.equal(a.frozen[COMMITTED].S, b.frozen[COMMITTED].S)
    assert torch.equal(a.frozen[COMMITTED].V, b.frozen[COMMITTED].V)
    x = _x()
    assert torch.equal(a(x), b(x))


def test_germline_cache_key_ignores_versions_but_others_do_not():
    delta = DeltaModel(_net(), rank=2)
    with delta.banks("germline"):
        germline_key = delta.banks_key()
    with delta.banks("committed"):
        committed_key = delta.banks_key()
    delta.bump_version()
    with delta.banks("germline"):
        assert delta.banks_key() == germline_key
    with delta.banks("committed"):
        assert delta.banks_key() != committed_key


def test_non_finite_factors_raise_rather_than_corrupt_the_bank():
    bad = torch.full((3, 2), float("nan"))
    with pytest.raises(ValueError):
        svd_of_product(bad, torch.zeros(3, 2))
