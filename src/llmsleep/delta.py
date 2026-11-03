"""Low-rank weight banks over a frozen base model.

Each wrapped layer computes `base(x) + x @ D^T` where D (out x in) is a sum over the active banks:
  frozen bank: D = U diag(S) V^T, with U out x r and V in x r
  fresh bank:  D = scale * B A^T, with A in x r, B out x r, B zero-initialised (standard LoRA)

All factors are float32, whatever the base dtype or autocast state. One model instance serves as both
teacher and student: selecting banks swaps a single tuple shared by every layer.
"""
from __future__ import annotations

import hashlib
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, Union

import torch
from torch import Tensor, nn

COMMITTED = "committed"
FRESH = "fresh"
DEFAULT_EXCLUDE = r"(^|\.)(lm_head|wte|wpe|embed_tokens|embeddings?)($|\.)"

BankSpec = Union[str, Sequence[str]]


@dataclass
class Factors:
    """Thin SVD factors of a frozen bank."""

    U: Tensor
    S: Tensor
    V: Tensor

    @property
    def rank(self) -> int:
        return int(self.S.numel())

    def clone(self) -> "Factors":
        return Factors(self.U.clone(), self.S.clone(), self.V.clone())


def empty_factors(out_features: int, in_features: int, device: torch.device) -> Factors:
    """A rank-0 bank."""
    return Factors(
        torch.zeros(out_features, 0, dtype=torch.float32, device=device),
        torch.zeros(0, dtype=torch.float32, device=device),
        torch.zeros(in_features, 0, dtype=torch.float32, device=device),
    )


def svd_of_product(left: Tensor, right: Tensor) -> Factors:
    """Thin SVD of `left @ right.T` (out x k, in x k), never forming the product.

    QR both sides, then SVD the small core: L = Ql Rl and R = Qr Rr give L R^T = Ql (Rl Rr^T) Qr^T.
    Runs in float32 on CPU: the core is small, and this keeps results independent of the device.
    """
    device = left.device
    out_features, in_features = left.size(0), right.size(0)
    if left.size(1) == 0:
        return empty_factors(out_features, in_features, device)
    left_cpu = left.detach().to("cpu", torch.float32)
    right_cpu = right.detach().to("cpu", torch.float32)
    if not (torch.isfinite(left_cpu).all() and torch.isfinite(right_cpu).all()):
        raise ValueError("non-finite values in delta factors")
    with torch.autocast("cpu", enabled=False):
        q_left, r_left = torch.linalg.qr(left_cpu, mode="reduced")
        q_right, r_right = torch.linalg.qr(right_cpu, mode="reduced")
        core_u, sigma, core_vh = torch.linalg.svd(r_left @ r_right.T, full_matrices=False)
    # Values below rounding noise (same tolerance as numpy.matrix_rank) are numerically zero rank.
    tolerance = max(out_features, in_features) * torch.finfo(torch.float32).eps * sigma[0]
    keep = sigma > tolerance
    return Factors(
        (q_left @ core_u[:, keep]).to(device),
        sigma[keep].to(device),
        (q_right @ core_vh.T[:, keep]).to(device),
    )


def _layer_dims(module: nn.Module) -> tuple[int, int] | None:
    """(in, out) for nn.Linear and HF Conv1D, else None. Conv1D is matched by name to avoid the import."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if type(module).__name__ == "Conv1D":
        return int(module.weight.shape[0]), int(module.weight.shape[1])  # Conv1D stores weight as (in, out).
    return None


class _Selector:
    """The active banks, shared by all layers so that switching is one attribute write."""

    __slots__ = ("active",)

    def __init__(self, active: tuple[str, ...]) -> None:
        self.active = active


class DeltaLinear(nn.Module):
    """Wraps a Linear or Conv1D. The delta acts on outputs, so the base's weight layout does not matter."""

    def __init__(self, base: nn.Module, rank: int, scale: float, selector: _Selector) -> None:
        super().__init__()
        dims = _layer_dims(base)
        if dims is None:
            raise TypeError(f"cannot wrap {type(base).__name__}")
        self.in_features, self.out_features = dims
        self.base = base
        self.scale = float(scale)
        self.selector = selector
        device = next(base.parameters()).device
        self.fresh_A = nn.Parameter(torch.zeros(self.in_features, rank, dtype=torch.float32, device=device))
        self.fresh_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=device))
        # Plain dict: ranks change between nights, and rank-changing buffers would break state_dict shapes.
        self.frozen: dict[str, Factors] = {COMMITTED: empty_factors(self.out_features, self.in_features, device)}

    def _apply(self, fn: Callable, *args, **kwargs):  # type: ignore[override]
        """Follow `.to(device)`, but stay float32 through `.half()` and the like."""
        result = super()._apply(fn, *args, **kwargs)
        if self.fresh_A.dtype != torch.float32:
            self.fresh_A.data = self.fresh_A.data.to(torch.float32)
            self.fresh_B.data = self.fresh_B.data.to(torch.float32)
        device = self.fresh_A.device
        for bank in self.frozen.values():
            bank.U = bank.U.to(device)
            bank.S = bank.S.to(device)
            bank.V = bank.V.to(device)
        return result

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        names = self.selector.active
        if not names:
            return y
        xf = x.to(torch.float32)
        delta = None
        for name in names:
            if name == FRESH:
                # (..., in) @ (in, r) -> (..., r); @ (r, out) -> (..., out)
                term = self.scale * ((xf @ self.fresh_A) @ self.fresh_B.t())
            else:
                bank = self.frozen[name]
                if bank.rank == 0:
                    continue
                term = ((xf @ bank.V) * bank.S) @ bank.U.t()
            delta = term if delta is None else delta + term
        return y if delta is None else y + delta.to(y.dtype)

    def left_right(self, names: Sequence[str]) -> tuple[Tensor, Tensor]:
        """(L, R) with `D = L R^T` summing `names`. The LoRA scale is absorbed into L here."""
        lefts, rights = [], []
        for name in names:
            if name == FRESH:
                lefts.append(self.scale * self.fresh_B.detach())
                rights.append(self.fresh_A.detach())
            else:
                bank = self.frozen[name]
                lefts.append(bank.U * bank.S)
                rights.append(bank.V)
        if not lefts:
            device = self.fresh_A.device
            return (
                torch.zeros(self.out_features, 0, dtype=torch.float32, device=device),
                torch.zeros(self.in_features, 0, dtype=torch.float32, device=device),
            )
        return torch.cat(lefts, dim=1), torch.cat(rights, dim=1)

    def dense_delta(self, names: Sequence[str]) -> Tensor:
        """The dense D. For tests only; production code never forms it."""
        left, right = self.left_right(names)
        return left @ right.t()


@dataclass
class DeltaSnapshot:
    """Exact copy of every bank, for rollback."""

    frozen: dict[str, dict[str, Factors]]
    fresh: dict[str, tuple[Tensor, Tensor]]


class DeltaModel:
    """Attach low-rank banks to a model, in place. The default selection is the "night" student."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 32,
        alpha: float | None = None,
        targets: str = r".*",
        exclude: str = DEFAULT_EXCLUDE,
        seed: int = 0,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be positive")
        self.model = model
        self.rank = rank
        self.scale = (alpha if alpha is not None else float(rank)) / rank
        self.version = 0
        self._selector = _Selector((COMMITTED, FRESH))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        targets_re, exclude_re = re.compile(targets), re.compile(exclude)
        chosen = [
            name
            for name, module in model.named_modules()
            if _layer_dims(module) is not None and targets_re.search(name) and not exclude_re.search(name)
        ]
        self.layers: dict[str, DeltaLinear] = {}
        for name in chosen:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            wrapped = DeltaLinear(getattr(parent, child_name), rank, self.scale, self._selector)
            setattr(parent, child_name, wrapped)
            self.layers[name] = wrapped
        # A = 0 with B = 0 would be a saddle: no gradient could ever leave it.
        self.reset_fresh(torch.Generator(device="cpu").manual_seed(seed))

    # ---- bank selection ----------------------------------------------------------------------------

    def _bank_names(self) -> set[str]:
        names = {COMMITTED, FRESH}
        for layer in self.layers.values():
            names.update(layer.frozen)
            break  # All layers hold the same bank names.
        return names

    def _resolve(self, spec: BankSpec) -> tuple[str, ...]:
        if spec == "germline":
            return ()
        if spec == "night":
            return (COMMITTED, FRESH)
        names = (spec,) if isinstance(spec, str) else tuple(spec)
        unknown = set(names) - self._bank_names()
        if unknown:
            raise KeyError(f"unknown banks: {sorted(unknown)}")
        # Canonical order: a fixed summation order keeps float results exactly reproducible.
        return tuple(sorted(set(names)))

    @contextmanager
    def banks(self, spec: BankSpec) -> Iterator[None]:
        """Select banks for the block: "germline", "committed", "night", a bank name, or a list of names."""
        names = self._resolve(spec)
        previous = self._selector.active
        self._selector.active = names
        try:
            yield
        finally:
            self._selector.active = previous

    @property
    def active(self) -> tuple[str, ...]:
        return self._selector.active

    def banks_key(self) -> tuple[tuple[str, ...], int]:
        """Identifies the current function, for caches. Germline never changes, so its version is 0."""
        active = self._selector.active
        return (active, self.version if active else 0)

    def bump_version(self) -> None:
        """Note that a bank changed. The training loop calls this after each optimizer step."""
        self.version += 1

    # ---- fresh bank --------------------------------------------------------------------------------

    def fresh_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers.values():
            params.extend((layer.fresh_A, layer.fresh_B))
        return params

    def assert_only_fresh_trainable(self) -> None:
        """Invariant 1."""
        fresh = {id(p) for p in self.fresh_parameters()}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and id(parameter) not in fresh:
                raise AssertionError(f"non-fresh parameter is trainable: {name}")

    @torch.no_grad()
    def reset_fresh(self, generator: torch.Generator) -> None:
        """B = 0; A ~ U(-1/sqrt(in), 1/sqrt(in)), which is kaiming_uniform(a=sqrt(5)), from a local generator."""
        for layer in self.layers.values():
            bound = 1.0 / math.sqrt(layer.in_features)
            draws = torch.rand(layer.in_features, self.rank, generator=generator, dtype=torch.float32)
            layer.fresh_A.copy_((2.0 * draws - 1.0) * bound)
            layer.fresh_B.zero_()
            layer.fresh_A.grad = None
            layer.fresh_B.grad = None
        self.bump_version()

    # ---- merge and compaction ----------------------------------------------------------------------

    @torch.no_grad()
    def merge_fresh_into_committed(self) -> None:
        """committed += fresh, exactly up to rounding, including the LoRA scale; then fresh becomes zero."""
        for layer in self.layers.values():
            if layer.fresh_B.any():
                layer.frozen[COMMITTED] = svd_of_product(*layer.left_right((COMMITTED, FRESH)))
            layer.fresh_B.zero_()
        self.bump_version()

    @torch.no_grad()
    def compact(self, threshold: float, max_rank: int | None = None) -> dict[str, float]:
        """Merge, then soft-threshold singular values per layer and keep at most `max_rank` of them.

        `threshold` is a fraction of the layer's largest singular value: sigma -> max(sigma - t * sigma_1, 0).
        threshold=0 with `max_rank` is plain truncation.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.merge_fresh_into_committed()
        energy_before, energy_after, rank_before, rank_after = 0.0, 0.0, 0, 0
        for layer in self.layers.values():
            bank = layer.frozen[COMMITTED]
            rank_before += bank.rank
            if bank.rank == 0:
                continue
            factors = svd_of_product(bank.U * bank.S, bank.V)  # Re-orthonormalises; also sorts sigma.
            energy_before += float((factors.S**2).sum())
            if factors.rank > 0:
                factors.S = (factors.S - threshold * factors.S[0]).clamp(min=0.0)
                keep = factors.S > 0
                if max_rank is not None:
                    keep[max_rank:] = False  # S is sorted, so this drops the smallest.
                factors = Factors(factors.U[:, keep], factors.S[keep], factors.V[:, keep])
            energy_after += float((factors.S**2).sum())
            rank_after += factors.rank
            layer.frozen[COMMITTED] = factors
        self.bump_version()
        return {
            "rank_before": rank_before,
            "rank_after": rank_after,
            "energy_before": energy_before,
            "energy_after": energy_after,
        }

    @torch.no_grad()
    def freeze_bank(self, name: str, source: BankSpec = "night") -> None:
        """Store the current sum of `source` as a frozen named bank, e.g. `pre_compaction`."""
        if name in (COMMITTED, FRESH):
            raise ValueError(f"{name!r} is reserved")
        names = self._resolve(source)
        for layer in self.layers.values():
            layer.frozen[name] = svd_of_product(*layer.left_right(names))

    def drop_bank(self, name: str) -> None:
        if name in (COMMITTED, FRESH):
            raise ValueError(f"{name!r} is reserved")
        if name in self._selector.active:
            raise ValueError(f"bank {name!r} is active")
        for layer in self.layers.values():
            layer.frozen.pop(name, None)

    # ---- rollback and persistence ------------------------------------------------------------------

    def snapshot(self) -> DeltaSnapshot:
        return DeltaSnapshot(
            frozen={n: {b: f.clone() for b, f in l.frozen.items()} for n, l in self.layers.items()},
            fresh={n: (l.fresh_A.detach().clone(), l.fresh_B.detach().clone()) for n, l in self.layers.items()},
        )

    @torch.no_grad()
    def restore(self, snapshot: DeltaSnapshot) -> None:
        """Restore every bank bit-exactly. The snapshot stays valid for reuse."""
        for name, layer in self.layers.items():
            layer.frozen = {b: f.clone() for b, f in snapshot.frozen[name].items()}
            fresh_a, fresh_b = snapshot.fresh[name]
            layer.fresh_A.copy_(fresh_a)
            layer.fresh_B.copy_(fresh_b)
            layer.fresh_A.grad = None
            layer.fresh_B.grad = None
        self.bump_version()

    def state_dict(self) -> dict[str, dict]:
        """Adapters only, on CPU. The base model is not included."""
        return {
            name: {
                "frozen": {
                    b: {"U": f.U.cpu().clone(), "S": f.S.cpu().clone(), "V": f.V.cpu().clone()}
                    for b, f in layer.frozen.items()
                },
                "A": layer.fresh_A.detach().cpu().clone(),
                "B": layer.fresh_B.detach().cpu().clone(),
            }
            for name, layer in self.layers.items()
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, dict]) -> None:
        if set(state) != set(self.layers):
            raise KeyError("adapter state does not match the attached layers")
        for name, layer in self.layers.items():
            entry = state[name]
            device = layer.fresh_A.device
            layer.frozen = {
                b: Factors(f["U"].to(device), f["S"].to(device), f["V"].to(device))
                for b, f in entry["frozen"].items()
            }
            layer.fresh_A.copy_(entry["A"].to(device))
            layer.fresh_B.copy_(entry["B"].to(device))
        self.bump_version()

    # ---- stats -------------------------------------------------------------------------------------

    def stats(self) -> dict:
        """Norm, rank and parameter count of the committed delta, per layer and in total."""
        per_layer, total_sq, total_rank, total_params = {}, 0.0, 0, 0
        for name, layer in self.layers.items():
            bank = layer.frozen[COMMITTED]
            norm_sq = float((bank.S**2).sum())
            params = bank.rank * (layer.in_features + layer.out_features + 1)
            per_layer[name] = {"norm": math.sqrt(norm_sq), "rank": bank.rank, "params": params}
            total_sq += norm_sq
            total_rank += bank.rank
            total_params += params
        return {
            "delta_norm": math.sqrt(total_sq),
            "delta_rank": total_rank,
            "delta_params": total_params,
            "layers": per_layer,
        }

    def base_fingerprint(self) -> str:
        """SHA-256 of all base parameters (invariant 1). Fresh factors are excluded."""
        fresh = {id(p) for p in self.fresh_parameters()}
        digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            if id(parameter) in fresh:
                continue
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().to("cpu", torch.float32).numpy().tobytes())
        return digest.hexdigest()
