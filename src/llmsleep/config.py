"""Night configuration and seed derivation. Nothing here reads global state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

import torch

from .arms import ArmConfig, get_arm

# Allocation of the night's tokens, for the stages an arm actually uses. Inactive stages are dropped and
# the rest renormalised, so a replay arm spends its whole matched allowance on training.
DEFAULT_STAGE_FRACTIONS: dict[str, float] = {
    "synthesis": 0.10,
    "signals": 0.15,
    "teacher": 0.15,
    "train": 0.45,
    "repair": 0.05,
    "audit": 0.10,
}


def derive_seed(world_seed: int, night: int, purpose: str) -> int:
    """A stable 63-bit seed for (world_seed, night, purpose). Python's hash() is salted, so not used."""
    digest = hashlib.sha256(f"{world_seed}:{night}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_generator(world_seed: int, night: int, purpose: str) -> torch.Generator:
    """A local CPU generator. Sampling draws on CPU, so results do not depend on the device."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(world_seed, night, purpose))
    return generator


@dataclass
class ModelConfig:
    rank: int = 32
    alpha: float = 32.0  # LoRA scale is alpha / rank.
    batch_size: int = 8  # Micro-batch for signals, sampling and teacher passes.
    fp16: bool = False  # Autocast on CUDA only; factors stay float32.
    prompt_template: str = "Context: {context}\nQuestion: {query}\nAnswer:"
    empty_context: str = "(none)"


@dataclass
class BudgetConfig:
    night_tokens: int = 2_000_000
    fractions: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STAGE_FRACTIONS))


@dataclass
class SignalConfig:
    k: int = 8  # Monte Carlo samples per CHD estimate.
    max_new_tokens: int = 24
    chunk: int = 4  # Queries per CHD call, so that exhaustion loses at most one chunk.


@dataclass
class QueryConfig:
    per_episode: int = 6
    model_samples: int = 2
    model_max_new_tokens: int = 64
    temperature: float = 0.8
    instruction: str = "Write questions, one per line, that the text above answers."
    max_pairs: int = 16


@dataclass
class SelectConfig:
    integrate_null_q: float = 0.90  # Integrate: CHD above this quantile of the null.
    anchor_null_q: float = 0.50  # Anchor: CHD at or below this quantile, and lexically near.
    gate_null_q: float = 0.50  # Write gate and retirement: at or below this quantile means no dependence.
    n_null: int = 32
    min_null: int = 8  # Below this the quantiles are not trusted; the null's extremes are used.
    lexical_min: float = 0.3
    distractor_length_tol: float = 0.5
    hard_boost: int = 2  # Extra sampling weight in the "selected" pool for episodes with integrate queries.
    # With selection off, draw counts per night: {str(night): report.selection of the paired CHD arm, same seed}.
    # Only "train" and "heldout", each with "integrate" and "anchor", are read. A night that is absent falls
    # back to the fractions below, and the report says so through selection["count_source"].
    matched_selection: dict[str, dict] | None = None
    uniform_integrate_frac: float = 0.5
    uniform_anchor_frac: float = 0.25


@dataclass
class TeacherConfig:
    samples: int = 2
    max_new_tokens: int = 24
    top_k: int = 64
    self_samples: int = 64  # B4: sequences sampled from germline.
    self_max_new_tokens: int = 32


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    hard_batch: int = 4
    kl_batch: int = 4
    anchor_batch: int = 4
    accum_steps: int = 1
    max_grad_norm: float = 1.0
    max_steps: int | None = None  # None: run until the stage's tokens are spent.
    w_hard: float = 1.0
    w_kl: float = 1.0
    w_anchor: float = 1.0
    gradient_checkpointing: bool = False


@dataclass
class CompactConfig:
    threshold: float = 0.01  # Soft threshold, as a fraction of each layer's largest singular value.
    max_rank: int | None = 256  # Per layer.
    repair_steps: int = 20


@dataclass
class CommitConfig:
    heldout_frac: float = 0.2
    max_anchor_drift: float = 0.1  # Nats per answer token, over held-out anchors.
    min_gap_drop: float = 0.0  # Mean gap must fall by more than this.
    retire_nights: int = 3
    audit_max_items: int = 64


@dataclass
class NightConfig:
    arm: ArmConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    compact: CompactConfig = field(default_factory=CompactConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    retrieval_k: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NightConfig":
        """Inverse of `to_dict`. `arm` may also be a PREREG arm name. Unknown keys raise."""
        kwargs: dict[str, Any] = {}
        types = {f.name: f.default_factory for f in fields(cls) if callable(f.default_factory)}
        for key, value in data.items():
            if key == "arm":
                kwargs[key] = get_arm(value) if isinstance(value, str) else ArmConfig(**value)
            elif key in types:
                kwargs[key] = types[key](**value)
            elif key == "retrieval_k":
                kwargs[key] = value
            else:
                raise KeyError(f"unknown config key {key!r}")
        return cls(**kwargs)

    def config_hash(self) -> str:
        """Stable identifier for results.jsonl."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
