"""Stage 4: merge fresh into committed; optionally threshold and repair. Acts on delta, never on base weights."""
from __future__ import annotations

from typing import Sequence

import torch

from ..config import CompactConfig, TrainConfig
from ..handle import ModelHandle
from .consolidate import HardExample, TokenTargets, train_loop


def merge_only(handle: ModelHandle) -> dict:
    """Exact low-rank addition; the delta accumulates without truncation."""
    handle.delta.merge_fresh_into_committed()
    return {"compacted": False}


def compact_and_repair(
    handle: ModelHandle,
    hard: Sequence[HardExample],
    integrate: Sequence[TokenTargets],
    anchor: Sequence[TokenTargets],
    cfg: CompactConfig,
    train_cfg: TrainConfig,
    repair_init: torch.Generator,
    repair_data: torch.Generator,
    repair_active: bool,
) -> dict:
    """Soft-threshold, then a short repair on the same loss. The repair's rank is cut back to `max_rank`.

    The repair trains a new fresh bank, whose merge adds up to `rank` per layer, so without this last cut the
    retained rank would exceed `max_rank` by that much. The caller enters the "repair" stage if it is active.
    """
    delta = handle.delta
    stats = delta.compact(cfg.threshold, cfg.max_rank)
    report: dict = {"compacted": True, **stats, "repair": None}
    if repair_active and cfg.repair_steps > 0:
        delta.reset_fresh(repair_init)
        result = train_loop(handle, hard, integrate, anchor, train_cfg, repair_data, max_steps=cfg.repair_steps)
        report["repair"] = result.summary()
        delta.compact(0.0, cfg.max_rank)
    return report
