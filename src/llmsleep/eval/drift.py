"""`drift_generic`: mean per-token KL(germline || model) on fixed generic prompts. An alarm only (PREREG).

The sequence KL is estimated by the chain rule: continuations are sampled once from germline, and at each of
their tokens the full-vocabulary KL is taken between germline and the current model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch.nn.functional as F

from ..config import make_generator

GENERIC_PROMPT_FILE = Path(__file__).with_name("generic_prompts.json")


def load_generic_prompts(path: Path = GENERIC_PROMPT_FILE) -> list[str]:
    with open(path, "r", encoding="utf-8") as stream:
        return list(json.load(stream)["prompts"])


@dataclass
class DriftProbe:
    prompt_ids: list[list[int]]
    cont_ids: list[list[int]]


def build_drift_probe(handle: Any, max_tokens: int, seed: int = 0, prompts: Optional[list[str]] = None) -> DriftProbe:
    """Sample germline continuations. Fixed by `seed` and the base model, so the same for every arm and world."""
    texts = prompts if prompts is not None else load_generic_prompts()
    prompt_ids = [handle.encode_prompt(text) for text in texts]
    with handle.ledger.exempt(), handle.banks("germline"):
        drawn = handle.sample(prompt_ids, 1, max_tokens, 1.0, make_generator(seed, 0, "drift:continuations"))
    return DriftProbe(prompt_ids, [row[0].token_ids for row in drawn])


def measure_generic_drift(handle: Any, probe: DriftProbe, bank: str = "committed") -> tuple[Optional[float], dict[str, list[float]]]:
    """(mean KL per token, {prompt id: [sum of KL, tokens]}). The caller holds `ledger.exempt()`."""
    clusters: dict[str, list[float]] = {}
    size = handle.batch_size
    for start in range(0, len(probe.prompt_ids), size):
        prompts = probe.prompt_ids[start : start + size]
        conts = probe.cont_ids[start : start + size]
        with handle.banks("germline"):
            teacher = [logits for _, logits in handle.iter_answer_logits(prompts, conts)]
        with handle.banks(bank):
            student = [logits for _, logits in handle.iter_answer_logits(prompts, conts)]
        # One micro-batch each, since the slice is at most handle.batch_size.
        log_p = F.log_softmax(teacher[0], dim=-1)
        log_q = F.log_softmax(student[0], dim=-1)
        per_token = (log_p.exp() * (log_p - log_q)).sum(dim=-1).cpu().tolist()
        cursor = 0
        for offset, cont in enumerate(conts):
            length = len(cont)
            clusters[f"prompt-{start + offset:04d}"] = [float(sum(per_token[cursor : cursor + length])), float(length)]
            cursor += length
    total = sum(entry[1] for entry in clusters.values())
    value = sum(entry[0] for entry in clusters.values()) / total if total > 0 else None
    return value, clusters
