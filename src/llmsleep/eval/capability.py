"""Frozen real closed-book QA: retention of what the base model could already do (`ret_real`).

Built once per base model: keep questions the germline answers correctly by greedy generation, then split
head/tail by how consistently germline samples get them right (T2 section 7, Gekhman's known/maybe-known idea).
Cached on disk, so that a resumed run and every arm use the same set.

LICENSE: TriviaQA's licence is NOT verified (T2 section 6 lists it as "NOT CHECKED"). Treat it as research use
until someone checks. The dataset is only downloaded at run time and is not redistributed by this repository;
the disk cache holds its questions, so do not commit or publish the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from ..config import derive_seed, make_generator
from ..worlds.names import SeededRng
from .answers import normalize_answer, score_answer
from .metrics import EvalConfig, Tally, eval_batch_size, generate_answers


@dataclass
class CapabilityConfig:
    n_candidates: int = 3000  # Questions tried. Small models get perhaps a fifth right (an assumption).
    n_keep: int = 500  # At most this many base-correct questions are kept.
    k: int = 8  # Samples for the consistency score.
    max_new_tokens: int = 24
    batch_size: int = 32
    seed: int = 0
    dataset: str = "trivia_qa"


@dataclass(frozen=True)
class CapabilityItem:
    id: str
    question: str
    answers: tuple[str, ...]
    consistency: float
    split: str  # "head" (consistently right) or "tail".


def load_trivia_qa(limit: Optional[int] = None) -> list[dict]:
    """Network: the dataset is fetched here, and only here. Imported lazily; tests never call this."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    data = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    items = []
    for row in data:
        answer = row["answer"]
        forms = [answer["value"]] + [a for a in answer["aliases"] if a != answer["value"]]
        items.append({"question": row["question"], "answers": forms})
        if limit is not None and len(items) >= limit:
            break
    return items


def usable_answers(answers: Sequence[str], min_chars: int = 3) -> list[str]:
    """The main answer, plus aliases long enough not to match by accident (review L5).

    Aliases are matched anywhere in the first sentence, so one like "A" or "US" would give false credit. The
    first form is the dataset's own value and always stays, however short.
    """
    if not answers:
        return []
    return [answers[0]] + [a for a in answers[1:] if len(normalize_answer(a).replace(" ", "")) >= min_chars]


def _atomic_write(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def cache_path(directory: str, base_key: str, cfg: CapabilityConfig, fixture: Optional[Sequence[dict]] = None) -> str:
    """One file per base model and settings; a fixture list has its own."""
    identity = {"base": base_key, "cfg": asdict(cfg)}
    if fixture is not None:
        identity["fixture"] = list(fixture)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(directory, f"capability-{digest}.json")


def build_capability(
    handle: Any,
    cfg: CapabilityConfig,
    path: Optional[str] = None,
    items: Optional[Sequence[dict]] = None,
) -> list[CapabilityItem]:
    """Load from `path` if there; otherwise filter `items` (default: download), and write the cache.

    `items` is a list of {"question": str, "answers": [str, ...]}. Only `build_capability` with no cache and
    no items touches the network.
    """
    if path is not None and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as stream:
            return [
                CapabilityItem(r["id"], r["question"], tuple(r["answers"]), r["consistency"], r["split"])
                for r in json.load(stream)
            ]
    raw = list(items) if items is not None else load_trivia_qa()
    raw = [{"question": row["question"], "answers": usable_answers(row["answers"])} for row in raw]
    order = SeededRng(derive_seed(cfg.seed, 0, "capability:order")).shuffled(raw)[: cfg.n_candidates]

    kept: list[dict] = []
    with handle.ledger.exempt(), handle.banks("germline"), eval_batch_size(handle, cfg.batch_size):
        answers = generate_answers(
            handle,
            [(row["question"], None) for row in order],
            cfg.max_new_tokens,
            0.0,
            make_generator(cfg.seed, 0, "capability:greedy"),
        )
        for row in order:
            if score_answer(answers[(row["question"], None)][0], row["answers"]).correct:
                kept.append(row)
            if len(kept) >= cfg.n_keep:
                break
        prompts = [handle.format(row["question"], None) for row in kept]
        drawn = handle.sample(prompts, cfg.k, cfg.max_new_tokens, 1.0, make_generator(cfg.seed, 0, "capability:samples"))
    consistency = [
        sum(score_answer(handle.decode(s.token_ids), row["answers"]).correct for s in samples) / cfg.k
        for row, samples in zip(kept, drawn)
    ]
    ranked = sorted(range(len(kept)), key=lambda i: (-consistency[i], kept[i]["question"]))
    head = set(ranked[: (len(kept) + 1) // 2])
    out = [
        CapabilityItem(f"q{i:05d}", row["question"], tuple(row["answers"]), consistency[i], "head" if i in head else "tail")
        for i, row in enumerate(kept)
    ]
    if path is not None:
        _atomic_write(path, [asdict(item) for item in out])
    return out


def evaluate_capability(
    handle: Any, items: Sequence[CapabilityItem], cfg: EvalConfig, world_seed: int, night: int
) -> tuple[dict, dict[str, dict]]:
    """`ret_real` and its split. Every item was base-correct when selected, so change from base is ret_real - 1."""
    answers = generate_answers(
        handle,
        [(item.question, None) for item in items],
        cfg.max_new_tokens,
        cfg.temperature,
        make_generator(world_seed, night, "eval:capability"),
    )
    overall, head, tail, control, lift = Tally(), Tally(), Tally(), Tally(), Tally()
    size = len(items)
    for index, item in enumerate(items):
        text = answers[(item.question, None)][0]
        correct = float(score_answer(text, item.answers).correct)
        # Real questions have no typed vocabulary, so the control is simply the next item's gold.
        other = items[(index + 1) % size].answers if size > 1 else ()
        permuted = float(score_answer(text, other).correct)
        overall.add(item.id, correct)
        control.add(item.id, permuted)
        lift.add(item.id, correct - permuted)
        (head if item.split == "head" else tail).add(item.id, correct)
    rate = overall.rate()
    metrics = {
        "ret_real": rate,
        "ret_real_change": rate - 1.0 if rate is not None else None,
        "ret_real_head": head.rate(),
        "ret_real_tail": tail.rate(),
        "ret_real_permuted": control.rate(),
        "ret_real_lift": lift.rate(),
        "ret_real_n": len(overall),
    }
    clusters = {
        "ret_real": overall.clusters,
        "ret_real_head": head.clusters,
        "ret_real_tail": tail.clusters,
        "ret_real_lift": lift.clusters,
    }
    return metrics, clusters
