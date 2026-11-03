"""E0: teacher and signal validity. Trains nothing. `python experiments/run_e0.py --config configs/e0.json --out DIR`.

For each candidate model it writes every PREREG criterion with its value, its threshold and pass or fail, to
`e0_verdict.json`, and names the smallest model passing all of H0. A candidate is written as soon as it finishes,
so a dead session loses only the model in progress.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

from llmsleep.config import make_generator  # noqa: E402
from llmsleep.eval.answers import score_answer  # noqa: E402
from llmsleep.eval.metrics import generate_answers  # noqa: E402
from llmsleep.handle import ModelHandle, PromptFormat  # noqa: E402
from llmsleep.ledger import ComputeLedger  # noqa: E402
from llmsleep.memory import DistractorSampler, format_context  # noqa: E402
from llmsleep.signals import chd, raw_kl_nocontext  # noqa: E402
from llmsleep.stats import auroc  # noqa: E402
from llmsleep.worlds import SyntheticWorld, WorldConfig  # noqa: E402
from llmsleep.worlds.familiarity import check_unknown  # noqa: E402

THRESHOLDS = {
    "h0_direct": 0.90,
    "h0_reverse": 0.70,
    "h0_two_hop": 0.60,
    "h0_contradiction": 0.70,
    "h1_episode_auroc": 0.85,
    "h1_known_auroc": 0.75,
}
REAL_FACTS_FILE = _ROOT / "src" / "llmsleep" / "eval" / "real_facts.json"


def load_real_facts(path: Path = REAL_FACTS_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        return list(json.load(stream)["facts"])


def _context_of(world: SyntheticWorld, episode_ids: Sequence[str]) -> str:
    return format_context([world.episode(i) for i in episode_ids])


def _accuracy(handle: Any, world: SyntheticWorld, kind: str, limit: Optional[int], max_new: int) -> tuple[Optional[float], int]:
    """In-context accuracy: each probe with exactly its gold episodes in context."""
    probes = [p for p in world.probes if p.kind == kind][:limit]
    requests = [(p.query, _context_of(world, p.episode_ids)) for p in probes]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(world.seed, 0, f"e0:{kind}"))
    hits = [
        score_answer(answers[request][0], p.gold, p.stale, world.vocabulary(p.vocab)).correct
        for p, request in zip(probes, requests)
    ]
    return (sum(hits) / len(hits) if hits else None), len(hits)


def h0_criteria(handle: Any, world: SyntheticWorld, facts: Sequence[dict], limit: Optional[int], max_new: int) -> dict:
    results = {}
    for name, kind in (("h0_direct", "direct"), ("h0_reverse", "reverse"), ("h0_two_hop", "two_hop")):
        value, count = _accuracy(handle, world, kind, limit, max_new)
        results[name] = _criterion(value, THRESHOLDS[name], count)
    known = known_facts(handle, facts, world.seed, max_new)
    requests = [(f["question"], f["claim"].format(value=f["alt"])) for f in known]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(world.seed, 0, "e0:contradiction"))
    follows = [
        score_answer(answers[request][0], [f["alt"]], [f["answer"]]).correct for f, request in zip(known, requests)
    ]
    value = sum(follows) / len(follows) if follows else None
    results["h0_contradiction"] = _criterion(value, THRESHOLDS["h0_contradiction"], len(follows))
    return results


def known_facts(handle: Any, facts: Sequence[dict], seed: int, max_new: int) -> list[dict]:
    """The real facts this base model answers correctly closed-book."""
    requests = [(f["question"], None) for f in facts]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(seed, 0, "e0:known"))
    return [f for f, r in zip(facts, requests) if score_answer(answers[r][0], [f["answer"]]).correct]


def _criterion(value: Optional[float], threshold: float, count: int) -> dict:
    return {"value": value, "threshold": threshold, "n": count, "passed": value is not None and value >= threshold}


def h1_criteria(
    handle: Any, world: SyntheticWorld, facts: Sequence[dict], limit: Optional[int], k: int, max_new: int, chd_tokens: int
) -> dict:
    """AUROC on own vs. unrelated queries; offset of CHD vs. raw KL; known vs. fictional."""
    probes = [p for p in world.probes if p.kind == "direct"][:limit]
    sampler = DistractorSampler(world.distractor_pool(), 0.5)
    pick = make_generator(world.seed, 0, "e0:distractors")
    own: list[tuple[str, str, str]] = []
    unrelated: list[tuple[str, str, str]] = []
    size = len(probes)
    for index, probe in enumerate(probes):
        episode = world.episode(probe.episode_ids[0])
        context = format_context([episode])
        # The same query with a context about something else: the query's difficulty cancels.
        other = None
        for step in range(1, size):
            candidate = world.episode(probes[(index + step) % size].episode_ids[0])
            if not {x.casefold() for x in candidate.entities} & {x.casefold() for x in episode.entities}:
                other = candidate
                break
        distractor = sampler.sample(episode.entities, len(context), pick)
        if other is None or distractor is None:
            continue
        distractor_text = format_context([distractor])
        own.append((probe.query, context, distractor_text))
        unrelated.append((probe.query, format_context([other]), distractor_text))
    results: dict[str, dict] = {}

    def run_chd(items: Sequence[tuple[str, str, str]], purpose: str) -> list[float]:
        if not items:
            return []
        out = chd(
            handle,
            [q for q, _, _ in items],
            [e for _, e, _ in items],
            [d for _, _, d in items],
            k,
            chd_tokens,
            make_generator(world.seed, 0, purpose),
        )
        return out.values.tolist()

    own_values = run_chd(own, "e0:chd_own")
    unrelated_values = run_chd(unrelated, "e0:chd_unrelated")
    area = auroc([1] * len(own_values) + [0] * len(unrelated_values), own_values + unrelated_values)
    results["h1_episode_auroc"] = _criterion(area, THRESHOLDS["h1_episode_auroc"], len(own_values))

    raw = (
        raw_kl_nocontext(
            handle,
            [q for q, _, _ in unrelated],
            [e for _, e, _ in unrelated],
            k,
            chd_tokens,
            make_generator(world.seed, 0, "e0:raw"),
        ).values.tolist()
        if unrelated
        else []
    )
    mean_chd = sum(unrelated_values) / len(unrelated_values) if unrelated_values else None
    mean_raw = sum(raw) / len(raw) if raw else None
    results["h1_lower_offset"] = {
        "mean_chd_unrelated": mean_chd,
        "mean_raw_kl_unrelated": mean_raw,
        "n": len(raw),
        "passed": mean_chd is not None and mean_raw is not None and mean_chd < mean_raw,
    }

    known = known_facts(handle, facts, world.seed, max_new)
    known_items = []
    for fact in known:
        context = fact["claim"].format(value=fact["answer"])
        distractor = sampler.sample([fact["answer"]], len(context), pick)
        if distractor is not None:
            known_items.append((fact["question"], context, format_context([distractor])))
    known_values = run_chd(known_items, "e0:chd_known")
    # Fictional is the positive class: a model that knows a fact needs the context less.
    fictional = own_values[: max(len(known_values), 1)]
    area = auroc([0] * len(known_values) + [1] * len(fictional), known_values + fictional)
    results["h1_known_auroc"] = _criterion(area, THRESHOLDS["h1_known_auroc"], len(known_values))
    return results


def evaluate_candidate(model: Any, tokenizer: Any, config: dict, device: str, facts: Sequence[dict]) -> dict:
    fp16 = bool(config.get("fp16", False))
    handle = ModelHandle(
        model,
        tokenizer,
        device,
        ComputeLedger(),
        PromptFormat(),
        int(config.get("batch_size", 32)),
        fp16,
    )
    generation = getattr(model, "generation_config", None)
    eos = getattr(generation, "eos_token_id", None)
    if eos is not None:
        handle.eos_ids |= set(eos if isinstance(eos, (list, tuple)) else [eos])
    world = SyntheticWorld(WorldConfig(**config["world"]), int(config["world_seed"]))
    limit = config.get("max_probes")
    max_new = int(config.get("max_new_tokens", 24))
    with handle.ledger.exempt():
        familiarity = check_unknown(handle, world, limit, max_new)
        h0 = h0_criteria(handle, world, facts, limit, max_new)
        h1 = h1_criteria(handle, world, facts, limit, int(config.get("k", 8)), max_new, int(config.get("chd_tokens", 24)))
    return {
        "names_unknown": familiarity,
        "h0": h0,
        "h1": h1,
        "h0_passed": all(c["passed"] for c in h0.values()),
        "h1_passed": all(c["passed"] for c in h1.values()),
    }


def verdict(candidates: dict[str, dict], order: Sequence[str]) -> dict:
    """PREREG: choose the smallest model passing all four H0 criteria; a name collision also disqualifies."""
    chosen = None
    for model_id in order:
        entry = candidates.get(model_id)
        if entry and entry["h0_passed"] and entry["names_unknown"]["passed"]:
            chosen = model_id
            break
    missing = [m for m in order if m not in candidates]
    if chosen is not None:
        action = "use this model"
    elif missing:
        action = "not finished: some candidates are still to run"
    else:
        action = "stop and report: the teacher is unusable at Colab scale; hard-label arms only"
    h1 = candidates[chosen]["h1_passed"] if chosen else None
    return {
        "chosen_model": chosen,
        "action": action,
        "h1_passed_on_chosen": h1,
        "h1_action": None if h1 is not False else "remove the control layer; E2 and E3 become replay-baseline comparisons",
        "not_run": missing,
    }


def run_e0(config: dict, out: Path, factory: Any = None, device: Optional[str] = None, facts: Optional[Sequence[dict]] = None) -> dict:
    """`config["model_ids"]` is in ascending size. Finished candidates in the verdict file are not re-run."""
    from run import default_device, load_hf_model, write_json_atomic

    factory = factory or load_hf_model
    device = device or default_device()
    facts = list(facts) if facts is not None else load_real_facts()
    out = Path(out)
    path = out / "e0_verdict.json"
    candidates: dict[str, dict] = {}
    if path.exists():
        candidates = json.loads(path.read_text(encoding="utf-8"))["candidates"]
    order = list(config["model_ids"])
    payload = {"thresholds": THRESHOLDS, "candidates": candidates, "verdict": verdict(candidates, order)}
    for model_id in order:
        if model_id in candidates:
            continue
        model, tokenizer = factory(model_id, bool(config.get("fp16", False)))
        candidates[model_id] = evaluate_candidate(model, tokenizer, config, device, facts)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payload = {"thresholds": THRESHOLDS, "candidates": candidates, "verdict": verdict(candidates, order)}
        write_json_atomic(path, payload)
    write_json_atomic(path, payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="E0: teacher and signal validity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    result = run_e0(config, Path(args.out), device=args.device)
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
