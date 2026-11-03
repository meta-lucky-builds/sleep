"""E0: teacher and signal validity. Trains nothing. `python experiments/run_e0.py --config configs/e0.json --out DIR`.

For each candidate model it writes every PREREG criterion with its value, its threshold and pass or fail, to
`e0_verdict.json`, and names the smallest model passing all of H0, overall and per family (A1.10). A candidate is
written as soon as it finishes, so a dead session loses only the model in progress.

`--smoke` instead runs the first real-model checks of the review's section D on one model and writes
`smoke_<model>.json`. It works on CPU without fp16 and on a GPU with fp16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

from llmsleep.config import QueryConfig, TrainConfig, make_generator  # noqa: E402
from llmsleep.delta import DeltaModel  # noqa: E402
from llmsleep.eval.answers import score_answer  # noqa: E402
from llmsleep.eval.metrics import generate_answers  # noqa: E402
from llmsleep.eval.probes import stratified_sample  # noqa: E402
from llmsleep.handle import ModelHandle, PromptFormat, is_instruct_model  # noqa: E402
from llmsleep.ledger import ComputeLedger  # noqa: E402
from llmsleep.memory import DistractorSampler, Episode, format_context  # noqa: E402
from llmsleep.phases.consolidate import hard_examples, train_loop  # noqa: E402
from llmsleep.phases.queries import ModelQueryGen, parse_questions  # noqa: E402
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
# Fictional attribute -> the type of real fact it is compared with, so known-vs-fictional is not question type.
ATTR_TYPE = {
    "hometown": "place",
    "hq_city": "place",
    "region": "place",
    "instrument": "thing",
    "product": "thing",
    "invention": "thing",
    "birth_year": "number",
    "person": "person",
}
SMOKE_MIN_PARSE_RATE = 0.5


def load_real_facts(path: Path = REAL_FACTS_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        return list(json.load(stream)["facts"])


# ---- H0 ----------------------------------------------------------------------------------------------------


def _context_of(world: SyntheticWorld, episode_ids: Sequence[str]) -> str:
    return format_context([world.episode(i) for i in episode_ids])


def _accuracy(handle: Any, world: SyntheticWorld, kind: str, limit: Optional[int], max_new: int) -> tuple[Optional[float], int]:
    """In-context accuracy: each probe with exactly its gold episodes in context."""
    probes = stratified_sample([p for p in world.probes if p.kind == kind], limit, world.seed, f"h0:{kind}")
    requests = [(p.query, _context_of(world, p.episode_ids)) for p in probes]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(world.seed, 0, f"e0:{kind}"), answer_mode=True)
    hits = [
        score_answer(answers[request][0], p.gold, p.stale, world.vocabulary(p.vocab), p.query).correct
        for p, request in zip(probes, requests)
    ]
    return (sum(hits) / len(hits) if hits else None), len(hits)


def h0_criteria(
    handle: Any, world: SyntheticWorld, known: Sequence[dict], limit: Optional[int], max_new: int
) -> dict:
    results = {}
    for name, kind in (("h0_direct", "direct"), ("h0_reverse", "reverse"), ("h0_two_hop", "two_hop")):
        value, count = _accuracy(handle, world, kind, limit, max_new)
        results[name] = _criterion(value, THRESHOLDS[name], count)
    requests = [(f["question"], f["claim"].format(value=f["alt"])) for f in known]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(world.seed, 0, "e0:contradiction"), answer_mode=True)
    follows = [
        score_answer(answers[request][0], [f["alt"]], [f["answer"]]).correct for f, request in zip(known, requests)
    ]
    value = sum(follows) / len(follows) if follows else None
    results["h0_contradiction"] = _criterion(value, THRESHOLDS["h0_contradiction"], len(follows))
    return results


def known_facts(handle: Any, facts: Sequence[dict], seed: int, max_new: int) -> list[dict]:
    """The real facts this base model answers correctly closed-book."""
    requests = [(f["question"], None) for f in facts]
    answers = generate_answers(handle, requests, max_new, 0.0, make_generator(seed, 0, "e0:known"), answer_mode=True)
    return [f for f, r in zip(facts, requests) if score_answer(answers[r][0], [f["answer"]]).correct]


def _criterion(value: Optional[float], threshold: float, count: int) -> dict:
    return {"value": value, "threshold": threshold, "n": count, "passed": value is not None and value >= threshold}


# ---- H1 ----------------------------------------------------------------------------------------------------


def _run_chd(
    handle: Any, world: SyntheticWorld, items: Sequence[tuple[str, str, str]], k: int, tokens: int, purpose: str
) -> list[float]:
    if not items:
        return []
    out = chd(
        handle,
        [q for q, _, _ in items],
        [e for _, e, _ in items],
        [d for _, _, d in items],
        k,
        tokens,
        make_generator(world.seed, 0, purpose),
    )
    return out.per_token.tolist()  # Nats per answer token: what the experiments' decision rules compare.


def _other_episode(episode: Episode, pool: Sequence[Episode], start: int) -> Optional[Episode]:
    """The next episode in `pool` that shares no entity with `episode`."""
    keys = {x.casefold() for x in episode.entities}
    size = len(pool)
    for step in range(1, size):
        candidate = pool[(start + step) % size]
        if not keys & {x.casefold() for x in candidate.entities}:
            return candidate
    return None


def _episode_vs_unrelated(
    handle: Any,
    world: SyntheticWorld,
    triples: Sequence[tuple[str, Episode]],
    pool: Sequence[Episode],
    k: int,
    tokens: int,
    label: str,
) -> tuple[dict, list[float], list[tuple[str, str, str]]]:
    """Own-episode vs. unrelated-episode CHD on the same queries, and the offset against raw KL."""
    sampler = DistractorSampler(world.distractor_pool(), 0.5)
    pick = make_generator(world.seed, 0, f"e0:distractors:{label}")
    own: list[tuple[str, str, str]] = []
    unrelated: list[tuple[str, str, str]] = []
    for index, (query, episode) in enumerate(triples):
        context = format_context([episode])
        # The same query with a context about something else: the query's difficulty cancels.
        other = _other_episode(episode, pool, index)
        distractor = sampler.sample(episode.entities, len(context), pick)
        if other is None or distractor is None:
            continue
        distractor_text = format_context([distractor])
        own.append((query, context, distractor_text))
        unrelated.append((query, format_context([other]), distractor_text))
    own_values = _run_chd(handle, world, own, k, tokens, f"e0:chd_own:{label}")
    unrelated_values = _run_chd(handle, world, unrelated, k, tokens, f"e0:chd_unrelated:{label}")
    area = auroc([1] * len(own_values) + [0] * len(unrelated_values), own_values + unrelated_values)
    results: dict[str, dict] = {"h1_episode_auroc": _criterion(area, THRESHOLDS["h1_episode_auroc"], len(own_values))}
    raw = (
        raw_kl_nocontext(
            handle,
            [q for q, _, _ in unrelated],
            [e for _, e, _ in unrelated],
            k,
            tokens,
            make_generator(world.seed, 0, f"e0:raw:{label}"),
        ).per_token.tolist()
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
    return results, own_values, own


def _known_vs_fictional(
    handle: Any, world: SyntheticWorld, known: Sequence[dict], k: int, tokens: int
) -> dict:
    """A1.10: within type. A model that knows a fact needs the context less, so fictional is the positive class."""
    sampler = DistractorSampler(world.distractor_pool(), 0.5)
    pick = make_generator(world.seed, 0, "e0:distractors:known")
    by_type: dict[str, list[tuple[str, str, str]]] = {}
    for fact in known:
        context = fact["claim"].format(value=fact["answer"])
        distractor = sampler.sample([fact["answer"]], len(context), pick)
        if distractor is not None:
            by_type.setdefault(fact["type"], []).append((fact["question"], context, format_context([distractor])))
    pool = [p for p in world.probes if p.kind in ("direct", "reverse") and p.vocab in ATTR_TYPE]
    areas: dict[str, dict] = {}
    weighted, weight = 0.0, 0
    for kind in sorted(by_type):
        candidates = [p for p in pool if ATTR_TYPE[p.vocab] == kind]
        chosen = stratified_sample(candidates, len(by_type[kind]), world.seed, f"h1:{kind}")
        fictional = []
        for probe in chosen:
            episode = world.episode(probe.episode_ids[0])
            context = format_context([episode])
            distractor = sampler.sample(episode.entities, len(context), pick)
            if distractor is not None:
                fictional.append((probe.query, context, format_context([distractor])))
        known_values = _run_chd(handle, world, by_type[kind], k, tokens, f"e0:chd_known:{kind}")
        fictional_values = _run_chd(handle, world, fictional, k, tokens, f"e0:chd_fictional:{kind}")
        area = auroc([0] * len(known_values) + [1] * len(fictional_values), known_values + fictional_values)
        areas[kind] = {"auroc": area, "known": len(known_values), "fictional": len(fictional_values)}
        if area is not None:
            weighted += area * len(known_values)
            weight += len(known_values)
    value = weighted / weight if weight else None
    return {**_criterion(value, THRESHOLDS["h1_known_auroc"], weight), "by_type": areas}


def h1_criteria(
    handle: Any, world: SyntheticWorld, known: Sequence[dict], limit: Optional[int], k: int, tokens: int
) -> dict:
    """E0 settings, on the fixed direct templates."""
    probes = stratified_sample([p for p in world.probes if p.kind == "direct"], limit, world.seed, "h1:direct")
    pairs = [(p.query, world.episode(p.episode_ids[0])) for p in probes]
    results, _, _ = _episode_vs_unrelated(handle, world, pairs, [e for _, e in pairs], k, tokens, "e0")
    results["h1_known_auroc"] = _known_vs_fictional(handle, world, known, k, tokens)
    return results


def h1_deployed(handle: Any, world: SyntheticWorld, deployed: dict) -> dict:
    """A1.10: the same test at the settings the experiments use, on questions the model writes itself."""
    cfg = QueryConfig(
        per_episode=int(deployed["per_episode"]),
        model_samples=int(deployed["model_samples"]),
        model_max_new_tokens=int(deployed["model_max_new_tokens"]),
        temperature=float(deployed["temperature"]),
    )
    gen = ModelQueryGen(handle, cfg)
    direct = stratified_sample(
        [p for p in world.probes if p.kind == "direct"], int(deployed["episodes"]), world.seed, "h1:deployed"
    )
    episodes = [world.episode(p.episode_ids[0]) for p in direct]
    pick = make_generator(world.seed, 0, "e0:deployed_queries")
    pairs: list[tuple[str, Episode]] = []
    parsed = 0
    for episode in episodes:
        questions = gen.generate([episode], pick)
        parsed += int(bool(questions))
        pairs.extend((question, episode) for question in questions)
    results, _, _ = _episode_vs_unrelated(
        handle, world, pairs, episodes, int(deployed["k"]), int(deployed["chd_tokens"]), "deployed"
    )
    results["parse_rate"] = {"value": parsed / len(episodes) if episodes else None, "queries": len(pairs)}
    return results


# ---- a candidate, and the verdict ----------------------------------------------------------------------------


def make_handle(model: Any, tokenizer: Any, model_id: str, device: str, fp16: bool, batch_size: int) -> ModelHandle:
    from run import stop_tokens

    return ModelHandle(
        model,
        tokenizer,
        device,
        ComputeLedger(),
        PromptFormat(chat=is_instruct_model(model_id)),  # A1.10: chat template on Instruct, raw on base.
        batch_size,
        fp16,
        stop_tokens(model.model if isinstance(model, DeltaModel) else model, tokenizer),
    )


def evaluate_candidate(
    model: Any, tokenizer: Any, model_id: str, config: dict, device: str, facts: Sequence[dict]
) -> dict:
    started = time.time()
    fp16 = bool(config.get("fp16", False)) and device.startswith("cuda")
    handle = make_handle(model, tokenizer, model_id, device, fp16, int(config.get("batch_size", 16)))
    world = SyntheticWorld(WorldConfig(**config["world"]), int(config["world_seed"]))
    limit = config.get("max_probes")
    max_new = int(config.get("max_new_tokens", 24))
    with handle.ledger.exempt():
        familiarity = check_unknown(handle, world, limit, max_new, answer_mode=True)
        known = known_facts(handle, facts, world.seed, max_new)
        h0 = h0_criteria(handle, world, known, limit, max_new)
        h1 = h1_criteria(
            handle,
            world,
            known,
            config.get("h1_probes", limit),
            int(config.get("k", 8)),
            int(config.get("chd_tokens", 24)),
        )
        deployed = h1_deployed(handle, world, config["deployed"]) if config.get("deployed") else None
    entry = {
        "family": "instruct" if is_instruct_model(model_id) else "base",
        "prompting": "chat template" if is_instruct_model(model_id) else "raw completion",
        "device": device,
        "fp16": fp16,
        "names_unknown": familiarity,
        "known_real_facts": len(known),
        "h0": h0,
        "h1": h1,
        "h1_deployed": deployed,
        "h0_passed": all(c["passed"] for c in h0.values()),
        "h1_passed": all(c["passed"] for c in h1.values()),
        "h1_deployed_passed": None
        if deployed is None
        else all(deployed[name]["passed"] for name in ("h1_episode_auroc", "h1_lower_offset")),
        "tokens": handle.ledger.exempt_forward,
        "seconds": time.time() - started,
    }
    return entry


def _smallest_passing(candidates: dict[str, dict], order: Sequence[str]) -> Optional[str]:
    return next((m for m in order if m in candidates and candidates[m]["h0_passed"]), None)


def verdict(candidates: dict[str, dict], order: Sequence[str]) -> dict:
    """PREREG: the smallest model passing all four H0 criteria. Name familiarity is reported only (A1.10)."""
    chosen = _smallest_passing(candidates, order)
    missing = [m for m in order if m not in candidates]
    if chosen is not None:
        action = "use this model"
    elif missing:
        action = "not finished: some candidates are still to run"
    else:
        action = "stop and report: the teacher is unusable at Colab scale; hard-label arms only"
    h1 = candidates[chosen]["h1_passed"] if chosen else None
    deployed = candidates[chosen].get("h1_deployed_passed") if chosen else None
    remove = h1 is False or deployed is False
    return {
        "chosen_model": chosen,
        "chosen_by_family": {
            family: _smallest_passing(candidates, [m for m in order if is_instruct_model(m) == (family == "instruct")])
            for family in ("base", "instruct")
        },
        "action": action,
        "h1_passed_on_chosen": h1,
        "h1_deployed_passed_on_chosen": deployed,
        "h1_action": "remove the control layer; E2 and E3 become replay-baseline comparisons" if remove else None,
        "names_to_replace": candidates[chosen]["names_unknown"]["replace"] if chosen else [],
        "not_run": missing,
    }


def run_e0(
    config: dict,
    out: Path,
    factory: Any = None,
    device: Optional[str] = None,
    facts: Optional[Sequence[dict]] = None,
) -> dict:
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
        fp16 = bool(config.get("fp16", False)) and device.startswith("cuda")
        model, tokenizer = factory(model_id, fp16)
        candidates[model_id] = evaluate_candidate(model, tokenizer, model_id, config, device, facts)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payload = {"thresholds": THRESHOLDS, "candidates": candidates, "verdict": verdict(candidates, order)}
        write_json_atomic(path, payload)
    write_json_atomic(path, payload)
    return payload


# ---- smoke ---------------------------------------------------------------------------------------------------


def _check(checks: dict, name: str, body: Any) -> Any:
    """Run one check, so that a crash in it is a failure on record and the others still run."""
    try:
        checks[name] = body()
    except Exception as error:  # noqa: BLE001 - a smoke test reports what broke, whatever it was.
        checks[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    return checks[name]


def newline_token_ids(tokenizer: Any) -> set[int]:
    """Every token whose text contains a newline. Computed here, so the check does not trust the handle's own set."""
    size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    return {i for i in range(size) if "\n" in tokenizer.decode([i], skip_special_tokens=False)}


def tokens_after_answer_line(token_ids: Sequence[int], stops: set[int]) -> int:
    """How many tokens follow the first stop token: what signals would wrongly sum over without the stop."""
    for position, token in enumerate(token_ids):
        if token in stops:
            return len(token_ids) - position - 1
    return 0


def run_smoke(
    model_id: str,
    config: dict,
    out: Path,
    factory: Any = None,
    device: Optional[str] = None,
) -> dict:
    """Section D of the review, for one model and 8 prompts. Each check's pass or fail goes in the JSON."""
    from run import default_device, load_hf_model, write_json_atomic

    factory = factory or load_hf_model
    device = device or default_device()
    fp16 = bool(config.get("fp16", False)) and device.startswith("cuda")
    model, tokenizer = factory(model_id, fp16)
    delta = DeltaModel(model, rank=4, alpha=4.0)
    handle = make_handle(delta, tokenizer, model_id, device, fp16, 8)
    world = SyntheticWorld(WorldConfig(**config["world"]), int(config["world_seed"]))
    direct = stratified_sample([p for p in world.probes if p.kind == "direct"], 8, world.seed, "smoke")
    # Half closed-book, half with context, so that a left-padded batch really mixes lengths.
    prompts = [
        handle.format(p.query, _context_of(world, p.episode_ids) if i % 2 else None) for i, p in enumerate(direct)
    ]
    prompt_ids = [handle.encode_prompt(text) for text in prompts]
    newline_ids = newline_token_ids(tokenizer)
    per_token_tolerance = 0.1 if fp16 else 0.01
    checks: dict[str, dict] = {}
    report: dict[str, Any] = {"model_id": model_id, "device": device, "fp16": fp16, "checks": checks}

    def logprob_match() -> dict:
        with handle.ledger.exempt():
            drawn = handle.sample(prompt_ids, 1, 12, 1.0, make_generator(world.seed, 0, "smoke:logprob"))
            conts = [row[0].token_ids for row in drawn]
            scored = handle.sequence_logprob(prompt_ids, conts).tolist()
        differences = [abs(row[0].logprob - s) / max(1, len(row[0].token_ids)) for row, s in zip(drawn, scored)]
        worst = max(differences) if differences else None
        return {
            "worst_per_token": worst,
            "tolerance_per_token": per_token_tolerance,
            "prompt_lengths": [len(ids) for ids in prompt_ids],
            "passed": worst is not None and worst <= per_token_tolerance,
        }

    def logits_finite() -> dict:
        with handle.ledger.exempt():
            drawn = handle.sample(prompt_ids, 1, 4, 1.0, make_generator(world.seed, 0, "smoke:finite"))
            conts = [row[0].token_ids for row in drawn]
            finite = all(bool(torch.isfinite(l).all()) for _, l in handle.iter_answer_logits(prompt_ids, conts))
        return {"passed": finite}

    def training_step() -> dict:
        before = delta.base_fingerprint()
        fresh_before = [p.detach().clone() for p in delta.fresh_parameters()]
        hard = hard_examples(handle, world.episodes(0)[:4])
        result = train_loop(
            handle,
            hard,
            [],
            [],
            TrainConfig(lr=1e-3, hard_batch=2, kl_batch=0, anchor_batch=0, gradient_checkpointing=True),
            make_generator(world.seed, 0, "smoke:train"),
            max_steps=3,  # The scaler may skip its first steps.
        )
        moved = any(not torch.equal(a, b) for a, b in zip(fresh_before, delta.fresh_parameters()))
        same_base = delta.base_fingerprint() == before
        return {
            "steps": result.steps,
            "nonfinite_steps": result.nonfinite_steps,
            "fresh_moved": moved,
            "base_unchanged": same_base,
            "passed": result.steps >= 1 and moved and same_base,
        }

    def raw_completions() -> dict:
        with handle.ledger.exempt():
            drawn = handle.sample(prompt_ids, 3, 24, 1.0, make_generator(world.seed, 0, "smoke:raw"))
        flat = [(prompts[i], sample) for i, row in enumerate(drawn) for sample in row][:20]
        report["completions"] = [handle.decode(sample.token_ids) for _, sample in flat]
        after = [tokens_after_answer_line(sample.token_ids, newline_ids | handle.eos_ids) for _, sample in flat]
        return {
            "mean_tokens_after_answer_line": sum(after) / len(after) if after else None,
            "max_tokens_after_answer_line": max(after) if after else None,
            "n": len(after),
            "passed": len(after) == 20,
        }

    def stop_works() -> dict:
        before = set(handle.eos_ids)
        stops = newline_ids | before
        with handle.ledger.exempt():
            drawn = handle.sample(prompt_ids, 3, 24, 1.0, make_generator(world.seed, 0, "smoke:stop"), answer_mode=True)
        samples = [sample for row in drawn for sample in row]
        clean = all(tokens_after_answer_line(s.token_ids, stops) == 0 for s in samples)
        flagged = all(s.finished == (s.token_ids[-1] in stops) for s in samples if s.token_ids)
        return {
            "passed": clean and flagged and handle.eos_ids == before,
            "stop_token_count": len(stops),
            "eos_ids": sorted(before),
        }

    def question_sets() -> dict:
        episodes = [e for night in range(world.nights) for e in world.episodes(night)][:20]
        instruction = QueryConfig().instruction
        rendered = [handle.format(instruction, e.text) for e in episodes]
        with handle.ledger.exempt():
            drawn = handle.sample(rendered, 1, 64, 0.8, make_generator(world.seed, 0, "smoke:questions"))
        sets = []
        parsed = 0
        for episode, row in zip(episodes, drawn):
            text = handle.decode(row[0].token_ids)
            questions = parse_questions(text)
            parsed += int(bool(questions))
            sets.append({"episode": episode.text, "raw": text, "questions": questions})
        report["question_sets"] = sets
        rate = parsed / len(sets) if sets else 0.0
        return {"parse_rate": rate, "n": len(sets), "minimum": SMOKE_MIN_PARSE_RATE, "passed": rate >= SMOKE_MIN_PARSE_RATE}

    for name, body in (
        ("logprob_match", logprob_match),
        ("logits_finite", logits_finite),
        ("raw_completions", raw_completions),
        ("answer_stop", stop_works),
        ("question_parse", question_sets),
        ("training_step", training_step),  # Last: it moves fresh, and the others should see the base model.
    ):
        _check(checks, name, body)
    report["passed"] = all(c["passed"] for c in checks.values())
    for index, text in enumerate(report.get("completions", [])):
        print(f"--- completion {index}\n{text!r}", flush=True)
    for index, entry in enumerate(report.get("question_sets", [])):
        print(f"--- question set {index}\n{entry['raw']!r}\nparsed: {entry['questions']}", flush=True)
    print(json.dumps(checks, indent=1), flush=True)
    write_json_atomic(Path(out) / f"smoke_{model_id.replace('/', '__')}.json", report)
    return report




def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="E0: teacher and signal validity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", default=None, help="comma-separated model ids, replacing the config's")
    parser.add_argument("--smoke", action="store_true", help="section D's checks on one model, not E0 itself")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    if args.models:
        config["model_ids"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.smoke:
        report = run_smoke(config["model_ids"][0], config, Path(args.out), device=args.device)
        sys.exit(0 if report["passed"] else 1)
    result = run_e0(config, Path(args.out), device=args.device)
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
