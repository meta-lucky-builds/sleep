"""E0b: replacement signals for H1a (PREREG A2.3). Trains nothing. Base 0.5B only.

    python experiments/run_e0b.py --config configs/e0b.json --out DIR
    python experiments/run_e0b.py --config configs/e0b.json --out DIR --confirm NAME --tune-output FILE

The first form tunes on the config's first seed: every candidate, on generated questions, and on template questions as a
sanity check. It picks a winner by a rule fixed in `select_winner` and writes `e0b_tune.json` and `winner.txt`. The
second form tests one candidate on the second seed, and refuses any name other than the winner recorded in the tune
output. Only that candidate is computed there. It writes `e0b_confirm.json`, holding A2.3's verdict.
"""
from __future__ import annotations

import argparse
import hashlib
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

from llmsleep.config import QueryConfig, derive_seed, make_generator  # noqa: E402
from llmsleep.eval.probes import stratified_sample  # noqa: E402
from llmsleep.eval.query_labels import POSITIVE, THIRD, label_for_episode  # noqa: E402
from llmsleep.handle import is_instruct_model  # noqa: E402
from llmsleep.memory import DistractorSampler, Episode, format_context  # noqa: E402
from llmsleep.phases.queries import ModelQueryGen, normalize_query  # noqa: E402
from llmsleep.signals import CANDIDATES, INCUMBENT, CandidateSpec, DistractorMemo, score_candidates  # noqa: E402
from llmsleep.stats import auroc, bootstrap_auroc  # noqa: E402
from llmsleep.worlds import SeededRng, SyntheticWorld, WorldConfig  # noqa: E402

PASS_POSITIVE_VS_NEGATIVE = 0.85
PASS_POSITIVE_VS_THIRD = 0.75
NEGATIVE = "negative"


class ConfirmRefused(ValueError):
    """The confirm run would not be the single pre-specified test that A2.3 allows."""


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def candidate_spec(config: dict) -> CandidateSpec:
    return CandidateSpec(
        k=int(config["k"]),
        max_new_tokens=int(config["chd_tokens"]),
        mix_k=int(config["chd_mix_k"]),
        mix_m=int(config["chd_mix_m"]),
        max_skip=int(config["max_skip"]),
    )


def make_world(config: dict, seed: int) -> SyntheticWorld:
    return SyntheticWorld(WorldConfig(**config["world"]), seed)


def ordered_episodes(world: SyntheticWorld, limit: Optional[int]) -> list[Episode]:
    """A seeded order, balanced over attributes, so that early stopping does not favour one of them."""
    direct = [p for p in world.probes if p.kind == "direct"]
    return [world.episode(p.episode_ids[0]) for p in stratified_sample(direct, limit, world.seed, "e0b:episodes")]


# ---- questions and their labels ------------------------------------------------------------------------------


def collect_generated(handle: Any, world: SyntheticWorld, config: dict) -> tuple[list[dict], dict]:
    """Model-written questions over the world's episodes, each labelled, until the quotas are met or episodes run out."""
    queries = config["queries"]
    generator = ModelQueryGen(
        handle,
        QueryConfig(
            per_episode=int(queries["per_episode"]),
            model_samples=int(queries["model_samples"]),
            model_max_new_tokens=int(queries["model_max_new_tokens"]),
            temperature=float(queries["temperature"]),
        ),
    )
    quota = config["quota"]
    pick = make_generator(world.seed, 0, "e0b:queries")
    labelled: list[dict] = []
    seen: set[str] = set()
    counts = {POSITIVE: 0, THIRD: 0}
    episodes = parsed = 0
    for episode in ordered_episodes(world, int(config["max_episodes"])):
        if counts[POSITIVE] >= int(quota["positive"]) and counts[THIRD] >= int(quota["third"]):
            break
        episodes += 1
        questions = generator.generate([episode], pick)
        parsed += int(bool(questions))
        for question in questions:
            key = normalize_query(question)
            if key in seen:
                continue
            seen.add(key)
            label = label_for_episode(world, question, episode.id)
            labelled.append(
                {
                    "query": question,
                    "episode_id": episode.id,
                    "episode_text": episode.text,
                    "attr": world.fact_of_episode(episode.id).attr,
                    "cls": label.cls,
                    "reason": label.reason,
                }
            )
            if label.cls in counts:
                counts[label.cls] += 1
    stats = {"episodes_tried": episodes, "episodes_with_a_question": parsed, "questions": len(labelled)}
    return labelled, stats


def collect_templates(world: SyntheticWorld, config: dict) -> list[dict]:
    """The world's oracle questions, under the same rule. Own attribute: positive. A sibling's question: third class."""
    quota = config["template_quota"]
    episodes = ordered_episodes(world, None)
    by_subject: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_subject.setdefault(world.fact_of_episode(episode.id).subject, []).append(episode)
    rows: list[dict] = []
    counts = {POSITIVE: 0, THIRD: 0}

    def add(query: str, episode: Episode) -> None:
        label = label_for_episode(world, query, episode.id)
        if label.cls in counts and counts[label.cls] < int(quota[label.cls]):
            counts[label.cls] += 1
            rows.append(
                {
                    "query": query,
                    "episode_id": episode.id,
                    "episode_text": episode.text,
                    "attr": world.fact_of_episode(episode.id).attr,
                    "cls": label.cls,
                    "reason": label.reason,
                }
            )

    for episode in episodes:
        before = counts[POSITIVE]
        for query in world.template_queries(episode):
            if counts[POSITIVE] > before:
                break
            add(query, episode)
        siblings = [e for e in by_subject[world.fact_of_episode(episode.id).subject] if e.id != episode.id]
        if siblings:
            add(world.template_queries(siblings[0])[0], episode)
    return rows


def choose_rows(labelled: Sequence[dict], quota: dict) -> list[dict]:
    """The first rows of each scored class, in generation order, up to its quota."""
    taken = {POSITIVE: 0, THIRD: 0}
    out = []
    for row in labelled:
        if row["cls"] in taken and taken[row["cls"]] < int(quota[row["cls"]]):
            taken[row["cls"]] += 1
            out.append(row)
    return out


def label_report(labelled: Sequence[dict], seed: int, per_class: int) -> dict:
    """How many fell where, and samples of every class and reason, for reading by eye."""
    by_class: dict[str, int] = {}
    by_reason: dict[str, list[dict]] = {}
    by_attr: dict[str, dict[str, int]] = {}
    for row in labelled:
        by_class[row["cls"]] = by_class.get(row["cls"], 0) + 1
        by_reason.setdefault(f"{row['cls']}:{row['reason']}", []).append(row)
        cell = by_attr.setdefault(row["attr"], {})
        cell[row["cls"]] = cell.get(row["cls"], 0) + 1
    samples = {}
    for name in sorted(by_reason):
        rng = SeededRng(derive_seed(seed, -1, f"e0b:label_samples:{name}"))
        chosen = rng.sample(by_reason[name], per_class)
        samples[name] = [{"query": r["query"], "episode": r["episode_text"], "attr": r["attr"]} for r in chosen]
    return {
        "by_class": by_class,
        "by_reason": {name: len(rows) for name, rows in sorted(by_reason.items())},
        "by_attr": by_attr,
        "samples": samples,
    }


# ---- contexts ---------------------------------------------------------------------------------------------------


def unrelated_episode(query: str, episode: Episode, pool: Sequence[Episode], generator: torch.Generator) -> Optional[Episode]:
    """A seeded draw of an episode sharing no entity with `episode`, none of which the query names or its text states."""
    own = {x.casefold() for x in episode.entities}
    lowered = query.casefold()
    eligible = [
        other
        for other in pool
        if other.id != episode.id
        and not own & {x.casefold() for x in other.entities}
        and not any(name in other.text.casefold() for name in own)
        and not any(x.casefold() in lowered for x in other.entities)
    ]
    if not eligible:
        return None
    return eligible[int(torch.randint(len(eligible), (1,), generator=generator))]


def draw_distractors(
    sampler: DistractorSampler, entities: Sequence[str], length: int, count: int, generator: torch.Generator
) -> Optional[list[str]]:
    """`count` different filler contexts, near `length`, about none of `entities`. None if the pool cannot supply them."""
    used: list[str] = []
    out: list[str] = []
    for _ in range(count):
        episode = sampler.sample(entities, length, generator, used)
        if episode is None:
            return None
        used.append(episode.id)
        out.append(format_context([episode]))
    return out


# ---- scoring ----------------------------------------------------------------------------------------------------


def score_rows(
    handle: Any,
    world: SyntheticWorld,
    rows: Sequence[dict],
    names: Sequence[str],
    spec: CandidateSpec,
    tag: str,
    checkpoint: Any = None,
) -> list[dict]:
    """One output row per context. A positive gives a positive and, with the same query and distractors, a negative."""
    pool = [e for night in range(world.nights) for e in world.episodes(night)]
    sampler = DistractorSampler(world.distractor_pool(), 0.5)
    count = spec.mix_m  # Always this many, so a candidate's contexts do not depend on which candidates ran.

    def generator_for(purpose: str) -> torch.Generator:
        return make_generator(world.seed, 0, f"e0b:{tag}:{purpose}")

    out: list[dict] = []
    for index, row in enumerate(rows):
        episode = world.episode(row["episode_id"])
        query = row["query"]
        context = format_context([episode])
        pick = generator_for(f"contexts|{row['episode_id']}|{query}")
        other = unrelated_episode(query, episode, pool, pick) if row["cls"] == POSITIVE else None
        if row["cls"] == POSITIVE and other is None:
            continue
        entities = list(episode.entities) + (list(other.entities) if other is not None else [])
        distractors = draw_distractors(sampler, entities, len(context), count, pick)
        if distractors is None:
            continue
        memo = DistractorMemo()
        own = score_candidates(handle, query, episode.text, context, distractors, names, spec, generator_for, memo)
        out.append(_output(row, row["cls"], episode, own))
        if other is not None:
            against = score_candidates(
                handle, query, other.text, format_context([other]), distractors, names, spec, generator_for, memo
            )
            out.append(_output(row, NEGATIVE, other, against))
        if checkpoint is not None and (index + 1) % 25 == 0:
            checkpoint(out)
    return out


def _output(row: dict, cls: str, context_episode: Episode, scores: Any) -> dict:
    return {
        "query": row["query"],
        "episode_id": row["episode_id"],  # The source of the question, also on a negative: the unit for resampling.
        "attr": row["attr"],
        "cls": cls,
        "context": context_episode.text,
        "values": scores.values,
        "costs": scores.costs,
        "answers_e": [answer.strip()[:60] for answer in scores.answers_e],
    }


def _auroc_entry(
    rows: Sequence[dict], name: str, first: str, second: str, n_boot: int, seed: int
) -> dict:
    high = [r for r in rows if r["cls"] == first and name in r["values"]]
    low = [r for r in rows if r["cls"] == second and name in r["values"]]
    high_values = [r["values"][name] for r in high]
    low_values = [r["values"][name] for r in low]
    value = auroc([1] * len(high) + [0] * len(low), high_values + low_values)
    return {
        "value": value,
        "n_first": len(high),
        "n_second": len(low),
        "ci_queries": bootstrap_auroc(
            high_values, low_values, [r["query"] for r in high], [r["query"] for r in low], n_boot, seed=seed
        ),
        "ci_episodes": bootstrap_auroc(
            high_values, low_values, [r["episode_id"] for r in high], [r["episode_id"] for r in low], n_boot, seed=seed
        ),
    }


def candidate_metrics(rows: Sequence[dict], names: Sequence[str], n_boot: int, seed: int) -> dict[str, dict]:
    """Per candidate: both AUROCs with bootstrap CIs, and mean ledger tokens per query, what it would pay alone."""
    out = {}
    for name in names:
        costs = [r["costs"][name] for r in rows if name in r["costs"]]
        out[name] = {
            "positive_vs_negative": _auroc_entry(rows, name, POSITIVE, NEGATIVE, n_boot, seed),
            "positive_vs_third": _auroc_entry(rows, name, POSITIVE, THIRD, n_boot, seed),
            "cost_per_query": sum(costs) / len(costs) if costs else None,
        }
    return out


def select_winner(
    metrics: dict[str, dict], incumbent: str, cost_cap: float, min_class: int
) -> tuple[Optional[str], dict[str, dict]]:
    """THE RULE. Highest min(positive vs. negative, positive vs. third), among candidates costing at most `cost_cap` times
    the incumbent, with each class at least `min_class` strong. Ties go to the cheaper, then to the name, so the outcome
    is a function of the numbers. (winner or None, one row per candidate saying why.)
    """
    limit = metrics[incumbent]["cost_per_query"] * cost_cap if metrics[incumbent]["cost_per_query"] is not None else None
    table: dict[str, dict] = {}
    for name in sorted(metrics):
        entry = metrics[name]
        first, second = entry["positive_vs_negative"], entry["positive_vs_third"]
        enough = all(
            part["n_first"] >= min_class and part["n_second"] >= min_class for part in (first, second)
        )
        complete = first["value"] is not None and second["value"] is not None
        score = min(first["value"], second["value"]) if complete else None
        affordable = limit is not None and entry["cost_per_query"] is not None and entry["cost_per_query"] <= limit
        table[name] = {
            "score": score,
            "cost_ratio": entry["cost_per_query"] / metrics[incumbent]["cost_per_query"] if limit else None,
            "affordable": affordable,
            "enough_data": enough,
            "eligible": bool(affordable and enough and complete),
        }
    eligible = [name for name in table if table[name]["eligible"]]
    if not eligible:
        return None, table
    winner = min(eligible, key=lambda name: (-table[name]["score"], metrics[name]["cost_per_query"], name))
    return winner, table


def _provenance() -> dict:
    from run import code_identity, package_versions

    return {"code_hash": code_identity(), "versions": package_versions()}


def _load(config: dict, device: str, factory: Any) -> Any:
    from run_e0 import make_handle

    model_id = config["model_id"]
    if is_instruct_model(model_id):
        raise ValueError("A2.3: E0b is base 0.5B only")
    fp16 = bool(config.get("fp16", False)) and device.startswith("cuda")
    model, tokenizer = factory(model_id, fp16)
    return make_handle(model, tokenizer, model_id, device, fp16, int(config.get("batch_size", 16)))


def _evaluate(handle: Any, world: SyntheticWorld, config: dict, names: Sequence[str], out: Path) -> dict:
    """What tune and confirm share: questions, labels, scores on both question sets, metrics. All exempt."""
    from run import write_json_atomic

    spec = candidate_spec(config)
    n_boot = int(config["n_boot"])
    started = time.time()
    with handle.ledger.exempt():
        labelled, stats = collect_generated(handle, world, config)
        chosen = choose_rows(labelled, config["quota"])

        def checkpoint(rows: list[dict]) -> None:
            write_json_atomic(Path(out) / "e0b_partial.json", {"world_seed": world.seed, "rows": rows})

        generated = score_rows(handle, world, chosen, names, spec, "generated", checkpoint)
        template_rows = collect_templates(world, config)
        templates = score_rows(handle, world, template_rows, names, spec, "template")
    return {
        "model_id": config["model_id"],
        "world_seed": world.seed,
        "world_fingerprint": world.fingerprint(),
        "config_hash": config_hash(config),
        "spec": {
            "k": spec.k,
            "max_new_tokens": spec.max_new_tokens,
            "mix_k": spec.mix_k,
            "mix_m": spec.mix_m,
            "max_skip": spec.max_skip,
        },
        **_provenance(),
        "generation": stats,
        "labels": label_report(labelled, world.seed, int(config["samples_per_class"])),
        "template_labels": label_report(template_rows, world.seed, int(config["samples_per_class"])),
        "metrics": candidate_metrics(generated, names, n_boot, world.seed),
        "template_metrics": candidate_metrics(templates, names, n_boot, world.seed),
        "rows": generated,
        "template_rows": templates,
        "tokens": handle.ledger.exempt_forward,
        "seconds": time.time() - started,
    }


def run_tune(config: dict, out: Path, factory: Any = None, device: Optional[str] = None) -> dict:
    """Every candidate on the tuning seed. Writes `e0b_tune.json`, and `winner.txt` if there is a winner."""
    from run import default_device, load_hf_model, write_json_atomic

    device = device or default_device()
    seed = int(config["world_seeds"][0])
    handle = _load(config, device, factory or load_hf_model)
    payload = {"mode": "tune", **_evaluate(handle, make_world(config, seed), config, list(CANDIDATES), Path(out))}
    winner, table = select_winner(
        payload["metrics"], INCUMBENT, float(config["cost_cap"]), int(config["min_class_n"])
    )
    payload["winner"] = winner
    payload["selection"] = table
    payload["rule"] = (
        f"highest min(positive_vs_negative, positive_vs_third) with cost at most {config['cost_cap']} times "
        f"{INCUMBENT} and at least {config['min_class_n']} per class"
    )
    write_json_atomic(Path(out) / "e0b_tune.json", payload)
    if winner is not None:
        (Path(out) / "winner.txt").write_text(winner + "\n", encoding="utf-8")
    return payload


def check_confirm(name: str, tune: dict, config: dict) -> None:
    """A2.3: "the single best candidate ... is then tested once". Anything else is refused, before a model is loaded."""
    if tune.get("mode") != "tune":
        raise ConfirmRefused("the tune output is not a tuning run")
    if tune.get("world_seed") != int(config["world_seeds"][0]):
        raise ConfirmRefused(f"the tune output is from seed {tune.get('world_seed')}, not {config['world_seeds'][0]}")
    if tune.get("config_hash") != config_hash(config):
        raise ConfirmRefused("the config has changed since tuning")
    if tune.get("winner") is None:
        raise ConfirmRefused("tuning found no winner, so there is nothing to confirm")
    if name != tune["winner"]:
        raise ConfirmRefused(f"{name!r} is not the tuning winner, {tune['winner']!r}")


def confirm_verdict(entry: dict, min_class: int) -> dict:
    """A2.3's pass rule, as written. The point estimates decide; the CIs are reported beside them."""
    first, second = entry["positive_vs_negative"], entry["positive_vs_third"]
    enough = all(part["n_first"] >= min_class and part["n_second"] >= min_class for part in (first, second))
    passed = (
        enough
        and first["value"] is not None
        and second["value"] is not None
        and first["value"] >= PASS_POSITIVE_VS_NEGATIVE
        and second["value"] >= PASS_POSITIVE_VS_THIRD
    )
    return {
        "passed": bool(passed),
        "enough_data": enough,
        "positive_vs_negative": first["value"],
        "positive_vs_third": second["value"],
        "thresholds": {"positive_vs_negative": PASS_POSITIVE_VS_NEGATIVE, "positive_vs_third": PASS_POSITIVE_VS_THIRD},
        "action": "selection may be restored by amendment" if passed else "selection is abandoned for this project",
    }


def run_confirm(
    name: str, tune_output: Path, config: dict, out: Path, factory: Any = None, device: Optional[str] = None
) -> dict:
    """The one candidate on the confirming seed. Refuses a second run into the same directory."""
    from run import default_device, load_hf_model, write_json_atomic

    tune = json.loads(Path(tune_output).read_text(encoding="utf-8"))
    check_confirm(name, tune, config)
    path = Path(out) / "e0b_confirm.json"
    if path.exists():
        raise ConfirmRefused(f"{path} already exists: A2.3 allows one test")
    device = device or default_device()
    seed = int(config["world_seeds"][1])
    handle = _load(config, device, factory or load_hf_model)
    payload = {"mode": "confirm", "candidate": name, **_evaluate(handle, make_world(config, seed), config, [name], Path(out))}
    payload["verdict"] = confirm_verdict(payload["metrics"][name], int(config["min_class_n"]))
    incumbent_cost = tune["metrics"][INCUMBENT]["cost_per_query"]
    cost = payload["metrics"][name]["cost_per_query"]
    # No other candidate may run on this seed, so the incumbent's cost is the one measured on the tuning seed.
    payload["verdict"]["cost_ratio_to_tuning_incumbent"] = cost / incumbent_cost if cost and incumbent_cost else None
    write_json_atomic(path, payload)
    return payload


def _print_table(metrics: dict[str, dict]) -> None:
    for name, entry in metrics.items():
        first, second = entry["positive_vs_negative"], entry["positive_vs_third"]
        print(
            f"{name}: pos/neg {first['value']} CI {first['ci_queries']} n={first['n_first']}/{first['n_second']}; "
            f"pos/third {second['value']} CI {second['ci_queries']} n={second['n_first']}/{second['n_second']}; "
            f"tokens per query {entry['cost_per_query']}",
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="E0b: replacement signals for H1a")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confirm", default=None, help="the candidate to confirm, which must be the tuning winner")
    parser.add_argument("--tune-output", default=None, help="e0b_tune.json from the tuning run")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.confirm is None:
        result = run_tune(config, Path(args.out), device=args.device)
        _print_table(result["metrics"])
        print(json.dumps({"winner": result["winner"], "selection": result["selection"]}, indent=1))
        print(json.dumps(result["labels"]["by_reason"], indent=1))
        return
    if args.tune_output is None:
        parser.error("--confirm needs --tune-output")
    result = run_confirm(args.confirm, Path(args.tune_output), config, Path(args.out), device=args.device)
    _print_table(result["metrics"])
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
