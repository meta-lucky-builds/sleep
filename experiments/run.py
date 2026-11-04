"""Run an experiment: `python experiments/run.py --config configs/e3.json --out DIR`.

A cell is one (model, variant, arm, world seed). Each cell's directory holds `cell.json`, `checkpoints/`,
`records/` and `results.jsonl`. For each night, in this order:

    run_night -> evaluate -> write records/night-N.json -> State.save (the commit) -> append to results.jsonl

The checkpoint's LATEST pointer is the commit point. A kill before it re-runs the night, identically, since all
seeds derive from (world_seed, night, purpose). A kill after it leaves a record that `results.jsonl` may lack
or hold half of, so on resume `results.jsonl` is rebuilt from the records of committed nights. Wall-clock times
go only in `timing.jsonl`, so `results.jsonl` is byte-identical with or without a kill.

Matched arms (config "matched_arms": {arm: source arm}) need the source's per-night selection counts, so sources
run first within each seed. If the source is incomplete the arm falls back to fraction counts; each night's
report then says `count_source: "fraction"`, and analyze.py will not call that run confirmatory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from llmsleep.config import NightConfig  # noqa: E402
from llmsleep.eval.capability import CapabilityConfig, build_capability, cache_path, question_overlap  # noqa: E402
from llmsleep.eval.drift import build_drift_probe  # noqa: E402
from llmsleep.eval.entrench import evaluate_entrenchment  # noqa: E402
from llmsleep.eval.metrics import EvalConfig, eval_batch_size, evaluate_night  # noqa: E402
from llmsleep.handle import is_instruct_model  # noqa: E402
from llmsleep.phases.night import State, run_night  # noqa: E402
from llmsleep.worlds import SyntheticWorld, WorldConfig  # noqa: E402

ModelFactory = Callable[[str, bool], tuple[Any, Any]]
Hook = Optional[Callable[[str, int], None]]


@dataclass(frozen=True)
class Cell:
    model_id: str
    variant: str
    arm: str
    seed: int


def merge(base: dict, overrides: dict) -> dict:
    """Deep merge; values in `overrides` win. Inputs are not modified."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def code_identity(provided: Optional[str] = None) -> str:
    """A provided hash (Colab receives code as a tarball, with no .git), else the git SHA, else "unknown"."""
    if provided:
        return provided
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_hf_model(model_id: str, fp16: bool) -> tuple[Any, Any]:
    """Weights are fp16 only on a GPU with fp16 asked for. Delta factors stay float32 regardless."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.half() if fp16 and torch.cuda.is_available() else model.float()
    return model.eval(), tokenizer


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(config: dict) -> None:
    """Refuse a confirmatory config that would leave credited budget unspent or skip paired seeds."""
    if config.get("seed_role") == "confirmatory":
        steps = config.get("night", {}).get("train", {}).get("max_steps")
        variant_steps = [v.get("night", {}).get("train", {}).get("max_steps") for v in config.get("variants", [])]
        if steps is not None or any(s is not None for s in variant_steps):
            raise ValueError("a confirmatory run must use train.max_steps = null, or the budget credit goes unspent")
    for arm, source in config.get("matched_arms", {}).items():
        if arm in config.get("arms", []) and source not in config["arms"]:
            print(f"warning: {arm} is matched to {source}, which is not in this config", file=sys.stderr)


def expand_cells(config: dict) -> list[Cell]:
    """Sources of matched arms come before everything else, within each seed."""
    sources = [a for a in config["arms"] if a in set(config.get("matched_arms", {}).values())]
    others = [a for a in config["arms"] if a not in sources]
    variants = [v["name"] for v in config.get("variants", [])] or ["default"]
    return [
        Cell(model_id, variant, arm, int(seed))
        for model_id in config["model_ids"]
        for variant in variants
        for seed in config["world_seeds"]
        for arm in sources + others
    ]


def cell_dir(out: Path, cell: Cell) -> Path:
    tag = cell.model_id.replace("/", "__")
    return Path(out) / tag / cell.variant / cell.arm / f"seed-{cell.seed}"


def _variant(config: dict, name: str) -> dict:
    for variant in config.get("variants", []):
        if variant["name"] == name:
            return variant
    return {}


def read_results(path: Path) -> list[dict]:
    """Records in a results file. A half-written line from a killed writer is skipped with a warning."""
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipped a partial line in {path}", file=sys.stderr)
    return out


def package_versions() -> dict[str, Optional[str]]:
    """What ran this, per result: CI and Colab may differ (review L7)."""
    from importlib import metadata

    versions: dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "numpy", "datasets"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def matched_selection(config: dict, cell: Cell, out: Path, nights: int) -> Optional[dict]:
    """The source arm's report.selection per night, or None unless it covers every night."""
    source = config.get("matched_arms", {}).get(cell.arm)
    if source is None:
        return None
    source_dir = cell_dir(out, Cell(cell.model_id, cell.variant, source, cell.seed))
    records = read_results(source_dir / "results.jsonl")
    selection = {str(r["night"]): r["report"]["selection"] for r in records}
    if sorted(selection, key=int) != [str(n) for n in range(nights)]:
        print(f"warning: {source} is incomplete for {cell}; falling back to fraction counts", file=sys.stderr)
        return None
    return selection


def build_cell_config(config: dict, cell: Cell, out: Path) -> tuple[dict, NightConfig, WorldConfig]:
    """The cell's world and night settings, with the variant applied and the matched counts filled in."""
    variant = _variant(config, cell.variant)
    world_dict = merge(config.get("world", {}), variant.get("world", {}))
    night_dict = merge(config.get("night", {}), variant.get("night", {}))
    # A1.10: per model, not per config: chat template on Instruct, raw completion on base.
    night_dict = merge(night_dict, {"model": {"chat": is_instruct_model(cell.model_id)}})
    world_config = WorldConfig(**world_dict)
    matched = matched_selection(config, cell, out, world_config.nights)
    if matched is not None:
        night_dict = merge(night_dict, {"select": {"matched_selection": matched}})
    night_cfg = NightConfig.from_dict({"arm": cell.arm, **night_dict})
    return {"world": world_dict, "night": night_dict}, night_cfg, world_config


def cell_identity(config: dict, cell: Cell, settings: dict, night_cfg: NightConfig, world: SyntheticWorld) -> dict:
    return {
        "experiment": config.get("experiment"),
        "seed_role": config.get("seed_role"),
        "cell": {"model_id": cell.model_id, "variant": cell.variant, "arm": cell.arm, "seed": cell.seed},
        "settings": settings,
        "eval": config.get("eval", {}),
        "capability": config.get("capability"),
        "night_config_hash": night_cfg.config_hash(),
        "world_fingerprint": world.fingerprint(),
    }


def identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def stop_tokens(model: Any, tokenizer: Any) -> Optional[list[int]]:
    """The tokenizer's EOS plus the generation config's: instruct checkpoints stop on more than one token."""
    ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is not None:
        ids |= set(eos if isinstance(eos, (list, tuple)) else [eos])
    return sorted(ids) or None


def _prepare_state(
    night_cfg: NightConfig, cell: Cell, world: SyntheticWorld, factory: ModelFactory, device: str
) -> State:
    # Weights go fp16 only where autocast will run: `--device cpu` on a GPU machine must stay float32.
    model, tokenizer = factory(cell.model_id, night_cfg.model.fp16 and str(device).startswith("cuda"))
    return State.create(model, tokenizer, world, cell.seed, night_cfg, device, stop_tokens(model, tokenizer))


def _build_suite(config: dict, state: State, out: Path, qa_items: Optional[Sequence[dict]]) -> tuple[Any, Any]:
    eval_cfg = EvalConfig(**config.get("eval", {}))
    capability = None
    if config.get("capability") is not None and eval_cfg.capability:
        cap_cfg = CapabilityConfig(**config["capability"])
        path = cache_path(str(Path(out) / "_cache"), state.handle.delta.base_fingerprint(), cap_cfg, qa_items)
        capability = build_capability(state.handle, cap_cfg, path, qa_items)
    drift = build_drift_probe(state.handle, eval_cfg.drift_tokens) if eval_cfg.drift else None
    return capability, drift


def _rebuild_results(directory: Path, committed_nights: int) -> None:
    """results.jsonl = the records of committed nights, in order, whatever a kill left behind."""
    lines = []
    for night in range(committed_nights):
        record = directory / "records" / f"night-{night:05d}.json"
        if not record.exists():
            raise FileNotFoundError(f"night {night} is committed but its record is missing: {record}")
        lines.append(record.read_text(encoding="utf-8").strip())
    tmp = directory / "results.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write("".join(line + "\n" for line in lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, directory / "results.jsonl")


def _append_result(directory: Path, record: dict) -> None:
    with open(directory / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prune_checkpoints(root: Path, keep: int) -> None:
    """Drop all but the newest `keep`. LATEST always names the newest, so it is never removed."""
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("night-"))
    for name in names[: max(0, len(names) - keep)]:
        shutil.rmtree(root / name)


def _drop_final_weights(config: dict, roots: Path) -> None:
    """With "drop_finished_weights", a finished cell keeps no adapters, to save disk. Its LATEST stays: it says the cell
    is done, so a rerun skips it. Nothing can resume or evaluate from these checkpoints afterwards."""
    if config.get("drop_finished_weights"):
        for path in roots.glob("night-*/adapters.pt"):
            path.unlink()


def run_cell(
    config: dict,
    cell: Cell,
    out: Path,
    factory: ModelFactory = load_hf_model,
    qa_items: Optional[Sequence[dict]] = None,
    code_hash: Optional[str] = None,
    device: Optional[str] = None,
    hook: Hook = None,
) -> Path:
    """Run or resume one cell to its last night. `hook(stage, night)` is for tests that simulate a kill."""
    directory = cell_dir(out, cell)
    directory.mkdir(parents=True, exist_ok=True)
    settings, night_cfg, world_config = build_cell_config(config, cell, out)
    world = SyntheticWorld(world_config, cell.seed)
    identity = cell_identity(config, cell, settings, night_cfg, world)
    cell_hash = identity_hash(identity)
    cell_file = directory / "cell.json"
    if cell_file.exists():
        stored = json.loads(cell_file.read_text(encoding="utf-8"))
        if stored["cell_hash"] != cell_hash:
            raise ValueError(f"{directory} was started with other settings; use a new --out")
    else:
        write_json_atomic(cell_file, {"cell_hash": cell_hash, **identity})

    roots = directory / "checkpoints"
    if (roots / "LATEST").exists() and (roots / "LATEST").read_text(encoding="utf-8").strip() == f"night-{world.nights:05d}":
        _rebuild_results(directory, world.nights)  # A kill may have struck between the last commit and append.
        _drop_final_weights(config, roots)
        return directory

    state = _prepare_state(night_cfg, cell, world, factory, device or default_device())
    if (roots / "LATEST").exists():
        state = State.load(str(roots), state.handle, world)
    _rebuild_results(directory, state.night)
    # Setup runs once per process and its cost depends on the disk cache, so it stays out of the counters,
    # which then hold evaluation only and are the same with or without a kill.
    exempt_forward, exempt_backward = state.ledger.exempt_forward, state.ledger.exempt_backward
    capability, drift = _build_suite(config, state, out, qa_items)
    state.ledger.restore_exempt(exempt_forward, exempt_backward)
    eval_cfg = EvalConfig(**config.get("eval", {}))
    code = code_identity(code_hash)
    versions = package_versions()

    while state.night < world.nights:
        night = state.night
        started = time.time()
        report = run_night(state, night_cfg)
        if hook:
            hook("after_night", night)
        exempt_before = state.ledger.exempt_forward + state.ledger.exempt_backward
        metrics = evaluate_night(
            state.handle,
            state.store,
            night_cfg.arm,
            world,
            night,
            cell.seed,
            eval_cfg,
            capability,
            drift,
            night_cfg.retrieval_k,
        )
        eval_tokens = state.ledger.exempt_forward + state.ledger.exempt_backward - exempt_before
        report_dict = report.to_dict()
        written = report_dict["teacher"].get("frame", {}).get("questions")
        if written is not None and capability is not None:
            metrics["frame_capability_overlap"] = question_overlap(written, capability)
        record = {
            "experiment": config.get("experiment"),
            "seed_role": config.get("seed_role"),
            "model_id": cell.model_id,
            "variant": cell.variant,
            "arm": cell.arm,
            "world_seed": cell.seed,
            "night": night,
            "config_hash": cell_hash,
            "night_config_hash": night_cfg.config_hash(),
            "code_hash": code,
            "versions": versions,
            "world_fingerprint": identity["world_fingerprint"],
            "ledger": report_dict["ledger"],
            "eval_tokens": eval_tokens,
            "decision": report_dict["decision"],
            "metrics": metrics,
            "report": report_dict,
        }
        write_json_atomic(directory / "records" / f"night-{night:05d}.json", record)
        if hook:
            hook("after_record", night)
        state.save(str(roots))  # The commit.
        if hook:
            hook("after_save", night)
        _append_result(directory, record)
        if hook:
            hook("after_append", night)
        _prune_checkpoints(roots, int(config.get("keep_checkpoints", 1)))  # LATEST alone suffices to resume.
        with open(directory / "timing.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"night": night, "seconds": time.time() - started}) + "\n")
        print(f"{cell.arm} seed {cell.seed} night {night} done in {time.time() - started:.0f}s", flush=True)
    _drop_final_weights(config, roots)
    return directory


def evaluate_checkpoint(
    config: dict,
    cell: Cell,
    out: Path,
    factory: ModelFactory = load_hf_model,
    code_hash: Optional[str] = None,
    device: Optional[str] = None,
) -> Path:
    """E4: the contradiction probe on another experiment's final checkpoints. Trains nothing."""
    directory = cell_dir(out, cell)
    result_file = directory / "results.jsonl"
    if result_file.exists():
        return directory
    source = cell_dir(Path(config["source_out"]), cell)
    stored = json.loads((source / "cell.json").read_text(encoding="utf-8"))
    world = SyntheticWorld(WorldConfig(**stored["settings"]["world"]), cell.seed)
    night_cfg = NightConfig.from_dict({"arm": cell.arm, **stored["settings"]["night"]})
    state = _prepare_state(night_cfg, cell, world, factory, device or default_device())
    state = State.load(str(source / "checkpoints"), state.handle, world)
    night = state.night - 1
    eval_cfg = EvalConfig(**config.get("eval", {}))
    handle = state.handle
    with handle.ledger.exempt(), handle.banks("committed"), eval_batch_size(handle, eval_cfg.batch_size):
        metrics, clusters = evaluate_entrenchment(handle, state.store, world, night, cell.seed, eval_cfg)
    metrics["night"] = night
    metrics["per_cluster"] = clusters
    record = {
        "experiment": config.get("experiment"),
        "seed_role": config.get("seed_role"),
        "model_id": cell.model_id,
        "variant": cell.variant,
        "arm": cell.arm,
        "world_seed": cell.seed,
        "night": night,
        "config_hash": stored["cell_hash"],
        "code_hash": code_identity(code_hash),
        "versions": package_versions(),
        "world_fingerprint": world.fingerprint(),
        "metrics": metrics,
        "report": {},
    }
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "results.jsonl.tmp"
    tmp.write_text(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, result_file)
    return directory


def run_experiment(
    config: dict,
    out: Path,
    factory: ModelFactory = load_hf_model,
    qa_items: Optional[Sequence[dict]] = None,
    code_hash: Optional[str] = None,
    device: Optional[str] = None,
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    variants: Optional[Sequence[str]] = None,
) -> list[Path]:
    validate_config(config)
    done = []
    for cell in expand_cells(config):
        if (arms is not None and cell.arm not in arms) or (seeds is not None and cell.seed not in seeds):
            continue
        if variants is not None and cell.variant not in variants:
            continue
        if config.get("mode") == "eval_checkpoints":
            done.append(evaluate_checkpoint(config, cell, Path(out), factory, code_hash, device))
        else:
            done.append(run_cell(config, cell, Path(out), factory, qa_items, code_hash, device))
    return done


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--code-hash", default=None, help="identifies the code when there is no .git")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arms", nargs="*", default=None, help="run only these arms, for sharding sessions")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="run only these variants")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    run_experiment(config, Path(args.out), code_hash=args.code_hash, device=args.device, arms=args.arms, seeds=args.seeds, variants=args.variants)


if __name__ == "__main__":
    main()
