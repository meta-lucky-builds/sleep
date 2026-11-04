#!/usr/bin/env python3
"""Apply A3.4 and A4.2 to E1b's results and write E3x's config. Standard library only.

    python scripts/pick_setting.py --results runs/e1b --config configs/e1b.json \
        --template configs/e3x.template.json --out runs/e3x.config.json

Readings this script had to make, each stated so that it can be argued with:
- A3.4's "best sleep arm", at a setting: of the deployable sleep arms that clear the threshold there, the one with the
  highest `ret_real`. `S_oracleq` is left out: its queries are oracle data, so nothing can be deployed at its setting.
- A3.4's "highest common value": over settings, the largest of min(B3_pairs, B4_replay_kl, the best sleep arm's
  `acq_direct`). The threshold is 0.7, or that value if it is lower.
- A4.2's upgrades are ranked at the shared setting, since E3x runs one setting, against that same threshold. If fewer than
  two qualify, E3x runs with fewer, and this says so: an upgrade that fails the rule does not advance by default.

The rule is here, and analyze.py imports it, so that the analysis and the run cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

THRESHOLD = 0.7
BASELINE_ARMS = ("B3_pairs", "B4_replay_kl")
SLEEP_ARMS = ("S_core", "S_coreq")
UPGRADE_ARMS = ("B4q_idk", "B4q_compact", "B4q_r8")
N_UPGRADES = 2
RESERVED_SEEDS = frozenset(range(1001, 1006))  # A4.2: untouched, for a later confirmatory run.
METRICS = ("acq_direct", "ret_real", "false_recall_neighbour")

Table = dict  # setting -> arm -> metric -> mean at the final night.


class PickError(RuntimeError):
    """A result is missing or a rule cannot be applied. E3x must not start."""


def _value(table: Table, setting: str, arm: str, metric: str) -> Optional[float]:
    return table.get(setting, {}).get(arm, {}).get(metric)


def best_setting_per_arm(table: Table, arms: Sequence[str], threshold: float = THRESHOLD) -> dict[str, Optional[str]]:
    """A3.4's first sentence: for each arm, the setting with the highest `ret_real` subject to `acq_direct`."""
    out: dict[str, Optional[str]] = {}
    for arm in arms:
        eligible = [
            setting
            for setting in sorted(table)
            if (_value(table, setting, arm, "acq_direct") or 0.0) >= threshold
            and _value(table, setting, arm, "ret_real") is not None
        ]
        out[arm] = max(eligible, key=lambda s: _value(table, s, arm, "ret_real")) if eligible else None
    return out


def choose_setting(table: Table) -> Optional[dict]:
    """A3.4's shared setting. None if no setting has every arm the rule needs, which only a partial run can show."""
    common: dict[str, float] = {}
    for setting in sorted(table):
        required = [_value(table, setting, arm, "acq_direct") for arm in BASELINE_ARMS]
        sleep = [_value(table, setting, arm, "acq_direct") for arm in SLEEP_ARMS]
        sleep = [value for value in sleep if value is not None]
        if None not in required and sleep:
            common[setting] = min(required + [max(sleep)])
    if not common:
        return None
    highest = max(common.values())
    threshold = min(THRESHOLD, highest)
    rows = []
    for setting in sorted(common):
        if common[setting] < threshold:
            continue
        sleep = [
            arm
            for arm in SLEEP_ARMS
            if (_value(table, setting, arm, "acq_direct") or 0.0) >= threshold
            and _value(table, setting, arm, "ret_real") is not None
        ]
        retained = [_value(table, setting, arm, "ret_real") for arm in BASELINE_ARMS]
        if not sleep or None in retained:
            continue
        best_sleep = max(sleep, key=lambda arm: _value(table, setting, arm, "ret_real"))
        rows.append(
            {
                "setting": setting,
                "best_sleep": best_sleep,
                "min_ret_real": min(retained + [_value(table, setting, best_sleep, "ret_real")]),
            }
        )
    if not rows:
        return None
    winner = max(rows, key=lambda row: row["min_ret_real"])
    return {
        "setting": winner["setting"],
        "best_sleep": winner["best_sleep"],
        "min_ret_real": winner["min_ret_real"],
        "threshold": threshold,
        "relaxed": threshold < THRESHOLD,
        "candidates": rows,
    }


def choose_upgrades(table: Table, setting: str, threshold: float, n: int = N_UPGRADES) -> list[str]:
    """A4.2: largest `ret_real` subject to `acq_direct`, ties to lower `false_recall_neighbour`."""
    eligible = [
        arm
        for arm in UPGRADE_ARMS
        if (_value(table, setting, arm, "acq_direct") or 0.0) >= threshold
        and _value(table, setting, arm, "ret_real") is not None
    ]

    def rank(arm: str) -> tuple[float, float]:
        false_recall = _value(table, setting, arm, "false_recall_neighbour")
        return (-_value(table, setting, arm, "ret_real"), false_recall if false_recall is not None else float("inf"))

    return sorted(eligible, key=rank)[:n]


def _final_record(path: Path, last_night: int) -> Optional[dict]:
    if not path.exists():
        return None
    final = None
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A half-written line: the cell then lacks its last night, and fails below.
            if record.get("night") == last_night:
                final = record
    return final


def load_table(results: Path, config: dict) -> Table:
    """Every expected cell's final night, averaged over seeds. Raises PickError naming all that is missing."""
    if len(config["model_ids"]) != 1:
        raise PickError("E1b is defined on one model")
    tag = config["model_ids"][0].replace("/", "__")
    last_night = config["world"]["nights"] - 1
    table: Table = {}
    missing: list[str] = []
    for variant in config["variants"]:
        name = variant["name"]
        for arm in config["arms"]:
            rows = []
            for seed in config["world_seeds"]:
                path = Path(results) / tag / name / arm / f"seed-{seed}" / "results.jsonl"
                record = _final_record(path, last_night)
                if record is None:
                    missing.append(f"{name}/{arm}/seed-{seed}: no record for night {last_night}")
                    continue
                absent = [m for m in ("acq_direct", "ret_real") if record.get("metrics", {}).get(m) is None]
                if absent:
                    missing.append(f"{name}/{arm}/seed-{seed}: no {', '.join(absent)}")
                    continue
                rows.append(record["metrics"])
            if rows:
                table.setdefault(name, {})[arm] = {
                    metric: _mean([row[metric] for row in rows if row.get(metric) is not None]) for metric in METRICS
                }
    if missing:
        raise PickError(f"{len(missing)} expected cells are missing or incomplete:\n  " + "\n  ".join(missing))
    return table


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _merge(base: dict, overrides: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in overrides.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def build_e3x(template: dict, e1b: dict, choice: dict, upgrades: Sequence[str]) -> dict:
    """The template, with the chosen setting's night values and the chosen upgrades. Refuses reserved seeds."""
    seeds = {int(seed) for seed in template["world_seeds"]}
    if seeds & RESERVED_SEEDS:
        raise PickError(f"seeds {sorted(seeds & RESERVED_SEEDS)} are reserved for a confirmatory run (A4.2)")
    if template.get("seed_role") != "exploratory":
        raise PickError("E3x is labelled exploratory whatever it shows (A4.2)")
    variant = next((v for v in e1b["variants"] if v["name"] == choice["setting"]), None)
    if variant is None:
        raise PickError(f"setting {choice['setting']!r} is not in E1b's config")
    config = _merge(template, {"night": variant.get("night", {})})
    config["arms"] = list(template["arms"]) + [arm for arm in upgrades if arm not in template["arms"]]
    config["picked"] = {"setting": choice["setting"], "threshold": choice["threshold"], "upgrades": list(upgrades)}
    return config


def explain(table: Table, choice: dict, upgrades: Sequence[str]) -> str:
    lines = []
    for row in choice["candidates"]:
        lines.append(f"  {row['setting']}: min ret_real {row['min_ret_real']:.3f}, best sleep arm {row['best_sleep']}")
    relaxed = " (relaxed from 0.7: no setting reached it on all three)" if choice["relaxed"] else ""
    head = [
        f"acq_direct threshold: {choice['threshold']:.3f}{relaxed}",
        "Settings where B3_pairs, B4_replay_kl and a sleep arm all clear it:",
        *lines,
        f"Chosen: {choice['setting']}, which has the highest minimum ret_real ({choice['min_ret_real']:.3f}).",
        "Upgrades at that setting:",
    ]
    for arm in UPGRADE_ARMS:
        row = table.get(choice["setting"], {}).get(arm, {})
        status = "advances" if arm in upgrades else "does not advance"
        fields = ", ".join(f"{m} {row[m]:.3f}" if row.get(m) is not None else f"{m} -" for m in METRICS)
        head.append(f"  {arm}: {fields}: {status}")
    if len(upgrades) < N_UPGRADES:
        head.append(f"WARNING: only {len(upgrades)} upgrades qualify, and A4.2 asks for {N_UPGRADES}. E3x runs with those.")
    return "\n".join(head)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    e1b = json.loads(Path(args.config).read_text(encoding="utf-8"))
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    try:
        table = load_table(Path(args.results), e1b)
        choice = choose_setting(table)
        if choice is None:
            raise PickError("no setting has results for every arm A3.4 needs")
        upgrades = choose_upgrades(table, choice["setting"], choice["threshold"])
        config = build_e3x(template, e1b, choice, upgrades)
    except PickError as error:
        print(f"pick_setting FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    print(explain(table, choice, upgrades), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}: arms {', '.join(config['arms'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
