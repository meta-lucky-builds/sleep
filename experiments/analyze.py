"""Apply the PREREG decision rule: `python experiments/analyze.py --results DIR [--json FILE]`.

Readings this script had to make of PREREG; each is stated here so it can be argued with:
- Margin: 3 points applies to accuracies. `drift_generic` has no margin (units are nats), only the CI rule.
- Holm family per model size: H2, H3, H4 (drift difference), H5 (the worst non-inferiority p), H6 (the larger p of
  its two arms), H7. A hypothesis is supported only if its own rule holds AND its Holm-adjusted p is below 0.05.
- H5 "within 3 points": the whole 95% CI of S_full minus S_nocompact lies above -3 points, on each of acq_direct,
  acq_implied and ret_real. "Bounded" delta_rank: final rank at most 1.25 times the mid-run rank, in every seed.
- H4 "widening": the per-seed slope over nights of (S_chain drift - S_full drift) has a CI above zero.
- H6 is within-arm: consolidated minus matched control, predicted negative, 3-point margin, both B3 and S_full.
- H9 is not measured by the current world: conflicts always arrive on a later night.

A run is labelled exploratory when any of these hold: fewer than 5 common seeds; a seed that is not marked
confirmatory; or a matched arm whose counts fell back to fractions (`count_source: "fraction"`). Exploratory results
are reported, but never as confirmatory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from llmsleep.stats import (  # noqa: E402
    BootstrapResult,
    holm,
    non_inferior,
    paired_cluster_bootstrap,
    seed_bootstrap,
    slope,
    supported_difference,
)

MIN_CONFIRMATORY_SEEDS = 5
MARGIN = 0.03
ALPHA = 0.05
MAIN_METRICS = (
    "acq_direct",
    "acq_implied",
    "acq_two_hop",
    "supersede",
    "stale",
    "preservation",
    "false_recall",
    "ret_real",
    "ret_real_head",
    "ret_real_tail",
    "drift_generic",
    "delta_norm",
    "delta_rank",
    "style_length",
    "style_artefact",
)
# Metric name in tables -> key under metrics["per_cluster"].
CLUSTER_KEY = {"acq_direct": "direct", "acq_two_hop": "two_hop"}


def load_records(root: Path) -> list[dict]:
    """Every results.jsonl below `root`. A half-written last line of a dead run is skipped, with a warning."""
    records: list[dict] = []
    for path in sorted(Path(root).rglob("results.jsonl")):
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"warning: skipped a partial line in {path}", file=sys.stderr)
    return records


class Group:
    """Records of one (model, variant): arm -> seed -> night -> record."""

    def __init__(self, records: Sequence[dict]) -> None:
        self.cells: dict[str, dict[int, dict[int, dict]]] = {}
        for record in records:
            self.cells.setdefault(record["arm"], {}).setdefault(record["world_seed"], {})[record["night"]] = record

    def seeds(self, arm: str) -> set[int]:
        return set(self.cells.get(arm, {}))

    def final(self, arm: str, seed: int) -> Optional[dict]:
        nights = self.cells.get(arm, {}).get(seed)
        return nights[max(nights)] if nights else None

    def clusters(self, arm: str, metric: str) -> dict[int, dict]:
        """seed -> cluster -> [num, den], at each seed's final night. Absent metrics are left out."""
        key = CLUSTER_KEY.get(metric, metric)
        out = {}
        for seed in self.seeds(arm):
            record = self.final(arm, seed)
            clusters = record["metrics"].get("per_cluster", {}).get(key) if record else None
            if clusters:
                out[seed] = clusters
        return out

    def series(self, arm: str, seed: int, metric: str) -> tuple[list[int], list[float]]:
        nights = self.cells.get(arm, {}).get(seed, {})
        xs, ys = [], []
        for night in sorted(nights):
            value = nights[night]["metrics"].get(metric)
            if value is not None:
                xs.append(night)
                ys.append(value)
        return xs, ys

    def count_sources(self, arm: str) -> set[str]:
        found = set()
        for nights in self.cells.get(arm, {}).values():
            for record in nights.values():
                source = record.get("report", {}).get("selection", {}).get("count_source")
                if source is not None:
                    found.add(source)
        return found

    def seed_roles(self, arm: str) -> set[Optional[str]]:
        return {
            record.get("seed_role")
            for nights in self.cells.get(arm, {}).values()
            for record in nights.values()
        }


def run_label(group: Group, arms: Sequence[str], seed_count: int) -> tuple[str, list[str]]:
    """("confirmatory" or "exploratory", reasons). A fraction-count run can never be confirmatory."""
    reasons = []
    if seed_count < MIN_CONFIRMATORY_SEEDS:
        reasons.append(f"only {seed_count} seeds; {MIN_CONFIRMATORY_SEEDS} are needed")
    for arm in arms:
        if group.seed_roles(arm) - {"confirmatory"}:
            reasons.append(f"{arm}: seeds not marked confirmatory")
        if "fraction" in group.count_sources(arm):
            reasons.append(f"{arm}: selection counts fell back to fractions")
    return ("exploratory" if reasons else "confirmatory"), reasons


def _common_seeds(group: Group, arms: Sequence[str]) -> int:
    sets = [group.seeds(a) for a in arms]
    return len(set.intersection(*sets)) if sets else 0


def _diff(group: Group, metric: str, a: str, b: str, n_boot: int, seed: int, null: float = 0.0) -> Optional[BootstrapResult]:
    left, right = group.clusters(a, metric), group.clusters(b, metric)
    if not (set(left) & set(right)):
        return None
    return paired_cluster_bootstrap(left, right, n_boot=n_boot, seed=seed, null=null)


def _entry(status: str, arms: Sequence[str], group: Group, p: Optional[float], **extra: Any) -> dict:
    label, reasons = run_label(group, arms, _common_seeds(group, arms))
    return {"status": status, "label": label, "reasons": reasons, "p": p, "arms": list(arms), **extra}


def _result(res: Optional[BootstrapResult]) -> Optional[dict]:
    return res.to_dict() if res is not None else None


def _difference_hypothesis(group: Group, metric: str, a: str, b: str, n_boot: int, seed: int) -> dict:
    result = _diff(group, metric, a, b, n_boot, seed)
    if result is None:
        return _entry("missing", [a, b], group, None, metric=metric)
    status = "supported" if supported_difference(result, +1, MARGIN) else "not supported"
    return _entry(status, [a, b], group, result.p_value, metric=metric, result=_result(result))


def h2(group: Group, n_boot: int, seed: int) -> dict:
    a, b = "S_full", "B4_replay_kl"
    entry = _difference_hypothesis(group, "acq_implied", a, b, n_boot, seed)
    guard = _diff(group, "ret_real", a, b, n_boot, seed)
    entry["guard_ret_real"] = _result(guard)
    if entry["status"] == "supported" and (guard is None or guard.estimate <= -MARGIN):
        entry["status"] = "not supported"
    return entry


def h4(group: Group, n_boot: int, seed: int) -> dict:
    a, b = "S_full", "S_chain"
    drift = _diff(group, "drift_generic", a, b, n_boot, seed)
    retention = _diff(group, "ret_real", a, b, n_boot, seed)
    if drift is None or retention is None:
        return _entry("missing", [a, b], group, None)
    slopes: dict[int, float] = {}
    for s in sorted(group.seeds(a) & group.seeds(b)):
        xs_a, ys_a = group.series(a, s, "drift_generic")
        xs_b, ys_b = group.series(b, s, "drift_generic")
        common = sorted(set(xs_a) & set(xs_b))
        gap = [ys_b[xs_b.index(n)] - ys_a[xs_a.index(n)] for n in common]
        slopes[s] = slope(common, gap)
    finite = {k: v for k, v in slopes.items() if v == v}
    if not finite:
        return _entry("missing", [a, b], group, drift.p_value, drift=_result(drift), retention=_result(retention))
    widening = seed_bootstrap(finite, n_boot=n_boot, seed=seed)
    ok = (
        supported_difference(drift, -1, 0.0)
        and supported_difference(retention, +1, MARGIN)
        and widening.ci_low > 0
    )
    return _entry(
        "supported" if ok else "not supported",
        [a, b],
        group,
        drift.p_value,
        drift=_result(drift),
        retention=_result(retention),
        widening=_result(widening),
    )


def h5(group: Group, n_boot: int, seed: int) -> dict:
    a, b = "S_full", "S_nocompact"
    results = {
        metric: _diff(group, metric, a, b, n_boot, seed, null=-MARGIN)
        for metric in ("acq_direct", "acq_implied", "ret_real")
    }
    if any(r is None for r in results.values()):
        return _entry("missing", [a, b], group, None)
    bounded = True
    ranks = {}
    for s in sorted(group.seeds(a)):
        xs, ys = group.series(a, s, "delta_rank")
        if not xs:
            bounded = False
            continue
        middle = ys[len(ys) // 2]
        ranks[s] = {"mid": middle, "final": ys[-1]}
        bounded = bounded and ys[-1] <= 1.25 * max(middle, 1)
    ok = bounded and all(non_inferior(r, MARGIN) for r in results.values())
    return _entry(
        "supported" if ok else "not supported",
        [a, b],
        group,
        max(r.p_value for r in results.values()),
        results={k: _result(v) for k, v in results.items()},
        ranks=ranks,
        bounded=bounded,
    )


def h6(group: Group, n_boot: int, seed: int) -> dict:
    arms = ("B3_replay", "S_full")
    results = {}
    for arm in arms:
        fact, control = group.clusters(arm, "entrench_follow_fact"), group.clusters(arm, "entrench_follow_control")
        results[arm] = paired_cluster_bootstrap(fact, control, n_boot=n_boot, seed=seed) if set(fact) & set(control) else None
    if any(r is None for r in results.values()):
        return _entry("missing", arms, group, None)
    ok = all(supported_difference(r, -1, MARGIN) for r in results.values())
    return _entry(
        "supported" if ok else "not supported",
        arms,
        group,
        max(r.p_value for r in results.values()),
        results={k: _result(v) for k, v in results.items()},
    )


def h8(group: Group) -> dict:
    """Not in the Holm family: fixed thresholds on AUROC and on the false-retirement rate."""
    arm = "S_full"
    aurocs, false_rates = [], []
    for s in sorted(group.seeds(arm)):
        metrics = group.final(arm, s)["metrics"]
        if metrics.get("audit_auroc") is not None:
            aurocs.append(metrics["audit_auroc"])
        if metrics.get("false_retirement") is not None:
            false_rates.append(metrics["false_retirement"])
    if not aurocs:
        return _entry("missing", [arm], group, None)
    mean_auroc = sum(aurocs) / len(aurocs)
    mean_false = sum(false_rates) / len(false_rates) if false_rates else None
    ok = mean_auroc >= 0.75 and (mean_false is None or mean_false < 0.10)
    return _entry(
        "supported" if ok else "not supported",
        [arm],
        group,
        None,
        mean_auroc=mean_auroc,
        mean_false_retirement=mean_false,
    )


def evaluate_hypotheses(group: Group, n_boot: int = 2000, seed: int = 0) -> dict:
    results = {
        "H2": h2(group, n_boot, seed),
        "H3": _difference_hypothesis(group, "acq_implied", "S_full", "S_noselect", n_boot, seed),
        "H4": h4(group, n_boot, seed),
        "H5": h5(group, n_boot, seed),
        "H6": h6(group, n_boot, seed),
        "H7": _difference_hypothesis(group, "acq_two_hop", "S_full", "S_nopair", n_boot, seed),
    }
    adjusted = holm({name: entry["p"] for name, entry in results.items()})
    for name, entry in results.items():
        entry["p_holm"] = adjusted[name]
        if entry["status"] == "supported" and (adjusted[name] is None or adjusted[name] >= ALPHA):
            entry["status"] = "not supported"
            entry.setdefault("reasons", []).append("did not survive Holm correction")
    results["H8"] = h8(group)
    results["H9"] = {"status": "not measured", "label": "exploratory", "reasons": ["needs a same-night conflict world"]}
    return results


def main_comparison(group: Group, n_boot: int, seed: int) -> dict:
    """PREREG: the result for B3_replay versus S_full is stated first, whatever the outcome."""
    out = {}
    for metric in ("acq_direct", "acq_implied", "supersede", "stale", "false_recall", "ret_real", "drift_generic"):
        result = _diff(group, metric, "S_full", "B3_replay", n_boot, seed)
        out[metric] = _result(result)
    label, reasons = run_label(group, ("S_full", "B3_replay"), _common_seeds(group, ("S_full", "B3_replay")))
    return {"label": label, "reasons": reasons, "differences_S_full_minus_B3": out}


def arm_table(group: Group) -> list[dict]:
    rows = []
    for arm in sorted(group.cells):
        seeds = sorted(group.seeds(arm))
        row: dict[str, Any] = {"arm": arm, "seeds": len(seeds)}
        for metric in MAIN_METRICS:
            values = [group.final(arm, s)["metrics"].get(metric) for s in seeds]
            values = [v for v in values if v is not None]
            row[metric] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def pilot_checks(group: Group) -> dict:
    """E1 kill criterion and budget check, read off whatever arms are present."""
    table = arm_table(group)
    best_implied = max((r["acq_implied"] for r in table if r["acq_implied"] is not None), default=None)
    replay = next((r for r in table if r["arm"] == "B3_replay"), None)
    return {
        "best_acq_implied": best_implied,
        "kill_no_arm_reaches_0.5_implied": best_implied is not None and best_implied < 0.5,
        "B3_acq_direct": replay["acq_direct"] if replay else None,
        "B3_reaches_0.7_direct": bool(replay and replay["acq_direct"] is not None and replay["acq_direct"] >= 0.7),
    }


def format_table(rows: Sequence[dict]) -> str:
    if not rows:
        return "(no arms)"
    columns = list(rows[0])
    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append("-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def format_hypothesis(name: str, entry: dict) -> str:
    text = f"{name}: {entry['status']} [{entry.get('label', '?')}]"
    if entry.get("p_holm") is not None:
        text += f" p_holm={entry['p_holm']:.3f}"
    result = entry.get("result")
    if result:
        text += f" diff={result['estimate']:+.3f} CI[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
    if entry.get("reasons"):
        text += " (" + "; ".join(entry["reasons"]) + ")"
    return text


def analyze(root: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    records = load_records(root)
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["model_id"], record["variant"]), []).append(record)
    out = {}
    for (model_id, variant), rows in sorted(groups.items()):
        group = Group(rows)
        out[f"{model_id}/{variant}"] = {
            "main": main_comparison(group, n_boot, seed),
            "hypotheses": evaluate_hypotheses(group, n_boot, seed),
            "table": arm_table(group),
            "pilot": pilot_checks(group),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Apply the PREREG decision rule to results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    summary = analyze(Path(args.results), args.n_boot, args.seed)
    for key, entry in summary.items():
        print(f"\n=== {key} ===")
        main_part = entry["main"]
        print(f"B3_replay versus S_full [{main_part['label']}]" + (f" ({'; '.join(main_part['reasons'])})" if main_part["reasons"] else ""))
        for metric, result in main_part["differences_S_full_minus_B3"].items():
            if result is None:
                print(f"  {metric}: not available")
            else:
                print(f"  {metric}: {result['estimate']:+.3f} CI[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] seeds={result['n_seeds']}")
        print("\nHypotheses")
        for name, hypothesis in entry["hypotheses"].items():
            print("  " + format_hypothesis(name, hypothesis))
        print("\nArms at the final night (means over seeds)")
        print(format_table(entry["table"]))
        print(f"\nPilot checks: {entry['pilot']}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
