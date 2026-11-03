"""Apply the PREREG decision rule, as amended: `python experiments/analyze.py --results DIR [--json FILE]`.

Readings this script had to make of PREREG; each is stated here so it can be argued with:
- Margin: 3 points applies to accuracies, measured on lift over the permuted-gold control (A1.3). `drift_generic`
  has no margin (units are nats), only the CI rule. H6 compares two follow rates, which are their own control.
- H2 (A1.2) is three members of the Holm family: `S_nopair` vs `B4_replay_kl` on `acq_reverse`, the same on
  `acq_alias`, and `S_full` vs `B3_pairs` on `acq_twohop`. Each has its own `ret_real` guard. H2 as a whole is
  supported only if all three are.
- Holm family per model size: the three H2 members, H3, H4 (drift difference), H5 (the worst non-inferiority p),
  H6 (the larger p of its two arms), H7. Anything missing counts as p = 1 (A1.7). A hypothesis is supported only if
  its own rule holds AND its Holm-adjusted p is below 0.05.
- H5 "within 3 points": the whole 95% CI of S_full minus S_nocompact lies above -3 points, on each of acq_direct,
  acq_reverse, acq_alias, acq_twohop and ret_real. "Bounded" delta_rank: final rank at most 1.25 times the mid-run
  rank, in every seed.
- H4 "widening": the per-seed slope over nights of (S_chain drift - S_full drift) has a CI above zero.
- H6 is within-arm: consolidated minus matched control, predicted negative, 3-point margin, both B3 and S_full.
- H8 is "not measured" with fewer than MIN_RETIRED retirements over all seeds: otherwise its second clause would pass
  on no data (review M7). The minimum is this script's choice, not PREREG's.
- H9 is not measured by the current world: conflicts always arrive on a later night.

A run is labelled exploratory when any of these hold: fewer than 5 common seeds; a seed not marked confirmatory; a
matched arm whose counts fell back to fractions; or A1.7: unequal nights across compared arms, differing world
fingerprints, an acceptance rate below 0.5, an unscored fraction above 0.25. Exploratory results are reported, but
never as confirmatory.
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
MIN_ACCEPTANCE = 0.5
MAX_UNSCORED = 0.25
MIN_RETIRED = 20
HOLM_FAMILY = ("H2_reverse", "H2_alias", "H2_twohop", "H3", "H4", "H5", "H6", "H7")
MAIN_METRICS = (
    "acq_direct",
    "acq_direct_lift",
    "acq_reverse",
    "acq_reverse_lift",
    "acq_alias",
    "acq_alias_lift",
    "acq_twohop",
    "acq_twohop_lift",
    "acq_implied",
    "supersede",
    "stale",
    "preservation",
    "taught_value_bleed",
    "false_recall_any_specific",
    "false_recall_neighbour",
    "ret_real",
    "ret_real_head",
    "ret_real_tail",
    "drift_generic",
    "delta_norm",
    "delta_rank",
    "style_length",
    "style_artefact",
)


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
        out = {}
        for seed in self.seeds(arm):
            record = self.final(arm, seed)
            clusters = record["metrics"].get("per_cluster", {}).get(metric) if record else None
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

    def records(self, arm: str) -> list[dict]:
        return [record for nights in self.cells.get(arm, {}).values() for record in nights.values()]

    def count_sources(self, arm: str) -> set[str]:
        found = set()
        for record in self.records(arm):
            source = record.get("report", {}).get("selection", {}).get("count_source")
            if source is not None:
                found.add(source)
        return found

    def seed_roles(self, arm: str) -> set[Optional[str]]:
        return {record.get("seed_role") for record in self.records(arm)}


def validity_row(group: Group, arm: str) -> dict:
    """What actually happened in an arm (review H8). None where the arm has no such thing, like a frozen arm."""
    decisions = [r.get("decision") for r in group.records(arm)]
    decided = [d for d in decisions if d]
    accepted = sum(1 for d in decided if d.get("accepted"))
    unscored, gate, chains, integrate, unspent = [], [], [], [], []
    spent, allowance = 0, 0
    for record in group.records(arm):
        report = record.get("report", {})
        selection = report.get("selection", {})
        if selection.get("unscored_fraction") is not None:
            unscored.append(selection["unscored_fraction"])
        if selection.get("train"):
            integrate.append(selection["train"].get("integrate", 0))
        if report.get("signals", {}).get("gate_rate") is not None:
            gate.append(report["signals"]["gate_rate"])
        chosen = report.get("synthesis", {}).get("pairs", {}).get("chains_chosen")
        if chosen is not None:
            chains.append(chosen)
        ledger = record.get("ledger") or {}
        if ledger.get("allowance"):
            spent += ledger.get("spent", 0)
            allowance += ledger["allowance"]
            if ledger.get("unspent") is not None:
                unspent.append(ledger["unspent"])
    control = [d["control_drop"] for d in decided if d.get("control_drop") is not None]
    return {
        "arm": arm,
        "seeds": len(group.seeds(arm)),
        "acceptance": accepted / len(decided) if decided else None,
        "unscored": _mean(unscored),
        "gate_rate": _mean(gate),
        "chains_per_night": _mean(chains),
        "integrate_per_night": _mean(integrate),
        "spent_fraction": spent / allowance if allowance else None,
        "unspent_per_night": _mean(unspent),
        "control_drop": _mean(control),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def anchor_drift_table(group: Group) -> list[dict]:
    """Each night's increase in anchor drift, per arm. `max_anchor_drift` is uncalibrated until E1 reads this."""
    rows = []
    for arm in sorted(group.cells):
        values = sorted(
            r["decision"]["anchor_drift"]
            for r in group.records(arm)
            if r.get("decision") and r["decision"].get("anchor_drift") is not None
        )
        if values:
            size = len(values)
            rows.append(
                {
                    "arm": arm,
                    "nights": size,
                    "min": float(values[0]),
                    "median": float(values[size // 2]),
                    "p90": float(values[min(size - 1, int(0.9 * size))]),
                    "max": float(values[-1]),
                }
            )
    return rows


def validity_reasons(group: Group, arms: Sequence[str]) -> list[str]:
    """A1.7, for the arms being compared."""
    reasons = []
    common = set.intersection(*[group.seeds(a) for a in arms]) if arms else set()
    for seed in sorted(common):
        last_nights = {max(group.cells[arm][seed]) for arm in arms}
        if len(last_nights) > 1:
            reasons.append(f"seed {seed}: nights are unequal across {', '.join(arms)}")
        prints = {group.final(arm, seed).get("world_fingerprint") for arm in arms} - {None}
        if len(prints) > 1:
            reasons.append(f"seed {seed}: world fingerprints differ")
    for arm in arms:
        row = validity_row(group, arm)
        if row["acceptance"] is not None and row["acceptance"] < MIN_ACCEPTANCE:
            reasons.append(f"{arm}: acceptance {row['acceptance']:.2f} is below {MIN_ACCEPTANCE}")
        if row["unscored"] is not None and row["unscored"] > MAX_UNSCORED:
            reasons.append(f"{arm}: unscored fraction {row['unscored']:.2f} is above {MAX_UNSCORED}")
    return reasons


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
    reasons += validity_reasons(group, arms)
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
    """The margin applies to lift over the permuted-gold control, so `<metric>_lift` is what is tested."""
    result = _diff(group, f"{metric}_lift", a, b, n_boot, seed)
    if result is None:
        return _entry("missing", [a, b], group, None, metric=metric)
    status = "supported" if supported_difference(result, +1, MARGIN) else "not supported"
    return _entry(status, [a, b], group, result.p_value, metric=metric, result=_result(result))


def h2_member(group: Group, metric: str, a: str, b: str, n_boot: int, seed: int) -> dict:
    entry = _difference_hypothesis(group, metric, a, b, n_boot, seed)
    guard = _diff(group, "ret_real_lift", a, b, n_boot, seed)
    entry["guard_ret_real"] = _result(guard)
    if entry["status"] == "supported" and (guard is None or guard.estimate <= -MARGIN):
        entry["status"] = "not supported"
    return entry


def h4(group: Group, n_boot: int, seed: int) -> dict:
    a, b = "S_full", "S_chain"
    drift = _diff(group, "drift_generic", a, b, n_boot, seed)
    retention = _diff(group, "ret_real_lift", a, b, n_boot, seed)
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
        metric: _diff(group, f"{metric}_lift", a, b, n_boot, seed, null=-MARGIN)
        for metric in ("acq_direct", "acq_reverse", "acq_alias", "acq_twohop", "ret_real")
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
    aurocs, false_rates, retired = [], [], 0
    for s in sorted(group.seeds(arm)):
        metrics = group.final(arm, s)["metrics"]
        if metrics.get("audit_auroc") is not None:
            aurocs.append(metrics["audit_auroc"])
        if metrics.get("false_retirement") is not None:
            false_rates.append(metrics["false_retirement"])
        retired += metrics.get("retired") or 0
    if not aurocs:
        return _entry("missing", [arm], group, None)
    mean_auroc = sum(aurocs) / len(aurocs)
    mean_false = sum(false_rates) / len(false_rates) if false_rates else None
    extra = {"mean_auroc": mean_auroc, "mean_false_retirement": mean_false, "retired": retired}
    if retired < MIN_RETIRED:
        return _entry("not measured", [arm], group, None, note=f"only {retired} retirements; {MIN_RETIRED} needed", **extra)
    ok = mean_auroc >= 0.75 and mean_false is not None and mean_false < 0.10
    return _entry("supported" if ok else "not supported", [arm], group, None, **extra)


def evaluate_hypotheses(group: Group, n_boot: int = 2000, seed: int = 0) -> dict:
    results = {
        "H2_reverse": h2_member(group, "acq_reverse", "S_nopair", "B4_replay_kl", n_boot, seed),
        "H2_alias": h2_member(group, "acq_alias", "S_nopair", "B4_replay_kl", n_boot, seed),
        "H2_twohop": h2_member(group, "acq_twohop", "S_full", "B3_pairs", n_boot, seed),
        "H3": _difference_hypothesis(group, "acq_implied", "S_full", "S_noselect", n_boot, seed),
        "H4": h4(group, n_boot, seed),
        "H5": h5(group, n_boot, seed),
        "H6": h6(group, n_boot, seed),
        "H7": _difference_hypothesis(group, "acq_twohop", "S_full", "S_nopair", n_boot, seed),
    }
    # A1.7: a missing member still takes a slot, so analysing a partial run cannot look stronger.
    adjusted = holm({name: (results[name]["p"] if results[name]["p"] is not None else 1.0) for name in HOLM_FAMILY})
    for name in HOLM_FAMILY:
        entry = results[name]
        entry["p_holm"] = adjusted[name]
        if entry["status"] == "supported" and adjusted[name] >= ALPHA:
            entry["status"] = "not supported"
            entry["reasons"].append("did not survive Holm correction")
    members = [results[name] for name in ("H2_reverse", "H2_alias", "H2_twohop")]
    if any(m["status"] == "missing" for m in members):
        whole = "missing"
    else:
        whole = "supported" if all(m["status"] == "supported" for m in members) else "not supported"
    results["H2"] = {
        "status": whole,
        "label": "confirmatory" if all(m["label"] == "confirmatory" for m in members) else "exploratory",
        "reasons": ["all three H2 members must hold"],
    }
    results["H8"] = h8(group)
    results["H9"] = {"status": "not measured", "label": "exploratory", "reasons": ["needs a same-night conflict world"]}
    return results


def main_comparison(group: Group, n_boot: int, seed: int) -> dict:
    """A1.1: the headline is S_full vs B3_pairs. B3_replay, the original headline, is reported beside it."""
    out = {}
    for baseline in ("B3_pairs", "B3_replay"):
        differences = {}
        for metric in ("acq_direct", "acq_reverse", "acq_alias", "acq_twohop", "supersede", "ret_real"):
            differences[metric] = _result(_diff(group, f"{metric}_lift", "S_full", baseline, n_boot, seed))
        for metric in ("stale", "taught_value_bleed", "false_recall_any_specific", "drift_generic"):
            differences[metric] = _result(_diff(group, metric, "S_full", baseline, n_boot, seed))
        label, reasons = run_label(group, ("S_full", baseline), _common_seeds(group, ("S_full", baseline)))
        out[baseline] = {"label": label, "reasons": reasons, "differences_S_full_minus_baseline": differences}
    return out


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


def age_table(group: Group) -> list[dict]:
    """Direct accuracy by nights since a fact was taught, at the final night (M6): is old knowledge decaying?"""
    rows = []
    for arm in sorted(group.cells):
        pooled: dict[int, list[int]] = {}
        for seed in group.seeds(arm):
            for age, (hits, count) in group.final(arm, seed)["metrics"].get("acq_by_age", {}).items():
                entry = pooled.setdefault(int(age), [0, 0])
                entry[0] += hits
                entry[1] += count
        if pooled:
            rows.append({"arm": arm, **{f"age_{a}": pooled[a][0] / pooled[a][1] for a in sorted(pooled)}})
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
    columns: list[str] = []
    for row in rows:
        columns.extend(c for c in row if c not in columns)
    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
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
    if entry.get("note"):
        text += f" {entry['note']}"
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
            "by_age": age_table(group),
            "validity": [validity_row(group, arm) for arm in sorted(group.cells)],
            "anchor_drift": anchor_drift_table(group),
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
        for baseline, part in entry["main"].items():
            reasons = f" ({'; '.join(part['reasons'])})" if part["reasons"] else ""
            print(f"S_full versus {baseline} [{part['label']}]{reasons}")
            for metric, result in part["differences_S_full_minus_baseline"].items():
                if result is None:
                    print(f"  {metric}: not available")
                else:
                    print(f"  {metric}: {result['estimate']:+.3f} CI[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] seeds={result['n_seeds']}")
        print("\nValidity (acceptance, unscored fraction, integrate count, budget spent)")
        print(format_table(entry["validity"]))
        print("\nAnchor drift increase per night, nats per token (E1 sets max_anchor_drift from this)")
        print(format_table(entry["anchor_drift"]))
        print("\nHypotheses")
        for name, hypothesis in entry["hypotheses"].items():
            print("  " + format_hypothesis(name, hypothesis))
        print("\nArms at the final night (means over seeds)")
        print(format_table(entry["table"]))
        print("\nDirect accuracy by fact age, at the final night")
        print(format_table(entry["by_age"]))
        print(f"\nPilot checks: {entry['pilot']}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
