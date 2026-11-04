"""Apply the PREREG decision rule, as amended: `python experiments/analyze.py --results DIR [--json FILE]`.

Readings this script had to make of PREREG; each is stated here so it can be argued with:
- A2.1: `S_core` replaces `S_full` in H2, H4, H5, H7 and the headline. The amendment does not name H6 or H8, whose arm
  was S_full. Since S_full no longer runs by default, both are read on `S_core` too.
- H3 is withdrawn (A2.1), and leaves the Holm family, unless both S_full and S_noselect have results, which only
  happens if E0b restores selection. S_full is then reported beside the headline, and replaces nothing.
- Margin: 3 points applies to accuracies, measured on lift over the permuted-gold control (A1.3). `drift_generic`
  has no margin (units are nats), only the CI rule. H6 compares two follow rates, which are their own control.
- H2 (A1.2) is three members of the Holm family: `S_nopair` vs `B4_replay_kl` on `acq_reverse`, the same on
  `acq_alias`, and `S_core` vs `B3_pairs` on `acq_twohop`. Each has its own `ret_real` guard. H2 as a whole is
  supported only if all three are.
- Holm family per model size: the three H2 members, H3 if restored, H4 (drift difference), H5 (the worst
  non-inferiority p), H6 (the larger p of its two arms), H7. Anything missing counts as p = 1 (A1.7). A hypothesis is
  supported only if its own rule holds AND its Holm-adjusted p is below 0.05.
- H5 "within 3 points": the whole 95% CI of S_core minus S_nocompact lies above -3 points, on each of acq_direct,
  acq_reverse, acq_alias, acq_twohop and ret_real. "Bounded" delta_rank: final rank at most 1.25 times the mid-run
  rank, in every seed.
- H4 "widening": the per-seed slope over nights of (S_chain drift - S_core drift) has a CI above zero.
- H6 is within-arm: consolidated minus matched control, predicted negative, 3-point margin, both B3 and S_core.
- H8 (A2.2) reads the exempt per-fact log, in which nothing retires, so its second clause is counterfactual: a fact would
  retire after `retire_nights` accepted nights in a row with its probe gap at or below that night's median null gap,
  and it counts as false if the fact is wrong at the final night. H8 is supported only with all of: mean AUROC at least
  0.75 for the held-out gap against correctness at the final night (PREREG's clause); mean AUROC at least 0.75 for the
  fall in gap among facts wrong before a night, gained vs. not gained (A2.2's within-fact test; carrying the threshold
  over is this script's choice); and a false rate under 0.10. It is "not measured" with fewer than MIN_RETIRED such facts
  over all seeds: otherwise the last clause would pass on no data (review M7), so a one-night world such as E2's can
  report the AUROCs but no verdict. The minimum is this script's choice, not PREREG's.
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
for _path in (_ROOT / "src", _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pick_setting  # noqa: E402

from llmsleep.stats import (  # noqa: E402
    BootstrapResult,
    auroc,
    holm,
    non_inferior,
    paired_cluster_bootstrap,
    pearson,
    seed_bootstrap,
    slope,
    supported_difference,
)

HEADLINE_ARM = "S_core"  # A2.1.
RESTORED_ARMS = ("S_full", "S_noselect")  # Run only if E0b restores selection.
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
    a, b = HEADLINE_ARM, "S_chain"
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
    a, b = HEADLINE_ARM, "S_nocompact"
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
    arms = ("B3_replay", HEADLINE_ARM)
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


def _fact_rows(record: dict) -> list[dict]:
    return (record.get("report") or {}).get("fact_audit", {}).get("items", [])


def _gap_auroc(rows: Sequence[dict], gap: str, correct: str) -> Optional[float]:
    """Does a low gap go with a right answer? The score is the negated gap, as in the nightly metric."""
    pairs = [(r[gap], r[correct]) for r in rows if r.get(gap) is not None and r.get(correct) is not None]
    return auroc([int(c) for _, c in pairs], [-g for g, _ in pairs]) if pairs else None


def _changes(rows: Sequence[dict], gap_before: str = "gap_before", gap_after: str = "gap_after") -> list[dict]:
    return [
        {
            "before": r[gap_before],
            "gap_change": r[gap_after] - r[gap_before],
            "correct_change": int(r["correct_after"]) - int(r["correct_before"]),
            "wrong_before": not r["correct_before"],
            "gained": bool(r["correct_after"]),
        }
        for r in rows
        if r.get(gap_before) is not None and r.get(gap_after) is not None
    ]


def _within_fact(changes: Sequence[dict]) -> Optional[float]:
    """Among facts wrong before: does the gap fall further for those now right? Fact by fact, on identical queries."""
    wrong = [c for c in changes if c["wrong_before"]]
    return auroc([int(c["gained"]) for c in wrong], [-c["gap_change"] for c in wrong]) if wrong else None


def _line(changes: Sequence[dict]) -> Optional[tuple[float, float]]:
    """(intercept, slope) of gap change on the gap before. On controls, the slope is regression to the mean."""
    xs, ys = [c["before"] for c in changes], [c["gap_change"] for c in changes]
    b = slope(xs, ys)
    if b != b:
        return None
    return sum(ys) / len(ys) - b * sum(xs) / len(xs), b


def fact_audit_night(rows: Sequence[dict]) -> dict:
    """One night of the log. Pure. What H8 asks, with the controls beside it.

    `taught_net_of_control` is the taught facts' mean gap change minus what the controls' regression predicts at each
    fact's own starting gap: the fall left over after regression to the mean and the night's general movement.
    """
    taught = [r for r in rows if r["role"] in ("tonight", "older")]
    tonight = [r for r in rows if r["role"] == "tonight"]
    controls = [r for r in rows if r["role"] == "control"]
    changes, control_changes = _changes(tonight), _changes(controls)
    line = _line(control_changes)
    net = (
        _mean([c["gap_change"] - (line[0] + line[1] * c["before"]) for c in changes]) if line is not None and changes else None
    )
    taught_line = _line(changes)
    return {
        "taught": len(taught),
        "tonight": len(tonight),
        "controls": len(controls),
        "auroc_probe": _gap_auroc(taught, "gap_after", "correct_after"),
        "auroc_heldout": _gap_auroc(taught, "heldout_after", "correct_after"),
        "auroc_probe_before": _gap_auroc(tonight, "gap_before", "correct_before"),
        "within_fact_auroc": _within_fact(changes),
        "within_fact_auroc_heldout": _within_fact(_changes(tonight, "heldout_before", "heldout_after")),
        "gap_vs_correct_change": pearson([c["gap_change"] for c in changes], [c["correct_change"] for c in changes]),
        "mean_gap_change": _mean([c["gap_change"] for c in changes]),
        "control_mean_gap_change": _mean([c["gap_change"] for c in control_changes]),
        "mean_correct_before": _mean([float(c["wrong_before"] is False) for c in changes]),
        "mean_correct_after": _mean([float(r["correct_after"]) for r in tonight]),
        "control_correct_before": _mean([float(r["correct_before"]) for r in controls]),
        "control_correct_after": _mean([float(r["correct_after"]) for r in controls]),
        "regression_slope": taught_line[1] if taught_line is not None else None,
        "control_regression_slope": line[1] if line is not None else None,
        "taught_net_of_control": net,
    }


def would_retire(records: Sequence[dict]) -> dict[str, bool]:
    """One seed's nights, in order. {fact: wrong at the final night}, for facts retirement would have taken by then.

    Counts as `update_episodes` does: a rejected night neither counts nor resets, and a high gap resets.
    """
    low: dict[str, int] = {}
    retired: set[str] = set()
    last: dict[str, bool] = {}
    for record in records:
        audit = (record.get("report") or {}).get("fact_audit", {})
        null = sorted(audit.get("null_gaps", []))
        threshold = null[len(null) // 2] if null else None
        for row in audit.get("items", []):
            if row["role"] == "control":
                continue
            last[row["key"]] = bool(row["correct_after"])
            if threshold is None or row.get("gap_after") is None or not audit.get("accepted"):
                continue
            low[row["key"]] = low.get(row["key"], 0) + 1 if row["gap_after"] <= threshold else 0
            if low[row["key"]] >= audit.get("retire_nights", 3):
                retired.add(row["key"])
    return {key: not last[key] for key in retired}


def fact_audit_table(group: Group, arm: str = HEADLINE_ARM) -> list[dict]:
    rows = []
    for seed in sorted(group.seeds(arm)):
        for night in sorted(group.cells[arm][seed]):
            items = _fact_rows(group.cells[arm][seed][night])
            if items:
                rows.append({"seed": seed, "night": night, **fact_audit_night(items)})
    return rows


def h8(group: Group) -> dict:
    """Not in the Holm family: fixed thresholds. See the module docstring for the reading."""
    arm = HEADLINE_ARM
    heldout, probe, within, false_flags = [], [], [], []
    for s in sorted(group.seeds(arm)):
        nights = [group.cells[arm][s][n] for n in sorted(group.cells[arm][s])]
        if not any(_fact_rows(r) for r in nights):
            continue
        final = [r for r in _fact_rows(nights[-1]) if r["role"] in ("tonight", "older")]
        for values, gap in ((heldout, "heldout_after"), (probe, "gap_after")):
            area = _gap_auroc(final, gap, "correct_after")
            if area is not None:
                values.append(area)
        # Nights the commit kept: on a rejected one, the weights did not change and neither did the fact.
        changes = []
        for record in nights:
            if (record.get("report") or {}).get("fact_audit", {}).get("accepted"):
                changes += _changes([r for r in _fact_rows(record) if r["role"] == "tonight"])
        area = _within_fact(changes)
        if area is not None:
            within.append(area)
        false_flags += list(would_retire(nights).values())
    if not probe and not heldout:
        return _entry("missing", [arm], group, None)
    mean_heldout, mean_within = _mean(heldout), _mean(within)
    mean_false = _mean([float(flag) for flag in false_flags])
    extra = {
        "mean_auroc_heldout": mean_heldout,
        "mean_auroc_probe": _mean(probe),
        "mean_within_fact_auroc": mean_within,
        "counterfactual_false_retirement": mean_false,
        "counterfactual_retired": len(false_flags),
    }
    if len(false_flags) < MIN_RETIRED:
        note = f"only {len(false_flags)} facts would have retired; {MIN_RETIRED} needed"
        return _entry("not measured", [arm], group, None, note=note, **extra)
    if mean_heldout is None or mean_within is None:
        return _entry("not measured", [arm], group, None, note="no AUROC could be computed", **extra)
    ok = mean_heldout >= 0.75 and mean_within >= 0.75 and mean_false is not None and mean_false < 0.10
    return _entry("supported" if ok else "not supported", [arm], group, None, **extra)


def evaluate_hypotheses(group: Group, n_boot: int = 2000, seed: int = 0) -> dict:
    restored = all(group.seeds(arm) for arm in RESTORED_ARMS)
    results = {
        "H2_reverse": h2_member(group, "acq_reverse", "S_nopair", "B4_replay_kl", n_boot, seed),
        "H2_alias": h2_member(group, "acq_alias", "S_nopair", "B4_replay_kl", n_boot, seed),
        "H2_twohop": h2_member(group, "acq_twohop", HEADLINE_ARM, "B3_pairs", n_boot, seed),
        "H3": _difference_hypothesis(group, "acq_implied", "S_full", "S_noselect", n_boot, seed)
        if restored
        else {
            "status": "withdrawn",
            "label": "exploratory",
            "reasons": ["A2.1: withdrawn unless E0b restores selection"],
            "p": None,
            "arms": list(RESTORED_ARMS),
        },
        "H4": h4(group, n_boot, seed),
        "H5": h5(group, n_boot, seed),
        "H6": h6(group, n_boot, seed),
        "H7": _difference_hypothesis(group, "acq_twohop", HEADLINE_ARM, "S_nopair", n_boot, seed),
    }
    # A1.7: a missing member still takes a slot, so analysing a partial run cannot look stronger. A withdrawn one
    # is not missing: it was removed by rule, before any of this data existed.
    family = [name for name in HOLM_FAMILY if restored or name != "H3"]
    adjusted = holm({name: (results[name]["p"] if results[name]["p"] is not None else 1.0) for name in family})
    for name in family:
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


def main_comparison(group: Group, n_boot: int, seed: int, arm: str = HEADLINE_ARM) -> dict:
    """A1.1, A2.1: the headline is S_core vs B3_pairs. B3_replay, the original headline, is reported beside it."""
    out = {}
    for baseline in ("B3_pairs", "B3_replay"):
        differences = {}
        for metric in ("acq_direct", "acq_reverse", "acq_alias", "acq_twohop", "supersede", "ret_real"):
            differences[metric] = _result(_diff(group, f"{metric}_lift", arm, baseline, n_boot, seed))
        for metric in ("stale", "taught_value_bleed", "false_recall_any_specific", "drift_generic"):
            differences[metric] = _result(_diff(group, metric, arm, baseline, n_boot, seed))
        label, reasons = run_label(group, (arm, baseline), _common_seeds(group, (arm, baseline)))
        out[baseline] = {"label": label, "reasons": reasons, "arm": arm, "differences_minus_baseline": differences}
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
            "restored": main_comparison(group, n_boot, seed, "S_full") if group.seeds("S_full") else None,
            "hypotheses": evaluate_hypotheses(group, n_boot, seed),
            "fact_audit": fact_audit_table(group),
            "table": arm_table(group),
            "by_age": age_table(group),
            "validity": [validity_row(group, arm) for arm in sorted(group.cells)],
            "anchor_drift": anchor_drift_table(group),
            "pilot": pilot_checks(group),
        }
    return out


# A3.3 and A4.1. name: (arm, baseline, the points it must gain on ret_real, and on false_recall_neighbour).
GAP_HYPOTHESES = {
    "H10": ("B4q", "B4_replay_kl", 0.10, 0.10),
    "H11": ("B4q_idk", "B4q", 0.0, 0.15),
    "H12": ("B4q_compact", "B4q", 0.05, 0.0),
    "H13": ("B4q_r8", "B4q", 0.05, 0.0),
}
ACQ_WITHIN = 0.05
E1B_NIGHT_METRICS = ("acq_direct", "ret_real", "false_recall_neighbour")


def gap_hypothesis(table: dict, arm: str, baseline: str, ret_margin: float, false_margin: float) -> dict:
    """A gap at each setting. Testable where both arms reach `acq_direct` 0.7, as A3.3 falsifies H10 on those settings.

    H11 to H13 name one gap each, and a margin of 0 for the other only asks that it is not lost.
    """
    rows = []
    for setting in sorted(table):
        left, right = table[setting].get(arm), table[setting].get(baseline)
        if not left or not right or any(side.get(m) is None for side in (left, right) for m in E1B_NIGHT_METRICS):
            continue
        ret_gap = left["ret_real"] - right["ret_real"]
        false_gap = right["false_recall_neighbour"] - left["false_recall_neighbour"]  # Lower is better, so reversed.
        acq_gap = left["acq_direct"] - right["acq_direct"]
        rows.append(
            {
                "setting": setting,
                "ret_real_gap": ret_gap,
                "false_recall_gap": false_gap,
                "acq_direct_gap": acq_gap,
                "testable": min(left["acq_direct"], right["acq_direct"]) >= pick_setting.THRESHOLD,
                "meets": ret_gap >= ret_margin and false_gap >= false_margin and abs(acq_gap) <= ACQ_WITHIN,
            }
        )
    testable = [row for row in rows if row["testable"]]
    verdict = "not testable" if not testable else "falsified"
    if any(row["meets"] for row in testable):
        verdict = "not falsified"
    return {"arm": arm, "baseline": baseline, "verdict": verdict, "rows": rows}


def e1b_report(records: Sequence[dict]) -> Optional[dict]:
    """Across settings: each night's three metrics, A3.4's choice, H10 to H13, and H8's AUROC. None unless E1b is here."""
    rows = [record for record in records if record.get("experiment") == "E1b"]
    if not rows:
        return None
    by_setting: dict[str, list[dict]] = {}
    for record in rows:
        by_setting.setdefault(record["variant"], []).append(record)
    groups = {setting: Group(part) for setting, part in sorted(by_setting.items())}
    nightly, audit = [], []
    table: dict[str, dict] = {}
    for setting, group in groups.items():
        table[setting] = {}
        for arm in sorted(group.cells):
            finals = [group.final(arm, seed)["metrics"] for seed in group.seeds(arm)]
            table[setting][arm] = {
                m: _mean([f[m] for f in finals if f.get(m) is not None]) for m in E1B_NIGHT_METRICS
            }
            nights = sorted({night for by_night in group.cells[arm].values() for night in by_night})
            for night in nights:
                at_night = [by_night[night]["metrics"] for by_night in group.cells[arm].values() if night in by_night]
                nightly.append(
                    {
                        "arm": arm,
                        "setting": setting,
                        "night": night,
                        **{m: _mean([x[m] for x in at_night if x.get(m) is not None]) for m in E1B_NIGHT_METRICS},
                    }
                )
        result = h8(group)
        audit.append(
            {
                "setting": setting,
                "auroc_heldout": result.get("mean_auroc_heldout"),
                "auroc_probe": result.get("mean_auroc_probe"),
                "within_fact_auroc": result.get("mean_within_fact_auroc"),
            }
        )
    nightly.sort(key=lambda row: (row["arm"], row["setting"], row["night"]))
    arms = sorted({arm for setting in table.values() for arm in setting})
    choice = pick_setting.choose_setting(table)
    return {
        "nightly": nightly,
        "per_arm_setting": pick_setting.best_setting_per_arm(table, arms),
        "shared_setting": choice,
        "upgrades": pick_setting.choose_upgrades(table, choice["setting"], choice["threshold"]) if choice else [],
        "gaps": {name: gap_hypothesis(table, *spec) for name, spec in GAP_HYPOTHESES.items()},
        "h8_by_setting": audit,
    }


def print_e1b(report: dict) -> None:
    print("\n=== E1b, across settings (exploratory, one seed) ===")
    print("\nEach arm, setting and night")
    print(format_table(report["nightly"]))
    print("\nA3.4, per arm: highest ret_real with acq_direct at least 0.7")
    for arm, setting in report["per_arm_setting"].items():
        print(f"  {arm}: {setting if setting is not None else 'no setting qualifies'}")
    choice = report["shared_setting"]
    if choice is None:
        print("\nA3.4, shared setting: not available, since results are incomplete")
    else:
        relaxed = " (relaxed from 0.7)" if choice["relaxed"] else ""
        print(
            f"\nA3.4, chosen setting: {choice['setting']}, min ret_real {choice['min_ret_real']:.3f}, "
            f"best sleep arm {choice['best_sleep']}, acq_direct threshold {choice['threshold']:.3f}{relaxed}"
        )
        print(f"A4.2, upgrades advancing: {', '.join(report['upgrades']) or 'none'}")
    for name, entry in report["gaps"].items():
        print(f"\n{name}: {entry['arm']} minus {entry['baseline']}: {entry['verdict']}")
        print(format_table(entry["rows"]))
    print("\nH8 per setting: AUROC against correctness (held-out gap is A3.7's test, 0.75)")
    print(format_table(report["h8_by_setting"]))


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
        for comparison in (entry["main"], entry["restored"] or {}):
            for baseline, part in comparison.items():
                reasons = f" ({'; '.join(part['reasons'])})" if part["reasons"] else ""
                print(f"{part['arm']} versus {baseline} [{part['label']}]{reasons}")
                for metric, result in part["differences_minus_baseline"].items():
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
        print(f"\nH8's log for {HEADLINE_ARM}, per seed and night: gap vs. correctness, and change against the controls")
        print(format_table(entry["fact_audit"]))
        print("\nArms at the final night (means over seeds)")
        print(format_table(entry["table"]))
        print("\nDirect accuracy by fact age, at the final night")
        print(format_table(entry["by_age"]))
        print(f"\nPilot checks: {entry['pilot']}")
    e1b = e1b_report(load_records(Path(args.results)))
    if e1b is not None:
        print_e1b(e1b)
        summary["E1b"] = e1b
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
