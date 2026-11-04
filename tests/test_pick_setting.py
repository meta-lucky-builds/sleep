"""A3.4 and A4.2's rules, E3x's config, and E1b's tables in analyze.py, on fabricated results. No model."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze  # noqa: E402
import pick_setting  # noqa: E402

from llmsleep.worlds import SyntheticWorld, WorldConfig  # noqa: E402

E1B = json.loads((ROOT / "configs" / "e1b.json").read_text(encoding="utf-8"))
TEMPLATE = json.loads((ROOT / "configs" / "e3x.template.json").read_text(encoding="utf-8"))


def row(acq=0.8, ret=0.5, false_recall=0.3):
    return {"acq_direct": acq, "ret_real": ret, "false_recall_neighbour": false_recall}


def table_of(**settings):
    """Every arm the rule reads at each setting, with `row()` unless the setting says otherwise."""
    arms = pick_setting.BASELINE_ARMS + pick_setting.SLEEP_ARMS + pick_setting.UPGRADE_ARMS
    return {name: {arm: overrides.get(arm, row()) for arm in arms} for name, overrides in settings.items()}


# ---- A3.4 ----------------------------------------------------------------------------------------------------


def test_the_shared_setting_maximises_the_worst_of_the_three():
    table = table_of(
        a={"B3_pairs": row(ret=0.9), "B4_replay_kl": row(ret=0.2), "S_core": row(ret=0.9), "S_coreq": row(ret=0.9)},
        b={"B3_pairs": row(ret=0.5), "B4_replay_kl": row(ret=0.5), "S_core": row(ret=0.5), "S_coreq": row(ret=0.5)},
    )
    choice = pick_setting.choose_setting(table)
    assert choice["setting"] == "b" and choice["min_ret_real"] == 0.5
    assert choice["threshold"] == 0.7 and not choice["relaxed"]


def test_the_best_sleep_arm_is_whichever_clears_the_threshold_with_most_retention():
    table = table_of(a={"S_core": row(acq=0.5, ret=0.9), "S_coreq": row(acq=0.75, ret=0.6)})
    choice = pick_setting.choose_setting(table)
    assert choice["best_sleep"] == "S_coreq" and choice["min_ret_real"] == 0.5


def test_a_setting_failing_the_threshold_cannot_win_however_well_it_retains():
    table = table_of(
        low={"B3_pairs": row(acq=0.4, ret=0.99), "B4_replay_kl": row(ret=0.99), "S_core": row(ret=0.99), "S_coreq": row(ret=0.99)},
        ok={},
    )
    assert pick_setting.choose_setting(table)["setting"] == "ok"


def test_the_threshold_relaxes_to_the_highest_common_value_and_says_so():
    weak = row(acq=0.6)
    table = table_of(
        a={"B3_pairs": weak, "B4_replay_kl": weak, "S_core": row(acq=0.65), "S_coreq": row(acq=0.1)},
        b={"B3_pairs": row(acq=0.3), "B4_replay_kl": row(acq=0.9), "S_core": row(acq=0.9), "S_coreq": row(acq=0.9)},
    )
    choice = pick_setting.choose_setting(table)
    assert choice["relaxed"] and choice["threshold"] == pytest.approx(0.6) and choice["setting"] == "a"
    assert "relaxed" in pick_setting.explain(table, choice, [])


def test_the_oracle_arm_cannot_be_the_best_sleep_arm():
    assert "S_oracleq" not in pick_setting.SLEEP_ARMS


def test_each_arm_has_its_own_best_setting():
    table = {"a": {"X": row(ret=0.9, acq=0.5)}, "b": {"X": row(ret=0.6)}, "c": {"X": row(ret=0.4)}, "d": {"Y": row(acq=0.1)}}
    assert pick_setting.best_setting_per_arm(table, ["X", "Y"]) == {"X": "b", "Y": None}


# ---- A4.2 ----------------------------------------------------------------------------------------------------


def test_upgrades_rank_by_retention_then_lower_false_recall_and_must_clear_the_threshold():
    table = table_of(
        a={
            "B4q_idk": row(ret=0.6, false_recall=0.4),
            "B4q_compact": row(ret=0.6, false_recall=0.1),
            "B4q_r8": row(ret=0.9, acq=0.2),
        }
    )
    assert pick_setting.choose_upgrades(table, "a", 0.7) == ["B4q_compact", "B4q_idk"]
    assert pick_setting.choose_upgrades(table, "a", 0.1) == ["B4q_r8", "B4q_compact"]


# ---- reading results, loudly --------------------------------------------------------------------------------


def write_results(root, config, skip=(), metrics=None):
    tag = config["model_ids"][0].replace("/", "__")
    last = config["world"]["nights"] - 1
    for variant in config["variants"]:
        for arm in config["arms"]:
            for seed in config["world_seeds"]:
                if (variant["name"], arm) in skip:
                    continue
                directory = Path(root) / tag / variant["name"] / arm / f"seed-{seed}"
                directory.mkdir(parents=True)
                lines = [{"night": night, "metrics": {**row(ret=0.1 * night), **(metrics or {})}} for night in range(last + 1)]
                (directory / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def _main(tmp_path):
    config, template = tmp_path / "e1b.json", tmp_path / "e3x.template.json"
    config.write_text(json.dumps(E1B), encoding="utf-8")
    template.write_text(json.dumps(TEMPLATE), encoding="utf-8")
    out = tmp_path / "out" / "e3x.config.json"
    code = pick_setting.main(
        ["--results", str(tmp_path / "e1b"), "--config", str(config), "--template", str(template), "--out", str(out)]
    )
    return code, out


def test_complete_results_write_e3x_and_say_why(tmp_path, capsys):
    write_results(tmp_path / "e1b", E1B)
    code, out = _main(tmp_path)
    assert code == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    printed = capsys.readouterr().out
    assert "Chosen:" in printed and config["picked"]["setting"] in printed
    assert config["experiment"] == "E3x" and config["seed_role"] == "exploratory"
    assert config["world_seeds"] == [2001, 2002, 2003, 2004, 2005] and config["world"]["nights"] == 10
    chosen = next(v for v in E1B["variants"] if v["name"] == config["picked"]["setting"])
    assert config["night"]["train"]["lr"] == chosen["night"]["train"]["lr"]
    assert config["night"]["budget"]["night_tokens"] == chosen["night"]["budget"]["night_tokens"]
    assert len(config["arms"]) == 7 and set(TEMPLATE["arms"]) < set(config["arms"])
    assert config["night"]["train"]["hard_batch"] == TEMPLATE["night"]["train"]["hard_batch"]  # The merge kept the rest.


def test_a_missing_cell_fails_loudly_naming_it_and_writes_nothing(tmp_path, capsys):
    write_results(tmp_path / "e1b", E1B, skip={("lr3e-5_t150k", "B4q_r8")})
    code, out = _main(tmp_path)
    assert code == 1 and not out.exists()
    assert "lr3e-5_t150k/B4q_r8/seed-7" in capsys.readouterr().err


def test_a_cell_short_of_its_last_night_counts_as_missing(tmp_path, capsys):
    write_results(tmp_path / "e1b", E1B)
    tag = E1B["model_ids"][0].replace("/", "__")
    path = tmp_path / "e1b" / tag / "lr1e-5_t50k" / "S_core" / "seed-7" / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + '\n{"night": 2, "metr', encoding="utf-8")
    code, out = _main(tmp_path)
    assert code == 1 and not out.exists()
    assert "lr1e-5_t50k/S_core/seed-7" in capsys.readouterr().err


def test_a_cell_with_no_retention_measure_fails_too(tmp_path):
    write_results(tmp_path / "e1b", E1B, metrics={"ret_real": None})
    code, out = _main(tmp_path)
    assert code == 1 and not out.exists()


# ---- E3x's config --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1001, 1003, 1005])
def test_a_confirmatory_seed_cannot_get_into_e3x(seed):
    bad = {**TEMPLATE, "world_seeds": [2001, seed]}
    choice = {"setting": E1B["variants"][0]["name"], "threshold": 0.7}
    with pytest.raises(pick_setting.PickError):
        pick_setting.build_e3x(bad, E1B, choice, [])


def test_the_shipped_template_holds_no_reserved_seed_and_is_exploratory():
    assert not set(TEMPLATE["world_seeds"]) & set(range(1001, 1006))
    assert TEMPLATE["seed_role"] == "exploratory"
    choice = {"setting": E1B["variants"][0]["name"], "threshold": 0.7}
    with pytest.raises(pick_setting.PickError):
        pick_setting.build_e3x({**TEMPLATE, "seed_role": "confirmatory"}, E1B, choice, [])


def test_the_upgrades_are_not_in_the_template_and_no_oracle_arm_is():
    assert not set(pick_setting.UPGRADE_ARMS) & set(TEMPLATE["arms"])
    assert set(TEMPLATE["arms"]) == {"B3_pairs", "B4_replay_kl", "B4q", "S_core", "S_coreq"}


def test_e3x_has_e1s_facts_per_night_and_its_world_can_be_built():
    """Worlds this size have never been built, and a name forge that runs dry would stop the run at its first cell."""
    e1 = json.loads((ROOT / "configs" / "e1.json").read_text(encoding="utf-8"))

    def per_night(world_settings, seed):
        world = SyntheticWorld(WorldConfig(**world_settings), seed)
        return sum(len(world.episodes(n)) for n in range(world.nights)) / world.nights

    small, large = per_night(e1["world"], 7), per_night(TEMPLATE["world"], 2001)
    assert large == pytest.approx(small, rel=0.05)


def test_pick_setting_uses_the_standard_library_only():
    tree = ast.parse((ROOT / "scripts" / "pick_setting.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= set(sys.stdlib_module_names)


# ---- E1b's tables --------------------------------------------------------------------------------------------


def record(arm, variant, night, **metrics):
    base = row()
    base.update(metrics)
    return {
        "experiment": "E1b", "arm": arm, "variant": variant, "night": night, "world_seed": 7, "model_id": "m",
        "metrics": base, "report": {},
    }  # fmt: skip


def test_other_experiments_get_no_e1b_section():
    assert analyze.e1b_report([{**record("B4q", "a", 0), "experiment": "E1"}]) is None


def test_the_report_has_each_arm_setting_and_night_and_the_chosen_setting(capsys):
    records = [
        record(arm, variant, night, ret_real=0.5 + 0.1 * night)
        for variant in ("a", "b")
        for arm in ("B3_pairs", "B4_replay_kl", "S_core", "B4q", "S_coreq")
        for night in range(3)
    ]
    report = analyze.e1b_report(records)
    assert len(report["nightly"]) == 2 * 5 * 3
    assert report["nightly"][0].keys() >= {"arm", "setting", "night", "acq_direct", "ret_real", "false_recall_neighbour"}
    assert report["shared_setting"]["setting"] in ("a", "b")
    analyze.print_e1b(report)
    printed = capsys.readouterr().out
    assert "chosen setting" in printed and "H10" in printed and "H8 per setting" in printed


def test_a_partial_run_can_still_be_read():
    report = analyze.e1b_report([record("B4q", "a", 0)])
    assert report["shared_setting"] is None and report["upgrades"] == []
    analyze.print_e1b(report)


def _h10(acq=0.8, base_ret=0.4, ret=0.55, base_false=0.5, false=0.35):
    return {"s": {"B4q": row(acq, ret, false), "B4_replay_kl": row(acq, base_ret, base_false)}}


def test_h10_needs_both_gaps_with_false_recall_reversed():
    spec = analyze.GAP_HYPOTHESES["H10"]
    assert analyze.gap_hypothesis(_h10(), *spec)["verdict"] == "not falsified"
    assert analyze.gap_hypothesis(_h10(ret=0.45), *spec)["verdict"] == "falsified"
    assert analyze.gap_hypothesis(_h10(false=0.45), *spec)["verdict"] == "falsified"
    higher_false_recall = analyze.gap_hypothesis(_h10(false=0.7), *spec)
    assert higher_false_recall["rows"][0]["false_recall_gap"] == pytest.approx(-0.2)
    assert higher_false_recall["verdict"] == "falsified"


def test_h10_is_read_only_where_direct_accuracy_reaches_the_threshold_and_acquisition_is_close():
    spec = analyze.GAP_HYPOTHESES["H10"]
    assert analyze.gap_hypothesis(_h10(acq=0.5), *spec)["verdict"] == "not testable"
    far = _h10()
    far["s"]["B4q"]["acq_direct"] = 0.95
    assert analyze.gap_hypothesis(far, *spec)["verdict"] == "falsified"


def test_h11_to_h13_read_their_own_gaps():
    table = {
        "s": {
            "B4q": row(ret=0.5, false_recall=0.5),
            "B4q_idk": row(ret=0.5, false_recall=0.3),
            "B4q_compact": row(ret=0.56),
            "B4q_r8": row(ret=0.52),
        }
    }
    verdicts = {
        name: analyze.gap_hypothesis(table, *analyze.GAP_HYPOTHESES[name])["verdict"] for name in ("H11", "H12", "H13")
    }
    assert verdicts == {"H11": "not falsified", "H12": "not falsified", "H13": "falsified"}
