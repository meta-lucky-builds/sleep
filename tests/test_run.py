"""The runner, E0 and the decision rule, on tiny models and fabricated records. No downloads."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from tiny import TinyTokenizer, tiny_gpt2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import analyze  # noqa: E402
import run as runner  # noqa: E402
import run_e0  # noqa: E402

from llmsleep.arms import ARMS  # noqa: E402
from llmsleep.config import NightConfig  # noqa: E402
from llmsleep.eval.capability import CapabilityConfig  # noqa: E402
from llmsleep.eval.metrics import EvalConfig  # noqa: E402
from llmsleep.worlds import WorldConfig  # noqa: E402

QA = [{"question": "Who wrote it?", "answers": ["Mira"]}, {"question": "What is it?", "answers": ["Kite"]}]


def factory(model_id, fp16):
    tok = TinyTokenizer()
    return tiny_gpt2(tok.vocab_size), tok


def smoke_config(arms=("B3_replay",), nights=2, **extra) -> dict:
    config = {
        "experiment": "smoke",
        "seed_role": "smoke",
        "model_ids": ["tiny/gpt2"],
        "world_seeds": [3],
        "arms": list(arms),
        "world": {
            "n_persons": 2, "n_orgs": 2, "n_cities": 2, "n_regions": 1, "nights": nights,
            "supersession_rate": 0.5, "n_distractors": 8, "n_control_persons": 2, "n_entrench": 2,
        },
        "night": {
            "model": {"rank": 4, "alpha": 4.0, "batch_size": 8},
            "budget": {"night_tokens": 40000},
            "signals": {"k": 2, "max_new_tokens": 3, "chunk": 4},
            "queries": {"per_episode": 3, "model_samples": 1, "model_max_new_tokens": 6, "max_pairs": 2},
            "select": {"n_null": 8, "min_null": 4},
            "teacher": {"samples": 1, "max_new_tokens": 3, "top_k": 8, "self_samples": 2, "self_max_new_tokens": 3},
            "train": {"lr": 0.001, "hard_batch": 2, "kl_batch": 2, "anchor_batch": 2, "max_steps": 2},
            "compact": {"threshold": 0.01, "max_rank": 6, "repair_steps": 1},
        },
        "eval": {
            "max_new_tokens": 3, "batch_size": 8, "max_per_kind": 3, "drift_tokens": 2,
            "entrench_nights": [nights - 1], "entrench_max": 2,
        },
        "capability": {"n_candidates": 2, "n_keep": 2, "k": 2, "max_new_tokens": 3, "batch_size": 4},
    }
    config.update(extra)
    return config


class Killed(Exception):
    pass


def kill_at(stage, night):
    def hook(current_stage, current_night):
        if current_stage == stage and current_night == night:
            raise Killed

    return hook


def only_cell(config):
    cells = runner.expand_cells(config)
    assert len(cells) == 1
    return cells[0]


def run_cell(config, out, hook=None):
    return runner.run_cell(config, only_cell(config), out, factory, QA, "code-1", "cpu", hook)


# ---- resume --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def straight_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("straight")
    return run_cell(smoke_config(), out)


@pytest.mark.parametrize(
    "stage, night",
    [("after_night", 0), ("after_record", 0), ("after_save", 0), ("after_append", 0), ("after_save", 1), ("after_night", 1)],
)
def test_resume_after_a_kill_gives_byte_identical_results(stage, night, straight_run, tmp_path):
    config = smoke_config()
    with pytest.raises(Killed):
        run_cell(config, tmp_path, kill_at(stage, night))
    resumed = run_cell(config, tmp_path)
    assert (resumed / "results.jsonl").read_bytes() == (straight_run / "results.jsonl").read_bytes()
    assert (resumed / "checkpoints" / "LATEST").read_text() == "night-00002"


def test_a_half_written_result_line_is_repaired_on_resume(straight_run, tmp_path):
    config = smoke_config()
    with pytest.raises(Killed):
        run_cell(config, tmp_path, kill_at("after_append", 0))
    with open(runner.cell_dir(tmp_path, only_cell(config)) / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write('{"partial": ')
    resumed = run_cell(config, tmp_path)
    assert (resumed / "results.jsonl").read_bytes() == (straight_run / "results.jsonl").read_bytes()


def test_finished_nights_are_skipped(straight_run, tmp_path):
    config = smoke_config()
    run_cell(config, tmp_path)

    def must_not_run(stage, night):
        raise AssertionError("a finished night ran again")

    directory = run_cell(config, tmp_path, must_not_run)
    assert len(runner.read_results(directory / "results.jsonl")) == 2


def test_another_config_may_not_resume_a_directory(tmp_path):
    config = smoke_config()
    run_cell(config, tmp_path)
    changed = copy.deepcopy(config)
    changed["night"]["budget"]["night_tokens"] = 123
    with pytest.raises(ValueError):
        run_cell(changed, tmp_path)


# ---- what a record holds -----------------------------------------------------------------------------


def test_a_record_has_everything_the_design_asks_for(straight_run):
    records = runner.read_results(straight_run / "results.jsonl")
    assert [r["night"] for r in records] == [0, 1]
    for record in records:
        assert record["config_hash"] and record["code_hash"] == "code-1" and record["world_fingerprint"]
        assert record["ledger"]["spent"] <= 40000
        assert record["report"]["ledger"] == record["ledger"]
        assert record["eval_tokens"] > 0
        assert record["decision"]["accepted"] is True
        metrics = record["metrics"]
        for name in ("acq_direct", "false_recall", "drift_generic", "delta_norm", "delta_rank", "style_length", "ret_real_n"):
            assert name in metrics, name
    assert "entrench_pairs" in records[1]["metrics"] and "entrench_pairs" not in records[0]["metrics"]


def test_exempt_counters_hold_evaluation_only_and_are_in_the_record(straight_run):
    """Each night's report is written before its evaluation, so it holds the earlier nights' tokens. Setup
    (capability, drift probe) is not counted: its cost depends on the disk cache, so it would differ on resume."""
    first, second = runner.read_results(straight_run / "results.jsonl")
    assert first["ledger"]["exempt_forward"] + first["ledger"]["exempt_backward"] == 0
    assert second["ledger"]["exempt_forward"] + second["ledger"]["exempt_backward"] == first["eval_tokens"]


def test_stop_tokens_join_tokenizer_and_generation_config(tmp_path):
    config = smoke_config()
    cell = only_cell(config)
    night_cfg = NightConfig.from_dict({"arm": cell.arm, **config["night"]})
    world = runner.SyntheticWorld(WorldConfig(**config["world"]), cell.seed)

    def several(model_id, fp16):
        model, tok = factory(model_id, fp16)
        model.generation_config.eos_token_id = [tok.eos_token_id, 7]
        return model, tok

    def scalar(model_id, fp16):
        model, tok = factory(model_id, fp16)
        model.generation_config.eos_token_id = 9
        return model, tok

    tokenizer_eos = TinyTokenizer().eos_token_id
    state = runner._prepare_state(night_cfg, cell, world, several, "cpu")
    assert state.handle.eos_ids == {tokenizer_eos, 7}
    state = runner._prepare_state(night_cfg, cell, world, scalar, "cpu")
    assert state.handle.eos_ids == {tokenizer_eos, 9}


def test_capability_cache_is_shared_on_disk(straight_run):
    root = straight_run
    while root.name != "tiny__gpt2":
        root = root.parent
    assert list((root.parent / "_cache").glob("capability-*.json"))


def test_old_checkpoints_are_pruned(tmp_path):
    config = smoke_config(nights=3, keep_checkpoints=1)
    directory = run_cell(config, tmp_path)
    names = sorted(p.name for p in (directory / "checkpoints").iterdir() if p.is_dir())
    assert names == ["night-00003"]


# ---- matched arms --------------------------------------------------------------------------------------


def test_sources_run_first_within_each_seed():
    config = smoke_config(
        arms=("S_noselect", "B3_replay", "S_full"), matched_arms={"S_noselect": "S_full"}, world_seeds=[1, 2]
    )
    cells = runner.expand_cells(config)
    assert [(c.seed, c.arm) for c in cells] == [
        (1, "S_full"), (1, "S_noselect"), (1, "B3_replay"),
        (2, "S_full"), (2, "S_noselect"), (2, "B3_replay"),
    ]


def test_noselect_gets_full_counts_and_falls_back_to_fractions_without_them(tmp_path):
    config = smoke_config(arms=("S_full", "S_noselect"), nights=1, matched_arms={"S_noselect": "S_full"}, capability=None)
    config["eval"]["drift"] = False
    noselect = next(c for c in runner.expand_cells(config) if c.arm == "S_noselect")
    alone = tmp_path / "alone"
    directory = runner.run_cell(config, noselect, alone, factory, None, "code-1", "cpu")
    record = runner.read_results(directory / "results.jsonl")[0]
    assert record["report"]["selection"]["count_source"] == "fraction"

    both = tmp_path / "both"
    dirs = runner.run_experiment(config, both, factory, None, "code-1", "cpu")
    assert [d.parent.name for d in dirs] == ["S_full", "S_noselect"]
    full = runner.read_results(dirs[0] / "results.jsonl")[0]
    matched = runner.read_results(dirs[1] / "results.jsonl")[0]
    assert full["report"]["selection"]["count_source"] == "chd"
    assert matched["report"]["selection"]["count_source"] == "matched"


def test_partial_source_gives_no_counts(tmp_path):
    config = smoke_config(arms=("S_full", "S_noselect"), matched_arms={"S_noselect": "S_full"})
    cells = {c.arm: c for c in runner.expand_cells(config)}
    source = runner.cell_dir(tmp_path, cells["S_full"])
    source.mkdir(parents=True)
    line = {"night": 0, "report": {"selection": {"train": {"integrate": 1, "anchor": 0}, "heldout": {"integrate": 0, "anchor": 0}}}}
    (source / "results.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert runner.matched_selection(config, cells["S_noselect"], tmp_path, 2) is None
    assert runner.matched_selection(config, cells["S_noselect"], tmp_path, 1) is not None
    assert runner.matched_selection(config, cells["S_full"], tmp_path, 1) is None


def test_confirmatory_config_with_capped_steps_is_refused():
    config = smoke_config(seed_role="confirmatory")
    with pytest.raises(ValueError):
        runner.validate_config(config)
    config["night"]["train"]["max_steps"] = None
    runner.validate_config(config)


# ---- E4: entrenchment on checkpoints ---------------------------------------------------------------------


def test_eval_checkpoints_mode_reads_a_finished_run(tmp_path):
    # Enough steps and tokens that replay reaches every episode, which is what marks a fact consolidated.
    source_config = smoke_config()
    source_config["night"]["train"]["max_steps"] = 200
    source_config["night"]["budget"]["night_tokens"] = 400000
    source = tmp_path / "source"
    run_cell(source_config, source)
    config = {
        "experiment": "E4",
        "seed_role": "smoke",
        "mode": "eval_checkpoints",
        "source_out": str(source),
        "model_ids": ["tiny/gpt2"],
        "world_seeds": [3],
        "arms": ["B3_replay"],
        "eval": {"max_new_tokens": 3, "batch_size": 8, "entrench_max": 2},
    }
    dirs = runner.run_experiment(config, tmp_path / "out", factory, None, "code-1", "cpu")
    record = runner.read_results(dirs[0] / "results.jsonl")[0]
    assert record["night"] == 1 and "entrench_pairs" in record["metrics"]
    assert record["metrics"]["entrench_pairs"] == 2  # With these steps, B3 reaches everything.


# ---- configs --------------------------------------------------------------------------------------------


CONFIG_DIR = ROOT / "configs"
ALL_CONFIGS = sorted(CONFIG_DIR.glob("e*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_experiment_has_a_config():
    assert {p.stem for p in ALL_CONFIGS} == {f"e{i}" for i in range(7)}


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_config_is_well_formed(path):
    config = _load(path)
    if path.stem == "e0":
        WorldConfig(**config["world"])
        return
    assert all(arm in ARMS for arm in config["arms"])
    runner.validate_config(config)
    cells = runner.expand_cells(config)
    assert cells
    if config.get("mode") == "eval_checkpoints":
        return
    EvalConfig(**config["eval"])
    CapabilityConfig(**config["capability"])
    for cell in cells:
        variant = runner._variant(config, cell.variant)
        WorldConfig(**runner.merge(config["world"], variant.get("world", {})))
        night = runner.merge(config["night"], variant.get("night", {}))
        parsed = NightConfig.from_dict({"arm": cell.arm, **night})
        if config.get("seed_role") == "confirmatory":
            assert parsed.train.max_steps is None


def test_confirmatory_seeds_are_never_piloted():
    dev, confirmatory = set(), set()
    for path in ALL_CONFIGS:
        config = _load(path)
        seeds = set(config.get("world_seeds", [config.get("world_seed")]))
        (confirmatory if config["seed_role"] == "confirmatory" else dev).update(seeds)
    assert confirmatory and dev and not dev & confirmatory


def test_every_matched_noselect_config_includes_its_source():
    for path in ALL_CONFIGS:
        config = _load(path)
        if "S_noselect" in config.get("arms", []):
            assert config["matched_arms"]["S_noselect"] == "S_full" and "S_full" in config["arms"]


def test_e3_and_e4_world_can_be_shared():
    e3, e4 = _load(CONFIG_DIR / "e3.json"), _load(CONFIG_DIR / "e4.json")
    assert e3["world_seeds"] == e4["world_seeds"] and e3["model_ids"] == e4["model_ids"]
    assert set(e4["arms"]) <= set(e3["arms"])


# ---- E0 ------------------------------------------------------------------------------------------------


def e0_config():
    return {
        "experiment": "E0",
        "seed_role": "dev",
        "model_ids": ["tiny/one", "tiny/two"],
        "world_seed": 7,
        "world": {
            "n_persons": 2, "n_orgs": 2, "n_cities": 2, "n_regions": 1, "nights": 1,
            "supersession_rate": 0.0, "n_distractors": 8, "n_control_persons": 1, "n_entrench": 0,
        },
        "batch_size": 8,
        "max_probes": 3,
        "max_new_tokens": 3,
        "k": 2,
        "chd_tokens": 2,
    }


FACTS = [{"question": "Is it?", "answer": "Yes", "alt": "No", "claim": "It is {value}."}]


def test_e0_writes_a_verdict_with_every_criterion(tmp_path):
    result = run_e0.run_e0(e0_config(), tmp_path, factory, "cpu", FACTS)
    stored = json.loads((tmp_path / "e0_verdict.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(result))
    assert set(stored["candidates"]) == {"tiny/one", "tiny/two"}
    entry = stored["candidates"]["tiny/one"]
    assert set(entry["h0"]) == {"h0_direct", "h0_reverse", "h0_two_hop", "h0_contradiction"}
    assert set(entry["h1"]) == {"h1_episode_auroc", "h1_lower_offset", "h1_known_auroc"}
    for criterion in list(entry["h0"].values()) + list(entry["h1"].values()):
        assert isinstance(criterion["passed"], bool)
    assert "passed" in entry["names_unknown"]


def test_e0_skips_finished_candidates(tmp_path):
    run_e0.run_e0(e0_config(), tmp_path, factory, "cpu", FACTS)

    def must_not_load(model_id, fp16):
        raise AssertionError("a finished candidate ran again")

    run_e0.run_e0(e0_config(), tmp_path, must_not_load, "cpu", FACTS)


def _candidate(h0, h1=True, names=True):
    return {"h0_passed": h0, "h1_passed": h1, "names_unknown": {"passed": names}}


def test_verdict_chooses_the_smallest_passing_model():
    order = ["small", "medium", "large"]
    result = run_e0.verdict({"small": _candidate(False), "medium": _candidate(True), "large": _candidate(True)}, order)
    assert result["chosen_model"] == "medium" and result["h1_action"] is None
    result = run_e0.verdict({"small": _candidate(True, h1=False)}, order)
    assert result["chosen_model"] == "small" and "control layer" in result["h1_action"]
    result = run_e0.verdict({m: _candidate(False) for m in order}, order)
    assert result["chosen_model"] is None and "stop" in result["action"]
    result = run_e0.verdict({"small": _candidate(True, names=False), "medium": _candidate(False)}, order)
    assert result["chosen_model"] is None and result["not_run"] == ["large"]


# ---- the decision rule, on fabricated records ------------------------------------------------------------


def clusters(rate, n=40):
    """n clusters, with the first round(rate * n) right."""
    hits = round(rate * n)
    return {f"c{i}": [1.0 if i < hits else 0.0, 1.0] for i in range(n)}


def record(arm, seed, night, count_source="chd", role="confirmatory", **rates):
    metrics = {"per_cluster": {}}
    for name, rate in rates.items():
        metrics[name] = rate
        metrics["per_cluster"][analyze.CLUSTER_KEY.get(name, name)] = clusters(rate)
    return {
        "arm": arm,
        "world_seed": seed,
        "night": night,
        "model_id": "m",
        "variant": "default",
        "seed_role": role,
        "metrics": metrics,
        "report": {"selection": {"count_source": count_source}},
    }


def group_of(rows):
    return analyze.Group(rows)


def _h3_rows(seeds=5, full=0.6, noselect=0.4, **kwargs):
    rows = []
    for seed in range(seeds):
        rows.append(record("S_full", seed, 0, acq_implied=full))
        rows.append(record("S_noselect", seed, 0, acq_implied=noselect, **kwargs))
    return rows


def test_clear_win_with_five_seeds_is_confirmatory_and_supported():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows()), n_boot=200)["H3"]
    assert result["status"] == "supported" and result["label"] == "confirmatory"
    assert result["result"]["estimate"] == pytest.approx(0.2, abs=1e-6)


def test_fewer_than_five_seeds_is_always_exploratory():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(seeds=3)), n_boot=200)["H3"]
    assert result["status"] == "supported" and result["label"] == "exploratory"
    assert any("seeds" in reason for reason in result["reasons"])


def test_fraction_counts_are_refused_as_confirmatory():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(count_source="fraction")), n_boot=200)["H3"]
    assert result["label"] == "exploratory"
    assert any("fractions" in reason for reason in result["reasons"])


def test_pilot_seeds_are_refused_as_confirmatory():
    rows = _h3_rows()
    for row in rows:
        row["seed_role"] = "dev"
    assert analyze.evaluate_hypotheses(group_of(rows), n_boot=100)["H3"]["label"] == "exploratory"


def test_below_margin_is_not_supported():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(full=0.5, noselect=0.475)), n_boot=200)["H3"]
    assert result["status"] == "not supported"


def test_h2_needs_retention_not_worse():
    rows = []
    for seed in range(5):
        rows.append(record("S_full", seed, 0, acq_implied=0.6, ret_real=0.5))
        rows.append(record("B4_replay_kl", seed, 0, acq_implied=0.4, ret_real=0.9))
    assert analyze.h2(group_of(rows), 100, 0)["status"] == "not supported"
    rows = []
    for seed in range(5):
        rows.append(record("S_full", seed, 0, acq_implied=0.6, ret_real=0.9))
        rows.append(record("B4_replay_kl", seed, 0, acq_implied=0.4, ret_real=0.9))
    assert analyze.h2(group_of(rows), 100, 0)["status"] == "supported"


def test_h4_needs_the_gap_to_widen():
    def rows_for(widening: bool):
        rows = []
        for seed in range(5):
            for night in range(10):
                chain = 0.2 + 0.05 * night if widening else 0.4 - 0.01 * night  # Narrowing, when not widening.
                rows.append(record("S_full", seed, night, drift_generic=0.1, ret_real=0.9))
                rows.append(record("S_chain", seed, night, drift_generic=chain, ret_real=0.5))
        return rows

    assert analyze.h4(group_of(rows_for(True)), 200, 0)["status"] == "supported"
    assert analyze.h4(group_of(rows_for(False)), 200, 0)["status"] == "not supported"


def test_h5_needs_non_inferiority_and_bounded_rank():
    def rows_for(loss, final_rank):
        rows = []
        for seed in range(5):
            for night, rank in ((0, 100), (5, 200), (9, final_rank)):
                common = {"acq_direct": 0.7, "acq_implied": 0.5, "ret_real": 0.9 - loss}
                full = record("S_full", seed, night, **common)
                full["metrics"]["delta_rank"] = rank
                rows.append(full)
                rows.append(record("S_nocompact", seed, night, acq_direct=0.7, acq_implied=0.5, ret_real=0.9))
        return rows

    assert analyze.h5(group_of(rows_for(0.0, 220)), 200, 0)["status"] == "supported"
    assert analyze.h5(group_of(rows_for(0.1, 220)), 200, 0)["status"] == "not supported"
    assert analyze.h5(group_of(rows_for(0.0, 900)), 200, 0)["status"] == "not supported"


def test_h6_is_within_arm_and_needs_both_arms():
    rows = []
    for seed in range(5):
        for arm in ("B3_replay", "S_full"):
            rows.append(record(arm, seed, 9, entrench_follow_fact=0.3, entrench_follow_control=0.8))
    assert analyze.h6(group_of(rows), 200, 0)["status"] == "supported"
    assert analyze.h6(group_of(rows[::2]), 200, 0)["status"] == "missing"  # B3 alone.


def test_h8_thresholds():
    def rows_for(value, false_rate):
        rows = []
        for seed in range(5):
            row = record("S_full", seed, 9)
            row["metrics"].update(audit_auroc=value, false_retirement=false_rate)
            rows.append(row)
        return rows

    assert analyze.h8(group_of(rows_for(0.8, 0.05)))["status"] == "supported"
    assert analyze.h8(group_of(rows_for(0.7, 0.05)))["status"] == "not supported"
    assert analyze.h8(group_of(rows_for(0.8, 0.2)))["status"] == "not supported"


def test_holm_can_remove_support():
    entries = analyze.evaluate_hypotheses(group_of(_h3_rows()), n_boot=200)
    assert entries["H3"]["p_holm"] is not None
    assert entries["H2"]["status"] == "missing" and entries["H9"]["status"] == "not measured"


def test_main_comparison_comes_first_and_analyze_reads_a_real_run(straight_run, capsys):
    root = straight_run
    while root.name != "tiny__gpt2":
        root = root.parent
    summary = analyze.analyze(root.parent, n_boot=50)
    assert list(summary) == ["tiny/gpt2/default"]
    analyze.main(["--results", str(root.parent), "--n-boot", "20"])
    output = capsys.readouterr().out
    assert output.index("B3_replay versus S_full") < output.index("Hypotheses")
    assert "exploratory" in output
    entry = summary["tiny/gpt2/default"]
    assert entry["table"][0]["arm"] == "B3_replay"


def test_a_partial_last_line_does_not_break_the_reader(tmp_path):
    good = record("S_full", 0, 0, acq_implied=0.5)
    path = tmp_path / "x" / "results.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(good) + "\n" + '{"partial": ', encoding="utf-8")
    assert len(analyze.load_records(tmp_path)) == 1
