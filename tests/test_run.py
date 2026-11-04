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
        for name in ("acq_direct", "taught_value_bleed", "drift_generic", "delta_norm", "delta_rank", "style_length", "ret_real_n"):
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


def test_a_cell_keeps_one_checkpoint_by_default(tmp_path):
    directory = run_cell(smoke_config(nights=3), tmp_path)
    names = sorted(p.name for p in (directory / "checkpoints").iterdir() if p.is_dir())
    assert names == ["night-00003"]


def test_each_record_says_which_packages_made_it(straight_run):
    for row in runner.read_results(straight_run / "results.jsonl"):
        assert row["versions"]["python"] and row["versions"]["torch"] and row["versions"]["transformers"]


def test_chat_follows_the_model_not_the_config(tmp_path):
    config = smoke_config()
    base = runner.Cell("Qwen/Qwen2.5-0.5B", "default", "B3_replay", 3)
    instruct = runner.Cell("Qwen/Qwen2.5-0.5B-Instruct", "default", "B3_replay", 3)
    assert runner.build_cell_config(config, base, tmp_path)[1].model.chat is False
    assert runner.build_cell_config(config, instruct, tmp_path)[1].model.chat is True


def test_weights_are_fp16_only_where_autocast_runs(tmp_path):
    seen = []

    def recording(model_id, fp16):
        seen.append(fp16)
        return factory(model_id, fp16)

    config = smoke_config()
    config["night"]["model"]["fp16"] = True
    cell = only_cell(config)
    night_cfg = NightConfig.from_dict({"arm": cell.arm, **config["night"]})
    world = runner.SyntheticWorld(WorldConfig(**config["world"]), cell.seed)
    runner._prepare_state(night_cfg, cell, world, recording, "cpu")
    assert seen == [False]


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
    assert {p.stem for p in ALL_CONFIGS} == {f"e{i}" for i in range(7)} | {"e0b", "e1b", "e3x.template"}


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_config_is_well_formed(path):
    config = _load(path)
    if path.stem in ("e0", "e0b"):
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
        "h1_probes": 3,
        "max_new_tokens": 3,
        "k": 2,
        "chd_tokens": 2,
        "deployed": {
            "k": 2, "chd_tokens": 2, "episodes": 2, "per_episode": 2, "model_samples": 1,
            "model_max_new_tokens": 4, "temperature": 0.8,
        },
    }


FACTS = [{"type": "place", "question": "Is it?", "answer": "Yes", "alt": "No", "claim": "It is {value}."}]


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
    assert "rejected" in entry["names_unknown"] and "by_type" in entry["h1"]["h1_known_auroc"]
    assert {"h1_episode_auroc", "h1_lower_offset", "parse_rate"} <= set(entry["h1_deployed"])
    assert entry["family"] == "base" and entry["prompting"] == "raw completion"
    assert entry["tokens"] > 0 and entry["seconds"] >= 0


def test_e0_skips_finished_candidates(tmp_path):
    run_e0.run_e0(e0_config(), tmp_path, factory, "cpu", FACTS)

    def must_not_load(model_id, fp16):
        raise AssertionError("a finished candidate ran again")

    run_e0.run_e0(e0_config(), tmp_path, must_not_load, "cpu", FACTS)


def _candidate(h0, h1=True, deployed=True, replace=()):
    return {"h0_passed": h0, "h1_passed": h1, "h1_deployed_passed": deployed, "names_unknown": {"replace": list(replace)}}


def test_verdict_chooses_the_smallest_passing_model():
    order = ["small", "medium", "large"]
    result = run_e0.verdict({"small": _candidate(False), "medium": _candidate(True), "large": _candidate(True)}, order)
    assert result["chosen_model"] == "medium" and result["h1_action"] is None
    result = run_e0.verdict({"small": _candidate(True, h1=False)}, order)
    assert result["chosen_model"] == "small" and "control layer" in result["h1_action"]
    result = run_e0.verdict({"small": _candidate(True, deployed=False)}, order)
    assert "control layer" in result["h1_action"] and result["h1_passed_on_chosen"] is True
    result = run_e0.verdict({m: _candidate(False) for m in order}, order)
    assert result["chosen_model"] is None and "stop" in result["action"]
    result = run_e0.verdict({"small": _candidate(False), "medium": _candidate(False)}, order)
    assert result["chosen_model"] is None and result["not_run"] == ["large"]


def test_name_familiarity_reports_but_never_gates():
    """A1.10, and review H3: a lucky guess must not silently move the study to a bigger model."""
    result = run_e0.verdict({"small": _candidate(True, replace=["Dara Quilmont"])}, ["small"])
    assert result["chosen_model"] == "small" and result["names_to_replace"] == ["Dara Quilmont"]


def test_verdict_reports_each_family_separately():
    order = ["m-0.5B", "m-0.5B-Instruct", "m-1.5B", "m-1.5B-Instruct"]
    candidates = {
        "m-0.5B": _candidate(False),
        "m-0.5B-Instruct": _candidate(True),
        "m-1.5B": _candidate(True),
        "m-1.5B-Instruct": _candidate(True),
    }
    result = run_e0.verdict(candidates, order)
    assert result["chosen_model"] == "m-0.5B-Instruct"
    assert result["chosen_by_family"] == {"base": "m-1.5B", "instruct": "m-0.5B-Instruct"}


def test_known_versus_fictional_compares_within_a_type():
    facts = run_e0.load_real_facts()
    assert len(facts) >= 30
    assert {f["type"] for f in facts} <= set(run_e0.ATTR_TYPE.values())
    assert {f["type"] for f in facts} == {"place", "thing", "person", "number"}


# ---- smoke ----------------------------------------------------------------------------------------------


def test_tokens_after_answer_line():
    assert run_e0.tokens_after_answer_line([5, 6, 7], {9}) == 0
    assert run_e0.tokens_after_answer_line([5, 9, 7, 8], {9}) == 2
    assert run_e0.tokens_after_answer_line([5, 9], {9}) == 0


def test_newline_tokens_are_found():
    tok = TinyTokenizer()
    found = run_e0.newline_token_ids(tok)
    assert tok("\n", add_special_tokens=False)["input_ids"][0] in found and len(found) == 1


def test_smoke_runs_every_check_on_cpu_and_writes_its_json(tmp_path):
    report = run_e0.run_smoke("tiny/gpt2", e0_config(), tmp_path, factory, "cpu")
    stored = json.loads((tmp_path / "smoke_tiny__gpt2.json").read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(report))
    checks = report["checks"]
    assert set(checks) == {
        "logprob_match", "logits_finite", "raw_completions", "answer_stop", "question_parse", "training_step",
    }
    for name in ("logprob_match", "logits_finite", "raw_completions", "answer_stop", "training_step"):
        assert checks[name]["passed"] is True, (name, checks[name])
    assert report["fp16"] is False and len(report["completions"]) == 20
    assert report["question_sets"] and "parse_rate" in checks["question_parse"]
    assert "mean_tokens_after_answer_line" in checks["raw_completions"]
    assert checks["training_step"]["fresh_moved"] and checks["training_step"]["base_unchanged"]


def test_a_crash_in_one_check_is_recorded_and_the_rest_go_on():
    checks: dict = {}

    def broken():
        raise ValueError("no")

    run_e0._check(checks, "bad", broken)
    run_e0._check(checks, "good", lambda: {"passed": True})
    assert checks["bad"]["passed"] is False and "no" in checks["bad"]["error"] and checks["good"]["passed"]


def test_e0_workflow_is_manual_and_bounded():
    """Triggered by hand or by a tag someone pushes on purpose, never by an ordinary push."""
    text = (ROOT / ".github" / "workflows" / "e0.yml").read_text(encoding="utf-8")
    triggers = text.split("permissions:")[0]
    assert "workflow_dispatch" in triggers and "pull_request" not in triggers
    assert "tags:" in triggers and "branches" not in triggers
    assert "timeout-minutes: 350" in text and "HF_HUB_DISABLE_TELEMETRY" in text and "secrets." not in text
    assert "actions/cache" in text and "upload-artifact" in text and "whl/cpu" in text


# ---- the decision rule, on fabricated records ------------------------------------------------------------


def clusters(rate, n=40):
    """n clusters, with the first round(rate * n) right."""
    hits = round(rate * n)
    return {f"c{i}": [1.0 if i < hits else 0.0, 1.0] for i in range(n)}


def record(arm, seed, night, count_source="chd", role="confirmatory", **rates):
    """Each metric comes with a lift equal to itself: a zero chance control, unless a test says otherwise."""
    metrics = {"per_cluster": {}}
    for name, rate in rates.items():
        metrics[name] = rate
        metrics["per_cluster"][name] = clusters(rate)
        metrics["per_cluster"][f"{name}_lift"] = clusters(rate)
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
    result = analyze.evaluate_hypotheses(group_of(_h3_rows()), n_boot=400)["H3"]
    assert result["status"] == "supported" and result["label"] == "confirmatory"
    assert result["result"]["estimate"] == pytest.approx(0.2, abs=1e-6)


def test_margin_applies_to_lift_not_to_raw_accuracy():
    """A1.3: a 20-point raw gap that the chance control explains is no win."""
    rows = _h3_rows()
    for row in rows:
        row["metrics"]["per_cluster"]["acq_implied_lift"] = clusters(0.1)
    result = analyze.evaluate_hypotheses(group_of(rows), n_boot=400)["H3"]
    assert result["status"] == "not supported" and result["result"]["estimate"] == 0.0


def test_fewer_than_five_seeds_is_always_exploratory():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(seeds=3)), n_boot=400)["H3"]
    assert result["status"] == "supported" and result["label"] == "exploratory"
    assert any("seeds" in reason for reason in result["reasons"])


def test_fraction_counts_are_refused_as_confirmatory():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(count_source="fraction")), n_boot=400)["H3"]
    assert result["label"] == "exploratory"
    assert any("fractions" in reason for reason in result["reasons"])


def test_pilot_seeds_are_refused_as_confirmatory():
    rows = _h3_rows()
    for row in rows:
        row["seed_role"] = "dev"
    assert analyze.evaluate_hypotheses(group_of(rows), n_boot=100)["H3"]["label"] == "exploratory"


def test_below_margin_is_not_supported():
    result = analyze.evaluate_hypotheses(group_of(_h3_rows(full=0.5, noselect=0.475)), n_boot=400)["H3"]
    assert result["status"] == "not supported"


# ---- A1.7: validity ----------------------------------------------------------------------------------------


def _label(rows):
    return analyze.run_label(group_of(rows), ("S_full", "S_noselect"), 5)


def test_unequal_nights_force_exploratory():
    rows = _h3_rows()
    rows.append(record("S_full", 0, 9, acq_implied=0.6))
    label, reasons = _label(rows)
    assert label == "exploratory" and any("unequal" in reason for reason in reasons)


def test_differing_world_fingerprints_force_exploratory():
    rows = _h3_rows()
    assert _label(rows)[0] == "confirmatory"
    rows[0]["world_fingerprint"] = "a"
    rows[1]["world_fingerprint"] = "b"
    label, reasons = _label(rows)
    assert label == "exploratory" and any("fingerprints" in reason for reason in reasons)


def test_low_acceptance_forces_exploratory():
    rows = _h3_rows()
    for index, row in enumerate(r for r in rows if r["arm"] == "S_full"):
        row["decision"] = {"accepted": index == 0}
    label, reasons = _label(rows)
    assert label == "exploratory" and any("acceptance" in reason for reason in reasons)
    assert analyze.validity_row(group_of(rows), "S_full")["acceptance"] == pytest.approx(0.2)


def test_high_unscored_fraction_forces_exploratory():
    rows = _h3_rows()
    for row in rows:
        if row["arm"] == "S_full":
            row["report"]["selection"]["train"] = {"integrate": 3, "anchor": 1, "discarded": {"unscored": 6}}
            row["report"]["selection"]["unscored_fraction"] = 0.6
            row["report"]["signals"] = {"gate_rate": 0.1}
            row["report"]["synthesis"] = {"pairs": {"chains_chosen": 4}}
    label, reasons = _label(rows)
    assert label == "exploratory" and any("unscored" in reason for reason in reasons)
    row = analyze.validity_row(group_of(rows), "S_full")
    assert row["unscored"] == pytest.approx(0.6) and row["integrate_per_night"] == pytest.approx(3.0)
    assert row["gate_rate"] == pytest.approx(0.1) and row["chains_per_night"] == pytest.approx(4.0)


def test_anchor_drift_distribution_is_tabulated_for_calibration():
    rows = []
    for night, drift in enumerate((0.1, 0.2, 0.3, None)):
        row = record("S_full", 0, night)
        row["decision"] = {"accepted": True, "anchor_drift": drift, "control_drop": 0.5}
        rows.append(row)
    table = analyze.anchor_drift_table(group_of(rows))
    assert table == [{"arm": "S_full", "nights": 3, "min": 0.1, "median": 0.2, "p90": 0.3, "max": 0.3}]
    assert analyze.validity_row(group_of(rows), "S_full")["control_drop"] == pytest.approx(0.5)


def test_validity_table_reads_the_ledger_and_tolerates_frozen_arms():
    frozen = record("B0_frozen", 0, 0, acq_direct=0.1)
    frozen["decision"] = {}
    frozen["ledger"] = {"allowance": 1000, "spent": 0}
    full = record("S_full", 0, 0, acq_direct=0.5)
    full["decision"] = {"accepted": True}
    full["ledger"] = {"allowance": 1000, "spent": 900}
    group = group_of([frozen, full])
    assert analyze.validity_row(group, "B0_frozen")["acceptance"] is None
    assert analyze.validity_row(group, "S_full")["spent_fraction"] == pytest.approx(0.9)


# ---- hypotheses --------------------------------------------------------------------------------------------


def _h2_rows(reverse=0.6, alias=0.6, twohop=0.6, retention=0.9):
    rows = []
    for seed in range(5):
        rows.append(record("S_nopair", seed, 0, acq_reverse=reverse, acq_alias=alias, ret_real=retention))
        rows.append(record("B4_replay_kl", seed, 0, acq_reverse=0.4, acq_alias=0.4, ret_real=0.9))
        rows.append(record("S_core", seed, 0, acq_twohop=twohop, ret_real=0.9))
        rows.append(record("B3_pairs", seed, 0, acq_twohop=0.4, ret_real=0.9))
    return rows


def test_h2_is_three_tests_on_the_amended_arms():
    """A1.2: the teacher is judged without pairs, and two-hop against the pair baseline."""
    group = group_of(_h2_rows())
    reverse = analyze.h2_member(group, "acq_reverse", "S_nopair", "B4_replay_kl", 200, 0)
    alias = analyze.h2_member(group, "acq_alias", "S_nopair", "B4_replay_kl", 200, 0)
    twohop = analyze.h2_member(group, "acq_twohop", "S_core", "B3_pairs", 200, 0)
    assert (reverse["status"], alias["status"], twohop["status"]) == ("supported",) * 3
    entries = analyze.evaluate_hypotheses(group, n_boot=1000)
    assert entries["H2_reverse"]["arms"] == ["S_nopair", "B4_replay_kl"]
    assert entries["H2_twohop"]["arms"] == ["S_core", "B3_pairs"]
    assert entries["H2"]["status"] == "supported"


def test_h2_as_a_whole_needs_all_three_members():
    entries = analyze.evaluate_hypotheses(group_of(_h2_rows(alias=0.4)), n_boot=1000)
    assert entries["H2_alias"]["status"] == "not supported"
    assert entries["H2_reverse"]["status"] == "supported" and entries["H2"]["status"] == "not supported"


def test_h2_needs_retention_not_worse():
    group = group_of(_h2_rows(retention=0.5))
    assert analyze.h2_member(group, "acq_reverse", "S_nopair", "B4_replay_kl", 100, 0)["status"] == "not supported"


def test_h7_uses_twohop():
    rows = []
    for seed in range(5):
        rows.append(record("S_core", seed, 0, acq_twohop=0.6))
        rows.append(record("S_nopair", seed, 0, acq_twohop=0.2))
    assert analyze.evaluate_hypotheses(group_of(rows), n_boot=400)["H7"]["metric"] == "acq_twohop"


def test_missing_hypotheses_count_as_p_one_in_holm():
    """A1.7, review M10: a partial run must not look stronger through a smaller family."""
    entries = analyze.evaluate_hypotheses(group_of(_h3_rows()), n_boot=400)
    assert entries["H3"]["p_holm"] == pytest.approx(entries["H3"]["p"] * len(analyze.HOLM_FAMILY))
    assert entries["H5"]["status"] == "missing" and entries["H5"]["p_holm"] == 1.0


def test_holm_can_remove_support():
    entries = analyze.evaluate_hypotheses(group_of(_h3_rows()), n_boot=100)  # p = 0.01, times eight members.
    assert entries["H3"]["status"] == "not supported"
    assert any("Holm" in reason for reason in entries["H3"]["reasons"])
    assert entries["H2"]["status"] == "missing" and entries["H9"]["status"] == "not measured"


def test_h4_needs_the_gap_to_widen():
    def rows_for(widening: bool):
        rows = []
        for seed in range(5):
            for night in range(10):
                chain = 0.2 + 0.05 * night if widening else 0.4 - 0.01 * night  # Narrowing, when not widening.
                rows.append(record("S_core", seed, night, drift_generic=0.1, ret_real=0.9))
                rows.append(record("S_chain", seed, night, drift_generic=chain, ret_real=0.5))
        return rows

    assert analyze.h4(group_of(rows_for(True)), 200, 0)["status"] == "supported"
    assert analyze.h4(group_of(rows_for(False)), 200, 0)["status"] == "not supported"


def test_h5_needs_non_inferiority_and_bounded_rank():
    def rows_for(loss, final_rank):
        rows = []
        accuracy = {"acq_direct": 0.7, "acq_reverse": 0.5, "acq_alias": 0.5, "acq_twohop": 0.5}
        for seed in range(5):
            for night, rank in ((0, 100), (5, 200), (9, final_rank)):
                full = record("S_core", seed, night, ret_real=0.9 - loss, **accuracy)
                full["metrics"]["delta_rank"] = rank
                rows.append(full)
                rows.append(record("S_nocompact", seed, night, ret_real=0.9, **accuracy))
        return rows

    assert analyze.h5(group_of(rows_for(0.0, 220)), 200, 0)["status"] == "supported"
    assert analyze.h5(group_of(rows_for(0.1, 220)), 200, 0)["status"] == "not supported"
    assert analyze.h5(group_of(rows_for(0.0, 900)), 200, 0)["status"] == "not supported"


def test_h6_is_within_arm_and_needs_both_arms():
    rows = []
    for seed in range(5):
        for arm in ("B3_replay", "S_core"):
            rows.append(record(arm, seed, 9, entrench_follow_fact=0.3, entrench_follow_control=0.8))
    assert analyze.h6(group_of(rows), 200, 0)["status"] == "supported"
    assert analyze.h6(group_of(rows[::2]), 200, 0)["status"] == "missing"  # B3 alone.


def _fact(key, role="tonight", gap_before=2.0, gap_after=1.0, correct_before=False, correct_after=True, heldout=True):
    return {
        "key": key,
        "role": role,
        "attr": "employer",
        "gap_before": gap_before,
        "gap_after": gap_after,
        "heldout_before": gap_before if heldout else None,
        "heldout_after": gap_after if heldout else None,
        "heldout_n": 1 if heldout else 0,
        "correct_before": correct_before,
        "correct_after": correct_after,
    }


def _audit_record(seed, night, items, null=(1.0,), accepted=True, retire_nights=2):
    row = record("S_core", seed, night)
    row["report"]["fact_audit"] = {
        "items": items, "null_gaps": list(null), "accepted": accepted, "retire_nights": retire_nights,
    }
    return row


def test_fact_audit_night_reads_within_fact_change_and_the_controls():
    taught = [
        _fact("gained-1", gap_before=2.0, gap_after=0.5),
        _fact("gained-2", gap_before=3.0, gap_after=1.0),
        _fact("missed-1", gap_before=2.0, gap_after=1.9, correct_after=False),
        _fact("missed-2", gap_before=3.0, gap_after=3.1, correct_after=False),
        _fact("knew", gap_before=0.1, gap_after=0.1, correct_before=True),
    ]
    # Nothing taught these, and high ones fall anyway: regression to the mean, with slope -0.5.
    controls = [
        _fact(f"control:{i}", "control", gap_before=g, gap_after=0.5 * g + 0.5, correct_after=False, heldout=False)
        for i, g in enumerate((0.0, 1.0, 2.0, 3.0))
    ]
    out = analyze.fact_audit_night(taught + controls)
    assert out["taught"] == 5 and out["controls"] == 4
    assert out["within_fact_auroc"] == pytest.approx(1.0)  # Among the four wrong before, knew is not counted.
    assert out["auroc_probe"] == pytest.approx(1.0)  # Right (0.5, 1.0, 0.1); wrong (1.9, 3.1).
    assert out["gap_vs_correct_change"] < 0  # A larger fall goes with gaining the fact.
    assert out["control_regression_slope"] == pytest.approx(-0.5)
    assert out["control_mean_gap_change"] == pytest.approx(-0.25)
    # Each taught fact against the control line, 0.5 - 0.5 * before: the fall left after regression to the mean.
    expected = sum(
        (r["gap_after"] - r["gap_before"]) - (0.5 - 0.5 * r["gap_before"]) for r in taught
    ) / len(taught)
    assert out["taught_net_of_control"] == pytest.approx(expected)


def test_fact_audit_night_is_none_where_it_has_nothing():
    out = analyze.fact_audit_night([_fact("a", correct_before=True)])
    assert out["within_fact_auroc"] is None and out["taught_net_of_control"] is None
    assert out["control_regression_slope"] is None and out["auroc_probe"] is None


def test_would_retire_counts_accepted_nights_in_a_row():
    def night(number, gap, accepted=True, correct=True):
        return _audit_record(0, number, [_fact("a", "older", gap_after=gap, correct_after=correct)], accepted=accepted)

    assert analyze.would_retire([night(0, 0.5)]) == {}  # One low night of two.
    assert analyze.would_retire([night(0, 0.5), night(1, 0.5)]) == {"a": False}
    assert analyze.would_retire([night(0, 0.5), night(1, 0.5, correct=False)]) == {"a": True}
    assert analyze.would_retire([night(0, 0.5), night(1, 5.0), night(2, 0.5)]) == {}  # A high gap resets.
    # A rejected night neither counts nor resets.
    assert analyze.would_retire([night(0, 0.5), night(1, 5.0, accepted=False), night(2, 0.5)]) == {"a": False}
    # The verdict is at the final night, not at retirement.
    assert analyze.would_retire([night(0, 0.5), night(1, 0.5), night(2, 0.5, correct=False)]) == {"a": True}


def _h8_rows(separates=True, within=True, wrong_at_end=False, nights=2):
    """Five seeds of ten facts. Five end right with low probe gaps, so they would retire; five stay wrong with high ones.

    Each flag spoils exactly one clause: the held-out gap's AUROC, the within-fact AUROC, or who retirement would take.
    """
    rows = []
    for seed in range(5):
        for number in range(nights):
            items = []
            for i in range(10):
                right = i < 5
                gap_after = 0.5 if right else 1.5
                fall = (1.0 if right else 0.1) if within else (0.1 if right else 1.0)
                spoiled = wrong_at_end and number == nights - 1 and i < 3
                row = _fact(f"f{i}", gap_before=gap_after + fall, gap_after=gap_after, correct_after=right and not spoiled)
                if not separates:
                    row["heldout_after"] = 1.5 if right else 0.5
                items.append(row)
            rows.append(_audit_record(seed, number, items, null=(1.0,)))
    return rows


def test_h8_reads_the_log_on_the_headline_arm():
    good = analyze.h8(group_of(_h8_rows()))
    assert good["status"] == "supported" and good["arms"] == ["S_core"]
    assert good["mean_within_fact_auroc"] == pytest.approx(1.0) and good["counterfactual_retired"] == 25
    assert analyze.h8(group_of(_h8_rows(separates=False)))["status"] == "not supported"
    assert analyze.h8(group_of(_h8_rows(within=False)))["status"] == "not supported"


def test_h8_fails_if_counterfactual_retirement_retires_what_is_wrong():
    assert analyze.h8(group_of(_h8_rows(wrong_at_end=True)))["status"] == "not supported"


def test_h8_is_not_measured_on_a_single_night_but_still_reports():
    """Review M7: with nothing retired its last clause would otherwise pass on no data. E2 has one night."""
    entry = analyze.h8(group_of(_h8_rows(nights=1)))
    assert entry["status"] == "not measured" and entry["mean_auroc_probe"] == pytest.approx(1.0)


def test_h8_is_missing_without_the_log():
    assert analyze.h8(group_of([record("S_core", 0, 0)]))["status"] == "missing"
    # An arm with the old audit metrics no longer stands in.
    row = record("S_full", 0, 0)
    row["metrics"].update(audit_auroc=0.9, false_retirement=0.0, retired=99)
    assert analyze.h8(group_of([row]))["status"] == "missing"


def test_h3_is_withdrawn_and_leaves_the_family_unless_selection_is_restored():
    rows = []
    for seed in range(5):
        rows.append(record("S_core", seed, 0, acq_twohop=0.6))
        rows.append(record("S_nopair", seed, 0, acq_twohop=0.2))
    entries = analyze.evaluate_hypotheses(group_of(rows), n_boot=400)
    assert entries["H3"]["status"] == "withdrawn" and "p_holm" not in entries["H3"]
    assert entries["H7"]["p_holm"] == pytest.approx(entries["H7"]["p"] * (len(analyze.HOLM_FAMILY) - 1))
    assert "H3" in analyze.format_hypothesis("H3", entries["H3"])


def test_a_restored_s_full_is_reported_beside_the_headline_and_replaces_nothing():
    rows = _h3_rows()
    rows += [record("S_core", seed, 0, acq_twohop=0.5) for seed in range(5)]
    rows += [record("B3_pairs", seed, 0, acq_twohop=0.5) for seed in range(5)]
    group = group_of(rows)
    assert analyze.evaluate_hypotheses(group, n_boot=100)["H2_twohop"]["arms"] == ["S_core", "B3_pairs"]
    assert analyze.main_comparison(group, 50, 0)["B3_pairs"]["arm"] == "S_core"
    assert analyze.main_comparison(group, 50, 0, "S_full")["B3_pairs"]["arm"] == "S_full"


def test_fixed_anchor_counts_are_not_refused_as_a_fallback():
    """A fallback fraction marks a matched arm whose source was missing. S_core's fraction is its design."""
    rows = []
    for seed in range(5):
        rows.append(record("S_core", seed, 0, count_source="fixed_fraction", acq_twohop=0.6))
        rows.append(record("B3_pairs", seed, 0, acq_twohop=0.4))
    label, reasons = analyze.run_label(group_of(rows), ("S_core", "B3_pairs"), 5)
    assert label == "confirmatory" and not reasons


def test_a_real_core_run_reaches_the_analysis(tmp_path, capsys):
    config = smoke_config(arms=("S_core",), nights=2, capability=None)
    config["eval"]["drift"] = False
    config["night"]["fact_audit"] = {"panel": 6, "controls": 2, "null_items": 2}
    directory = runner.run_cell(config, only_cell(config), tmp_path, factory, None, "code-1", "cpu")
    first, second = runner.read_results(directory / "results.jsonl")
    audit = second["report"]["fact_audit"]
    assert audit["items"] and audit["tokens"] > 0
    assert {r["role"] for r in audit["items"]} >= {"control"}
    assert first["report"]["selection"]["count_source"] == "fixed_fraction"
    assert "signals" not in first["ledger"]["caps"]
    assert second["ledger"]["exempt_forward"] >= first["eval_tokens"] + first["report"]["fact_audit"]["tokens"]
    summary = analyze.analyze(tmp_path, n_boot=20)
    assert summary["tiny/gpt2/default"]["fact_audit"]
    analyze.main(["--results", str(tmp_path), "--n-boot", "10"])
    assert "H8's log" in capsys.readouterr().out


def test_the_arm_lists_follow_amendment_2():
    for stem in ("e2", "e3", "e5", "e6"):
        arms = _load(CONFIG_DIR / f"{stem}.json")["arms"]
        assert "S_core" in arms and "S_full" not in arms and "S_noselect" not in arms, stem
        assert _load(CONFIG_DIR / f"{stem}.json")["night"]["select"]["fixed_anchor_frac"] == 0.5
    needed = {"S_core", "S_nopair", "S_chain", "S_nocompact", "B3_pairs", "B3_replay", "B4_replay_kl"}
    for stem in ("e2", "e3"):
        assert needed <= set(_load(CONFIG_DIR / f"{stem}.json")["arms"])
    assert "S_core" in _load(CONFIG_DIR / "e4.json")["arms"]
    # Restoring selection needs only the two arm names.
    assert _load(CONFIG_DIR / "e3.json")["matched_arms"] == {"S_noselect": "S_full"}


def test_age_table_pools_seeds():
    rows = []
    for seed, hits in ((0, 3), (1, 1)):
        row = record("S_full", seed, 9)
        row["metrics"]["acq_by_age"] = {"0": [hits, 4], "9": [0, 4]}
        rows.append(row)
    table = analyze.age_table(group_of(rows))
    assert table == [{"arm": "S_full", "age_0": 0.5, "age_9": 0.0}]


def test_headline_is_the_pair_baseline_and_analyze_reads_a_real_run(straight_run, capsys):
    root = straight_run
    while root.name != "tiny__gpt2":
        root = root.parent
    summary = analyze.analyze(root.parent, n_boot=50)
    assert list(summary) == ["tiny/gpt2/default"]
    analyze.main(["--results", str(root.parent), "--n-boot", "20"])
    output = capsys.readouterr().out
    assert output.index("S_core versus B3_pairs") < output.index("S_core versus B3_replay") < output.index("Hypotheses")
    assert "exploratory" in output and "Validity" in output
    entry = summary["tiny/gpt2/default"]
    assert entry["table"][0]["arm"] == "B3_replay"
    assert entry["validity"][0]["acceptance"] == 1.0
    assert entry["by_age"][0]["arm"] == "B3_replay"
    assert "taught_value_bleed" in entry["table"][0] and "false_recall_any_specific" in entry["table"][0]


def test_a_finished_cell_can_drop_its_weights_and_a_rerun_still_skips_it(tmp_path):
    config = smoke_config(drop_finished_weights=True)
    directory = run_cell(config, tmp_path)
    assert not list((directory / "checkpoints").glob("night-*/adapters.pt"))
    assert (directory / "checkpoints" / "LATEST").read_text() == "night-00002"
    results = (directory / "results.jsonl").read_bytes()

    def no_model(model_id, fp16):
        raise AssertionError("a finished cell must not load a model")

    runner.run_cell(config, only_cell(config), tmp_path, no_model, QA, "code-1", "cpu")
    assert (directory / "results.jsonl").read_bytes() == results


def test_weights_stay_without_the_flag_so_that_e3_and_e4_are_unaffected(straight_run):
    assert list((straight_run / "checkpoints").glob("night-*/adapters.pt"))


def test_a_partial_last_line_does_not_break_the_reader(tmp_path):
    good = record("S_full", 0, 0, acq_implied=0.5)
    path = tmp_path / "x" / "results.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(good) + "\n" + '{"partial": ', encoding="utf-8")
    assert len(analyze.load_records(tmp_path)) == 1
    assert len(runner.read_results(path)) == 1  # Review L6: the matched-arm reader too.


def test_a_partial_source_line_means_no_matched_counts(tmp_path):
    config = smoke_config(arms=("S_full", "S_noselect"), matched_arms={"S_noselect": "S_full"})
    cells = {c.arm: c for c in runner.expand_cells(config)}
    source = runner.cell_dir(tmp_path, cells["S_full"])
    source.mkdir(parents=True)
    line = {"night": 0, "report": {"selection": {"train": {"anchor": 0}, "heldout": {"anchor": 0}}}}
    (source / "results.jsonl").write_text(json.dumps(line) + "\n" + '{"nig', encoding="utf-8")
    assert runner.matched_selection(config, cells["S_noselect"], tmp_path, 2) is None
