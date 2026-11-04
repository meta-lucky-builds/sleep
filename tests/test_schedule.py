import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_schedule", ROOT / "scripts" / "schedule.py")
schedule_module = importlib.util.module_from_spec(spec)
sys.modules["gpu_schedule"] = schedule_module
spec.loader.exec_module(schedule_module)
Job = schedule_module.Job


def config(**extra):
    return {
        "arms": ["B0_frozen", "B3_replay", "S_core"],
        "once_arms": ["B0_frozen"],
        "world_seeds": [7],
        "variants": [
            {"name": "big", "night": {"budget": {"night_tokens": 2000}}},
            {"name": "small", "night": {"budget": {"night_tokens": 500}}},
        ],
        **extra,
    }


def test_once_arms_run_for_one_variant_and_first():
    jobs = schedule_module.plan_jobs(config())
    assert jobs[0] == Job("small", "B0_frozen", 7)
    assert sum(job.arm == "B0_frozen" for job in jobs) == 1
    assert len(jobs) == 5


def test_cheapest_budget_comes_before_the_expensive_one():
    jobs = schedule_module.plan_jobs(config())
    variants = [job.variant for job in jobs]
    assert variants.index("small") < variants.index("big")
    assert variants[-1] == "big"


def test_matched_arms_are_refused():
    with pytest.raises(ValueError):
        schedule_module.plan_jobs(config(matched_arms={"S_noselect": "S_full"}))


def test_a_config_without_variants_still_plans():
    jobs = schedule_module.plan_jobs({"arms": ["S_core"], "world_seeds": [1, 2]})
    assert [job.variant for job in jobs] == [None, None]
    assert "--variants" not in schedule_module.job_command(jobs[0], "c.json", "out")


def test_the_command_names_exactly_one_cell():
    command = schedule_module.job_command(Job("small", "S_core", 7), "c.json", "out")
    assert command[command.index("--arms") + 1] == "S_core"
    assert command[command.index("--seeds") + 1] == "7"
    assert command[command.index("--variants") + 1] == "small"


def test_the_job_sees_only_its_gpu():
    env = schedule_module.job_env("3", 4)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["OMP_NUM_THREADS"] == "4"


def _fake_launcher(fail, seen, live):
    """Real short-lived processes. `live` records how many ran at once on each slot."""

    def launch(job, slot, logs):
        seen.append((job.name, slot))
        code = 1 if fail(job, seen) else 0
        return subprocess.Popen([sys.executable, "-c", f"import time, sys; time.sleep(0.05); sys.exit({code})"])

    return launch


def test_every_job_runs_once_when_nothing_fails(tmp_path):
    jobs = schedule_module.plan_jobs(config())
    seen = []
    failed = schedule_module.schedule(jobs, ["0", "0", "1"], _fake_launcher(lambda j, s: False, seen, {}), tmp_path, poll=0.01)
    assert failed == []
    assert sorted(name for name, _ in seen) == sorted(job.name for job in jobs)


def test_the_first_job_runs_alone_and_its_failure_stops_everything(tmp_path):
    jobs = schedule_module.plan_jobs(config())
    seen = []
    failed = schedule_module.schedule(jobs, ["0", "1"], _fake_launcher(lambda j, s: True, seen, {}), tmp_path, poll=0.01)
    assert len(seen) == 1
    assert failed == jobs


def test_a_failure_is_retried_once_on_distinct_gpus(tmp_path):
    jobs = schedule_module.plan_jobs(config())
    target = jobs[2]
    tries = []

    def fail(job, seen):
        if job == target:
            tries.append(job)
            return len(tries) == 1
        return False

    seen = []
    failed = schedule_module.schedule(jobs, ["0", "0", "1", "1"], _fake_launcher(fail, seen, {}), tmp_path, poll=0.01)
    assert failed == []
    assert len(tries) == 2


def test_a_job_that_keeps_failing_is_reported(tmp_path):
    jobs = schedule_module.plan_jobs(config())
    target = jobs[2]
    seen = []
    failed = schedule_module.schedule(jobs, ["0", "1"], _fake_launcher(lambda j, s: j == target, seen, {}), tmp_path, poll=0.01)
    assert failed == [target]


def test_two_cells_may_share_a_slot_id_without_being_confused(tmp_path):
    jobs = [Job(None, f"arm{i}", 1) for i in range(6)]
    seen = []
    failed = schedule_module.run_pass(jobs, ["0", "0"], _fake_launcher(lambda j, s: False, seen, {}), tmp_path, poll=0.01)
    assert failed == []
    assert len(seen) == 6


def test_e1_config_is_schedulable_and_has_the_core_arm():
    import json

    e1 = json.loads((ROOT / "configs" / "e1.json").read_text(encoding="utf-8"))
    jobs = schedule_module.plan_jobs(e1)
    arms = {job.arm for job in jobs}
    assert "S_core" in arms and "S_full" not in arms
    assert sum(job.arm == "B0_frozen" for job in jobs) == 1


def test_a_cell_goes_to_the_gpu_with_the_fewest_running_cells():
    assert schedule_module.pick_slot(["0", "0", "1"], ["0", "0", "0"]) == 2
    assert schedule_module.pick_slot(["3", "1", "2"], ["3", "1", "1"]) == 2  # Whatever order slots came back in.
    assert schedule_module.pick_slot(["1", "0"], []) == 0  # A tie takes the earliest.


def test_cells_spread_over_gpus_even_when_the_slots_are_grouped(tmp_path):
    jobs = [Job(None, f"arm{i}", 1) for i in range(2)]
    seen = []
    failed = schedule_module.run_pass(jobs, ["0", "0", "1", "1"], _fake_launcher(lambda j, s: False, seen, {}), tmp_path, poll=0.01)
    assert failed == []
    assert {slot for _, slot in seen} == {"0", "1"}


def test_load_stays_level_as_slots_free_up_in_any_order():
    """Fill eight GPUs, free some on one GPU, and refill: the new cells go back where the load is lowest."""
    free = ["0", "1", "0", "1", "2", "2"]
    busy: list[str] = []
    for _ in range(3):
        busy.append(free.pop(schedule_module.pick_slot(free, busy)))
    assert sorted(busy) == ["0", "1", "2"]
    busy.remove("1")
    free.append("1")
    busy.append(free.pop(schedule_module.pick_slot(free, busy)))
    assert sorted(busy) == ["0", "1", "2"]


def _gpu_run() -> str:
    return (ROOT / "scripts" / "gpu_run.sh").read_text(encoding="utf-8")


def test_slots_are_round_robin():
    text = _gpu_run()
    outer, inner = text.index('for _ in $(seq 1 "$PER_GPU"); do\n  for gpu in'), text.index('ALL_SLOTS="$ALL_SLOTS,$gpu"')
    assert outer < inner


def test_torch_must_really_work_on_the_gpu_before_the_smoke_test_with_a_fallback_wheel():
    text = _gpu_run()
    check, fallback, again = text.index("if ! cuda_works; then"), text.index('"torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130'), text.index("cuda_works || stop")
    assert check < fallback < again < text.index('say "GPU smoke test"')
    assert "torch.cuda.is_available()" in text and "x @ x" in text and 'device="cuda:0"' in text


def test_e0b_is_guarded_by_its_committed_result_and_then_every_slot_goes_to_the_stage():
    text = _gpu_run()
    assert 'E0B_RESULT="results/e0b/e0b_confirm.json"' in text
    guard = text.index('git cat-file -e "HEAD:$E0B_RESULT"')
    start = text.index('run_e0b "$LAST" &')
    reserve = text.index('SLOTS="${SLOTS%,"$LAST"}"')
    otherwise = text.index("else", guard)
    assert guard < otherwise < start < reserve  # Only the branch that runs E0b starts it, or gives up a slot.
    assert text.count('run_e0b "$LAST" &') == 1 and text.count('SLOTS="${SLOTS%,') == 1
    assert 'if [ -n "$E0B_PID" ]' in text  # An empty pid must not reach `wait`.


def test_the_stage_comes_from_the_environment_and_is_printed():
    text = _gpu_run()
    assert 'STAGE="${STAGE:-e1b+e3x}"' in text
    assert re.search(r'say "Stage \$STAGE, \$GPUS GPUs', text)


def test_the_chain_goes_through_pick_setting_and_stages_can_be_skipped():
    text = _gpu_run()
    first, pick, last = text.index("run_stage e1b"), text.index("scripts/pick_setting.py"), text.index("run_stage e3x")
    assert first < pick < last
    assert 'touch "runs/.done-$name"' in text and 'if [ -f "runs/.done-$name" ]' in text
    assert 'PICKED=0' in text and 'if [ "$PICKED" -eq 1 ]' in text


def test_defaults_are_computed_and_the_environment_wins():
    text = _gpu_run()
    assert 'if [ -n "${PER_GPU:-}" ]' in text and 'if [ -n "${THREADS:-}" ]' in text
    assert "memory.total --format=csv,noheader,nounits | sort -n | head -1" in text
    assert "THREADS=$((CPUS / TOTAL_SLOTS))" in text


def test_archives_and_the_gpu_log_are_written_in_the_background():
    text = _gpu_run()
    assert "write_archive partial" in text and "write_archive results" in text
    assert "-not -path '*/checkpoints/*' -not -path '*/_cache/*'" in text
    assert "index,utilization.gpu,memory.used,memory.total --format=csv,noheader" in text
    assert "gpu_util.log" in text and "'*.log'" in text  # The archive's pattern takes it in.


def test_e1b_is_the_eighty_cells_on_e1s_world():
    e1 = json.loads((ROOT / "configs" / "e1.json").read_text(encoding="utf-8"))
    e1b = json.loads((ROOT / "configs" / "e1b.json").read_text(encoding="utf-8"))
    assert e1b["world"] == e1["world"] and e1b["world_seeds"] == e1["world_seeds"] and e1b["model_ids"] == e1["model_ids"]
    assert e1b["night"]["commit"] == e1["night"]["commit"]
    assert e1b["eval"]["max_per_kind"] == 60 and e1b["capability"]["n_keep"] == 150
    jobs = schedule_module.plan_jobs(e1b)
    assert len(jobs) == 80 and not {"B0_frozen", "B1_rag"} & {job.arm for job in jobs}
    settings = {
        (v["night"]["train"]["lr"], v["night"]["budget"]["night_tokens"]) for v in e1b["variants"]
    }
    assert settings == {(lr, tokens) for lr in (1e-5, 3e-5, 1e-4) for tokens in (50000, 150000)}
    assert sum(e1b["night"]["queries"]["synthesis_shares"].values()) == pytest.approx(1.0)


def test_run_accepts_a_variant_filter():
    sys.path.insert(0, str(ROOT / "experiments"))
    import run

    cfg = {
        "model_ids": ["m"],
        "world_seeds": [1],
        "arms": ["a"],
        "variants": [{"name": "x"}, {"name": "y"}],
    }
    called = []
    original = run.run_cell
    run.run_cell = lambda config, cell, *args, **kwargs: called.append(cell.variant) or Path(".")
    try:
        run.run_experiment(cfg, Path("."), variants=["y"])
    finally:
        run.run_cell = original
    assert called == ["y"]
