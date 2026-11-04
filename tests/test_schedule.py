import importlib.util
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
