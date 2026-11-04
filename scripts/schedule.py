#!/usr/bin/env python3
"""Run an experiment's cells across GPU slots, one process per cell, handing the next cell to whichever slot is free.

Standard library only, so it costs no GPU memory. Cells are resumable (`run.py` skips finished nights), so running this
again picks up where it stopped. Arms in the config's `once_arms` train nothing and run for one variant only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    variant: Optional[str]
    arm: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.variant or 'default'}__{self.arm}__seed-{self.seed}"


def _night_tokens(variant: dict) -> float:
    return float(variant.get("night", {}).get("budget", {}).get("night_tokens", 0))


def plan_jobs(config: dict) -> list[Job]:
    """Runs that train nothing come first, then the cheapest budgets, so that results appear early.

    Matched arms need their source's report, which this order does not guarantee, so they are refused.
    """
    if config.get("matched_arms"):
        raise ValueError("schedule.py runs cells independently, and matched arms need their source to finish first")
    variants = sorted(config.get("variants", []), key=_night_tokens)
    names = [v["name"] for v in variants] or [None]
    once = [arm for arm in config["arms"] if arm in set(config.get("once_arms", []))]
    others = [arm for arm in config["arms"] if arm not in once]
    jobs = [Job(names[0], arm, int(seed)) for seed in config["world_seeds"] for arm in once]
    jobs += [Job(variant, arm, int(seed)) for variant in names for seed in config["world_seeds"] for arm in others]
    return jobs


def job_command(job: Job, config_path: str, out: str) -> list[str]:
    command = [
        sys.executable, str(ROOT / "experiments" / "run.py"),
        "--config", config_path, "--out", out, "--arms", job.arm, "--seeds", str(job.seed),
    ]  # fmt: skip
    if job.variant is not None:
        command += ["--variants", job.variant]
    return command


def job_env(slot: str, threads: int) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": slot,
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


Launcher = Callable[[Job, str, Path], "subprocess.Popen"]


def pick_slot(free: Sequence[str], busy: Sequence[str]) -> int:
    """Index in `free` of a slot on the GPU with the fewest running cells; the earliest such slot on a tie."""
    return min(range(len(free)), key=lambda index: (list(busy).count(free[index]), index))


def run_pass(jobs: Sequence[Job], slots: Sequence[str], launch: Launcher, logs: Path, poll: float = 5.0) -> list[Job]:
    """Everything in `jobs`, one per slot at a time. Returns the jobs that failed."""
    pending = list(jobs)
    running: dict[str, tuple[Job, "subprocess.Popen", float]] = {}
    free = list(slots)
    failed: list[Job] = []
    done = 0
    started = time.time()
    while pending or running:
        while pending and free:
            job = pending.pop(0)
            slot = free.pop(pick_slot(free, [name.split("#", 1)[0] for name in running]))
            key = f"{slot}#{job.name}"
            running[key] = (job, launch(job, slot, logs), time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start {job.name} on GPU {slot}", flush=True)
        for key in list(running):
            job, process, began = running[key]
            code = process.poll()
            if code is None:
                continue
            del running[key]
            free.append(key.split("#", 1)[0])
            done += 1
            elapsed = time.time() - started
            left = elapsed / done * (len(pending) + len(running))
            status = "ok" if code == 0 else f"FAILED ({code})"
            print(
                f"[{time.strftime('%H:%M:%S')}] {status} {job.name} in {(time.time() - began) / 60:.1f} min; "
                f"{done}/{len(jobs)} done, roughly {left / 60:.0f} min left",
                flush=True,
            )
            if code != 0:
                failed.append(job)
        if running:
            time.sleep(poll)
    return failed


def make_launcher(config_path: str, out: str, threads: int) -> Launcher:
    def launch(job: Job, slot: str, logs: Path) -> "subprocess.Popen":
        stream = open(logs / f"{job.name}.log", "a", encoding="utf-8")
        return subprocess.Popen(
            job_command(job, config_path, out), stdout=stream, stderr=subprocess.STDOUT, env=job_env(slot, threads), cwd=ROOT
        )

    return launch


def schedule(jobs: Sequence[Job], slots: Sequence[str], launch: Launcher, logs: Path, poll: float = 5.0) -> list[Job]:
    """The first job runs alone, since it fills the shared caches. Failures get one more try, one per GPU, for memory."""
    if not jobs:
        return []
    logs.mkdir(parents=True, exist_ok=True)
    if run_pass(jobs[:1], slots[:1], launch, logs, poll):
        print("the first job failed, so its caches don't exist: stopping. See its log.", flush=True)
        return list(jobs)
    failed = run_pass(jobs[1:], slots, launch, logs, poll)
    if failed:
        print(f"{len(failed)} failed; retrying with one job per GPU", flush=True)
        failed = run_pass(failed, list(dict.fromkeys(slots)), launch, logs, poll)
    return failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--slots", required=True, help="one GPU id per slot: '0,0,1,1' runs two cells on each of two GPUs")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    failed = schedule(plan_jobs(config), slots, make_launcher(args.config, args.out, args.threads), Path(args.logs))
    for job in failed:
        print(f"still failed: {job.name}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
