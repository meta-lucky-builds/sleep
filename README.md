# sleep

A study of "sleep" for language models: can new facts be consolidated offline into a small model's weights, over
many cycles, without destroying what it already knows? It began as a bet that a measured signal could control this
better than a schedule, against plain uniform replay plus a KL penalty. That signal failed its tests (E0, E0b), and
selection was abandoned.

**Status: experiments have run, and all results so far are exploratory; nothing is confirmed yet.** The best result
is E5s (3 seeds, 2 nights, Qwen2.5-0.5B): anchoring on the model's own questions kept 26 points more real knowledge
than replay plus KL while still learning new facts, and composed facts told outright reached reworded questions.
False recall on look-alike names is unsolved. A ten-night confirmatory run is planned, not done.

- [docs/results/E5s.md](docs/results/E5s.md): the latest results and their limits; earlier ones are in the same folder.
- [docs/THEORY.md](docs/THEORY.md): the original idea and what could be wrong with it.
- [docs/PREREG.md](docs/PREREG.md): metrics, hypotheses, falsifiers, the decision rule.
- [docs/DESIGN.md](docs/DESIGN.md): code spec and the invariants the tests enforce.

## Layout

```
src/llmsleep/        delta banks, model handle, store, signals, ledger, arms
src/llmsleep/phases/ the night: queries, selection, consolidation, compaction, commit
src/llmsleep/worlds/ SyntheticWorld, its names and templates; eval_templates.json is fixed by hand
src/llmsleep/eval/   frozen probes and every PREREG metric
src/llmsleep/stats.py paired hierarchical bootstrap, Holm, AUROC
experiments/         run.py, run_e0.py, analyze.py
configs/             one per experiment; assumptions in configs/README.md
tests/               unit tests, no downloads
```

## How to run

Real runs so far used rented GPUs through `scripts/gpu_run.sh`. By hand, on a GPU machine, in order, stopping at any kill criterion in PREREG:

```
pip install -e ".[capability]"
python experiments/run_e0.py --config configs/e0.json --out runs/e0
python experiments/run.py --config configs/e1.json --out runs/e1
python experiments/run.py --config configs/e3.json --out runs/e3
python experiments/analyze.py --results runs/e3
```

`run.py` resumes where a killed session stopped: run the same command with the same `--out`. Pass `--code-hash` when
there is no `.git`. Use `--arms` and `--seeds` to split a long experiment across sessions.

Tests run in CI only: `pytest -q`. E0 can also run on CPU in GitHub Actions: `.github/workflows/e0.yml`, started by hand, with a `smoke` option for the first real-model checks.
