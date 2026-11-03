# sleep

A study of "sleep" for language models: can a measured signal control offline consolidation of new facts into
a small model's weights better than a schedule, over many cycles? The main competitor is plain uniform replay
plus a KL penalty, and it may well win.

**Status: no experiment has been run, and there are no results here.** The only planned execution so far is CI's
unit tests, which use tiny random models and check the machinery, never learning quality.

- [docs/THEORY.md](docs/THEORY.md): the idea and what could be wrong with it.
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

Install on a GPU machine (Colab T4 is the target), in order, stopping at any kill criterion in PREREG:

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
