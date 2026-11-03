# Configs

JSON has no comments, so assumptions live here. **Every time and size below is an estimate; nothing has been run.**

## Before running

- Models: E1 onward name `Qwen/Qwen2.5-0.5B` as a placeholder. Replace it with E0's chosen model; E6 with the next size up. Hub IDs and licences of the Qwen models have not been checked; the 3B family may carry a different licence.
- `train.lr` 3e-4 and `budget.night_tokens` are placeholders until E1 picks them on the dev world. Set them in E2, E3, E5, E6 before any confirmatory run, and note the values in PREREG.
- E6's arms are placeholders for "the two best arms".

## Seeds

- 7: dev world (E0, E1). Pilot tuning uses this only.
- 2001-2003: E5 sweep, labelled exploratory.
- 1001-1005: confirmatory worlds. Never used in piloting. A test enforces that the dev seed appears in no confirmatory config.

## Assumptions for a Colab T4

From T2 section 5.2, not measured: 0.5B training at roughly 1 to 2 million token passes per 10 minutes under fp16 with gradient checkpointing. The ledger counts forward and backward separately, so 1M ledger tokens is about 500k token passes in training.

| Config | Size | Rough cost |
|---|---|---|
| e0 | 30 persons, 60 probes per kind, 40 for H1, 6 candidates | under 1 hour per candidate at 3B, less below |
| e1 | 40 persons, 3 nights, 9 variants, 6 arms, 1 seed | 15 to 25 hours in total, so split it with `--arms`; start with `B3_replay` |
| e2 | 100 persons (about 650 facts), one 4M-token night, 15 arms, 5 seeds | 40 to 70 hours |
| e3 | 120 persons (about 800 facts, 72 changes), 10 nights at 1M tokens, 15 arms, 5 seeds | 60 to 100 hours; the first arms in the list are the ones hypotheses need. Run those first |
| e4 | contradiction probe on E3's checkpoints | minutes per cell |
| e5 | 9 soft settings and 3 hard truncations, 3 seeds | 40 to 60 hours; cut by need |
| e6 | 1.5B, smaller batches with accumulation, 2 arms, 5 seeds | about 3x e3 per cell |

A T4 session dies at about 12 hours, so a run spans sessions: same command, same `--out`, on durable storage. Each night evaluates up to 300 probes per kind, 500 real questions and 64 generic prompts, a few minutes per night.

## E0 on CPU in GitHub Actions

`.github/workflows/e0.yml` is manual only and spends the account's Actions minutes. **These are estimates, not measurements**, from T2 section 5.2's rate scaled to a 4-core runner: roughly 50 to 100 tokens per second for 0.5B, 20 to 35 for 1.5B. E0 needs about 175k forward tokens at `configs/e0.json` sizes, assuming the answer stop keeps answers short.

| Model | Smoke | Full E0 |
|---|---|---|
| 0.5B | 5 to 15 minutes | 30 to 60 minutes |
| 1.5B | 15 to 40 minutes | 1.5 to 2.5 hours |
| 3B | about an hour | 3 to 5 hours, close to the 350-minute limit |

Each verdict records `tokens` and `seconds`, so replace these numbers after the first run. A CPU run does not exercise fp16; that needs one GPU `--smoke`.

## Choices to look at

- `max_pairs` is well above the code's default of 16, because the cap decides how many chains ever get a pair context (review M3).
- `n_regions` is close to `n_cities`, and two-hop is reported per chain, so that guessing the type does not clear a margin (review H5).
- `self_samples` is null: `B4` draws until its teacher allocation is spent, in thousands of sequences (A1.11).
- E5's `hard_*` variants set `threshold` to 0, which is plain truncation at `max_rank` (A1.13).
- `queries.per_episode` 3 and `signals.k` 4 are lower than the defaults, so that signals can score most candidates at 15% of 1M tokens: about 500 tokens per query at k=4.
- `train.max_steps` is null in all confirmatory configs. `run.py` refuses otherwise, since the signals budget credited to `S_noselect` would go unspent.
- `matched_arms` makes `S_full` run before `S_noselect` in each seed. If it is missing, `S_noselect` falls back to fraction counts and `analyze.py` labels the comparison exploratory.
- `e5` lists `S_nocompact` so each threshold has a reference. It ignores `compact`, so its cell repeats per variant; run it with one variant if the cost matters.
- `n_entrench` is the same in all multi-night worlds, so E4 can use E3's world exactly.
