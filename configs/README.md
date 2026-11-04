# Configs

JSON has no comments, so assumptions live here. **Every time and size below is an estimate; nothing has been run.**

## Before running

- Models: E1 onward name `Qwen/Qwen2.5-0.5B` as a placeholder. Replace it with E0's chosen model; E6 with the next size up. Hub IDs and licences of the Qwen models have not been checked; the 3B family may carry a different licence.
- `train.lr` 3e-4 and `budget.night_tokens` are placeholders until E1 picks them on the dev world. Set them in E2, E3, E5, E6 before any confirmatory run, and note the values in PREREG.
- E6's arms are placeholders for "the two best arms".

## Seeds

- 7: dev world (E0, E1). Pilot tuning uses this only.
- 8: E0b tuning. 9: E0b's single confirming test (A2.3).
- 2001-2003: E5 sweep, labelled exploratory.
- 1001-1005: confirmatory worlds. Never used in piloting. A test enforces that the dev seed appears in no confirmatory config.

## Assumptions for a Colab T4

From T2 section 5.2, not measured: 0.5B training at roughly 1 to 2 million token passes per 10 minutes under fp16 with gradient checkpointing. The ledger counts forward and backward separately, so 1M ledger tokens is about 500k token passes in training.

| Config | Size | Rough cost |
|---|---|---|
| e0 | 30 persons, 60 probes per kind, 40 for H1, 6 candidates | under 1 hour per candidate at 3B, less below |
| e1 | 40 persons, 3 nights, 9 variants, 6 arms, 1 seed | 15 to 25 hours in total, so split it with `--arms`; start with `B3_replay` |
| e2 | 100 persons (about 650 facts), one 4M-token night, 14 arms, 5 seeds | 40 to 70 hours |
| e3 | 120 persons (about 800 facts, 72 changes), 10 nights at 1M tokens, 14 arms, 5 seeds | 60 to 100 hours; the first arms in the list are the ones hypotheses need. Run those first |
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

## E0b on CPU in GitHub Actions

Tags `e0b-tune-*` and `e0b-confirm-*`. After tuning, commit the artifact's `e0b_tune.json` and `winner.txt` to `results/e0b/`, read the label samples and the template sanity table, and only then push a confirm tag. Confirm refuses any other name, a changed config, or a second run.

**Estimate, not a measurement.** Rate: about 60 tokens per second, from E0 on this runner. Assumed per row: prompt P = 40 tokens, answer A = 8 tokens, 1.2 passes per side for the entropy. A sampled answer costs P + A - 1 = 47, since the last token's step isn't run. A scored answer costs P + A = 48.

| Work | Tokens |
|---|---|
| CHD alone, k = 4: 4 x 47 + 4 x 48 | 380 |
| CHD_mix alone, k = 2, M = 4: 2 x 47 + 2 x 4 x 48 | 478, or 1.26 x CHD |
| Grounding or agreement alone: 2 x 4 x 47 | 376 |
| Entropy drop alone: 2 sides x 1.2 x 40 | about 100 |
| A positive or third-class row, all candidates, sharing samples: 188 + 192 + 3 x 2 x 48 + 188 + 100 | 956 |
| A negative row, reusing its positive's distractor side: 188 + 192 + 288 + 50 | 718 |

| One tuning seed | Rows | Tokens |
|---|---|---|
| Question generation, at most: 250 episodes x 2 x (30 + 64) | | 47k |
| Positives | 150 | 143k |
| Negatives | 150 | 108k |
| Third class | 60 | 57k |
| Template sanity, 20 of each class | 60 | 52k |
| **Total** | | **407k, or about 1.9 hours** |

Confirm computes one candidate, so about 1 hour if it is CHD. The margin against the 350-minute limit is about 3x, and `e0b_partial.json` keeps finished rows if a run dies. If the rule labels under about a quarter of questions positive, 250 episodes will fall short of 150; the output says how many it got, and below `min_class_n` no candidate can win or pass.

- **CHD_mix runs at k = 2, not the deployed 4.** At k = 4 it would cost 4 x 47 + 4 x 4 x 48 = 956, about 2.5 x CHD, and the 1.5 x cap would exclude it by arithmetic before any data. Its estimate is therefore noisier than the others'. Its two samples are CHD's first two.
- Negatives are pairs of positives, so the query-level CI resamples queries, which keeps the pairs together. `ci_episodes` also resamples source episodes, since several questions about one fact are not independent evidence. The task asked for the first; read the second as well.

## S_core and the ablations (A2.1)

- `S_core`: teacher KL on all generated queries, no gate, germline anchors per A1.5, pairs, compaction, commit on anchor drift only. It uses the hard pool `all_real`, the same as `B3_replay`, so the comparison with B3 and B4 attributes to the teacher and pairs alone.
- **Anchor count: `select.fixed_anchor_frac` = 0.5 of the training anchor pool, set explicitly in E2, E3, E5, E6.** It reads no CHD and no other arm. 0.5 because S_full's anchor rule kept what fell at or below the null's median, about half of a pool that behaves like the null. All held-out anchors (up to `commit.audit_max_anchors`) enter the drift test, since it alone decides the commit.
- Derive from `S_core`, one flag each: `S_oracleq`, `S_nokl`, `S_chain`, `S_nocompact`, `S_nopair`, `S_nocommit`, `S_bm25`. Derive from `S_full`: `S_noselect`. Only `S_core` has the exempt H8 log, because of its cost.
- `S_nokl` still generates pair queries it never uses, as it did under `S_full`. That handicaps it by part of the synthesis stage.
- A night with no drift measurement is rejected, and that shows in the acceptance rate. `commit.max_anchor_drift` 0.1 is uncalibrated until E1 reads the table `analyze.py` prints, and it is now the only thing that can reject a night.
- `S_full` and `S_noselect` are off the arm lists: A2.1 runs them only if E0b restores selection. To restore them, add both names to `arms`. `matched_arms` is left in place for that, and `analyze.py` then evaluates H3 and reports `S_full` beside the headline.
- E4 now reads `S_core` from E3's checkpoints, since E3 no longer writes `S_full`.

## H8's log

Each `S_core` night writes `report.fact_audit`: gap on the frozen direct probe and on held-out generated queries, and closed-book correctness on the probe, before and after, per fact. `fact_audit.panel` facts are followed from their first night, chosen by a hash of the seed and key, plus `controls` facts about never-taught people. Exempt from the ledger. **Estimate:** about 800 tokens per fact per night at k = 4, so at most 150k tokens per E3 night at the end of the run, about 1.5 minutes on a T4 at T2's rate, and no effect on the training budget. E5 sets the panel to 0, since it tests compaction only.

## Choices to look at

- `max_pairs` is well above the code's default of 16, because the cap decides how many chains ever get a pair context (review M3).
- `n_regions` is close to `n_cities`, and two-hop is reported per chain, so that guessing the type does not clear a margin (review H5).
- `self_samples` is null: `B4` draws until its teacher allocation is spent, in thousands of sequences (A1.11).
- E5's `hard_*` variants set `threshold` to 0, which is plain truncation at `max_rank` (A1.13).
- `queries.per_episode` 3 and `signals.k` 4 are lower than the defaults, so that signals can score most candidates at 15% of 1M tokens: about 500 tokens per query at k=4.
- `train.max_steps` is null in all confirmatory configs. `run.py` refuses otherwise, since the signals budget credited to `S_noselect` would go unspent.
- `matched_arms` makes `S_full` run before `S_noselect` in each seed, if they are restored. If it is missing, `S_noselect` falls back to fraction counts and `analyze.py` labels the comparison exploratory.
- `e5` lists `S_nocompact` so each threshold has a reference. It ignores `compact`, so its cell repeats per variant; run it with one variant if the cost matters.
- `n_entrench` is the same in all multi-night worlds, so E4 can use E3's world exactly.
