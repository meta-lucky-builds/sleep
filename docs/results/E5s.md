# E5s: question-frame anchors and written-out composed facts

Exploratory. Seeds 3001 to 3003, 16 people, 2 nights, 250k tokens per night at lr 3e-4, Qwen2.5-0.5B, one RTX 5090, git 6131efe. Each arm had 100 real questions per seed. Criteria were stated before the run in `toy1.md`. Raw: `results/e5s_run1/`.

Final night, mean of 3 seeds:

| Arm | New facts | Reverse | Alias | Two-hop | Updated facts | Real knowledge | Look-alike false recall | Generic drift |
|---|---|---|---|---|---|---|---|---|
| `B0_frozen` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| `B4_replay_kl` | 0.99 | 1.00 | 0.17 | 0.07 | 1.00 | 0.47 | 0.75 | 0.05 |
| `B4q` | 0.89 | 0.95 | 0.14 | 0.14 | 0.87 | 0.73 | 0.72 | 0.23 |
| `B4qd` | 0.99 | 0.92 | 0.89 | 0.58 | 1.00 | 0.73 | 0.80 | 0.24 |

Real knowledge is scored on questions chosen because the frozen model answers them correctly, so its 1.00 is by construction. 0.73 means 73% of those questions, not 73% of all the model knows.

## Criteria

| Test | Criterion | Result | Verdict |
|---|---|---|---|
| H10 | `B4q` minus `B4_replay_kl` on `ret_real` at least 10 points | +26 points: +26, +23, +30; 95% t CI 18 to 35 | pass |
| H10 | all seeds positive | 3 of 3 | pass |
| H10 | `acq_direct` at least 0.7 in both | 0.99 and 0.89; lowest seed 0.717 | pass, narrowly on one seed |
| H14 | `B4qd` alias lift at least 0.25 over `B4q` | +75 points: +78, +83, +63 | pass |
| H14 | `B4qd` two-hop lift at least 0.25 over `B4q` | +45 points: +38, +69, +28 | pass |
| H14 | `ret_real` within 10 points of `B4q` | -1 point: 0, -5, +3 | pass |

## What this shows

- Question-frame anchors protect real knowledge while the model still learns. The earlier gap might have come from not learning; here `B4q` learns and keeps it. The anchor questions are the model's own, often nonsense, and none appears word for word in the test set. Reworded overlap wasn't checked.
- The cost is real. Generic drift is 5 times higher (0.23 vs. 0.05), and `B4q` still loses 27 points of real knowledge, mostly on questions the base model answers less consistently (head 0.88, tail 0.59).
- Real knowledge barely moved between nights (`B4q` 0.74 to 0.73, `B4_replay_kl` 0.48 to 0.47, `B4qd` 0.77 to 0.73). Those are means; one `B4qd` seed fell 12 points (0.77 to 0.65). The damage mostly arrives on the first night. Two nights can't show that it stops.
- Telling the model composed facts outright reaches reworded questions at no visible cost to real knowledge. Train and test templates are checked disjoint, and permuted-gold controls are near zero.
- Without those statements, composition is near chance. `B4q` alias is 0.14 against 0.11 permuted. Its two-hop of 0.14 against 0.04 permuted is a small gap on 32 probes per seed, nearly all headquarters-city questions, and not evidence of inference.

## What it doesn't show

- **`B4qd` isn't inference.** It is trained on the answers, from oracle data. H14 shows that the budget can hold such facts, not that a model can derive them. The deployable form is unbuilt.
- **False recall isn't fixed, and it grows.** `false_recall_neighbour` rises from about 0.5 to 0.72 to 0.80 between nights in the learning arms. The frozen base also gives a specific answer to every look-alike stranger and never abstains (`false_recall_any_specific` 1.00, `abstain` 0.00), so these answers were already wrong. What changes is that the made-up answer becomes another person's taught value, which suggests that facts bind loosely to names.
- **Scale.** Two nights and about 117 episodes, not ten nights. `B4q`'s low score on seed 3003 (direct 0.72, updated facts 0.60 on 5 probes) doesn't show strain from scale: it was already 0.73 after the first night, the run with the most skipped steps (see below), then barely moved. The other two arms stayed at 0.97 or above on that seed. It looks more like one bad run, though the cause isn't proven.
- **Noise floor.** No near-zero control ran on these questions. On other question sets, near-zero runs lost 8 points (Toy 1) and 11 to 27 points (E1b) with almost no training, so much of the 27 may be fragile answers rather than damage.
- **Sample.** Three seeds; CIs are descriptive. E5s also departs from A5.2: one setting instead of a grid, 2 nights instead of 3, 16 people instead of E1's 40, 250k tokens per night instead of 500k or 1M, no near-zero control, and no `S_coreq`.

## Oddities to check

- Some cells have non-finite steps, nearly all on night 0 (0 to 9 of about 320). The worst is `B4q` seed 3003 (9 steps, hard loss 1.51 against about 0.4 elsewhere), the same cell with the low direct score. `B4qd`'s apparent +10 points on direct comes almost entirely from that seed, so it should not be read as a benefit. fp16 at lr 3e-4 is the first suspect, unconfirmed.
- The GPU averaged 23% busy, so it wasn't the bottleneck. Find out what was before renting for ten nights. Night time is flat with a fixed token budget (`B4q` 510 s and 506 s).
