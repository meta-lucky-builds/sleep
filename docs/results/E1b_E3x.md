# E1b and partial E3x

Exploratory. E1b: dev seed 7, 3 nights, 80 cells, one seed. E3x: seeds 2001 to 2005, 10 nights, **read while still running**: `B3_pairs` on all 5 seeds; `B4_replay_kl` and `B4q` on 4; `S_core` on 2 to 4 depending on night; `S_coreq` not yet. Git `b1da143`, 4×RTX 5090. Tables for E1b: `results/e1b_run1/analysis.txt`. The E3x numbers come from a partial download whose result files aren't in this repo; only its config is. Note: a read-only script parsed these result files on the coordinator's machine; no model or test ran locally.

## E3x at night 10 (setting lr 1e-4, 150k tokens per night)

| Arm | New facts | Reverse | Two-hop | Supersede | Real knowledge kept | Look-alike false recall |
|---|---|---|---|---|---|---|
| `B3_pairs` | 0.73 | 0.60 | 0.02 | 0.72 | 0.07 | 0.31 |
| `B4_replay_kl` | 0.16 | 0.06 | 0.03 | 0.23 | 0.43 | 0.14 |
| `B4q` | 0.13 | 0.03 | 0.03 | 0.15 | **0.80** | 0.07 |
| `S_core` (2 seeds at night 10) | 0.05 | 0.01 | 0.03 | 0.10 | 0.40 | 0.04 |

## Findings

1. **The chosen setting is wrong for this comparison, and that is my rule's fault.** A3.4 forces one setting on every arm. No KL arm reached 0.7 anywhere in E1b, so the threshold relaxed to 0.18. KL arms barely learn here, while plain replay learns and wrecks the model. So E3x can't answer whether sleep beats replay with learning held equal. E1 shows KL arms can reach 0.96 at lr 1e-4 and 500k, which E1b's grid stopped short of.

2. **Question-frame anchors look like a real effect: H10's most promising signal.** At similar low learning, `B4q` keeps 0.80 vs. `B4`'s 0.43 on real knowledge at night 10 (4 seeds, SD about 0.03), and has half the look-alike false recall. The gap held at all 8 E1b settings (+0.11 to +0.42). **H10 is not yet confirmed:** its rule needs `acq_direct` of at least 0.7, which `B4q` never reached. A model that learned little has little reason to be damaged. Whether the gap survives real learning is the open question.

3. **`S_core`, this design's sleep arm, is losing.** At the same budget, it learns less than `B4q` (0.05 vs. 0.13) and keeps much less (0.40 vs. 0.80), and it worsens across nights (0.63, 0.42, 0.40) while `B4q` improves (0.68, 0.76, 0.80). `S_core` has much less training per token, so its setting is even less fair, but nothing here favours it.

4. **Composition is possible when written out.** At lr 1e-4 and 150k, `B3_derived` scores 0.43 two-hop and 0.42 alias on held-out templates, vs. `B3_pairs` at 0.00 and 0.08 (one seed, few items). So weights can hold composed facts. `S_oracleq` reached 0.12 and 0.13, but with only 0.13 direct, so A3.5's branch can't be read cleanly. Teacher KL over queries hasn't been shown to work; explicit statements have.

5. **`ret_real` overstates damage, and earlier statements of mine were too strong.** Greedy answers are fragile. At lr 1e-5, 50k, with almost no training (drift about 0.002 nats), six arms fail 16 to 40 of 150 base-correct questions, mostly different ones (46 in the union, 10 in all six). A near-zero change costs 11 to 27 points. The questions are selected for being correct under the exact base weights, so some regression is built in. **So "0.997 down to 0.53" overstated the damage; the true untouched baseline is around 0.85.** `B0_frozen` was not run on E1b's question set, which was my mistake. Comparisons between arms still stand.

6. **Upgrades can't be judged.** H11, H12 and H13 are untestable, since none reached 0.7. Directionally: compaction changes little (0 to +4 points); rank 8 is mixed; "I don't know" lowered learning at the one setting with learning.

7. **The audit is dropped, per A3.7.** Held-out AUROC at the chosen setting is 0.58, under 0.75. Write gate, pressure and retirement go with it. Gap on template probes scored 0.82 to 0.98, but deployment has no template probes, so that isn't usable as it stands.

8. **`B3_pairs` shows the damage that accumulates.** Over 10 nights, real knowledge goes 0.15, 0.08, 0.07; false recall 0.07, 0.19, 0.31; supersede falls to 0.72 with stale at 0.12. Direct recall is inconsistent across seeds (SD 0.14 to 0.23).

## Next

The question has narrowed: **with learning held equal at a useful level, do question-frame anchors still protect real knowledge, and can explicit derived statements add composition?** That needs per-arm budgets high enough to learn, and `B0_frozen` plus a near-zero control on the same questions. See PREREG Amendment 5.
