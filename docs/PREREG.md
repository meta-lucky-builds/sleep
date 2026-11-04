# Pre-registration (version 1, written before any model has been run)

Changes after data exists must be appended with date and reason, not edited in place. Pilot data (E0, E1) may
tune parameters on a development world seed only. Confirmatory runs use world seeds never used in piloting.

## Metrics

All accuracy metrics are from **free generation, closed-book**, with normalised string match against gold, not
teacher-forced likelihood (T2 §7). Measured after every night.

| Name | Definition |
|---|---|
| `acq_direct` | Accuracy on direct questions about facts taught so far, using templates never seen in training |
| `acq_implied` | Mean accuracy on reverse, alias and two-hop probes of facts taught so far |
| `ret_real` | Accuracy on frozen real closed-book QA the base model answered correctly; reported as change from base, split head/tail by the base model's sampled consistency |
| `supersede` | Fraction of superseded facts answered with the current value; `stale` is the fraction with the old value |
| `false_recall` | Rate of confident specific answers about never-taught look-alike entities |
| `drift_generic` | Mean KL(germline || model) on fixed generic prompts. An alarm only; not validated against factual damage |
| `delta_norm`, `delta_rank` | Frobenius norm and total retained rank of committed delta |
| `style` | Mean response length, rate of context-artefact phrases ("based on the text") |

Budget: each arm in a comparison receives an identical `ComputeLedger` allowance per night, counted in forward and backward tokens
including query synthesis, signal evaluation and teacher generation. Evaluation is not counted.

## Decision rule

Confirmatory: at least 5 world seeds, paired by seed. A hypothesis about a difference is **supported** only if the 95% CI of the paired mean
difference (fact-clustered bootstrap within seed, seeds resampled) excludes zero in the predicted direction **and** the point estimate meets the practical margin:
3 points for accuracies. Holm correction across H2–H7 at each model size. With 3 seeds a result is labelled **exploratory**, whatever it shows.

## Experiments, in order, with kill criteria

**E0. Teacher and signal validity. No training.** For candidate models Qwen2.5-0.5B, 1.5B, 3B (instruct and base where available).
- **H0 (teacher).** In-context accuracy: direct at least 0.90, reverse at least 0.70, two-hop with both facts in context at least 0.60, follows a contradicting context on real facts at least 0.70.
  *Choose the smallest model passing all four. If none pass, stop and report: the teacher mechanism is unusable at Colab scale, and the framework proceeds with hard-label arms only.*
- **H1 (signal).** CHD on queries about an episode vs. unrelated queries: AUROC at least 0.85. CHD has lower offset than raw KL against no context on unrelated queries (lower mean on unrelated pairs). CHD predicts whether the model already knows a real fact (known vs fictional entity) with AUROC at least 0.75.
  *If H1 fails: remove the control layer; E2 and E3 become replay-baseline comparisons.*

**E1. Pilot on dev world.** Learning-rate and step sweep for each arm family, at equal budget. Also determines the night budget at which `B3` (uniform replay) reaches `acq_direct` of 0.7. Kill: if no arm reaches 0.5 `acq_implied` with reversals written into training data, revise the data builder first (T2 §10).

**E2. Single-night ablation ladder.** Arms below; one big night of N facts.

**E3. Ten-night ladder with supersession.** The main result. Per-night curves.

**E4. Entrenchment, on E3 checkpoints plus a dedicated contradiction probe.**

**E5. Compaction sweep.** Thresholds, repair steps.

**E6. Scale check** on next model size for the two best arms.

## Arms

All through one code path with config flags, so baselines don't diverge in implementation.

| Arm | Description |
|---|---|
| `B0_frozen` | No learning, closed book. Floor. |
| `B1_rag` | No learning, retrieval. Ceiling proxy for in-context ability. |
| `B2_seq` | New facts only, hard labels. No replay. |
| `B3_replay` | Uniform replay of all real episodes, hard labels, paraphrases and reversals in the data. **The main competitor (I-8).** |
| `B4_replay_kl` | B3 plus KL penalty against germline on self-sampled text (Marek A14). **Second competitor.** |
| `S_full` | CHD selection, teacher KL, germline anchors, pair contexts, compaction, commit. |
| `S_oracleq` | S_full with oracle templates as queries. Upper bound on query quality. |
| `S_noselect` | Teacher KL on all generated queries, no CHD filter. Ablates selection. |
| `S_nokl` | Selection, hard labels only. Ablates the teacher. |
| `S_chain` | Anchors to yesterday's model rather than germline. |
| `S_nocompact` | Delta accumulates with no truncation. |
| `S_nopair` | No pair contexts. |
| `S_nocommit` | No transactional audit. |
| `S_bm25` | Real retrieval rather than oracle retrieval. |

## Hypotheses and falsifiers (E2, E3 at night 10, on the chosen model)

- **H2 (teacher).** `S_full` beats `B4_replay_kl` on `acq_implied` by 3+ points, with no worse `ret_real`. *Falsified if CI includes zero or `ret_real` is worse by 3+.*
- **H3 (selection).** `S_full` beats `S_noselect` on `acq_implied` at equal budget. *Falsified if not. I-8 predicts this may fail.*
- **H4 (germline).** `S_full` has lower `drift_generic` and `ret_real` loss than `S_chain` at night 10, with the gap widening per night (night × arm interaction). *Falsified if the gap is flat or zero.*
- **H5 (compaction).** `S_full` holds `delta_rank` bounded and is within 3 points of `S_nocompact` on `acq_*` and `ret_real`. The two-sided claim that compaction *improves* `false_recall` or `ret_real` is exploratory. *Falsified if a 3+ point loss.*
- **H6 (entrenchment).** At night 10, open-book contradiction-following is lower for consolidated facts than for matched never-consolidated ones. Directional, not "good": if true, supersession needs a mechanism. Measured for `B3`, `S_full`.
- **H7 (pair contexts).** `S_full` beats `S_nopair` on two-hop probes by 3+ points. *Falsified if not, and T2 finding 4 says this is the best-founded of the set.*
- **H8 (audit).** The closed-book gap on held-out generated queries predicts per-fact closed-book correctness, AUROC at least 0.75. Retiring at the threshold gives a false-retirement rate under 10%. *Falsified otherwise; retirement is removed.*
- **H9 (spacing, exploratory).** Conflicting facts separated by a night interfere less than the same facts in one night (I-30).

Whatever the outcome, the report states the result for `B3_replay` versus `S_full` first.

## Amendment 1 (2026-11-04, before any model has been run)

Reason: independent code review (`docs/REVIEW_1.md`) found that the design tilted comparisons toward the method. No data exists, so these are design corrections. They override the text above where they conflict.

**A1.1 Pair baseline (C3).** New arm `B3_pairs`: `B3_replay` plus concatenated pairs of real episodes that share an entity, under the same pair cap as `S_full`. It uses no model and no signal. The headline comparison is now `S_full` vs. `B3_pairs`, with `B3_replay` also reported.

**A1.2 Metrics are reported by type (C3).** `acq_reverse`, `acq_alias`, `acq_twohop` separately. `acq_implied` remains as a summary only. **H2** becomes two tests: `S_nopair` vs. `B4_replay_kl` on `acq_reverse` and `acq_alias` (attributes to the teacher, without pairs) and `S_full` vs. `B3_pairs` on `acq_twohop`. **H7** stays `S_full` vs. `S_nopair` on `acq_twohop`.

**A1.3 Chance correction (H5).** Each accuracy is also scored against the gold of a different fact of the same type (`permuted-gold`). Margins apply to lift over that control.

**A1.4 `S_noselect` (C2).** Teacher KL on all usable generated queries with no CHD filter. Anchors at the same count as `S_full`, drawn only from queries naming no entity from tonight's episodes.

**A1.5 Anchors (C1, M5).** In all arms with anchors, anchors name no entity from tonight's episodes. Commit tests the *increase* in drift on identical targets before and after the night, and the budget is set in E1 on the dev world.

**A1.6 Audit (H7).** Items are flagged using an independent measurement, and the gap must fall by more than it does on a null-control set.

**A1.7 Validity rule (H8).** A run is labelled exploratory if nights are unequal across compared arms, world fingerprints differ, the acceptance rate is below 0.5, or the unscored fraction exceeds 0.25. Hypotheses missing from a family count as p = 1 in Holm (M10).

**A1.8 `false_recall` (M4).** Renamed `taught_value_bleed`, since it counts answers naming taught vocabulary. `false_recall_any_specific` is tabulated alongside it, and confidence is not measured. PREREG's "confident" is withdrawn.

**A1.9 Answer scope (H1).** CHD, teacher targets and audit are all scored over the answer only: generation stops at EOS or the first newline, on both sides of each comparison. E0's criterion H1 uses this estimator.

**A1.10 E0 (H3, H4, M2).**
- Instruct models are prompted through their chat template; base models by raw completion. E0 reports both, separately.
- Probes are a seeded sample stratified by attribute. Known-vs-fictional compares within attribute.
- H1 is evaluated twice: at E0 settings and at deployed settings (`k`, tokens, model-generated queries), with the parse rate reported.
- Name familiarity is reported and does not gate. Entities are replaced if a binomial test against chance rejects at p < 0.01.
- E0 runs first on CPU in the cloud, on models where that's feasible. It has no training, so hardware doesn't change its validity; only the fp16 path remains untested until a GPU run.

**A1.11 `B4` (H9).** It spends its full teacher allocation on self-samples, drawn fresh, in thousands of sequences. Size set in E1.

**A1.12 Write gate (M8).** Low-CHD episodes are deferred, not dropped: stored without active retrieval and still replayed. The gate rate is reported.

**A1.13 Compaction (M6).** Retention and acquisition are reported by fact age. E5 includes hard truncation as well as soft thresholding.

## Amendment 2 (2026-11-04, after E0 on 0.5B; written before E0b or any training)

E0 is pilot data on dev seed 7 (`docs/results/E0.md`). Base 0.5B passed H0. **H1 failed on both 0.5B models, at E0 settings and deployed settings, so the kill rule applies.**

**A2.1 Rule applied.** CHD-based selection and the write gate leave the main arms for E2 and E3. H3 is withdrawn unless A2.3 restores it. New arm `S_core`: teacher KL on all generated queries with no gate, anchors per A1.5, pair contexts, compaction, and commit on the drift test only. **`S_core` replaces `S_full` in H2, H4, H5, H7 and the headline.** `S_full` runs only if restored. Gap and audit values are still measured and logged, exempt from the ledger, and drive nothing.

**A2.2 H1 is split, post hoc and labelled so.** H1 lumped two questions:
- **H1a (relevance):** does the signal tell whether this episode bears on this query? Selection needs this. Failed.
- **H1b (state):** does it tell whether the model already knows the answer? Write gate, audit, pressure and retirement need this. Between-entity AUROC was 0.997 (base), 0.995 (Instruct).

H1b's result was not predicted separately. Known real entities versus fictional ones is confounded by familiarity with the names. So it earns no control role yet. The clean test is **H8, within fact, before vs. after training**, measured passively in E2 and E3. Only if H8 passes may audit or retirement drive anything, and only in a later, separately pre-registered run.

**A2.3 E0b: replacement signals for H1a. The only route back for selection.**
Observed: own-episode positives in the deployed test include generated questions the episode doesn't answer, so that test mislabels some positives. And replacing no-context with a single distractor barely reduced the offset (1.08 vs. 1.26 nats), so the noise comes from context *content*.
- Candidates, all oracle-free: (a) CHD per token, single distractor, the incumbent; (b) `CHD_mix`, KL to the mixture over M = 4 distractors; (c) grounding: rate at which e-context answers contain a content word from the episode that is absent from the query, minus the same rate under distractors; (d) agreement: modal-answer agreement across samples under e, minus under distractors; (e) entropy drop of the answer's first token, d minus e.
- Labels: positive = query with its own episode, where the query's answer in the world is the episode's stated value, labelled by the world's facts (benchmark only); negative = same query with an unrelated episode. Questions the episode doesn't answer are excluded from positives and reported as a separate third class, because selection must also reject them.
- Tuning on dev seed 8; the single best candidate at equal ledger cost is then tested once on dev seed 9. No other candidate is tested on seed 9.
- **Pass: AUROC at least 0.85 on seed 9, and at least 0.75 for answer-bearing vs. non-answer-bearing own-episode queries, at deployed cost.** Fail: selection is abandoned for this project. Confirmatory seeds are untouched.
- Base 0.5B only. Its result does not rescue H1 as originally stated; it adds a differently specified signal.

**A2.4 Model.** H0's rule picks the smallest passing model: base 0.5B, unless 1.5B changes the reading. Instruct 0.5B's two-hop 0.20 vs. base 0.85 is unexplained; refusals under the chat template are a possible cause, unverified.

**A2.4 addendum (2026-11-04, 1.5B results).** H1 also failed on both 1.5B models, so the kill rule holds at every size tested. Base 0.5B stays chosen; base 1.5B passes H0 at 0.92 to 1.00 and remains the scale check. Instruct failed H0 at both sizes.

**A2.5 Readings.** H6 and H8 are measured on `S_core`. E1 runs with rollback disabled (`max_anchor_drift` at 1e9), because E1 exists to calibrate that limit from the drift distribution; `S_core` nights are all accepted there, and E1's numbers on rollback mean nothing. E1 uses `S_core`, not `S_full`. `CHD_mix` runs at k = 2 in E0b, since k = 4 exceeds the cost cap. `B0_frozen` and `B1_rag` run once per seed, since they train nothing.

## Threats accepted

Synthetic facts are not real experience; templated eval could be easier than real use. Results on 1 to 3B models may not transfer up or down (T2 §9). Ten nights is short of deployment timescales. The world generator was written by the same author as the method, so eval templates are fixed and reviewed before any arm runs.
