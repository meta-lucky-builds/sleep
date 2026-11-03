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

## Threats accepted

Synthetic facts are not real experience; templated eval could be easier than real use. Results on 1 to 3B models may not transfer up or down (T2 §9). Ten nights is short of deployment timescales. The world generator was written by the same author as the method, so eval templates are fixed and reviewed before any arm runs.
