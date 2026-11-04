# Sleep for language models: theory (draft 1)

**Status (2026-11-05): this is the original theory, kept as written apart from citation corrections. Its central bet has since failed.** The signal in section 3 failed its tests (E0, E0b), so selection was abandoned, and the audit, write gate, pressure and retirement were dropped (PREREG A3.7, A5.1). Current results: `docs/results/`.

Draft 0 is in git history (commit `ec2f7a7`), written before the literature came back. This draft states
what the evidence changed. Source IDs refer to `docs/literature/` (A1, C5...) and `insights.md` (I-n, Sn).
Original claims here are **Speculative** until an experiment says otherwise.

## 0. What the literature did to draft 0

| Draft 0 claim | Verdict | Evidence |
|---|---|---|
| Consolidation = make closed-book match open-book, and this is the new idea | **Not new.** Snell 2022 onward. Mechanism is solved at 7B+ | C1–C5, A8 |
| It should work on small models | **In doubt.** SDFT loses to plain fine-tuning at 3B; teacher quality limits it | C5, C4 |
| Context changes knowledge, so the distribution gap is knowledge | **Wrong as stated.** Unrelated context also moves the distribution; part of "transfer" is teacher behaviour and style | C7, C8, C5, C6 |
| Teacher-of-yesterday compounds | **Conditionally.** Continuing from a checkpoint on own outputs reliably amplifies traits under DPO, and reinitialising removes that; under SFT and SDF traits mostly decay or hold. SDPO collapsed in one sequence | 2605.01130, 2607.01763, C4 |
| Downscaling restores plasticity | **Motive dropped at this scale.** Only LM-scale measure extrapolates to about 1.8T tokens before onset at 1B. A few dozen nights is nowhere near | T3 §1 (Zyphra 2606.24752) |
| Downscaling has another motive | **Partly supported.** In one merging test, spectral merging raised some out-of-domain scores and lowered others, with the average nearly unchanged; singular-value clipping claimed to help retention; freeze-and-grow methods grow capacity without bound | 2607.01763, 2608.12332 (ABS), A1, A8 |
| Self-generated anchors can protect old knowledge | **Supported, single-task.** This is now the baseline to beat | A14, 2610.12602, SSR |
| Real data must stay in the loop | **Supported; no safe threshold known.** | T3 §3.2, B1 (SRC loses to 0.75% rehearsal) |
| Reversal and two-hop can be fixed by consolidation | **Conditionally.** Reversal must be written explicitly; composition works only if both facts share a document | T2 findings 3, 4 |
| Learning rate is a detail | **Wrong.** Retention fell 97% to 33% across a 100x lr range; it may dominate every mechanism here | 2610.12602 |

## 1. What remains unfilled (bounded search, not a proof of absence)

1. **Many-cycle control** (T1 §9). No 10+ cycle comparison on a pretrained LLM with matched-compute uniform and
   self-sampled replay, per-cycle capability probes, and seeds. The A1 sleep paper has no replay baseline, no seeds,
   and unequal budgets.
2. **A control layer for sleep.** Published methods consolidate what they are given, on a schedule. What should be
   consolidated, when, whether it worked, and when a memory can leave retrieval are open. This is where the design
   below makes its bet.
3. **Costs that appear only across cycles:** entrenchment, drift compounding, bounded capacity.

Prior art on the control layer (T1 §12): ReSAIL selects steps by JSD between teacher with and without context (top 62.2,
random 49.7, bottom 40.1, over three decoding seeds; training runs not stated), so *integration selection by divergence* already has a precedent; GATES gates questions by agreement
among document-conditioned rollouts. No distractor-controlled version found. Nothing found on using the gap for consolidation timing, ground-truth-free audit,
retirement from retrieval, or on same-fact entrenchment. Those nulls come from a bounded search; retirement's rests on one web query. Anchoring to original
vs. previous checkpoint has no head-to-head found. Kotha and Liang (2603.04964) add a warning for the baselines: replaying generic pretraining-like data
also improved target performance, so B4 may gain on acquisition as well as retention.

So the claim is not "we invented consolidation". The claim is narrow: **a measured signal can control consolidation
better than a schedule, over many cycles**, and it may well fail against uniform replay plus a KL penalty.

## 2. The object of study

Time is split into nights. By day the weights are frozen and answers use a fast store `M`. By night the system maps
`(theta, M) -> (theta', M')`. Success is judged after every night on: acquisition (closed-book, including implied
queries), retention (real capability, tail knowledge), supersession, false recall on look-alikes, drift, delta size, and
store size. All arms get the same compute ledger.

Biological constraints that do not hold here, and are used (I-68): immutable `theta_0`, unlimited real-episode storage,
transactional rollback.

## 3. The signal: controlled dependence on memory

Raw `KL(p(y|x,c) || p(y|x))` is contaminated: any context moves the distribution (C7, C8). So the signal subtracts
the effect of having context at all. For query `x`, episode context `e`, and a matched distractor context `d` drawn from unrelated
episodes:

    CHD(x; e, d) = KL( p(y | x, e) || p(y | x, d) )

Both sides have context, so format and style effects largely cancel. It estimates how much *this content* changes the
answer. Estimated by Monte Carlo with samples from the left side, which is unbiased for sequence KL:

    (1/K) * sum_k [ log p(y_k | x, e) - log p(y_k | x, d) ],  y_k ~ p(.|x, e)

No ground truth is needed. Note what it cannot do: it measures whether the model's behaviour depends on the episode. It says nothing about
whether the episode is true. And whether it tracks knowledge at all is hypothesis H1, tested first, with no training (I-50).

| Job | Rule | Principle |
|---|---|---|
| Write gate | Episode with low CHD on its own queries is already known; don't store it | I-15 |
| Teacher gate | Query whose open-book answer ignores the episode (low CHD where the episode states the answer) has an unreliable teacher: drop or hard-label it | C5, C9 |
| Integration selection | Among generated queries, high CHD = implied by the episode as the model sees it | I-27 |
| Interference anchors | Lexically near queries with low CHD: must not change, replay with old target | I-7, I-10 |
| Audit | Closed-book gap on held-out queries measures what remains unconsolidated | I-51 |
| Pressure | Mean remaining gap over active store; triggers sleep | no clock |
| Retirement | Gap low for k nights: leave active retrieval, keep in archive | I-16, I-3, I-70 |

The closed-book gap used in audit, pressure and retirement is `KL(p(y|x,e) || p(y|x,d))` evaluated on the *current*
weights: it falls as the student absorbs the episode, because the distractor-context answer stops differing.

## 4. The night

**Stage 1, query synthesis.** Candidates per episode: oracle templates (upper bound, benchmark only) or model-generated
(the deployable path). Pair contexts combine two episodes that share a node, since composition needs co-occurrence (T2
finding 4). CHD sorts candidates into integration targets, interference anchors, and discards.

**Stage 2, consolidation.** Student = frozen base + committed delta + fresh low-rank delta. Only fresh trains. Loss:

    L = w_h * CE(real episode statements and explicit reversals)            [anchored to reality, I-67]
      + w_k * KL(teacher(x, e) || student(x)) on integration queries         [answer tokens only: C5 style leakage]
      + w_a * KL(germline or committed(x) || student(x)) on anchor queries

*Teacher sources are frozen for the whole night.* Never a moving self-teacher (C4, 2607.01763). Past real episodes are replayed from the archive.
Generated queries widen input coverage. Targets come only from a teacher given a real episode. Nothing generated is ever stored as fact.

**Stage 3, anchors: germline by default.** Queries with no overlap with anything consolidated are anchored to `theta_0`, which cannot compound
across nights (2605.01130: under DPO, re-initialising removes amplification). Queries touching consolidated knowledge use the committed model
with the relevant episodes in context.

**Stage 4, compaction.** Add fresh into committed delta and truncate by singular-value thresholding per layer, followed by
a short repair using the same loss. Motive: bounded capacity over many nights, unlike freeze-and-grow. It also tests whether trimming removes
collateral (2607.01763). Plasticity is measured but no longer a motive. Always acts on delta, never on base weights (Biderman 2405.09673).

**Stage 5, commit.** Audit on held-out queries. Reject the night and roll back if anchor drift exceeds budget or gap failed to fall.

## 5. What could be wrong (updated)

1. **CHD does not track knowledge.** Then the control layer goes and we report a many-cycle replay study. Tested first.
2. **Uniform real replay plus self-sampled KL wins** (A14, I-8). The most likely single outcome. Still a publishable finding at this rigor.
3. **Small models can't teach** (C5). May force 3B+. E0 decides the model before anything is trained.
4. **Entrenchment.** Facts in weights may resist later contradicting context, so consolidation breaks the teacher for supersession. Tested as H6.
5. **Subliminal transfer** across shared-init distillation is not filterable (Cloud 2025). We can only monitor drift, not exclude it.
6. **Learning rate swamps everything.** All arms get the same tuning.
7. **Oracle queries beat generated ones by a wide margin.** Then the benchmark results don't transfer to deployment. Reported separately.
8. **Retrieval noise.** Oracle retrieval hides this; BM25 arm exposes it.

Falsifiers and thresholds: `docs/PREREG.md`. Code spec: `docs/DESIGN.md`.
