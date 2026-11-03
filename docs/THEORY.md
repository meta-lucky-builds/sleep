# Sleep for language models: theory (draft 0, written BEFORE the literature threads returned)

Status: this draft is deliberately frozen as a prior. The literature threads (`docs/literature/`) may
show parts of it are wrong or already done. Revisions go in later drafts, with the change stated, so
that the record shows what the evidence did to the idea. Confidence labels follow `insights.md`:
Established / Supported / Tentative / Speculative. Everything original in this file is **Speculative**
until an experiment says otherwise.

## 1. What is being copied

I-68: copy the function, not the workaround. Sleep in animals does at least five jobs. Which of them
apply to a machine whose weights are frozen in deployment?

| Function of sleep | Constraint it answers in brains | Does the constraint hold for an LLM? |
|---|---|---|
| Separate encoding from consolidation (I-28) | Cannot learn and consolidate at once | **Yes, for a different reason.** Online weight edits in deployment are unvalidated and unreversible in effect. Also reality monitoring (I-67). |
| Fast store to slow store (CLS) | Slow store cannot learn fast without interference | **Yes.** Context is one-shot; weights need many gradient steps. RAG/agent memory is already the hippocampus; nothing is doing the transfer. |
| Protect from interference (I-30) | Overlapping codes | **Yes.** |
| Synaptic downscaling (SHY) | Energy, saturation, SNR | **Partly.** No energy budget. Whether LLM fine-tuning saturates is an open empirical question (T3). |
| Transform to gist, integrate (I-27) | Unknown | **Yes, sharply.** Known LLM failures (reversal curse, two-hop curse, ripple effects) are integration failures. |
| Clear metabolites | Biological | No. |

Three biological constraints that do NOT hold, and which the design should exploit:

1. **Checkpoints exist.** Brains have no copy of yesterday's self. We have `theta_0`, immutable, forever. 
2. **Real episodes can be stored indefinitely.** Generative replay is a workaround for lack of storage.
   Only *old base knowledge* (whose training data we don't have) needs a self-generated stand-in.
3. **Sleep can be a transaction.** A night can be audited and rolled back.

## 2. The central proposal: consolidation is closing the open-book / closed-book gap

Let `p_theta(y | x)` be the closed-book model and `p_theta(y | x, M(x))` the same model given retrieved
episodes from a fast store `M`. Define **hippocampal dependence** of a query:

    HD_theta(x) = KL( p_theta(y | x, M(x)) || p_theta(y | x) )

HD is zero when the store changes nothing about what the model would say. It requires no ground truth.

**Sleep, defined.** Snapshot the teacher. Find new weights minimising

    L(theta') = E_{x ~ Q} KL( p_teacher(y | x, M(x)) || p_theta'(y | x) ) + lambda * Omega(theta' - theta_0)

The observation that makes this more than ordinary distillation: **retention and acquisition are the
same loss term.** When `M(x)` is empty, the target is the old model and the term protects old knowledge.
When `M(x)` is non-empty, the target is the model's own knowledge-integrating behaviour, and the term
writes new knowledge. The fixed point is the state in which the fast store is no longer needed:
`HD = 0` everywhere on Q.

Why expect this to beat hard-label fine-tuning on the episode text:

- In-context reasoning does not show the reversal or two-hop curses that closed-book recall does, to the
  extent the model can do those tasks given context. So the teacher can supply targets for implied
  queries that the episode text never states. (Supported as a premise; the size of the gap for small
  models is the risk.)
- Soft targets carry uncertainty, which hard labels discard (candidate fix for overconfidence).

### What HD does beyond the loss

One signal, several jobs. This is the part I would most like to be true, so it gets the hardest tests.

| Job | How HD does it | Principle |
|---|---|---|
| **Write gate** | An episode whose queries have HD near 0 is already known. Don't store it. | I-15, I-9 |
| **Query selection (REM)** | Sample candidate queries from the model given the episode. High HD = the episode changes the answer, so this is an implication of the episode, as the model itself understands it. | I-27, ripple effects |
| **Interference anchors (NREM)** | Candidate queries *near* the episode with HD near 0 are exactly the neighbours whose answers must not change. Replay them with the old target. | I-7, I-10: "old memories overlapping new learning" |
| **Self-audit** | HD on held-out queries reports how much is still unconsolidated. An oracle-free answer to machine anosognosia. | I-51, I-71 |
| **Sleep pressure** | Mean HD over the active store. Fires when there is something to consolidate. No clock. | Two-process model without Process C |
| **Retirement** | HD low on held-out queries across k successive nights means the episode can leave active retrieval. It is archived, not deleted. | I-16, I-3, I-70 |

Measurement warning (I-50): HD is a *correlate* of consolidation. It must be validated against ground
truth on a benchmark where truth is known before it is trusted for any of those jobs.

## 3. The phases

**Wake.** Weights frozen. Answers use `p(y | x, M(x))`. Episodes are written to `M` through the write gate
with provenance (source, time, real vs generated). Query logs are kept: real prompts are the best
coverage of the distribution that matters.

**NREM (fidelity).** Real episodes, plus HD-near-0 neighbours and generic anchors. Preserves item detail.
Hard labels stay in the mix: self-referential loops need real data (I-67, B35).

**REM (integration).** Queries generated from single episodes and from episode pairs that share a node
(I-22), filtered by HD and a grounding test (I-36: generate then evaluate). Generated *queries* widen coverage;
the *targets* still come from the frozen teacher with the real episode in context. Nothing generated enters
`M` as fact. Budgeted, source-tagged.

**Downscale (homeostasis).** Merge tonight's adapter into the accumulated delta. Each layer's delta is
truncated by singular-value thresholding, then repaired by a few steps of the same loss. Downscaling acts on
`Delta = theta - theta_0`, **never on `theta`**: weight decay on base weights would attack knowledge we have
no way to rebuild (I-6). This is the machine form of "scale down, protect what matters". Motivations:
(a) bounded storage; (b) rate-distortion: consolidation is compression of `M` into `Delta`; (c) PAC-Bayes
bounds with a prior centred at `theta_0` tighten as `||Delta||` falls, which predicts better generalisation
from downscaling, a claim that may well be too clean.

**Commit (transaction).** Morning audit on held-out queries. Accept if HD fell and anchor KL stayed in budget,
else roll back. Delayed tests, because performance leads consolidation (I-16).

### Which teacher for anchors?

Teacher-of-yesterday compounds error like a photocopy of a photocopy. Teacher-at-`theta_0` cannot compound but
would fight anything learned. Proposal: **germline anchoring.** Queries with empty `M(x)` and no overlap with
anything ever consolidated are anchored to `theta_0`. Queries touching consolidated knowledge use yesterday's
teacher, with their episodes in context.

## 4. Pre-mortem: how this could be wrong

1. **The teacher can't do it.** Small models follow context badly, especially against parametric knowledge
   (conflict). Then the targets are noise. This is the largest risk; it may force larger models or a hard-label
   term carrying the weight.
2. **HD is uninformative.** KL on sampled tokens may be dominated by formatting or length effects of context
   rather than knowledge. Prompt format alone shifts distributions, so `M(x)` = irrelevant filler gives nonzero HD.
   Needs a control: HD against a neutral distractor context, not against the raw prompt.
3. **Uniform replay wins anyway (I-8).** Paraphrase augmentation with uniform real replay may already get all the
   benefit, and everything above is decoration. It is a baseline, and it could win.
4. **No plasticity problem exists** at the scale tested, removing a stated motive for downscaling.
5. **Entrenchment.** Once a fact is in weights, the in-context teacher may stop following contradicting context
   (parametric override). Then consolidation makes later correction *harder*. A night's gain becomes a later cost.
   This would be a real and interesting negative result. Test directly (supersession stream).
6. **Query coverage.** Self-generated queries occupy the model's high-probability region, leaving tail knowledge
   unprotected. Test real-prompt vs self-generated anchors, and measure tail knowledge separately.
7. **Already done.** See T1.

## 5. Hypotheses (first version; each needs a falsifier before it is run)

- **H1. Open-book teacher.** On the same step budget, soft targets from the open-book teacher beat hard labels with
  paraphrase augmentation and uniform real replay on implied queries (paraphrase, reverse, two-hop).
- **H2. HD selection.** HD-selected queries beat unselected generated queries.
- **H3. Locality.** HD-near-0 neighbour anchors reduce collateral damage on overlapping facts more than generic anchors.
- **H4. Germline.** Germline anchoring has lower cumulative drift by night N than chain anchoring, with no acquisition cost.
- **H5. Downscale.** SVT plus repair bounds delta size with no loss in retention, and improves later learning or generalisation.
- **H6. Audit.** HD predicts closed-book correctness on held-out queries well enough to drive retirement.
- **H7. Entrenchment.** Consolidated facts resist later contradicting context more than unconsolidated ones.
- **H8. Spacing.** Conflicting facts learned across a night interfere less than when massed (I-30).
- **H9. Coverage.** Self-generated anchors lose tail knowledge that real-prompt anchors keep.

Every mechanism gets an ablation (I-50). Baselines always include uniform random replay (I-8).
