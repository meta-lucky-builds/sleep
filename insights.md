# insights.md (v2): Principles, cross-paper understanding, and honest verdicts

`ewk.md` describes what each source found. This file connects those findings. For each principle it gives:

- the claim,
- how strong the evidence is,
- which sources support it (by ID from `papers.md`),
- which sources weaken or complicate it,
- what remains open.

**What changed from v1.**
- Principles **I-1 to I-39 keep their numbers.** Where v1 was wrong, overstated or mislabelled, the text was corrected in place and the change is flagged **(v2)**.
- New principles **I-40 to I-71** were added in Parts 11–15.
- The verdict table on the original idea (Part 0) was redone.
- The tensions table (Part 8) and the corrections list (Part 9) were extended.
- The open-questions list (Part 10) was updated.
- **⚠ unread** marks evidence from a source we only saw as a listing summary, press release or abstract. No principle rests on ⚠ sources alone.

**Confidence labels used throughout:**

| Label | Meaning |
|---|---|
| **Established** | Several strong, independent sources agree |
| **Supported** | Good evidence, with clear limits (species, task, sample size, or a single lab) |
| **Tentative** | One source, a model only, a preprint, or indirect evidence |
| **Speculative** | My own inference or hypothesis, not tested |

---

## Part 0. The starting idea, and what the evidence did to it

The project began with this idea:

- The single most important property of an intelligent system is **fluid adaptability**.
- Human babies are not built to survive immediately. They are built to adapt like sponges.
- Forgotten skills leave a **"pre-skill buffer"** that makes relearning faster.
- **Replay** should follow what is encountered most (frequency and recency).
- The brain has a **cache**. The more a skill is practised, the closer the cache matches reality and the cheaper it becomes to run. A newer, over-optimised cache weakens older ones.
- **Imagination** (for example, practising football through a video game) builds a transferable skill buffer.
- A **"memory stamper"** decides what to store, using three signals:
  1. prediction error (delta),
  2. information gain (KL divergence),
  3. task utility.
- The whole system is a **flywheel**: improving one part improves many others.

The verdict on each piece, after all 91 sources (v2):

| Original idea | Verdict | Main sources |
|---|---|---|
| Adaptability is the key property | **Right, but it is not one property.** It splits into learning speed, retention, relearning speed (savings), transfer, lifelong plasticity and stability. These trade off against each other. Adaptability comes from plasticity plus good priors plus counterweights plus gating plus self-monitoring. | S11, S25, S50, S58, S80, S82, B06 |
| Babies are sponges, not ready-made | **Partly right.** Babies are not blank. Circuits are pre-structured and pre-trained by self-generated activity. Some cross-modal alignment exists before any task (S87). Declarative memory develops later (around 22 months for allocentric memory). Some maps are fixed early. | S25, S58, S87; ⚠ S44, S57 |
| Forgotten skills leave a buffer that speeds relearning | **Right, and one of the oldest findings in memory research (savings, B28).** The mechanism has at least **three sources**: a hidden slow trace, a remembered learning rate, and a recallable strategy. Traces persist in silent or dormant forms. Relearning *rebuilds* guided by the old trace rather than simply reactivating it. The skill buffer lives largely in procedural memory. *(v2: v1 rested this on memory-only references B07–B09; v2 adds S80 and the motor-savings models B44–B46.)* | S16, S26, S80; B07–B09, B28, B44–B46 |
| Replay by frequency and recency | **Closer to right than v1 said.** *(v2 correction.)* Uniform random replay samples in proportion to how often things were encountered, and **at scale it is near-unbeatable** (I-8). Coverage and distribution matching also work (B05). Brains do favour **recency** (S07). What brains add on top are refinements: weakly learned, valued, and overlapping memories. These mainly help *learning*, with small gains for retention at scale. | S07, S19, I-8, B05 |
| The cache gets closer to reality and cheaper | **Right.** This is automatization, chunking and priming (repetition suppression), and the shift from deliberate to habitual control. Code-as-policy (S91) is the explicit, engineered version of a compiled cache. | S26, S27, S91; ⚠ S42 |
| Over-optimising one skill weakens others | **Right, now with direct human evidence.** Taxi drivers' hippocampal reallocation came with worse learning of new visuospatial information (B53). Learning to learn increased adaptability but reduced robustness (S80). Mechanistically, it's interference and saturation. | S01, S50, S80, S81 (B53) |
| Imagination and game play transfer to physical skill | **Partly right, with a sharper mechanism.** Imagined and observed learning is real but smaller than doing. Transfer happens when the practice drives the **same shared latent variables** (timing, trajectories, spatial relations) *and* attaches rules consistent with the real task. Where the mapping conflicts, expect **negative** transfer, possibly worse than no practice (S87). Ungrounded imagination becomes inert (S10). | S30, S31, S33, S10, S87, B24 |
| Delta, KL and utility as stamper signals | **Useful, but in different jobs.** Delta is good for deciding what to attend to and write; *(v2)* its biological support for *sensory* learning is Tentative (S72). Delta is poor for choosing retention replay. Pure delta gets stuck on noise; **learning progress** is the better curiosity signal. Missing channels: interference, salience/relevance, and self-confidence (metacognition). | S01, S07, S72, S82, S84, B05, B38 |
| The flywheel | **Real, but it needs brakes where it isn't anchored to reality.** *(v2 refinement.)* Learning that is corrected by external error slows down on its own. Self-referential loops (Hebbian amplification, training on one's own outputs, replay or imagination feeding itself) run away without counterweights. | S13, S14, S15, S19, S70, S71, S10, B34 |

---

## Part 1. Forgetting and retention

### I-1. Retaining something is far cheaper than learning it. *Supported.* (v2: downgraded from "Established", because only one strong ML source tests it directly.)
- **Evidence:**
  - S06: one replayed sample per 128 new ones, or very poor generated samples, still protected old knowledge. The same small amount failed when everything had to be relearned from scratch.
  - S50: repetition after behaviour has improved makes a skill resistant to decay.
  - S80, B44: relearning after washout is faster than first learning (savings).
- **Why it matters:** a small, cheap maintenance budget can protect a lot of knowledge. This is the economic case for keeping a pre-skill buffer.

### I-2. In artificial networks, forgetting is mostly interference, but not only interference. In brains, time-driven change is real too. *Supported.* (v2: corrected twice.)
- **ANNs:**
  - Forgetting comes mainly from new updates overlapping with old ones (S01, S11). Its severity depends on the geometry between tasks (S01; see the v2 corrections under S01 about where the danger zone lies).
  - **(v2)** Weight changes are not only gradient-driven. Weight decay, drifting normalization statistics and noise-driven drift change networks with elapsed training steps (B33). So time-based models like MSSR (S03) are not purely a scheduling convenience.
- **Brains:**
  - Synapses turn over while coarse fidelity stays stable (S64). Representations drift even in stable environments (⚠ S38).
- **Correction history:** v1 first said "forgetting is interference, not time", then limited that to ANNs. v2 limits it further: it holds for ANNs *without* weight decay or normalization drift.

### I-3. A memory can be stored but inaccessible. Access decays faster than storage. *Supported.*
- **Evidence:**
  - B08: storage strength vs retrieval strength.
  - B09/B10: silent engrams.
  - B07: lasting spines that speed up later re-adaptation.
  - S16: the original engram is silenced, not shown to be erased.
  - S26 (Packard & McGaugh): the hippocampal place memory persisted unused and came back when the caudate was inactivated.
  - S53: stress-retracted dendrites regrow.
  - S09: silent connections keep updating their statistics.
  - **(v2)** B44: in the two-state model, the slow state stays partly adapted while the output is back at zero. That's hidden storage in the simplest possible model.
  - **(v2)** B31: in ANNs, probes on frozen features often recover old tasks, so forgetting is often in the readout.
- **Why it matters:** "forgotten" and "gone" are different states. A design can deliberately keep traces silent instead of deleting them. A cheap diagnostic, storage vs access, follows from this (I-71).

### I-4. Relearning, and recalling remote memories, means rebuilding, not retrieving. *Supported (mouse, one paradigm).*
- **Evidence:**
  - S16: remote recall recruits **new** hippocampal neurons, guided by the cortical gist and the current context, and **updates** the memory.
  - S17: patterns before and after sleep are not directly similar. The memory's format changes.
  - S28 argues episodic memory exists for constructive recombination (theory).
- **Complication:** S16's mechanism depends on adult neurogenesis, which is disputed in humans (B20). Bats and whales have good spatial memory with little hippocampal neurogenesis (S53). The *rebuild* principle may hold without the neurogenesis step.
- **(v2) What "rebuild" must mean to be distinctive.** "Copy the archive and fine-tune the copy" only protects the archive. S16's actual content is that the new trace is built from a **compressed gist plus the current context**, in new units. The distinctive ML prediction: rebuilding from a *lossy* summary of an old skill plus fresh data should beat fine-tuning the full old weights *when the context has changed*, because the gist is less tied to the old context. *Speculative; tested in `design.md` E4.*
- **Interesting:** this matches the user's original wording, "not memory extraction but rebuilding skill."

### I-5. Recalling a memory opens a window in which it can change, both to update and to corrupt. *Supported.*
- **Evidence:**
  - S16: memories updated with the context present at recall. The authors propose this as a mechanism for false memory.
  - S20: reconsolidation.
  - S19: the misinformation effect.
- **Why it matters:** retrieval is a high-plasticity, high-risk moment.

### I-6. Brains do not "use it or lose it" in a simple way. *Supported (refined).*
- **Old view (S50):** disuse degrades circuits, and their territory is reallocated (Merzenich digit maps, B26).
- **Refinement:**
  - S58 shows that large-scale reallocation is mostly **early and homeostatic**, not driven by use.
  - Adult maps are more stable than the classic story implies.
  - Meanwhile, traces persist in reversible forms (I-3).
- **(v2) Scope correction.** S58 studied *congenital* deprivation. "Homeostatic takeover threatens dormant traces in adults" is an inference, not a tested result. Its ML analogue is mechanical, though: weight decay and global normalization act on unused parameters exactly as a homeostatic force would. Excluding archived parameters from them is a concrete design rule.

---

## Part 2. What to replay and what to write

### I-7. Replay can increase forgetting. Selection by error alone can pick exactly the harmful samples. *Established in theory (S01). Tentative empirically.*
- **Mechanism:** reducing one sample's error moves error onto overlapping samples. In S01's worst case, the high-error sample (x2) was the harmful one to replay.
- **Small replay** between moderately similar tasks is the risky zone (S01 MNIST: 1–2 samples hurt, 4+ helped). **(v2)** "Moderately similar" means null-space angles *below* π/4. Above π/4, replay is guaranteed safe.
- **Why it matters:** any "replay what has high error" scheme (S03, B01, the original stamper) needs an interference check.

### I-8. At scale, random replay is very hard to beat in supervised learning. *Established (S07).*
- Clever selection helps in small benchmarks and in reinforcement learning, but often only marginally in large supervised settings.
- **(v2)** Uniform random sampling from a reservoir buffer *is* frequency-proportional replay. This is why the original "replay by frequency" intuition deserves more credit than v1 gave it (Part 0).
- **Why it matters:** random replay must always be the baseline.

### I-9. Prediction error is good for attention and writing. Coverage and interference are what matter for retention. *Supported, with a v2 downgrade for sensory learning.*
- **Evidence:**
  - B05 (via S07): TD-error and reward selection did not prevent forgetting. Coverage and distribution matching did.
  - S01: error-driven selection can be harmful.
  - S27: deviation from the internal model captures attention and determines what enters working memory.
  - S64: familiarity (low novelty) suppresses writing.
- **(v2) Caveat.** S72 states that "prediction errors update internal models" has **not been tested causally** for sensory prediction errors (only for dopamine reward-prediction errors). For sensory learning, "delta gates writing" is a *reasonable design rule with Tentative biological support*.
- **(v2) Two kinds of delta should be separated (I-48):**
  - *local novelty* (adaptation, "has this just repeated?"): cheap, automatic, good for suppressing redundant writes
  - *model-based surprise* ("does this violate learned structure?"): sparse, held in higher areas, better for triggering structural updates
- **Synthesis:** **delta gates what is written and attended. Coverage and interference drive replay for retention.**

### I-10. The brain replays by recency, risk, value, intermediate strength and overlap. *Supported.* (v2: recency added.)
- **Evidence:**
  - S07: replay favours recent experience, weakly learned items, reward and fear, and old memories that overlap with new learning.
  - S19: targeted reactivation works best for memories of intermediate strength (strong enough to reactivate accurately, with room to improve).
  - S26: amygdala value signals modulate consolidation.
  - S50: salience.
- **(v2) Correction.** v1 said "intermediate strength amounts to learning progress". They are related but not the same. *Intermediate strength* is a **state** (how well something is known); *learning progress* is a **rate** (how fast it is improving). Both avoid the two extremes (hopeless noise; already mastered). They can be measured separately and should be compared (`design.md` E2).

### I-11. Importance can be assigned after the fact. *Supported.*
- **Evidence:**
  - S20: synaptic tagging and capture, behavioural tagging, and retroactive enhancement (weak memories become strong when something important follows).
  - S61: BTSP eligibility traces span about ±5 seconds, and inputs *before* the plateau count over a longer window.
- **(v2) ML counterparts already exist:** eligibility traces in reinforcement learning (TD(λ)) and Hindsight Experience Replay (B51). They are the baselines for any "delayed tagging" design.
- **Why it matters:** a stamper does not have to decide importance at the moment of experience. Cheap tags plus a later teaching signal work.

### I-12. Replay quantity matters. Small replay is risky; tiny replay can be enough. *Tension, see Part 8.*
- S01: small replay can hurt.
- S06: tiny replay is enough.
- **(v2) Reconciliation.** S01's harm is measured against an unusually benign no-replay baseline (minimum-distance updates in linear regression). In deep networks the no-replay baseline is catastrophic, so replay's occasional extra harm is small by comparison. S01's harm also needs a specific task geometry; S06 reports benchmark averages.

---

## Part 3. Stability and plasticity

### I-13. Every self-reinforcing process needs a counterweight, and the counterweight must be dosed correctly. *Supported by many independent lines.* (v2: scope sharpened; see I-67.)

| Counterweight | When it acts | Source |
|---|---|---|
| Fast inhibitory decorrelation (firing rates kept stable, correlations reduced) | Waking, minutes | S13 (experiment) |
| BARRs inhibit the just-replayed groups of neurons; reactivation returns to baseline | Sleep | S14, B11 |
| Anti-Hebbian plasticity during replay keeps replay stable | Offline | S15 (model) |
| Synaptic down-scaling alongside replay (synaptic homeostasis) | Sleep | S07, S19, S20 |
| Homeostatic plasticity against self-reinforcing Hebbian change | Hours | S52, S54, B12 |
| Built-in normalization in the learning rule (pointwise mutual information) | Always | S09 |
| Feedback inhibition preventing mass simultaneous plateaus | Always | S61 |
| **(v2)** Inhibition stabilizing strong recurrent excitation (inhibition-stabilized network) | Always, milliseconds | S71 |
| **(v2)** Top-down (prefrontal) suppression of sensory responses; removing it increases oddball gamma | Always | S75 |
| **(v2)** Decorrelation of deep-layer activity by antipsychotics (hypothesized therapeutic mechanism) | Pharmacological | S70 |

- **Dose matters:**
  - Depression alone erases the memory (S15).
  - Overtrained anti-Hebbian learning collapses (S08).
  - Overuse right after an injury enlarges the damage (S50).
- **Failure modes when counterweights are missing:**
  - epilepsy (S35, S25)
  - chronic pain and addiction (S52)
  - rumination (S19)
  - saturated synapses blocking new learning (S50: Moser 1998)
  - **(v2)** excess long-range correlation as a candidate mechanism of psychotic false inference (S70)
  - **(v2)** model collapse when generative models train on their own outputs (B34)

### I-14. Plasticity requires a teaching signal. Untargeted plasticity is harmful. *Established.*
- **Evidence:**
  - S50 salience: tones without reward or acetylcholine pairing produced no reorganisation; attention was required.
  - S50 interference: stimulation not shaped by behaviour hurt learning.
  - S61: plateaus need instructive input and released inhibition.
  - S72: VIP disinhibition plus higher-order thalamic input gates plasticity in cortex.
  - S70: noradrenaline drives prediction-error-induced plasticity.
  - ⚠ S40 (inhibition reduced at goals), ⚠ S41 (dopamine increases plasticity when new information arrives).

### I-15. Plasticity responds to learning, not to repetition of what is already known. *Established.*
- **Evidence:**
  - S50 specificity: repeating learned movements produced no new synapses or map change.
  - S61: BTSP saturates, and plateaus inside an existing field change little.
  - S64: familiarity suppresses writes.

### I-16. Consolidation lags behind performance. *Supported.*
- **Evidence:**
  - S50 repetition: behaviour improved days before synaptic and map changes appeared. LTP needed about 5–15 days to stabilise.
  - S51: spines matured over 30–60 days.
  - S50 time: early changes are more fragile.
- **Why it matters:** plateaued performance does not mean the knowledge is consolidated. Delayed retention is the test.

### I-17. Stable and plastic parts can coexist, even inside a single neuron. Locking is driven by experience, not by time. *Supported.* (v2: evidence reorganized.)
- **Evidence:**
  - S57 (press release only): thalamic visual inputs land on a dendritic zone with almost no NMDA receptors and no plasticity; other zones of the same neuron stay plastic. Locking needs visual experience: dark-reared mice never lock.
  - S59: representations stretch flexibly on top of a frozen feature extractor in the model.
  - S06: internal replay works only if early layers stay stable.
  - **(v2)** S87: the model reproduces cross-modal transfer with *only* readout synapses plastic.
  - ⚠ S45: stable and dynamic engrams exist side by side.
- **Why it matters:** the answer to stability vs plasticity can be **per input type or per pathway**, not per network.

### I-18. Persistence can come from rare writes rather than frozen synapses. *Tentative (S64 preprint, S63 abstract).*
- **Evidence:**
  - S64: microscopic synaptic turnover coexists with stable memory fidelity. Raising the threshold makes rewrites rarer.
  - S63: BTSP "could mitigate the need for widespread synapse stabilization."
  - Consistent with drift (⚠ S38).
- **Why it matters:** it offers a way out of the dilemma between losing plasticity (B06) and forgetting. Keep writes rare and aligned instead of freezing.

### I-19. Locking too much kills learning. *Established in ML (B06).*
- Loss of plasticity is the mirror image of catastrophic forgetting.
- Critical periods in biology close plasticity (S25), but the brakes can be removed (perineuronal-net digestion; S51 mentions perineuronal-net changes after interventions).
- **(v2)** Metaplasticity (B47) and complex-synapse theory (B48) are the formal treatments of this trade-off and are missing from the library (`agenda.md`).

---

## Part 4. Representation: separation, sharing, gating, structure

### I-20. Separating tasks protects against interference but blocks transfer. *Established as a trade-off (S11). Partly resolved in v2 by I-57.*
- **Partial answers:**
  - **Factorised, compositional codes** (S11).
  - **Content and context in separate populations, bound when used**, with few conjunctive neurons in humans (S22).
  - **Learned routing to reuse old modules** (S04).
  - **(v2)** **A shared latent variable with task-specific readouts** (S87, I-57): rules attach to the latent, so they transfer wherever the latent is shared.
  - **(v2)** **Share where updates align, separate where they conflict** (B39): decide by the sign of gradient alignment, not by input similarity.
- **(v2) Correction about hierarchy.** v1 proposed "shared slow low-level features, with sparse gated task-specific parts above". That is the vision layout. In motor skill (S79) the *specific* learning sits low (execution) and the *general* learning sits high (sequencing). **Where to share depends on where the task invariants live** (I-59).
- **Complication:** rodent hippocampus is more conjunctive (S24's event codes, S22's comparison). Both kinds of code probably exist.

### I-21. Context gating can be learned, but discovering what the contexts are is still open. *Supported, with a corrected claim; v2 adds a candidate mechanism.*
- **Evidence:**
  - S12: Hebbian learning on a **given** context cue produces gating automatically.
  - S59: stretching along the relevant dimension emerges from minimising error.
  - S22: context codes are reinstated during the task.
- **Correction (v1):** S12 does not discover contexts; the cue is given.
- **(v2) Candidate mechanism, Speculative:** a *relevance* detector (salience network, S84) flags a meaningful, persistent change, and a separate *task-control* system maintains or switches the task set (S84, B57). A mismatch strong enough to fail a "vigilance" test recruits a new category (ART, B40). Combined with volatility estimation (B42), this gives a testable recipe: **relevant, persistent surprise → allocate a new context code; irrelevant or one-off surprise → ignore** (`design.md` E20).

### I-22. Memory is organised as a graph of linked events, not only as a map of space. *Supported (S24, with B23).*
- **Structure:**
  - event (conjunctive) codes
  - sequence codes linking events within an episode
  - nodal codes for features shared across episodes
- **What this structure enables:** walking along linked episodes supports transitive inference and shortcuts without metric calculation. Sleep improves this kind of inference (S18).
- **Complication:** grid cells and goal-vector place cells (⚠ S49) mean space is more than one node among many. The modern view: entorhinal cortex provides structure, and the hippocampus binds experiences to it. Supporting non-spatial mapping: ⚠ S47, ⚠ S48.

### I-23. Sparse storage combined with a dense, compressed readout supports fast learning. *Tentative-to-supported (S37 abstract, S62, S64).*
- **Evidence:**
  - S37: CA3 ultrasparse → CA1 dense, which helps learn new maps quickly (model).
  - S62: sparse one-shot writes with binary weights give a high-capacity memory retrievable from partial cues.
  - S64: sparse one-shot writes to selected units limit interference.

### I-24. Perception is partly postdictive. The stored version of an event can be revised by what came after. *Supported for the phenomena; the theory is speculative.*
- **Evidence:**
  - S28 collects postdiction illusions (cutaneous rabbit, colour phi, colour fusion, cues after the stimulus improving awareness).
  - The phenomena are real.
  - The "consciousness is memory" theory built on them is speculative.
- **Engineering reading:** write the hindsight-revised (smoothed) estimate into memory, and act on the online (filtered) estimate. **(v2)** ML counterpart: smoothing, and Hindsight Experience Replay (B51).

---

## Part 5. Memory systems, working memory, consolidation modes

### I-25. Several memory systems run in parallel, each with different rules. *Established (S26).*
- **The systems:**
  - declarative: fast, flexible
  - habit/skill: slow, rigid, insensitive to changes in reward value
  - priming: reduced activity on repeated input
  - conditioning
  - an emotional/value system that modulates all the others
- Control of behaviour shifts between them with training, stress and age.
- **Three consequences for the buffer idea:**
  1. **Skill savings can exist without episodic memory.** H.M. learned mirror drawing without remembering any practice.
  2. **A dormant system can keep its knowledge while another system controls behaviour** (Packard & McGaugh).
  3. **Format change exposes rigid knowledge** (Bayley et al. 2005).
- **(v2)** Motor adaptation adds a further split: cerebellar error-driven trial-to-trial learning vs early cognitive/striatal support (S80).

### I-26. Working memory is a tiny bottleneck that decides what enters, what gets linked, and what is understood. *Established (S27).*
- About 3–4 chunks.
- Associations form between items held together in the focus of attention.
- Understanding a concept needs its related elements held at once (relational complexity).
- Chunks stored in long-term memory extend effective capacity (long-term working memory).
- **Link to the original "cache" idea:** chunks are the cached, compressed patterns. Expertise lets the same capacity do more with less effort.
- **(v2)** Why the limit is 3–4 is unknown (S27). An artificial system should treat bottleneck size as a parameter to tune, not copy the number 4 (I-68).

### I-27. Consolidation has at least two modes that pull in opposite directions. *Supported.*
- **Evidence:**
  - S17: slow-wave sleep preserves item detail; REM transforms memories toward gist (the REM/SWS ratio predicts the shift).
  - S19: waking replay is faithful; sleep replay is noisy and distributed.
  - S07: non-REM transfers; REM organises and abstracts.
  - S29: REM favours distant associations.
- **Why it matters:** "replay" is not one operation. Preservation and transformation are separate objectives, and their balance trades retention against generalisation.

### I-28. Encoding and consolidation are different global modes. *Supported.*
- **Evidence:**
  - S20: acetylcholine switches between encoding (high) and consolidation (low) modes.
  - S29: sleep is poor at forming new memories and good at reorganising existing ones.
  - S35: replay events briefly shut down ongoing processing.
  - S23: awake replay occurs in inward-focused, lower-arousal states.
- **(v2) A functional reason a machine also needs modes:** keeping generated (replayed or imagined) data from being mistaken for real experience (I-67). The mode switch is a form of reality monitoring.

### I-29. The state during replay determines whether replay helps. *Tentative (S23 preprint).*
- Replay that engaged sensory, memory/limbic and default networks together helped retention.
- Replay that was mostly sensory went with more forgetting.
- **Speculative ML reading:** replay that runs through the whole stack (content, context and value) may help more than replay that refreshes only low-level layers.

### I-30. Consolidating between tasks protects against interference. *Supported (S18: Ellenbogen 2006).*
- Recall was 76% vs 32% when conflicting learning came after sleep vs after wake.

### I-31. How long consolidation takes depends on how many links a memory needs. Schema fit speeds it up. *Supported (S24, S20; B50 Tse et al. from memory).*

---

## Part 6. Imagination and observation

### I-32. Learning from imagined and observed experience is real. *Supported.*
- **Evidence:**
  - S30: watching movements creates a specific motor trace of the same kind as practice, but smaller.
  - B24: observation speeds up later physical learning, and the benefit needs the motor system to be free.
  - S31 (press release only): imagined encounters produce reward prediction errors and change preferences.
  - S33: self-imagination modestly improves memory.
  - S19: puzzles cued during sleep are solved more often.
  - ⚠ S43: looking at a location activates place codes.
- **Limits:**
  - The effects are smaller than doing the real thing.
  - In S31, imagining negative encounters didn't change liking.
  - Transfer to complex physical skill (the video-game-to-football question) has not been measured.
- **(v2)** I-57 gives the condition under which transfer from practice in another medium should be expected: shared latents plus consistent rules.

### I-33. Imagination is recombining memory parts, and it needs an intact memory system. *Supported.*
- **Evidence:**
  - S32 (abstract only): retrieval and imagination share representations.
  - S35 (expert view).
  - S22: separable content and context codes are what recombination requires.
  - S33: no imagination benefit when the medial temporal lobe is degraded.
  - S28: imagination is constrained by episodic memory.

### I-34. Intermediates that the objective doesn't force to be used get bypassed. Ungrounded imagination is one case. *Supported (S10; with S50 and S26).* (v2: generalized.)
- **Evidence:**
  - S10: latent "imagination" tokens collapsed, ignored the input, and barely affected answers. Explicit, grounded imagination worked better.
  - S50: "learned non-use": an easy compensatory path entrenches and the impaired path withers.
  - S26: the habit system takes control from the hippocampus with training.
  - Contrast: world models whose latents are trained to predict the future, and whose decisions depend on those latents, do use them.
- **Why it matters:** imagination must be **grounded** (trained against real outcomes) and **causally load-bearing** (decisions must depend on it), and this must be verified with interventions. See also I-69 on which bypassed modules survive.

### I-35. Imagination and replay can run away. *Supported.*
- **Evidence:**
  - S19: replay biased toward negative content resembles rumination; cueing negative memories worsened ratings in anxious adolescents.
  - S31: "you can paint the world black just by imagining it."
  - S35: epilepsy as self-reinforcing plasticity.
  - S10: collapse.
  - **(v2)** B34: model collapse from training on one's own outputs; B35: the remedy is to keep real data.
  - **(v2)** S81 (B54): runaway thought loops are maintained by meta-level beliefs; changing those beliefs works.
- **Why it matters:** imagination needs the same counterweights and diversity checks as replay (I-13), plus **source tagging** (real vs generated) and a meta-level regulator (I-55).

### I-36. Generate, then evaluate. *Weak evidence, sound logic (S34).*
- Imagination involves a generative phase and an exploratory/evaluative phase.

---

## Part 7. Development and lifespan

### I-37. Brains are pre-trained on self-generated activity and develop in stages. *Supported (S25; v2 adds S87, S86).*
- **Evidence:**
  - Bursts in entorhinal cortex and hippocampus triggered by spontaneous twitches.
  - Stage-wise maturation driven by stellate cells.
  - Self-centred (egocentric) memory before world-centred (allocentric) memory.
  - Trajectory replay appears only after the underlying structure matures.
  - **(v2)** S87: aligned visual–tactile maps exist in task-naive mice, probably refined by correlated spontaneous and sensory activity.
  - **(v2)** S86: motor babbling as the source of the earliest useful invariants (argument).
  - ⚠ S44: pre-existing sequence repertoires selected by action plans.

### I-38. Critical periods exist, and abnormal activity during them causes lasting damage. *Supported (S25; ⚠ S57).*
- Removing Arc in the first postnatal month permanently impairs learning; removing it later does not.
- Early-life seizures disrupt timing codes.
- Lock-in of stable synapses depends on experience (S57, press release only).

### I-39. Plasticity declines with age but persists. Lifestyle and enrichment matter. *Established (S50, S51, S52, S55).*
- Varying or increasing environmental complexity beats constant enrichment (S51).
- Voluntary exercise beats forced exercise (S51).
- Isolation reduces plasticity (S51).
- **(v2)** Older adults keep savings and learning to learn even when initial learning is slower (S80).

---

## Part 8. Tensions between sources

| Tension | Sides | Current reading |
|---|---|---|
| Does small replay help or hurt? | S01: it can hurt. S06: tiny replay is enough. | (v2) Different baselines: S01's no-replay baseline is benign; deep networks' is catastrophic. Harm needs specific geometry (angles below π/4). Test it. |
| Separate or share representations? | S11/S22 separation vs the need for transfer | (v2) Share a latent where task invariants align; separate where updates conflict (I-57, B39). |
| Random or instructive BTSP gating? | S62: random, near-optimal rate. S63/S64: instructive input links plasticity to experience. | A low random baseline plus instructive modulation, kept sparse. |
| Winner-take-all allocation? | S62: BTSP avoids global WTA. S64: optimised allocation became effectively WTA. | Open. Compare both. |
| Remapping: use-driven or homeostatic? | S50 (Merzenich): use. S58: mostly early and homeostatic. | Adult large-scale maps are more stable than the classic story implies. |
| Do dreams matter for consolidation? | S29: dreams are a by-product. Zadra & Stickgold: they matter. | Unresolved. |
| Does sleep improve motor skills? | S18/S35: yes. B19: gains partly artefactual. | Sleep *protecting* memory is better supported than sleep *improving* skills. |
| Does adult human neurogenesis matter? | S53: about 700 new neurons per day. B20: disputed. Bats and whales have little. | Don't rely on it as the only route to new capacity. |
| Blocked or interleaved practice? | S11/S12: blocked is better for human category learning. Motor learning often benefits from variety (S80). | It depends on the domain. |
| Rich vs lazy regime | S11: depends on initial weight scale and task demands. | A design choice to measure, not assume. |
| **(v2)** Are prediction errors residuals? | S70 (classical layer-2/3 error neurons) vs S72 (amplified features) vs S75 (no dedicated error neurons) | Residuals for low-dimensional mismatches (reward, visuomotor); amplified features for high-dimensional ones; routing for complex predictions (I-41, I-42). |
| **(v2)** Do neuromodulators implement precision? | S73: yes. S70: too coarse in mouse V1. | Broad modulators set global learning rate and plasticity; specific precision comes from local circuits, normalization and routing (I-47). |
| **(v2)** Amplify the expected or the unexpected? | S71: natural (familiar) patterns amplified. S72: unexpected features amplified. | Different timescales: long-term statistics amplified as a prior; violations of short-term expectations amplified as surprise (I-43). |
| **(v2)** Psychosis mechanism | S70: deep-layer over-connectivity, strong high-level priors. S75: weak bottom-up gamma (PV cells). S73: aberrant precision either way. | They agree on the outcome (an internal model insufficiently anchored to evidence) and disagree on mechanism. Treat as hypotheses (Bowman's warning, S73). |
| **(v2)** Dedicated cross-modal hub or not? | S87: area RL necessary and sufficient. S88: no polysensory area; insula–claustrum only. S86: invariants pervade cortex. | Different invariants, different architectures (spatial vs shape); methods differ hugely (I-63). |
| **(v2)** Must cross-modal invariants be learned from scratch? | S86: no physical ground truth, so learned individually. S87: aligned maps exist before any task. | Developmental alignment builds a prior; task learning attaches rules. The ground truth is the shared physical latent (I-58). |
| **(v2)** Does prediction error drive learning (predictive coding) or does match drive learning (ART)? | S70/S72 vs B40 (via S77) | Both, in different roles: mismatch allocates new capacity, match refines existing structure (I-62). |

---

## Part 9. Corrections made during the project (so they are not repeated)

**From v1:**
1. **"Hebbian gating discovers contexts without labels" (about S12): wrong.** The context cue was given.
2. **"Forgetting is interference, not time": too strong for brains.** Drift exists (⚠ S38, S64).
3. **Using Merzenich digit remapping as clean evidence of use-driven reallocation: qualified** by S58.
4. **Recommending reactivation of archived modules: corrected** to rebuilding guided by the archive (after S16).
5. **"Train on what you encounter most is wrong": refined.** Wrong for replay, right for learning weights (S09).
6. **The Wu & Maass citation was uncertain, then verified** (S62).
7. **"Delta-prioritised replay prevents forgetting": corrected.** Delta belongs to writing and attention, not to retention replay (I-9).
8. **Elman's "starting small": flagged as contested** (B15).

**New in v2:**

9. **"Forgetting is interference, not time" for ANNs: also too strong.** Weight decay, normalization drift and noise-driven drift cause step-dependent forgetting (I-2).
10. **"Replay by frequency is wrong": corrected again.** Uniform random replay is frequency-proportional and near-unbeatable at scale (I-8, Part 0). Recency is a documented replay bias (S07).
11. **"Worst forgetting around π/4": corrected.** The harmful zone is below π/4; above it, replay is guaranteed safe (S01 v2 note). There's an ambiguity about whether "worst at intermediate angles" refers to replay at all.
12. **I-1 "Established": downgraded to Supported** (one strong ML source).
13. **"Intermediate strength = learning progress": corrected.** State vs rate (I-10).
14. **"Ungrounded imagination becomes inert": generalized.** Unforced intermediates get bypassed (I-34).
15. **Several unread sources (S38–S49) were used as evidence without marking.** Now marked ⚠.
16. **The pre-skill buffer was supported mainly by memory-only references.** Now also by S80 and B44–B46; and savings itself is old (B28), so "measuring savings" is not novel (I-60).
17. **"Prediction error gates writes" was labelled Supported for sensory learning.** Downgraded to Tentative (S72).
18. **"Shared low-level, specific high-level" as a universal layout: corrected.** It depends on the domain (S79; I-59).
19. **"Homeostatic takeover threatens adult dormant traces": scoped** as an inference from congenital data (I-6).
20. **"Counterweights everywhere": sharpened.** Supervised error-corrected learning already has a brake. Counterweights matter most where learning isn't anchored to external error (I-67).

---

## Part 10. Genuinely open questions (where new work could matter)

1. **Engineered savings.** Measuring savings isn't new (B28–B30); *engineering* it deliberately and decomposing its sources (I-60) is under-explored. This is the clearest gap tied to the original idea.
2. **Rebuild vs reactivate vs fine-tune** for relearning under changed context, including rebuilding from a *lossy* gist (I-4). Untested in ML.
3. **A replay selection rule that beats random replay at scale** in supervised learning.
4. **Discovering contexts without supervision** from raw, non-stationary streams. Candidate mechanism in I-21 and I-54.
5. **The dose and schedule of counterweights**, and *where* they are needed (I-67).
6. **Where the instructive (allocation) pathway comes from:** fixed random, slowly learned, or derived from context codes.
7. **How long a silent trace stays useful** for relearning.
8. **How much generated replay or imagination is safe** before its errors compound, and what fraction of real data prevents it (B35).
9. **REM-style transformative replay** in ML: abstracting and recombining rather than repeating.
10. **Whether observation or simulation pre-training transfers to complex motor skill**, and when it transfers *negatively* (I-57).
11. **Whether a working-memory bottleneck helps generalisation** ("starting small"), and whether arbitrary structure needs a limited workspace (I-64).
12. **Experience-gated locking** of feedforward weights, versus never locking or locking by time.
13. **(v2)** **"Machine anosognosia":** do continual learners know when they have forgotten? (I-51)
14. **(v2)** **The cost of learning to learn:** does meta-learning across varied tasks reduce robustness to perturbation, as in humans (I-61)?
15. **(v2)** **Is there a single write-gate motif** (novelty → disinhibition + instructive input → plateau) shared across brain regions (I-45)?
16. **(v2)** **Do sensory prediction errors cause model updates?** Untested in biology (S72).

---

## Part 11. Predictive processing, revised (new in v2)

### I-40. Prediction errors have two jobs: updating the current representation and teaching the internal model. *Supported (S70 argument; B44's error-driven updates; standard in learning theory).*
- A network that computes prediction errors has a local learning signal at every node. S70 argues this, not efficiency, is the main advantage of predictive processing.
- **Caveat:** S72 notes that the *teaching* role is untested for sensory prediction errors.

### I-41. Sensory cortex keeps a faithful feature code. Prediction acts on it through gain and routing, not mainly through subtraction. *Supported.*
- **Evidence:**
  - S72: the same unexpected stimulus under different predictions gives the same layer-2/3 response; responses are feature-specific and amplified in proportion to surprise.
  - S75: global (learned-structure) violations are absent from sensory spiking; sensory cortex fires to fully predictable input; prediction acts by alpha/beta pre-inhibition of predicted channels.
  - S74: predictable stimuli give *lower* BOLD but are *more* detectable, so lower activity isn't less information (possibly sharpening).
  - S76: no early sensory areas in the fMRI meta-analysis of prediction.
- **Why it matters:** an artificial system should keep a full feature stream and modulate it with surprise-scaled gain, rather than replacing features with residuals.

### I-42. True residual errors exist where the mismatch is low-dimensional. *Supported.*
- Dopamine reward-prediction error; visuomotor mismatch in V1 (faster or slower optic flow than predicted). One number fits in one neuron (S72, S70).

### I-43. Two kinds of "prediction" act on different timescales: long-term statistics are amplified; violations of short-term expectations are amplified. *Tentative (synthesis of S71 and S72).*
- **Long-term:** local recurrent networks amplify patterns resembling natural input. A prior stored in recurrent weights (S71).
- **Short-term:** expected stimuli in the current context are suppressed (S74), and violations are amplified (S72).
- **Resolves** the apparent contradiction "amplify the expected" (S71) vs "amplify the unexpected" (S72).

### I-44. A broad modulatory signal combined with local selectivity produces specific effects. *Supported as a recurring motif.*
- **Instances:**
  - BTSP: a local eligibility trace × a global plateau signal (S61)
  - noradrenaline: synchronized across dorsal cortex, yet it drives prediction-error-specific plasticity (S70)
  - VIP and pulvinar: broadly responsive, yet the amplification they produce is feature-selective because it lands on sharply tuned neurons (S72)
  - evidence uncertainty encoded locally, prior uncertainty centrally (S85)
- **Why it matters:** a learning controller can stay coarse (a few channels) as long as plasticity is gated by local eligibility. In ML this is a three-factor learning rule (B21).

### I-45. One write-gate motif may be shared by hippocampus and cortex: novelty opens disinhibition, an instructive higher-order input arrives, a dendritic plateau forms, and a write occurs. *Tentative.*
- Hippocampus (S61): plateaus need escape from local inhibition (disinhibition recruited by novelty) plus instructive entorhinal input.
- Cortex (S72): VIP disinhibition plus higher-order thalamic (pulvinar) input can trigger plateaus and plasticity without somatic spikes.
- Goal-related disinhibition (⚠ S40).

### I-46. Precision (attention) can be implemented as divisive normalization. *Supported formally; Tentative as the brain's mechanism.*
- S83: the normalization model of attention and the predictive-coding precision term are interchangeable. Whether attention changes gain or selectivity depends on whether the attended and ignored features share neural populations.

### I-47. Precision works at several scales and lives in several places. *Supported (synthesis; individual links vary).*
- **Evidence precision is local:** encoded in the area processing the evidence (S85); implemented as gain/normalization (S83).
- **Prior precision is central:** insula, orbitofrontal cortex, putamen, amygdala (S85).
- **The global learning rate follows volatility:** behaviour (B42), formal model (B43), proposed noradrenergic role (S73).
- **Neuromodulators broadcast "learn now"** rather than precise per-feature precision (S70).
- **Reconciles** S73 (neuromodulators *are* precision) with S70 (too coarse): coarse modulators set the global dial; specific precision comes from local circuits.

### I-48. Prediction has two tiers: local and automatic vs global and structural. *Supported (S75; with S82, S83).*

| Local tier | Global tier |
|---|---|
| Adaptation, repetition suppression, local oddballs | Violations of learned structure (global oddballs) |
| Present everywhere, including sensory cortex | Sparse; prefrontal and other higher areas |
| Survives anaesthesia | Disappears under anaesthesia |
| Gain change (S83: SSA) | Can change selectivity; more accessible to awareness (S83) |
| Online metacognition: ERN, ACC (S82) | Offline metacognition: Pe, reportable (S82) |
| Non-declarative (priming; S26) | Declarative / working memory (S26, S27) |

### I-49. A default system runs, and error hands control to another. *Supported as a recurring pattern.*
- **Instances:**
  - S70 (speculation): layer 2/3 intervenes in layer 5 only when layer 5's predictions fail.
  - S74: the putamen's generic error increases perceptual-to-motor coupling.
  - B14: uncertainty-based arbitration between habit and model-based control.
  - S26: control shifts between memory systems.
  - S91: code-as-policy falls back and retries when its checks fail.

### I-50. Measured correlates are not mechanisms. Decodable is not the same as used. *Established as a methodological principle.*
- **Instances:**
  - S75: imaging tracks an area's inputs, while spikes track its outputs, and the two disagree about prediction errors.
  - S10: latent tokens looked like imagination but weren't used.
  - S75: global oddballs are decodable in PFC yet aren't its main representation.
  - S86: many low-dimensional embeddings "work" on the same data.
  - S80: some activity "during learning" tracks performance, not learning.
- **Consequence:** every claimed mechanism in the design needs an intervention test (ablate, perturb, substitute).

---

## Part 12. Self-monitoring, salience and control (new in v2)

### I-51. Self-monitoring is a separable system, and it can fail independently of the skill it monitors. *Supported.*
- **Evidence:**
  - S81: TMS to lateral PFC impairs metacognition while task performance is spared; anosognosia; preserved metamemory in early Alzheimer's disease.
  - S82: metacognitive *efficiency* (meta-d′/d′) is separable from task sensitivity.
- **ML consequence ("machine anosognosia"):** a network that forgets old tasks usually stays confident on them. A learner that forgets without *knowing* it has forgotten cannot trigger rebuilding, replay or fallback. *Speculative as a claim about ML systems; testable (`design.md` E18).*

### I-52. Monitoring has a fast online form and a slow offline form. *Supported (S82).*
- **Online:** automatic error and conflict detection (ACC, posterior medial frontal; ERN).
- **Offline:** reflective judgements and planning (anterior/lateral PFC; Pe for reportable errors).
- Offline metacognition appears more trainable and more strongly tied to learning outcomes.

### I-53. What transfers furthest may be the meta level: how to learn, not what was learned. *Tentative.*
- **Evidence:**
  - S82: confidence training on perception transferred to memory metacognition, whereas object-level executive-function training mostly transfers only near.
  - S80: learning to learn transferred to *unrelated* motor tasks; better general error detection is one proposed mechanism.
  - S79: most of the motor speed-up was sequence-general.
- **ML consequence:** put shared machinery for monitoring, learning-rate control and error evaluation *above* task modules, and share it across all of them.

### I-54. A salience system detects what matters and decides whether to maintain or switch the current task set. *Supported (S84), with a speculative extension.*
- **Evidence (S84):**
  - The ventral anterior insula and ACC respond to homeostatically relevant events regardless of valence (B56).
  - A neighbouring task-control network maintains task sets (B57).
  - Salience engages it to maintain the set or switch to a new one.
  - Noradrenaline and dopamine modulate the network.
- **It unifies the generic "something's off" hubs** seen elsewhere: insula/IFG for violation (S76), dACC + insula for online monitoring (S82), the putamen's generic error (S74), unsigned surprise (S72, S73).
- **Extension (Speculative):** a concrete mechanism for context discovery (I-21).

### I-55. Runaway loops can be maintained at the meta level. *Tentative.*
- S81 (B54): metacognitive therapy, which targets beliefs *about* worrying rather than the worries' content, outperformed CBT in one RCT.
- **ML consequence:** a replay or imagination regulator should control the *policy that allocates* processing, not only filter content.

### I-56. Error detection is hierarchical: generic comparators higher up, specialized mismatch lower down. *Supported.*
- **Evidence:**
  - S89: the supramarginal gyrus and medial PFC respond to any haptic mismatch; SI distinguishes shape from texture mismatches; feature-selective regions don't carry mismatch at all.
  - S75: global errors only in higher areas.
  - S76: insula/IFG for violation across modalities.

---

## Part 13. Transfer, savings and learning to learn (new in v2)

### I-57. Transfer works by attaching learned rules to a shared latent variable. Positive and negative transfer travel through the same channel. *Supported (one strong, multi-method causal study, S87; consistent with B39 and S11).*
- **Evidence (S87):**
  - Mice transferred a whisker rule to vision instantly when the spatial rule was preserved.
  - When the rule was reversed, they fell to or below chance and recovered **more slowly than mice with no relevant prior**.
  - Bimodal neurons with congruent spatial tuning carry the shared code.
  - Silencing area RL abolished *both* the benefit and the harm, so mice learned like naive animals.
  - Stimulating RL substituted for the visual stimulus.
  - A model with only readout synapses plastic reproduces all of this.
- **Why it matters:** it turns the transfer–interference trade-off (I-20) from an abstract dilemma into a concrete mechanism with a concrete test (ablate the shared latent).

### I-58. Shared latents are built before tasks, from correlated experience. *Supported (S87; consistent with S25, S86).*
- Aligned visual–tactile maps exist in task-naive mice. They are probably refined by Hebbian plasticity on correlated spontaneous and sensory activity.
- **The ground truth for cross-modal invariants is the shared physical latent** (location, force, timing), not an analytic mapping between sensor spaces. This corrects S86's overstatement.

### I-59. Where general and specific learning live depends on the domain. *Tentative (S79).*
- In motor-sequence learning, practice-specific learning is low-level (M1/S1 execution) and transferable learning is higher (sequence representations). In vision, the usual layout is the reverse.
- **Rule:** share at the level where task invariants live.

### I-60. Savings has at least three separable sources. *Supported in the motor literature (B44–B46; known here via S80 and memory).*
1. **A hidden slow state** that persists after washout (B44). This is storage without access (I-3).
2. **A remembered learning rate:** higher sensitivity to errors that have been seen before (B45). This links to volatility-tuned learning rates (I-47).
3. **A recallable explicit strategy or reinforced action** (B46). Declarative and cognitive.
- **Also:** ageing impairs initial learning but not savings (S80), so savings is not just "learning again".
- **Measuring savings is not new** (B28–B30). Decomposing and engineering it is the open contribution.

### I-61. Learning to learn is real and has a cost. Specialization is real and has a cost. *Supported.*
- **Evidence:**
  - S80: varied learning experience sped up learning of unrelated tasks but made movements less robust to perturbations.
  - B53 (via S81): taxi drivers' spatial-memory specialization came with worse learning of new visuospatial information.
  - S51: varying or increasing enrichment beats constant enrichment.
- **Reading (Speculative):** the brain sets its plasticity dial from how variable its history has been. Varied history → plastic and adaptable; constant history → stiff and robust.

### I-62. Novelty should allocate new capacity; familiarity should refine existing structure. *Tentative (synthesis).*
- **Evidence:**
  - ART (B40, via S77): a match above the vigilance threshold refines a category; a mismatch recruits a new one.
  - S64: allocation is partly separate from storage; familiar items get weaker writes.
  - S61: a plateau away from a place field creates a *new* field.
  - S16: remote recall recruits *new* neurons rather than rewriting the old engram.
- **Reconciles** predictive coding (error drives learning) with ART (match drives learning): **error drives *allocation*, match drives *refinement*.**

### I-63. Different invariants may use different cross-modal architectures. *Tentative.*
- Spatial (peri-personal) rules: a dedicated dorsal-stream hub (RL) is necessary and sufficient (S87).
- Object shape: no amodal area found with PET; only an insula–claustrum region specific to cross-modal matching (S88). The ventral visuo-haptic region (mentioned in S87) may have been missed.
- Cross-modal structure pervades cortex (S86, argument).

---

## Part 14. Consciousness, capacity and coarse-to-fine inference (new in v2)

### I-64. High-capacity statistical learning runs outside awareness. Learning arbitrary structure across seconds seems to need the limited-capacity conscious workspace. *Tentative (a convergent hypothesis).*
- **Evidence:**
  - Non-conscious: priming and implicit learning intact without awareness (S26, S77); local oddballs survive anaesthesia (S75); statistical learning is present at birth (S77).
  - Conscious: global-oddball decoding disappears under anaesthesia (S75); stable evaluative conditioning needs momentary contingency awareness (S77); declarative memory is fast and flexible but conscious (S26); working memory holds only 3–4 chunks (S27); reportable errors carry the Pe (S82).
- **Caveat:** "conscious" means different things across these sources (reportability, anaesthesia, masking).

### I-65. Perception runs coarse-to-fine: a vague early guess is refined toward a crisp interpretation. *Supported for vision (B52 via S78).*
- Orbitofrontal cortex sends a low-spatial-frequency "gist" before object recognition completes.
- **ML reading:** stay soft (high-temperature, probabilistic assignments) early and commit late. This is standard in mixture fitting and annealing (B60); worth using deliberately in memory allocation.

### I-66. The signal that drives exploration and model improvement is the rate of improvement, not the error itself. *Tentative (three independent framings).*
- Learning progress (I-10), compression progress (B38), and the "knowledge instinct" felt as a change in model fit (S78).
- Raw error is attracted to unlearnable noise; progress is not.

---

## Part 15. Cross-cutting principles from the v2 critique (new in v2)

### I-67. Counterweights are most needed where learning is not anchored to an external error signal. *Supported.*
- Supervised, error-corrected learning slows down by itself as error approaches zero.
- Self-referential loops have no such brake:
  - Hebbian amplification (S13, S71)
  - replay or imagination feeding itself (S19, S10)
  - training on one's own outputs (B34)
- Remedies:
  - inhibition and normalization
  - **keeping real data in the mix** (B35)
  - **tagging what is generated**, i.e. reality monitoring (I-28)

### I-68. Copy the function, not the workaround. *Speculative (methodological).*
- Many biological mechanisms solve *biological* constraints: limited synapses, no addressable memory, energy budgets, the inability to learn and consolidate at once, and slow wiring.
- Binary synapses, 3–4 working-memory slots, sleep as a separate phase, and neurogenesis may be answers to problems a computer doesn't have.
- Example: if memory is cheap, storing raw or latent episodes may beat a compressed binary associative memory. The BTSP store's advantages are clearest under neuromorphic constraints.
- **Constraint audit for each component:** which function is being copied, and does the original constraint apply?

### I-69. A bypassed module survives only if it keeps an objective of its own. *Speculative.*
- In S10 and S50 the bypassed pathway degrades. In Packard's rats (S26) the bypassed hippocampal knowledge survives.
- The difference may be whether the bypassed system keeps learning by its own rules.
- **Design consequence:** archived modules may need a local self-supervised objective ("silent learning", `design.md` E10).

### I-70. Forgetting can be a feature. *Supported (literature from memory, B37).*
- Active forgetting supports generalization and flexibility (B37).
- Archive-not-delete is a partial answer. Deciding *when to let go* deserves its own policy.

### I-71. Whether forgetting was a loss of access or a loss of storage predicts how much savings to expect. *Speculative, cheap to test.*
- If a probe on frozen features still decodes the old task, forgetting was mainly a loss of **access** (readout or gating). Savings should be high and re-routing cheap.
- If not, **storage** was overwritten, and savings should be low.
- B31 suggests access loss is common in ANNs.
- Test: `design.md` E0.
