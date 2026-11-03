# ewk.md (v2): Everything Worth Knowing

**What changed from v1.**
- Entries S01–S69 are kept. Where v1 contained an error or an overstated limit, the entry now carries a **v2 correction** paragraph. The original text is left in place so the change is visible.
- The glossary gained the terms needed for the new sources (precision, oddballs, normalization, metacognitive sensitivity, and others).
- New entries **S70–S91** (sections I–N) were added, written to the same template.
- A new section **H** explains the most important background references (savings models, ART, metaplasticity, model collapse and others). They come from memory unless stated, so every claim there is marked accordingly.

This file explains each source listed in `papers.md`. For every source it covers:

- **Question.** What the authors were trying to find out.
- **What they did.** The method, in plain language.
- **What they found.** The results, with numbers where the source gave them.
- **Their explanation.** How the authors themselves interpret the result.
- **Limits.** Why the result might be weaker, or narrower, than it sounds.

Molecular and biochemical detail is left out unless it changes how a finding should be understood.

Two rules for reading this file:

1. **Access level matters.** Each source is tagged FULL, PART, EXCERPT, ABS or PRESS (defined in `papers.md`). If a source is tagged PRESS or ABS, everything here comes from that press release or abstract, and the study itself may contain more or different detail.
2. **"Their explanation" is not "the truth."** Where I think an interpretation is weak, that goes under *Limits*.

How these findings combine across sources is in `insights.md`. What could be built from them is in `design.md`.

---

## Glossary (read this first)

- **Catastrophic forgetting.** Training a network on a new task wrecks its performance on old tasks, because the weights that served the old tasks get overwritten.
- **Continual learning.** Learning a sequence of tasks or a changing data stream without forgetting earlier ones.
- **Task-incremental (Task-IL) vs class-incremental (Class-IL).** In Task-IL the system is told which task it is doing at test time, which is easy. In Class-IL it must pick among all classes ever seen without being told the task, which is hard.
- **Replay (rehearsal).** Mixing stored or regenerated old examples into training on new data.
- **Generative replay.** Replay where a generative model produces "pseudo-examples" of old data instead of storing real ones.
- **Hippocampus.** A brain structure crucial for quickly forming memories of events and places.
- **Neocortex (cortex).** The large outer brain sheet that learns slowly and holds long-term knowledge and skills.
- **Complementary Learning Systems (CLS).** The theory that the hippocampus learns fast and the cortex learns slowly, and that the hippocampus "teaches" the cortex by replaying memories.
- **Sharp-wave ripples (SWRs).** Brief bursts of high-frequency activity in the hippocampus during rest and sleep. Replay of recent experience happens during them, compressed in time.
- **NREM / SWS / REM.** Non-REM sleep, whose deepest stage is slow-wave sleep (SWS), versus rapid-eye-movement (REM) sleep, the stage most associated with vivid dreams.
- **Engram.** The physical trace of a memory: the set of neurons and synapses that hold it.
- **Place cell.** A hippocampal neuron that fires when the animal is in a particular location.
- **Grid cell.** An entorhinal neuron that fires at many locations arranged in a triangular grid, acting like a coordinate system.
- **LTP / LTD.** Long-term potentiation and long-term depression: lasting strengthening or weakening of a synapse.
- **Hebbian plasticity.** "Cells that fire together wire together": a synapse strengthens when the neurons on both sides are active together within milliseconds.
- **Homeostatic plasticity.** Slow self-correcting adjustments that keep a neuron's overall activity within a healthy range. For example, *synaptic scaling* multiplies all of a neuron's input weights up or down together.
- **Dendrite.** A neuron's branching input structure. Synapses land on dendrites.
- **Dendritic plateau potential.** A long (tens to hundreds of milliseconds) electrical event in a dendrite, which makes the cell body fire a burst.
- **Eligibility trace.** A temporary, decaying "tag" left at a synapse that was recently active. It marks the synapse as available for change if a later teaching signal arrives.
- **Neuromodulators.** Chemicals that adjust how plastic or how active large brain areas are, without carrying specific content. Examples: dopamine, acetylcholine, noradrenaline.
- **TMS.** Transcranial magnetic stimulation: a magnetic pulse through the skull that activates or disrupts a cortical area.
- **fMRI / EEG / MEG.** Brain-imaging methods. fMRI measures blood-flow changes (slow, but precise in space). EEG and MEG measure electrical and magnetic activity (fast, but coarse in space).
- **Representational similarity analysis (RSA).** Comparing the *pattern* of brain activity across items, to ask whether items are represented as similar or different.

**Terms added in v2**

- **Savings.** Relearning a forgotten skill faster than it was first learned. Measured as (trials to learn originally) ÷ (trials to relearn), or as the head start at the beginning of relearning. This is the scientific name for the user's "pre-skill buffer".
- **Predictive processing / predictive coding.** The idea that the brain predicts its input and uses the mismatch (prediction error) to update. *Predictive processing* is the broad framework. *Predictive coding* is a specific algorithm in which lower areas send up only the residual (input minus prediction).
- **Prediction error (PE).** A signal that input differs from expectation. Three possible forms (S72):
  - a *residual*: the signed difference
  - an *amplified sensory response*: the stimulus itself, boosted in proportion to the surprise
  - a *general surprise signal*: "something changed", without content
- **Positive / negative prediction error.** More input than predicted / less input than predicted (for example, an expected stimulus that is omitted).
- **Precision.** How much weight a signal gets, formally the inverse of its variance. *Evidence precision* is trust in the input; *prior precision* is trust in the expectation. Psychosis and autism have both been described as precision imbalances.
- **Volatility.** How fast the statistics of the environment are changing. In Bayesian models (B42, B43), higher volatility should raise the learning rate.
- **Oddball paradigm.** A frequent "standard" stimulus is occasionally replaced by a rare "deviant".
  - *Local oddball:* violates a simple repetition (AAAB, where B is the oddball). Confounded with adaptation.
  - *Global oddball:* violates a learned pattern across sequences (after many AAAB trials, AAAA is the oddball even though it is a repetition). Separates prediction from adaptation.
- **Mismatch negativity (MMN).** An EEG/MEG response to deviant sounds in an oddball sequence.
- **Stimulus-specific adaptation (SSA) / repetition suppression.** Weaker responses to a repeated stimulus. A low-level effect that can mimic prediction.
- **Divisive normalization.** A canonical neural computation: each neuron's response is divided by the pooled activity of a neighbourhood. The normalization model of attention (B58) uses it to explain attentional gain.
- **Gain vs selectivity.** A *gain* change multiplies all responses by the same factor. A *selectivity* change alters the shape of tuning (some responses boosted more than others).
- **Inhibition-stabilized network (ISN) / balanced state.** A regime where excitatory–excitatory connections are strong enough to be unstable on their own, and fast inhibition keeps the network stable. It allows fast, selective amplification of input patterns.
- **Interneuron classes.** *PV* (parvalbumin) cells give fast perisomatic inhibition. *SOM* (somatostatin) cells inhibit dendrites. *VIP* cells inhibit SOM cells, so activating VIP *disinhibits* pyramidal neurons.
- **Pulvinar.** A higher-order thalamic nucleus that exchanges signals with many cortical areas.
- **Cortical layers.** Layer 2/3 (superficial) neurons project mostly to other cortical areas (feedforward). Layer 5 (deep) neurons are the main output to subcortical targets and send long-range feedback.
- **Efference copy / corollary discharge.** A copy of a motor command sent to sensory areas, used to predict the sensory consequences of one's own movement.
- **Salience network.** Anterior cingulate (ACC) plus ventral anterior insula ("frontoinsula") and subcortical partners. Responds to anything homeostatically relevant, positive or negative.
- **Metacognition.** Monitoring and controlling one's own cognition.
  - *Meta-knowledge:* monitoring, flowing upward (e.g. confidence).
  - *Meta-control:* regulation, flowing downward (e.g. choosing a strategy, setting a reminder).
  - *Metacognitive sensitivity (meta-d′):* how well confidence separates correct from incorrect answers.
  - *Metacognitive bias:* overall confidence level.
  - *Metacognitive efficiency:* meta-d′ ÷ d′, i.e. sensitivity corrected for task performance.
- **Anosognosia.** Unawareness of one's own deficit (for example, after a right-hemisphere stroke).
- **Adaptive Resonance Theory (ART).** A family of neural models (Grossberg; Carpenter & Grossberg) in which input that matches a stored category well enough (a "vigilance" threshold) refines that category, while a mismatch resets and recruits a new one.
- **Cross-modal generalization.** Applying something learned through one sense (e.g. touch) to another (e.g. vision).
- **Hyperalignment.** Aligning brains by function (shared response patterns) rather than anatomy, so data from different people can be compared in a common space.
- **Tetanus toxin light chain (TeNT).** Expressed in neurons to block their transmitter release: a chronic, cell-targeted silencing method.

---

## A. Continual learning and replay in machine learning

### S01. "Replay Can Provably Increase Forgetting" (continual linear regression)
**Access: EXCERPT.**

**Question.** Replay is widely used to reduce forgetting. Can it ever *increase* forgetting, even in a very simple, friendly setting?

**What they did.**
- **Setting.** A sequence of linear regression tasks, with no noise, where *one* solution solves all tasks at once.
  - Each task has fewer samples than the input dimension, so a single task doesn't pin down the solution.
  - Training with gradient descent on each task in turn gives the solution closest to the previous weights (minimum-distance updates).
  - With this setup, the error after each task is a projection of the previous error onto that task's "null space" (the directions the task doesn't constrain).
- **Replay.** A few stored samples from earlier tasks are mixed into training on the current task.
- **Forgetting.** The average error on previous tasks after all training.
- **Two theorems:**
  - a worst-case one, where the replay samples are chosen adversarially
  - an average-case one, where samples are random draws from task subspaces and replay samples are chosen at random
- **Experiments:**
  - linear models and one-hidden-layer networks on constructed tasks
  - Rotated MNIST (task 2 = rotated digits)
  - two Split-MNIST orderings: (0 vs 1 → 6 vs 7) and (0 vs 6 → 1 vs 7)

**What they found.**
1. **Worst case.** Without replay, forgetting can vanish as tasks accumulate (it shrinks like 1/T). Replaying one carefully chosen sample makes forgetting **catastrophic**: it stays at a constant level and never vanishes.
2. **Average case.** There exist pairs of tasks where replaying *randomly chosen* samples increases forgetting *on average*. This can hold even when the number of replay samples is a sizeable fraction of the input dimension.
3. **When replay is guaranteed safe.**
   - If forgetting without replay is already zero.
   - If the smallest angle between the tasks' null spaces is larger than π/4.
   - Forgetting is worst at *intermediate* angles between tasks, not when tasks are very similar or very different.
4. **Experiments.**
   - Forgetting first rises, then falls, as the number of replay samples increases.
   - In Rotated MNIST, replaying a single "4" reduced forgetting by about 3% for 45° rotation, while a single "5" did nothing on average and often made it worse.
   - In Split MNIST, replay always helped for (0/1 → 6/7). For (0/6 → 1/7), 1–2 replay samples *increased* forgetting (statistically significant), and only 4 or more helped.

**Their explanation.**
- When a replayed sample's error is reduced, the update moves the weights in a direction that also changes the error on other stored samples that overlap with it. Replay *moves* error; it doesn't simply remove it.
- In the three-sample construction:
  - x1 and x3 are orthogonal (they don't overlap); x2 overlaps both.
  - Training on x3 creates error on x2.
  - Replaying x2 to fix that error pushes error onto x1.
  - If x1 appears in nearly every task, the total forgetting becomes large.
- **Interference between samples of the *same* task** matters. Without replay, all of a task's samples are fitted together, so they can't hurt each other. With replay, only a few are revisited, so they can.

**Limits.**
- The theory covers over-parameterised linear regression trained to zero loss on each task, with minimum-distance gradient-descent solutions.
- Replay is mixed only into the next task.
- The neural-network and MNIST effects are small, a few percent.
- It shows replay *can* hurt, not that it usually does. In practice, large-scale replay is still one of the most reliable continual-learning methods (S07).
- It measures forgetting, **not how fast an old task is relearned**.

**v2 corrections and clarifications.**
1. **Where the danger zone actually is.** The excerpt says replay is *guaranteed safe* when the smallest angle between the tasks' null spaces is **larger than π/4**. So the harmful zone is at angles *below* π/4. v1 elsewhere (I-2, E1) placed the worst case "around π/4", which is the edge of the safe region, not its centre. Any experiment meant to reproduce the harm must use angles below π/4.
2. **An ambiguity to resolve before relying on the angle result.** "Forgetting is worst at intermediate angles" might describe forgetting *without* replay (a known result in this setting, B32) rather than the *extra* forgetting caused by replay. The excerpt does not make this unambiguous. Check the full paper.
3. **Why the harm is real but small in practice.** The comparison is against no-replay forgetting, which in this setting is unusually benign: minimum-distance updates keep the solution close to old ones. In deep networks, no-replay forgetting is usually catastrophic, so replay's occasional extra harm is small relative to its benefit. This explains why S01 and S06 can both be right.
4. **The setting has no weight decay and no normalization layers.** Both of those add forgetting that depends on time or step count (see the S03 correction).

---

### S02. CORE: COgnitive REplay
**Access: EXCERPT.**

**Question.** Most replay methods treat every task and every sample equally. Can buffer space be allocated more intelligently, the way people review material?

**What they did.**
- **After each task:**
  1. Measure each old task's **forgetting rate**: its best past accuracy minus its current accuracy.
  2. Estimate the newest task's future vulnerability from how much it hurt older tasks. This is an **interference rate**, computed as a softmax over accuracy drops.
  3. Convert each into an **attention** value, att = −log(1 − f).
  4. Softmax-normalise the attention values.
  5. Tasks below a threshold get a guaranteed minimum share of the buffer ("spaced repetition"). Tasks above it share the rest in proportion to their attention ("targeted recall").
  6. Fill each task's share with "representative" samples ("quality-focused data selection").
- **Framing.** The authors map this onto human memory:
  - "recall-based" memory (hippocampal, orthogonal representations, so forgetting comes from capacity overload)
  - "familiarity-based" memory (overlapping representations, so forgetting comes from interference)

**What they found.**
- **Average accuracy** on split-CIFAR10 was 37.95%, 6.52 points above the best baseline.
- **Worst-task accuracy** was 6.30 points better.

**Their explanation.** Reviewing more of what is being forgotten, while keeping a minimum review of everything, makes the buffer more useful.

**Limits.**
- **It reacts after the fact.** It responds to forgetting that has already happened.
- **Interference is measured coarsely,** at the task level through accuracy drops. That can't see the sample-level error transfer that S01 describes.
- **The formula blows up** as f approaches 1, because −log(1 − f) goes to infinity.
- **The cognitive framing doesn't drive the maths.** The recall/familiarity distinction isn't what produces the equations.
- **The absolute accuracy is low.**
- **"First to use minimum task accuracy"** is a strong novelty claim that I would doubt.
- **It conflicts with S01.** Giving more replay to the tasks that forget most is exactly what can hurt between moderately similar tasks. The guaranteed minimum share may protect against this by accident.

---

### S03. MSSR: Memory-Inspired Sampler and Scheduler Replay (LLM continual fine-tuning)
**Access: EXCERPT.**

**Question.** Existing ways of deciding when to replay all have drawbacks:
- fixed intervals ignore how training is going
- loss-triggered replay is set off by noise
- accuracy-triggered replay lags and needs expensive evaluation

Can a forgetting-curve model schedule replay better?

**What they did.**
- **A memory model for each sample:**
  - a **memory strength** m that decays over training steps, faster when the sample's loss is high and slower when its **stability** S is high
  - reviewing a sample resets m to 1
  - S grows on each review, and grows more when the review came late (when m had dropped further), with saturation and noise
- **Sampling.** Samples are drawn in proportion to m^(−ζ), so weak memories are replayed more.
- **Scheduling.** Replay sessions get further apart over time (expanding intervals), and the replay ratio decays toward a minimum.
- **Setup.** LoRA fine-tuning of Qwen2.5-7B, Gemma2-9B, Llama-3.1-8B and Mistral-7B. Two task sequences:
  - 3 tasks: Alpaca → GSM8K → Competition Math
  - 11 tasks: AGNews, SQuAD, SciQ, BoolQ, ARC, MATH subsets
- **Baselines:** no replay, fixed replay, loss-based replay, accuracy-based replay.

**What they found.**
- The full method was best on most tasks and models.
- The sampler-only and scheduler-only variants came second.
- Accuracy-based replay was competitive but costly.
- Gains were largest on ARC, up to +0.108, and on early tasks in the long sequence.

**Their explanation.** Modelling when each memory weakens lets replay happen before forgetting, which gives a more stable retention curve. They say explicitly that the Ebbinghaus forgetting curve is used as a heuristic, not as a literal model of LLM forgetting.

**Limits.**
- **Memory strength is modelled, not measured.** It decays with training steps along an assumed curve. Networks don't forget because time passes; they forget because new gradients interfere (see `insights.md` I-2). A sample whose features are orthogonal to the new task can sit untouched for any number of steps without being forgotten.
- **It is close to replaying by error.** Low m is largely high loss, and S01 shows that error-driven selection can pick exactly the harmful samples. There is no interference check.
- **It has about 14 hyperparameters**, and robustness across random seeds isn't shown in the excerpt.
- **The setting is mild.** LoRA fine-tuning already forgets relatively little.

**v2 correction to the first limit.** "Networks don't forget because time passes" is too strong. Three mechanisms make artificial networks forget with elapsed training steps even without interfering gradients:
- **weight decay**, which shrinks every weight a little at every step, including weights no current task uses
- **drifting normalization statistics** (e.g. batch-norm running averages), which change the function applied to old inputs
- **noise-driven drift** along directions the current loss doesn't constrain (B33)

So MSSR's time-based decay is not purely a scheduling convenience. What remains true is that interference is the *dominant* cause in most continual-learning settings, and a sample whose features are untouched by new gradients is not forgotten *if* none of the three mechanisms above is present.

---

### S04. TD-MCL (brain-development-inspired spiking network)
**Access: EXCERPT.**

**Question.** Children learn perception, then movement, then interaction with objects, while their brain connections first grow and are later pruned. Can that developmental schedule give continual learning across very different tasks, with no replay, regularisation or freezing, and a shrinking network?

**What they did.**
- **Network.** A spiking ResNet-18. Each new task adds a new module (extra channels per block).
- **Reuse of old modules.** An online evolutionary search chooses which old task modules feed into the new one ("long-range connections"), with about 50% left unconnected. Connections that are rarely chosen but perform well get more probability.
- **Pruning.** During later tasks, local connections in old modules are weakened or removed. A synapse's threshold depends on two things:
  - local Hebbian co-activity
  - how often that module is reused by later tasks ("global generality")
- **Tasks.** 9 tasks in order:
  - perception: sketch, cartoon and photo classification
  - motor control: DeepMind Control reacher/stand/walk-style tasks, learned by imitating demonstrations
  - interaction: Meta-World button press, drawer, hammer

**What they found.**
- **Old tasks stayed mostly intact as their modules shrank.** For example, tasks 1, 2, 3 and 6 were roughly unchanged at 60–78% of their peak parameter count. Tasks 4, 5 and 7 lost a little (task 5 lost 1.93%).
- **Visual-perception comparison:** 68.88% average accuracy, 6.15 points above the next method (SOR-SNN).
- **Fine-tuning pruned modules improved them.** Task 4 rose from 45% to 70%.
- **Learning in sequence beat learning each task alone.** Arm motion: 70.3% vs 29.9% for direct training.
- **Network size** grew, then shrank, then stabilised.
- **The evolved reuse made sense.** The finger task connected mainly to the arm and standing tasks and avoided the photo task.

**Their explanation.** Long-range connections let new tasks reuse useful old knowledge (positive transfer). Feedback-guided pruning removes redundancy from old modules without forgetting. This parallels brain development.

**Limits.**
- **"No freezing" is mostly wording.** Old modules appear not to be retrained, which makes this parameter isolation, like Progressive Neural Networks plus pruning. Whether gradients flow into old modules isn't stated in the excerpt.
- **The setting is the easy one.** Each task has its own path through the network (task-incremental).
- **Network size shrinks only relative to each module's peak.** Every task still adds a module, and 9 tasks is too few to show scaling.
- **Pruning is where the forgetting comes from.** It deletes task-specific knowledge.
- **The baselines are weak.** Fine-tuning gains come from extra training the baselines didn't get. The direct-training baseline starts from scratch. Motor tasks are imitation, not reinforcement learning.
- **No seed variance is shown.**

---

### S05. Kim (2025). Re-analysis of "brain-inspired replay"
**Access: FULL.**

**Question.** In S06's brain-inspired replay model, internal (hidden-layer) replay was the most important part. What does it actually do?

**What they did.**
- Compared S06's model **with and without internal replay**, each **with and without Synaptic Intelligence** (SI, a method that protects important weights).
- Benchmark: class-incremental CIFAR-100, 10 tasks.
- Measured:
  - initial and final accuracy per task, retention ratio and forgetting score
  - log-likelihood and reconstruction error of the generative model
  - silhouette scores (how well clusters separate) and UMAP plots of hidden representations

**What they found.**
- **Internal replay, and especially internal replay with SI, reduced forgetting**, giving the best retention.
- **But it lowered accuracy when each task was first learned.** Initial accuracy was best for the model without internal replay or SI.
- **Internal replay improved the generative model's fit:** higher log-likelihood and lower reconstruction error.
- **All models had near-zero silhouette scores**, meaning clusters overlapped heavily in the hidden layer.

**Their explanation.**
- Internal replay and SI trade plasticity for stability.
- They claim internal replay "increases representational overlap," which would limit task-specific separation.

**Limits.**
- It's a single-author report with one benchmark and no reported number of seeds.
- **The overlap claim isn't supported by their own data.** All models had near-zero silhouette scores, whether or not they used internal replay. The data show overlap everywhere, not overlap caused by internal replay.
- **The useful takeaway:** report initial (plastic) accuracy as well as forgetting.

---

### S06. van de Ven, Siegelmann & Tolias (2020). Brain-inspired replay
**Access: PART.**

**Question.** Generative replay avoids storing data, but it fails to scale to many tasks or to natural images. Can brain-inspired changes fix that?

**What they did.**
- Split MNIST in Task-IL and Class-IL versions; permuted MNIST with 100 tasks; split CIFAR-100.
- **Compared:** generative replay (GR) with a separate variational autoencoder, EWC, SI, Learning-without-Forgetting (LwF) and context gating (XdG).
- **Tested how efficient GR is:** how many replayed samples it needs, and how good the generator has to be.
- **Control experiment:** re-initialising the network before each task, so old tasks had to be relearned from scratch using replay alone.
- **Proposed "brain-inspired replay" (BI-R):**
  1. **Replay through feedback.** The generator is merged into the main model as feedback connections.
  2. **Conditional replay.** A Gaussian-mixture prior with one mode per class, so specific classes can be generated.
  3. **Gating by internal context.** Different units are silenced depending on the class being generated, in the decoder only.
  4. **Internal replay.** Hidden representations are replayed rather than pixels.
  5. **Distillation.** Soft labels for replayed samples.

**What they found.**
- **In class-incremental split MNIST, EWC and SI failed badly; only GR worked.** Some form of replay may be needed whenever classes from different episodes must be told apart.
- **Retaining is much cheaper than learning:**
  - **one** replayed sample per mini-batch of 128 new samples was still competitive (Task-IL) or better than all non-replay methods (Class-IL)
  - a tiny 10-unit generator that produced poor-quality samples only moderately hurt performance
  - in the re-initialisation control, where everything had to be relearned, the same small or low-quality replay was **not** enough
- **Plain GR collapsed** after about 15 permuted-MNIST tasks and failed on CIFAR-100, even in Task-IL.
- **BI-R** beat SI on 100 permuted-MNIST tasks, and BI-R + SI did better still. It gave reasonable class-incremental CIFAR-100 performance without storing data.

**Their explanation.**
- Learning something new is much harder than not forgetting it, so imperfect replay is enough to protect old knowledge.
- Brain-like features make replay scale:
  - the hippocampus sits on top of cortex, hence replay through feedback
  - recall can be controlled, hence conditional replay
  - context shapes processing, hence gating
  - memories aren't replayed down to the retina, hence internal replay

**Limits.**
- Class-IL CIFAR-100 accuracy is still low in absolute terms. Methods that store data do better.
- **Internal replay needs the early layers to stay unchanged.** The text says this directly. I recall (not verified here) that the CIFAR experiments used pre-trained, frozen convolutional layers.

**v2 note.** S06's frozen convolutional layers, S59's frozen CNN, S57's locked thalamic input zone and S87's "only the readout is plastic" model point the same way: **stable shared features plus a plastic readout** is a recurring answer to stability vs plasticity. In modern ML the same pattern (a frozen pretrained backbone with lightweight heads or adapters) is a very strong continual-learning baseline (B36).

---

### S07. Hayes et al. (2021). Replay in deep learning vs biology (review)
**Access: EXCERPT.**

**What it covers.** A systematic comparison of replay in the brain with replay in artificial networks, listing biological features that ML lacks.

**Key biological facts the authors summarise:**
- **Replay prioritises recent experience,** and its strength fades over hours.
- **Replay is selective.** The brain favours:
  - weakly learned items (Schapiro 2018)
  - rewarding or frightening experiences
  - old memories that overlap with new learning (McClelland 2020)
- **Replay is partial.** Only fragments are replayed, which can be blended, supporting generalisation.
- **Replay is not an exact copy.** It includes shortcuts never actually taken and possible future paths.
- **Reverse replay** happens and is linked to reward.
- **Replay happens in many regions,** coordinated across hippocampus and cortex by nested sleep rhythms (ripples, spindles, slow oscillations).
- **NREM and REM sleep do different jobs.** NREM is linked to transferring and storing recent memories; REM to organising, abstracting and preventing interference.
- **Sleep may re-assign which neurons hold which memory**, making memories less overlapping (González 2020).
- **Normalisation and synaptic down-scaling co-occur with replay** in biology. In ML, replay and regularisation are designed separately.

**Key ML facts the authors summarise:**
- **Replay is the most effective family** of continual-learning methods.
- **Storing hidden representations** (representational replay) is more biologically plausible than storing raw inputs.
- **Generative replay** avoids storage but is hard to train.
- **At large scale, uniform random replay is almost as good as clever selection** in supervised learning (Chaudhry 2018; Hayes 2020; Wu 2019).
- **In reinforcement learning, selection helps more.** Isele & Cosgun (2018) found that **selection by TD error or reward did not prevent forgetting**, while selection by distribution matching or state-space coverage worked about as well as an unlimited buffer.
- **MIR** (replaying the samples that an update would damage most) beats random replay in small-scale tests.

**Gaps the authors identify in ML:**
- partial replay
- temporally structured replay
- REM-like replay
- spontaneous self-generated replay
- removing old items from the buffer
- explicit wake and sleep phases
- tight coupling between replay and regularisation
- forward transfer

They also flag an open biological question: after consolidation, is the hippocampal trace erased, or kept but never used?

**Limits.** It was written in 2021, so it predates newer work.

---

### S08. Lin, Piao & Fung (2025). Unsupervised Hebbian learning from an AI perspective
**Access: FULL.**

**Question.** Do ML ideas such as epoch tuning, activation functions and pre-training help unsupervised Hebbian learning?

**What they did.**
- Used the Krotov & Hopfield Hebbian rule with winner-take-all competition and an optional anti-Hebbian term, on Fashion-MNIST.
- Trained a linear classifier on top of the learned features.
- Varied:
  - the number of Hebbian epochs
  - the anti-Hebbian strength Δ
  - the activation function
  - how auxiliary datasets (MNIST, KMNIST) were combined
- Used 3 random seeds.

**What they found.**
- **Unsupervised Hebbian learning overfits.** There is an optimal number of epochs, about 20, beyond which the features get worse. One epoch is not enough.
- **Anti-Hebbian learning cuts both ways.** It reaches good features sooner, but its performance **collapses** if trained too long.
- **Activation functions that keep negative values help.** Softplus was best, about +0.5 points test accuracy over ReLU.
- **Sequential pre-training on KMNIST before Fashion-MNIST hurt.**
- **Merging weights helped.** Training separately on MNIST and Fashion-MNIST, then mixing the weights at roughly 1:9 (α ≈ 0.1), raised test accuracy by about 0.9 points.

**Their explanation.** Hebbian learning shares problems with ANNs (overfitting, wasted negative values) and benefits differently from extra data: merging weights works where sequential pre-training doesn't.

**Limits.** One dataset, three seeds, small effects. Published in an MDPI journal.

---

### S09. BCPNN (Bayesian Confidence Propagation Neural Network) preprint
**Access: EXCERPT.**

**Question.** Can a network built from brain-like ingredients learn useful representations with no labels? The ingredients:
- unsupervised learning
- local Hebbian learning
- online, sample-by-sample updates
- sparse activity
- sparse, patchy connectivity

**What they did.**
- **Architecture.** Units are grouped into modules ("hypercolumns" made of "minicolumns"), with competition inside each module through a softmax.
- **Learning rule.**
  - Each synapse keeps running averages (exponential moving averages) of how often the input side is active, how often the output side is active, and how often both are active together.
  - The weight is log(joint ÷ product of the individual rates), which is the pointwise mutual information.
  - The bias is the log of the output unit's activity rate.
- **Structural plasticity.** Connections between modules are either **active** or **silent**. Silent connections keep updating their running averages but don't pass any signal. Every so often, for each module, the least useful active connection swaps with the most useful silent one, provided the silent one is better by a factor ρ. The number of incoming connections stays fixed.
- **Inputs** are encoded with Gaussian mixture models or difference-of-Gaussian filters.
- **Evaluation:** linear classifiers on top of the learned features, compared with RBMs, autoencoders, local PCA/ICA, Krotov-Hopfield, feedback alignment, target propagation and MLPs.

**What they found** (Discussion section). Performance was on par with other brain-like models and MLPs on standard benchmarks, it converged within a few epochs, and it ran on neuromorphic and spiking hardware.

**Their explanation.**
- The rule is Hebbian but derived from Bayesian inference, so it is **stable and bounded without extra constraints**.
- Dividing by the individual activity rates acts like homeostasis.
- Structural plasticity finds good receptive fields.

**Limits.**
- It's a preprint, not peer reviewed.
- Only one hidden layer.
- Evaluated only with a linear readout.
- The input encoding is not local.
- **Not tested on continual learning.** Exponential moving averages forget gradually, as old statistics wash out.

---

### S10. Li et al. (2026). Imagination helps visual reasoning, but not yet in latent space
**Access: PART.**

**Question.** Multimodal language models that "imagine" in hidden vectors (latent visual reasoning: Monet, LVR, Mirage) score well on benchmarks. Do the latent "imagination" tokens actually do anything?

**What they did.** Causal mediation analysis on the chain input X → latent tokens Z → answer Y:
1. Changed X and measured how much Z changed.
2. Intervened on Z (replaced it with a fixed tensor, Gaussian noise, or values near zero) and measured how much Y changed.
3. Probed whether Z alone could answer new questions about the same image.

They then built **CapImagine**: the model describes imagined visual changes (zooming, highlighting) **in text**, trained on the same Monet data after rewriting and filtering it down to 17k examples.

**What they found.**
1. **The input barely shapes the latent tokens.** Latent tokens were highly similar across very different inputs and even different tasks, and became more and more identical as reasoning went on (collapse). Ordinary text and image tokens didn't show this.
2. **The latent tokens barely affect the answer.** Even drastic changes to them shifted accuracy by about ±1% (V* even rose 0.5%). The exception was one strong intervention in one model, which caused repetitive output.
3. **The latent tokens carry little visual information.** Using them alone did worse than guessing from text.
4. **Writing the imagination out in text worked better.** CapImagine beat the latent methods, for example +4.0 points on HR-Bench-8K and +4.9 on MME-RealWorld-Lite compared with Monet. Ablations showed that both the rewriting and the filtering mattered.

**Their explanation.** The models learned a **shortcut around their own latent tokens**. The tokens act as soft prompts or placeholders, not as carriers of imagined content.

**Limits.**
- It's specific to current latent visual-reasoning methods.
- It doesn't show that latent imagination is impossible. World-model agents whose latents are trained to predict the future, and whose policies depend on those latents, do use them.
- CapImagine's improvement also comes partly from filtering the data. They ablated this.
- ICML acceptance is not verified.

**v2 note: the general lesson is broader than imagination.** What S10 actually shows is: **an intermediate representation that the training objective doesn't force the model to use will be bypassed.** The same pattern appears in S50 ("learned non-use" after stroke) and in S26 (Packard's rats, where the habit system takes control from the hippocampus). The difference matters:
- In S10 and learned non-use, the bypassed pathway degrades or collapses.
- In Packard's rats, the bypassed hippocampal knowledge **survived**.
- A plausible explanation **[inferred]**: a bypassed module survives only if it keeps an objective of its own. The hippocampus keeps learning by its own rules; S10's latent tokens had no strong objective of their own.

---

## B. Neuroscience of continual learning, replay, consolidation and memory

### S11. Flesch, Saxe & Summerfield (2023). Continual task learning in natural and artificial agents (review)
**Access: FULL.**

**Main points.**
1. **Two ways a network can learn: "rich" and "lazy."**
   - Networks started with **small** initial weights learn structured, low-dimensional representations shaped by the task ("rich").
   - Networks started with **large** initial weights mostly learn just the output weights on top of random high-dimensional features ("lazy").
   - Brain recordings show signs of both. For example, in monkey prefrontal cortex, 29% of neurons were active for one categorisation scheme but only 2% for both.
2. **In the human trees task, people separate tasks into orthogonal subspaces.**
   - In the task, people learn to plant trees in two "gardens," where only leaf density matters in one and only branch density in the other.
   - After learning in blocks, fMRI in parietal and dorsomedial prefrontal cortex showed the two relevant dimensions on **orthogonal planes**, matching rich-regime networks.
3. **Humans learn better in blocks; standard networks learn better interleaved.**
   - Standard networks forget catastrophically under blocked training and do best when tasks are interleaved.
   - Humans do **better** when trained in blocks, because blocking helps them apply the two category boundaries independently.
4. **Gating can protect old tasks.**
   - If the context input is forced to switch hidden units on for one task and off for the other (anti-correlated weights), a network can learn both tasks in blocks without forgetting.
   - Hebbian learning (Oja's rule) on the context inputs, combined with "sluggish" inputs that blend in recent trials, can produce this gating automatically. See S12.
5. **Lasting dendritic spines.** New spines formed while learning motor tasks persist for the animal's lifetime.
6. **Replay links to Complementary Learning Systems.** The authors connect replay to CLS and to its uses in ML: DQN's replay buffer, prioritised replay, generative replay.
7. **Transfer vs interference is the key unsolved problem.**
   - Strict separation of tasks removes interference but also removes transfer.
   - Their candidate answer is **compositional, factorised codes**: reusable building blocks that can be recombined. Example: a chef combining dough-making and sauce-making skills to make pizza.

**Limits.** The modelling evidence comes from toy tasks. The authors call the transfer/interference trade-off unsolved.

---

### S12. Flesch, Nagy, Saxe & Summerfield (2023). Hebbian context gating with sluggish task signals
**Access: PART.**

**Question.** What simple mechanisms would make a network (a) avoid forgetting under blocked training and (b) pay a cost under interleaved training, as humans do?

**What they did.**
- **Network.** A feedforward network with one hidden layer, fed Gaussian-blob images plus a **one-hot context cue**.
- **Two tasks.** Classify the blob by its x-position in one task and by its y-position in the other.
- **Two mechanisms added:**
  1. **"Sluggish" task units.** The task signal is an exponential moving average over recent trials.
  2. **A Hebbian step (Oja's rule) after every SGD update**, applied to the weights from the context cue into the hidden layer.
- **Comparison data.** Behaviour from 50 network runs per condition, compared with earlier human data (about 200 trials per task).

**What they found.**
- **A plain SGD network** forgot task 1 under blocked training and applied task 2's boundary to both tasks. Under interleaved training it learned both with orthogonal representations, and its context weights became anti-correlated.
- **Hand-set anti-correlated context weights** prevented forgetting under blocked training.
- **The Hebbian step learned that gating automatically.** Because the context signal is the main direction of variation in the input, Oja's rule pushes the two context weights to opposite signs. This partitions the hidden units and prevents forgetting.
- **Sluggishness made interleaved training costly.** The network slid toward a single shared diagonal boundary and showed a congruency effect (better on trials where both tasks give the same answer).
- **The full model reproduced the human results:** blocked > interleaved, a bigger congruency effect after interleaved training, and errors in estimating the boundary.

**Limits (important).**
- **The network is given the context cue.** The Hebbian step learns *how to gate* using that cue. It does **not** discover what the contexts are.
- Only two tasks. Oja's rule finds the single main direction of variation, which happens to be the context axis here; how this scales to many tasks is unclear.
- It needs a carefully balanced pair of learning rates.
- Forgetting grows as training blocks get longer.

---

### S13. Andrei et al. (2023). Rapid compensatory plasticity in monkey visual cortex
**Access: ABS plus the user's summary (which matches the abstract).**

**What they did.** Repeatedly activated excitatory neurons in macaque primary visual cortex (V1) with light (optogenetics). They recorded neural populations during waking and during rest.

**What they found.**
- **The strength of interactions between neurons (correlations) dropped across the population within minutes.**
- **This happened only while the monkeys were awake,** not during rest.
- **Firing rates stayed stable.** Only the correlation structure changed.

**Their explanation.** A network model in which excitation and inhibition balance each other reproduced the effect, with **plasticity of inhibitory-onto-excitatory synapses** responsible. The authors call this the first experimental evidence for fast homeostatic plasticity that works mainly during waking, stabilising neuronal interactions when the network is strongly co-activated. This fits the theoretical prediction that Hebbian learning needs fast compensating processes (B12).

**Limits.**
- The inhibitory mechanism comes from the model, not from direct measurement.
- The drive was artificial (optogenetic), not natural learning.
- Recordings were from V1 only.

---

### S14. Rudroff, Rainio & Klén (2024). Hippocampus-inspired approach to stability–plasticity (perspective)
**Access: FULL.**

**What it says.**
- Mostly a restatement of Complementary Learning Systems theory: fast hippocampus, slow cortex, replay during sleep.
- It proposes AI designs with dual learning rates, offline "sleep" phases and changing plasticity over time.

**The one new element: BARRs** (from Karaba et al., B11).
- In non-REM sleep, sharp-wave ripples replay and strengthen recent memories.
- **BARRs** ("barrages of action potentials") are synchronised bursts involving a subset of CA2 pyramidal cells and CCK+ basket cells (a type of inhibitory neuron).
- **During BARRs, the neurons and groups of neurons that were active in learning and reactivated in ripples are *inhibited*.**
- Reactivation strength rises at first, then returns to baseline over the course of sleep. Silencing the CCK+ cells during BARRs abolished that return to baseline, made the groups of neurons more synchronised, and **impaired memory consolidation**.

**Limits.**
- No experiments, no maths.
- The ML proposals are vague ("selective inhibition" is never turned into an algorithm).
- The Pareto-frontier figure is conceptual, with nothing derived to support it.
- It overstates how little humans forget.
- Use it as a pointer to Karaba et al. (B11).

---

### S15. Baron, Diba & Amarasingham (2026). Anti-Hebbian rules stabilise replay
**Access: PART.**

**Question.** Which synaptic plasticity rules, if switched on *during* offline replay, keep replay healthy?

**What they did.**
- Used a published spiking model of hippocampal area CA3 (Ecker et al., 2022): 8,000 excitatory neurons and 150 inhibitory basket cells. Place cells were trained on runs along a linear track, and the network then produced spontaneous replays during ripples.
- Ecker et al. had switched plasticity off during the offline phase. These authors switched it on and tested every combination of the four "kernel components" of spike-timing plasticity:
  - strengthening or weakening when the presynaptic neuron fires first (positive timing)
  - strengthening or weakening when the postsynaptic neuron fires first (negative timing)
- Simulations lasted 10–15 seconds.

**What they found.**

| Rule during replay | Result |
|---|---|
| Strengthening only (symmetric) | Synapses saturate, rigid attractors form, and the memory trace is corrupted |
| Weakening only | The memory trace is erased |
| Standard Hebbian timing rule | Replay drifts strongly to one direction, speeds up, and degrades |
| **Anti-Hebbian** (weaken when the presynaptic neuron fires first, strengthen when the postsynaptic neuron fires first) | **Replay stays bidirectional and speed stays stable**, with no saturation or erasure |

- **The negative-timing component sets replay speed:** strengthening there slows replay down, weakening speeds it up.

**Their explanation.** Anti-Hebbian depression may be what stabilises replay. This would explain why replay slows down in real animals, which has been observed experimentally.

**Limits.**
- One model, and very short simulations.
- A single memory (one track).
- The anti-Hebbian rule is "noncanonical": not established experimentally in CA3.

---

### S16. Lei et al. (2025). Recalling a remote memory builds a new hippocampal engram
**Access: FULL (main text).**

**Background.**
- **Systems consolidation.** Memories start out dependent on the hippocampus and become dependent on the cortex ("remote") over weeks.
- **Systems reconsolidation.** Recalling a remote memory brings the hippocampus back into play.
- **Open question:** does that re-engagement reuse the original hippocampal engram, or create a new one?

**What they did.**
- **Paradigm.** Contextual fear conditioning in mice, a remote recall test 21 days later, then a test the next day.
- **Methods:**
  - optogenetic silencing of the dentate gyrus (DG) or medial prefrontal cortex (mPFC) at specific moments
  - genetically encoded blocking of protein synthesis
  - a new **triple-event labelling method ("Fos-Tree")** to tag the neurons active at training, at recall and at test
  - live two-photon imaging of the same DG neurons over time
  - silencing and activating specific labelled groups of neurons (loss and gain of function)
  - X-ray irradiation or chemogenetic silencing to remove adult neurogenesis or young neurons
  - tracing and silencing of the mPFC → DG pathway, and testing whether DG and amygdala (BLA) need to be active at the same time

**What they found.**
1. **Remote recall re-engages the hippocampus.** It needs DG activity during recall and DG protein synthesis, but not mPFC protein synthesis. If DG is blocked during recall, the memory stays generalised and dependent on mPFC.
2. **The hippocampus recruits a new set of neurons.**
   - DG neurons active during **training** were rarely reactivated at remote recall or at the next day's test.
   - Neurons recruited **at recall** were reactivated at test, above chance.
   - Live imaging confirmed this.
   - At *recent* recall, by contrast, the training neurons *were* reactivated.
3. **The new neurons carry the memory.**
   - Silencing the neurons recruited at recall impaired the next-day memory. Silencing the training neurons had no effect.
   - Activating the neurons recruited at recall produced fear in a new context.
4. **The memory is updated with the current context.** Recalling it in a slightly changed context incorporated that context, so fear the next day was even higher in the changed context. This needed DG activity and protein synthesis.
5. **Adult neurogenesis is needed.** Removing it, or silencing 4–6-week-old newborn neurons, prevented reconsolidation and updating. Without neurogenesis, the **original training neurons were reactivated more**.
6. **Value is re-attached through coordination.** A sparse mPFC → DG projection is needed, and DG and amygdala must be able to be active at the same time. Inhibiting them alternately broke updating.

**Their explanation (the "reconstruction model").**
- Recalling a remote memory reconstructs a **new** hippocampal engram, combining the cortical gist with the current context.
- Neurogenesis helps silence the original engram so a new one can be recruited.
- mPFC coordinates the new contextual engram with the amygdala's value (fear) engram.
- This explains both **updating** remote memories and **recall-induced false memories** (the misinformation effect).

**Limits.**
- One paradigm (fear conditioning), male mice only.
- c-Fos is an indirect proxy for activity.
- "Remote" means 21 days.
- X-ray irradiation has side effects.
- **Adult neurogenesis in humans is disputed** (B20).
- The authors concede the original engram may still contribute in other ways.

---

### S17. Liu et al. (2025). Slow-wave sleep vs REM sleep in transforming memories
**Access: PART.**

**Question.** Over sleep, do memories lose item-specific detail while keeping category-level gist? Which sleep stage drives this?

**What they did.**
- 35 participants learned 96 word–picture pairs. Pictures came from 4 categories: animals, plants, electronic devices and vehicles.
- They slept overnight with EEG recording. Half the pairs were cued during slow-wave sleep with sounds (targeted memory reactivation, TMR).
- The next morning: cued recall, mental imagery with written reports, and recognition.
- **Analysis.** EEG representational similarity analysis (RSA) separated two levels:
  - **item-level** representation: same item vs other items in the same category
  - **category-level** representation: same category vs different categories
- Analysis focused on items remembered after sleep.

**What they found.**
- **Before sleep,** both item-level and category-level representations were present.
- **After sleep,** item-level representations were **not detectable**, while category-level representations persisted.
- **Patterns before and after sleep weren't directly similar.** The format of the memory changed.
- **The REM/SWS ratio predicted the shift:** more REM relative to SWS meant more loss of item detail and more gain of category structure. The SWS × REM product did not predict it, and neither stage alone did.
- **REM theta and beta power** went with transformation. **SWS slow-oscillation power** went the opposite way, with preservation.
- TMR made no difference.

**Their explanation.** SWS and REM play **different, opposing** roles. SWS preserves item-specific detail; REM transforms memories toward gist and integration. This contradicts the "sequential/complementary" view that the two stages work together.

**Limits.**
- 34–35 participants, with correlations between people.
- EEG RSA is a coarse measure.
- One night, mostly female participants.

---

### S18. Deak & Stickgold (2010). Sleep and cognition (review)
**Access: PART.**

**Key points.**
1. **Chronic sleep restriction** (4–6 hours a night for 14 days) degrades attention, working memory and throughput to the level of 1–2 nights of total sleep deprivation. Doctors in training on long shifts made 36% more serious errors.
2. **Procedural and perceptual skills seem to improve over sleep.**
   - A visual discrimination task improved only after sleep, correlating with early-night SWS and late-night REM.
   - Finger-tapping speed rose about 20% after sleep but not after an equal time awake (Walker 2002).
   - Motor adaptation was better after sleep.
3. **Declarative memories benefit from sleep.**
   - Early-night sleep, which is rich in SWS, helps word-pair recall more than late-night sleep.
   - **Sleep protects against interference.** After learning a conflicting list of pairs, recall was 76% after sleep vs 32% after wake (Ellenbogen 2006).
   - Emotional memories are preferentially kept.
4. **Insight and inference improve after sleep.**
   - 59% of people who slept discovered a hidden shortcut rule, vs 23% of those who stayed awake (Wagner 2004).
   - People were better at inferring relationships between items several steps apart (Ellenbogen 2007).

**Limits.**
- Dated (2010).
- **Later work argues the "offline motor gains after sleep" were partly artefacts** of fatigue and averaging (B19). Sleep *protecting* memories is better supported than sleep *improving* skills.

---

### S19. Paller, Creery & Schechtman (2021). Memory and sleep (review)
**Access: EXCERPT.**

**Key points.**
1. **Systems consolidation.**
   - The hippocampus lets the cortex store whole experiences.
   - Memories may become schematic and independent of the hippocampus, though the hippocampus may stay involved for episodic detail.
   - Damage produces temporally graded retrograde amnesia: recent memories are lost and older ones spared.
2. **Two hypotheses about how sleep helps memory, not mutually exclusive:**
   - **Active systems consolidation:** selective replay in non-REM sleep reshapes cortex.
   - **Synaptic homeostasis:** sleep scales synapses down globally, improving signal-to-noise.
3. **Nested sleep rhythms.**
   - Hippocampal ripples sit inside thalamocortical spindles, which ride on the up-states of cortical slow oscillations.
   - Disrupting ripples impairs memory; extending them helps.
   - Coupling between slow oscillations and spindles predicts overnight memory gains.
4. **Waking replay vs sleep replay.**
   - **Waking ripples** carry high-fidelity, local replay (useful for planning).
   - **Sleep ripples** carry noisier, widely distributed replay coordinated with slow oscillations (useful for generalisation).
5. **A cortex → hippocampus → cortex loop.** A partial cortical trace selects what gets reactivated. The hippocampus pattern-completes it. The whole memory is then reinstated across cortex.
6. **Targeted memory reactivation (TMR) works.**
   - Sounds or smells linked to learning, played during non-REM sleep, selectively improve the cued memories. A meta-analysis covered more than 90 experiments.
   - Rodent work showed tones bias which hippocampal place cells are reactivated.
   - **TMR works best for memories of intermediate strength:** strong enough to be reactivated accurately, but with room to improve.
   - Cues work best just after the refractory period following a spindle, about 3–6 seconds.
7. **Sleep helps problem solving.**
   - Puzzles cued during SWS were solved more often the next day (Sanders 2019).
   - Hidden-rule discovery improved after sleep (Wagner 2004).
   - Some studies found no benefit.
8. **Sleep and emotion.** Emotional memories get preferential consolidation, but findings on REM are mixed.
9. **Maladaptive sleep processing.**
   - Depression and PTSD may involve over-consolidation of negative memories.
   - In socially anxious adolescents, cueing negative memories made ratings more negative.
   - Rumination-like activity may happen during sleep.
10. **Some TMR effects failed to replicate,** such as the social-bias study (Humiston & Wamsley 2019).

**Limits.** It's a review, and the REM findings are mixed.

---

### S20. Pasted review: "Memory from a cognitive neuroscience perspective"
**Access: EXCERPT.**

**What it covers.**
- **Working memory:** Baddeley's model, the state-based model, persistent activity, and prefrontal gating.
- **Declarative vs non-declarative memory.**
- **LTP and LTD**, early and late phases, and the role of the gene-regulator CREB.
- **Synaptic tagging and capture (STC):** a weak event sets a temporary tag at a synapse. If proteins produced by a strong event nearby arrive within a time window, they "capture" the tag and the weak event becomes a lasting memory.
- **Behavioural tagging:** the same principle at the level of behaviour.
- **Emotional tagging** by the amygdala.
- **Retroactive enhancement:** weakly encoded items become better remembered when later paired with shock (Dunsmoor 2015).
- **Schema-dependent consolidation:** information that fits existing knowledge consolidates fast. The medial temporal lobe binds the parts; the medial prefrontal cortex integrates with schemas.
- **Acetylcholine switches modes:**
  - high acetylcholine (active waking, REM) = encoding mode
  - low acetylcholine (quiet waking, NREM) = consolidation mode
- **Theories of sleep:** the sequential vs dual-process hypotheses, active systems consolidation, and synaptic homeostasis.

**Limits.**
- Broad and textbook-like, with some imprecise claims. For example, it says stage N1 sleep is "responsible for initial encoding."
- Its value is the pointers above (STC, schema-dependent consolidation, acetylcholine mode switching), not new evidence.

---

### S21. Mujawar et al. (2021). Memory: mechanisms and assessment
**Access: FULL.**

A brief textbook overview covering:
- types of memory and the multi-store model
- forgetting through interference and decay
- amnesia and clinical tests

One relevant motif: the **CaMKII / PP1 "push-pull"** between strengthening and its counterweight. It adds nothing new for this project.

---

### S22. Bausch et al. (2026). Separate neurons for content and context in the human brain
**Access: PART.**

**Question.** How are "what" (items) and "in which context" combined at the level of single neurons in the human medial temporal lobe?

**What they did.**
- Recorded **3,109 neurons from 16 epilepsy patients**, across the amygdala, parahippocampal cortex, entorhinal cortex and hippocampus.
- **Task.** Each trial started with a question context (e.g. "Bigger?", "Last seen in real life?", "More expensive?/Older?", "Like better?", "Brighter?"). Two of four pictures followed, pre-selected because neurons responded to them. Patients then compared the two pictures according to the question.
- Neurons were classified by ANOVA as **stimulus**, **context** or **interaction** neurons.
- Decoders tested whether context and content generalised across each other.
- Cross-correlations tested timing relationships between neuron pairs.

**What they found.**
- **597 neurons responded to the picture.** Most (524) did **not** respond to context.
- **200 neurons responded to context:** 2.95% of amygdala, 7.68% of parahippocampal cortex, 5.68% of entorhinal cortex and 9.42% of hippocampal neurons.
- **Only 50 neurons (1.61%) were conjunctive,** responding to specific picture–question combinations.
- **The two codes generalise.** Context could be decoded across different pictures and positions, and pictures across different contexts.
- **Context is reinstated.** Activity from the question period was **reactivated** during the pictures, especially in the hippocampus.
- **Both codes lasted until the decision,** and were stronger when the decision was correct.
- **A new timing link formed during the task.** In 3 of 6 sessions (2 patients), entorhinal stimulus neurons predicted hippocampal context-neuron firing about 40 ms later. This link was absent before the task, appeared during it and persisted after.

**Their explanation.**
- Humans use largely **separate, orthogonal** populations for content and context, bound at the moment of use by co-firing and reinstatement, possibly through synaptic changes.
- This favours **flexible generalisation over rigid conjunctive coding.** That contrasts with rodents, where hippocampal coding is more context-dependent.

**Limits.**
- The patients have epilepsy.
- "Context" here means the task question, not the background environment.
- The pictures were pre-selected because neurons responded to them.
- Evidence for synaptic change is limited (3 sessions from 2 patients).

---

### S23. Preprint (2026). Brain state shapes awake reactivation and whether it helps memory
**Access: EXCERPT.**

**Question.** Is spontaneous replay during waking rest spread uniformly over time, or does it depend on the whole brain's ongoing state? And does that state decide whether replay helps memory?

**What they did.**
- Re-analysed an fMRI dataset (Tambini & D'Esposito 2020): object–face pairs were encoded, then TMS was applied (to lateral occipital cortex, LOC, or a control site), then 27 minutes of rest, then a delayed test.
- **Reactivation events** were time points where the activity patterns in both the hippocampus and LOC matched encoding templates (about 42 events per participant).
- **Whole-brain states.** Eight patterns, found by clustering baseline rest data, formed four pairs of opposite poles.
- Also measured slow fluctuations in the whole-brain global signal (its phase, and its amplitude in 1-minute windows), which have been linked to arousal.

**What they found.**
- **Reactivation happened preferentially during:**
  - an **inward-focused state**: default mode network up, dorsal attention network down. This state was *suppressed* during encoding.
  - states that **reinstated encoding-related visual activity**
  - the **peak** of the global signal and windows of **high global-signal amplitude**, both previously linked to lower arousal
- **Whether replay helped depended on the state it occurred in:**

| State during reactivation | Effect |
|---|---|
| Visual + limbic + default mode networks co-active, with ventral attention/salience and somatomotor networks suppressed ("4A") | **Less** forgetting |
| Mostly visual-network activity ("2A") | **More** forgetting |
| More reactivation during high global-signal amplitude windows | Better retention (r = −0.40). No relationship in low-amplitude windows |

- The overall prevalence of each state across the rest period did **not** predict memory. Only the state *during* reactivation did.
- **TMS to LOC changed how reactivation lined up with brain states,** but not how it lined up with the global signal.

**Their explanation.**
- Opportunities for consolidation fluctuate *within* rest.
- Reactivation may stabilise memories in some brain states and destabilise them in others. This could explain why awake reactivation and TMR results are inconsistent across studies.

**Limits.**
- An unreviewed preprint.
- Correlations between people.
- The global signal is not a specific measure of arousal.
- Many analyses were run.
- fMRI is slow, so "events" are coarse.

---

### S24. Eichenbaum et al. (1999). Spatial memory or a memory space?
**Access: EXCERPT.**

**Question.** Is the hippocampus a map of space (the "cognitive map" theory), or something more general?

**The evidence they review against a pure spatial map:**
- **Place fields aren't spread evenly** over space. They cluster near cues and walls.
- **Place fields aren't bound into one coherent map.** Different cells follow different subsets of cues at the same time, and changing the cues produces **partial** remapping (some cells keep their fields, others change).
- **Place cells depend on behaviour.** They fire differently on outward vs inward journeys and are tuned to speed and turning angle. "True" place cells that ignore behaviour appear mainly in random foraging, where location is the only regularity.
- **Many hippocampal neurons code non-spatial things:** odours, whether an item matches or doesn't, task phases, conditioned stimuli and responses. These codes are as robust as spatial ones. In Wood et al. (1999), most active cells coded odour or match status regardless of location.
- **Non-spatial coding appears even in tasks that don't need the hippocampus.**
- **Primate and human hippocampal cells** respond to faces, words and their combinations.

**Their alternative: a "memory space."**
- **Event codes:** specific combinations of stimulus, action and place, formed quickly by Hebbian learning within about 200 ms.
- **Sequence codes:** firing that spans successive events, linking them into an episode and telling similar episodes apart.
- **Nodal codes:** features shared across episodes, such as a place or an odour, linking *different* episodes. A "true place cell" is just a nodal code for a location.
- The whole network of linked episodes supports:
  - **flexible inference.** Rats can infer B > D from training on A > B, B > C, C > D, D > E, but not after the hippocampus is disconnected (Dusek & Eichenbaum 1997).
  - **shortcuts**, through overlapping episodes rather than metric map calculations.
- **Consolidation happens in stages:** hippocampus → parahippocampal region → neocortex, as sequence and nodal codes are gradually built in cortex through repeated reactivation. Its **duration depends on how many useful links a memory needs**: days for simple memories, possibly a lifetime for richly connected ones.
- **A testable prediction:** T-maze "splitter cells" that fire on the shared stem depending on which way the rat will turn. Later confirmed (Wood et al. 2000; Frank et al. 2000).

**Limits.** It's from 1999. Grid cells (2005) show that a metric-like map exists in the neighbouring entorhinal cortex, and goal-vector coding (S49) adds spatial content. The modern view is that the entorhinal cortex supplies structure and the hippocampus binds experiences onto it.

---

### S25. Baram, Donato & Holmes (2019). Building and disrupting spatial memory networks in development
**Access: EXCERPT.**

**Key points.**
1. **Hippocampal (declarative) memory is not innate.**
   - Egocentric, self-to-object spatial memory develops first and doesn't need the hippocampus.
   - **Allocentric, object-to-object memory** emerges at about **22 months** in children. In one study, 30% of 20–24-month-olds succeeded vs 84% of 25–39-month-olds (Ribordy 2013/2017). In rats it emerges around postnatal day 20.
   - Three-year-olds *form* episodic memories but don't *retain* them over a delay. Four-year-olds do. So childhood amnesia may be about retention, not formation.
2. **Rodent development order:**
   - place cells stable by about day 16
   - boundary cells by about day 17
   - grid cells by about day 21, adult-like within about a day of appearing
   - theta and gamma rhythms in the first postnatal week
   - sharp-wave ripples by the end of the second week
   - **trajectory sequences (theta sequences, replay) appear gradually after weaning.** Before that, replay reflects mostly stationary positions.
3. **The circuit is built by its own activity.**
   - Bursts in entorhinal cortex and hippocampus are triggered by **spontaneous body twitches**, like retinal waves in the visual system.
   - **Maturation spreads stage by stage** from entorhinal layer-2 stellate cells. Silencing those cells arrests maturation of the whole circuit downstream; silencing pyramidal cells doesn't (Donato 2017).
4. **There are critical periods.** Removing the memory-linked gene Arc in the first postnatal month permanently changes hippocampal rhythms and impairs adult spatial learning. Removing it later doesn't.
5. **Abnormal activity damages development.**
   - Prolonged febrile seizures in infancy (40–50 minutes in a rat model) cause adult spatial memory deficits, with no cell loss.
   - Instead there is dendritic loss, abnormal synapses, disrupted theta-phase coding and an altered gamma balance.
   - **Blocking the transcription repressor NRSF after the seizures prevented the deficits.**

**Limits.** A review that relies heavily on rodent data.

---

### S26. Squire & Dede (2015). Conscious and unconscious memory systems
**Access: EXCERPT.**

**The framework.** Several memory systems run **in parallel**, each with its own anatomy and principles:

| System | Brain basis | Character |
|---|---|---|
| **Declarative** (facts, events) | Hippocampus and nearby cortex, diencephalon | Fast (single episode), flexible, conscious, relational |
| **Working memory** | Separate from long-term memory; prefrontal and parietal | About 3–4 items |
| **Skills and habits** | Striatum | Gradual, feedback-driven, rigid, **ignores changes in reward value** |
| **Priming** | Neocortex | Unconscious, faster processing of repeated input with **reduced** cortical activity |
| **Perceptual learning** | Sensory cortex, including V1 | Gradual |
| **Delay eyeblink conditioning** | Cerebellum | Independent of awareness |
| **Emotional (fear) learning** | Amygdala | Also **modulates** the strength of both declarative and non-declarative memory through stress hormones |

**Key evidence.**
- **H.M.** learned mirror drawing over three days with no memory of having practised. Amnesic patients learn mirror-reading at a normal rate.
- **Priming is intact in amnesia** even when recognition is at chance.
- **Probabilistic classification ("weather prediction"):** hippocampal patients learn the task but can't report facts about it. Parkinson's patients (striatal damage) remember the facts but can't learn the task.
- **Bayley et al. (2005).** Severely amnesic patients learned 8 object pairs only after more than 1,000 trials (controls needed about 40). Their knowledge **collapsed when the format changed**, and each day they couldn't describe the task.
- **Packard & McGaugh (1996), the plus-maze:**
  - Early in training, rats use **place** learning (hippocampus). With extended training, they shift to a fixed **turn-left response** (caudate).
  - Inactivating the hippocampus early abolishes place learning.
  - **Inactivating the caudate late brings place behaviour back.** The hippocampal system had kept its knowledge the whole time.
- **Which system controls behaviour can shift.** Stress, ageing and drug history push toward the striatum. Prefrontal cortex may arbitrate.
- **Working memory vs long-term memory.** Hippocampal patients are intact for 1–3 object locations but fail sharply at 4, where capacity is exceeded and long-term memory is needed.

**Limits.** It's a review that relies on lesion studies. Some argue the hippocampus contributes to unaware memory too, and the authors discuss this.

---

### S27. Cowan (2014). Working memory, development, learning and education
**Access: EXCERPT.**

**Key points.**
1. **Capacity is about 3–4 chunks** in adults when rehearsal and chunking are blocked. Miller's "7 ± 2" reflects those strategies. Why the limit exists is unknown.
2. **The embedded-processes model:**
   - **activated long-term memory**: a large, loose mix of active features that decays within about 30 seconds unless refreshed
   - inside it, a **focus of attention** holding about 3–4 integrated chunks
   - attention is captured by **changes from the internal model**. Matching input habituates, and deviating input triggers orienting (Sokolov).
3. **Chunking and long-term working memory.**
   - The limit counts chunks, not items.
   - Example: runner SF reached about 80 digits by recoding them as running times, but stayed at about 7 for letters.
   - Ericsson & Kintsch: experts use retrieval structures in long-term memory to extend working memory.
4. **Associations form in the focus of attention.** Words that were in the focus together got linked without any intention to learn: 59% vs 53% accuracy for short vs long lists.
5. **Halford's relational complexity.** Understanding a concept requires holding its related elements at once: "bigger than" relates 3 elements, ratios relate 4. Rote procedures can get around this but don't build understanding.
6. **Maintenance takes active effort.**
   - Rehearsal and attentional refreshing are needed.
   - Cognitive load (the share of time spent on other processing) reduces recall roughly linearly (Barrouillet).
   - Better-consolidated items decay more slowly (Ricker & Cowan).
   - Whether working-memory forgetting is decay or interference is unresolved.
7. **Development.**
   - Capacity rises through childhood, and in Cowan's experiments this wasn't explained by knowledge, filtering, encoding speed or rehearsal (tentative).
   - Children under about 4 don't use attentional refreshing.
8. **Working-memory capacity predicts staying on task:** fewer Stroop lapses and less mind-wandering when people *want* to focus (Kane & Engle).
9. **Working-memory training mostly improves the trained task,** likely through task-specific strategies, and transfers little to reasoning (Melby-Lervåg & Hulme 2013; Redick 2013). This is still debated.
10. **Cognitive load theory:** separate intrinsic load from extraneous load. Too little load disengages learners.
11. **People don't use process of elimination in working memory** (Chen & Cowan), which resembles confirmation bias.

**Limits.** From 2014, with some citations "in press." Cowan himself notes that small method changes produce large differences in results.

---

### S28. Budson, Richman & Kensinger (2022). Consciousness as a memory system (theory)
**Access: EXCERPT.**

**The claim.**
- Consciousness evolved as part of **episodic memory**. Its original job was to bind experiences into memories and to **flexibly recombine** them to imagine and plan.
- Perceptions, decisions and actions are made **unconsciously and in parallel**. Conscious experience is a **memory** of them, delayed about 350–500 ms and back-dated.
- Their metaphor: an unconscious "horse" (System 1) acts; a slow conscious "rider" (System 2) advises.

**Evidence they use.**
- Postdictive illusions: the cutaneous rabbit, colour phi, colour fusion, and post-stimulus cues improving awareness (Sergent 2013).
- Chronostasis (the stopped-clock illusion).
- Libet's timing results.
- The Grey Walter slide-projector anecdote.
- Blindsight and visual apperceptive agnosia.
- How hard mindfulness is.
- They also argue every cortical region supports its own domain of consciousness.

**Limits.**
- No new data.
- **Libet's backward referral and the readiness-potential interpretation are contested** (B18).
- The Grey Walter story is second-hand, via Dennett.
- The fixed ~500 ms delay is an over-generalisation.
- "Every cortical region is autonomously conscious" and the claims about autism are speculation.
- **Useful parts:**
  - the functional idea that episodic memory exists for flexible recombination aimed at the future
  - the fast-acting / slow-advising split
  - postdiction (see `insights.md` I-24)

---

### S29. Memory theory of consciousness applied to sleep and dreams (perspective)
**Access: EXCERPT.**

**Key claims and the evidence cited.**
- **Sleep consolidation is massively parallel and unconscious.** Schechtman et al.: many memories can be reactivated at once in non-REM sleep without interfering with one another, which suggests the reactivation isn't conscious.
- **Dreams are a narrow, serial storyline** stitched from a small subset of the memories being reactivated. The authors suggest dreaming is a by-product of consolidation. This is untested; Zadra & Stickgold disagree.
- **After REM awakenings,** distant word associations prime more strongly. After non-REM, close associations do (Stickgold).
- **TMR works without conscious recall:** more than 90 experiments and over 2,000 participants. The Guitar Hero study (Antony 2012) is cited.
- **Sleep is a mode** that is poor at forming new memories and good at reorganising existing ones.
- **"Sleep amnesia":** we forget most of what happens during sleep.
- **Continuity on waking depends on episodic memory.** Patients with Alzheimer's who wake somewhere unfamiliar become disoriented.
- **"Local sleep":** different parts of the brain can be in different states at once.

**Limits.** A perspective piece. Whether dreams play any role in consolidation is explicitly unresolved.

---

## C. Imagination, observation and learning from simulated experience

### S30. Stefan et al. (2005). Formation of a motor memory by action observation
**Access: EXCERPT (Discussion only); identification inferred.**

**The method** (inferred from the Discussion).
- A TMS pulse over the thumb area of motor cortex makes the thumb twitch in a habitual direction.
- **Physically practising** thumb movements in a different direction shifts the twitch toward the practised direction. That shift is a marker of motor memory forming in M1.

**What they found.**
- **Just watching** thumb movements in a new direction shifted the twitch direction and the balance of muscle excitability toward the observed movement.
- The effect was **much smaller** than physical practice, but **the same in kind and specificity**.
- Watching movements in the baseline direction, or with no overall direction, did nothing.
- **Unintentional muscle activity** during watching was orders of magnitude smaller than during practice, so it couldn't explain the effect.
- **It didn't depend on what people were consciously doing** while watching. Counting the frequent movements and counting the rare ones gave similar results.

**Their explanation.** Mirror-neuron activity in ventral premotor cortex, passed on to M1 (or mirror activity within M1 itself), builds a motor memory trace from observation alone. Watching movement also activates somatosensory areas, which may supply the body signal motor memories normally need.

**Related cited evidence (B24).** Mattar & Gribble (2005): watching someone learn reaching in a novel force field made later learning of that task faster. The benefit disappeared when the watcher's motor system was busy with an unrelated movement task, but not under a cognitive distraction.

**Limits.**
- A simple single-thumb, single-direction test.
- The outcome measure is TMS twitch direction, not skilled performance.
- "Lasting" likely means minutes (unverified).
- Small sample.
- The mirror-neuron explanation is debated.

---

### S31. Dabas, Benoit et al. (2025). Learning from imagined experiences (via press release)
**Access: PRESS.**

**What they did** (from the press release).
- 50 participants listed 30 people they knew and ranked them from liked, to neutral, to disliked.
- In fMRI, for neutral people, they vividly imagined (for 8 seconds) either a positive or a negative encounter.

**What they found.**
- Participants came to **like the people they had imagined positive encounters with**, both during the task and afterwards.
- **Ventral striatum activity** (the classic reward-prediction-error region) was stronger on imagination trials with larger prediction errors.
- It worked together with **dorsomedial prefrontal cortex**, which stores knowledge about specific people. The authors interpret this as the stored knowledge being changed by imagination.
- **Imagining negative encounters did *not* reduce liking.** The authors don't yet know why.

**Their explanation.** Learning from imagined events uses much the same machinery as learning from real ones: reward prediction error updating stored knowledge. They describe imagination as "not passive."

**Limits.**
- Only the press release was read: no effect sizes, no controls.
- The content of the imaginations wasn't controlled.
- Expectation (demand) effects are possible.

---

### S32. Ye et al. (2024). Imagination builds new scenes and objects from memory (fMRI)
**Access: ABS.**

**What they did.**
- Participants learned four rooms. Each had five wall segments of different shapes, each paired with an object.
- In fMRI, they **retrieved** the associated objects or walls, then **imagined** either the two objects side by side or a scene combining the two walls.
- What they imagined was checked with same/different judgements.

**What they found.**
- Widespread activity in occipito-temporal cortex during object retrieval and scene imagination.
- **A classifier trained on either imagination data or retrieval data** could tell imagined scenes apart from imagined objects.

**Their explanation.** Different representations underlie scenes vs objects, and imagination recombines remembered elements.

**Limits.** Abstract only.

---

### S33. Zhang, Budson & Gutchess (2021/22). Self-imagination and memory in older adults and amnestic MCI
**Access: PART.**

**What they did.**
- 40 healthy older adults and 26 patients with amnestic mild cognitive impairment (aMCI).
- Each studied 90 object photos under two conditions:
  - imagining buying the object in a store (self-imagination)
  - judging whether it would fit in a shoebox (control)
- Recognition test after 15 minutes, with "similar" lures (same object name, different details) and new lures.

**What they found.**
- **Healthy older adults** recognised more in the self-imagination condition: corrected recognition .71 vs .66 against similar lures (p = .02), and marginal against new lures.
- **aMCI patients** showed **no reliable benefit**.

**Their explanation.** Self-imagination engages elaborative, self-referential processing that helps memory, but it relies on intact medial temporal function that aMCI degrades.

**Limits.** Small groups, a modest effect, and a shallow control task.

---

### S34. Lamb & Puryear (2026). Imagination and the creative process (systematic review)
**Access: PART.**

**What they did.** Reviewed 42 peer-reviewed studies from 2012–2024, mostly correlational, with convenience samples and educational settings, often from Taiwan and China. They looked at how imagination is defined and how it relates to creativity.

**The useful distinction.** Imagination involves:
- a **generative** phase: producing candidate ideas
- an **exploratory** phase: examining and judging them

Creativity additionally requires the result to be useful and intentional.

**Limits.** Low relevance to how the mechanism works.

---

### S35. Loren Frank interview (2017)
**Access: FULL.**

**Expert views.**
- **Memory supplies the building blocks of imagination.**
- **Sleep** may extract regularities across experiences: you know a watermelon's colour without remembering how you learned it.
- **During sharp-wave ripples,** replay includes possible futures and distant old experiences.
- **During movement,** theta sequences **alternate between options** (for example, left vs right).
- **Replay events briefly shut down ongoing processing** across the brain.
- **Plasticity is regulated** by inputs, circadian rhythms and neuromodulators.
- **Epilepsy** reflects plasticity that has run away and keeps reinforcing its own pattern.

**Limits.**
- It's an interview.
- His statement that motor skills "clearly" improve after sleep is contested (B19).

---

## D. Spatial memory, hippocampal coding, development

### S37. Maimon, Eliav, … Ulanovsky (2026). Sparse-to-dense coding from CA3 to CA1
**Access: ABS + intro.**

**What they did.** Recorded CA1 and CA3 place cells at the same time in bats flying through tunnels 6–200 m long.

**What they found.**
- In **small tunnels**, CA1 and CA3 coding looked the same.
- In **large tunnels**:
  - **CA1** was **dense**: many place fields per cell.
  - **CA3** was **ultrasparse**: mostly one field per cell.
  - Field *sizes* were similar in both.
- **A network model showed the sparse-to-dense transformation speeds up learning of new maps.**
- **Trajectory history** modulated place cells, with effects lasting more than 100 m.

**Their explanation.** CA3 → CA1 reformats spatial information into a more efficient, compressed code, combining fast learning with compact representation.

**Limits.** Only the abstract and introduction were read.

### S38–S49. Nature spatial-memory papers (listing summaries only; NOT read)
Each line below is the one-line summary on Nature's listing page. Read the paper before relying on any of them.

**v2 warning.** v1's `insights.md` cited S38 (drift), S40 and S41 (gated plasticity), S42 (recalibration), S43 (gaze-driven place codes), S44 (action-aligned assemblies), S45 (stable and dynamic engrams), S47 and S48 (non-spatial mapping, multiplexing) and S39 (trajectory error) as evidence. In v2, each such use is marked as resting on an unread source, and no principle depends on them alone.

- **S38 (Climer 2025).** Individual CA1 place cells drift over time even in stable environments with unchanged behaviour. More excitable cells drift less.
- **S39 (Brown 2026).** Dopamine across the striatum encodes **trajectory errors** (speed and direction of movement relative to the optimal path to the goal), separated in space and time from value coding.
- **S40 (Jeong 2025).** Parvalbumin inhibitory interneurons in CA3 **reduce firing** on the approach to and at goal locations while mice learn where food is.
- **S41 (Fisher 2022).** In fruit flies, dopamine increases head-direction plasticity during turning movements that bring in new spatial information.
- **S42 (Jayakumar 2019).** The gain of path integration (how movement is converted into position) is **not fixed**; it re-tunes quickly to match visual landmarks.
- **S43 (Payne 2025).** Chickadee place cells fire for locations the bird **looks at** as well as places it visits, coordinated by head saccades.
- **S44 (Zutshi 2025).** Internally generated hippocampal cell assemblies are updated by action plans.
- **S45 (Hainmueller 2018).** Stable and dynamic memory engrams emerge in parallel.
- **S46 (Chung 2021).** Cognitive-control training improves hippocampal information processing and later learning, with effects lasting months.
- **S47 (Aronov 2017).** The hippocampal–entorhinal circuit maps a **non-spatial** continuous variable (sound frequency) the way it maps space.
- **S48 (Moore 2021).** Hebbian plasticity lets rats encode journey-specific episodes through multiplexed tuning.
- **S49 (Ormond & O'Keefe 2022).** Place cells have goal-oriented vector fields that support flexible navigation.

---

## E. Neuroplasticity

### S50. Kleim & Jones (2008). Ten principles of experience-dependent plasticity
**Access: EXCERPT.**

**Context.** Decades of animal work on skill learning and recovery from brain damage. The authors stress that the principles don't automatically become clinical recipes.

1. **Use it or lose it.**
   - Circuits that go unused degrade.
   - Depriving a kitten's eye of light reduced light-responsive neurons in visual cortex (Hubel & Wiesel).
   - After a monkey's digit was removed, its cortical area came to respond to neighbouring digits (Merzenich 1984).
   - Blind people's visual cortex is active during Braille reading.
   - **Often the territory is reallocated rather than simply lost.**
2. **Use it and improve it.**
   - Skill training expands the trained body part's map in motor cortex and adds synapses (in monkeys and rats).
   - Complex environments and "acrobatic" training improve recovery after damage.
3. **Specificity.**
   - **Learning** a skill drives gene expression, dendrite growth, synapse addition and map changes.
   - **Repeating movements that are already learned does not.**
   - Training is regionally specific.
4. **Repetition matters.**
   - In rats, behaviour improved within days, but synaptic strength, synapse number and map reorganisation changed only **after several more days**.
   - Lasting LTP from daily stimulation took 5 days to appear and plateaued at day 15.
   - Repetition may make a skill resistant to decay, getting it "over the hump."
5. **Intensity matters.**
   - 400 reaches a day added synapses; 60 a day didn't.
   - Low-intensity stimulation causes LTD, high-intensity causes LTP.
   - 1,800 TMS pulses induced lasting change; 150 didn't.
   - **But forced overuse in the first week after an injury enlarged the damage** (through excitotoxicity), and early exercise reduced plasticity-related molecules. Delayed exercise helped.
6. **Time matters.**
   - Plasticity unfolds in a sequence: gene expression, then synapses, then maps.
   - Early synaptic changes are more fragile.
   - Rehabilitation starting **5 days after stroke worked much better than starting at 30 days**.
   - Delays let compensatory habits set in.
7. **Salience matters.**
   - A tone reorganised auditory cortex only when paired with **reward or with stimulation of the basal forebrain's acetylcholine system**. Tones alone did not.
   - Lesioning the acetylcholine system prevented learning and reorganisation.
   - Paired nerve-and-cortex stimulation **failed when attention was directed elsewhere**.
   - Emotion modulates consolidation.
8. **Age matters.**
   - Plasticity is reduced and slower in older brains, but it persists.
   - After lesions, sprouting started in about 2–4 days in young rats vs about 20 days in old rats.
9. **Transference.**
   - Plasticity in one circuit can boost later plasticity.
   - Brain stimulation timed **together with** training enhanced skill acquisition.
   - Previously enriched environments improved outcomes after damage.
   - Exercise creates a supportive environment (blood vessel growth, growth factors).
10. **Interference.**
    - Brain stimulation **after** training reduced the training gains. Plasticity "not shaped by behavioral signals" is harmful.
    - **Saturating hippocampal LTP impaired later spatial learning** (Moser 1998).
    - Easy **compensatory "bad habits"** (relying on the unaffected limb, "learned non-use") block recovery of the impaired side. Early training of the unaffected limb worsened later use of the impaired one.
    - Explicit instructions helped healthy people learn a motor sequence implicitly but **hurt stroke patients**.

**Limits.** From 2008, and mostly based on rodents and primates.

---

### S51. Review: exercise, enrichment and isolation (2017–2023 studies)
**Access: EXCERPT.**

**Key points.**
- **Aerobic exercise** raises growth factors (BDNF, VEGF, IGF-1), new neurons, new glia, new blood vessels and myelin remodelling.
- **Voluntary exercise beats forced exercise.** Forced training raises stress and inflammation.
- **Stopping exercise caused a rebound:** neurogenesis fell **below** control levels for a few months (Nishijima).
- **Resistance training** refines existing connections. Training one limb strengthens the untrained limb too ("cross-education").
- **Motor skill learning:**
  - spines are added and stabilised specifically on neurons that form the engram
  - spines mature over 30–60 days
  - inhibitory changes depend on the interneuron subtype (somatostatin synapses decrease, parvalbumin synapses increase)
- **Environmental enrichment** (physical + cognitive + social) increases synapse formation, neurogenesis and myelin.
  - It reduced anxiety more than exercise did.
  - **Varying or increasing complexity beat constant enrichment** (Ramirez-Rodriguez 2022).
  - Social enrichment alone matched full enrichment for neurogenesis markers.
  - Effects can last after enrichment ends.
- **Social isolation** reduces plasticity markers.
  - Adolescent isolation shifts striatal decision-making from goal-directed toward habitual.
  - Isolation effects depend on age, sometimes paradoxically: isolation from adolescence into adulthood *reduced* anxiety.
- **The same stimulus can have opposite effects at different ages.** Low-dose corticosterone helped adolescent rats but raised anxiety in adults.

**Limits.** Mostly rodent studies, many of them small.

---

### S52. Overview of neuroplasticity mechanisms, interventions and ethics (c. 2025)
**Access: EXCERPT.**

**Key points.**
- **Four mechanisms:** synaptic plasticity (LTP, LTD, synaptic scaling), structural plasticity, neurogenesis and functional reorganisation.
- **Plasticity can be adaptive or maladaptive.** Maladaptive examples are **self-reinforcing over-strengthening**:
  - chronic pain: excessive LTP leading to hyperconnected pain networks
  - addiction: drug-driven LTP in reward circuits
  - possibly schizophrenia: excessive pruning
- **Homeostatic plasticity** is named as the stabilising counterweight.
- **Plasticity across the lifespan:** highest early, declining with age but persisting.
- **Interventions:** TMS, tDCS, brain–computer interfaces, psychedelics, BDNF-targeting drugs, lifestyle. An evidence table rates most of these as "moderate" or "emerging."
- **Ethics:** cost and access, data privacy, therapy vs enhancement, consent.

**Limits.**
- Broad, with little new.
- **Some numbers need checking.** For example, it quotes a 60% remission rate for TMS in depression, which looks high compared with what I remember from trials. Not verified here.

---

### S53. Fuchs & Flügge (2014). Adult neuroplasticity: 40 years of research
**Access: EXCERPT.**

**Key points.**
1. **Chronic stress causes reversible dendrite retraction,** not neuron death.
   - Stress shortens dendrites of CA3 neurons, while amygdala neurons grow more branches.
   - **Synapses return after the stress ends.**
   - Prefrontal dendrites even change with the daily rhythm.
2. **"Stress kills neurons" was an artefact.**
   - Early monkey studies saw "dead" neurons because brain tissue was fixed too late after death, producing "dark neurons."
   - Better methods found no stress-induced neuron loss.
   - The hippocampus shrinks under stress, but through changes to dendrites and surrounding tissue, not cell loss.
3. **The history of adult neurogenesis.**
   - Altman's early autoradiography results.
   - Songbird neurogenesis linked to song learning.
   - Rakic's negative primate results.
   - BrdU labelling showed it exists in primates and humans.
   - Carbon-14 dating estimated about **700 new hippocampal neurons per day** in adult humans, plus new striatal interneurons.
4. **Bats and whales have low or absent hippocampal neurogenesis yet excellent spatial memory.**
5. **Neurogenesis and depression.** Its role is debated. Hippocampal shrinkage in depression likely isn't explained by reduced neurogenesis.
6. **Chronic stress increases heterochromatin** (condensed chromatin) in CA3 neurons. Chronic cortisol affects CA1 instead, so stress and glucocorticoids act through different pathways.

**Limits.** From 2014. Later work disputes how much neurogenesis happens in adult humans (B20).

---

### S54. MDPI review: neuroplasticity and nervous-system recovery
**Access: EXCERPT.**

**What it covers.** Regenerative medicine:
- Schwann-cell repair of peripheral nerves
- oligodendrocytes and myelin
- neural stem cells
- electrical stimulation, nerve grafts, hydrogels, nanofibres and extracellular vesicles

**Relevant points.**
- Homeostatic plasticity counteracts the self-reinforcing nature of Hebbian plasticity, and can act within hours.
- Habituation has short-term (synaptic depression) and long-term (gene expression) forms.

**Limits.** It contains laboratory protocol paragraphs (resazurin assays, nanofibre membrane preparation) that don't belong in a narrative review, a sign of patchwork writing. Low relevance.

---

### S55. Zihl (2021). Adapting to the adaptive brain (opinion)
**Access: FULL.**

**Points made.**
- **"Functional reserve."** The ageing brain keeps skills despite losing neurons, by switching between structures within a functional system and by recruiting compensating programs.
- **Recovery has two routes:**
  - **restitution**: recovering the function itself
  - **compensation**: working around the loss, including through aids or environment changes
- **Older people have many entrenched routines** that give reliability but are hard to change.
- Rehabilitation should be individually tailored. Happiness and satisfaction act as rewards that drive learning.

**Limits.** An opinion piece without data.

---

### S56. Hauptman, Liu & Bedny (2024). Built to Adapt
**Access: NONE.** Only the title page was seen; no abstract was available.

**From background knowledge of Bedny's *other* work (not this review):** in people born blind, "visual" cortex responds to language and mathematics. This suggests human cortex has broad potential, limited by its connectivity and by sensitive periods. **Read the review before relying on this.**

---

### S57. Yaeger & Harnett (2024). A stable dendritic zone in plastic neurons (via MIT News)
**Access: PRESS.**

**What they found** (layer-5 pyramidal neurons in mouse visual cortex).
- Input from the visual thalamus (dLGN) lands in a narrow band on the **apical oblique dendrites**.
- Synapses there:
  - **don't interact with each other** (no cooperative boosting)
  - have **very few NMDA receptors**
  - showed **no activity-dependent plasticity** under protocols that strengthen other synapses ("as far as we have tested")
- **Other dendritic zones on the same neuron stay plastic.**
- **The zone stabilises a few weeks after eye opening, and only with visual experience.** Mice raised in the dark never stabilise.

**Their explanation.** Basic visual features need a robust, high-fidelity readout that doesn't change ("imagine forgetting what a vertical line looks like"). Mixing stable and flexible synapses in one neuron balances stability and flexibility. They are exploring this for AI catastrophic forgetting.

**Limits.** Only the press release was read.

---

### S58. Tucciarelli … Makin (2026). Remapping of the body map is set in early childhood
**Access: PART.**

**What they did.**
- **Participants.** People born with one hand: 16 children (5–7 years) and 16 adults (25–61), plus matched controls (21 children, 16 adults).
- **Measures:**
  - fMRI of higher-level somatosensory cortex (areas 1 and 2)
  - video analysis of which body parts were used to do 15 normally two-handed tasks
- **Modelling** compared two explanations:
  - homeostatic plasticity, driven by the missing input
  - Hebbian plasticity, driven by use

**What they found.**
- Children born with one hand used **more diverse body parts** to do the tasks.
- **Remapping of the whole body map** in the deprived hand area was **already established by age 5–7 and maintained into adulthood**, even though behaviour kept changing.
- **The modelling:** deprivation-driven **homeostatic** plasticity accounts for the global remapping. Use-driven **Hebbian** learning adds to the **differences between individuals**.

**Their explanation.** Cortical maps are established early and are stable despite extensive changes in everyday behaviour.

**Limits.** Cross-sectional (different people at each age, not followed over time), about 16 per group, fMRI.

**v2 note on how far this generalizes.** The participants were *born* with one hand: this is congenital deprivation during development. v1 (I-6) extended it to "idle capacity in adults is taken over by neighbours, which threatens dormant traces". That extension is an inference, not something S58 tested. The ML analogue that *does* hold is mechanical: weight decay and global normalization act on unused parameters in the same way a homeostatic force would.

---

### S59. Zhang … Love (2025). Representations stretch along task-relevant dimensions
**Access: PART.**

**What they did.**
- **Data.** Re-analysed spiking data from monkeys cued trial-by-trial to judge either the colour or the motion direction of dots. Recordings came from V4, MT, lateral PFC, frontal eye fields, LIP and IT.
- **Model.** A network made of a frozen pretrained CNN plus a recurrent LSTM, trained by trial and error on the same task.
- **Measure.** Whether representations "stretched": distances growing along the relevant dimension and shrinking along the irrelevant one.

**What they found.**
- **Representations stretched along the cued, relevant dimension in every area,** including V4 and MT, not only frontal areas. The effect was strongest in frontal areas.
- **Spike timing** contributed to the code beyond firing rate.
- **The model's LSTM also stretched,** with no built-in attention mechanism.

**Their explanation.** Stretching is an adaptive strategy that emerges from reducing error. Control structures can be **learned**, not presupposed.

**Limits.** A re-analysis of one dataset (two monkeys).

---

## F. BTSP (behavioral timescale synaptic plasticity)

### S60. Quanta Magazine (2026). A new type of plasticity after a single experience
**Access: PART.**

**What it says.** Science-journalism history of BTSP:
- Magee's lab, with Bittner, Grienberger, Milstein and others, discovered in the mid-2010s that a **single dendritic plateau** could turn a silent CA1 neuron into a place cell.
- When plateaus were induced artificially, cells then fired at that location 99.5% of the time.
- The plasticity window spans **seconds**, not milliseconds.
- They named it in a 2017 Science paper. It met early scepticism and is now widely studied.
- **Proposed mechanism:**
  - active synapses carry **eligibility traces**
  - a plateau triggers a dendrite-wide voltage change
  - that change strengthens the tagged synapses
- Losonczy suggests it could be useful for AI.

**Limits.** Journalism. Use S61 and S63 for the substance.

---

### S61. Madar et al. (2025). BTSP review
**Access: PART.**

**Why Hebbian timing rules (STDP) are insufficient:**
- They work over milliseconds, so they can't link events seconds apart.
- They need many pairings, so they're too weak for one-shot learning.
- They are unsupervised, with no explicit way to decide which synapses should change.

**How BTSP differs:**
1. **Triggered by occasional dendritic plateaus** (with a "complex spike" burst at the cell body), not by individual spikes.
2. **Operates over seconds,** in a window of about ±5 s.
3. **Produces large changes in one shot.**

**Key properties:**
- **The window is asymmetric.**
  - Inputs before the plateau are strengthened over a longer stretch.
  - Place-field width correlates with the animal's speed at the moment of the plateau, which confirms the seconds-long window.
- **It is bidirectional and depends on the existing weight.**
  - A plateau away from the current field creates a new field **and depresses the old one** (Milstein 2021).
  - Plateaus inside the field change little: **it saturates after a few events**.
- **It is synapse-specific.** A plateau alone doesn't potentiate inactive synapses. It is modelled as a **local eligibility trace × a global plateau signal**.
- **Plateaus are induced by** coincident CA3 (proximal) and entorhinal (distal) input. The entorhinal input looks like an instructive signal.
- **Plateaus require escaping local inhibition**, through disinhibition circuits engaged by novelty or learning.
- **Neuromodulators** (acetylcholine, β-adrenergic signalling) facilitate plateaus.
- **Activating many cells at once doesn't work,** because strong feedback inhibition stops it.
- **The rate of BTSP events is scheduled.**
  - From thousands of place fields compared against simulations: at most about 0.5 plateaus per 100 spikes in novel environments, 0.2 in familiar ones.
  - The probability is high right after a field forms, then decays to a low nonzero level.
  - Plateaus outside the field are more likely in novel environments.

**Limits.**
- Mostly CA1. How widespread BTSP is in the rest of the brain is still being explored.
- The molecular basis of eligibility traces is unknown.
- In-vivo plateau rates differ between studies.

---

### S62. Wu & Maass (2025). A simple BTSP model gives content-addressable memory
**Access: PART.**

**Question.** Can a minimal BTSP rule create a high-capacity memory that can be retrieved from partial cues (content-addressable), in one shot, with binary weights?

**The model.**
- **Input:** sparse binary patterns from CA3-like units.
- **Memory units:** CA1-like McCulloch-Pitts threshold units with no lateral connections.
- **Gating:** each memory unit receives a random plateau with probability *f_q* per memory item. The experimental value is about **0.005**.
- **Rule:** within the plateau window (about −6 s to +4 s), each active input's **binary weight flips with probability 0.5** (0→1 or 1→0). Plasticity **doesn't depend on the memory unit firing**.
- **Recall:** present a partial cue and see which memory units fire. Feedback connections can be added to create attractors.

**What they found** (from the parts read).
- **A high-capacity content-addressable memory with binary weights.** Hopfield networks need weights with many levels and lose capacity when rounded to binary.
- **The plateau probability matters.**
  - Values much larger than 0.005 make memory traces bloat and degrade partial-cue recall.
  - At 0.005, recall works with up to about a third of the cue missing.
  - The experimental value appears **near-optimal**, helped by the seconds-long window.
- **Better than random projection (the fly algorithm).** BTSP produces a learned, input-tailored random projection. It doesn't need a 40-fold expansion or global winner-take-all competition, and it supports attractors for recall.
- **Repulsion.** Traces for **similar items are pulled apart**, reproducing a known effect in human memory that Hopfield networks don't show.
- **It suits memristor hardware** because the rule is local with two weight states.

**Their explanation.** BTSP lets CA1 allocate memory traces on the fly with one-shot learning. Stochastic gating by entorhinal input is a key, previously unmodelled parameter.

**Limits.**
- A simplified model.
- **It assumes plateaus are random and independent of the input.** That contrasts with the "instructive" view in S63 and S64.
- The capacity analysis wasn't fully read here.

---

### S63. Magee (2026). BTSP: properties, elements, functions (review)
**Access: ABS (paywalled).**

**Claims in the abstract.**
- BTSP is strong, bidirectional and spans many seconds. It is induced by single plateaus, so it can create place cells in one trial.
- Plateau initiation is controlled partly by **local feedback inhibition** and by an **instructive input from a higher-order region**, potentially linking plasticity to current experience.
- Its **new way of assigning credit** "could mitigate the need for widespread synapse stabilization."
- It may allow hippocampal networks both to form memories of specific episodes **and** to generalise across them.

---

### S64. Li, Romani & Magee (2026). Plateau-gated one-shot plasticity supports continual recognition memory (preprint)
**Access: PART.**

**Question.** How does one-shot plasticity limit interference with existing memories?

**The single-neuron model.**
- Binary plastic weights *w*, and an input pattern *x*.
- A **plateau (write) happens if the total dendritic drive η⟨w, x⟩ + ⟨v, x⟩ exceeds a threshold κ**:
  - η⟨w, x⟩ is the plastic, proximal drive (CA3-like), attenuated by η
  - ⟨v, x⟩ is a **fixed instructive, distal pathway** *v* (entorhinal-like)
- If a write happens, *w* becomes *x* (every mismatched synapse switches). Otherwise *w* stays the same ("jump or stay").
- **This creates a loop:** the current weights help trigger the event that rewrites them.
- κ represents how strong inhibition is.

**What they found (single neuron, exact results).**
- For random, unstructured inputs, the dynamics reduce exactly to **one variable: how well the plastic weights align with the instructive weights**.
- **Three regimes:**
  - **frozen:** too misaligned, so a plateau can never fire
  - **transient:** states that a write pulls into the memory band
  - **a closed but dynamic memory band:** writes keep happening, but alignment stays stable
- **Raising κ** increases alignment (fidelity) and makes rewrites much rarer, until everything freezes.
- **Individual synapses keep turning over while overall fidelity stays stable.** The authors offer this as a mechanism for representational drift.
- **Persistence is "kinetic":** memories last because rewrites are rare, not because synapses are permanently stabilised.
- **With correlated, structured inputs (two modes),** the instructive pathway **selects which mode is stored**, the one aligned with *v*, rather than averaging the two.

**What they found (network).**
- **Task: continual recognition.** A stream of sparse binary patterns (5% active). Each pattern is judged novel or familiar *before* plasticity acts. Repeats come R steps later.
- **The memory network:**
  - a binary plastic memory matrix
  - **familiarity-dependent suppression of plasticity**: familiar items get weaker writes
  - for each item, a softmax over drives picks **the memory unit to write**. In practice this was effectively winner-take-all.
- **Compared:**
  - **BTSP-inspired:** fixed instructive drive + plastic drive pick the unit
  - **Hebbian control:** plastic drive alone picks the unit
  - both matched for storage, synapse type, number of writes, readout and optimisation, and both optimised at every lag
- **Results:**
  - Longest lag recognised at ≥90% accuracy: **169 for BTSP vs 31 for Hebbian** (input dimension 400, 32 memory units).
  - The advantage grew from **5.5× to 42×** across the tested sizes.
  - BTSP capacity grew roughly as N × d^0.67, where N is the number of memory units and d the input dimension. The Hebbian control's capacity **fell** as d increased.
  - **Intermediate coupling of the two pathways was best.**
    - Too much plastic drive → the network re-selects units that already respond, so turnover slows but less new encoding happens.
    - Too much instructive drive → **novel items look familiar**.
  - **A reduced theory predicted accuracy and capacity** from the optimised parameters with nothing fitted to the simulations. It was about 1.2× too optimistic on capacity.
  - **Memory-trace strength** fell from 0.99 to 0.82 after 169 steps for BTSP, vs 0.94 to 0.63 for the matched Hebbian control.

**Their explanation.**
- **Partially separating *allocation* (credit assignment: which unit gets written) from *storage* (the plastic weights)**, combined with sparse one-shot writes, limits interference.
- Cross-time depression between successive plateaus also helps.

**Limits.**
- An unreviewed preprint.
- **Recognition only.** No recall from cues, no associations, no transfer.
- Random sparse patterns, each repeated at most once.
- **The instructive weights are fixed by assumption,** with no account of where they come from.
- **Allocation is effectively winner-take-all,** which is the very thing S62 says BTSP avoids.
- The Hebbian control is one particular baseline, not the strongest possible competitor.

---

## G. The user's own list of "brain stability mechanisms," fact-checked

The user listed mechanisms from web sources (S65–S69, not read). Here is how each claim holds up against background knowledge:

| Claim | Verdict |
|---|---|
| Perineuronal nets wrap neurons and close critical periods | **Correct.** Dissolving them with the enzyme chondroitinase reopens adult plasticity, so the brake is removable. |
| Fast hippocampal encoding; slow cortical consolidation through sleep replay | **Correct** (Complementary Learning Systems; see S07, S19). |
| Sparse representations | **Correct.** |
| Non-overlapping activity patterns protect old skills | **Partly correct.** They protect, but they also block transfer (S11). |
| Variable plasticity: important synapses are protected | **Correct** in principle (B02 is the ML counterpart; S57 shows a biological case). |
| Synapses settle into a stable energy landscape | **Overstated.** Spines turn over and representations drift (S38, S64) while behaviour stays stable. The stability is at the level of what the network computes, not individual synapses. |
| New capacity through neurogenesis and new spines | **New spines: yes** (B07). **Adult neurogenesis in humans: disputed** (B20), and essentially absent in neocortex. |
| E–I balance through parvalbumin interneurons prevents runaway activity | **Correct.** |
| Dendritic gating compartmentalises plasticity | **A well-supported hypothesis.** S57 is a concrete example. |
| "Rapid compensatory plasticity" | **Real** (S13). The user's source was a press release (S69). |
| Molecular brakes (perineuronal nets, myelin-associated inhibitors) lock in adult networks | **Correct** in outline. |


---

## I. Predictive processing: circuits, precision and prediction-error signals (new in v2)

**Why this section exists.** Several design ideas in v1 rest on prediction error: the "stamper" uses delta (prediction error) to gate writes, and the counterweight principle concerns runaway predictive loops. These sources test what prediction-error signals actually are in the brain, where they are computed, and how their weight (precision) is set. The short answer is that the classical picture is under pressure, and a revised one is emerging.

### S70. Keller & Sterzer (2024). Predictive processing: a circuit approach to psychosis
**Access: EXCERPT (essentially full). Identification inferred.**

**Question.** Psychosis can be described as a failure of inference about the causes of sensory input. Can it be tied to specific cortical circuit elements?

**Key content.**
1. **Two historical roots of predictive processing:**
   - *efficient coding:* transmit only what wasn't predicted, as in video compression
   - *internal models for motor control:* a forward model predicts the sensory result of a movement; the mismatch teaches the model
2. **The authors' main conceptual point: prediction errors have two jobs.** They update the current representation of the world, and they serve as the teaching signal for learning internal models. A network that computes them therefore has a learning signal available at every node. The authors argue this, not efficiency, is the real advantage of predictive processing.
3. **The canonical psychosis account:**
   - Priors become too weak relative to sensory evidence, so ordinary events seem significant ("aberrant salience").
   - Contrary evidence also exists. A hierarchical version reconciles them: weak *low-level* priors and overly strong *high-level* priors.
4. **The four elements of a minimal cortical circuit for predictive processing:**
   - **Prediction-error neurons.** Separate positive and negative populations exist in layer 2/3. Prediction-error computation survives adult cortical NMDA-receptor knockout (Widmer 2022). Layer 2/3, but not layer 5, excitatory neurons show intracellular signals consistent with computing errors (Jordan & Keller 2020).
   - **Internal-representation neurons.** They integrate errors over time. Their identity is unknown; layer 5 is often proposed.
   - **Precision signals.** Their circuit implementation is unknown, and neuromodulators look too coarse:
     - **Noradrenaline** responses are synchronized across the whole dorsal cortex. Its main effect is to *drive plasticity after prediction errors* (Jordan & Keller 2023).
     - **Acetylcholine** in visual cortex is driven mainly by locomotion, not prediction errors, and acts mainly on layer 5.
     - **Dopamine** innervation of posterior cortex is sparse and value-related.
   - **Internal models.** These are the transformations between areas, stored in the synaptic weights of the connections between them.
5. **Superficial and deep layers may be partly independent "compartments":**
   - Anaesthesia *synchronizes* layer-5 activity cortex-wide; antipsychotics *decorrelate* it. Both leave superficial layers largely unchanged.
   - Bulk activation of layer 2/3 mainly *suppresses* layer 5.
   - **Speculation:** the two compartments run partly separate models; layer 2/3 intervenes in layer 5 only when layer 5's predictions fail. This is likened to a joint-embedding architecture, or to layer 2/3 supplying a precision-like signal.
6. **The psychosis hypothesis:**
   - Three antipsychotics (clozapine, haloperidol, aripiprazole) decorrelate layer-5 activity across dorsal cortex in mice, and the effect outlasts the drug (Heindorf & Keller 2023).
   - Untreated first-episode patients show *increased* functional connectivity across cortex.
   - **Proposal:** spurious long-range layer-5 correlation causes false inference. Core regions, being more densely connected, would show stronger priors than peripheral ones.
   - This fits the observed pattern of thalamocortical connectivity in schizophrenia: increased in sensory and motor regions, decreased in prefrontal regions.

**Limits.**
- Much of the circuit mapping is from mouse V1 and visuomotor tasks.
- The layer-5 decorrelation hypothesis is explicitly speculative.
- "Prediction error" here is mostly the visuomotor mismatch case, which is low-dimensional (see S72).

**Why it matters here.**
- The dual role of prediction errors supports giving every module a local learning signal.
- Neuromodulators as coarse plasticity broadcasts, rather than precise precision signals, supports the "broad signal × local selectivity" design (`insights.md` I-44).
- Too much correlation as a failure mode joins the counterweight principle (I-13).

---

### S71. Active filtering by recurrent cortical networks (review, 2025)
**Access: EXCERPT (essentially full). Identification guessed (probably M. H. Histed).**

**Question.** Each cortical area has up to ~500 million local excitatory-to-excitatory synapses per cubic millimetre. What are they for in *sensory* cortex, where activity is mostly locked to the stimulus rather than self-generated?

**Main proposal: "active filtering".** The local recurrent network **selectively amplifies input patterns that resemble natural sensory input and attenuates others**, such as spontaneous noise. It is:
- a *predictive* function: it privileges expected patterns
- a *form of memory*: which patterns are amplified encodes the statistics of the world
- **not generative:** responses start and stop with the input, within tens of milliseconds

**Evidence.**
- **Anatomy:**
  - 50–80% of an excitatory cell's synapses come from excitatory neurons less than 1 mm away.
  - Connection probability averages about 22%, up to about 40% for similarly tuned cells.
  - Individual synapses are weak, but patterned population input sums.
  - About two-thirds of the input to a visual cortical neuron comes from the local recurrent network.
- **Causal stimulation (two-photon holographic optogenetics):**
  - Co-tuned orientation ensembles in V1 self-amplify (Marshel 2019; Oldenburg 2024).
  - Played-back natural-vision patterns are amplified (Deveau 2024).
  - Whisker-tuned ensembles in somatosensory cortex amplify too (Peron 2020, via lesions).
- **Mechanism (theory):** balanced amplification (Murphy & Miller 2009) in an **inhibition-stabilized network**. Recurrent excitation is strong enough to be unstable on its own, and inhibition stabilizes it. Recent studies support strong recurrence across areas and species.
- **Other senses:** auditory cortex shows sharply frequency-tuned recurrent excitation and offset responses. Offset suppression lasting a few hundred milliseconds appears in many areas, which suggests filtering over time as well.

**Two meanings of "engram" (after Semon).**
- **(1) A latent physical change** that stores a memory. Active filtering *is* an engram in this sense.
- **(2) The activity pattern produced when the memory is revived.** Active filtering is *not* an engram in this sense, because it doesn't reactivate past patterns. It changes responses to *present* input based on the past.

**Open questions the author lists.**
- How much of what is amplified is genetic vs learned?
- Which homeostatic mechanisms preserve the filter during learning?
- What are the learning rules?
- Is it a fixed-point attractor or a continuous (metastable) amplification?

**Limits.** A proposal that synthesizes a small number of recent stimulation studies, mostly in mouse V1. Dendritic nonlinearities could contribute as well; the author argues that the dimensionality of synaptic weights favours a recurrent-weight mechanism.

**Why it matters here.**
- It gives a mechanism for *perceptual* memory stored in sensory cortex: the slow, long-term statistics that the design calls the "shared slow core".
- It's a concrete case of a counterweight: amplification is only possible because inhibition stabilizes an otherwise unstable network (I-13).
- **Possible link to savings [guess]:** a recurrent filter that still amplifies an old skill's input patterns would speed relearning without any explicit memory of the skill.

---

### S72. What sensory prediction-error signals encode (review, 2026)
**Access: EXCERPT (essentially full). Identification inferred (Furutachi & Hofer, probable).**

**Question.** Neural responses to unexpected stimuli are taken as prediction errors. But what information do they carry?

**Three candidate forms** (E = mismatch, S = sensory input, a = gain):
1. **Residual difference:** R = ±a·E. The classical predictive-coding view.
2. **Amplified sensory response:** R = (1 + a·|E|)·S. The stimulus representation itself is boosted in proportion to the surprise.
3. **General surprise:** R = a·|E|. "Something changed", without content.

**Evidence.**
- **Classic signatures are confounded.** Mismatch negativity and stimulus-specific adaptation can often be explained by adaptation to repetition. Better designs (global oddballs, learned sequences, cross-modal contingencies) still find deviance responses across species.
- **Residuals do exist when the mismatch is low-dimensional:**
  - dopamine reward-prediction error
  - visuomotor mismatch in mouse V1, where optic flow is faster or slower than predicted from running speed
- **The decisive test (Furutachi et al. 2024, mouse V1 layer 2/3 in virtual reality):**
  - **The same unexpected stimulus under different predictions gave nearly identical responses**, so individual neurons carried little information about *what was expected*.
  - **Different unexpected stimuli under the same prediction drove different neurons**, so the signal is feature-specific, not general surprise.
  - Only sharply tuned neurons were amplified; broadly tuned neurons were not.
  - **Conclusion for this paradigm: an amplified sensory response.**
- **Circuit:**
  - Unexpected stimuli recruit **VIP interneurons** and **pulvinar** inputs.
  - Pulvinar input alone suppresses layer 2/3 via SOM interneurons. With VIP active (disinhibiting), pulvinar drives layer 2/3.
  - The amplification scales with the size of the deviation.
  - VIP and pulvinar responses are *broad*, yet the amplification they produce is *feature-selective*: it lands on sharply tuned neurons.
- **Plasticity link:** the same VIP-plus-thalamic motif can trigger dendritic plateau potentials and plasticity without somatic spikes (Gambino 2014; Williams & Holtmaat 2019).
- **A critical admission:** "this central premise of predictive coding theories" (that prediction errors are what update internal models) **has not been experimentally tested** for sensory prediction errors. It has been tested only for dopamine reward-prediction errors.

**Their revised model.** Feedforward neurons represent *features*, not residuals. Their gain is scaled by the size of the prediction error. Residuals may coexist, in other populations or in a two-stage process.

**Limits.** Mostly one paradigm in mouse V1. How far it generalizes across areas, modalities and hierarchy levels is open.

**Why it matters here.**
- **Correction to v1 I-9:** "prediction error gates what gets written" is a reasonable design rule, but its biological support for *sensory* learning is Tentative, not Supported.
- The "broad modulator, selective effect" circuit is one of three instances of a general motif (I-44).
- **Novelty → disinhibition + higher-order input → plateau → plasticity** is close to BTSP's recipe (S61). It suggests one write-gate motif shared by hippocampus and cortex (I-45).

---

### S73. Precision weighting, neuromodulation and psychopathology (perspective, 2026)
**Access: EXCERPT (essentially full). Identification guessed.**

**Claims.**
- Predictive coding's explanatory power depends on **precision**: how reliable each signal is judged to be.
- Neuromodulators are proposed to set different kinds of precision:
  - **acetylcholine:** sensory uncertainty. In auditory cortex it enhances mismatch responses while sparing repetition suppression.
  - **dopamine:** confidence in policies and action-related prediction errors
  - **noradrenaline:** environmental volatility, and therefore the learning rate
  - **serotonin:** uncertainty about affective and bodily states
- Psychopathology is framed as maladaptive precision:
  - **Schizophrenia:** aberrant precision of priors or of sensory errors; NMDA hypofunction; reduced MMN.
  - **Autism (Arthur et al. 2023):** not globally weak priors, but **difficulty adapting learning rates to volatility**.
  - **Autism (Adamou et al. 2026):** excess precision of low-level sensory errors, with repetitive behaviours as attempts to reduce uncertainty.
  - **Anxiety:** hyper-precise threat-related bodily cues.

**The strongest content is what it cites:**
- **Ernst & Banks (2002, B41):** people combine visual and haptic cues almost optimally, weighting each by its reliability.
- **Behrens et al. (2007, B42):** people raise their learning rate when the environment is volatile and lower it when it is stable.
- **The Hierarchical Gaussian Filter (Mathys et al. 2014, B43):** a formal model in which volatility sets how much each prediction error updates beliefs.

**Its own stated weakness.** If precision is inferred from the symptoms it is meant to explain, the theory becomes unfalsifiable (Bowman 2023). Partial remedies: estimate precision from behaviour with computational models (HGF), and from independent markers (MMN, connectivity modelling, pupil responses tracking information gain, Colizoli 2026).

**Limits.**
- A perspective with no new data.
- Its claim that neuromodulators *are* the precision signal conflicts with S70's evidence that they are too coarse (in mouse V1). See `insights.md` I-47 for the reconciliation.

**Why it matters here.** Volatility → learning rate is the most directly useful idea in this cluster for continual learning. It gives v1's rule "ramp plasticity up after distribution shifts" a formal model.

---

### S74. Journal-club commentary (2010) on Alink et al. and den Ouden et al.
**Access: EXCERPT (essentially full).**

**Two fMRI studies.**
- **Alink et al. (2010):**
  - A bar presented on the path of an illusory moving bar evoked a *lower* V1 signal when its timing fitted the illusion (predictable) than when it didn't.
  - **Predictable bars were *more* detectable**, which rules out a simple attention explanation.
- **den Ouden et al. (2010):**
  - Arbitrary tone→face/house contingencies, with probabilities changing over the experiment, reduced responses in face and place areas to predicted stimuli. This shows sensitivity to **short-term, arbitrary** contingencies.
  - The **putamen** signalled a *generic* prediction error, the same for faces and houses.
  - Effective-connectivity modelling suggested it **gates** the coupling between perceptual and motor areas when predictions fail.

**The commentary's caution.** A lower BOLD signal is compatible with "explaining away" (predictive coding) and also with **sharpened tuning**: a lower average with better signal-to-noise. fMRI cannot tell them apart.

**Limits.** fMRI only; 2010; a commentary.

**Why it matters here.**
- Lower activity is not less information (detectability went up).
- Another generic, content-free error signal (putamen) that switches routing. That's the third instance of error-gated override, alongside S70's layer 2/3 → layer 5 and S91's fallback logic (I-49).

---

### S75. "Predictive routing" vs classical predictive coding (review, c. 2024–2025)
**Access: EXCERPT (essentially full). Identification inferred (Bastos lab).**

**Question.** Classical predictive coding predicts that feedforward neurons in sensory cortex carry prediction errors, and that predictions suppress responses to expected input. Is that true at the level of spikes?

**Evidence (mostly primate, local–global oddball paradigm).**

| Signal | Local oddballs (repetition violated) | Global oddballs (learned pattern violated) |
|---|---|---|
| fMRI / M-EEG / LFP | Widespread, including sensory cortex | Widespread, including sensory cortex |
| **Spikes** | Widespread, sensory to prefrontal | **Absent in V1, V4, mid-level auditory cortex (Tpt). Present only sparsely in PFC (~2% and ~9% of neurons in two studies); decodable at population level** |
| Under propofol anaesthesia | Persist in sensory cortex | Decoding in FEF (prefrontal) falls to chance |

- Local oddballs survive anaesthesia, when frontal cortex is inactive. So they reflect adaptation, not top-down prediction.
- **Why imaging and spikes disagree:** BOLD and LFP track an area's *input* currents more than its *output* spiking, and BOLD correlates with gamma-band LFP more than with spikes.
- The same imaging-vs-spike divergence has been reported for working memory and attention.

**Their alternative: predictive routing.**
- No dedicated prediction-error neurons.
- Higher-order cortex (mixed-selectivity PFC ensembles) sends **alpha/beta-band** feedback that **pre-inhibits** the predicted sensory channels.
- Input arriving at an *unprepared* channel produces more **gamma** activity and spiking. That is the "error".
- Sensory cortex keeps a **redundant** code (it fires even to fully predictable input). Complex predictions live higher up.
- **Supporting result:** under propofol (PFC inactivated, alpha/beta modulation gone), local-oddball gamma in auditory cortex *increased*. Removing top-down input disinhibits sensory cortex.

**Clinical framing (speculative).**
- Schizophrenia: reduced sensory gamma and parvalbumin-interneuron dysfunction, so weak bottom-up input leaves the internal model unanchored.
- Autism: abnormally large local-oddball responses.

**Limits.**
- An advocacy review; much of the spike evidence comes from the authors' lab.
- Passive paradigms. The authors argue attention can't explain the absence, since attention modulates sensory responses by only ~5–23%.
- Anaesthesia also lowers firing rates.

**Why it matters here.** It supports the revised picture in `insights.md` Part 11: faithful feature coding in sensory cortex, cheap local adaptation everywhere, complex prediction in higher areas, and prediction acting on sensory cortex as **gain and routing** rather than subtraction. It also gives the clearest **local vs global** split (I-48).

---

### S76. Ficco et al. (2021). Meta-analysis of predictive processing in fMRI
**Access: FULL (read via browser; reference list skipped).**

**What they did.**
- **Search:** PubMed for "predictive coding" + fMRI, up to January 2019. 106 articles collected, 70 retained.
- **Inclusion criterion (g):** studies had to *explicitly support* predictive coding.
- **Contrasts classified as:**
  - **Prediction Violation:** 45 experiments, 511 foci, 939 subjects
  - **Prediction Encoding:** 39 experiments, 444 foci, 750 subjects. A mixed category: repetition suppression, adaptation, learned vs unfamiliar, high vs low probability, belief updating, explicit prediction.
  - **Pooled:** 70 experiments, 930 foci, 1,419 participants
- **Methods:**
  - Activation Likelihood Estimation (ALE) at family-wise error p < 0.05
  - "Seed-voxel correlation consensus": each activation peak's resting-state connectivity map from Neurosynth (1,000 subjects), then the overlap across peaks
  - a fail-safe test against unpublished null results
  - leave-N-out heterogeneity checks

**Findings.**
- **Violation:** left inferior frontal gyrus (IFG) / precentral, and left anterior insula / claustrum.
- **Encoding:** **no convergence**, even at FDR p < 0.01. Leave-N-out showed the dataset is homogeneous, so the scatter is real and not caused by a few outlier studies.
- **Pooled:** left IFG, both insulae, right superior frontal gyrus, both inferior parietal lobules, left precuneus.
- **Connectivity:** nearly the same network for violation and encoding (cosine similarity 0.68–0.93). It resembles the task-positive network (salience + dorsal and ventral attention) and is anticorrelated with the default mode network.
- **No early visual areas** in any result.
- **Robustness:** clusters survived adding 250% (pooled) and 425% (violation) random "unpublished" data.

**Limits (some admitted by the authors).**
- **Confirmatory sample:** only studies supporting predictive coding, found by searching for the theory's name. The fail-safe test addresses missing null results, not this selection effect.
- The "Encoding" category mixes adaptation with prediction, which S75 shows are different.
- fMRI only.
- No neutral control condition, so the network might reflect any attention-demanding task.
- Resting-state maps seeded from frontal and parietal peaks tend to produce the attention network anyway.
- Coding done by one author, cross-checked by another.

**What survives.**
1. Prediction and error are **not separable at the network level**, which is consistent with S75's "no dedicated error neurons".
2. Cross-modal prediction converges on **higher-order, attention-linked** regions, not sensory cortex, which is consistent with S75.
3. The violation hubs (insula, IFG) are **salience-network** territory (S84).


---

## J. Consciousness, non-conscious representation and coarse-to-fine inference (new in v2)

### S77. Dresp-Langley (2012). Non-conscious representations and conscious experience
**Access: EXCERPT (essentially full). Identification inferred.**

**Claims.**
1. Non-conscious processing has very large capacity; conscious processing is very limited. Most knowledge is acquired outside awareness, starting with statistical learning, which is present at birth.
2. **Mechanism, from Adaptive Resonance Theory (B40):**
   - *Bottom-up automatic activation* processes input without awareness.
   - *Top-down matching* compares input with learned memory traces.
   - A **positive match amplifies** the input and brings it toward awareness; a **negative match suppresses** it.
   - Grossberg's argument, cited here: such matching mechanisms are what prevent catastrophic forgetting in learned representations.
3. Conscious experience arises when coincident activity traces "resonate" in long-range reentrant circuits. These temporal signatures are consolidated by developmental selection of frequently co-active patterns.
4. The temporal signatures are said to be independent of spatial maps and to survive destruction of the circuits that built them.

**Evidence, sorted by strength.**
- **Solid:**
  - masked semantic priming
  - priming and implicit learning without awareness
  - blindsight
  - collinear context making sub-threshold contrast targets visible (with monkey V1 correlates)
  - ERP markers of conditioning to unseen faces
  - **evaluative conditioning becomes reliable only when people are momentarily aware of the contingency** (Pleyers et al., cited)
- **Weak or contested; discount these:**
  - subliminal psychodynamic activation and subliminal relaxation messages
  - the uncontrolled BBC broadcast study (the author notes laboratory replications were unclear)
  - weak magnetic fields raising hypnotizability. In my memory, this line of work has a poor replication record. **[memory]**
- **Speculative:** claim 4. It leans on phantom-limb remapping, and S58 qualifies use-driven remapping.

**Why it matters here.**
- It points to ART (B40), a classic stability–plasticity solution the v1 library lacked. In ART, *match* gates refinement and *mismatch* recruits new capacity (I-62).
- "Contingency awareness is needed for stable valence learning" is independent evidence that learning arbitrary relations may require the limited-capacity conscious workspace (I-64).

---

### S78. "Dynamic logic", the knowledge instinct, and language–cognition interaction (review)
**Access: EXCERPT (essentially full). Identification guessed (probably Perlovsky & Ilin).**

**Claims.**
1. **Matching bottom-up and top-down signals runs into combinatorial complexity** ("CC"), which the authors tie to Gödel's incompleteness. Rule systems, learning systems and model-fitting systems each hit it in a different form.
2. **"Dynamic logic" avoids it.** Associations between data and models start *vague* (fuzzy, near-uniform) and become *crisp* (0 or 1) as model parameters improve.
   - Vague = unconscious; crisp = conscious.
   - Examples:
     - finding "smile" and "frown" patterns at signal-to-noise below 1 in a 100×100 image: about 10⁹ operations instead of the 10⁵⁰⁰⁰ of exhaustive search
     - learning 10 "situations" (10 essential objects each among 50 perceived, from 1,000 possible objects; 16,000 samples) in about 3 iterations
3. **Evidence cited for coarse-to-fine perception (B52, Bar et al. 2006):**
   - Orbitofrontal cortex activates about 130 ms after visual cortex but about 50 ms *before* the fusiform object-recognition area.
   - It projects a *low-spatial-frequency* "gist" that facilitates recognition.
4. **A "knowledge instinct":** an inborn drive to improve the fit between internal models and the world. Its satisfaction or dissatisfaction is felt as "aesthetic emotions" (harmony, beauty, the sublime).
5. **A "dual model":**
   - Every concept has a *language* part, learned fast and crisply from surrounding speech, and a *cognitive* part, grounded slowly in experience.
   - This would explain why children can talk about abstract ideas long before they understand them.
6. Speculative extensions: music reduces cognitive dissonance; how inflected a language is shapes cultural evolution.

**Assessment.**
- **The mathematics is overclaimed. [memory]** Dynamic logic appears to be expectation–maximization for mixture models with a fuzziness (variance) that shrinks over iterations, closely related to deterministic annealing (B60).
  - EM never enumerates all M^N assignments of data to models, so the "10⁵⁰⁰⁰" baseline is a straw man.
  - "Unsolvable by other methods" and "10,000% improvement" are not credible as stated.
  - The situation-learning demo is a Bernoulli mixture fitted by EM.
- **The Gödel → combinatorial complexity argument is rhetoric**, not a mathematical result.
- **The language and culture claims are unsupported and lean ethnocentric** (e.g. that English's loss of inflections "spread democracy", or that highly inflected languages go with "stagnating" cultures). Do not use them.

**What survives.**
1. **Coarse-to-fine inference**, supported by B52: stay soft and vague early, commit late.
2. **The "knowledge instinct" felt as a change in model fit** is essentially *learning progress* / compression progress (B38). That's the same signal as v1's I-10 and the stamper's learning-progress channel (I-66).
3. **The dual model as a lens on language models [inferred]:** fluent language can run far ahead of grounded understanding. Compare S10: explicit text "imagination" beat latent imagination, so a crisp language scaffold can guide a vaguer internal model.

---

## K. Motor learning, transfer, savings and learning to learn (new in v2)

**Why this section exists.** The user's central idea, that forgotten skills relearn faster, is studied most directly in motor-adaptation research under the name **savings**. v1 had almost none of this literature.

### S79. Marins et al. (2025). Training vs transfer effects in motor-sequence learning (preprint)
**Access: FULL (read via browser; references skipped).**

**Data.** A re-analysis of a public 7-Tesla dataset (Berlot, Popp & Diedrichsen 2020; OpenNeuro ds002776).
- 13 right-handed adults (9 female, mean age 21.4).
- Six 9-digit finger sequences practised for about 4,000 trials over 5 weeks. Six matched sequences never practised.
- Scans before and after training, with paced execution in the scanner.

**Method.**
- Searchlight **hyperalignment** (functional alignment across brains), then a between-subject linear classifier (leave one subject and one run out; 104 folds).
- Permutation tests with FDR correction.
- Three questions:
  - *training effect:* pre vs post, for trained sequences
  - *transfer effect:* pre vs post, for untrained sequences
  - *resulting engrams:* trained vs untrained, after training

**Findings.**
- **Behaviour:**
  - Trained sequences: movement time 2.44 s → 1.30 s.
  - Untrained sequences: 2.46 s → 1.46 s.
  - **About 88% of the speed-up transferred to sequences never practised** (1.00 s of 1.14 s). This is my calculation from the reported means; the paper doesn't highlight it.
- **The training effect decoded better than the transfer effect** (F(1,12) = 9.29, p = 0.01), especially in the central sulcus (primary motor and sensory cortex, M1/S1).
- **The central sulcus was specific to the training effect.** Transfer was coded mainly in the surrounding premotor and parietal territory.
- Representations shifted for both trained and untrained sequences, more for trained.
- The ipsilateral (right) hemisphere carried more decodable information (F(1,12) = 8.43, p = 0.013).
- **Anatomical alignment largely failed** (e.g. 0 significant vertices for the transfer effect in the left hemisphere). The shared code is functional, not anatomical.

**Their interpretation.** Motor skill learning is hierarchical. Repetition builds rigid, execution-level codes in M1/S1; transfer relies on higher-level sequence representations.

**Limits.**
- n = 13; a preprint.
- **Pre vs post decoding mixes learning with session differences** (different day, head position, arousal). This affects the training and transfer effects equally, so their *difference* is the defensible result, not the absolute accuracies of 70–75%.
- The "two distinct engrams" result may partly reflect differences in sequence identity and familiarity.

**Why it matters here.**
- **Correction to v1 design §3.1 and I-20 [inferred]:** v1 assumed the vision-style layout (general features low, task-specific parts high). In motor skill the *specific* learning sits at the low execution level and the *transferable* learning sits higher. Share at whatever level holds the task invariants; specialize where tasks conflict. That's a property of the domain, not a fixed rule about layers.
- Most of practice's benefit here was **general**, not sequence-specific.

---

### S80. Seidler (2010). Neural correlates of motor learning, transfer of learning, and learning to learn
**Access: FULL (read via browser; references skipped).**

**Key points.**
1. **Separate performance from learning.** In a control study (aiming at targets of different sizes), performance changed without any learning. Activity in some regions tracked performance itself. Brain changes "during learning" must be checked against such controls.
2. **Sensorimotor adaptation uses two systems:**
   - **Cerebellum:** forward models, error-driven trial-to-trial updating, storage. Patients with cerebellar damage can correct *within* a movement but don't adapt *across* trials.
   - **Basal ganglia (plus prefrontal, ACC, parietal):** active early in learning. Probably cognitive support (attention, working memory, error monitoring) rather than storage. Patients with Huntington's disease show the reverse deficit.
3. **Transfer and savings:**
   - Adapting after other adaptive experience is faster.
   - How much savings is seen depends on task order and on the measure used.
   - **Older adults learn more slowly but show intact savings.** Learning and savings are dissociable mechanisms.
   - The **two-state model** (B44, Smith, Ghazizadeh & Shadmehr 2006): a fast process that learns and forgets quickly, plus a slow process that learns and forgets slowly. It reproduces savings.
   - At transfer, early-learning regions (right IFG, M1, early cerebellar regions) are *less* active, and savings correlates with late-learning and storage regions (lateral cerebellum, parietal).
   - Interpretation: transfer = **retrieving a stored model and moving quickly through the early phase**.
4. **Learning to learn:**
   - After several *unrelated* motor tasks (three rotations, plus a gain change and a sequence task), people learned new unrelated tasks faster than controls.
   - **Cost: their movements were less stable against brief perturbations.** "More adaptable … but less resistant to external perturbation."
   - Proposed mechanisms: lower limb stiffness (so bigger errors and more information), or better general error detection (anterior cingulate).
   - Older adults retain learning to learn.
   - Recommendation: varied practice, not narrowly focused practice.

**Limits.** A short review, mostly of the author's own lab. The neural claims are fMRI-based, and the learning-to-learn mechanism is hypothesis.

**Why it matters here.**
- It is the entry point to the computational models of savings: B44 (two-state), plus B45 (memory of errors) and B46 (explicit strategy), which come from memory. See section H.
- The adaptability–robustness cost is the stability–plasticity trade-off appearing in behaviour, and it suggests **the brain learns where to set the plasticity dial from how variable its history has been** (I-61).


---

## L. Metacognition, salience and uncertainty (new in v2)

**Why this section exists.** Most of the "when to change" mechanisms collected so far need a signal for "how reliable am I, here and now?" These include: the learning rate set by volatility, replay of items of intermediate strength, arbitration between systems, rebuilding a degraded skill, and fallback to explicit procedures. These sources address that self-monitoring signal, the system that flags what *matters*, and where uncertainty is represented.

### S81. Metacognition and neuroplasticity (narrative review, 2026)
**Access: EXCERPT (essentially full). Identification guessed.**

**Content.**
- **Framework:** Nelson & Narens' two levels. An *object level* does the task; a *meta level* monitors it (upward flow) and controls it (downward flow).
- **Neural substrates:**
  - medial and lateral prefrontal cortex, especially right rostrolateral and anterior PFC (Vaccaro & Fleming 2018 meta-analysis of 47 studies, B55)
  - dorsal anterior cingulate plus anterior insula: conflict and uncertainty
  - parietal cortex (precuneus, inferior parietal lobule): evidence and confidence
  - hippocampus–precuneus connectivity: memory confidence
- **Evidence that the monitor can be dissociated from what it monitors:**
  - TMS to lateral PFC impairs metacognitive accuracy while task performance stays intact (Lapate 2020; Miyamoto 2021).
  - **Anosognosia:** unawareness of one's own deficit. Occurs in 10–18% of acute stroke patients with hemiparesis, linked to right parietal, temporoparietal, insular and thalamic lesions. In mild cognitive impairment and early Alzheimer's disease, 20–30%.
  - In early Alzheimer's disease, metamemory judgements can be relatively preserved while memory itself declines.
- **Clinical evidence:**
  - **Metacognitive therapy (MCT) vs CBT for depression** (Callesen et al. 2020, n = 174, B54):
    - recovery 74% vs 52% (odds ratio 2.42, p = 0.014)
    - at 6 months, 74% vs 56%
    - fewer sessions (5.5 vs 6.7)
  - MCT targets *how one responds to thoughts* (worry and rumination loops held in place by beliefs such as "worrying helps me cope"), not what the thoughts say.
  - MERIT (for schizophrenia) and brain-injury awareness training show promise.
- **Educational evidence:**
  - Metacognitive strategy use correlates r = 0.32 with mathematics achievement (147 studies, 698,096 participants).
  - The Education Endowment Foundation estimates up to +7 months of learning progress.
- **Taxi drivers (Maguire 2000/2006, B53):**
  - Larger posterior hippocampus and smaller anterior hippocampus than controls.
  - Changes absent in bus drivers matched for driving experience and stress.
  - **Reduced ability to acquire new visuospatial information.**

**Limits.**
- The link between metacognition and neuroplasticity is mostly hypothesis. Taxi driving isn't metacognitive training, and the BDNF, exercise and meditation pathways are speculative, as the authors admit mid-paper.
- **The conclusion contradicts that admission:** it says the evidence "demonstrates" lasting structural reorganization from metacognitive interventions.
- The figure mapping meta levels onto first-, second- and third-person language is the authors' own construct.
- The educational figures are correlational.

**Why it matters here.**
- A separable self-monitor is biologically plausible (I-51).
- **Specialization has a cost** (taxi drivers): direct human evidence for the user's intuition that "over-optimizing one cache weakens others" (I-61).
- **Runaway thought loops are maintained at the meta level** (MCT's evidence): relevant to rumination-like failures in replay and imagination (I-55).

---

### S82. Metacognition in cognitive neuroscience and educational science (conceptual review)
**Access: EXCERPT (essentially full). Identification guessed.**

**Content.**
- **Meta-knowledge vs meta-control** (upward monitoring vs downward regulation).
- **Online vs offline metacognition (the most useful distinction in this paper):**
  - **Online:** fast, automatic, no reflection needed. Error detection, conflict monitoring, executive function. Linked to posterior medial frontal cortex and ACC.
  - **Offline:** slow and reflective, built on meta-representations. Confidence judgements, planning, strategy selection. Linked to anterior and lateral PFC.
  - **ERP evidence:** the error-related negativity (ERN) accompanies errors generally; the later **error positivity (Pe)** specifically accompanies errors the person can *report*.
- **Measurement:**
  - **Sensitivity (meta-d′):** how well confidence discriminates one's correct from incorrect choices.
  - **Bias:** overall confidence level. Sensitivity and bias are independent.
  - **Efficiency (meta-d′/d′):** sensitivity corrected for task performance. A ratio of 1 means all the evidence available for the decision was also available for the confidence judgement.
- **Neural meta-analysis (B55):**
  - a domain-general network (medial and lateral PFC, precuneus, insula)
  - right anterior dorsolateral PFC specific to perceptual decisions
  - parahippocampal cortex specific to memory
- **Training and transfer:**
  - Executive-function training gives mostly near transfer and little far transfer (mixed in adults).
  - **Feedback on confidence in a perceptual task improved metacognition in an *untrained memory* task.**
  - Two weeks of mindfulness improved memory metacognition but not perceptual metacognition.
  - **Offline metacognition looks more trainable** and is more strongly tied to learning outcomes.
  - Learning journals help when students reflect on their learning *activities*, but hurt when they reflect only on the *material*.
  - Think-aloud protocols correlate highly with achievement; self-report questionnaires correlate weakly.
- **Development:** meta-knowledge emerges around age 5 and meta-control around age 8. Both start domain-specific and become domain-general with expertise.

**Limits.** Mostly conceptual. The authors stress ecological validity: it's unclear whether lab metacognition matches real-world metacognition.

**Why it matters here.**
- The online/offline split matches the local/global split of predictive processing (S75) and the non-conscious/conscious split (S26, S77). See I-52 and I-64.
- **Far transfer at the meta level** (perception → memory) supports "what transfers furthest is how to learn, not what was learned" (I-53).

---

### S83. Attention × adaptation: gain vs selectivity (fMRI)
**Access: EXCERPT (Discussion only).**

**Findings.**
- Adaptation here = stimulus-specific adaptation (SSA), measured with fMRI.
- **Experiment 1, attend to identity vs spatial location** (separate neural populations): attention multiplied the adapted response by a constant factor at every level of adaptation. A pure **gain** change. Same result in voice areas (auditory) and face areas (visual).
- **Experiment 2, attend to voice identity vs loudness** (overlapping populations): attention boosted the *less-adapted* responses disproportionately. A **selectivity** change.
- **Rule:**
  - When attended and ignored features are processed by *largely separate* populations, attention changes gain.
  - When they are processed by *overlapping* populations, there's competition and attention reshapes selectivity.
- **Formal bridge:**
  - Normalization model of attention (B58): response = enhancement ÷ normalization.
  - Murray & Wojciulik's adaptation model, with attention gain nested inside it.
  - The predictive-coding form, where **precision (1/σ) plays the role of the normalization term**.
  - So precision weighting and divisive normalization are mathematically interchangeable in this setting.
- SSA (a gain decrease via inhibitory interneurons) is separated from expectation suppression, which is more accessible to consciousness. A monkey study (Vinken 2018) found no repetition-expectation effect in face-area local field potentials.

**Limits.** fMRI; sample sizes and venue unknown; Discussion section only.

**Why it matters here.**
1. A candidate **mechanism for precision**: divisive normalization (I-46).
2. **The interference principle at the level of attention:** separate codes can be modulated independently (gain); overlapping codes force trade-offs (selectivity changes). Same geometry as S01, S11 and S59.

---

### S84. Seeley (2019). The salience network
**Access: EXCERPT (most of the text). Identification inferred.**

**Content.**
- **Origin:**
  - Behavioural-variant frontotemporal dementia (bvFTD) removes concern, drive and restraint ("both the brakes and the gas pedal were broken").
  - It causes atrophy in the anterior cingulate (ACC) and ventral anterior insula ("frontoinsula"), where von Economo neurons (VENs) cluster.
  - Mapping the frontoinsula's intrinsic connectivity produced the "salience network" (Seeley 2007), alongside a separate executive-control network.
- **Meaning of "salience":** *homeostatic relevance*.
  - Frontoinsula is the input hub, perceiving visceral and autonomic feedback.
  - ACC is the output hub, generating visceral, autonomic, behavioural and cognitive responses.
- **A neighbouring but distinct network:**
  - **Salience network** (ventral insula, many VENs): detects what matters.
  - **Cingulo-opercular task-control network** (dorsal insula, nearly VEN-free; Dosenbach 2006, B57): maintains a task set (tonic alertness).
  - In Seeley's view, the salience network engages the task-control network to **either maintain the current task set or switch to a new one** as the salience landscape changes. In Dosenbach's data the frontoinsula appeared only in the map derived from *behavioural errors*.
- **Findings since 2007:**
  - ACC and insula respond to salience **independent of valence**, positive or negative (Bartra 2013, >206 studies, B56).
  - Beta-adrenergic blockade (propranolol) weakens the network's response to a stressful film.
  - Dopamine depletion reduces its connectivity.
  - Coma-causing brainstem lesions near the locus coeruleus connect mainly to ACC and frontoinsula.
  - **Direct ACC stimulation produced a sense of a looming challenge, a faster heart rate, and a will to persevere** (Parvizi 2013).
  - It's implicated across disorders (bvFTD, schizophrenia, bipolar disorder, depression, ADHD, anxiety, autism, substance use). Alzheimer's disease and Williams syndrome may *increase* salience-network function.
  - **Neurodegeneration spreads along connectivity:** a region's connection to the disease's epicentre predicts its vulnerability.

**Limits.** A personal account; the author says the network's core function remains uncertain.

**Why it matters here.**
- It unifies the generic "something's off" hubs seen across sources: insula/IFG (S76), dACC + insula for online meta-knowledge (S82), unsigned surprise (S72, S73), ACC in early learning (S80).
- **A concrete hypothesis for context discovery (v1's open problem I-21) [inferred]:** a relevance detector signals a meaningful change, and a separate controller maintains or switches the task set (I-54).
- **Damage spreads along connectivity [inferred]:** in a modular learner, a corrupted shared hub damages everything that depends on it (`design.md`, hub protection).

---

### S85. Commentary on Vilares et al. (2012): where prior and evidence uncertainty are represented
**Access: PRESS (commentary only; the study itself was not read).**

**Findings as reported.**
- **Task:** a coin-toss estimation game. Participants weighted a prior (where coins usually land) against evidence (noisy cues) according to their relative uncertainties. That's behavioural support for precision weighting.
- **fMRI:**
  - **Uncertainty about the evidence** was represented in the same visuomotor regions that process the evidence (*co-localized*).
  - **Uncertainty about the prior** was represented elsewhere: insula, orbitofrontal cortex, putamen, amygdala.
- Belief formation was separated from value: the events carried no reward. The commentators note that reinforcement learning handles poorly the problem of learning beliefs about events that carry no value.

**Limits.** Secondhand; fMRI correlations.

**Why it matters here.**
- It gives precision an "address": **local for evidence, central for priors** (I-47).
- The central regions overlap the salience network (insula) and the generic-error region (putamen; S74).
- **Reward-free belief learning** is a separate function. It supports a world-model / curiosity channel kept apart from value.


---

## M. Cross-modal learning and transfer (new in v2)

**Why this section exists.** Transfer vs interference (v1 I-20) was the least-resolved principle in v1, and it rested on toy models and theory. S87 supplies strong causal evidence for one concrete mechanism.

### S86. Cross-modal invariants and biological "plen-functions" (theory)
**Access: EXCERPT (essentially full). Identification guessed (probably Jörntell's group).**

**Claims.**
1. **"Plen-functions".** Each sense has an all-encompassing function describing every possible pattern on its sensor sheet. Proposed biological versions (dimensions):
   - **touch** (frequency, amplitude, time, 3-D skin location): 6 dimensions
   - **vision** (wavelength, intensity, time, 2-D retinal location): 5 dimensions, or **10** once focus, pupil and gaze control are included
   - **hearing** (frequency, intensity, time): 3 dimensions, or at least 5 with ear movements and cochlear tuning
2. **Single-sense "kinematic invariants"** (stable patterns of sensor activity over time) are partly extracted *before* cortex: in the cuneate nucleus for touch (shown by the group's earlier work), and plausibly in the retina. They are grounded in physics (skin mechanics, optics, acoustics).
3. **Cross-modal invariants have no analytic mapping** between light, air-pressure waves and contact forces. So they must be learned, individually and pragmatically: first through motor babbling, then through whatever is most useful at each stage of development.
4. **They need long-lasting working memory.** Planning steps several seconds ahead from vision is one example. Cats remember an obstacle for over half an hour and adjust their hindleg step to it. The authors argue this requires the cortex.
5. **They pervade the whole cortex:**
   - Any cortical neuron is within 2–3 synapses of any thalamic input.
   - Primary sensory cortices respond to other senses.
   - "Association cortex" is a property of most neurons, not a region.
   - This is why cross-modal integration has been hard to localize.
6. **Methods warning:** in a signal rich enough, low-dimensional embeddings will correlate with almost any variable, and different embeddings tell different stories. So an embedding that "works" doesn't reveal the brain's actual computation.
7. Supporting evidence they cite: sensorimotor mismatch signals (visuomotor in V1, audiomotor in A1) are large, and mismatches in two senses together amplify each other.
8. **Synaesthesia** is framed as overtraining of one particular pairing of senses. VR illusions are presented as a test of understanding.

**Assessment.**
- **Claim 3 is overstated [inferred].** There is no analytic mapping *between sensor spaces*. But there *is* a shared ground truth: **the latent physical state of the world** (where an object is, what force acts on it, when an impact happens). These are the same facts whether seen, heard or felt.
- S87 shows exactly this: shared spatial location makes transfer instantaneous.
- The accurate version: **physics grounds the latent causes; learning has to discover them.**

**Why it matters here.**
- Learning cross-modal structure needs long working memory and anticipation over seconds (compare S27, S75).
- The embedding warning is the same lesson as S10 and S75: **decodable is not the same as used**.

---

### S87. A cortical circuit for cross-modal generalization in mice (primary study)
**Access: EXCERPT (most of Results and Discussion). Authors and venue not given (probably Nature Communications).**

**Question.** How does the brain generalize a learned rule from one sense to another?

**Behaviour.**
- Head-fixed mice learned a Go/No-go task: lick for the top whisker (B2), withhold for the bottom whisker (C2).
- They were then switched to vision: drifting squares at the matching top or bottom positions in the visual field.
- **Rule-preserving switch:** expert performance on the very first visual session, within the first few tens of trials.
- **Rule-reversing switch:**
  - performance fell to chance **or below** (mice licked for the spatially congruent visual stimulus)
  - then they **disengaged** for several sessions, despite being thirsty
- **No-prior control (non-spatial auditory task first):** performance dropped to chance and recovered over a few sessions, **faster than the rule-reversed group** (p = 0.02).
- **Conclusion: a conflicting prior is worse than no prior.**
- Transfer worked in both directions. Vision → touch showed a small drop, attributed to whiskers being harder to discriminate.
- Control: removing the whiskers from the stimulators dropped tactile performance to chance, so the mice weren't secretly using visual cues.
- Reversing the rule *within* touch produced rigid persistence on the old rule (habit-like).

**Maps and anatomy (task-naive mice).**
- Wide-field imaging (29 mice): **aligned visual and tactile maps of vertical space** across associative areas A, RL, AL, LI, and even in S1. The two maps are offset by about 30°.
- Multisensory enhancement was strongest where the maps aligned.
- Anterograde and retrograde tracing: topographically matched **feedforward** projections (S1 → associative areas, V1 → associative areas) and matched **feedback** projections (associative areas → S1 and V1).
- There's no direct V1 → S1 projection, so visual responses in S1 must come via feedback.

**Neurons (two-photon imaging).**
- Four response types: visual-only, tactile-only, "gated" (responding only to the combination), and **bimodal**.
- Bimodal neurons form a dorsal cluster (A, RL; 124 neurons) and a ventral cluster (AL, LI; 75 neurons).
- They are tuned to **spatially congruent** visual and tactile positions (71–77% in congruent quadrants). Congruent combinations are enhanced and incongruent ones suppressed, giving sharper tuning than the unisensory responses predict.
- **A decoder trained on whisker responses generalized to visual responses only when built from bimodal neurons.**

**Causal tests.**
- **Loss of function:** tetanus toxin (TeNT) expressed in associative areas *before any training*.
  - Mice learned the tactile task normally.
  - Silencing **RL** abolished transfer: performance after a rule-preserving switch fell to chance.
  - Relearning then proceeded at the rate of the no-prior auditory group.
  - With RL silenced, rule *reversal* was also relearned faster than in controls. **Silencing RL removes both the benefit and the harm of the prior.**
  - Ventral silencing (AL) caused only a small, transient drop.
- **Gain of function:** after a brief habituation phase, patterned optogenetic stimulation of RL's top- and bottom-coding subregions **substituted for the visual stimuli**.
  - Rule preserved: discrimination maintained.
  - Rule reversed: chance level, and about twice as long to start responding to the stimulation.
  - Stimulating **AL** was not even detected over 7 days.

**Model.**
- Areas V1, S1, RL and AL, each with top and bottom excitatory–inhibitory populations.
- **Only the sensorimotor (readout) synapses are plastic**, learning by reward-modulated Hebbian rules.
- RL's strong bidirectional connections keep the visual and tactile codes for the same location linked, so a readout learned from touch also works for vision.
- The model reproduces instant transfer, slow reversal, and "learning from scratch" when RL is silenced.

**Limits.**
- Mice; spatial (peri-personal) rules only. Object recognition may use the ventral stream.
- TeNT spread was estimated from the cortical surface.
- "Sufficiency" holds only after optogenetic habituation.

**Why it matters here (strongest new source in v2).**
1. **Transfer and interference are the same channel with opposite sign**, shown behaviourally and causally (I-57).
2. **Mechanism: the rule attaches to a shared latent variable (location), not to the sense.** Nothing in sensory cortex has to change; only the readout learns.
3. **The shared code exists before any task:** developmental alignment, probably Hebbian learning on correlated spontaneous and sensory activity (I-58).
4. **Removing the hub turns the learner naive.** That's the causal test to copy in ML (`design.md`, experiment E17).

---

### S88. Hadjikhani & Roland (1998). PET study of tactile–visual shape matching
**Access: EXCERPT (essentially full). Identification inferred.**

**Design.**
- 8 right-handed young men; ¹⁵O-butanol PET.
- 15 ellipsoids of equal weight and surface area differing only in shape.
- Conditions: tactile–tactile (TT), visual–visual (VV), tactile–visual (TV) matching, and a motor control.
- Motor activity and task difficulty (d′) were balanced across conditions.
- Boolean intersections of subtraction images were used to isolate components.

**Findings.**
- **Tactile shape** (TT−control ∩ TV−VV): left postcentral gyrus, anterior intraparietal sulcus, superior parietal lobule; right cerebellum.
- **Visual shape** (VV−control ∩ TV−TT): lingual, fusiform, middle occipital gyri, cuneus.
- **No region was active in all three matching conditions**: no "polysensory" area for shape.
- **The only region specific to cross-modal matching** (TV−TT ∩ TV−VV): **right insula–claustrum**.
- Proposal: cross-modal transfer happens by communication *between* modality-specific areas, via the claustrum, rather than through an amodal shape area.

**Limits.** 1998 PET, n = 8. The authors themselves note that PET cannot separate the claustrum from the insula.

**Why it matters here.**
- **It conflicts with S87's dedicated hub**, but they test different things: spatial rules vs object shape. **Different invariants may use different architectures [inferred]** (I-63).
- The insula/claustrum also appears for prediction *violation* (S76) and as a salience hub (S84). Cross-modal *matching* is partly mismatch checking, so this may reflect the generic comparator rather than the transfer channel **[guess]**.

---

### S89. Hierarchical haptic monitoring (fMRI)
**Access: EXCERPT (essentially full). Identification guessed (probably Gurtubay-Antolin et al., eNeuro).**

**Design.**
- 22 participants (17 analysed in the main task), blindfolded.
- An independent localizer identified shape-selective regions (left anterior intraparietal sulcus, bilateral lateral occipital complex) and a texture-selective region (left SII).
- Main task: a spoken word set up an expectation, then participants touched a stimulus. On 50% of trials, its shape or texture didn't match.
- Behavioural accuracy 91%.

**Findings.**
- **Feature-selective haptic regions did not distinguish expected from unexpected stimuli.**
- **Any mismatch** (shape or texture) activated the left supramarginal/angular gyrus (SMG), left middle temporal gyrus and medial PFC.
- **Left primary somatosensory cortex (SI) responded differently to shape vs texture mismatches.**
- SMG–SI coupling increased during mismatches. The SMG also coupled with sensorimotor cortex and, at a liberal threshold, the cerebellum.

**Their interpretation.**
- A higher-order **generic comparator** (SMG), comparing actual input with predictions possibly computed in the cerebellum.
- A lower-order site for **specialized** mismatch (SI).
- Medial PFC as an amodal mismatch area.

**Limits.**
- fMRI.
- The localizer was weak (lenient threshold, p uncorrected 0.005); the authors admit it missed typical regions.
- The expectation was verbal (semantic).

**Why it matters here.** Error detection is hierarchical. Feature-blind comparators sit higher; feature-specific mismatch sits lower; feature-selective areas themselves don't carry the mismatch. That's consistent with S75's spike data and with the "broad × local" motif (I-56).

---

### S90. Editorial: Frontiers Research Topic on crossmodal learning
**Access: EXCERPT (full editorial).**

An overview of 22 papers spanning neuroscience, psychology, AI and robotics. Points worth keeping:
- **Bayesian integration is the shared theoretical framework.** AI systems remain domain-specific deep networks, while the brain uses a strikingly uniform structure.
- **Visual recalibration of auditory space** is ongoing and subject to top-down modulation (ventriloquism after-effect; macular degeneration distorting auditory space).
- One summarized mouse study (Li et al.): superficial-layer activity patterns during multimodal memory formation were consistent across areas and stable over days, while deep-layer dynamics were area-specific and history-dependent. That's weak support for S70's two-compartment idea.
- Unsupervised audio–visual binding from co-occurrence (Barros et al.).

**Weight: weak** (an editorial).

---

## N. Robotics: explicit state and code as policy (new in v2)

### S91. Hu et al. (2026). Embodied Turing Machines / Code-Only-as-Policy (COAP)
**Access: EXCERPT (most of the text).**

**Claim.** If the robot and environment state can be measured accurately and stored explicitly, the decision policy can be written entirely in code, with no vision-language model or vision-language-action model running at test time. Coding agents (Claude Opus 5.5) wrote and improved a shared library offline.

**Method.**
- **State:** environment (object pose, size, category, relations, plus each measurement's error, source and time), robot (joints, end-effector, gripper, reachable poses), and task (stage, arm assignment, retry counts, remembered values).
- **Perception is ordinary code** (NumPy/OpenCV): mask the robot, find candidate regions, intersect camera rays with the known table plane, and fit *known* object meshes.
- **Library:** layered, with dependencies pointing downward. 83% of the code a new task runs comes from the library.
- **Changes:** default-off parameters guarantee unchanged behaviour on existing tests. A change is merged only if it improves the validation success count by a margin above noise.

**Results (RoboDojo, 42 bimanual tasks).**
- **70.24% success vs 31.38%** for the best leaderboard entry.
- Highest in all five dimensions: Memory 89.9%, Precision 75.1%, Open 64.1%, Generalization 65.7%, Long-horizon 56.4%.
- The same program scored **83% on oracle state vs 63% on measured state**.
- **Failure analysis:**
  - 62% state representation: perception errors and contact (e.g. cloth slipping)
  - 37% logic: missing retries, hard-coded values
- **Generalization tasks:** 77.3% on standard layouts vs 54.1% on randomized ones.
- **Standard layouts, validation vs test:** 76.7% vs 74.9% (a small gap).
- **Speed:** a control step takes a median of 0.3 ms on a CPU.

**Limits.**
- **Simulation with published assets:** camera models, meshes and object sizes are exact constants. The authors call this a major limitation.
- Hours of per-task development by coding agents; development compute isn't counted in the efficiency comparison.
- About 17% of the code is task-specific.
- The "exponential capability growth" figure is schematic.
- The compatibility guarantee covers only the test fixtures.

**Why it matters here (`map.md`, section 6).**
- It is the opposite of the brain in where state is held (explicit vs distributed) and where learning happens (offline, outside the policy).
- Yet it mirrors several brain motifs:
  - an object-permanence-like world state
  - content bound to context
  - compositional skill libraries
  - archive-not-overwrite
  - offline consolidation
- Its strongest margins (Memory, Precision) are exactly where human working memory and motor precision fail, and where people reach for external aids. **Code-as-policy is a model of cognitive offloading**, which S82 identifies as a form of meta-control.

---

## H. Background references: what we know about the key ones (new in v2)

**Every claim in this section comes from memory unless it says "cited in Sxx".** These references were not read during the project. They are summarized because several are central to the project's goals and are missing from the library (see `agenda.md`).

### B44–B46. Computational models of savings (motor adaptation)
- **Two-state model (B44; cited in S80; equations from memory).** Output x = x_fast + x_slow.
  - Each state updates as x(n+1) = A·x(n) + B·e(n), where e is the error.
  - The fast state has a high learning rate B and low retention A; the slow state the opposite.
  - **Explains savings:** after washout the output returns to zero while the slow state is still partly adapted, held in balance by the fast state. Relearning starts with a head start.
  - **Also explains spontaneous recovery:** after counter-adaptation and a pause, the old adaptation partly returns as the fast state decays.
  - **Known limit:** it doesn't capture all savings data.
- **Memory of errors (B45; memory).** The nervous system raises its sensitivity to errors it has experienced before. Savings then comes from *learning faster*, not only from residual state.
- **Explicit strategy and reinforced actions (B46; memory).** Part of savings comes from recalling a strategy ("aim 30° to the left") or from previously reinforced actions. This component depends on cognitive, early-learning systems.
- **Synthesis (I-60):** savings has at least three separable sources: a hidden slow state, a remembered learning rate, and a recallable explicit strategy.

### B40. Adaptive Resonance Theory (ART)
- **Principle cited in S77; details from memory.**
- Input activates a candidate category. Its top-down template is compared with the input.
- If the match exceeds a **vigilance** threshold, the system resonates and *refines* that category.
- If not, the category is reset and the search continues; if nothing matches, a **new category is recruited**.
- Old categories aren't overwritten by mismatching input, which is how ART addresses stability vs plasticity.
- Known weaknesses: sensitivity to the vigilance setting and to input order; category proliferation; it hasn't scaled like deep networks.

### B47–B49. Metaplasticity, synaptic memory theory, neuromodulation as meta-learning (memory)
- **Metaplasticity (B47):** the threshold for potentiation vs depression shifts with a neuron's activity history (the BCM sliding threshold). Plasticity itself is plastic.
- **Cascade / complex synapse models (B48):** synapses with internal states spanning many timescales can combine fast learning with long retention. Memory capacity can scale far better than in simple synapse models. This is the mathematical theory of the stability–plasticity dilemma at the synapse level.
- **Doya (2002) (B49):** dopamine, serotonin, noradrenaline and acetylcholine map onto meta-parameters of learning (roughly: reward-prediction error, time horizon/discounting, exploration/randomness, learning rate/memory update).

### B34–B35. Model collapse (memory)
- Training generative models repeatedly on their own outputs loses the tails of the distribution and eventually collapses (B34).
- Accumulating real data alongside synthetic data, rather than replacing it, prevents collapse in studied settings (B35).
- This is the strongest ML evidence for v1's I-35 (imagination and replay can run away). It identifies the remedy: **keep real data in the loop, and tag what is generated.**

### B29–B31, B28. Savings and forgetting in artificial networks (memory)
- **Ebbinghaus (B28)** introduced the savings method in 1885.
- **Hetherington & Seidenberg (B29):** networks that forget catastrophically still relearn faster than from scratch. Some savings comes "for free" from residual weight structure.
- **Davidson & Mozer (B30):** measured savings in a sequential continual-learning setting.
- **Davari et al. (B31):** linear probes on frozen features often recover much of an old task, so a large part of "forgetting" sits in the readout.
- **Consequence:** measuring savings is not new. *Engineering* savings deliberately, and decomposing where it comes from, is still under-explored.

### B39. Gradient alignment (memory)
- When two tasks' gradients point in similar directions (positive inner product), training one helps the other (transfer). When they conflict (negative inner product), it hurts (interference).
- Methods such as MER (B39) and PCGrad (gradient surgery) act on this directly.
- This is the formal version of "transfer and interference are the same quantity with opposite sign", which S87 demonstrates in a brain.
