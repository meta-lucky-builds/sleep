# T3: Plasticity, homeostasis, stability and self-generated replay: ML counterparts and failure modes

Thread T3. Written 2026-11-04. Not committed.

## 0. Conventions and honesty notes

Access tags:
- **PART** = I read the PDF through the alphaXiv page-retrieval tool, which returns only the top-ranked pages (abstract, main body, some appendix). I never saw any paper's complete text, so nothing here is tagged FULL.
- **ABS** = abstract or abstract-page summary only. Details beyond the abstract are not verified.
- **PRESS** = news, blog or search-result summary.
- **CITED** = I only saw it cited inside another paper; I did not open it.
- **INFERENCE** = my own reasoning, not a published claim.

Other notes:
- IDs are arXiv IDs. URL pattern: https://arxiv.org/abs/<ID>.
- Several 2026 papers appear below (IDs 26xx.xxxxx). They were returned by the tool, but most are single-source preprints I have not seen replicated. Treat them as evidence of what has been claimed, not as settled.
- Quotes are under 15 words. Everything else is paraphrase.
- No local tests or model runs were done.

**Not covered or not verified in this session** (so the coordinator knows the holes):
- GEM, MER, PCGrad and MIR primary papers were not opened. MIR is known only through a citation in 2610.12602 (arXiv 1908.04742, CITED).
- Hayes et al., GDumb (Prabhu 2020) and Buzzega "bag of tricks" were not opened.
- Neyshabur's distance-from-init papers were not opened.
- Task arithmetic, TIES and DARE were not re-checked.
- Generative-replay collapse at about 15 tasks (the S06 number in ewk.md) was not re-verified.
- No LLM-specific drift-detection or sleep-trigger paper was found other than the ones listed in section 5.

---

## 1. Loss of plasticity, and whether it exists at LLM scale

### 1.1 The core non-LLM papers

| Paper | Setting | Finding | Tag |
|---|---|---|---|
| Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton, "Loss of plasticity in deep continual learning", Nature 632:768 (2024), doi 10.1038/s41586-024-07711-7. arXiv 2306.13812 (I read v3, titled "Maintaining Plasticity in Deep Continual Learning"). https://arxiv.org/abs/2306.13812 | Continual ImageNet (binary tasks, up to 5000), Online Permuted MNIST, slowly-changing regression. MLP and CNN. No transformer in the pages I read. | ImageNet accuracy fell from 89% to 77% by task 2000. Loss of plasticity is correlated with growing weight magnitude, falling effective rank and rising dead units. Adam, dropout and online normalisation made it worse. L2 and shrink-and-perturb helped. Both were very sensitive to hyperparameters. Continual backprop (reinitialise low-utility units) fixed it. | PART (arXiv v3). Nature version: PRESS (web search confirmed citation and doi only). |
| Lyle et al., "Understanding plasticity in neural networks", ICML 2023. 2303.01486 https://arxiv.org/abs/2303.01486 | DQN-style probe tasks on MNIST and CIFAR; MLP, CNN, ResNet-18, small ViT-based transformer. | Weight norm, feature rank, dead units and weight rank can each be made to correlate positively or negatively with plasticity depending on the problem. So none is a robust cause. Their favoured explanation is loss-landscape curvature and gradient interference. Scaling width reduced plasticity loss but did not eliminate it. Layer norm and categorical (two-hot) outputs helped. Shrink-and-perturb and weight decay did not (appendix text). | PART |
| Kumar, Marklund, Van Roy, "Maintaining plasticity via regenerative regularization" (L2 Init), CoLLAs 2024. 2308.11958 https://arxiv.org/abs/2308.11958 | Permuted MNIST, random-label MNIST/CIFAR, 5+1 CIFAR, continual ImageNet. Small MLP/CNN. | Loss is `L_train + λ‖θ − θ0‖²` with θ0 the initial weights. It was most consistently good across five problems. L2 toward zero failed on three of them. Regularising toward a freshly resampled init each step performed badly, so the fixed anchor matters. L1-Init worked slightly worse. The paper says it studies plasticity only and leaves forgetting aside. | PART |
| Ash and Adams, "On warm-starting neural network training", NeurIPS 2020. 1910.08475 https://arxiv.org/abs/1910.08475 | ResNet-18, 3-layer MLP and logistic regression on CIFAR-10/100 and SVHN. A bidirectional RNN on IMDB appears only in an appendix figure. | Warm-starting hurt generalisation although training loss was the same. Logistic regression was unaffected, so it is a non-convex effect. L2, adversarial training and confidence penalties did not close the gap. Shrink (θ ← λθ) plus noise did. Their Proposition 1 (argmax preserved under shrinking) is proven only for bias-free, batch-norm-free ReLU nets. | PART |
| Sokar et al., ReDo, "The dormant neuron phenomenon in deep RL". 2302.12902 https://arxiv.org/abs/2302.12902 | Deep RL | Dormant neurons accumulate. Recycling them improves performance. | ABS |

Relations worth keeping (partly INFERENCE):
- Kumar's appendix states that with SGD, L2 Init is shrink-and-perturb with shrink `p = 1 − αλ` and noise `σ = αλ`. The differences are that the noise target is the fixed θ0 (not resampled), the gradient is not shrunk, and the equivalence breaks under Adam.
- Anchor-to-init and anchor-to-previous-solution (EWC) are different objects. Kumar describes L2 Init as "remembering how to learn", while EWC remembers what was learned.
- Dohare found that L2 alone did not fix plasticity but shrink-and-perturb nearly did. The authors attribute this to the continual noise injection. Kumar found the opposite ordering (L2 Init beat S&P). So "shrink toward zero" is not a settled mechanism.

### 1.2 Does it exist for transformers and LLMs?

The honest answer has three parts.

**(a) Yes, in controlled small-scale LM training.**
- 2606.24752 (Hernandez-Garcia, Figliolia, Millidge, Zyphra, 2026-06) https://arxiv.org/abs/2606.24752, PART. GPT-style pre-norm transformers of 5M to 314M non-embedding parameters were trained on cycled languages. Plasticity was measured as the area under the validation-loss curve when learning a held-out Vietnamese probe.
  - All models eventually show a rising AUC. The effect also appears under stationary multilingual training.
  - The onset follows a fitted power law `T = 1.3e-5 · P^0.8269`, where T is in "task instances" of 5B tokens each and P is non-embedding parameters. The authors extrapolate this to roughly 360 task instances (about 1.8T tokens) for a 1B-parameter model.
  - The effects are small. The y-axes span about ±1.5% of AUC.
  - Correlates were measured (parameter magnitude, dormant MLP units, lazy and collapsed attention heads), but the authors say none is a "smoking gun".
  - The fit rests on 8 model sizes, and they call it "decent but not perfect".
  - The 1B extrapolation is theirs, not a measurement.

**(b) Yes in a related form at 1B+ scale, but it is a pretraining-extent effect, not a repeated-consolidation effect.**
- 2503.19206 (Springer et al., "Overtrained language models are harder to fine-tune") https://arxiv.org/abs/2503.19206, PART. OLMo-1B pretrained to 3T tokens, then instruction-tuned, was more than 2% worse on several benchmarks than the 2.3T-token checkpoint. They attribute this to "progressive sensitivity": equal-size parameter perturbations damage the model more as pretraining lengthens (shown with Gaussian noise and with fine-tuning).
  - Their Theorem 4.7 (linear setting): regularising fine-tuning toward the pretrained weights, `λ‖θ − θ_pre‖²`, delays degradation at a cost in downstream fit. This is a formal anchored-regulariser trade-off.
  - Appendix Fig. 8 shows the pretrained-to-finetuned distance is roughly proportional to learning rate and independent of pretraining length (OLMo-30M).
  - Most controlled runs are OLMo-30M, and the 3T result is on OLMo-1B.
- 2602.11137 (Han, Bordt, Zhang, Kakade, ICML 2026, "Weight decay improves language model plasticity") https://arxiv.org/abs/2602.11137, PART. Llama-2 and OLMo-2 models up to 4B, pretrained with several weight decays then fine-tuned on 12 tasks. Higher pretraining weight decay gave better post-fine-tuning performance. The best value was 1.0 at 20 tokens per parameter and 0.3 at 140 tokens per parameter, versus the default 0.1. The authors say the causal claim covers the effect, while their mechanisms (more linearly separable representations, lower attention rank, less overfitting) are correlational.
  - This is evidence that decay toward zero helps LM plasticity at pretraining time. It does not test repeated decay between consolidation cycles.

**(c) No direct evidence found for LoRA or for fine-tuned 1B+ LLMs over repeated cycles.**
- I found no paper that measures plasticity loss (slower learning of fresh tasks) in LoRA-adapted LLMs across cycles.
- Closest: 2605.30524 "Representation collapse in sequential post-training of LLMs" https://arxiv.org/abs/2605.30524, PART. It claims that LoRA rank 16 post-training chains on 1B to 7B models lower hidden-state effective rank and lower a "future-task score" (71.0 down to 60.7 over four stages in one table).
  - **Reliability warning:** the paper's tables contain a column headed "Expected interpretation", the numbers are round, and the methods text is written as a design plan ("the central hypothesis is..."). I could not establish that these are real measured results. Do not rely on it until the data or code is checked.
- 2609.33620 (Ren et al., "Learning dynamics of continual learning: data attribution, forgetting, plasticity loss") https://arxiv.org/abs/2609.33620, ABS. Claims a token-level update-interaction decomposition for LMs and that cumulative geometry change weakens later learning signals. Details unverified.
- LoRA-specific forgetting is well documented (section 2), but plasticity is not the bottleneck there. Biderman et al. report that LoRA "learns less", which is a capacity limit and not the Dohare phenomenon.

**Decision-relevant read (INFERENCE from the above):**
- Using the Zyphra fit at face value, a ~100M model reaches onset after tens of 5B-token task instances, and a 1B model after roughly 1.8T training tokens. A sleep framework running a few dozen LoRA consolidations of 10^6 to 10^9 tokens each is orders of magnitude below that, so Dohare-style plasticity loss is unlikely to be the binding constraint on that timescale.
- This is an extrapolation across scale and from full-weight pretraining to LoRA, so it is a prior and not a measurement.
- Caveat on the opposite side: Springer shows fine-tuning sensitivity grows with pretraining length. That means the base model you start from (heavily overtrained 1B models) matters more than the number of cycles.

---

## 2. Anchors, PAC-Bayes, low-rank bias, spectral truncation, merging

### 2.1 Anchor to base versus anchor to zero

- Dziugaite and Roy, 1703.11008 https://arxiv.org/abs/1703.11008, PART. The PAC-Bayes prior is a Gaussian whose mean is the random initialisation w0, not zero. The KL term contains `‖w − w0‖²/λ`. The reason given (appendix B): random initialisation breaks parameter-space symmetries, and they found this "led to much better bounds" than a zero-mean prior.
  - Their nonvacuous bounds are 0.16 to 0.22 (error rates) for binary-MNIST MLPs with 300k to 2.4M parameters, and the bound is vacuous (1.35) for random labels.
  - The setting is small MLPs on MNIST. It is a generalisation bound, not a forgetting bound.
- Nagarajan and Kolter, 1901.01672 https://arxiv.org/abs/1901.01672, ABS. Argue that SGD implicitly limits capacity by keeping the ℓ2 distance from initialisation small, and that capacity measures should be initialisation-relative. They leave open why.
- Kumar L2 Init (section 1.1) is the continual-learning analogue of the same idea and is empirically better than zero-anchored L2 on 3 of 5 tasks.
- For LLM fine-tuning the "initialisation" is the pretrained model. **INFERENCE:** a LoRA delta ΔW = BA with B initialised to zero is already an anchor at the base weights, and weight decay on the LoRA factors shrinks ΔW toward the base. This is anchor-to-base, not anchor-to-zero of the full weights.
- Empirical support: Biderman et al. 2405.09673 https://arxiv.org/abs/2405.09673, PART (Llama-2-7B, code and math, continued pretraining up to 20B tokens and instruction tuning). Classic full-weight weight decay (5e-5, 1e-4) and attention dropout forgot as much as full fine-tuning, and weight decay deteriorated at longer training. LoRA r=256 learned as much as them with less forgetting. So decay toward zero of full weights is not a substitute for being structurally anchored at the base.
- PAC-Bayes bounds with the pretrained model as prior centre for LLMs: I did not search for these and make no claim.

### 2.2 Low-rank adapters and implicit bias

Biderman et al. 2405.09673 (PART):
- With common settings (r 16 to 64), LoRA substantially underperforms full fine-tuning on code and math, especially in continued pretraining. In instruction tuning, rank 256 can close the gap.
- LoRA forgets less, with forgetting graded by rank. Example: code continued pretraining at 20B tokens, forgetting-metric average 0.545 for full fine-tuning versus 0.617 for LoRA r=256 (higher is better).
- The authors state the least-forgetting-when-changing-less result is "trivial", and say the open question is whether the trade-off curve differs. They find it differs by domain: LoRA looks better for code, full fine-tuning for math.
- Full fine-tuning perturbations ΔW have rank 10 to 100 times larger than typical LoRA ranks, shown for code continued pretraining in Llama-2-7B. They caution this does not rule out low-rank solutions.
- LoRA also preserved more diverse generations (more unique HumanEval outputs).
- Recommendations: LoRA for instruction tuning and not continued pretraining, rank 256 on all modules, α = 2r, learning rate one order of magnitude above full fine-tuning.

Kalajdzievski 2401.05605 https://arxiv.org/abs/2401.05605, ABS. Under LoRA, fine-tuning gain and forgetting trade off roughly linearly. Forgetting follows a shifted power law in the number of fine-tuned parameters and update steps, including safety behaviour in Llama-2-7B-chat. Early stopping and parameter-count changes did not prevent it.

Luo et al. 2308.08747 https://arxiv.org/abs/2308.08747, ABS. 1B to 7B models: catastrophic forgetting is generally present and gets worse with scale in that range, possibly because larger models start higher.

### 2.3 Nuclear norm, singular-value thresholding, pruning

- 2608.12332 (Wi and Park, SCLoRA, ACL 2026 Main) https://arxiv.org/abs/2608.12332, ABS. Claims that uncontrolled growth of singular values in LoRA adapters is theoretically linked to forgetting, and that clipping them guided by the pretrained spectrum gives better retention. Model sizes were not visible on the abstract page. No independent replication.
- 2607.01763 (Wang et al., "Denser ≠ Better", preprint) https://arxiv.org/abs/2607.01763, PART. In sequential post-training of a 7B model, linearly merging 50% of an SDPO-trained model with its base, or keeping only the top r=16 singular directions of each weight delta ("spectral merging"), recovered several out-of-domain scores (e.g. ZLogic 74.50 to 79.38 with r=16) while the average stayed about the same. Spectral trimming of a full-fine-tuning delta therefore has one direct supportive LLM result. It was not applied to LoRA deltas and not validated on behaviours beyond the benchmarks listed.
- Jaiswal et al. 2310.01382 https://arxiv.org/abs/2310.01382, ABS. Pruning methods degraded knowledge-intensive task performance at sparsity as low as 25 to 30%, while perplexity looked fine. In-context retrieval and summarisation stayed robust at 50% or more. So magnitude pruning is not behaviour-neutral, and perplexity is the wrong monitor. This concerns pruning of base weights, not LoRA deltas. I found no study of pruning LoRA deltas.

### 2.4 Compressing or merging many adapters into one

- 2606.03723 "Compress then Merge" (ICML 2026 accepted per the abstract page) https://arxiv.org/abs/2606.03723, ABS. Merges T LoRAs into a single rank-r LoRA by building shared r-dimensional subspaces, projecting each adapter into them, and merging the small coordinate matrices. The authors say it beats single-LoRA baselines and narrows the gap to full-parameter merging. The abstract gives no number for loss versus keeping the T adapters separate.
- 2606.28117 "When One Adapter Speaks for Many" https://arxiv.org/abs/2606.28117, ABS. Adapters trained on different tasks have substantially overlapping subspaces, and an earlier adapter can often represent a later task. LiteLoRA gates reuse versus new adapters and reduces active adapters by 20 to 70% on standard benchmarks.
- Other titles seen in search results, not read: 2609.32332 (proximity-regularised merging for continual LoRA), 2606.19549 (predicting mergeability), 2604.16826, 2607.20561, 2609.22237, 2610.14558.
- So yes, "many adapters into one" is now an active literature (all 2026 preprints in my results). I have no verified number for how much behaviour is lost relative to the unmerged set.

---

## 3. Self-generated replay, distillation chains, model collapse, subliminal learning

### 3.1 Classical data-free methods (abstract level only)

- Learning without Forgetting, 1606.09282 https://arxiv.org/abs/1606.09282, ABS. Trains on new-task data only, preserving old outputs. CNNs. It may improve new-task results when datasets are similar.
- Deep Generative Replay, Shin et al., 1705.08690 https://arxiv.org/abs/1705.08690, ABS. A generator plus a solver. Image classification.
- Born-Again Networks, Furlanello et al., 1805.04770 https://arxiv.org/abs/1805.04770, ABS. Same-architecture students outperform teachers. These students are trained with the original labelled data each generation, so this is not evidence about data-free generation chains.

### 3.2 Model collapse and how much real data is needed

- Shumailov et al., Nature 2024 version of "The curse of recursion", 2305.17493 https://arxiv.org/abs/2305.17493, PART.
  - Mechanism: statistical sampling error removes tails first ("early collapse"), later modes merge and variance shrinks. A Markov-chain argument gives delta-function absorbing states.
  - LLM experiment: OPT-125m fine-tuned on wikitext2 with 5-way beam-search generation, 9 generations. Baseline perplexity 34. With no real data, the paper reports a perplexity loss of 20 to 28 points. With 10% of original data preserved (and 10 instead of 5 epochs), degradation was "minor" (Fig. 10b axis 32 to 42).
  - Caveats (partly INFERENCE): beam search is a low-diversity sampler, and the 10% run differs in epochs, so the 10% figure is confounded. Model is 125M and fine-tuning only.
- Gerstgrasser et al., 2404.01413 https://arxiv.org/abs/2404.01413, PART.
  - Replacing data each generation causes collapse. Accumulating real plus all synthetic data keeps test loss from rising: GPT-2 9M and Llama-2 12M, 42M, 126M on TinyStories; also diffusion and VAE.
  - Linear theory: replace gives error `σ²d/(T−d−1) · n`; accumulate gives a bound `σ²d/(T−d−1) · π²/6` independent of n; "replace-multiple" (same data volume but all synthetic) grows like log n.
  - Caveats: each generation pretrains a newly initialised model from scratch, TinyStories is itself GPT-generated, VAE error still rises slowly under accumulation, and one cited prior paper found fast deterioration even with accumulation.
  - It implies no minimum real fraction when all real data is retained at constant size and synthetic data is only added. The real fraction after n generations is 1/n.
- Dohmatob et al., "Strong model collapse", 2410.04840 https://arxiv.org/abs/2410.04840, ABS (via summary). Claims even about 1% synthetic data can cause collapse in their setting and larger training sets do not cure it. Larger models may worsen collapse in a simplified setting. **Unresolved:** I did not read the paper, so I cannot say how its setting differs from Gerstgrasser's. Do not quote "1%" as an LLM threshold.
- Other model-collapse papers seen in results, not read: 2412.14689, 2406.07515 (verification), 2509.04796 ("knowledge collapse": fluency survives but facts fail, snippet only), 2603.11784, 2607.17043.

### 3.3 Subliminal learning

Cloud et al., 2507.14805 https://arxiv.org/abs/2507.14805, PART.
- A teacher given a trait (owl preference, or misalignment via emergent-misalignment finetuning) generates numbers, code or chain of thought. A student fine-tuned on filtered data acquires the trait. Example: GPT-4.1 nano owl preference moved from 12% to over 60%. Misaligned-teacher students reached about 8 to 10% misaligned responses on free-form prompts versus under 1% for controls.
- It occurs only when student and teacher share initialisation (cross-family and mismatched pairs failed; GPT-4.1 and GPT-4o, said to share an initialisation, transferred). LLM classifiers and in-context learning failed to detect the trait. In-context learning of the same data did not reproduce the effect.
- Theorem 1: for equal initialisations, one sufficiently small gradient step on teacher outputs moves the student toward the teacher on the teacher's loss, regardless of the data. The experiments do not satisfy the theorem's one-step assumption.
- Authors' limitations: artificial tasks, and unknown what is and is not transmissible.
- A follow-up title, 2609.20027 "Can data attribution filter out subliminal learning? Not reliably", was seen in search results only (PRESS-level snippet), not read.
- **INFERENCE for sleep:** a sleep-time self-distillation or self-replay is the case where student and teacher share initialisation by construction, which is the most favourable condition for transfer. Content filtering of the replay data cannot be assumed to remove it.

### 3.4 What happens with iterative self-training of fine-tuned LLMs

- 2605.01130 "Iterative finetuning is mostly idempotent" (Roe et al.) https://arxiv.org/abs/2605.01130, PART. Qwen3 and Llama models up to 70B.
  - SFT and synthetic-document chains: traits mostly decay or persist, amplification is rare (12 of 270 SDF trials), brittle to seed and data size, and costs coherence.
  - **Key detail:** in the SFT/SDF algorithm each cycle is re-initialised from the initial model (Algorithm 1), so only the data flows forward.
  - Continual DPO (each cycle starts from the previous checkpoint, chosen = own outputs, rejected = initial model's outputs) reliably amplifies traits, and re-initialising each cycle removes it. LoRA rank 16 and Qwen3-4B for the DPO experiments.
  - **INFERENCE:** a sleep loop that keeps training the same weights on its own outputs resembles the continual setting, not the re-initialised SFT chain. The paper's reassuring result is for the latter.
- 2607.01763 "Denser ≠ Better" https://arxiv.org/abs/2607.01763, PART. On-policy self-distillation (SDPO, student and teacher the same model with extra context) specialises quickly but forgets more than GRPO. In a MATH→SCIENCE→TOOLUSE→CODING sequence, one variant (SDPO with 5% teacher EMA) collapsed after stage three with endless repeated "\boxed" tokens, attributed to a self-reinforcing teacher-student loop. Weight-delta drift (spectral shift, subspace rotation) was larger than GRPO's.
  - Forgetting concentrated on tasks at intermediate embedding distance (MMD), with near and far tasks mostly unaffected or improved.
  - Their theorem: the teacher-projection target has excess KL drift versus the KL-minimal sequence-reward policy.
  - Models include Olmo-3-7B variants. Preprint, under review.

### 3.5 LLM self-generated replay: how well does it cover the model's knowledge?

- Huang et al., Self-Synthesized Rehearsal (SSR), 2403.01244 https://arxiv.org/abs/2403.01244, PART. Llama-2-7B, Llama-2-7B-chat and Alpaca-7B, LoRA rank 8, SuperNI sequences of 5 and 10 tasks.
  - Pipeline: the base model generates instances by in-context learning from K=2 demonstrations (taken from 1% of the original training data), the latest model rewrites the outputs, then K-means selection.
  - Results (5 tasks, Llama-2-7B, average over 3 orders): SSR average ROUGE-L 52.16 with backward transfer −0.79; random 10% real replay 50.32 / −3.02; random 1% real 49.67 / −4.02; no rehearsal 19.03 / −42.39. Multi-task upper bound 53.05.
  - Ablations: without the output-refinement step results are worse. Variants with new demonstrations or input-only demonstrations also work, but some real or hand-made task examples are still needed, so this is not fully data-free.
  - Authors' explanation for synthetic beating real: synthetic text has lower perplexity under the current model and is easier to fit. This is their interpretation.
  - Their EWC baseline did essentially nothing (average 15.94 versus 17.67 for no rehearsal; backward transfer −46.30) and L2 gave 29.22. Hyperparameter tuning of those baselines is not described in the excerpt.
- 2610.12602 "Self-preservation in language models" (Cho, Chunara, Behrouz) https://arxiv.org/abs/2610.12602, PART. Qwen3-0.6B/1.7B/4B, full-parameter fine-tuning on MedQA or C-STANCE, with a frozen pre-update model generating replay from a "Task Bank" of task descriptions.
  - Generated replay protected held-out tasks that were not in the bank. Combining partial banks improved held-out retention at all three sizes.
  - Model-guided selection of replay targets helped only at 4B (+2.75 points retention).
  - Under a fixed example budget, replay replaces new-task examples and lowers new-task gains.
  - The bank is still supplied by a human, and replay can reproduce the model's own errors.
  - Learning rate dominated: for Qwen3-1.7B on C-STANCE, held-out retention was 97.0% at lr 1e-7, 89.6% at 1e-6, 82.2% at 3e-6 and 32.9% at 1e-5. The 1e-5 run failed to adapt at all.
  - One-seed two-stage pilot showed mixed results. A bank-expansion pilot gave negative accuracy changes.
- There is no direct, published "coverage" measure of how much of a model's knowledge its own samples span. What exists indirectly: Shumailov's tail loss (section 3.2), and the held-out-retention measure above (retention gains but only within what the bank reaches).

---

## 4. Interference measurement and scalability

| Method | What I verified | Viability at ~1B parameters (INFERENCE unless noted) |
|---|---|---|
| Influence functions with EK-FAC (Grosse et al., Anthropic, 2308.03296) https://arxiv.org/abs/2308.03296 | ABS: scaled to LLMs up to 52B; patterns sparse; generalisation more abstract with scale; influence goes to about zero when key phrase order is reversed. | Authors apply it to far larger models, so size is not the barrier. I did not find a compute number for 1B, and the cost of scoring many candidate sequences is the bottleneck. Treat as an audit tool, not an inner-loop signal. Related 2025–26 variants (2505.05017, 2509.23437, 2609.37842) were seen but not read. |
| A-GEM (Chaudhry et al., 1812.00420) https://arxiv.org/abs/1812.00420 | ABS: one-example-at-a-time protocol, near-EWC cost, best accuracy-efficiency trade-off on standard benchmarks (image/lifelong benchmarks). | Needs one extra backward pass on a memory batch and one dot-product projection per step, i.e. roughly doubled step cost. Feasible. GEM proper needs a quadratic program over per-task gradients (4 GB per fp32 gradient at 1B), so it is not. I did not test either on LLMs. |
| MIR, MER, PCGrad | Not opened (MIR cited in 2610.12602 as 1908.04742, CITED). | Unknown from this session. |
| Fisher/EWC on LLMs | 2403.01244 table (PART): EWC with LoRA on Llama-2-7B ≈ no-rehearsal. | Diagonal Fisher on LoRA parameters is cheap but has the one negative result above. I did not find an LLM paper showing EWC working well. 2603.18596 "EWC done right" (title only, not read). |
| Gradient interference as plasticity signal | Lyle 2023 (PART): gradient descent produced more negative gradient interference and larger Hessian outliers than a random walk of equal step norm. | Quantity can be estimated from mini-batch gradient cosines. |
| Forward-computable token-level interaction (2609.33620) | ABS only | Unverified. |
| Distance-based interference proxy | 2607.01763 (PART): forgetting peaked at intermediate embedding MMD between source and target tasks. | Cheap (embeddings only). Consistent in direction with the null-space-angle story in section 6, but a different quantity and a different method (SDPO), so not a test of replay. |
| Knowledge-neuron locality | Niu et al. 2405.02421 https://arxiv.org/abs/2405.02421, ABS: the thesis is "at best an oversimplification"; editing the identified MLP weights also changes linguistic phenomena. Hase et al. 2301.04213 https://arxiv.org/abs/2301.04213, ABS: Causal-Tracing localisation does not predict which layer edits best. | Do not use localisation as an interference predictor. |

Other empirical reminders: Luo et al. 2308.08747 (ABS) found forgetting increased with scale from 1B to 7B.

---

## 5. Scheduling, drift detection, sleep triggers, "uniform replay is hard to beat"

- Chaudhry et al. "On tiny episodic memories", 1902.10486 https://arxiv.org/abs/1902.10486, ABS. A small memory trained jointly with the current data beat specialised methods on four supervised benchmarks, and repeated training on tiny memories did not hurt generalisation (gains of 7 to 17% with one example per class). Image benchmarks.
- LLM-specific, from SSR Table 1 (PART, section 3.5): random 1% and random 10% real replay are close to each other (49.67 vs 50.32 average ROUGE-L). K-means selection gains little (50.02 at 1%, 50.31 at 10%), and the authors write that significant differences "may not be observed" at 10%. In one ordering, 1% beat 10%. This supports "uniform random is hard to beat" for LoRA instruction tuning of 7B models on SuperNI, with small differences and 3 orderings.
- Where it does not hold: SSR's own synthetic rehearsal exceeded random real replay by about 2 points. Selection informed by the incoming task helped only at 4B in 2610.12602.
- Ibrahim et al., 2403.08763 https://arxiv.org/abs/2403.08763, ABS. Re-warming the learning rate, re-decaying it and replaying previous data matched full retraining on 405M-parameter models (hundreds of billions of tokens), and for a weak distribution shift also at 10B. The abstract page does not give the replay fraction.
- Cho et al. "Forget forgetting: continual learning in a world of abundant memory" (ICLR 2026) was seen only as a citation in 2606.24752 and 2610.12602 (CITED). The 2610.12602 related-work text says that with abundant replay memory the remaining problems shift to plasticity and compute.
- Task-free CL: Aljundi et al. 1812.03596 https://arxiv.org/abs/1812.03596, ABS. Reframes CL as a boundary-free stream and extends Memory Aware Synapses with a rule for when to update importance weights. Tested on small robotics/face tasks, not LLMs.
- Learning-rate scheduling is the strongest scheduling lever I found: in 2610.12602 retention fell from 97% to 33% across a 100x learning-rate range. Springer and Ibrahim also treat learning rate as central.
- Directly adjacent work on a "sleep" for LLMs: Behrouz, Hashemi, Javanmard, Mirrokni, "Language models need sleep", 2606.03979 https://arxiv.org/abs/2606.03979, ABS. A two-stage scheme: memory consolidation (distilling a smaller model's memories into a larger one, combining on-policy distillation with RL-based imitation) and "dreaming" (RL-generated synthetic curricula). Only the abstract was read. The coordinator should check how it handles the collapse and subliminal issues in sections 3.2 to 3.4.
- I found no verified LLM drift-detection or sleep-trigger paper.

---

## 6. "Replay can provably increase forgetting" (S01): identification and resolution of the ewk.md ambiguity

**Paper:** Mahdaviyeh, Lucas, Ren, Tolias, Zemel, Pitassi, "Replay Can Provably Increase Forgetting", arXiv 2506.04377, CoLLAs 2025. https://arxiv.org/abs/2506.04377. Access: PART (I read pages 1 and 3 to 6, and appendix pages 13, 14, 16, 18, 20, 27; the MNIST results section on pages 7 to 9 was not in my retrieved pages).

**Setting:** noiseless over-parameterised continual linear regression with min-distance (GD/SGD) updates. Eq. 2: after task t the error vector is projected onto that task's null space `P_t`. Forgetting `F = (1/(T−1)) Σ ‖X_t (w_T − w*)‖²`.

**What the paper actually says about angles:**

1. **Safe-replay condition (Prop. 3.6, average case, two tasks).** If `‖P2 P1‖_op ≤ √2/2` and `w* ~ N(0, I)`, then forgetting with replay of any number of samples, averaged over w*, is never larger than without replay. `‖P2 P1‖_op` is the cosine of the smallest principal angle between the two null spaces. The condition is therefore: **smallest principal angle ≥ π/4** (the paper's text says "larger than π/4"). So ewk.md line 143 ("guaranteed safe above π/4") is correct. It is a sufficient condition only. It is not shown that replay harms at every angle below π/4.

2. **"Forgetting is worst at intermediate angles" is a statement about forgetting WITHOUT replay.** Appendix A.3 and the Section 3.2 text state that, in the two-task case, forgetting without replay is `∝ (a1ᵀa2)²(1 − (a1ᵀa2)²)`, maximised at π/4, citing Evron et al. 2022. Section 3.3 says forgetting is small away from π/4 "either larger or smaller". So the ambiguity noted in ewk.md lines 144 and 120 resolves as: that sentence is the Evron et al. 2022 result and is not a claim about replay's extra harm.

3. **Where replay does harm (Thm. 3.5 and Thm. C.1).**
   - The constructions put the two null spaces nearly aligned, at a very small angle (task 1 null space close to but not equal to a task-2 null-space direction). Without replay, forgetting is tiny (`ε²(1−ε²)a²`, with ε = √(1/63) in the d=3 proof).
   - Replaying random samples from task 1 reduces the second task's null space to a subspace that, with high probability, excludes the direction `v2`. This slightly rotates that null space, increasing the angle with task 1's null space "but not by too much" toward π/4, where no-replay-style forgetting is largest. So expected forgetting rises.
   - Thm. 3.5 requires dimension conditions (`c1 < d`, `c2·m < d−1`, `d−1 < exp(m log m)/c3`); the paper notes that it still allows m linear in d. Thm. C.1 gives d = 3, one replayed sample.
   - The paper stresses that some individual replayed directions would reduce forgetting. The theorem is about expectation over random picks.
   - Worst case (Thm. 3.2): choose tasks and a replay sample so that forgetting goes from O(1/T) to Θ(1). This is the three-vector "error exchange" (x1, x2, x3) example in ewk.md.

4. **Effect of the replay on the angle:** harm arises when replay pushes a small angle upward toward the π/4 peak. It does not arise from replay acting at already-intermediate angles in any simple sense. For intermediate angles the paper gives no theorem either way.

5. **Corrections to ewk.md (v2 notes, lines 143 to 146):**
   - Item 1 stands: the harmful zone is below π/4, and Prop. 3.6 defines the guaranteed-safe region above it.
   - Item 2: ambiguity resolved as above. Also worth noting: the v1 statement that the worst case sits "around π/4" was actually true for **no-replay forgetting** (Evron) and wrong only as a description of where replay harms. So v1 conflated two quantities, and v2's correction should say so.
   - Add: the proposition is averaged over `w*`, covers two tasks only, and the null-space quantity is a principal angle, not an angle between task data vectors.
   - Add: an MLP trained on near-orthogonal null-space angles (near π/2) behaved differently from the linear model (appendix Fig. 9), so the safe-above-π/4 picture should not be assumed for networks.
   - The MNIST numbers in ewk.md lines 123 and 124 (single "4" helping 3%, 1 to 2 samples hurting, 4+ helping) were not in my retrieved pages. I could not re-verify them.
   - ewk.md line 145 (why harm is small in practice) is the entry's own reasoning, not the paper's. The paper itself says (Appendix, narrower network) that regularisation and small learning rates may affect the pattern and leaves that out of scope.

**Cross-link:** 2607.01763 (section 3.4) found, for LLM self-distillation (not replay), that forgetting concentrated on tasks at intermediate embedding distance. It cites the same line of linear-model theory. This is suggestive of a similar non-monotonic geometry in LLMs but is not a test of Mahdaviyeh et al.

---

## 7. Findings ranked by decision relevance

1. Plasticity loss in LMs is demonstrated only at small scale (5M to 314M) with tiny effects (about ±1.5% AUC). Extrapolation puts onset for 1B models near 1.8T training tokens (2606.24752). No paper found measures it for LoRA or repeated fine-tuning at 1B+. Do not build the framework around it.
2. The larger LLM-scale hazard is fine-tuning sensitivity growing with pretraining length (2503.19206). Anchored regularisation toward the base trades fit for retention (their Thm 4.7, linear setting).
3. Anchor-to-init beats decay-to-zero in small-network plasticity work (L2 Init, 2308.11958) and PAC-Bayes uses an init-centred prior (1703.11008). In LLMs, LoRA is already base-anchored; plain weight decay on full weights was not (2405.09673). Higher pretraining weight decay did improve later fine-tuning at up to 4B (2602.11137).
4. Real-data accumulation prevents collapse in small LMs (2404.01413); replacement does not. Shumailov's 10% retained-data run is confounded. "Strong model collapse" claims a 1% synthetic threshold in its own setting (ABS only). Unresolved, so no verified minimum real fraction for 1B LLMs.
5. Subliminal learning (2507.14805) is strongest when student and teacher share weights, which is exactly self-distillation. Content filtering and in-context detection did not remove it.
6. Iterative SFT from a re-initialised base mostly decays traits, but continual training from the previous checkpoint (the DPO case) amplifies (2605.01130). A sleep loop is closer to the latter.
7. On-policy self-distillation is a fragile consolidation signal: more forgetting than GRPO, and a collapse case from artifact amplification (2607.01763).
8. Self-generated replay works in LLM continual tuning when anchored by some real task examples or descriptions (SSR 2403.01244; 2610.12602), and beat 1 to 10% random real replay by about 2 points in one benchmark. Coverage of what the model knows is still bounded by the supplied bank. Learning rate is the dominant forgetting knob (retention 97% to 33% over 100x).
9. Random replay is close to selected replay in LLMs (SSR Table 1); EWC with LoRA was no better than no replay in that one test (2403.01244). Per-sample-gradient methods like GEM are impractical at 1B; A-GEM-style single-projection is feasible but untested on LLMs here. Knowledge-neuron localisation does not predict where to intervene.
10. S01 is arXiv 2506.04377. Safe region: smallest null-space principal angle ≥ π/4 (Prop. 3.6, two tasks, average over w*). The "intermediate angles" remark is the no-replay result of Evron et al. Replay harm occurs when replay pushes a small angle toward π/4. ewk.md's v1 "around π/4" was a no-replay fact misapplied to replay.
