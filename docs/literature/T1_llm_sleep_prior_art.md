# T1: What has actually been done under the name "sleep" / offline consolidation for LLMs and neural nets

Thread T1. Written 2026-11-04. Not committed.

**Access tags** (same scheme as `ewk.md`, with two additions):
- **FULL**: whole body text read (reference lists skipped).
- **PART**: only some pages or sections read. Most alphaXiv `answer_pdf_queries` reads are PART, because that tool returns the most relevant pages, not the whole paper.
- **PART-S**: PART, and the text I saw was a summary written by the fetch tool's small model, not the paper itself. Treat numbers as needing a re-check.
- **ABS**: abstract or search snippet only.
- **BG**: not retrieved in this thread. Stated from background knowledge; unverified. Do not build on it.

**Not done:** OpenReview was not reached. A web search for the OpenReview page did not return it. The Behrouz paper was read as arXiv v2 (10 Jul 2026), whose footnote says an OpenReview version has existed since September 2025. Differences between the OpenReview version and v2 are unknown.

---

## 0. Ten things that matter most (short form)

1. **Two different papers are titled "Language Models Need Sleep."** Behrouz et al. (arXiv 2606.03979) is the Google one. Lee et al. (arXiv 2605.26099) is a different CMU/UMD paper that was originally titled the same and is now "Do Language Models Need Sleep?" Lee's footnote says it renamed itself to avoid confusion with the OpenReview paper. They share no method.
2. **Behrouz's "sleep" is architecture-bound, not a drop-in procedure for an existing LLM.** It needs a multi-frequency MoE (Hope/CMS) that grows low-rank experts, and it distils self to self with RL imitation. Protection of old knowledge is by freezing plus capacity growth. No old-data replay.
3. **Behrouz has no uniform-random-replay baseline, no seeds or confidence intervals, and unequal baseline budgets.** Its ablation gaps are 1 to 2.4 points on avg@16. I found no measurement of general-capability drift (MMLU-style) across repeated sleeps.
4. **Lee's "sleep" is a compute-depth trick with fast weights.** It is trained end-to-end through the sleep. It does not study forgetting.
5. **Context to weights is well-trodden.** Context distillation (Askell 2021, Snell 2022), prompt distillation for knowledge (Kujanpää), DiSC, OPCD, SDFT, Cortex and SCoL all exist. Details in section 7.
6. **Self-generated replay with a KL penalty nearly eliminates forgetting in LLMs** (Marek et al., 2605.26097; small models plus one 1B model; single new task). This is pseudo-rehearsal at LLM scale. Any sleep claim has to beat it.
7. **The ANN-sleep literature mostly has few cycles.** Tadros SRC: one sleep per task, 4 sleeps. Golden: 2 tasks. Bazhenov 2026: one sleep after up to 5 tasks. Deperrois: not a continual-learning paper. Nobody I found ran 10 or more wake/sleep cycles on a pretrained LLM with held-out capability checks.
8. **SRC alone loses to small real-data rehearsal** on 3 of 4 reported benchmarks. It wins only on CIFAR-10-on-pretrained-features. Its best result is rehearsal plus SRC.
9. **Several "sleep" papers are weak evidence.** SleepGate's baselines score about 8% even at n=1, where there is no interference. SCM reports zero variance on a self-built 22-fact test and no code. Dennis et al. retrain a fresh LoRA each cycle, so it is not multi-cycle continual updating.
10. **The gap:** a controlled, many-cycle (10 or more) sleep protocol on a pretrained LLM, with matched-compute uniform and self-sampled replay baselines, a per-cycle general-capability probe, and an ablation of the replay-selection policy. I did not find it. Section 9.

---

## 1. Table of sources

| ID | Source | URL / ID | Access |
|---|---|---|---|
| A1 | Behrouz, Hashemi, Javanmard, Mirrokni. *Language Models Need Sleep* (v2) | arXiv 2606.03979 | FULL |
| A2 | Lee, McLeish, Goldstein, Fanti. *Do Language Models Need Sleep? Offline Recurrence...* | arXiv 2605.26099 | PART |
| A3 | Lin, Snell et al. (Letta). *Sleep-time Compute* | arXiv 2504.13171 | PART |
| A4 | Xie. *Learning to Forget / SleepGate* | arXiv 2603.14517 | PART |
| A5 | Shinde. *SCM: Sleep-Consolidated Memory* | arXiv 2604.20943 | PART |
| A6 | Dennis et al. *Weight-based consolidation vs cascading compaction* | arXiv 2605.24657 | PART |
| A7 | Wang et al. *Self-Consolidating LMs (SCoL)* | arXiv 2605.07076 | PART |
| A8 | Zhang et al. *Cortex: Consolidating Context into Parametric Memory* | arXiv 2610.20399 | PART |
| A9 | Ye et al. *Auto-Dreamer* (agent memory, text) | arXiv 2605.20616 | PART |
| A10 | Haber, Tononi. *Do agents need to sleep?* (position paper) | arXiv 2611.02216 | PART |
| A11 | Jayasinghe et al. *When Robots Sleep* | arXiv 2606.17493 | PART |
| A12 | Lu et al. *MSSR* (replay scheduling) | arXiv 2603.09892 | PART |
| A13 | Chen et al. *On-Policy Replay (OPR)* | arXiv 2605.29495 | PART |
| A14 | Marek et al. *Forgetting in LMs: Capacity, Optimization, Self-Generated Replay* | arXiv 2605.26097 | PART |
| A15 | Kim, Choi. *Hippocampal consolidation without replay (DeltaHippo)* | arXiv 2611.01952 | PART |
| A16 | Zhang et al. *Replay in the Silent Degrees of Freedom* | arXiv 2609.31630 | PART |
| B1 | Tadros, Krishnan, Ramyaa, Bazhenov. *Sleep-like unsupervised replay (SRC)*, Nat Commun 2022 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9755223/ | PART-S |
| B2 | A. Bazhenov, Delanois, Krishnan. *Not Just After One* (2026) | arXiv 2606.08447 | PART (4 pages, short paper) |
| B3 | Golden et al. *Sleep prevents catastrophic forgetting in spiking networks*, PLoS CB 2022 | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010628 | PART-S |
| B4 | González et al. *Can sleep protect memories from catastrophic forgetting?* eLife 2020 | https://elifesciences.org/articles/51005 | **PART-S** (see note below) |
| B5 | Sorrenti et al. *Wake-Sleep Consolidated Learning (WSCL)* | arXiv 2401.08623 | PART |
| B6 | Deperrois et al. *Perturbed and adversarial dreaming*, eLife 2022 | https://elifesciences.org/articles/76384 | PART-S |
| B7 | Robins 1995 pseudo-rehearsal; Crick and Mitchison 1983; Hopfield, Feinstein, Palmer 1983; Hinton et al. 1995 wake-sleep | search snippets only | ABS / BG |
| C1 | Snell, Klein, Zhong. *Learning by Distilling Context* | arXiv 2209.15189 | PART |
| C2 | Kujanpää et al. *Efficient Knowledge Injection via Self-Distillation (prompt distillation)* | arXiv 2412.14964 | PART |
| C3 | Padmanabhan et al. *DiSC: Context distillation retains post-training capabilities* | arXiv 2602.16093 | PART |
| C4 | Ye et al. *On-Policy Context Distillation (OPCD)* | arXiv 2602.12275 | PART |
| C5 | Shenfeld et al. *Self-Distillation Enables Continual Learning (SDFT)* | arXiv 2601.19897 | PART |
| C6 | Kim et al. *Why Does Self-Distillation (Sometimes) Degrade Reasoning?* | arXiv 2603.24472 | PART |
| C7 | Ichihara et al. *Privileged Solutions or Context-Induced Teacher Behavior? (OP2SD)* | arXiv 2608.09228 | PART |
| C8 | Davion, Rui. *Privileged Context as Drift in OPSD* | arXiv 2610.07842 | PART |
| C9 | Goyal et al. *Context-Parametric Inversion* | arXiv 2410.10796 | PART |
| C10 | Neeman et al. *DisentQA* | arXiv 2211.05655 | PART |

**Note on B4 (González 2020).** The page I fetched (elifesciences.org/articles/51005) returned a description of a *thalamocortical Hodgkin-Huxley model with sequence replay* (STDP, no external input in sleep, two-to-three sequences). Whether that is the paper titled "Can sleep protect memories from catastrophic forgetting?" I did not verify against the title. The Golden et al. summary and the Bazhenov 2026 text both cite González 2020 for "co-replay of recent and old memories during sleep forms orthogonal representations", which matches the model described. I treat the match as likely but unconfirmed.

**Leads seen but not read:** GATES (arXiv 2602.20574, consensus-gated self-distillation), Cartridges (Eyuboglu 2025), Deep Context Distillation, SELF-PARAM, SuRe (arXiv 2511.22367), Kotha and Liang "Replaying pre-training data improves fine-tuning" (arXiv 2603.04964), Retaining by Doing (arXiv 2510.18874), "Nested Learning" (Behrouz 2025; known here only through A1's description), HippoRAG (not read; retrieval over an external graph, no weight change, so a different mechanism from everything here), MemGPT, Mem0.

---

## 2. Per-source detail: "sleep" for LLMs

### A1. Behrouz et al., *Language Models Need Sleep* (FULL)

**Algorithm.** Two stages.
1. *Memory consolidation ("knowledge seeding").* The model is a continuum-memory system (CMS): a chain of MoE-MLP blocks, each updated at its own frequency. Before a faster block's base-weight update is applied, the procedure runs *compute, consolidate, update*:
   - Teacher = the model before the update, with its accumulated experts.
   - Student = the model with the prospective update applied, plus a **newly added low-rank expert** (A, B) in the next slower block. The new expert is the only thing trained. "Freeze all the parameters in the student model" except the expanded ones.
   - The loss is a generalised-KD on-policy distillation mix plus an RL "Learning to Imitate" reward. The reward is semantic similarity from a frozen reward model plus a Levenshtein term.
   - After consolidation, low-rank experts in the faster block are reset ("synaptic pruning").
2. *Dreaming.* Built on SEAL (Zweiger 2025). The model samples synthetic "dreams" from context. A random expert is chosen at the MoE routers "for novelty". Dreams are scored by the gradient of the SFT loss; the top-k plus b random ones are kept. Each is fine-tuned with LoRA in an isolated copy and rewarded 1 if the downstream score improves, using ReST-EM.

**What weights change.** Added low-rank experts in slower blocks, then base blocks per schedule. Dreaming: LoRA, then the policy via ReST-EM.

**Replay data source.** Self-generated (teacher samples). No old real data. Teacher is the pre-update self.

**Protection of old knowledge.** Freezing plus capacity growth. The paper reframes forgetting as "insufficient capacity." There is no replay of old tasks and no explicit retention probe that I saw.

**Cycles tested.** The notion of "levels" (Hope-1/2/3; 1k to 5k to 10k-step frequencies) is the number of consolidation stages in the hierarchy, not the number of full wake/sleep days. In the example given, a fast block is consolidated 10 times before the slow one updates. Total counts of sleeps in each experiment are not reported in the text I read.

**Models and benchmarks.**
- Class-incremental text classification (CLINC, Banking77, DBpedia) on Llama-3B/8B, vs ICL, EWC, InCA and Hope.
- Long-context QA (LongHealth, QASPER, MK-NIAH) vs ICL, DuoAttention, Cartridges.
- Continual translation (MTOB plus Manchu).
- BABILong up to 10M tokens.
- Math: Qwen3-1.7B/8B, AIME-24/25 and HMMT-25.
- Knowledge incorporation (SQuAD, SEAL setup) and ARC few-shot (Llama-3.2-1B).

**Numbers (Tables 1 to 4).**
- Qwen3-8B math avg@16 (AIME24/AIME25/HMMT): Base 73.8/68.1/42.4; SFT 75.5/66.4/43.7; GRPO 76.4/68.1/44.9; OPSD 76.6/67.4/45.1; Sleep 79.2/69.0/46.1.
- Ablations at 8B: removing imitation learning gives 76.8/67.9/45.0; removing the semantic reward 78.9/69.2/44.5; removing expansion 78.2/67.9/44.9.
- Knowledge incorporation: SEAL 46.7 (n=1) and 43.2 (CPT n=200); Sleep four-level 48.9 and 46.2; "removing dreaming" 35.7 and 36.2.
- ARC: ICL 0, TTT 10, SEAL 72.5, Sleep 80.

**What is wrong with it.**
- *No uniform or random replay baseline.* The comparison set is ICL, EWC, InCA, SFT, GRPO, OPSD, SEAL. Nothing replays old data.
- *Unequal budgets.* Table 5: GRPO 500 steps, SFT 100, Sleep 100. B.5 says SFT is 4x cheaper per step, and wall-clock-matched claims are made only against SFT.
- *Small ablation gaps, no variance.* The ablation deltas are 0.4 to 2.4 points. No seeds or confidence intervals are mentioned. I read the main text and appendix B and did not find any.
- *ARC numbers are coarse.* 8 held-out tasks and 5 dreams each gives 40 trials. 80% and 72.5% would then be 32/40 and 29/40, a 3-trial difference. This is my inference from the stated protocol, not a number in the paper.
- *Retention claims rest on the translation task and classification, not on general capability.* I found no MMLU/IFEval-style probe across repeated sleeps. The paper says SFT and Cartridges "faced catastrophic forgetting" on the translation task but gives no figures in the text.
- *Architecture dependence.* Headline results use Hope, the authors' own architecture. Only the math and SEAL-style rows use a standard transformer variant.
- *Editorial slips.* Ablations are said to be "reported in Figure 1" (they are in a table), and the same Tables 1 and 2 numbers appear under different captions.
- *Self-priority claim.* The paper says its work "is older" than OPSD-family methods. That is an assertion about OpenReview timing I could not check.

### A2. Lee et al., *Do Language Models Need Sleep?* (PART)

**Algorithm.** An SSM-attention hybrid with a fixed context window L. When the window fills, the model does N recurrent passes ("sleep") over the window, updating fast weights in the SSM blocks by a learned local rule, then clears the KV cache. Trained end-to-end by backpropagating through all passes (Algorithm 1). No external data in sleep.

**What changes.** Fast weights (state), not slow weights. They are lost when the sequence/state is dropped.

**Benchmarks.** Cellular automaton Rule 110 (t=32), Depo (k-hop graph retrieval), GSM-Infinite on Jet-Nemotron 2B and Ouro 1.4B.

**Numbers.** Rule 110: no loop about 10%, two loops about 20%, three or four loops above 30%. GSM-Infinite, Ouro 1.4B, six ops: 0.419 to 0.615 with four loops. Jet 6-op: 0.742 to 0.812.

**Weaknesses.**
- *It is about reasoning depth over evicted context, not consolidation of knowledge across deployments.* No forgetting or retention across sleeps is studied.
- *Seeds fixed to match data order*, but no repeats or error bars appear in the pages I read.
- *Training cost grows linearly with N* and the authors report instability as a limitation.
- Good for ruling out one reading of "sleep" (more compute before eviction helps on deep sequential tasks). It says nothing about long-run weight consolidation.

### A3. Sleep-time compute, Letta (PART)

Not a weight update. The model is prompted offline to re-write the context into an inference-enriched context c'. Test-time compute for the same accuracy is about 5x lower on Stateful GSM-Symbolic and Stateful AIME; scaling sleep-time compute adds up to 13% and 18%. Amortised cost per query falls about 2.5x with 10 queries per context. Limits stated: gains depend on query predictability; at high test-time budgets test-time-only is better (also on the SWE-Features case study). Behrouz explicitly distinguishes his work from this one.

### A4. SleepGate (PART)

A learned forgetting gate and consolidation module over the **KV cache**, with "sleep micro-cycles" triggered by attention entropy. Trained from scratch, **793K-parameter, 4-layer transformer on synthetic data.** Reported: 99.5% retrieval at interference depth 5 and 97.0% at depth 10, vs all five baselines under 18%. **Problems:**
- Baselines at n=1 (no interference at all) score 6.5 to 17.5% (full KV cache 8.0%) while SleepGate gets 82.5%. At n=1 the task has no stale values, so low baseline numbers point to under-training or a mismatched setup, not to interference.
- Accuracy collapses at larger depth (73.5% at n=15, 30.5% at n=20, 16.5% at n=30).
- Not tested on pretrained models or real language. The paper says so.
- No weights consolidate. This is cache management.

### A5. SCM (PART)

Pure external memory: a NetworkX graph with sleep-like "NREM" (Hebbian edge strengthening, 20% downscale per cycle), "REM" (random walks adding edges) and thresholded forgetting. LLM (Llama 3.2 2B) is used only for concept extraction. Benchmark: eight self-built tests, "perfect average score of 1.00", zero variance across five runs. 22 explicitly stated facts. Not tested: implicit preferences, temporal reasoning, inferential links from REM (the paper says REM-generated associations "are never tested"). Code not public. A forgetting-weight bug that gave 0% noise removal was fixed on the same benchmark. Low evidential weight.

### A6. Dennis et al., nightly LoRA vs compaction (PART)

Pipeline: LLM reflection extracts facts, a strong LLM writes about 20 paraphrased Q/A conversations per fact, a LoRA (r=16) on Qwen2.5-7B-Instruct trains for 8 epochs. Results over n=10 synthetic software-dev conversations: consolidation 80.4 ± 1.3% (no context at test) vs cascading compaction 36.8 ± 3.0% after three cycles; floor 11.8%, full-context ceiling 90.1%.

**What it does and does not show.**
- *Each cycle trains a fresh LoRA on facts from all conversations so far.* The consolidation curve is flat at 80 to 82% because it is a retrain, not a sequential update of one adapter. The text says "fresh LoRA."
- *Both conversations and tests are LLM-generated, and judging is by an LLM* (Claude Sonnet 4 throughout). Circularity risk.
- *No general-capability evaluation* (stated: no MMLU or HumanEval; also safety unmeasured).
- *Baseline is compaction only.* No retrieval-over-extracted-facts baseline, no "keep the fact list in context" baseline.
- Useful as a cost datapoint: about 1.5 h on an A100-40GB, 8.2 h on an RTX 5090 per conversation.

### A7. SCoL (PART)

A model learns to emit text naming which of its own layers to update by LoRA for each incoming context; trained with meta-RL (IPO) over a drifting model. SQuAD stream of 100 passages: immediate accuracy 35.4% and **retention 20.5%**, vs prompting-only 28.2%, SEAL-continual retention 1.3%. Fixed "last 10 layers" ablation gets 28.8% immediate and 20.1% retention, which is close to SCoL's retention. Long-context: SCoL 42.3% vs Batch TTT 39.5% (short), 37.0% vs 41.5% (long). Limits stated: overhead, mode collapse over IPO rounds. Retention of 20% means most consolidated content is lost over 100 passages; it shows partial control of forgetting, not solution of it.

### A8. Cortex (PART)

Formalises "memory consolidation" as internalising context-induced gains while preserving non-target capability (CR/RR metrics). Method: the frozen pre-update model generates prospective queries and answers conditioned on the context; a *context-conditioned teacher* supervises a context-free student with KL plus hidden-state MSE, on mergeable FFN updates (low-rank "synaptic" or added-channel "structural"). Sequential mode freezes everything old and trains only new parameters; **query replay** keeps past queries and regenerates targets with the pre-update checkpoint.

Results: 4B to 120B models. After 50 sequential factual edits on a 14B model, Structural gets 88.9% CR and 99.8% RR (exceeds AlphaEdit by 66.2 and 3.8 points). Failures: on GPT-OSS-20B τ²-Bench, Structural CR 38.5% and Synaptic **−92.3%** (below vanilla), Synaptic RR 98.5%; batch API learning CR drops from 91 to 93% (single) to 45 to 59% (batch) on 20B; suppression costs 9 to 20 points on neighbouring knowledge; on GPT-OSS-20B ALFWorld Structural RR 95% vs Synaptic 23%. Contexts were "manually constructed to elicit substantial gains" per model-task pair. No sleep-vs-random-replay control is described; the "replay" is of past queries.

### A9. Auto-Dreamer (PART)

Dreaming here is a learned LLM that **rewrites regions of a text memory bank**. No weights of the task agent change. GRPO-trained consolidator (Qwen3-14B init), ScienceWorld SR 41.1% vs UMEM 34.1% with about 12x smaller bank; held-out ALFWorld 60.2%. The failure case shows over-abstraction deleting a needed fact (agent score 1.0 untrained vs −1.0 trained for one task). A memory-system result, not a weights result.

### A10. Haber and Tononi, *Do agents need to sleep?* (PART)

Position paper. Proposes mapping "ringing, resonance, renormalisation" to digital memory operations and states a **falsifiable prediction**: selectionist consolidation will beat instructionist (Mem0-style ADD/UPDATE/DELETE) strategies as memory stores grow. No experiments. The prediction is untested.

### A11. When Robots Sleep (PART)

Wake-sleep for shared-policy robot skills: after each new skill, offline consolidation uses frozen per-skill memories (critic plus 1000-state buffer, or frozen actor plus observation buffer) and Nash-bargained gradients. MT5: +64% average success and 2.0x pairwise reliability over the strongest non-oracle baseline. Ablations use **seed 42 only** (Table III). Baselines include replay-style methods (CLEAR, DER, A-GEM, EWC). Sleep skipped for the first skill. Five-skill streams only.

---

## 3. Per-source detail: replay, scheduling and LLM continual fine-tuning (non-"sleep" but the right baselines)

### A12. MSSR (PART)

Ebbinghaus-inspired per-sample memory strength and expanding replay intervals, with LoRA on Mistral-7B, Llama-3.1-8B, Qwen2.5-7B, Gemma2-9B. 3-task (Alpaca-GPT4, GSM8K-RFT, MATH) and 11-task sequences. Baselines: No Replay, **Fixed Replay (uniform, constant subset)**, loss-triggered, accuracy-triggered. 3-task averages Qwen: fixed 52.8, MSSR_full 54.5. Gains of about 1 to 2 points over fixed. **Single seed (42).** Internal inconsistencies: Table 3 and Table 6 both report "buffer size vs average retention under the Ebbinghaus scheduler" with different values (512: 62.1 vs 68.7; 4096: 80.3 vs 76.8), and Table 4 vs Table 8 give different accuracy values for the same interval comparison. MSSR_full is not best on every cell (e.g. Gemma MATH1: Fixed 0.873 vs 0.867). Real stored data is replayed during training; there is no offline phase.

### A13. On-Policy Replay (PART)

Replays model-generated responses (reward-filtered) to historical prompts as plain SFT. TRACE, 8 tasks, Qwen2.5-7B-Instruct, Qwen3-8B, Llama3.1-8B; 3 seeds. At 1% replay budget Qwen2.5: SFT BWT −13.93; vanilla replay −4.27; OPR-RU −2.29; SDFT −1.00 but **SDFT loses 23 to 32 points on MeetingBank** and 3 to 5 overall accuracy points. A low-score replay ablation is *worse* than vanilla replay. OPR-SC (self-confidence filter) fails on confidently wrong rollouts (stated limitation). Takeaways: vanilla uniform replay of gold data is already a strong 1%-budget baseline; model-generated replay competes with it only when filtered.

### A14. Marek et al., self-generated replay (PART)

KL penalty on the model's own samples (prompted with a corpus identifier or a BOS token) nearly eliminates forgetting: toy 2M model on four sequential tasks (add, reversal, sort, modadd) shows forgetting "entirely eliminated"; 205M model on FineWeb-Edu to math; Llama-3.2-1B-Instruct on Verilog, where KL on self-generated data matches KL on substitute OLMo data. NTP on OLMo data *increased* forgetting vs standard finetuning. Limits stated: **single new task in most experiments, small models (up to 46M for most), one billion-scale result.** Capacity floor: models pretrained near saturation cannot absorb new data without overwriting. Replay also lets high learning rates be used without forgetting (38x compute difference in the toy setup).

This is the closest LLM-scale analogue to Robins' pseudo-rehearsal. It is also, in effect, the "uniform random replay" baseline that sleep papers need to beat.

### A15. DeltaHippo (PART)

"Sleep" = a gradient-free re-read of the just-finished task's own data that writes a low-rank activation memory (gated delta rule). That memory then constrains every weight update in later tasks via directional projection. Nothing is replayed to the model. Class-incremental TACRED, FewRel, Banking77, CLINC150 on six LLMs up to 8B (3 seeds); e.g. Llama-3.1-8B TACRED 76.8/78.1/77.1 vs SEQ* (frozen backbone) 67.1/64.3/69.4 and SEQ 41.3/33.6/38.0. Generative-replay baselines (LAMOL, PCLL) and L2KD score 11 to 56%, which looks weak relative to plain SEQ in places (L2KD 9 to 21% on TACRED, below SEQ), so baseline tuning is a question I could not resolve. The "sleep" here protects an index and does not transfer knowledge. Authors claim "first demonstration"; self-reported novelty.

### A16. Replay in the Silent Degrees of Freedom (PART)

Local-learner MLP on split-MNIST/CIFAR. Concurrent masked replay (no offline phase) reaches 91.6 ± 0.3% vs best offline "night" 88.5 ± 2.8 (narrow substrate); DER++ ties at 91.5 ± 0.5; at large buffers (K=5000) backprop references win. In a single pass the offline night falls to 76.9%. Relevant because it measures an offline-night schedule against real-sample interleaved replay, with several baselines tuned, and finds the night **high-variance across seeds** (89.0 ± 3.5).

---

## 4. Per-source detail: sleep in neural nets (pre-LLM and bio-ANN)

### B1. Tadros et al. 2022, Sleep Replay Consolidation (SRC) (PART-S)

- *What changes:* all weights in the fully connected layers, by a local Hebbian rule (strengthen when pre and post fire, weaken when post fires and pre is silent) in a spiking-converted copy (Heaviside activation, layerwise weight scaling). Conv layers were frozen.
- *Replay data:* none stored. Input is Poisson noise whose per-pixel rate is the mean input intensity over past training data. "No inputs representing specific memories" are presented in sleep.
- *Protection:* none explicit; the sleep rule rebalances weights toward old-task outputs. Applied after every task.
- *Cycles:* 4 sleeps for 5-task MNIST/FMNIST/CIFAR-10; 1 for CUB-200 (2 tasks).
- *Tasks:* class-incremental MNIST, Fashion-MNIST (5 tasks of 2 classes), CIFAR-10 (5 tasks, pretrained features), CUB-200, plus a cross-modal MNIST to FMNIST case.
- *Baselines:* sequential, parallel, EWC, SI, OWM, rehearsal (0.75% stored data), rehearsal+SRC, iCaRL (K=50 to 2000).
- *Numbers (overall test accuracy):* MNIST sequential 19.49, EWC 20.37, OWM 77.04, **SRC 48.47, rehearsal 79.91, rehearsal+SRC 86.47**, parallel 98.02. FMNIST: SRC 41.68 vs rehearsal 55.19. CIFAR-10: SRC 44.55 vs rehearsal 39.39 (SRC wins here). Cross-modal: SRC 61.33 vs rehearsal 83.13.
- *Flaws:* SRC alone loses to a tiny real-data rehearsal buffer on three of four benchmarks. No generative-replay baseline was run. Key thresholds were tuned with a genetic algorithm on training performance. Helps under-trained tasks more than well-trained ones (stated). The gap between SRC and rehearsal is the evidence that *noise-driven replay without data is much weaker than replay with a few real samples*, at least on MNIST-scale.
- *Random-noise control:* not reported as a separate condition. (In SRC, noise is the input; a control that swaps in unrelated noise would be the natural ablation. The Golden paper below does one.)

### B2. Bazhenov, Delanois, Krishnan 2026, *Not Just After One* (PART)

SRC applied **once** after a sequence of 2 to 5 tasks, 10 trials, MNIST, FMNIST, CIFAR-10. After 5 MNIST tasks, each earlier task is near 0% before sleep and about 20 to 75% after, mean about 0.5 (Fig. 1 values 0.53/0.39/0.67/0.75/0.33). Average accuracy falls gradually with number of tasks trained before sleep. Mechanism claim: sleep suppresses most synapses and keeps a few. **No baseline other than pre-sleep.** Four-page paper; no comparison to rehearsal, no random control. Useful as an existence proof that a single consolidating phase after several tasks can recover partially "forgotten" tasks (an access-versus-storage point, cf. `insights.md` I-3).

### B3. Golden et al. 2022, spiking networks (PART-S)

Two sequential foraging tasks in a small spiking network (842 neurons). Only hidden-to-output weights learn (reward-STDP); sleep = unsupervised STDP with Poisson noise matched to training firing rates; uniform noise (same rate for every hidden neuron) gave similar results. Sequential: Task 1 falls to 0.52 ± 0.02 after Task 2. Interleaved (uses old data): Task 1 0.68 ± 0.03. Interleaved *sleep with Task 2* (no old-task data): Task 1 0.70 ± 0.03. Freezing 1% of Task 1 weights failed; 10% preserved Task 1 but blocked Task 2. **Weakness:** two tasks, one small model, ten trials. The paper's own claim that sleep beats interleaved training "is argued on theoretical grounds; the measured difference is small." Methods mention "constant reward" in sleep, in tension with "unsupervised." Includes the one explicit random-noise control I found: a naive network given noise matched to Task 2 rates did not improve on Task 2.

### B4. González et al. 2020 (PART-S, identity unconfirmed; see note under table)

Biophysical thalamocortical model; sleep = slow oscillations with spontaneous replay; only cortical pyramidal-pyramidal AMPA synapses change via STDP. Sequence S1 then S2 then S1*, then sleep, 10 seeds. Reported: sleep reorganises connectivity (about 15% of recurrent connections remain) and recovers S1; interleaved training in supplement has smaller changes. Single biophysical model, no hippocampus, no homeostasis.

### B5. Sorrenti et al., WSCL (PART)

Wake: fast adaptation with layer-wise parameter freezing, episodic samples stored in short-term memory. NREM: trains on short-term plus long-term *real stored samples* (so it is not unsupervised). REM: trains on an external "dreaming" dataset with no class overlap. ResNet-18, CIFAR-10, Tiny-ImageNet half, FG-ImageNet, 5 tasks, 10 sleep epochs per task. Plugged on top of ER, DER++, ER-ACE: FAA CIFAR-10 (buffer 200) ER 48.76 to 51.86, DER++ 57.35 to 63.97, ER-ACE 59.98 to 71.15; forward transfer goes from negative to positive. Ablation: NREM and REM contribute about equally, REM drives forward transfer; REM works with noisy or downscaled dream images. **Caveats:** WSCL adds a multi-instance wake stage (one training run per candidate freezing depth) and 10 sleep epochs; whether baselines got matched compute is not stated in the pages I saw. External dreaming data is a free extra dataset, so "dreaming helps" may be "more data helps." Table 3 has error bars; the WSCL-vs-baseline margin on FG-ImageNet is 2 points.

### B6. Deperrois et al. 2022 (PART-S)

*Not a continual-learning paper.* Unsupervised representation learning with a generator and encoder: wake (reconstruction), NREM-like perturbed dreaming (encoder maps occluded generated images back to the replayed latent), REM-like adversarial dreaming. CIFAR-10 and SVHN linear-readout accuracy: full 58.25 and 78.92; wake-only 42.25 and 41.93; no REM 46.00 and 42.30. Four initial conditions. The authors note continual learning as future work.

### B7. Classical references (ABS / BG)

- **Robins 1995, pseudo-rehearsal.** Search snippets: the network is probed with random inputs and its outputs are used as pseudo-items, with no old data. (ABS)
- **Crick and Mitchison 1983 (reverse learning); Hopfield, Feinstein, Palmer 1983 (unlearning spurious attractors).** Attribution confirmed by secondary snippets only. Full texts not retrieved. (ABS / BG)
- **Hinton, Dayan, Frey, Neal 1995, wake-sleep.** Cited by A2/A5/A11 as the origin of the wake-sleep analogy for generative training. Not read. (BG)

---

## 5. Comparison matrix

"Random replay baseline" means a replay baseline that samples uniformly (from stored data or from the model itself) at comparable budget. "Retention probe" means a held-out general-capability measure tracked per cycle.

| Source | Weights changed in sleep | Replay / sleep data source | Old-knowledge protection | Cycles tested | Uniform / random replay baseline? | General-capability probe per cycle? | Seeds / CIs |
|---|---|---|---|---|---|---|---|
| A1 Behrouz | new low-rank experts; later base blocks; LoRA in dreaming | self-generated; no old data | freeze plus grow capacity | multi-level hierarchy; counts not given | **No** | **None found** | none stated |
| A2 Lee | fast SSM weights (state) | the context window itself | not studied | one per window; N = 1 to 6 passes | n/a | n/a | fixed seed; no CIs seen |
| A3 Letta | none (text) | model-written inferences | n/a | one | n/a | n/a | 3 to 10 runs for some models |
| A4 SleepGate | KV-cache gate params (trained) | synthetic PI streams | n/a | micro-cycles | n/a | no | 200 episodes per depth; one training run |
| A5 SCM | none (graph) | working-memory episodes | n/a | up to 20 (simulated) | n/a | no | "zero variance" |
| A6 Dennis | new LoRA each cycle | LLM-synthesised paraphrases | LoRA isolation; retrain from base | 3 | **No** (compaction only) | **No** (stated) | n=10 conversations |
| A7 SCoL | LoRA on selected layers | the context + pre-generated implications | sparse layer selection; RL forgetting term | 100-passage stream | Sequential FT only | no | 2 meta-rounds; SE reported on LongBench |
| A8 Cortex | new FFN params (frozen backbone) | self-generated queries + answers; past queries | freeze old; KL+MSE; query replay | 50 sequential edits | "regenerated query replay" ablation | MMLU/SimpleQA/math retention (RR) | seeds in some ablations |
| A9 Auto-Dreamer | none (text bank) | trajectories | provenance rewriting | k-episode cadence | n/a | n/a | single training |
| A11 Robots | policy | frozen critics/actors + 1000-state buffers | Nash-bargained gradients, anchor | 4 sleeps (5 skills) | replay baselines yes (CLEAR, DER) | pairwise reliability | seed 42 ablations |
| A12 MSSR | LoRA | stored real data | scheduling of replay | 3 and 11 tasks | **Yes (fixed uniform)** | MMLU per stage | one seed (42) |
| A13 OPR | full fine-tune | model rollouts on old prompts | replay buffer | 8 tasks | **Yes (vanilla gold)** | BWT only | 3 seeds |
| A14 Marek | full / LoRA | **self-sampled from base model** | KL on those samples | mostly 1 new task | the method is itself random self-replay | pretraining loss; MMLU+ARC+CSQA | single runs mostly |
| A15 DeltaHippo | all weights, constrained | none | directional projection from activation memory | 8 to 15 tasks | **No** (no replay allowed by design) | AA only | 3 seeds |
| B1 SRC | all FC weights | Poisson noise at mean input rate | none explicit | 4 (or 1) | rehearsal 0.75% **Yes** | n/a | not in table |
| B2 Not Just After One | all FC weights | same | none | 1 (after 2 to 5 tasks) | **No** | n/a | 10 trials |
| B3 Golden | H to O synapses | Poisson / uniform noise | STDP rebalancing | 1 to 2 (2 tasks) | interleaved + freezing | n/a | 10 trials |
| B5 WSCL | all weights | stored real samples (NREM) + external data (REM) | freezing, buffer | per task, 5 tasks | buffer baselines (ER etc.) as base | n/a | ± shown |
| B6 Deperrois | encoder + generator | latent replay + noise | n/a (not CL) | continuous | n/a | n/a | 4 inits |

---

## 6. Negative results and failures (all tagged to a source above)

1. **Sleep-like noise replay is weaker than a few real samples.** SRC alone: MNIST 48.5 vs 79.9 for 0.75% rehearsal; FMNIST 41.7 vs 55.2; cross-modal 61.3 vs 83.1 (B1). The offline night in A16 is high-variance and drops to 76.9% in a single pass.
2. **Offline phases are seed-fragile.** A16: offline night 89.0 ± 3.5 vs 91.5 ± 0.5 on a second split; "reported, not explained."
3. **SleepGate collapses at depth** (n ≥ 15) and its baselines fail even at no interference (A4).
4. **Retention in SCoL is still only about 20%** after 100 passages (A7).
5. **Dreaming-style self-edit loops can lose capability.** Behrouz himself cites iterative self-improvement as a forgetting risk (A1) and cites Kim et al. (C6) on OOD drops up to 40%.
6. **Auto-Dreamer over-abstracts and drops needed facts** (A9).
7. **Model-generated replay can be worse than naive replay if it is off the good-output region.** OPR: low-score on-policy replay is "uniformly worse" than vanilla replay (A13). OPR-SC fails on confidently wrong rollouts.
8. **Cortex failures:** Synaptic CR −92.3% on τ²-Bench (A8); batch API consolidation falls from 91 to 93% to 45 to 59%; suppression damages neighbours by 9 to 20 points.
9. **Capacity floor:** models trained near saturation cannot learn new data without overwriting, even with replay (A14).
10. **Context-distillation teachers are unreliable in several documented ways.** See section 7B.
11. **Single-cycle sleep proofs are not cycle-count proofs.** B2 shows recovery after one sleep; it does not show stability over many.
12. **Evaluation circularity in LLM memory papers.** A6 (LLM-generated data, LLM-extracted tests, LLM judge), A5 (own benchmark, zero variance).

---

## 7. Added question: has anyone trained a model's closed-book distribution to match its own context-conditioned (or RAG) distribution?

**Short answer: yes, repeatedly. It is a well-trodden area (2021 to 2026).** Searches run: alphaXiv `discover_papers` on "context distillation / on-policy context distillation / self-distillation / context-parametric inversion / knowledge conflict / DisentQA"; follow-ups on each hit; WebSearch for "Language Models Need Sleep openreview". I did not run separate searches for "retrieval-augmented distillation into weights" using other phrasings (e.g. "RAG-to-parameter", "parametric RAG", "DyPRAG", "knowledge internalization"), so papers under those labels may exist beyond those below; A8 mentions PRAG, DyPRAG, Cartridges, SELF-PARAM as neighbours, which I did not read.

### 7A. Papers that did it, and how well it worked

| Paper | Teacher / student | What was distilled | Result |
|---|---|---|---|
| Askell et al. 2021 (HHH prompt distillation) | same model with prompt vs without | alignment prompt | cited by C1/C3/C4; not read (ABS via citations) |
| **C1 Snell 2022** | same model; teacher sees instructions + examples, student minimal | instructions, SPIDER examples, 8-digit scratch-pad | Natural-Instructions T5-11B: student Rouge-L 9.0 to 34.7 vs teacher 43.4; SPIDER text-to-SQL +9% over directly training on the examples; fact editing: paraphrase score 79 (teacher 73) but **neighbourhood 48 (teacher 58, pre-distill student 80)** |
| **C2 Kujanpää 2024 (prompt distillation)** | same model with doc in prompt vs LoRA student | free-form documents (Squadshifts, HotpotQA) | Llama-3-8B Instruct closed-book PD: Amazon 86.1, NYT 93.6, Reddit 79.5 vs SFT 75.9/87.5/69.8 and RAG 86.3 (Amazon), 96.3 (NYT), 78.6 (Reddit). PD+RAG beats RAG. 3B model: weaker data generation; larger "expert" helps. HotpotQA closed-book Llama-3-8B: PD 73.7 vs SFT 68.9 vs base 53.0. With Tülu-3 regularisation MMLU-Pro stays within about 0.4 to 1 point of the instruct model |
| **C3 DiSC 2026** | same model; teacher sees document prefix, student sees none; KL on the suffix | raw documents | Qwen2.5-7B-Instruct KUP: 15.5 to 47.6 (best) with IFEval 80.7 to 74.5 and MATH 49.9 to 44.0; fine-tuning reached 40.9 on KUP but IFEval 65.8 and MATH 24.5 |
| **C4 OPCD 2026** | frozen teacher with context vs student, on-policy reverse KL | experiences, system prompts | Qwen3-8B math 75.0 to 79.7 (in-context 77.6); Frozen Lake 1.7B 6.3 to 26.5; IF-Eval roughly unchanged (81.3 to 81.7). Distillation of optimised safety/medical system prompts beat the in-context (prompted) result in all 6 model-task cells reported (e.g. Llama-3.2-3B medical 59.4 base, 66.4 in-context, 76.3 OPCD); off-policy context distillation tied OPCD on Llama-3.2-3B safety (83.3 vs 83.1) |
| **C5 SDFT 2026** | same model; teacher sees demonstration | demonstrations (skills; 2025 disasters corpus) | New knowledge strict acc 89% vs SFT 80%; sequential 3-skill learning without regression; 7B beats SFT by 4.0, 14B by 6.9 |
| **A8 Cortex 2026** | frozen pre-update model with context | contexts incl. API docs, persona, edits | 50 edits: 88.9% CR, 99.8% RR (14B). See failures |
| **A1 Behrouz 2026** | pre-update self with accumulated experts | in-context knowledge into slower blocks | see section 2 |
| **A7 SCoL, A6 Dennis** | different pipelines | context to weights | see section 2 |

**So the specific operation, "train the closed-book output distribution to match the model's own context-conditioned output distribution" (KL, reverse KL, or sampled), has been done by C1, C2, C3, C4, C5, A8, and in partial form A1.** C2 and C3 are the closest to "knowledge from documents/RAG context into weights." C2 explicitly benchmarks against RAG and finds it competitive on closed-book Q&A (not uniformly: NYT 93.6 vs RAG 96.3).

**What I did not find:** a paper that distils the *retrieved-passages-in-noisy-conditions* distribution (i.e., teacher conditioned on top-k retrieval including irrelevant or conflicting passages) and evaluates the student on conflict cases. C2 uses RAG as a baseline and as a complement, not as the teacher context. The nearest items are DisentQA-style counterfactual studies (below), which are about context use, not internalisation. This is a gap candidate but my search was not exhaustive.

### 7B. Evidence the approach fails or is fragile

1. **Teacher is only as good as its in-context learning, and small models are bad at it.**
   - SDFT: at 3B the method is **−3.3 points vs SFT**, +4.0 at 7B, +6.9 at 14B. Stated: "smaller models with weak ICL abilities fail to provide meaningful teacher signals" (C5).
   - OPCD: injecting experiences from a Qwen3-8B teacher into the *context* of 1.7B and 4B students **degrades** their performance below base ("In-Context" curve); distilling instead helps (C4).
   - C2: the 3B model "struggles to generate high-quality training data."
2. **Teacher can be hurt by its own context, and the student inherits it.**
   - C1: the T5-11B teacher is poor on neighbourhood prompts for fact editing; the student's neighbourhood score is worse still (48 vs 58). This is an old precedent for exactly the failure the coordinator asked about.
   - C4: raw previous-problem traces as context *lower* validation accuracy (70.5 vs 75.1 without experience), while extracted knowledge raises it.
3. **Teachers change style, not just knowledge.**
   - C5: students reproduce "Based on the text..." artifacts; mitigated by masking the first tokens ("a heuristic fix"). Fundamental behaviour shifts (non-reasoning to chain-of-thought) were difficult.
   - C6: conditioning the teacher on rich information suppresses epistemic markers ("wait", "hmm"); drops up to 40% on AIME-24 (DeepSeek-R1-Distill-Qwen-7B), 45.9% on Qwen3-1.7B, in math with broad task coverage; benefit in narrow-coverage chemistry. SFT on solution-guided data: AIME24 54.79 to 20.21.
   - C7: the gain of privileged-solution self-distillation is reproduced when the teacher sees an *unrelated* worked problem (OP2SD), and, in the 1.7B setting, also when the teacher sees only the target problem but runs in thinking mode (the student is non-thinking), so part of the gain is mode asymmetry. So part of "knowledge transfer" is "context-induced teacher behaviour." In the matched 4B non-thinking setting, target-only teacher collapses AIME24 to 1.18% (base 23.19%). Physics examples degrade and cause non-terminating generations. Longer training (400 vs 100 updates) lowers OP2SD AIME24 Avg@12 from 55.35 to 51.39. Evaluation in thinking mode did not transfer.
   - C8: the content of privileged context moves KL drift 7.2x per token vs source 1.4x; on Big-Math, 8 of 9 target-accuracy intervals include zero and external demonstrations drop accuracy by 7.7 points; prior-task accuracy declines in most cells (up to 3 points; IFEval −8.3, HumanEval −6.1). One seed per condition.
4. **Large behaviour shifts break the anchor.** A13: SDFT loses 23 to 32 points on MeetingBank and 3 to 5 points overall accuracy vs vanilla replay on TRACE; its loss plateaus on tasks that need large distribution shifts.
5. **Self as teacher is unstable.** C4: self-OPCD (teacher = evolving student) gives Sokoban 18.8 vs 53.9 teacher-student, and can diverge. C5 notes the student-as-teacher diverges and an EMA teacher is the compromise.
6. **Context reliance itself is not stable across instruction tuning.** C9: in Llama2-7B, Pythia-6.9B, Mistral-7B, reliance on a counterfactual context first rises then falls during instruction tuning (Llama2-7B: about 30% to 60% to about 35% on one measure) while standard benchmarks keep rising. Cause: non-context-critical examples. Implication: an instruction-tuned model's context-conditioned distribution may *itself* underweight the context when it conflicts with parametric memory, so the "teacher" is biased toward parametric memory exactly where distillation is supposed to add something. This has not, to my knowledge, been tested as a limit on context distillation specifically; it is my inference.
7. **Parametric/contextual disentanglement is imperfect.** C10 (T5-11B): vanilla counterfactual accuracy 66.81%; with counterfactual+answerability augmentation 84.98%; parametric answers leak the fine-tuning set ("strong tendency to repeat answers from fine-tuning"); parametric accuracy on counterfactual contexts 44.69% vs 27.69% closed-book baseline for the best model. Only named-entity answers handled.
8. **Batch consolidation is weaker than single.** A8: 91 to 93% CR single vs 45 to 59% batch on GPT-OSS-20B API learning.

### 7C. Reading for the sleep project

- "Context to weights by matching distributions" is a solved *mechanism* with reproducible gains on 7B+ models and clear failure modes at ≤3B or under distribution shift. It is not novel by itself.
- What is *not* settled: whether doing it **repeatedly across cycles**, with a **changing context source** (accumulated usage, noisy retrieval) and **teacher-quality gating**, accumulates or erodes. A8 (50 edits) is the longest sequence found. GATES (arXiv 2602.20574; unread) may address gating.

---

## 8. What the field has NOT tested (as far as I could find)

I could not prove absence. These are "not found in the sources above and in the searches listed."

1. **10 or more wake/sleep cycles on a pretrained LLM** with weights actually updated each cycle, new knowledge accumulating from live-like inputs, and capability checked after every cycle. Longest: 50 edits (A8, with replay of queries), 100-passage stream with 20% retention (A7), 3 retrains (A6), 8 tasks (A13, A15).
2. **A sleep method vs uniform self-sampled replay at matched compute** on an LLM. Marek (A14) shows self-sampled KL replay is strong but on one task. Behrouz (A1) never compares to any replay.
3. **A replay-selection-policy ablation inside a sleep phase** (random vs error-based vs recency vs interference-aware) on LLMs. MSSR (A12) varies *scheduling* and sample weighting with a single seed. `insights.md` I-7/I-8 predict random is hard to beat; no LLM sleep paper tests it.
4. **Whether synthetic/dream data helps over just more replay.** WSCL's REM (B5) uses an external dataset, so it confounds with data quantity.
5. **Savings/relearning speed after forgetting** in a sleep framework. Not measured in any LLM source I read.
6. **Interaction of sleep with weight decay/normalisation drift** (`insights.md` I-2, I-6). Not examined.
7. **Seeds and confidence intervals** are mostly missing in the LLM "sleep" papers (A1, A12 single seed, A11 ablation single seed, C8 single seed per condition).
8. **Safety/alignment drift across cycles.** A6 says "has not been measured." C8 measures IFEval/HumanEval but with one cycle.
9. **Teacher-quality gating of context distillation under conflict**, and distilling noisy-RAG contexts (section 7A).

---

## 9. Gap statement (precise)

> No published work I found runs a **controlled, repeated (≥10 cycles) offline consolidation protocol on a pretrained LLM** in which (i) each cycle updates weights from newly accumulated inputs, (ii) old-knowledge protection is compared against **matched-compute uniform replay of stored data and uniform self-sampled replay** as baselines, (iii) both **new-knowledge retention across all earlier cycles** and **held-out general capability** are tracked after every cycle, with seeds, and (iv) **which items get replayed** (random vs error, recency, interference-aware) is an experimental variable.

The pieces exist separately: self-sampled KL replay at small scale (A14), context-to-weights distillation (C1 to C5, A8), scheduled replay (A12), concurrent vs offline consolidation (A16), and cycle-count dependence in tiny ANNs (B1, B2). The combination, at LLM scale, with a clean ablation ladder, is not in what I read.

**Is the space crowded?** Yes, in three ways.
- *"Sleep"/"dreaming" as a name for consolidation:* at least ten 2026 preprints use it (A1, A2, A4, A5, A9, A10, A11, plus B2, B5-like earlier ones). The name is cheap. A paper that merely adds a nightly fine-tune will look like A6 or A1.
- *Context-to-weights distillation:* crowded (section 7).
- *Replay-for-LLM:* crowded (A12, A13, A14, A15 and others in search results such as Kotha-Liang).

Less crowded: the **rigorous ablation ladder**, **many-cycle dynamics**, **selection-policy comparison**, and **savings measurements**.

---

## 10. Design implications (with confidence)

1. **Always include two baselines: uniform replay of stored data and uniform self-sampled replay with a KL penalty (Marek-style).** Matched compute. *Confidence: High.* Supported by A14, A12, A13, `insights.md` I-8, and by SRC's loss to rehearsal (B1).
2. **Test many cycles, and report per-cycle curves, not just final numbers.** Single-cycle recoveries (B2) and 3-cycle results (A6) do not predict cycle-10 behaviour. Include a general-capability probe (MMLU/IFEval/HumanEval-style) after every cycle. *Confidence: High* that this is the gap; *Medium* on the best probe set.
3. **If using context distillation as the consolidation step, use a teacher-student (frozen) setup, not a continually updated self-teacher, and use on-policy reverse KL.** *Confidence: Medium-High.* C4 (Sokoban 18.8 vs 53.9), C5 (EMA), C3.
4. **Gate by teacher reliability.** The teacher is unreliable at ≤3B, under large distribution shift, and when its own context conflicts with parametric memory (C5, C4, A13, C9). Cheap gate: compare teacher accuracy with and without context on a probe set before distilling, and skip or down-weight items where the context does not help. *Confidence: Medium* (the gate is my proposal; the failure modes are documented). GATES (unread) may be a precedent.
5. **Track the epistemic-style side effect.** Check response length and hedging-token rates across cycles; C6/C7 show distillation shifts style in ways that hurt OOD. *Confidence: Medium.*
6. **Keep protection mechanisms separable from the sleep claim.** Freezing plus capacity growth (A1, A8) is already a continual-learning method with or without a "sleep" schedule. Ablate "growth only," "replay only," and "growth + replay," otherwise the gain cannot be attributed to sleep. *Confidence: High.*
7. **Measure both storage and access.** B2 recovers apparently-forgotten tasks by sleep; A15 notes decodable-but-unreachable knowledge. A linear probe on frozen features at each cycle separates the two. *Confidence: Medium-High* (consistent with `insights.md` I-3, I-71).
8. **Use real-data replay as the anchor even in a "self-generated" design, then reduce it.** SRC alone loses to 0.75% real rehearsal (B1), and rehearsal+SRC is best. A simple dose-response (replay fraction 0, 1, 5, 10%) shows how much the sleep mechanism adds on top. *Confidence: Medium-High.*
9. **Run ≥3 seeds and report intervals; avoid LLM-written data judged by the same LLM family.** The weakest LLM-sleep results (A5, A6, A12 single seed) fail on this. *Confidence: High.*
10. **Do not claim novelty for "context to weights" or for the word "sleep."** Claim the controlled many-cycle comparison and the selection-policy results, if they come out. *Confidence: High* on the novelty assessment (crowded), *Low-Medium* on whether the many-cycle result will be interesting; it could well show that uniform self-replay plus a nightly KL step is enough.

---

## 11. Limits of this report

- Most reads were partial (PART), and several came through a summariser (PART-S). Numbers marked PART-S need a re-check against the papers.
- OpenReview was not accessed. Behrouz v2 on arXiv may differ from the September 2025 OpenReview version.
- B4 (González 2020) identity is likely but unconfirmed. Hopfield/Crick-Mitchison/Hinton/Robins are not read.
- "Not found" statements reflect a bounded search (alphaXiv discovery with two to three queries per topic, a handful of web searches), not an exhaustive survey.
- No models or code were run.

---

## 12. Follow-up (2026-11-04): GATES, Kotha-Liang, divergence as a control signal, entrenchment, anchoring

Access tags as above. All items here are PART unless stated; WebSearch-only items are ABS (snippet).

### 12.1 GATES (arXiv 2602.20574) and Kotha and Liang (arXiv 2603.04964)

**GATES (PART).** What is gated: *which training questions are distilled*. For each question the model, given the source document ("tutor"), samples k=8 reasoning rollouts. If at least τ of them agree on the final answer, the question's agreeing document-conditioned trajectories are distilled into the document-free "student" (same weights). Otherwise the question contributes zero loss. So the gate is **agreement among context-conditioned rollouts**, not a divergence between context-conditioned and closed-book outputs.
- Setup: Qwen3-4B-Base, math word problems generated from documents, 551 train questions, **50 held-out** questions, one epoch. The paper states ≥4/8 agreement in the training details and 5/8 in the dataset filter; I did not resolve which applies to the gate in the main run.
- Numbers: held-out in-domain student accuracy 46.0% to 62.0% (31/50 vs 23/50, my arithmetic). Average maj@8 on MATH/AMC/Minerva/OlympiadBench 20.2% to 35.4%. Removing the gate (same loss weights): 54.0% in-domain and 31.1% benchmark average (so the gate is worth about 8 points in-domain and 4.3 points on benchmarks). Tutor-trajectory SFT without gating: 54% and 32.3%. Answer-only SFT collapses to 10 to 12% in-domain.
- Caveats: under **greedy decoding GATES and ungated tutor-trajectory SFT are tied** (40.0 vs 40.3 average); the gate's benchmark advantage is in maj@8 only. The in-domain reference answers come from a Qwen2.5-32B-Instruct 8-rollout consensus. Math only. Authors name the failure mode: confident but wrong consensus; with an adaptive question generator, adding an oracle-correctness loss helped. Without the gate, the student inherits "document-dependent" reasoning (tutor 74% vs student 54%).

**Kotha and Liang (PART).** What is replayed: **fresh samples from the generic (pretraining-like) distribution, mixed into fine-tuning on a target domain** (a fraction ρ of Stage-2 steps). Not self-generated, not old-task replay in the continual-learning sense. They report that replay *improves the target-domain loss*, not only retention.
- Controlled setup: 150M-parameter models, 4M target tokens (FineMath, StarCoder, Flan), up to 4B generic C4 tokens. Data-efficiency gains from tuned replay: 1.87x (Flan), 1.49x (FineMath), 1.09x (StarCoder) for fine-tuning; up to 2.06x with a mid-training schedule. Benefit persists to 1.9B-scale models in their size sweep (qualitatively; exact numbers not read). Replay helps more when the target domain is rare in pretraining; with 75% of the target data already in pretraining it stops helping or hurts.
- Real models: Llama 3.1 8B Instruct on Weblinx web navigation, baseline 32.86%, up to +4.5 points with OpenHermes/UltraChat replay; Llama 3.1 8B on 200M Basque tokens, up to +2 points on Basque COPA with SlimPajama replay.
- Caveats: replay costs 1/(1-ρ) more steps; the ρ was tuned per setting; real-model gains are single-benchmark and small; weight decay did not substitute for replay. Relevance: it supports a regularisation reading (replay curbs overfitting to few target samples), which is separate from the sleep reading.

### 12.2 Is the divergence between in-context and closed-book outputs used as a control signal?

**Yes in pieces; I did not find the full set (select / audit / schedule / retire).**

| Use | Found? | Source (tag) | Result |
|---|---|---|---|
| Select which items to distill | **Yes** | ReSAIL, arXiv 2609.39306 (PART) | "PI sensitivity" = Jensen-Shannon divergence between the teacher's output distribution with and without the privileged context, per step. Keep the top 5% most sensitive steps. ALFWorld, Qwen3-4B, cycle 1, success without PI, ID/OOD: Top 62.2/57.3, Random 49.7/45.8, **Bottom 40.1/40.4**; ratio sweep: top 5% best (62.2/57.3), 100% gives 56.8/50.3. Cuts student-optimisation cost about 74% (total loop about 6%). Also +1.06 points on AITZ as an offline filter. One training seed; error bars are decoding seeds only |
| Arbitrate source at inference | Yes | TrustMargin, arXiv 2606.08397 (PART) | Compares closed-book likelihood of the RAG answer vs the direct answer (parametric-prior margin), plus context-only likelihood subtraction (evidence-binding margin). LLaMA-3.1-8B average F1: Direct 37.25, BM25-RAG 36.28, TrustMargin 42.61. Recovers 24.9 to 41.3% of the Direct/RAG oracle F1 gap; selects Direct when Direct is better 60 to 66% of the time. Inference-time selection; no weights updated; bounded by the two candidates. Needs no gold labels at decision time |
| Detect whether context or memory governed output | Partial | CRM, arXiv 2605.26778 (PART) | Per-layer hidden-state displacement (with context minus without) detects whether the context document was in the pretraining data: AUC 0.71 to 0.95 (WikiMIA, nine models), 0.84 to 0.97 (BookMIA), **0.48 to 0.55 on MIMIR (fails)**. The authors say it does not certify which source any single generation used |
| Context-sensitive token identification | Background | PECoRe / P-CXMI (search snippets, ABS) | Per-token likelihood ratio, P-CXMI, KL between with- and without-context distributions are standard measures of context influence (from machine translation interpretability). I did not read them |
| Contrastive decoding (CAD, COIECD, AutoCAD) | Cited only | via C9 (Goyal), PART | Amplify the with-vs-without-context difference at inference; C9 says they give "limited gains, especially in instruction finetuned models" |
| Decide *when* to consolidate | **Not found** | - | - |
| Audit consolidation without ground truth (closed-book now matches context-conditioned) | **Not found as a stated method.** Cortex and DiSC *train* toward that match (A8, C3) but measure with labelled evals (CR/RR, QA accuracy) | - | - |
| Decide when a memory may leave retrieval | **Not found** | - | - |

Searches run for the "not found" rows: alphaXiv discovery on "knowledge conflict detection / context influence / parametric knowledge / retrieval-augmented / context-aware decoding"; WebSearch on retiring or evicting retrieved memory after internalisation and on closed-book-vs-RAG agreement as an audit (returned background on parametric vs external memory, CAMeR, MemoryLLM, MEGa; nothing on retirement or audit); WebSearch on P-CXMI/CCI/context-mixing. Listing-only leads, unread: 2606.10298 (conflict-aware contrastive decoding), 2605.14473 (does RAG know when retrieval is wrong), 2610.38296, 2609.00753, 2610.20067. The "memory retirement for RAG" phrasing was searched once (WebSearch, above) and not via alphaXiv; treat that null as weaker than the others.

**Takeaways.** (i) Divergence-based *selection* of consolidation targets has one positive result (ReSAIL, 3 cycles, text-agent tasks, one training seed). (ii) TrustMargin shows the closed-book likelihood of a candidate is a usable trust signal at inference. (iii) Both are untested for the retirement/audit role. (iv) CRM's MIMIR failure and the paper's own caveat argue against reading the with/without-context divergence as a per-item source certificate.

### 12.3 Entrenchment: does fine-tuning a fact into weights make the model follow contradicting context less?

**I found no paper that measures this on the same fact** (fine-tune or edit fact F, then compare context-following under a contradicting context for F against an unlearned control fact). Null result.

What I found that is adjacent:
- **Context-parametric inversion (C9, PART).** Instruction tuning on data that mixes in non-context-critical examples *lowers* reliance on a contradicting context over training, on facts *not* in the tuning data (they removed overlap and still saw the drop): Llama2-7B counterfactual reliance rose from about 30% to about 60% and fell back to about 35%. A web-search snippet adds that the drop persists with a context-only Alpaca subset of about 30% (ABS). This is a general shift toward parametric answering, not an entrenchment measurement on edited facts.
- **Parametric-vs-context arbitration by confidence (ABS, search snippet only, arXiv 2510.02370 and 2410.08414 listed; unread):** reported as preferring parametric knowledge for high-confidence facts. If true, entrenchment would be predicted by confidence, but I did not verify.
- **Context reliance in unstructured editing (arXiv 2602.19043, PART).** The opposite problem: edited knowledge becomes *dependent on its training context* (recall fell by up to 56% ROUGE on later-positioned facts when the context is absent; restoring the context restores recall; worse in larger models, 14B vs 0.5B; more training steps did not fix it). This is about retrieval of edited facts without context, not override of contradicting context.
- **Engram Injection (arXiv 2610.15882, PART).** Dated-update setup: after storing an old value and its replacement in a LoRA, the model prefers the old value before the update at 0.99 and the new one afterward at 0.79 to 0.95 with a margin loss; **0/62 on the held-out-age generation test** (retrieval 62/62). That shows weights-stored facts are poor at version arbitration, but it never presents contradicting *context* to a fact trained into weights.
- **Spurious forgetting / suppressed-not-erased (arXiv 2610.08718, 2609.18985; listing only, unread)** and **Rewired or Gated (2609.25602; listing only)** may bear on it.

Inference, labelled as such: there is a plausible route to entrenchment (training on non-context-critical examples reduces context reliance, C9; a fact distilled from context so that closed-book matches context-conditioned output is, by construction, a non-context-critical example afterwards). Nobody I found tested it. A direct test is cheap: after a consolidation cycle, present a counterfactual context for consolidated facts and for matched never-consolidated facts and compare the follow rate.

### 12.4 Anchor to original pretrained weights vs previous checkpoint across repeated cycles; accumulated drift

**No head-to-head comparison of the two anchors across many cycles found. Null result on that exact contrast.** Evidence on the pieces:

- **Which reference each system uses (read):**
  - Frozen *initial* model: DiSC (teacher = frozen copy of the post-trained model, single adaptation, C3), OPCD default (frozen teacher; self-teacher that tracks the student diverged: Sokoban 18.8 vs 53.9, C4), Marek KL on samples from the frozen reference model (A14), Cortex for generic-query targets in continual editing ("generated once by the initial model M0 and cached", A8).
  - Frozen *previous checkpoint*: Cortex query replay regenerates targets with the pre-update checkpoint (A8); CAGD (arXiv 2610.06940) freezes the previous-stage model each stage; ReSAIL freezes the cycle-start model each cycle and carries the final student into the next cycle (2609.39306).
  - SDFT appendix (C5): frozen base teacher is stable but underperforms; student-as-teacher diverges; EMA is the compromise. This is a within-run anchor comparison, not across cycles.
- **Drift / collapse curves over cycles (read):**
  - ReSAIL (ALFWorld OOD, Qwen3-4B, OEL baseline, three cycles): success 47.1, 46.1, **38.5** across cycles 1 to 3; ID 54.9, 56.5, 47.4. With selection, TLB and retention: ID 62.2, 70.1, 74.2; OOD 57.3, 63.5, 67.2. PI-conditioned competence (what the next teacher can do) also declines under the baselines. Retention toward the **cycle-start frozen teacher in the privileged view** is the component that keeps cycle 3 from falling (74.2 vs 56.5 with none; ordinary-view retention 72.9). Decoding-seed error bars; one training run.
  - Denser is not Better (arXiv 2607.01763, PART): SDPO on four sequential domains: MATH 56.42% after its stage falls to **34.38%** by the end, TOOLUSE 9.93%, collapse in a variant; GRPO retains. Parameter-drift metrics (spectral shift, principal-angle rotation) are larger for SDPO. Linear merging with the base weights at 50% recovered GPQA 58.72 to 62.25 and ZLogic 74.50 to 76.42. This is indirect evidence that an anchor to the original weights (merging) repairs accumulated drift; it is a post-hoc fix, not an in-loop anchor comparison.
  - CAGD (2610.06940, PART): previous-checkpoint distillation; four-task final held-out loss 2.927 to 1.114 (SMDM-219M, one order); but **GSM8K exact-match retention is seed-mixed at 0.6B and worse at 1.7B** (-22.2 vs -15.8 points sequential), format compliance 89% vs under 1%. The authors note CAGD "cannot repair knowledge that the previous model has already lost": anchoring to the previous checkpoint inherits its errors by design.
  - Cortex (A8): 50 sequential edits keep 99.8% retention measured against the *original* model on a 14B model, with replay of past queries and cached initial-model targets for generic queries. Closest thing to a many-step drift number in the literature I read. No curve over steps in the text I saw, and no ablation swapping the anchor.
- **Not found:** a study that runs more than about 10 cycles of context distillation and plots divergence from the *original* weights (or capability on a fixed held-out suite) for the two anchoring choices; any treatment of why a previous-checkpoint anchor should accumulate drift (it is a random walk on a chain of local constraints; this is my reasoning, not a cited result).

### 12.5 Added design implications (confidence)

11. **Use teacher-with-context vs teacher-without-context divergence (JSD or KL) to pick which items to consolidate, and compare against random and bottom-ranked selection at the same budget.** ReSAIL's Top/Random/Bottom ladder is the control to copy. *Medium* (one positive result, one training seed, agent tasks).
12. **Add an agreement gate on the context-conditioned teacher (GATES-style) but report greedy and maj@k separately**; the gate's benefit may live in consistency rather than peak accuracy. *Medium-Low.*
13. **Measure entrenchment explicitly each cycle** (contradicting-context follow rate on consolidated vs never-consolidated facts). *High* that it is untested; *Low* on the size of the effect.
14. **Anchor generic-capability targets to the original model's cached outputs, and use the previous checkpoint only for newly consolidated content.** This is what Cortex does in practice and it is the only multi-step (50) result with near-complete retention; no direct ablation exists, so *Low-Medium*. Include a base-weight merge or a distance-to-original curve as a diagnostic (2607.01763 shows merging repairs drift).
15. **Do not use with/without-context divergence as a per-item "this memory can be retired" certificate without a labelled audit.** CRM fails on a confounded benchmark and disclaims per-generation attribution; no retirement method was found. *Medium.*
