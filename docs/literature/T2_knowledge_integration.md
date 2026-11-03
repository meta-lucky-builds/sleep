# T2: Knowledge integration when fine-tuning an LLM on new facts

Thread T2, written 2026-11-04. No local tests or model runs were done. Everything below comes from reading papers or web pages.

## 0. How to read this file

Access tags (honest about depth; I did NOT read any paper end to end):
- **PART** = I retrieved selected pages of the paper PDF text (alphaXiv `answer_pdf_queries`). Numbers were read from those pages.
- **REPORT** = alphaXiv AI-generated report of the paper (`get_paper_content`). Treat numbers as one notch less reliable than PART.
- **ABS** = abstract or arXiv/ACL listing page only.
- **PRESS** = web-search summary or blog or third-party page. Not checked against the paper.
- **NONE** = I looked for it and could not verify. Said explicitly where it matters.

"My inference" marks anything that is my reasoning rather than a cited result. Quotes are under 15 words. No FULL tags are claimed.

---

## 1. Executive findings (the 10 that matter for the "sleep" design)

1. **Capacity is not the bottleneck; extraction and integration are.** A GPT-2-style model stores about 2 bits/parameter after about 1000 exposures per fact (Allen-Zhu & Li 3.3, arXiv 2404.05405, PART). A 0.5B model has about 10^9 bits. A few thousand synthetic facts need about 10^5 bits (my inference). What fails is whether the fact can be retrieved by a differently worded or reversed query.
2. **Rewording the training text is the single best-supported fix for forward extraction, but it works at pretraining or continued-pretraining time, with a lot of data.** In Part 3.1 (2309.14316, PART), GPT-2 on 100k bios scored 9.7% on held-out QA with unaugmented bios and 96.6% with 5 rewrites plus sentence permutation. Part 3.1 also reports that LoRA QA fine-tuning applied after a memorized-but-unaugmented pretrain only slightly improves test accuracy. So augmenting at "sleep" time must put the paraphrases in the consolidation corpus itself.
3. **Reversal is not fixed by paraphrase.** Berglund et al. (2309.12288, PART): GPT-3 175B reaches 96.7% / 50.0% in the trained direction and 0.0-0.1% reversed. Same near-0% on GPT-3 350M, Llama-7B, and with 30 paraphrases per fact, with both orders present for other facts, and with a 40k-document ablation. Allen-Zhu Part 3.2 (2309.14402, PART) finds inverse search about 0-1% on GPT2/Llama/Mistral unless the reversed text is in training. The practical fix is to write reversed statements into the consolidation data.
4. **Composition of two newly learned facts fails without CoT, and works if the facts are co-located.** Balesni et al. (2411.16353, PART): Llama 3 8B, Qwen2.5 7B, GPT-4o(-mini) at chance on two-hop with two synthetic facts in separate documents; nonzero when both facts sit in the same training document, or when one fact is pretrained; about 20% average no-CoT accuracy in a semi-synthetic setup. With 10 unrelated distractor triplets between the facts, same-document accuracy dropped to chance. Design implication: the sleep data builder must link related facts, and CoT-style training targets are the only reliably working form of multi-hop.
5. **Parametric editing (ROME/MEMIT/MEND) is not a safe base for a sleep procedure.** On RippleEdits (2307.12976, PART), ROME/MEMIT/MEND average 38-66% over ripple criteria, and an in-context edit baseline beats them. Sequentially, ROME and MEMIT show gradual then abrupt collapse (Gupta et al., 2401.07453, PART): MEMIT onset about 1400 edits (GPT-J), ROME onset anywhere from about 100 to 1000 edits. Mirage (2502.11177, PRESS) says teacher-forcing evaluation inflates results and about 10% success at 1000 sequential edits under autoregressive evaluation.
6. **Fine-tuning on facts the model does not know is slow and harmful if run to convergence, and the harm shows up as hallucination about things the model did know.** Gekhman et al. (2405.05904, PART; full FT, lr 1e-5, 50 epochs, EntityQuestions): unknown examples are fit much slower; at convergence a fine-tuning set with more unknown examples lowers closed-book test accuracy (up to 14 points in-distribution, up to 6 OOD). A linear model fit (R^2 0.86) has beta_unk about -8.3 and beta_kn about +7.3, roughly equal and opposite. Early stopping or dropping the unknown examples removes most of it.
7. **Mix-ins fix a lot, and they are exactly what a sleep phase can supply.** Evidence for replay/self-generated/known-fact mixing: Gekhman (MaybeKnown examples best, 43.6 vs 40.5 for HighlyKnown-only, PART); Dang et al. (2511.02626, ABS) say adding a small amount of known knowledge late in training restores entity attention and reduces hallucination; Kaplan et al. (2604.15574, ABS) propose self-distillation SFT that learns new facts while limiting output-distribution drift; Slocum et al. (2510.17941, PART) mix 1:1 with C4 so the SDF model does not over-mention the implanted fact; Yang et al. EntiGraph (2409.07431, REPORT) replays 10% RedPajama; Zucchet et al. (2503.21676, REPORT) find replay only partly mitigates the early loss of old facts.
8. **Synthetic document fine-tuning (SDF) does produce beliefs that behave like knowledge at 70B, but not for facts that clash with basic world knowledge, and nobody has shown it on 0.5-1.5B models.** Slocum et al. (PART): Llama 3.3 70B Instruct, 40k synthetic + 40k C4 docs, LoRA; implanted beliefs survived adversarial prompting and a 4-turn debate except for "egregious" facts. Prompting-only and AlphaEdit-style editing were shallower. They say belief depth is constant or increases with scale (footnote, PART), so small-model behavior is likely worse (my inference; NONE verified).
9. **Teaching "I don't know" is the one calibration fix with a number attached, and it is a partial fix.** Gekhman (PART): relabeling unknown examples as "I don't know" raises accuracy on answered questions to 61.8% (vs 43.0% baseline at early stop; 38.8% at convergence) while answering 58.7% / 55.6% of questions. I found no paper that measures over-confidence on fictional neighbors of newly learned entities in a small LoRA-tuned model (NONE). A LoRA study (Marina et al., NAACL Findings 2025, ABS) reports more confidence and fewer refusals after injecting facts.
10. **No off-the-shelf benchmark is a stream with conflicts over time.** RippleEdits (MIT) and MQuAKE (MIT repo) are edit-by-edit with ripple or multi-hop probes; MQuAKE-T has temporal updates (1,825 items); TemporalWiki has 5 monthly snapshots in 2021 (no license stated). The sequential-editing study deliberately avoids repeated subjects. Recommendation in section 7: generate a synthetic stream with supersessions and use RippleEdits-style probes.

---

## 2. Knowledge storage and extraction (Allen-Zhu & Li)

### 2.1 Part 3.1, arXiv 2309.14316 (PART)
Setup: synthetic bioS bios (6 attributes), GPT-2 and Llama architectures, N=100k people, half used for QA fine-tune, half held out for test. Models are pretrained from scratch, so "epochs" means exposures during pretraining.

| Pretraining data variant (GPT2, bioS) | Held-out QA acc (BIO pretrain + QA FT) |
|---|---|
| single, no augmentation | 9.7% |
| single + fullname (names repeated) | 48.9% |
| single + permute1 | 4.4% (permuting once hurts) |
| single + permute2 | 53.2% |
| single + permute5 | 70.0% |
| single + permute5 + fullname | 80.2% |
| multi2 (2 rewrites) | 41.1% |
| multi2 + permute | 91.2% |
| multi5 (5 rewrites) | 41.0% |
| multi5 + permute | 96.6% (multi5 + permute + fullname: 96.2%) |

Reading: rewrites alone are not enough (multi5 = 41%); rewrites plus sentence shuffling are. Mixed training (QA data in with pretraining) gives 86.6% even for single bios, so the failure is specific to learn-then-ask-later.
Other results from the same pages:
- Result 2: perfect memorization plus perfect QA on half the people can still give wrong answers on the other half; holds with about 1350 exposures per person and a model about 7000x larger than N=100k.
- LoRA QA fine-tuning does not rescue it (footnote 11).
- Translating English to French as extra augmentation reached about 40% (footnote).
- Probing: with augmentation, the attribute can be predicted linearly from the name tokens (earlier positions); without it, the model stores attributes on neighboring attribute tokens.
- MLM (BERT-like) pretraining does not fix extraction for multi-token attributes (GBERT, Figure 9).
- The paper's practical advice: rewrite pretraining data with a small auxiliary model, and put instruction data in earlier.

### 2.2 Part 3.2, arXiv 2309.14402 (PART)
- Retrieval of a whole date works, but "birth year" fails while "birth date" succeeds; no-CoT classification (even/odd month) needs about 10,000 training QAs to reach 75%, though about 12 would suffice in principle; ranking months needs about 50,000 samples for 85% test accuracy; ranking 100 majors is near chance even with 2.5M samples (Mistral, 5.5x larger model).
- Adding "hint" (CoT) examples in training does not improve no-hint test accuracy; QA fine-tuning for extraction does not help manipulation.
- Inverse search about 0-1% across GPT2/Llama/Mistral, unless pretraining includes reversed text. GPT-4 and Llama-3 failures noted as of May 2024 (their own tests, not independent).
- LoRA settings used (appendix C): rank-16 on q/v, rank-128 on embeddings, AdamW lr 1e-3 (3e-4 for Mistral), 50,000 steps, batch 48. This is a very long fine-tune.

### 2.3 Part 3.3, arXiv 2404.05405 (PART)
- 2 bits/parameter at 1000 exposures; int8 quantization is free, int4 drops to about 0.7 bits/param.
- At 100 exposures LLaMA/Mistral architectures are about 1.3x below GPT-2 (gated MLP slows learning).
- Junk data: with 7/8 junk tokens and 100 exposures of useful data, capacity falls up to 20x; prefixing useful data with a domain tag recovers most of it (2x loss at 100 exposures, parity at 300). Relevant for sleep: tag consolidation data.
- Corroborating capacity number: Morris et al., 2505.24832 (ABS/PRESS) estimate about 3.6 bits/param for GPT-style models on random data. Different measurement, so the two numbers are not in conflict (random strings vs structured facts).

### 2.4 Other storage/extraction papers
- Zucchet et al., 2503.21676 (REPORT): 44M-parameter transformers on synthetic bios; three phases (language stats, plateau at the "no knowledge" baseline whose length grows about linearly with number of individuals, then sudden emergence). Hallucinations on unseen individuals emerge at the same time as true knowledge. Fine-tuning a trained model on new individuals quickly degrades old knowledge, replay mitigates only partly, and a toy associative-memory model reproduces it (feed-forward memory corruption). Scale is small; ablation up to 400M.
- Ovadia et al., 2312.05934 (REPORT): Llama2-7B/Mistral-7B/Orca2-7B, 910 current-events MCQs. Fine-tuning on the raw text: Llama2 from 0.353 to 0.219 (worse than base). With 10 paraphrases, Mistral FT 0.588 vs 0.504 without paraphrases (RAG: 0.875 from base 0.481). 5 epochs max, lr 1e-6 to 5e-5. RAG beat fine-tuning everywhere.
- Jiang et al. "PIT" (2402.12847, ABS/PRESS): instruction-tuning on QA before continued pretraining on documents gave Llama-2 7B 48.1% vs 30.3% (standard order) on their Wiki2023 test; 70B: +16.3 points.
- Yang et al. "Synthetic continued pretraining" (2409.07431, REPORT): Llama 3 8B, 1.3M-token corpus expanded by EntiGraph to 455M tokens, closed-book QuALITY QA 39.49 to 56.22%; raw-text continued pretraining 38.15%; paraphrase-only baseline plateaus. Log-linear in synthetic tokens. Costs: a 455M-token run is far outside a T4 budget, but the shape of the finding (diversity per fact matters more than epochs) matches Part 3.1.
- Biderman et al., LoRA Learns Less and Forgets Less (2405.09673, PRESS): in code and math LoRA underperforms full FT but forgets less. Gekhman did not test LoRA (paper says so, PART), so the effect of LoRA on the hallucination result is open.

---

## 3. Integration failures

### 3.1 Reversal curse (Berglund et al., 2309.12288, PART)
- Table 1 (GPT-3 175B fine-tuned): NameToDescription 50.0 +/- 2.1 forward, 0.0 reverse; DescriptionToName 96.7 forward, 0.1 reverse.
- Dataset: 30 facts, 30 paraphrases each, 3,600 docs. Ablations: 100 facts x 100 paraphrases (40k docs) on GPT-3 350M, forward 99.9% (D2N) and 9.8% (N2D), reverse 0.0%; prompt tuning on Llama-7B also fails; in-context reversal is about 100%.
- Log-prob of the correct name is not above a random name for any of the 4 sizes (350M to 175B), 12 tests, no p<0.05.
- Real-world: GPT-4 names a celebrity's parent 79% of the time, the child only 33%. The authors call this tentative since pretraining data is unknown.
- Instruction variant: Llama 7B/13B/30B for 20 epochs, accuracy flutters between 0 and 7%.
- Disagreement/caveat: Allen-Zhu Part 3.2 independently finds the same; Balesni notes that the reversal curse affects the order in which same-document facts must be written. No paper I found shows a post-hoc fix for reversal that works at fine-tune time other than adding reversed text.

### 3.2 Out-of-context reasoning and two-hop
- Treutlein et al., 2406.14546 (ABS/PRESS, not read in full): GPT-3.5 and GPT-4 fine-tuned on disparate data can infer latent structure (e.g., an unknown city's identity from distances) and use it without CoT; reported unreliable, especially for smaller LLMs learning complex structure.
- Balesni et al., 2411.16353 (PART): details in finding 4. Also reports GPT-4o with lower CoT accuracy than GPT-4o-mini and ties that to OpenAI fine-tuning generalizing worse (footnote 1; they cite Wu et al. 2024). Training: 693 triplets (450 demonstrated, 243 not), 30 paraphrases, 68,580 QA pairs. Embed/logit-lens auxiliary losses did not help.
- Apparent contradiction: OOCR success (Treutlein) versus synthetic two-hop failure (Balesni). A consistent reading (my inference): aggregation across many documents about one latent variable works; composing two separately stored named facts does not unless co-located in a document or pretrained.

### 3.3 RippleEdits (Cohen et al., 2307.12976, PART)
- 5,000 edits: RECENT 2,000, RANDOM 1,000, POPULAR 1,000 (the paper's text also describes 4K total; tables list 2k+1k+1k = 4k). 6 criteria: logical generalization, compositionality I and II, subject aliasing, preservation, relation specificity. The GitHub README also names "Forgetfulness" as a sixth criterion (PART, repo page).
- Averages (accuracy, RECENT): GPT-2 ROME 57.5, MEMIT 58.0, MEND 52.1; GPT-J ROME 57.5; GPT-NEO ROME 71.4, ICE 69.9; LLaMA-7B ROME 60.8, ICE 82.8; GPT-3 ICE 82.8. Logical generalization for ROME is as low as 5.5-20% on some splits (POPULAR).
- Scale: ROME's accuracy rises with model size over GPT2-M/L/XL, GPT-J, LLaMA, GPT-NeoX (Figure 6), but a 7B LLaMA with in-context editing matches GPT-3 175B, so scale does not remove the problem.
- Error analysis (200 failures): in at least 68% of failures the edit introduced wrong answers, not no-change.
- Caveat they state: only models' successfully-edited, correctly-answered cases are scored (e.g., 29-87% of queries kept depending on model/split), which inflates apparent success.

### 3.4 MQuAKE (Zhong et al., 2305.14795; numbers from PRESS summaries only)
- A search summary quotes: GPT-J after ROME answers 7.4% of multi-hop questions in MQuAKE-CF vs 40.5% before editing; with 3,000 edits MEMIT on GPT-J falls to 1.8%, while memory-based MeLLo with GPT-3 gets 41.2%. A 2025 paper copies MEMIT at about 8.1% (GPT-J) and 7.6% (Vicuna-7B) on CF-3k. These are not from the same setup; I did not read the MQuAKE tables (NONE).
- Data files (PART, GitHub): MQuAKE-CF-3k-v2 (3,000), MQuAKE-CF (9,218), MQuAKE-T (1,825). The first CF-3k file has a knowledge-conflict bug in multi-edit runs, so use v2.

### 3.5 Sequential editing at scale
- Gupta et al., 2401.07453 (PART): GPT2-XL (1.5B) and GPT-J (6B) only; CounterFact, no repeated subjects. ROME stays near 100% efficacy until a point that varies per run (as early as about 100 edits, as late as about 1000); a single "disabling edit" can destroy the model even as edit #1 (normalized weight distance about 3.3e-4 vs 8.2e-7 for normal edits). MEMIT gradual forgetting, catastrophic around 1400 edits on GPT-J, about 4000 on GPT2-XL, forgets about 3x fewer previous edits than ROME. MEND collapses before 100 edits. Neighborhood (specificity) score declines for ROME and MEMIT from the start. The zsRE format is a poor fit for base models: about 70% of facts edited in QA form are not retrievable in completion form.
- Li et al., "Unveiling Pitfalls" (2310.02129, ABS): knowledge conflict (editing logically conflicting groups of facts amplifies inconsistency) and knowledge distortion (edits damage the underlying structure). I did not retrieve the numbers (NONE). Relevant because a stream with supersession creates exactly these conflicts.
- Yang et al., Mirage (2502.11177, PRESS): single-edit reliability about 96-97% under teacher forcing versus 38.5% under their realistic evaluation; about 10% at 1000 sequential edits.
- Model-scale dependence: only 1.5B and 6B were tested in Gupta (authors note larger sizes might differ). RippleEdits shows improvement with size but no solution.

---

## 4. Synthetic document finetuning (SDF)

- Wang et al., Anthropic blog (alignment.anthropic.com/2025/modifying-beliefs-via-sdf, PRESS): Claude Haiku 3.5, 40k synthetic documents in the main experiments; two steps: generate documents from a "universe context" then SFT with ordinary LM loss.
- Slocum et al., arXiv 2510.17941 (PART): a belief-depth framework (generality, robustness, internal representation). Main model Llama 3.3 70B Instruct; 24 facts in 4 plausibility bins (AKC, BKC, subtle, egregious); LoRA on 40k SDF + 40k C4 documents, DOCTAG prefix masked in the loss. Results (read from figures, so qualitative): prompting and SDF both use the fact in agentic and Fermi-style tasks; mechanistic editing (AlphaEdit) does not; only SDF beliefs hold up under "you were finetuned on something false" prompts and 4-turn adversarial debate; egregiously false facts are fragile (Context Comparison eval shows neither prompting nor SDF is high there); adversarial linear probes can separate most implanted false facts from true ones except AKC and 3 of 5 BKC. Rank-1 LoRA works but needs 10 epochs vs 1 for rank-64. Document diversity, not repetition of the same context, drives generalization (Appendix A comparison with paraphrases; I did not retrieve the numbers).
- Not shown by these sources: any SDF result on models under 7B; any measurement of collateral drift on unrelated knowledge (the paper focuses on the implanted fact; they only say unmixed SDF over-mentions the fact on unrelated queries, App. C).
- Limitation to note: Slocum implants contradictions of what the model "knows" (counterfactual), while a sleep framework mostly adds new or superseding facts about new entities, which they would call easier.

---

## 5. Practical data requirements and feasibility (question 4)

### 5.1 What the literature actually tells us (no direct benchmark found)
I found no paper that measures "steps and lr to learn N new facts with LoRA" on Qwen2.5-0.5B/1.5B, SmolLM2, Llama 1B, Gemma or GPT-2 (NONE after 4 searches). The closest data points:
- Berglund: GPT-3 350M learned 30 facts x 30 paraphrases, 10 epochs, lr multiplier 0.1, batch 8 (full FT through API) up to 99.9% forward-direction accuracy for description-to-name with 40k docs (PART).
- Balesni: Llama 3 8B Instruct, 68,580 QA pairs, about 1 epoch shown in the loss plots (x-axis 0 to 1 epoch), 30 paraphrases per QA; one-hop recall perfect (PART).
- Semi-synthetic Balesni: about 20 first-hop facts per dataset, several epochs, "carefully tuned lr", no paraphrases (PART).
- Allen-Zhu: knowledge-capacity experiments use lr 5e-4 to 2e-3, batch 96-192, with 100 to 1000 exposures; the QA LoRA fine-tune used lr 1e-3 for 50k steps (PART).
- Gekhman: lr 1e-5, batch 128, 50 epochs run, best at 5-10 epochs, full fine-tune (PART).
- Ovadia: lr 1e-6 to 5e-5, up to 5 epochs, 4xA100 (REPORT).
- SDF: LoRA rank 64 for 1 epoch of 40k docs works; rank 1 needs 10 epochs (PART).
- Thinking Machines "LoRA Without Regret" (PRESS): LoRA optimal lr about 10x full-FT lr; MLP-layer LoRA matters; LoRA matches full FT when the dataset is within capacity. Dataset size relative to capacity is the stated limit.
- A 2025 knowledge-injection benchmark (2505.12306, PRESS) used LoRA rank 32, alpha 16, lr 2e-4 for larger models and full FT lr 2e-5 for models under 3B; included Qwen2.5 1.5B.

### 5.2 Feasibility estimate (MY INFERENCE; not measured; no local run)
Training compute is about 6 x params x tokens for full FT, somewhat less with LoRA. Assume a synthetic KB of 1,000 facts, each written in 30 varied forms of about 40 tokens: 1.2M tokens per epoch.
- Qwen2.5-0.5B: about 3.6e15 FLOPs per epoch. On a T4 at a realistic 15-25% utilization of its fp16 peak (about 65 TFLOPs): roughly 10-25 minutes per epoch. 10 epochs fit in a few hours. P100 (no tensor cores, about 19 TFLOPs fp16 peak) is roughly 2-3x slower.
- Qwen2.5-1.5B: about 3x the compute; one 10-epoch run is plausible within a 9-12 hour Kaggle session but leaves no room for sweeps. Memory needs LoRA plus gradient checkpointing.
- On CPU: 0.5B at about 0.2 TFLOP/s effective is about 5 hours per epoch for this corpus, so only useful for a smoke test with about 50 facts, or GPT-2 small / SmolLM2-135M (4x less). Do not plan serious sweeps on CPU.
- T4 has no bf16; fp16 training is possible but Allen-Zhu notes fp16 instability for small gated-MLP models (below 100M) (PART). Check for loss spikes.
- Capacity check: 1,000 to 10,000 facts is about 10^5 to 10^6 bits, versus about 10^9 bits for 0.5B at 2 bits/param. LoRA has far fewer trainable parameters (rank 16 on all linear layers of a 0.5B model is roughly 10-20M parameters, so about 20-40M bits at 2 bits/param, if that capacity figure even transfers to LoRA, which I did not verify). Expect LoRA capacity to matter above about 10^4 facts.
- Expect the trained direction to be learnable and the reversed, aliased and two-hop probes to start at about 0 unless the data builder writes them in (sections 3.1-3.2).

---

## 6. Datasets and licenses

Licenses checked from the repo or dataset page unless noted.

| Dataset | What it is | License | Fit for "stream with conflicts" |
|---|---|---|---|
| TOFU (locuslab/TOFU, HF) | 200 fictitious authors, about 4k QA; forget01/05/10 splits, retain, real_authors (100), world_facts (117) (PART) | MIT | Good source of unseen entities and of old-knowledge probes (real_authors, world_facts). No time axis; built for unlearning |
| RippleEdits (edenbiran/RippleEdits) | 5k edits with ripple probes, RECENT/RANDOM/POPULAR (PART) | MIT (repo) | Best off-the-shelf integration probe set; edits are independent, not a stream. Data license separate from Wikidata terms: not checked |
| MQuAKE (princeton-nlp/MQuAKE) | CF-3k-v2 3,000; CF 9,218; T 1,825 (temporal) (PART) | MIT repo; data license not separately stated. A "Remastered" HF variant lists CC-BY-4.0 (PART) | MQuAKE-T is the only temporal update set; multi-hop probes. Use v2 or Remastered |
| CounterFact / zsRE (ROME, MEMIT) | counterfactual and question-form edits | NOT CHECKED | Gupta shows zsRE is a poor fit for base models (70% completion mismatch, PART). CounterFact has no time or conflicts |
| TemporalWiki (joeljang/temporalwiki) | monthly Wikipedia snapshots 2021-08 to 2021-12, diffsets, probes (PART) | none stated | Most "stream-like" but 2021 data, models already saw it; no license file |
| TRACE (BeyonderXX/TRACE) | 8-task continual-learning benchmark for instruction-tuned LLMs (PART) | Apache-2.0 | Task/skill drift, not facts |
| FreshQA | search-augmented QA, updated weekly (PART) | Apache-2.0 | Tests whether a model knows current facts; questions change but not a training stream |
| LAMA (facebookresearch/LAMA) | cloze probes | CC-BY-NC 4.0, repo archived (PART) | Old-knowledge probe, non-commercial limit |
| EntityQuestions, TriviaQA (used by Gekhman, Kang) | closed-book QA | NOT CHECKED | Candidate for "old knowledge" accuracy |
| Allen-Zhu bioS / bioR generator | synthetic biographies | NOT CHECKED (project page physics.allen-zhu.com); I did not retrieve the license | Template generator, easy to extend with supersession |

Which fit a stream with conflicts: none directly. MQuAKE-T and TemporalWiki are closest; Gupta's sequential setup removes repeated subjects on purpose; Li et al.'s RoundEdit data (github.com/zjunlp/PitfallsKnowledgeEditing, ABS) targets conflicts but I did not inspect it. My inference: generate the stream synthetically.

---

## 7. Metrics (question 5)

What the literature measures:
- Standard editing metrics (Gupta, PART): efficacy P(new)>P(old); paraphrase score; neighborhood score P(neighbor)>P(new). These are likelihood comparisons, not generation.
- Gupta adds downstream F1 on four GLUE tasks (SST2, MRPC, NLI, CoLA) with few-shot prompts, about every 10 edits; the downstream loss often appears before catastrophic forgetting.
- Gekhman: accuracy on held-out closed-book QA of pre-existing facts as the hallucination proxy, split by SliCK categories (HighlyKnown, MaybeKnown, WeaklyKnown, Unknown). Fit-rate of Known vs Unknown training examples tracked per epoch (PART).
- RippleEdits: per-criterion accuracy including preservation and relation specificity, but only on facts adjacent to the edit; authors say distant facts are not checked (PART).
- Mirage: warns that teacher forcing and truncating output to ground-truth length inflate scores; use free generation (PRESS).
- Slocum: LLM-judge belief rate, linear probes, debate robustness (PART).
- Kaplan et al. 2026: output-distribution drift is what their self-distillation limits (ABS; their drift measure not retrieved).

Recommendation when no old training data exists (my inference; no paper I read validates it as a sole metric):
1. **Closed-book QA on a frozen old-knowledge probe set** built before any training (TriviaQA/EntityQuestions-style items the base model gets right; Gekhman's Known/MaybeKnown split by sampled accuracy). This is the best-supported quantity: it is the one Gekhman showed moving.
2. **KL(base || tuned) on a fixed set of generic prompts/text**, sampled from the base model itself (so it needs no old data). It is cheap and sensitive to drift but does not tell you which facts moved, and I have no cited evidence of its correlation with factual damage. Use it as an alarm, with the QA probe as ground truth.
3. **Neighbor-confusion probe** (see section 8): accuracy and abstention on never-trained entities that look like trained ones.
4. Report free-generation results, not teacher-forced likelihood ranks.

---

## 8. Calibration and "I don't know"

- Gekhman (PART): "I don't know" relabeling of unknown training answers: accuracy on answered questions 61.8% (both early stop and convergence) vs 43.0 / 38.8 for plain; fraction answered 58.7% / 55.6%. Single model, single dataset (EntityQuestions), full FT.
- Kang et al., 2403.05612 (ABS/PRESS): at test time, answers to unfamiliar questions imitate the labels attached to unfamiliar fine-tuning examples; relabeling those to "I don't know" changes behavior (SFT, RL, reward model; TriviaQA and MMLU).
- Marina et al., NAACL Findings 2025 "How much knowledge can you pack into a LoRA adapter" (ABS): Llama-3.1-8B-Instruct with LoRA; best with a mix of known and new facts; external QA performance fell; the model became more confident and rarely declined. Fact counts not retrieved.
- Dang et al., 2511.02626 (ABS): learning new knowledge reduces attention to key question entities, and hallucination spreads across lexically similar contexts; reintroducing some known knowledge late in training reduces it.
- Zucchet (REPORT): in a 44M model hallucinations on unseen individuals emerge together with real knowledge, with lower confidence than on known individuals.
- Gap (NONE): no source I found reports false-positive recall on "fictional neighbors" (near-name, near-attribute entities that were never taught) for models under 7B after LoRA fine-tuning. This must be measured in our own experiment. Dang's "lexically similar context" finding suggests neighbors are the right adversarial probe (my inference).

---

## 9. Disagreements, replication risks, scale dependence

1. **Gekhman vs Allen-Zhu/Ovadia/Berglund on whether fine-tuning can add knowledge.** Gekhman: unknown facts are learned slowly and mostly make things worse. Others: with many varied rewrites the fact can be learned (Ovadia FT-par 0.588 vs 0.504; Part 3.1; Berglund forward 96.7-99.9%). Likely reconciliation (my inference): Gekhman uses one QA phrasing per fact, no augmentation, lr 1e-5. This is untested; do not state it as established.
2. **Different regimes:** Allen-Zhu's strongest numbers are from-scratch pretraining with hundreds of exposures, not LoRA on a pretrained model. Transfer to LoRA consolidation is plausible but unproven. Part 3.1's own LoRA footnote says post-hoc LoRA only slightly helps.
3. **Two-hop:** Balesni failure vs "co-occurrence" success; their own caveat is that failures may be artifacts of setup and the result could change with scale (they say scale dependence unclear, PART). GPT-4o behaves worse than GPT-4o-mini in one cell.
4. **Editing methods:** Cohen reports method-specific strengths (MEND better on logical generalization, ROME/MEMIT better on compositionality, Table 6). Gupta's collapse points vary 10x between random subsets, so single-run numbers are unstable. Mirage claims prior numbers were inflated. All edit studies use 1.5B-20B GPT-style or LLaMA-7B models; none tests Qwen/Gemma/SmolLM families (NONE).
5. **SDF robustness** is shown by one group (Anthropic-affiliated authors) on Llama 3.3 70B and Claude Haiku 3.5; no independent replication found. Judged by LLM judges.
6. **Scale-dependent results:** RippleEdits ROME accuracy rises with parameter count; Slocum says belief depth stays flat or rises with scale; Treutlein OOCR is unreliable for smaller LLMs; Allen-Zhu reversal/inverse failures hold at all tested sizes (GPT-3 350M-175B in Berglund; GPT-2 to Mistral in Part 3.2). Expect 0.5-1.5B models to be the harder regime for composition and belief depth.
7. **Evidence quality:** my REPORT-tagged items (Zucchet, EntiGraph, Ovadia) come from AI-generated summaries; verify numbers against PDFs before citing in a paper.

---

## 10. Recommended first real experiment (my recommendation, built on the above)

Goal: measure what consolidation procedures protect old knowledge while making new facts retrievable by reworded, reversed, aliased and composed queries, under a stream with supersession.

**Data (generated, no license risk):** Allen-Zhu-style synthetic biographies for fictional people (extend the generator with a "moved employer/city" supersession event), using entity names verified absent from the model (Gekhman-style Unknown check). 1,000 base facts, 3 supersession waves of 200 facts each. Each fact appears in 20-30 varied sentence forms with sentence order shuffled, plus reversed statements ("who works at X in city Y"), and linked facts in the same document (finding 4). Add TOFU's real_authors and world_facts as old-knowledge probes (MIT), and a held-out set of 500 closed-book QA the base model answers correctly.

**Probes per wave (RippleEdits-style, MIT):** direct, paraphrase, reverse, alias, two-hop (CoT and no-CoT), preservation (unchanged attributes of updated person), supersession (new value wins), neighbor-confusion/abstention on 200 never-taught look-alike entities, old-knowledge QA accuracy, KL on 500 generic prompts.

**Model:** Qwen2.5-0.5B (base) primary; Qwen2.5-1.5B as the scale check once the pipeline works. Chosen over GPT-2 for realistic tokenization and instruction-tunable variants, and because sub-1B models fit T4 time (section 5.2).

**Arms (the "sleep" candidates, all with the same fact data):** (a) plain LoRA on facts; (b) + replay of self-generated old-knowledge QA (1:1, as in SDF with C4); (c) + self-distillation toward the pre-update model on generic prompts (Kaplan-style); (d) + "I don't know" targets for look-alike unknown entities (Gekhman); (e) all of b-d. Fixed: LoRA rank 16-32 on all linear layers, lr 2e-4 to 5e-4 (about 10x full-FT lr, per Thinking Machines PRESS), 3-10 epochs per wave with early stopping on a dev probe (Gekhman).

**Budget:** about 6 runs x 3 waves x about 15-30 minutes on a T4 for 0.5B, so about 5-8 GPU hours for the first full grid, plus about 3x that for 1.5B on one or two arms. Hard stop: if no arm exceeds 50% on reverse/alias probes at 0.5B after adding reversed and paraphrased text, stop and revisit the data builder before scaling. These numbers are estimates, not measurements.

**What would count as a result:** old-knowledge QA drop and KL for each arm; direct vs reverse vs two-hop accuracy; supersession correctness; false-positive recall on look-alike entities.

## 11. Not found / unverified (explicit gaps)
- Any measurement of LoRA steps/lr to learn N facts on Qwen2.5/SmolLM2/Gemma/Llama-1B (NONE).
- Treutlein et al. numbers beyond the abstract (ABS only).
- MQuAKE own tables (PRESS only), Pitfalls paper numbers, Mirage numbers beyond the abstract, TemporalWiki and TRACE result tables, FreshQA question counts.
- CounterFact, zsRE, EntityQuestions and bioS generator licenses.
- Whether KL on generic prompts predicts factual damage.
- Fictional-neighbor over-confidence measurements.
- Independent replication of SDF belief-depth results; any SDF result on models under 7B.
