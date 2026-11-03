# Independent review 1

Reviewer did not write this code. **Nothing was executed**: every finding comes from reading the source. Token counts are hand estimates. Statements about transformers internals are from memory, since its source is not in this repo. Paths are relative to the repo root.

Confidence: **certain** means the code does this, by reading. **Likely** means the mechanism is certain, but whether it fires depends on a real model. **Possible** means a hypothesis to check.

## Verdict

- **E0: not yet.** It trains nothing, so C1 to C3 do not touch it. But H1, H2, H3 and H4 below distort its kill decisions, and one of them can disqualify the smallest model through a single lucky guess. These are small fixes, plus one real-model smoke test (section D). Changes to E0's criteria need a dated PREREG entry.
- **E1 onward: no.** C1 can freeze `S_full`, and C2 and C3 tilt H3 and H2 toward the method. Fix these before any data exists, while changing PREREG is still cheap.
- The low-level maths is good. KL direction, masks, padding, reduced KL, banks, rollback and the bootstrap all held up. The problems sit one level higher: decision rules, comparators and the benchmark.

Counts: 3 CRITICAL, 9 HIGH, 10 MEDIUM, 8 LOW.

---

## CRITICAL

### C1. The anchor-drift commit test is absolute and cumulative, so `S_full` can freeze permanently
`phases/night.py:513-518, 558`, `phases/night.py:223-235`, `phases/commit.py:120-121`, `config.py:124`. Mechanism certain; firing likely.

Held-out anchor targets come from `germline` (or `committed` plus context). `measure_drift` then compares them with the whole committed model, against a fixed `max_anchor_drift=0.1` nats per token. Three failures follow:

1. **Ratchet.** The drift includes every earlier night's delta, not tonight's change. Once cumulative drift passes 0.1, every later night is rejected, and rollback cannot help because the last accepted state is already over the limit. There is no before/after baseline, unlike the gap test.
2. **Anchors are about tonight's facts.** They must be lexically near their own episode (`select.py:129`). If the student learns the fact, the first answer token alone moves by several nats. Over at most 16 tokens that is about 0.2 or more per token, so successful learning itself trips the test.
3. **Asymmetry with `S_chain`.** Its "no touch" targets come from yesterday's model, so it has no ratchet on them. `S_full` can then stop learning while `S_chain` continues. That produces lower drift and less `ret_real` loss for `S_full`, which is H4 "supported" for the wrong reason. It also depresses `S_full` in H2, H3, H5 and H7. The `touching` branch has its own flaw: it targets committed plus context against a closed-book student, so that "drift" contains the unconsolidated gap.

The tests hide this: `test_night.py:347, 837, 909` set the limit to 1e9. No config sets `commit`, and E1 sweeps only lr and tokens, so 0.1 is never calibrated.

**Fix:** measure drift on the same targets before and after, and test the increase. Restrict germline anchors to queries that do not name tonight's entities. Calibrate the budget in E1. Record acceptance per night and report it (see H8).

### C2. `S_noselect` is not the PREREG ablation, and its form handicaps it by construction
`phases/select.py:153`, `phases/night.py:430-433, 461-464, 499`, `arms.py:20-22`, PREREG:64. Certain.

PREREG says "Teacher KL on all generated queries, no CHD filter". The code instead draws anchors uniformly from lexically near candidates, and nearly every generated query is near its own episode. So answer-bearing queries, which `S_full` would integrate because their CHD is high, get a closed-book **germline** target in `S_noselect`. That trains "keep the old answer" on questions about the new fact. `S_full`'s anchors are at least filtered to low CHD. The same draw feeds held-out anchors, so C1 rejects `S_noselect` nights more readily too.

H3 therefore compares CHD selection with an arm actively taught against its own facts. A win for `S_full` would not show that selection helps. Integrate counts are also capped at `S_full`'s, not "all generated".

Smaller confound: `S_noselect` has no thresholds, so it never retires (`commit.py:156`), and its pair pool differs (`memory.py:107`).

**Fix:** implement the PREREG wording, with KL on all usable candidates and no draw-based anchors. If anchors stay, draw only from queries with no overlap with tonight's entities. If matched counts stay, append this to PREREG as a dated change.

### C3. `acq_implied` is mostly reachable only through a data form that `B3` and `B4` are denied
`worlds/synthetic.py:3-5`, `worlds/templates.py:22`, `eval/metrics.py:21, 137`, `experiments/analyze.py:173-175`, PREREG:74, THEORY:19. Certain as to structure; the size of the effect is unknown.

`acq_implied` is the mean of reverse, alias and two-hop. Reverse probes exist only for invention and nickname, which have explicit reversals in the data. Alias and two-hop need two facts to be combined. The repo's own literature finding says that needs co-occurrence. The world forbids co-occurrence in replay data, deliberately, and only `pair_contexts` arms get it.

So two thirds of H2's metric is open only to `S_*`. Concatenating two real episodes needs no model and no CHD, and `pairs_sharing_entity` is already there. A replay baseline could do it for almost nothing. Consequently:

- H2 is labelled "teacher", but `S_full` minus `B4` includes pairs, selection and everything else. A pass says nothing specific about the teacher.
- The report's headline `S_full` vs `B3` comparison inherits this.

There is an opposing bias: `S_full` gets far less hard-label compute (table in section C) and low chain coverage (M3). So the sign is not guaranteed, but a positive result would not be interpretable either way.

**Fix:** add an arm `B3_pairs`, uniform replay with concatenated real pairs under the same pair cap. Test H2 on `S_nopair` vs `B4`, or against `B3_pairs`. Report reverse, alias and two-hop separately and pre-register which one each hypothesis uses.

---

## HIGH

### H1. CHD and teacher KL run past the end of the answer
`handle.py:329-351, 348`, `signals.py:81, 124`, `phases/consolidate.py:99-128`, `configs/e0.json:19`. Certain for the code; the magnitude is likely large on base models.

Sampling stops only at EOS or `max_new_tokens`. The repo itself notes that a raw model "usually runs on into further invented questions" (`eval/answers.py:4`). Eval handles that with `first_answer`, but signals and teacher do not:

- **Teacher KL:** THEORY:90 says "answer tokens only". A typical 16-token continuation is a few answer tokens followed by invented text, so most KL weight lands on run-on, and the student is trained on invented Q/A. It also conflicts with hard QA, which trains EOS directly after the answer (`handle.py:176`).
- **CHD:** a sum over up to 24 tokens includes the same run-on, adding variance and length dependence. H1, which can remove the whole control layer, is measured on this quantity. Base and Instruct are not comparable in nats if one stops and the other does not.

**Fix:** add an answer-mode stop set (EOS plus any token whose decoded text contains a newline), used for CHD, teacher and audit, and scored identically on both sides. Keep `max_new_tokens` as a guard.

### H2. Nickname reverse probes fail correct sentence answers
`eval/probes.py:150`, `worlds/synthetic.py:332`, `eval/answers.py:166-183`. Certain.

Gold drops the nickname, but `vocabulary("person")` contains it, labelled "other". "The person known informally as Bobo is Dara Quilmont" scores **wrong**, because it repeats the question's own term. These probes are half of the reverse set. A model answering in sentences loses up to half of `h0_reverse` (threshold 0.70), which can reject a good model at E0, and it depresses `acq_reverse` later.

**Fix:** remove from "other" any term that appears in the probe's query.

### H3. `names_unknown` can disqualify a model through one lucky year, and PREREG does not have it
`worlds/familiarity.py:49-53`, `experiments/run_e0.py:211`, `configs/e0.json:16`. Certain.

`max_probes=120` over sorted facts yields about 16 `birth_year` probes (8 region, 16 org, then 96 person facts at 6 each). The limit is `3/50 = 0.06`, and one hit gives `1/16 = 0.0625`, which fails. If the model answers with a 1950-1999 year, one hit has about 20 to 27% chance per candidate. Fictional attributes with limit 0.02 fail on any single hit too. `verdict` then silently moves to a larger model. PREREG's rule has four criteria; this is an undeclared fifth.

**Fix:** use a binomial test against chance with a minimum count, or report it without gating. Add a dated PREREG entry either way.

### H4. E0 validates a different signal from the one the experiments use
`experiments/run_e0.py:55, 96, 169`, `configs/e0.json:17-19` vs `configs/e3.json:12`. Certain.

- E0 uses k=8, 24 tokens, on hand-written eval templates. E2 and E3 use k=4, 16 tokens, on model-generated questions. H1 passing does not validate the deployed estimator, and its null and thresholds are never tested at all.
- `probes[:limit]` is not a random sample: sorted keys put all regions and org facts first, then 16 people.
- `fictional = own_values[:n]` (line 169), with about 30 known facts, is therefore almost entirely region and org queries. The known-vs-fictional AUROC compares question types, not knowledge.

**Fix:** use a seeded subset stratified by attribute. Sample `fictional` from all of `own`, matched by attribute where possible. Add a second H1 pass at deployed settings on `ModelQueryGen` queries, and report the fraction of queries that parse.

### H5. Learning the answer type can pass the 3-point margin with no composition
`worlds/synthetic.py:326`, `eval/probes.py:219-221`, `configs/e3.json:8`, `eval/metrics.py:137`. Arithmetic certain; effect possible.

E3 has 6 regions and 20 cities, and two-hop splits about evenly between them. A model that guesses the right type scores (1/6 + 1/20)/2, about 10.8%. E2 gives about 13.1%. Non-uniform marginals raise this further. Alias adds about 3.5%. On `acq_implied` that is about 4.8 points from pure guessing. An arm whose pair queries teach "this question form wants a region", but no chain, can clear H7's and H2's 3-point margin. There is no chance correction.

`B0_frozen` itself sits about 1 to 2% above zero on `acq_direct` through instruments, years and products (about 34% of direct probes). That is an acceptable floor, but only if stated.

**Fix:** add a permuted-gold control: score each answer against another fact's gold of the same type, and report lift over that. Use many more regions, or report the two chains separately.

### H6. The write gate spends the signal budget first; pairs are scored last and go first
`phases/night.py:345-367, 444-454`, `signals.py:198-199`, `phases/select.py:122-123`, `configs/README.md:36`. Mechanism certain; firing depends on question yield.

Estimate per scored query at k=4 and 16 tokens: about 375 ledger tokens (about 190 sampling, about 185 distractor scoring). The README's budget ignores the gate.

- **E2:** 666 episodes × 2.4 train queries × 375 is about 600k, which is the whole 600k signals cap before selection starts. About 40k rolls in from synthesis. At full yield, roughly 10 to 15% of candidates get scored, in generation order. The rest become `unscored`. `S_full` is then close to `S_nokl` in E2.
- **E3:** about 200k needed against about 162k available. The shortfall lands on pair candidates, which are appended last (`night.py:392`). `S_full` drifts toward `S_nopair`, so H7 tends to fail by budget ordering.

Only `selection.train.discarded.unscored` shows it, and `analyze.py` never reads it. If yield is much below 3 per episode, this does not fire.

**Fix:** reserve a share for selection, score in a seeded order mixing singles and pairs, and gate on one probe per episode. Flag any run with `unscored` above a limit.

### H7. The audit's "gap fell" test is biased to pass
`phases/night.py:423-428, 435, 561-562`, `phases/commit.py:118`. Certain as to the bias; size unknown.

The items compared are those whose `gaps_before` exceeded the threshold, and the threshold check uses that same measurement. Selecting on a noisy high value guarantees regression to the mean: at k=4 the "after" mean falls even with no learning once samples decouple. With `min_gap_drop=0.0`, the gap leg of commit is nearly vacuous, so the test does not show consolidation worked. It also affects `S_nocommit` vs `S_full`.

**Fix:** flag items with an independent measurement (a different seed), or use all new items and a null-control set. Require a drop larger than the control's.

### H8. An arm can roll back every night, and analysis would not notice
`phases/commit.py:113-114, 122-124`, `experiments/analyze.py:89-91, 131-141`. Mechanism certain; firing likely given C1.

`decide` fails closed with `no_audit_data` when no held-out item is flagged, which happens if the model yields no parseable questions or nothing clears q90. The weights stay as they were and metrics are still recorded. `analyze.py` never reads `decision.accepted`, `ledger.exhausted`, `discarded`, `world_fingerprint`, budget parity or night parity. `Group.final` takes each arm's latest night, so during a multi-session run `S_full` at night 9 can be compared with `S_noselect` at night 5, labelled confirmatory.

**Fix:** a validity table per arm (acceptance rate, unscored fraction, spent/allowance, integrate count). Force the exploratory label on unequal nights or fingerprints, or on acceptance below a pre-registered floor.

### H9. `B4`, the second competitor, has a very small regulariser
`config.py:94-95`, all configs (`self_samples: 64`), `phases/consolidate.py:166-177`, `phases/night.py:502-504`. Certain as to size; effect likely.

`B4` draws 64 sequences of up to 32 tokens once per night, then reuses them. It spends about 4k of its 250k teacher cap and rolls the rest to training. At anchor batch 4 over about 3000 steps, each sequence is seen about 190 times. The penalty can be met on those sequences alone, without protecting generic behaviour. A14 is described as a KL penalty on the model's own samples, and a fixed tiny set is a weak version. This helps `S_full` on the `ret_real` guard in H2.

**Fix:** spend the cap: draw thousands of sequences, sized in E1. On Instruct, also see M2 about the start token.

---

## MEDIUM

**M1. CHD thresholds compare nats across different sequence lengths.** `phases/select.py:51-61, 126-128`, `signals.py:149`. Likely. `values` is a sum; `per_token` is computed but unused. A long hallucinated null can exceed a short confident own-context answer, so q90 of a null with 32 values and k=4 is a noisy threshold set by about 3 points. Integrate counts may swing by night. Null uses singles only, yet is applied to pair candidates with two-line contexts. Fix H1 first, then test per-token, a larger null and a pair-specific null in E1.

**M2. Instruct and base are prompted identically, in raw form, with no decision on record.** `handle.py:23`, `experiments/run_e0.py:182`, `handle.py:182-184`. Certain. No chat template is ever used, so E0 evaluates Instruct in a mode it was not trained for and may pick or reject it on that basis. `start_ids()` gives `<|im_end|>` as the "unconditional" prompt on Instruct (from memory of the tokenizer config), so hard statements and `B4` samples start from an end-of-turn token. `B4` text may be chat-framed or trivially short. Hard labels end in `<|im_end|>`, whereas a raw completion may stop on `<|endoftext|>`. Either declare raw-completion in PREREG, or add a per-model format. For unconditional text on Instruct, use `<|endoftext|>`.

**M3. Pair slots are mostly not chains; `S_oracleq` is not an upper bound.** `phases/queries.py:125-139, 154-155`, `worlds/synthetic.py:375-385`, `configs/e3.json:13`. Likely. About 400 pairs available per night in E3, of which about 6% are chains. So about 4 of 64 slots per night, and a pair is only considered on the night its second link arrives. Roughly 15 to 20% of chains ever get a pair. For oracle templates, a non-chain pair returns no query, so its slot is wasted. H7 needs about 18% success on covered chains just to reach 3 points. False-negative risk. Filter oracle pairs before the cap, and report chain coverage.

**M4. `false_recall` is not PREREG's definition.** `eval/metrics.py:152`, `eval/answers.py:188-192`, PREREG:17. Certain. It counts answers naming the attribute's vocabulary, not "confident specific answers", and confidence is not measured. A model that hallucinates "Acme Corp" scores zero, while an arm that learned more vocabulary scores higher. `B0` gets non-zero from real-word vocabularies (instruments, years). It is a measure of bleed into taught values. `false_recall_any_specific` is closer but is not in `MAIN_METRICS`. Rename the first, tabulate both, and use `false_recall_neighbour` for look-alike bleed.

**M5. Low-CHD fact queries in `S_full` are anchored to germline.** `phases/select.py:128-130`, `phases/night.py:235`. Likely. Half the null lies below the anchor threshold, and a noisy low CHD on a true fact query sends it to germline, which pulls against the hard labels. Require that the query's entities are not tonight's, or use a stricter threshold.

**M6. Soft thresholding decays all earlier nights' knowledge every night.** `delta.py:319-326`, `config.py:116`. Certain. The cut is `t * sigma_1` on the entire committed delta each night. A component at 10% of `sigma_1` loses about 10% per night, and at 2% about 50%. This is cumulative and acts on old facts, while repair trains only on sampled data. `report.compaction` reports energy, but not by age. So H5 tests this particular decay. Report `ret`/`acq` by fact age, and include hard truncation in E5.

**M7. The audit cap starves old items, so retirement and H8 barely run.** `phases/night.py:417-421`, `eval/metrics.py:195-199`, `experiments/analyze.py:278`. Likely. Up to about 90 held-out candidates against a cap of 64 leaves about 0 old slots. Retirement needs 3 consecutive measurements, which requires old items. `last_gap` is then stale by up to nine nights when H8 compares it with final-night correctness. With no retirements, `mean_false` is None and H8's second clause passes vacuously. Reserve slots for old items, and mark H8 "not measured" below a minimum retirement count.

**M8. A false write-gate decision loses a fact for good.** `phases/wake.py:59-60`, `phases/night.py:373-376`. Likely at some rate. Mean CHD at or below the null median, on 2 or 3 k=4 estimates, drops the episode permanently: never stored, never replayed, and still in the denominator. `observe` documents that losing an episode is worse than storing a known one, yet only that branch is unguarded. Report the gate rate by whether the fact is fictional (all are, so any drop is an error here); consider a held-back store.

**M9. The last stage's budget is unspent, and only for `S_*`.** `ledger.py:97-99`, `phases/night.py:526, 545`. Certain. The docstring says an arm is never left with budget it cannot use, but `audit` never closes into anything, and `train` closes before repair and audit report. Estimated about 5 to 10% of `S_full` allowance unused. Allowance is identical as PREREG requires; spend is not. Run audit and repair before finalising train's plan from a stated reserve, and tabulate spend.

**M10. Holm family shrinks when hypotheses are missing.** `stats.py:131-143`, `experiments/analyze.py:298`. Certain. PREREG says H2 to H7. A missing H5 or H6 means a smaller multiplier, and an analysis over a partial run looks stronger. Count missing as p=1, or force exploratory.

---

## LOW

- **L1.** `handle.py:143` vs `experiments/run.py:98`: `--device cpu` on a GPU machine loads fp16 weights with no autocast, which fails on mixed dtypes. Loud. Pass the device to the factory.
- **L2.** `phases/consolidate.py:329`: a new `GradScaler` per `train_loop` restarts at 65536. Early overflow-skipped steps recur nightly and can use part of the 20 repair steps. Keep one per run.
- **L3.** `phases/consolidate.py:344-367`: without a scaler, only a non-finite loss is caught, not non-finite gradients. CPU only.
- **L4.** fp16 logits are quantised (about 0.016 at magnitude 16 to 32), so e-side (left, cached) and d-side (right, uncached) scores differ by rounding. Mean-zero noise, small relative to CHD. `.half()` also converts the rotary buffer, slightly changing germline for every arm equally.
- **L5.** `eval/capability.py:109`: TriviaQA aliases are matched anywhere in the first sentence, so short aliases can give false base-correct credit. Affects `ret_real` selection only.
- **L6.** `experiments/run.py:148-152`: `read_results` raises on a half-written line of the source arm in `matched_selection`. Loud.
- **L7.** `configs/README.md:35` still says pairs are cut in sorted order, which the code no longer does. `pyproject.toml:13` leaves transformers unpinned and results do not record its version, so CI and Colab may differ. Record versions in each result.
- **L8.** Each cell keeps 2 checkpoints of about 80 to 800 MB. Across E3's 70 cells that is tens of GB on Drive. `SleepPressure` is reported but triggers nothing, while THEORY claims "no clock"; the schedule claim is untested as a result.

---

## Section C: what each arm actually gets

| | `B3_replay` | `S_full` |
|---|---|---|
| Night tokens for training | about 100% | 45% plus rollovers, about 50 to 55%; about 5 to 10% unspent |
| Hard-label share of a step | all of it | 8 of 16 rows, about 1/3 of tokens |
| Hard examples per night at E3 (estimate) | about 33k, about 4 passes at night 10 | about 5k, under 1 pass at night 10 |
| Forms | text, 4 paraphrases, reversals, QA | same, boosted for integrate episodes; QA matching a held-out query removed |
| Episodes | all current | minus gated |
| Pairs of real facts | never (C3) | up to `max_pairs`, only for new episodes |
| Past KL targets | n/a | never replayed; only tonight's |
| Commit | always accepted | can roll back (C1, H8) |

No arm receives free compute that I could find. All model passes go through the ledger.

---

## Section D: first real run

Likely to work by reading, unverified: tokenizer call shape, `return_dict` through `**kwargs`, `DynamicCache` round trip, `eos_ids` with both Instruct stop tokens, fp16 off on CPU.

Covered only by stubs or tiny models, so unproven on a real model:

- **fp16, autocast and GradScaler never execute in CI.** `handle.fp16` requires CUDA and no test mentions `autocast` or `GradScaler`. `test_delta.py:279` only checks dtypes on a toy net.
- **SDPA never executes.** `tests/tiny.py:114, 137` force `attn_implementation="eager"`. Real Qwen2 defaults to SDPA, which handles fully masked left-padded rows differently. Under fp16 this is the likely place for NaN, and NaN would crash `multinomial` (`handle.py:340`).
- **Real tokenizer never executes.** The tiny one is char-level, has pad distinct from EOS, one EOS, and no BOS logic in use. Instruct has two stop tokens, and pad equals a stop token.
- **No test asserts a second EOS actually stops sampling**, only that the set holds it.
- **Question parsing yield is unknown** (`queries.py:75-82`). Near-zero yield means the `no_audit_data` rollback path (H8).
- TriviaQA loading runs only on Colab. Its fields look right from memory, unverified.

**Smoke test to run on GPU before E0**, for one model, 8 prompts:
- `sequence_logprob` equals sampled `logprob` within fp16 tolerance, on left-padded mixed-length batches.
- Logits are finite.
- One `train_loop` step changes fresh under checkpointing plus scaler, and the base hash is unchanged.
- Print 20 raw completions and 20 generated question sets.
- Report how many tokens follow the answer line.

---

## Section E: five mutations that pass the current tests

1. **Teacher ignores context.** `consolidate.py:153`: `handle.format(spec.query, None)`. Stored targets become closed-book while samples stay open-book, so the student is taught nothing. Tests check only the student's prompt and shapes. KL tests use context None.
2. **Evaluation under germline.** `eval/metrics.py:227`: `handle.banks("germline")`. Every arm then scores like `B0`. `banks_seen` in `test_eval.py:43` is recorded but never asserted, and drift nests its own bank.
3. **CHD reversed at call sites.** `phases/night.py:447`: `(c.query, c.distractor, c.context)`, or `commit.py:68`. Selection and audit then use the wrong KL. Closed-form tests call `chd` directly; night tests only count.
4. **Student trains without committed.** `consolidate.py:330`: `delta.banks("fresh")`. Fresh learns ignoring earlier nights, then merges. Single-night tests and gradient tests set banks themselves.
5. **Two-hop gold points at the first hop.** `eval/probes.py:219`: `world.answer_forms(first)`. `test_an_oracle_gets_full_marks` takes gold from the probe, so it is circular. No test computes expected gold independently.

Close behind: `null_distribution` with `second = first` (`select.py:86`), making the null about 0 so everything integrates. `handle.py:348` using only `eos_id`. `night.py:249` using `all_real()`, replaying stale values.

**Add:** a test comparing stored `top_logp` with logits under the context; two-hop and alias gold computed from the facts; call-site argument checks with a recording stub.

---

## Section F: statistics summary

`stats.py` is correct as written: pairing, per-slot draws, ratio estimator, Holm step-down, AUROC ties, margins. `analyze.py` applies the rule as its header says, and its readings of H2 to H6 are defensible. Issues are M10 (Holm), H8 (no validity or parity checks), M7 (H8 vacuous). The percentile bootstrap with 5 seeds is anti-conservative; the file says so and PREREG requires it, so I report it as a known limit. Consider adding a paired t interval on seed-level differences as a sensitivity check.

---

## Checked and cleared

- KL direction: `reduced_kl`, `measure_drift`, drift and CHD all use teacher or e-side as the sampling law.
- Answer mask and log-prob alignment, under both padding sides; row-major order in `compute_loss`.
- Padding is built by hand, so the HF `padding_side` is irrelevant; position ids are explicit.
- Reduced distribution: head plus tail is a true coarsened KL; gradients finite with vanishing tail.
- Teacher isolation: targets stored under `committed` or `germline` before training; repair reuses them; fresh B is zero when they are built.
- Bank selection reaches every layer through one shared selector; germline is a true no-op; E0's plain model has no banks and is the base.
- Checkpointing: `use_reentrant=False` means `enable_input_require_grads` is not needed, fresh gets gradients with base frozen, `use_cache=False` is passed in training.
- float32 factors with GradScaler: fresh grads are float32, so unscale works; fp16 base has no grads. Autocast casts matmul inputs to fp16 but gradients reach float32 leaves.
- CPU: `handle.fp16` off, model `.float()`, in E0 too.
- Hard data and eval templates: I read both, no shared wording beyond common words; the 4-gram check is sound; no probe's gold is in its prompt.
- Probe windows, stale forms, `replayable` excluding superseded values, entrenchment pairing.
- Two-hop: chain facts always exist, `valid_from` correct, contexts contain both facts.
- E0 H0 thresholds, contradiction restricted to known facts, H1 offset comparison and verdict order match PREREG (except H3).
- Ledger: all model passes charged; evaluation and setup exempt; stage rollovers conserve the total.
- Rollback, resume and atomic checkpoint logic; seeds paired across arms; held-out split.
- World name generation looks feasible at E3 size (120 nicknames at distance 3), by estimate only.
- `max_rank` and compaction maths match the dense SVD tests.

Not checked: Hub IDs and licences (no network); `docs/literature` claims beyond A14 and T2's co-occurrence finding; `insights.md`.
