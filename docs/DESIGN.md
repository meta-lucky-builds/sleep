# Code design (implementation spec)

Read `THEORY.md` and `PREREG.md` first. Those define *why*; this defines *what*.

Package: `src/llmsleep`. Python 3.10+. Dependencies: `torch`, `transformers`, `numpy`; `datasets` only in the capability-probe
loader, imported lazily. Tests with `pytest`, using tiny randomly initialised models built from a config, never downloads.

## Invariants the code must enforce (tested)

1. **Base weights never change.** `requires_grad=False` everywhere but fresh factors. A test hashes base parameters before and after a full night.
2. **Teachers are frozen for the night.** Banks are immutable during consolidation; training cannot alter them.
3. **Generated text never becomes a fact.** `Episode.provenance` is `real`; the store rejects anything else on the fact path.
4. **Budget is matched.** All arms draw from `ComputeLedger`; exceeding it stops the phase, doesn't warn. Evaluation is exempt and tracked separately.
5. **Same code path for every arm.** Arms are config values only. No per-arm subclasses or `if arm == "..."` branches outside `arms.py`.
6. **Eval templates are disjoint from training templates.** Enforced at world construction, and tested.
7. **Runs are resumable and deterministic.** A night is an atomic checkpoint. Colab sessions die. Seeds derive from `(world_seed, night, purpose)`.

## Modules

### `delta.py`: low-rank weight banks
`DeltaLinear(nn.Module)` wraps an `nn.Linear` (and adapts HF `Conv1D` for GPT-2-family). Output is `base(x) + sum over active banks of (x @ V) * S @ U^T`, in each bank's factor form. Banks:
- `committed`: frozen factors `(U, S, V)` from all prior nights.
- `fresh`: trainable `(A, B)`, B zero-init, standard LoRA scale.
- arbitrary named frozen snapshots, e.g. `pre_compaction`.

`DeltaModel` attaches to named modules by regex, excluding `lm_head` and embeddings by default. Context manager `model.banks("germline" | "committed" | "night" | [names])` selects active banks **so a single model instance is teacher and student**, avoiding copies. Key methods:
- `merge_fresh_into_committed()`: exact low-rank addition.
- `compact(threshold, max_rank)`: QR-based truncated SVD of the combined factors, soft-thresholding singular values. Must not form the full matrix. Test: result equals `torch.linalg.svd` of the dense sum on small layers.
- `snapshot()/restore()` for rollback; `state_dict` for adapters only.
- stats: per-layer norm, rank, total parameter count.

### `handle.py`: model access
`ModelHandle(model, tokenizer, device, ledger)`. `format(query, context=None) -> prompt` from a single `PromptFormat`: the same wrapper with context slot used by open, distractor and closed conditions, so the only difference is content. `sequence_logprob(prompts, continuations)`, `sample(prompts, n, max_new_tokens, temperature)`, `answer_token_mask` (positions of continuation only). All charged to ledger, batched, with left-padding handled correctly, right-padding for training.

### `memory.py`: fast store
`Episode(id, text, night, provenance, entities, supersedes, meta, status)`, statuses `active | retired`. `EpisodicStore`: `add`, `retire`, `active()`, `all_real()`, `neighbours(entity)`, `pairs_sharing_entity()`, JSON persist. Retriever protocol: `BM25Retriever` (pure Python), `OracleRetriever` (benchmark only). Recency rule: with a supersession chain, return the newest and mark it so in context. Distractor sampler: draws episodes sharing no entity with the query's, matching length to within a tolerance.

### `signals.py`
`chd(handle, queries, ctx_e, ctx_d, k, max_new_tokens) -> tensor` per query, Monte Carlo sequence KL per the theory. Also `raw_kl_nocontext` (control), and per-token normalised variants. Returns samples too for reuse so budget isn't paid twice. `SignalCache` keyed on `(query, ctx, banks_version)`.

### `phases/`
- `wake.py`: `observe(episode)`: write gate by CHD, optional bypass.
- `queries.py`: `TemplateQueryGen` (from world), `ModelQueryGen` (model given the episode, parse questions, de-duplicate), `PairQueryGen`.
- `select.py`: partition by CHD quantiles and lexical overlap into `integrate`, `anchor`, `discard`. Thresholds relative to a null distribution from distractor-vs-distractor CHD, not absolute numbers.
- `consolidate.py`: training loop on L = hard CE + teacher KL + anchor KL, over answer tokens. Teachers precomputed (samples and logprobs) at night start; targets stored, so banks can be released. Options: gradient accumulation, grad clip, fp16 autocast, gradient checkpointing, cosine lr.
- `compact.py`: merge, compact, repair steps.
- `commit.py`: audit, accept/rollback, returns a `Decision` with reasons.
- `night.py`: `run_night(state, cfg) -> NightReport`, orchestrates; `SleepPressure` computes trigger value.

### `arms.py`
`ArmConfig` dataclass and registry for every arm in PREREG. The only place arms differ.

### `worlds/`
`SyntheticWorld(seed, n_entities, nights, ...)`. Fictional, name-checked entities. Attributes, and linking relations for two-hop. Per-night `episodes` with train templates; supersession events at a set rate; look-alike never-taught entities; held-out templates (direct, paraphrase, reverse, alias, two-hop, preservation, supersession). Deterministic. Episode text never contains eval templates. Gold answers include aliases.

### `eval/`
- `probes.py`: builds frozen probe sets at world creation.
- `metrics.py`: each PREREG metric; free generation with `normalize_answer`. Batch efficient; subsampling is seeded and stored.
- `capability.py`: real QA (TriviaQA, license checked at download, plus TOFU `world_facts`) filtered to base-correct; head/tail split by sampled consistency. Disk-cached.
- `drift.py`: fixed generic prompts and KL to germline bank.
- `entrench.py`: contradiction-following probes.

### `stats.py`
Paired, seed-level and fact-clustered bootstrap; Holm. Pure numpy.

### `experiments/`
`run.py --config cfg.json`: for each night, skip if finished; append one line per night to `results.jsonl`, with config hash, git SHA, ledger, metrics, decision. Atomic checkpoint writes. `analyze.py` reads results, applies the PREREG decision rule, and prints tables. Configs in `configs/` for E0–E6. E0 has its own script since it trains nothing.

### Compute target
Colab T4: fp16, gradient checkpointing, LoRA rank 32 on attention and MLP linears. The code must run on CPU for tests, with no change other than config.

## Tests
`tests/` builds a tiny Qwen2-config or GPT-2-config model (2 layers, small dims, random weights, small char-level or trained tokenizer built in-process, no network). Tests assert the invariants above and the math (low-rank equality, KL estimator on a known categorical, ledger stops, rollback restores exact outputs, resume yields identical results with fixed seed). Tests never assert learning quality. CI: GitHub Actions on push, unit tests only. **No local execution.**

## Non-goals for now
Linear probes for storage vs. access (I-71) and savings measures come after E3. Reverse-KL on-policy is a config option only if it costs nothing to add.
