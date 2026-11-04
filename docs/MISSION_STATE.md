# Mission state

Resume point for the coordinator. Update at every phase boundary.

## Rules (from the user)
- Long continuous mission. The user may be away and has authorized autonomous decisions, as long as it's safe.
- No local tests or model runs, and no CPU toy batches. Write code. Real runs on Colab via MCP. CI on GitHub Actions is unit tests only.
- Coordinator plus threads, no swarm. Opus 5.5 for implementation and thinking; Sonnet 5.5 for execution and search; Haiku 5.5 for retrieval. All medium effort. Agents use how-to-think when needed.
- Commit and push to `feat/sleep-framework` in meta-lucky-builds/sleep. Original `main` stays untouched.
- After a usage limit, resume automatically.
- Keep context and usage lean without lowering quality. Adapt: when an approach yields nothing, form new hypotheses instead of polishing a local optimum. A pivot gets a dated entry below, stating what failed and why. Changes to confirmatory hypotheses are appended to `PREREG.md`, not edited in place.

## Phase log
- [done] Literature T1, T2, T3 (`docs/literature/`).
- [done] Theory draft 1, pre-registration, design spec (`docs/THEORY.md`, `PREREG.md`, `DESIGN.md`).
- [done] IMPL-A core, IMPL-B benchmark, eval, runner, CI.
- [done] Independent review (`docs/REVIEW_1.md`); all CRITICAL and HIGH fixed; PREREG Amendment 1. CI: 398 passed.
- GPU run: `scripts/gpu_run.sh` with tag `gpu-ready`; user rents and runs it, and uploads results.tar.gz.
- [done] E0 on 4 models, E0b, E1 (one seed, 3 nights). See docs/results/. Selection abandoned; E1 kill rule fired; operating point not bracketed.
- [next] E1b per PREREG Amendment 3. Then E3.

## Blockers



