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
- [in progress] Real-model smoke via GitHub Actions. Tag `e0-smoke-N` triggers it, `e0-run-N` runs E0, models in `configs/e0_models.txt`. Results as run artifacts.
- [next] E0 proper, then decide model and whether the control layer survives.
- [next] E1 onward need a GPU: Colab or Kaggle.

## Blockers
- Kaggle MCP needs the user's sign-in via /mcp; not required while Colab works.
- Colab connection untested until code is ready. It may need the user's browser. If unavailable, code still ships and experiments wait.
- Private repo: don't place a GitHub token in a notebook. Send the code to Colab as a tarball in a cell.
