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
- [in progress] IMPL-A core: delta, handle, memory, signals, ledger, config, arms, phases, tests.
- [next] IMPL-B: worlds, eval, stats, experiments, configs, CI, packaging, README.
- [next] Independent review by a fresh agent, fix findings, push, read CI.
- [next] Colab: open connection, E0 (no training, picks model, validates signal), then E1 onward per PREREG.

## Blockers
- Kaggle MCP needs the user's sign-in via /mcp; not required while Colab works.
- Colab connection untested until code is ready. It may need the user's browser. If unavailable, code still ships and experiments wait.
- Private repo: don't place a GitHub token in a notebook. Send the code to Colab as a tarball in a cell.
