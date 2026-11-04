# Toy 1: one seed of E5's question

A check, not evidence. Dev seed 41, one night, 8 people, 200k tokens at lr 3e-4, CPU on GitHub Actions. 60 real questions, 16 to 40 probes per kind. Raw: `results/toy1/summary.txt`.

| Arm | New facts | Reverse | Alias | Two-hop | Real knowledge | Look-alike false recall | Generic drift |
|---|---|---|---|---|---|---|---|
| `B0_frozen` | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| `B4q`, near zero | 0.00 | 0.00 | 0.00 | 0.00 | 0.92 | 0.00 | 0.00 |
| `B4_replay_kl` | 0.98 | 1.00 | 0.23 | 0.00 | 0.35 | 0.93 | 0.06 |
| `B4q` | 0.95 | 1.00 | 0.25 | 0.00 | 0.67 | 0.98 | 0.24 |
| `B4qd` | 0.93 | 0.81 | 0.80 | 0.50 | 0.67 | 0.90 | 0.25 |

- First time a question-frame arm has reached useful learning: it keeps its gap over `B4` (0.67 vs. 0.35). Per-fact budget here is about 7,700 tokens, higher than E1b's, and the world is tiny. That may not scale.
- Written-out composed facts reach held-out templates (alias 0.80, two-hop 0.50) at no visible cost to real knowledge. This is oracle data; the deployable form is unbuilt. Note: alias and two-hop use small item counts.
- **False recall on look-alikes is nearly total in all learning arms (0.90 to 0.98): none of this fixes it.** The model answers about strangers with taught values.
- The near-zero control lost 8 points with almost no change, confirming answers on these questions are fragile.
- Question-frame anchors cost more generic drift (0.24 vs. 0.06), so this isn't free.

## Next: E5s

Smallest test that's still somewhat reliable (user's request, 2026-11-04): 3 new seeds (3001 to 3003), 16 people, 2 nights, 250k tokens per night at lr 3e-4, planned for a Colab T4 (it ran on one RTX 5090; see `E5s.md`). Arms: `B0_frozen`, `B4_replay_kl`, `B4q`, `B4qd`. Stated before any data, from PREREG A5.4 and A5.5, on the final night:
- H10: `B4q` minus `B4_replay_kl` on `ret_real` at least 10 points, all 3 seeds positive, with `acq_direct` at least 0.7 in both.
- H14: `B4qd` lift on `acq_alias` and `acq_twohop` at least 0.25 over `B4q`, with `ret_real` within 10 points.
Exploratory whatever it shows. Seeds 1001 to 1005 stay untouched.
