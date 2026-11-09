# E5s run 1: which file to read

`analysis.txt` is the repository's analysis tool run on this output. It scores only seeds marked confirmatory, and these
seeds are exploratory, so it reports "only 0 seeds; 5 are needed" and no verdicts. That does not mean the run failed.

The numbers in [`docs/results/E5s.md`](../../docs/results/E5s.md) come from [`per_seed.txt`](per_seed.txt), which lists
each seed. For example, B4q minus B4_replay_kl on real knowledge is +0.26, +0.23 and +0.30 across the three seeds.
All E5s results are exploratory.
