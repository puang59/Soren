# Final configuration, five seeds

Issue #33. Run on 2026-10-06.

```bash
for s in 0 1 2 3 4; do python scripts/run_arm.py --group final --name base --seed $s --timesteps 1000000; done
python scripts/collect_runs.py --group final --out experiments/final/final.csv
```

The default configuration (Tier L features, K = 6, one declaration, no shaping, no
curriculum), 1M timesteps per seed, best validation checkpoint per seed.

| seed | 0 | 1 | 2 | 3 | 4 | mean ± std |
|---|---|---|---|---|---|---|
| validation success | 0.188 | 0.203 | 0.172 | 0.188 | 0.172 | 0.184 ± 0.012 |

Success on 300 training graphs averages 0.522. Successful episodes take 4.3 actions on average.

**Read this number with the leakage control next to it** (`../leakage/README.md`): each
checkpoint is the best of about twenty evaluations on these same 128 graphs, which inflates
the validation figure by roughly 0.04. The test-set evaluation is the unbiased one.
