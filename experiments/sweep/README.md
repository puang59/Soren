# Reward and hyperparameter sweep

Issue #32. Run on 2026-10-06. Validation split only (128 graphs); the test split is untouched.

```bash
python scripts/run_arm.py --group sweep --timesteps 500000 --seed 0 --name wrong1.0 --set reward.wrong=-1.0
python scripts/collect_runs.py --group sweep --out experiments/sweep/sweep.csv
```

Each arm is the default configuration with the listed change, trained for 500k timesteps with
one seed; the best validation checkpoint is scored. Full table: `sweep.csv`.

| arm | change | val success | train success | return |
|---|---|---|---|---|
| base | none | 0.203 | 0.527 | -0.253 |
| step0.005_wrong1.0 | step -0.005, wrong -1.0 | 0.203 | 0.517 | -0.661 |
| net128 | network 128x128 | 0.188 | 0.493 | -0.246 |
| wrong1.0 | wrong declaration -1.0 | 0.188 | 0.490 | -0.671 |
| step0.005 | step cost -0.005 | 0.180 | 0.543 | -0.258 |
| wrong0.25 | wrong declaration -0.25 | 0.180 | 0.523 | -0.050 |
| ent0.03 | entropy bonus 0.03 | 0.172 | 0.493 | -0.270 |
| wrong1.0_ent0.03 | wrong -1.0, entropy 0.03 | 0.156 | 0.453 | -0.714 |

## Reading the results

**Nothing separates the arms.** Validation success runs from 0.156 to 0.203. On 128 graphs one
standard error of a 0.19 success rate is about 0.035, so the whole range is within noise of a
single value.

**The reward was not the obstacle.** Doubling the wrong-declaration penalty or halving the
step cost does not make the agent search longer: successful episodes still end after 4 to 6
actions in every arm. This matches the classifier result in `../feature-signal.md`: the agent
guesses early because the features give it nothing to search for, not because guessing is
too cheap.

**Every arm memorises.** Training success is 0.45 to 0.54 against 0.16 to 0.20 on validation,
and a smaller network does not close the gap.

## Selected configuration

The defaults in `configs/ppo.yaml` and `configs/env.yaml`, unchanged. No arm improved on them
and there is no basis in this sweep to prefer another. The curriculum was not needed:
training does not stall on large graphs, it plateaus on all sizes.

## Limits

One seed per arm, so the ranking of arms is not meaningful. Eight arms instead of the plan's
full grid of 54, since the first eight show no effect to map. Rewards differ between arms,
so the return column is only comparable within the same reward settings.
