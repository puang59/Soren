# Ablations

Issue #35. Run on 2026-10-06. Validation split (128 graphs), 3 seeds per arm, 500k timesteps,
best validation checkpoint. Full table: `ablations.csv`.

```bash
python scripts/run_arm.py --group ablations --name declares_3 --seed 0 --timesteps 500000 --set env.max_declares=3
python scripts/collect_runs.py --group ablations --out experiments/ablations/ablations.csv
```

| arm | change from the default | val success (mean ± std) | train success |
|---|---|---|---|
| tier_l | none (the default) | 0.169 ± 0.026 | 0.491 |
| tier_s | structural features only | 0.148 ± 0.006 | 0.370 |
| tier_e | plus statement embeddings | 0.190 ± 0.013 | 0.520 |
| no_rel_line | relative line position zeroed | 0.190 ± 0.021 | 0.500 |
| no_backtrack | BACKTRACK disabled | 0.185 ± 0.004 | 0.493 |
| no_revisit_penalty | revisit penalty 0 | 0.182 ± 0.022 | 0.511 |
| shaping | potential-based shaping on | 0.159 ± 0.013 | 0.434 |
| declares_3 | declare budget 3 | 0.404 ± 0.021 | 0.653 |
| declares_5 | declare budget 5 | 0.471 ± 0.004 | 0.676 |

Chance on this split is 0.136 for one guess, 0.308 for three and 0.417 for five.

## One line per ablation

- **Tier S vs Tier L:** lexical flags add about 2 points (0.148 to 0.169), within the spread of
  the arms. They help the classifier more than they help the agent.
- **Tier E:** no gain over Tier L beyond noise, as the classifier predicted.
- **Relative line position:** removing it changes nothing, so the agent is not exploiting a
  positional shortcut.
- **Backtrack:** disabling it changes nothing, because the agent rarely moves far enough to
  need it. Successful episodes last about four actions.
- **Revisit penalty:** no effect, for the same reason.
- **Shaping:** slightly worse, not better. Shaping pulls the agent toward labelled nodes during
  training, which does not transfer when the features cannot identify them.
- **Declare budget:** the only arm that moves the number. With three declarations the agent
  succeeds on 40% of functions and with five on 47%, about 10 points above what the same
  strategies score on randomly reassigned labels (0.306 and 0.366, see the leakage note).

## Caveats

All arms share the validation-selection bias described in the leakage note, so compare arms
with each other and with chance plus about 0.04, not with raw chance. Three seeds and 128
graphs do not resolve differences of 2 or 3 points. The arms ran for 500k timesteps and the
final configuration for 1M, which is why `tier_l` here (0.169) is below the final 0.184.
