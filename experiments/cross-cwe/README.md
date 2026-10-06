# Cross-CWE transfer

Issue #43. Run on 2026-10-06. Validation split only; 3 seeds per arm, 500k timesteps.

```bash
python scripts/run_arm.py --group cross_cwe --name train119_val125 --seed 0 --timesteps 500000 \
    --train data/processed/by_cwe/train_CWE-119.jsonl --val data/processed/by_cwe/val_CWE-125.jsonl
```

The agent was trained on the training graphs of one CWE and scored on the validation graphs
of the same or the other CWE (805 and 218 training graphs; 101 and 27 validation graphs).

| trained on | scored on CWE-119 (101 graphs) | scored on CWE-125 (27 graphs) |
|---|---|---|
| CWE-119 | 0.158 ± 0.016 | 0.123 ± 0.017 |
| CWE-125 | 0.191 ± 0.009 | 0.148 ± 0.000 |
| random guess | 0.141 | 0.118 |

## Reading the results

**There is no transfer effect to report, in either direction, because there is barely an
in-distribution effect to transfer.** Every cell is within a few points of a random guess.
Training on the matching CWE is not better than training on the other one: on CWE-119 the
agent trained on CWE-125 scores higher (0.191) than the one trained on CWE-119 (0.158).

This is consistent with the rest of the project: the single-declaration agent learns a
generic early guess, not CWE-specific patterns.

## Caveats

- CWE-125 has only 27 validation graphs, so one graph is 3.7 points; the CWE-125 column
  cannot resolve anything.
- Each checkpoint was selected on the validation graphs it is scored on, including in the
  transfer cells, which inflates every cell by a few points (see `../leakage/README.md`).
- The test split was not used.
