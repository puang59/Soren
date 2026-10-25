# Test-set evaluation with the from-scratch PPO

The same evaluation as [`experiments/test`](../test/README.md), with every agent retrained by
Soren's own PPO (`soren/agents/ppo.py`) instead of sb3-contrib's `MaskablePPO`. The
configuration, data split, seeds, step budgets and Protocol B thresholds are unchanged; only
the implementation of the learner, the network and the optimiser differ.

```bash
for s in 0 1 2 3 4; do
  python scripts/run_arm.py --group final --name base --seed $s --timesteps 1000000
done
for n in 3 5; do for s in 0 1 2; do
  python scripts/run_arm.py --group ablations --name declares_$n --seed $s --timesteps 500000 \
      --set env.max_declares=$n
done; done
for s in 0 1 2; do
  python scripts/train_classifier.py --seed $s --out runs/v2/classifier/tier_L_seed$s.npz
done
python scripts/final_evaluation.py --allow-test --runs runs/v2 --out experiments/test-v2
```

The test split had already been evaluated once, for `experiments/test`. Nothing here was
tuned on it: the runs reuse the frozen configuration and select checkpoints on validation.

## Library and from-scratch implementation side by side

| | sb3-contrib `MaskablePPO` | Soren's PPO |
|---|---|---|
| validation success, one declaration (5 seeds) | 0.184 | 0.189 |
| **test success, one declaration (5 seeds)** | **0.211 ± 0.030** | **0.216 ± 0.021** |
| 95% interval over graphs | [0.153, 0.274] | [0.157, 0.282] |
| test success, three declarations (3 seeds) | 0.407 | 0.388 |
| test success, five declarations (3 seeds) | 0.480 | 0.520 |
| the agent's strategy on random labels, one declaration | 0.122 | 0.115 |
| actions per successful episode | 4.3 | 4.2 |
| classifier top-1 (3 seeds; PyTorch, then `soren.nn`) | 0.194 | 0.189 |

Every difference is within the seed-to-seed spread of either implementation. The conclusions
of the report do not change: the agent is above chance, below the structured baselines by 4 to
6 points without a significant difference (McNemar p = 0.18 to 0.42), and significantly above only the
random-order baseline (p = 0.04, uncorrected).

| method | success | 95% interval | p vs agent |
|---|---|---|---|
| DFS + heuristic stop | 0.276 | [0.197, 0.354] | 0.18 |
| Line order + heuristic stop | 0.276 | [0.197, 0.354] | 0.18 |
| Heuristic-first + heuristic stop | 0.268 | [0.197, 0.346] | 0.25 |
| BFS + heuristic stop | 0.252 | [0.181, 0.331] | 0.42 |
| **PPO agent (5 seeds)** | **0.216** | [0.157, 0.282] | |
| Random walk + heuristic stop | 0.172 | [0.110, 0.239] | 0.36 |
| Random order + heuristic stop | 0.157 | [0.112, 0.208] | 0.04 |

The baselines do not involve the learner, so their rows are identical to `experiments/test`.

`python scripts/final_evaluation.py --allow-test --runs runs/v2 --out <dir> --explain` prints
how the success rate, its interval and each test are computed.
