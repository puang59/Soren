# Protocol B thresholds and first comparison

Issue #29. Run on 2026-10-06, on the validation split (128 graphs).

Under Protocol B a baseline walks in its own order and declares at the first node whose score
reaches a threshold. The thresholds in `configs/eval.yaml` are the ones with the highest
validation success rate per baseline (ties: fewer nodes inspected). The full curves are
`threshold_curve_heuristic.csv` and `threshold_curve_classifier.csv`.

```bash
python scripts/tune_thresholds.py --graphs data/processed/graphs_val.jsonl --write-config
python scripts/tune_thresholds.py --graphs data/processed/graphs_val.jsonl \
    --classifier runs/classifier/model.pt --write-config
python scripts/evaluate.py --graphs data/processed/graphs_val.jsonl --split val \
    --methods dfs bfs line_order random_order heuristic_first random_walk --protocol-b \
    --checkpoints runs/bigvul_first/tier_l_seed0/best_model.zip --out runs/protocol_b/eval_val.parquet
```

## Success when every method must declare

| method | heuristic score | classifier score |
|---|---|---|
| DFS | 0.211 | 0.125 |
| LineOrder | 0.211 | 0.148 |
| RandomOrder | 0.206 | 0.164 |
| BFS | 0.203 | 0.133 |
| HeuristicFirst | 0.188 | 0.141 |
| RandomWalk | 0.159 | 0.125 |
| **PPO, Tier L, untuned** | **0.188** | |
| uniform guess | 0.136 | |

## Reading the results

- **Once the baselines have to declare, they fall from 100% to about 20%.** The oracle stop
  was doing all the work under Protocol A.
- **The untuned agent is level with them**: 0.188 against 0.19 to 0.21, with the best return
  of any method (-0.242) because it guesses early and cheaply.
- **Traversal order hardly matters.** DFS, BFS, line order and a random order are within one
  graph of each other, which is what weak per-node scores imply: whichever node first clears
  the threshold is rarely the right one.
- **The classifier makes a worse stopping rule than the heuristic** even though it ranks about
  as well, because its probabilities are poorly calibrated.

## Caveats

- These thresholds were chosen on the same 128 graphs the table reports, so the baseline
  numbers are optimistic. The test split has not been touched.
- All tuned heuristic thresholds sit at the low end (0.05 to 0.10): the best rule is close to
  "declare at the first statement with any flag set".
- The classifier thresholds depend on `runs/classifier/model.pt`, which is not in the
  repository; retrain it with `scripts/train_classifier.py` before reusing them.
