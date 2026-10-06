# Test-set evaluation

Issue #41. Run once, on 2026-10-06, from the commit tagged `v0.1-frozen`.

```bash
python scripts/final_evaluation.py --allow-test
```

127 test graphs (100 CWE-119, 27 CWE-125). No checkpoint, threshold or hyperparameter was
chosen on this split, so these numbers do not carry the selection bias that the validation
numbers do. Per-episode results are in `results.parquet`; every table here derives from it.

## Headline: one declaration per function

Share of test functions localized, with 95% bootstrap intervals over graphs. p-values compare
each method with the agent on the same graphs (exact McNemar test).

| method | success | 95% interval | p vs agent |
|---|---|---|---|
| DFS + heuristic stop | 0.276 | [0.197, 0.354] | 0.14 |
| Line order + heuristic stop | 0.276 | [0.197, 0.354] | 0.14 |
| Heuristic-first + heuristic stop | 0.268 | [0.197, 0.346] | 0.17 |
| BFS + heuristic stop | 0.252 | [0.181, 0.331] | 0.33 |
| **PPO agent (5 seeds)** | **0.211** | [0.153, 0.274] | |
| Random walk + heuristic stop | 0.172 | [0.110, 0.239] | 0.52 |
| Random order + heuristic stop | 0.157 | [0.112, 0.208] | 0.04 |
| the agent's own strategy on random labels | 0.122 | | |
| random guess | 0.125 | | |

Across the five seeds the agent's success is 0.211 ± 0.031.

Under Protocol A (oracle stop) DFS, BFS, line order, heuristic-first and random order reach a
vulnerable node in 99 to 100% of functions after inspecting 17.6 to 18.9 nodes; the random
walk reaches one in 65%.

## More than one declaration

| | 1 | 3 | 5 |
|---|---|---|---|
| PPO agent | 0.211 | 0.407 | 0.480 |
| the agent's strategy on random labels | 0.122 | 0.292 | 0.345 |
| heuristic score, ranking every statement | 0.236 | 0.465 | 0.559 |
| classifier, ranking every statement (3 seeds) | 0.194 | 0.388 | 0.480 |
| random ranking | 0.125 | 0.288 | 0.394 |

The three- and five-declaration agents come from the ablation runs (3 seeds each).

## Breakdowns (agent against DFS + heuristic stop)

| | n | PPO agent | DFS + heuristic |
|---|---|---|---|
| up to 20 nodes | 45 | 0.36 | 0.38 |
| 21 to 50 nodes | 38 | 0.18 | 0.24 |
| 51 to 100 nodes | 28 | 0.13 | 0.25 |
| over 100 nodes | 16 | 0.00 | 0.12 |
| one vulnerable statement | 75 | 0.17 | 0.21 |
| several vulnerable statements | 52 | 0.27 | 0.37 |
| CWE-119 | 100 | 0.21 | 0.25 |
| CWE-125 | 27 | 0.21 | 0.37 |

## What the test set says

1. **The agent beats chance.** 0.211 against 0.125 for a random guess and 0.122 for its own
   strategy on random labels. The interval's lower end (0.153) is above both. On validation
   this could not be established, because checkpoint selection added about as much.
2. **The agent does not beat the simple baselines.** Walking the function in DFS or line order
   and declaring at the first statement with a suspicious flag scores 0.276. The difference
   from the agent (0.065) is not significant at this sample size (p = 0.14), but nothing here
   supports a claim that learned navigation is better.
3. **Order matters a little, in favour of structure.** Structured orders (DFS, BFS, line
   order) score 0.25 to 0.28; a random order scores 0.16. Reading code in an order that
   follows the program helps more than the agent's learned policy does.
4. **Rankings that see the whole function are as good or better.** With five guesses the
   hand-weighted heuristic reaches 0.559, against 0.480 for the agent. The agent's one-hop
   view costs it, and it gains nothing from navigating.
5. **Everything degrades with size.** The agent localizes 36% of functions of up to 20 nodes
   and none of the 16 functions over 100 nodes.

## Caveats

- 127 graphs: intervals are about ±0.07 wide, and differences under 0.10 are not resolved.
- The Protocol B thresholds were tuned on validation with the heuristic scorer; no
  multi-declaration baseline with a stopping rule was built, so the 3- and 5-declaration rows
  compare the agent with rankings, not with traversals.
- p-values are not corrected for the six comparisons.
- Labels are the lines a fix changed, a noisy proxy for the vulnerable statement.
