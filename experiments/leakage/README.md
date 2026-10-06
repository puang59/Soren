# Leakage controls

Issue #36. Run on 2026-10-06, on the validation split.

## 1. Node-id permutation: passes

Each validation graph's node ids were randomly permuted, with lines, edges, labels and
features moving with their node. All five final checkpoints score exactly the same on the
permuted graphs as on the originals (0.188, 0.203, 0.172, 0.188, 0.172). The policy does not
read node ids or node order.

## 2. Shuffled labels: passes, and exposes a selection bias

Agents were trained with every graph's vulnerable nodes replaced by the same number of random
reachable statements (3 seeds, 500k timesteps).

| evaluated on | one declaration | three declarations |
|---|---|---|
| the shuffled validation set used to pick the checkpoint | 0.177 ± 0.019 | 0.320 ± 0.019 |
| fresh shuffles of validation (10 per model) | **0.131** | **0.301** |
| chance | 0.136 | 0.308 |

On labels they were not selected on, the shuffled-label agents score chance. Nothing leaks.

The first row is the important part. The same models score 0.177 on the validation labels
used for checkpoint selection. Picking the best of about ten evaluations on 128 graphs is
enough to lift a chance-level policy by about 0.045. **Every validation number in this
project that comes from a best-checkpoint carries that bias.**

## 3. What is left of the real-label result

The real-label agents were then scored on randomly reassigned labels, to see what their
strategy earns when the labels mean nothing:

| agent | real labels | random labels | difference |
|---|---|---|---|
| one declaration (5 seeds) | 0.184 | 0.137 | 0.047 |
| three declarations (3 seeds) | 0.404 | 0.306 | 0.098 |
| five declarations (3 seeds) | 0.471 | 0.366 | 0.105 |

With one declaration the gap to random labels (0.047) is the same size as the selection bias
(0.045). On validation data alone, the single-declaration agent cannot be told apart from
one that learned nothing. With three or five declarations the gap is about twice the bias,
so some real signal is likely there.

The test split, which no checkpoint was selected on, settles this without the bias.
