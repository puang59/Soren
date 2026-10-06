# How much do the features say? Node classifier on BigVul

Issue #30. Run on 2026-10-06.

## Question

The first PPO runs landed near chance (see `bigvul-first-runs.md`). Is that the policy, or do
the features simply not identify the vulnerable statement? A supervised classifier that sees
every node of a function at once gives an upper reference for what the features support.

## Setup

```bash
python scripts/train_classifier.py --out runs/classifier/model.pt
```

A two-layer MLP (64 units) on the 30 Tier L features of each statement, trained on the 1,023
training graphs with class weighting, keeping the epoch with the best validation top-1.
Statements of each validation function are ranked by predicted probability.

## Results on the validation split (128 graphs)

| ranking | top-1 | top-3 | top-5 |
|---|---|---|---|
| classifier, Tier L | 0.234 | 0.336 | 0.469 |
| classifier, Tier S | 0.164 | 0.289 | 0.391 |
| heuristic score (hand-set weights) | 0.242 | 0.422 | 0.500 |
| random ranking (expected) | 0.136 | 0.308 | 0.417 |

The Tier L classifier scores 0.485 / 0.664 / 0.737 on its own training graphs.

## Reading the results

**The features carry little signal.** With every node visible, the best ranking puts a
vulnerable statement first in 24% of functions, against 14% at random. At top-3 and top-5 the
gain over random is 3 to 11 points.

**This bounds the agent.** PPO reached 0.188 with one declaration and a one-hop view. The
ceiling for any single declaration from these features is about 0.24, so most of the gap to a
useful localizer is in the features, not in the navigation policy.

**The lexical flags do help.** Tier L beats Tier S by 7 points at top-1 and the hand-weighted
heuristic matches the trained classifier, so the flags point the right way; they are just weak.

**The classifier overfits** (0.49 on training, 0.23 on validation), like the agent. 1,023
functions are few for this.

## Consequence

More RL tuning on these features can gain a few points at most. The lever is the
representation: statement text embeddings (Tier E, #50), data-flow features such as whether an
index or length is checked before use, or a GNN over the whole function (#49).
