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

## Follow-up: statement embeddings (Tier E) do not lift the ceiling

Issue #50. Run on 2026-10-06.

Each statement was encoded with a frozen CodeBERT (`microsoft/codebert-base`, `[CLS]` vector),
reduced to 32 dimensions with PCA fitted on the training split, and appended to the Tier L
features. The same classifier was trained on Tier L and on Tier E with three seeds each.

```bash
python scripts/07_embed_statements.py --model .tools/codebert-base
python scripts/train_classifier.py --tier E --seed 0
```

Validation split (128 graphs), mean over 3 seeds, with the range in brackets:

| features | top-1 | top-3 | top-5 | top-1 on training |
|---|---|---|---|---|
| Tier L | 0.232 [0.227, 0.234] | 0.354 [0.336, 0.367] | 0.477 [0.461, 0.500] | 0.415 |
| Tier E | 0.232 [0.227, 0.242] | 0.393 [0.359, 0.430] | 0.513 [0.500, 0.531] | 0.541 |
| heuristic score | 0.242 | 0.422 | 0.500 | |
| random ranking | 0.136 | 0.308 | 0.417 | |

**Top-1 is unchanged at 0.232.** Top-3 and top-5 move up by about 4 points, which on 128
graphs is 5 graphs and not clearly more than seed noise. Training accuracy rises from 0.42 to
0.54, so the extra dimensions mostly add capacity to memorise.

**What this suggests.** Looking harder at a statement on its own does not tell you whether it
is the vulnerable one. That fits the task: whether `memcpy(dst, src, len)` is a flaw depends
on whether `len` was checked earlier, which no per-statement feature can see. It also fits the
labels: a "vulnerable" line is any line the fix changed, which includes renames and
restructuring around the real defect.

**What was not tried.** Fine-tuning the encoder, a different pooling or model, more PCA
dimensions, or any feature that relates a statement to others (data flow, dominating checks,
a GNN over the function). The last group is where the remaining signal is most likely to be.
