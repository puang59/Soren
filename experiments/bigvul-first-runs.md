# First PPO runs on BigVul graphs

Issue #28, the gate for milestone M2. Run on 2026-10-06.

## Question

Does the pipeline run end to end on real data, and where does an untuned agent land?

## Setup

```bash
python scripts/train.py --train data/processed/graphs_train.jsonl \
    --val data/processed/graphs_val.jsonl --name tier_l_seed0 --timesteps 1000000
```

- Data: 1,023 training and 128 validation graphs (CWE-119 and CWE-125, split by commit).
- Agent: MaskablePPO with the defaults in `configs/ppo.yaml` and `configs/env.yaml`: K = 6, one
  declaration per episode, no shaping, no curriculum. One seed, 1M timesteps, once with Tier L
  features and once with Tier S. The best validation checkpoint is evaluated deterministically.
- Baselines: Protocol A (oracle stop), so they succeed by reaching a vulnerable node.
  Stochastic ones use 3 seeds.

## Results on the validation split

| method | success | nodes inspected (successes) | return |
|---|---|---|---|
| **PPO, Tier L** | **0.188** | 3.4 | -0.242 |
| **PPO, Tier S** | **0.172** | 3.5 | -0.272 |
| uniform guess over statements | 0.136 | | |
| always declare the first statement | 0.117 | | |
| RandomOrder (oracle stop) | 1.000 | 15.6 | 0.854 |
| HeuristicFirst (oracle stop) | 1.000 | 16.4 | 0.846 |
| DFS (oracle stop) | 1.000 | 17.9 | 0.762 |
| LineOrder (oracle stop) | 1.000 | 17.9 | 0.831 |
| BFS (oracle stop) | 1.000 | 19.3 | 0.817 |
| RandomWalk (oracle stop) | 0.677 | 8.4 | -0.846 |

On 300 training graphs the same checkpoints score 0.473 (Tier L) and 0.360 (Tier S).

## Reading the results

**The pipeline works end to end.** Real graphs train, validate and evaluate through the same
code paths as the synthetic ones.

**The untuned agent is barely above chance.** 0.188 against 0.136 for a uniform guess. It
declares after about three actions and never searches: it visits a vulnerable node at any
point in only 26% of episodes. This is the behaviour the synthetic no-signal condition
produced. Under the default rewards, a quick guess (about -0.5 when wrong) beats a long
uncertain search.

**It memorises training graphs.** Training success is 0.47 against 0.19 on validation for
Tier L. With 1,023 graphs and a 248-value observation, the policy can recognise functions it
has seen.

**Tier L helps a little, within noise.** 0.188 against 0.172 on 128 graphs and one seed is a
difference of two graphs.

**The baselines' 100% is not comparable.** Under the oracle stop they cannot be wrong. What
their rows do show is search cost: an uninformed search inspects 16 to 19 nodes before it
reaches a vulnerable one, about 40% of a function. Ordering by the heuristic score barely
improves on that (16.4 against 15.6 for a random order), which suggests the lexical flags
carry little signal about *which* statement is vulnerable.

## What this does not show

- That learned navigation cannot work here. One seed, default hyperparameters, one
  declaration, no curriculum and no shaping is the weakest configuration.
- How the agent compares with baselines that must also declare. That needs Protocol B
  thresholds tuned on this validation split (#29).

## Follow-ups

1. **Measure the signal in the features directly** with the supervised node classifier (#30).
   If a classifier that sees every node cannot rank the vulnerable statement highly, no
   navigation policy will, and the features need work before more RL tuning.
2. **Allow several declarations** (declare budget 3 and 5, #35). Functions have a median of 2
   and a mean of 6 vulnerable statements, and synthetic runs showed one declaration caps
   success under noise.
3. **Tune the reward** (#32): a larger wrong-declaration penalty or a smaller step cost should
   stop the immediate guess.
4. **Counter the memorisation**: smaller network, K = 2 to shrink the observation, or pooling
   the memory-safety CWEs for more data.
5. **Tune Protocol B thresholds** (#29) so the agent has baselines it can fairly be compared to.
