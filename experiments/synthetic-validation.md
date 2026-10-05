# PPO on synthetic graphs: signal and no-signal controls

Issue #12, the gate for milestone M1. Run on 2026-10-05.

## Question

Does the agent learn to navigate when the features carry a signal, and does it fall back to
chance when they do not? The second half matters as much as the first: if the agent still
succeeded without a signal, something other than features (node order, ids, position) would
be leaking the label.

## Setup

```bash
python scripts/exp_synthetic_validation.py --timesteps 500000 --seeds 0 1 2 --out runs/exp_synthetic
```

- Graphs: synthetic CFGs of 10 to 40 target nodes (mean 25.0), one vulnerable statement each.
  2,000 for training, 300 for checkpoint selection, 500 held out for the numbers below.
- The three conditions use identical graph structures and labels. Only the marking differs:

  | condition | P(dangerous call \| vulnerable) | P(dangerous call \| other statement) |
  |---|---|---|
  | clean | 1.0 | 0.0 |
  | noisy | 0.9 | 0.1 |
  | none | 0.1 | 0.1 |

- Agent: MaskablePPO with the defaults in `configs/ppo.yaml` and `configs/env.yaml` (Tier L
  features, K = 6, one declaration per episode, no shaping), 500k timesteps, 3 seeds. The best
  validation checkpoint is evaluated deterministically.
- Baselines: DFS, BFS and RandomWalk under Protocol A. They succeed on merely *reaching* the
  vulnerable node, so DFS and BFS score 100% by construction. The agent must reach it and
  declare it.
- Feature ceiling: the success rate of the best single guess given every node's features,
  i.e. the most any one-declaration policy can score.

## Results

Held-out graphs, mean ± standard deviation over 3 seeds for PPO.

| condition | method | success | nodes inspected | actions | return |
|---|---|---|---|---|---|
| clean | **PPO** | **0.981 ± 0.002** | **11.84 ± 0.25** | 15.33 ± 0.56 | 0.812 ± 0.007 |
| clean | feature ceiling | 1.000 | | | |
| noisy | **PPO** | **0.459 ± 0.007** | 7.80 ± 0.12 | 9.28 ± 0.34 | 0.091 ± 0.015 |
| noisy | feature ceiling | 0.468 | | | |
| none | **PPO** | **0.060 ± 0.007** | 3.17 ± 0.13 | 3.45 ± 0.21 | -0.437 ± 0.009 |
| none | feature ceiling (chance) | 0.075 | | | |
| all | DFS (oracle stop) | 1.000 | 13.72 | 18.44 | 0.826 |
| all | BFS (oracle stop) | 1.000 | 13.92 | 13.92 | 0.871 |
| all | RandomWalk (oracle stop) | 0.600 | 12.13 | 62.52 | -0.700 |

## Reading the results

**Clean signal: the gate is met.** The agent localizes the vulnerable statement in 98.1% of
held-out graphs and inspects 11.8 nodes on average against 13.7 for DFS (14% fewer), with 15.3
actions against 18.4. It does this without the oracle stop the baselines enjoy. Its return is
marginally below DFS's (0.812 vs 0.826) because DFS's 100% success is given, not earned.

**Noisy signal: the agent is at the ceiling, and the ceiling is low.** 0.459 against a best
possible 0.468. With about 12 candidate statements per graph there is usually at least one
decoy, so a single guess among the flagged statements is right less than half the time. The
agent declares the first flagged statement it meets (7.8 nodes inspected), which is as good as
any other flagged one. The limit here is the one-declaration budget, not the policy.

**No signal: performance collapses to chance.** 0.060 against a chance level of 0.075. The
agent learns that searching cannot pay and guesses after about 3 nodes. Nothing leaks the
label through node order, ids or position.

## Consequences for later work

- **Top-k matters.** On noisy data a single declaration caps success well below what the
  features support across several guesses. The declare-budget ablation (#35) should be read
  with this in mind, and real BigVul features will be noisier than the `noisy` condition.
- **Blind guessing is the rational fallback.** Under the default rewards a hopeless search
  ends in an early wrong declaration, not a timeout. That is the intended reading of the
  reward scale, but it means "wrong declare" rates on real data partly measure how
  uninformative the features are.
- **Compare on nodes inspected.** DFS and BFS returns include a success they did not have to
  decide on. Protocol B (#29) is what makes the return comparison fair.

## Not tested

- Sensitivity to graph size beyond 40 nodes, and to hyperparameters (#32).
- Statistical tests: 3 seeds and one held-out set give a spread, not a confidence interval.
