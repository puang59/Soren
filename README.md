# Soren

[![CI](https://github.com/puang59/Soren/actions/workflows/ci.yml/badge.svg)](https://github.com/puang59/Soren/actions/workflows/ci.yml)

A reinforcement learning agent for source-level vulnerability localization. Instead of
classifying a whole function in one pass, Soren learns to walk a function's control flow
graph and declare the statement responsible for a vulnerability.

The design and roadmap live in [`plan/IMPLEMENTATION_PLAN.md`](plan/IMPLEMENTATION_PLAN.md).

## Setup

Requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
ruff check .
ruff format --check .
pytest                 # full suite
pytest -m "not slow"   # fast suite, as run in CI
```

## Training

Train a masked PPO agent on any JSONL file of graph records. Synthetic graphs are enough to
try the pipeline:

```bash
python -m soren.data.synthetic --out data/synthetic/train.jsonl --n 2000 --seed 0
python -m soren.data.synthetic --out data/synthetic/val.jsonl --n 300 --seed 1
python scripts/train.py --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl \
    --name synthetic-demo --timesteps 200000
tensorboard --logdir runs/synthetic-demo/tb
```

Each run writes its resolved configuration, the best and final checkpoints, the validation
history and TensorBoard logs to `runs/<name>/`. Hyperparameters live in `configs/ppo.yaml`,
environment and reward settings in `configs/env.yaml`.

## Layout

| Path | Contents |
|---|---|
| `soren/data` | Graph schema, dataset loading, CFG construction, features |
| `soren/env` | The CFG navigation environment, rewards and wrappers |
| `soren/agents` | PPO and DQN agents |
| `soren/baselines` | Traversal baselines and reference models |
| `soren/eval` | Evaluation harness, metrics and statistics |
| `soren/viz` | Trace format and traversal visualizer |
| `scripts` | Data pipeline, training and evaluation entry points |
| `configs` | YAML configuration |
