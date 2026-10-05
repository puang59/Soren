# Soren

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
