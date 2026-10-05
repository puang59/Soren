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

## Data

Soren trains on [BigVul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)
(Fan et al., MSR 2020). Download one of the following into `data/raw/` (the directory is
git-ignored):

- `MSR_data_cleaned.csv` from the BigVul repository. Flaw lines are derived from the diff
  between the pre-fix and post-fix function.
- The preprocessed CSVs from [LineVul](https://github.com/awsm-research/LineVul), which
  already carry `flaw_line` and `flaw_line_index`.

`soren.data.bigvul.load_bigvul` accepts either and maps it to one set of canonical columns.
It stops with an error naming the column if a file lacks something it needs.

### Preparing the data

```bash
python scripts/01_filter_bigvul.py --input data/raw/MSR_data_cleaned.csv
python scripts/02_write_sources.py
```

The first script keeps vulnerable functions of the CWEs in `configs/data.yaml` that have
usable flaw lines, and records the row count after every filter in
`data/processed/attrition.json`. The second writes one C file per function under
`data/interim/src/`, in batches of 500, for Joern.

### Joern

CFG extraction uses [Joern](https://joern.io). Install it with the installer script from the
Joern documentation and make sure `joern` and `joern-parse` are on your `PATH`. Joern runs on
the JVM; check the Joern release notes for the JDK version your release needs.

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
