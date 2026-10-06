# Soren

[![CI](https://github.com/puang59/Soren/actions/workflows/ci.yml/badge.svg)](https://github.com/puang59/Soren/actions/workflows/ci.yml)

A reinforcement learning agent for source-level vulnerability localization. Instead of
classifying a whole function in one pass, Soren learns to walk a function's control flow
graph and declare the statement responsible for a vulnerability.

- **Report:** [`docs/report.md`](docs/report.md)
- **Demo script:** [`docs/demo.md`](docs/demo.md)
- **Experiment notes and figures:** [`experiments/`](experiments/)
- **Design and roadmap:** [`plan/IMPLEMENTATION_PLAN.md`](plan/IMPLEMENTATION_PLAN.md)

## Results in brief

On 127 held-out BigVul functions (CWE-119 and CWE-125), with one declaration per function:

| Method | Functions localized |
|---|---|
| DFS or line order, stopping at the first suspicious statement | 27.6% |
| **PPO agent** | **21.1%** |
| Random guess | 12.5% |

The agent learns something but does not beat simple baselines; the difference between them is
not statistically significant on this sample. A classifier that sees every statement at once
reaches about 23% at top-1, with or without CodeBERT embeddings, so the limit is what
per-statement features can say, not the search policy. On synthetic graphs with a clean
signal the same agent localizes 98% while inspecting fewer nodes than DFS. The report has the
details and the caveats.

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

- The [`bstee615/bigvul`](https://huggingface.co/datasets/bstee615/bigvul) mirror on Hugging
  Face: three Parquet files (252 MB) with the original column names. Put them in
  `data/raw/bigvul_hf/`. This is what the results in this repository use.

`soren.data.bigvul.load_bigvul` accepts any of these, including a directory of Parquet files,
and maps it to one set of canonical columns. It stops with an error naming the column if a
file lacks something it needs.

**CWEs.** The project uses CWE-119 and CWE-125, which leave 1,578 usable functions after
filtering (1,251 and 327). The numbers behind that choice are in
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

### Preparing the data

```bash
python scripts/01_filter_bigvul.py --input data/raw/bigvul_hf
python scripts/02_write_sources.py
scripts/03_run_joern.sh
python scripts/04_build_graphs.py
python scripts/05_split.py
python scripts/06_graph_stats.py
```

1. `01_filter_bigvul.py` keeps vulnerable functions of the CWEs in `configs/data.yaml` that
   have usable flaw lines, and records the row count after every filter in
   `data/processed/attrition.json`.
2. `02_write_sources.py` writes every function under `data/interim/src/`, in batches of 500,
   once as `.c` and once as `.cpp`. BigVul mixes C and C++, and Joern picks its parser by
   extension; parsing both ways recovers 97% of the functions, against 65% for C alone.
3. `03_run_joern.sh` parses each batch with Joern and exports its control flow graphs to
   `data/interim/joern/batch_XXXX.jsonl`. Finished batches are skipped on a rerun; failed ones
   are listed in `failed_batches.txt`. Set `JOBS` to change how many batches run at once.

4. `04_build_graphs.py` collapses each Joern CFG to one node per source line, drops badly
   parsed functions, maps the flaw lines to nodes and writes `data/processed/graphs_all.jsonl`.
   Every drop is counted by reason in `attrition.json`.
5. `05_split.py` splits by fixing commit into `graphs_{train,val,test}.jsonl` (80/10/10) and
   fails if a commit or a function body would be shared between splits.
6. `06_graph_stats.py` writes size, branching and label statistics for the training split to
   `data/processed/stats.json`.

To check the result by eye, `python scripts/inspect_graph.py --graphs
data/processed/graphs_train.jsonl --sample 5` prints graphs next to their source.

On the Hugging Face mirror this yields 1,278 graphs from the 1,578 filtered functions:
1,023 for training, 128 for validation and 127 for test.

### Joern

CFG extraction uses [Joern](https://joern.io). The pipeline was developed against Joern
4.0.648 on JDK 17. Download `joern-cli-<platform>.zip` from the
[Joern releases](https://github.com/joernio/joern/releases) and unpack it to
`.tools/joern-cli` (git-ignored), or set `JOERN_HOME` to wherever `joern` and `joern-parse`
live, or put them on your `PATH`.

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

## Evaluation and visualizer

```bash
# Baselines and a checkpoint through one harness; per-episode results go to Parquet.
python scripts/evaluate.py --graphs data/synthetic/val.jsonl --split val \
    --methods dfs bfs random_walk random_order line_order \
    --checkpoints runs/synthetic-demo/best_model.zip --out runs/synthetic-demo/eval_val.parquet

# Confidence intervals and paired tests against a reference method.
python scripts/summarize_results.py --results runs/synthetic-demo/eval_val.parquet --reference ppo

# Record traces, then replay them in the browser.
python scripts/make_traces.py --graphs data/synthetic/val.jsonl --limit 20 --methods dfs bfs \
    --checkpoints runs/synthetic-demo/best_model.zip --out runs/synthetic-demo/traces
streamlit run soren/viz/app.py -- --graphs data/synthetic/val.jsonl \
    --traces runs/synthetic-demo/traces
```

The visualizer shows the control flow graph next to the source and steps through an episode.
Ground truth stays hidden until you switch it on, and "Compare with" puts a second method
beside the first.

## Reproducing the BigVul experiments

After the data pipeline above has produced `data/processed/graphs_{train,val,test}.jsonl`:

```bash
# Final configuration, five seeds (about 4 minutes each on a laptop CPU).
for s in 0 1 2 3 4; do
  python scripts/run_arm.py --group final --name base --seed $s --timesteps 1000000
done

# An ablation arm is the default plus overrides; collect a group into a table.
python scripts/run_arm.py --group ablations --name declares_3 --seed 0 --timesteps 500000 \
    --set env.max_declares=3
python scripts/collect_runs.py --group ablations

# Reference classifier, and Protocol B thresholds tuned on validation.
python scripts/train_classifier.py --seed 0 --out runs/classifier/tier_L_seed0.pt
python scripts/tune_thresholds.py --graphs data/processed/graphs_val.jsonl --write-config

# The test-set evaluation. Run it once, after everything else is frozen.
python scripts/final_evaluation.py --allow-test
python scripts/make_figures.py
```

The frozen configuration is the default in `configs/ppo.yaml` and `configs/env.yaml`: Tier L
features, K = 6, one declaration, no shaping, no curriculum. The commit it was evaluated
from is tagged `v0.1-frozen`.

Statement embeddings (Tier E) are optional: `pip install -e ".[embed]"`, then
`python scripts/07_embed_statements.py`.

To try the pipeline without BigVul or Joern, use the synthetic graphs in the Training section.

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
| `experiments` | Experiment notes, result tables and figures |
| `docs` | Report and demo script |
