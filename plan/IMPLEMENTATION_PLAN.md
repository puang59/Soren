# Soren — Implementation Plan

**A Reinforcement Learning Agent for Source-Level Vulnerability Localization**

Source: `project_description.pdf`
Team: Karan Kumar (23BPS1034), Aryan Kumar (23BPS1193)
Plan date: 2026-10-05

---

## Table of Contents

1. [Goal and Scope](#1-goal-and-scope)
2. [Key Design Decisions](#2-key-design-decisions)
3. [System Architecture](#3-system-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Tech Stack](#5-tech-stack)
6. [Phase 0 — Project Setup and Synthetic Sandbox](#6-phase-0--project-setup-and-synthetic-sandbox)
7. [Phase 1 — Dataset Acquisition and Filtering](#7-phase-1--dataset-acquisition-and-filtering)
8. [Phase 2 — CFG Extraction and Label Alignment](#8-phase-2--cfg-extraction-and-label-alignment)
9. [Phase 3 — Node Feature Engineering](#9-phase-3--node-feature-engineering)
10. [Phase 4 — The RL Environment](#10-phase-4--the-rl-environment)
11. [Phase 5 — Baselines](#11-phase-5--baselines)
12. [Phase 6 — Agent Training](#12-phase-6--agent-training)
13. [Phase 7 — Evaluation and Ablations](#13-phase-7--evaluation-and-ablations)
14. [Phase 8 — Traversal Visualizer](#14-phase-8--traversal-visualizer)
15. [Phase 9 — Report and Demo](#15-phase-9--report-and-demo)
16. [Testing Strategy](#16-testing-strategy)
17. [Timeline and Work Split](#17-timeline-and-work-split)
18. [Risks and Mitigations](#18-risks-and-mitigations)
19. [Stretch Goals](#19-stretch-goals)
20. [Definition of Done](#20-definition-of-done)

---

## 1. Goal and Scope

### 1.1 What the project description commits to

| Requirement from the PDF | Where it is covered |
|---|---|
| Convert each function into a CFG (nodes = statements/basic blocks, edges = execution paths) | Phase 2 |
| State = current node + lightweight structural features (node type, in/out degree, neighbours, branch/call/assignment/return flags) | Phases 3, 4 |
| Three actions: move to a neighbour, backtrack, declare current node vulnerable | Phase 4 |
| Train with PPO or DQN | Phase 6 |
| Reward: small per-move penalty, large positive for correct localization, negative for wrong localization or excessive traversal | Phase 4 |
| Train/evaluate on BigVul, restricted to one or two CWEs | Phase 1 |
| Compare against Random Walk, BFS, DFS | Phase 5 |
| Metrics: localization accuracy, average search steps, success rate, cumulative reward | Phase 7 |
| Real-time visualization of the agent's traversal through the CFG | Phase 8 |

### 1.2 Problem statement (formal)

Given a function `f` that is **known to be vulnerable**, with CFG `G = (V, E)` and a ground-truth set of vulnerable statements `V* ⊆ V`, learn a policy `π(a | o)` that walks `G` from the entry node and issues a `DECLARE` action on some node `v ∈ V*` in as few steps as possible.

### 1.3 Explicitly out of scope (core deliverable)

- Function-level vulnerability *detection* (vulnerable vs. not). Every episode uses a vulnerable function.
- Inter-procedural analysis. Each function is one self-contained graph.
- Languages other than C/C++ (BigVul is C/C++).
- Beating state-of-the-art statement-level localizers. The claim is about *learned search behaviour versus uninformed traversal*, not SOTA.

---

## 2. Key Design Decisions

These are the decisions the PDF leaves open. Each has a recommendation; the rest of the plan assumes it.

| # | Decision | Recommendation | Reason |
|---|---|---|---|
| D1 | CFG granularity | **Statement (line) level**, not basic blocks | Ground truth in BigVul is line-level. Basic blocks would make "declare" ambiguous when a block has several lines. |
| D2 | CFG extractor | **Joern** (`c2cpg`), with CFG collapsed to one node per source line | Fuzzy parser handles isolated, non-compilable functions, which is exactly what BigVul provides. |
| D3 | CWEs | **CWE-119** as primary, **CWE-125** as second (fallback: pool the memory-safety family 119/120/125/787) | Largest classes in BigVul, and both have a strong lexical signature (buffer ops, indexing) the agent can learn from. |
| D4 | Ground truth | **Lines deleted or modified by the fixing commit** (`flaw_line` / `flaw_line_index`) | Standard practice for BigVul statement-level work. Add-only fixes have no label and are dropped. |
| D5 | RL library | **Gymnasium + Stable-Baselines3 + sb3-contrib `MaskablePPO`** | The neighbour count varies per node, so invalid-action masking is required. `MaskablePPO` supports it natively. |
| D6 | Primary algorithm | **PPO (masked)**; DQN as a secondary comparison | SB3's stock DQN has no action masking, so DQN needs a small custom masked implementation. Do it only after PPO works. |
| D7 | Action space | `Discrete(K + 2)`: `K` successor slots + `BACKTRACK` + `DECLARE`, with `K = 6` | Fixed-size action space over a variable-degree graph. `K` is chosen from the out-degree distribution (Phase 2 output). |
| D8 | Observation | Flat vector: current-node features + `K` neighbour feature slots + episode context | Lets a plain MLP policy work; the agent can see what each move leads to. |
| D9 | Features beyond pure structure | Add **lexical flags** (dangerous API call, array subscript, pointer dereference, …) | See 2.1. This is the single most important decision in the plan. |
| D10 | Baseline comparison | Two protocols: oracle-stop search efficiency **and** declare-based localization | BFS/DFS/Random Walk have no native notion of "declare". See Phase 5. |
| D11 | Data split | **By fixing commit**, after de-duplicating functions | BigVul contains near-duplicate functions; random splits leak. |

### 2.1 Why purely structural features are not enough

The PDF lists node type, in/out degree, and statement-kind flags as the state. With only those, two `assignment` nodes with the same degree are indistinguishable, so the best any policy can do is learn a prior such as "vulnerabilities tend to sit N nodes deep in a loop body". That is a weak, hard-to-defend result.

The plan therefore defines three feature tiers and treats them as an **ablation axis**, which turns a weakness into an experiment:

- **Tier S (structural):** exactly what the PDF lists. This is the faithful baseline configuration.
- **Tier L (structural + lexical):** adds cheap regex/AST-derived flags per statement. Still "lightweight"; no neural encoder. **This is the default.**
- **Tier E (embedding, stretch):** adds a frozen CodeBERT embedding of the statement text, PCA-reduced.

The report should show Tier S vs. Tier L. If Tier L is much better, that is a finding, not a problem.

---

## 3. System Architecture

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                    OFFLINE DATA PIPELINE                 │
                 │                                                          │
  BigVul CSV ──► │ filter (CWE, labels) ─► write .c files ─► Joern (CPG)    │
                 │                                             │            │
                 │                              export CFG + AST info (JSON)│
                 │                                             │            │
                 │       collapse to line-level CFG ◄──────────┘            │
                 │                │                                         │
                 │       align flaw lines ─► V*                             │
                 │                │                                         │
                 │       featurize nodes (Tier S / L / E)                   │
                 │                │                                         │
                 │       dedupe + split by commit ─► graphs/{train,val,test}│
                 └────────────────┼─────────────────────────────────────────┘
                                  ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ CFGNavEnv (Gymnasium)                                                 │
   │   reset(): sample graph, place agent at ENTRY                         │
   │   step(a): MOVE_i | BACKTRACK | DECLARE  → obs, reward, term, trunc   │
   │   action_masks(): valid actions for the current node                  │
   └───────────┬───────────────────────────────┬───────────────────────────┘
               ▼                               ▼
     MaskablePPO / masked DQN        Baselines: RandomWalk, BFS, DFS,
     (training + checkpoints)        heuristic, node classifier
               │                               │
               └──────────────┬────────────────┘
                              ▼
                 Evaluation harness ─► metrics tables, plots
                              │
                              ▼
                 Episode traces (JSON) ─► Streamlit traversal visualizer
```

The single shared artefact between every stage is the **processed graph record** (Section 8.5). Everything downstream of Phase 2 reads only that format, so the environment, baselines, and visualizer can be built against synthetic graphs before the real data is ready.

---

## 4. Repository Layout

```
Soren/
├── project_description.pdf
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml               # CWE list, size filters, K, split ratios, seeds
│   ├── env.yaml                # reward values, max steps, feature tier
│   ├── ppo.yaml                # PPO hyperparameters
│   ├── dqn.yaml
│   └── eval.yaml
├── data/                       # git-ignored
│   ├── raw/                    # BigVul CSV
│   ├── interim/                # per-function .c files, Joern output
│   └── processed/              # graphs_{train,val,test}.jsonl + stats
├── scripts/
│   ├── 01_filter_bigvul.py
│   ├── 02_write_sources.py
│   ├── 03_run_joern.sh
│   ├── 04_build_graphs.py
│   ├── 05_split.py
│   ├── train.py
│   ├── evaluate.py
│   └── make_traces.py
├── joern/
│   └── export_cfg.sc           # Joern script: CPG → JSON
├── soren/
│   ├── data/
│   │   ├── bigvul.py           # loading + filtering
│   │   ├── cfg_builder.py      # Joern JSON → line-level networkx graph
│   │   ├── labels.py           # flaw-line → node alignment
│   │   ├── features.py         # Tier S / L / E featurizers
│   │   ├── schema.py           # GraphRecord dataclass + (de)serialisation
│   │   └── synthetic.py        # synthetic CFG generator
│   ├── env/
│   │   ├── cfg_nav_env.py      # Gymnasium environment
│   │   ├── rewards.py          # reward config + shaping
│   │   └── wrappers.py         # curriculum sampler, trace recorder
│   ├── agents/
│   │   ├── ppo.py              # MaskablePPO construction
│   │   ├── masked_dqn.py       # custom DQN with action masking
│   │   └── policies.py         # feature extractor(s)
│   ├── baselines/
│   │   ├── base.py             # common Searcher interface
│   │   ├── random_walk.py
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── heuristic.py        # dangerous-API-first ordering
│   │   └── node_classifier.py  # supervised top-k ranker (no navigation)
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── runner.py           # runs any policy/baseline over a split
│   │   └── stats.py            # bootstrap CIs, paired tests
│   └── viz/
│       ├── trace.py            # trace schema
│       └── app.py              # Streamlit app
├── tests/
├── notebooks/                  # EDA, result plots
└── runs/                       # git-ignored: checkpoints, tensorboard, traces
```

---

## 5. Tech Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Environment API | `gymnasium` | |
| RL | `stable-baselines3`, `sb3-contrib` (`MaskablePPO`) | Pin both to matching versions. |
| DL backend | `torch` | CPU is enough for the MLP policy; MPS/GPU optional. |
| Graphs | `networkx` | Graphs are small (tens to low hundreds of nodes). |
| CFG extraction | Joern | Needs a JDK; check the required version for the installed Joern release. |
| Data | `pandas`, `pyarrow` | BigVul CSV is large; convert to Parquet once. |
| Config | plain YAML + dataclasses | Hydra is unnecessary at this size. |
| Logging | TensorBoard (SB3 built-in) | |
| Stats/plots | `scipy`, `matplotlib` | |
| Visualizer | `streamlit` + `graphviz` | |
| Tests | `pytest` | |
| Lint/format | `ruff` | |

---

## 6. Phase 0 — Project Setup and Synthetic Sandbox

**Purpose:** de-risk the RL half of the project before any real data exists. The Joern pipeline is the most likely thing to slip, and the environment must not wait for it.

### Tasks

1. `git init`, `pyproject.toml`, `ruff`, `pytest`, `.gitignore` (`data/`, `runs/`).
2. Define the `GraphRecord` schema (Section 8.5) first. It is the contract between the two halves of the team.
3. Write `soren/data/synthetic.py`:
   - Generate random structured CFGs by recursively composing `sequence`, `if/else`, `while`, and `switch` templates until a target node count is reached. This yields realistic shapes (single entry, single exit, reducible loops).
   - Plant one vulnerable node. Give it a **noisy feature signal**: for example, set the `dangerous_call` flag on the vulnerable node with probability 0.9 and on each other node with probability 0.1.
   - Expose difficulty knobs: graph size, signal strength, number of decoys.
4. Build the environment (Phase 4) and train PPO against synthetic graphs.

### Exit criteria

- On synthetic graphs with a clean signal, PPO reaches >90% success and beats DFS on steps.
- With the signal turned off (pure noise), PPO degrades to roughly baseline level. This confirms the agent is using features and not exploiting a bug in the environment (for example, node ordering leaking the label).

---

## 7. Phase 1 — Dataset Acquisition and Filtering

### 7.1 Source

BigVul (Fan et al., MSR 2020). Use the cleaned CSV (`MSR_data_cleaned.csv`) or the LineVul-preprocessed release, which already contains `flaw_line` and `flaw_line_index` columns. The relevant columns are the pre-fix function body, the post-fix function body, the vulnerable flag, the CWE ID, the commit ID, and the project name.

> Column names differ between releases. `soren/data/bigvul.py` should map them to canonical names in one place and fail loudly if an expected column is missing.

### 7.2 Filtering steps (`scripts/01_filter_bigvul.py`)

Apply in order and **log the row count after each step** into `data/processed/attrition.json`. The attrition table goes into the report.

1. Keep `vul == 1`.
2. Keep rows whose CWE is in `configs/data.yaml: cwes` (default `["CWE-119", "CWE-125"]`).
3. Drop rows with no flaw lines (add-only fixes).
4. Drop rows where every flaw line is blank, a lone brace, or a comment.
5. Drop functions shorter than 5 or longer than 300 source lines (tune after EDA).
6. De-duplicate: normalise whitespace and strip comments, hash the pre-fix body, and keep one row per hash.

If flaw lines are not present in the chosen release, derive them: run `difflib.unified_diff` between the pre-fix and post-fix bodies and take the line indices on the `-` side.

### 7.3 EDA notebook (`notebooks/01_eda.ipynb`)

Produce and save:

- Count of usable functions per CWE after each filter.
- Distribution of function length in lines.
- Distribution of the number of flaw lines per function.
- Relative position of flaw lines in the function (0 = first line, 1 = last). **This matters:** if flaw lines cluster at a specific relative position, a trivial positional policy can score well, and the evaluation must include that as a baseline.

### 7.4 Decision gate

If fewer than about 1,000 usable functions survive for the chosen CWEs, widen to the memory-safety family (CWE-119, 120, 125, 787) and treat it as one category. Record the decision in the README.

### Exit criteria

- `data/interim/filtered.parquet` exists with canonical columns and a stable `sample_id`.
- Attrition table and EDA plots are saved.

---

## 8. Phase 2 — CFG Extraction and Label Alignment

This is the highest-risk engineering phase. Budget accordingly.

### 8.1 Write sources (`scripts/02_write_sources.py`)

- One file per function: `data/interim/src/batch_XXXX/<sample_id>.c`, containing only the pre-fix function body **starting at line 1**. Starting at line 1 makes the flaw-line index map directly to Joern line numbers (`joern_line = flaw_line_index + 1` for a 0-based index).
- Group into batches of about 500 files per directory so the JVM startup cost is amortised.

### 8.2 Run Joern (`scripts/03_run_joern.sh`, `joern/export_cfg.sc`)

For each batch directory:

1. `joern-parse <batch_dir> --output <batch>.cpg.bin`
2. `joern --script joern/export_cfg.sc --param cpgFile=<batch>.cpg.bin --param outFile=<batch>.json`

The script iterates over methods (skipping synthetic `<global>` and `<operator>.*` methods) and emits, per method:

- source filename (to recover `sample_id`)
- CFG nodes: `id`, `label` (CALL, IDENTIFIER, LITERAL, RETURN, CONTROL_STRUCTURE, METHOD, METHOD_RETURN, …), `code`, `lineNumber`
- CFG edges: `(src_id, dst_id)`
- control structures: `controlStructureType` (IF, WHILE, FOR, DO, SWITCH, BREAK, CONTINUE, GOTO) and `lineNumber`
- calls: callee `name` and `lineNumber`

Run batches in parallel with `xargs -P` or GNU `parallel`. Log failures per batch and continue; do not let one bad file abort the run.

### 8.3 Collapse to a line-level CFG (`soren/data/cfg_builder.py`)

Joern's CFG is at **expression** granularity: a single statement such as `buf[i] = foo(x)` becomes several CFG nodes. The agent needs one node per statement.

Algorithm:

1. Group Joern CFG nodes by `lineNumber`. Each distinct line becomes one node.
2. Add the synthetic `ENTRY` (Joern `METHOD`) and `EXIT` (`METHOD_RETURN`) nodes.
3. For every Joern CFG edge `(u, v)` with `line(u) != line(v)`, add edge `line(u) → line(v)`.
4. Drop self-loops produced by intra-line flow.
5. Attach to each node: the raw source line text, the set of Joern node labels on that line, any control-structure type, and callee names.
6. Compute and store: depth from `ENTRY` (BFS distance), back-edge flags (edge whose target dominates its source, or more simply an edge to a node with smaller DFS discovery time that is an ancestor), and a `in_loop` flag for nodes inside a cycle (non-trivial strongly connected component).

Known wrinkles to handle:

- **Multi-line statements.** A call spanning three lines produces nodes on several lines. Use a statement span map (from Joern AST line ranges, or `tree-sitter-c` as a fallback) to merge all lines of one statement into the node of its first line.
- **Multiple statements on one line.** Accept the merge; the ground truth is line-level anyway.
- **Macros and unparsable code.** Joern will still emit something. Reject a graph if fewer than 60% of non-blank, non-comment source lines map to a node.

### 8.4 Label alignment (`soren/data/labels.py`)

1. Map each flaw-line index to a line number, then to the node containing that line (via the statement span map).
2. **Verify by text:** the stripped flaw-line text must be a substring of the node's stripped source text. If the index is off by one in a given release, this check catches it. Fail the sample, not the run.
3. `V*` = set of nodes that received at least one flaw line.
4. Drop the sample if `V*` is empty or if no node in `V*` is reachable from `ENTRY`.

### 8.5 Processed graph record (`soren/data/schema.py`)

One JSON object per line in `data/processed/graphs_{split}.jsonl`:

```json
{
  "sample_id": "bigvul_000123",
  "project": "linux",
  "commit_id": "abc123...",
  "cwe": "CWE-119",
  "source_lines": ["static int foo(...)", "{", "..."],
  "nodes": [
    {"id": 0, "line": 1, "kind": "ENTRY", "code": "", "depth": 0, "in_loop": false},
    {"id": 1, "line": 3, "kind": "ASSIGN", "code": "len = hdr->len;", "depth": 1, "in_loop": false}
  ],
  "edges": [[0, 1], [1, 2]],
  "back_edges": [[7, 4]],
  "entry": 0,
  "exit": 12,
  "vuln_nodes": [5],
  "features": {"tier_s": [[...]], "tier_l": [[...]]}
}
```

Successor ordering must be **canonical and label-independent**: sort each node's successors by line number. Never order by anything correlated with `vuln_nodes`.

### 8.6 Split (`scripts/05_split.py`)

- Group by `commit_id`; assign whole groups to train/val/test at 80/10/10 with a fixed seed.
- Stratify by CWE where possible.
- Assert no normalised-body hash appears in more than one split.

### 8.7 Graph statistics (`data/processed/stats.json`)

Compute over the training split: node count distribution, out-degree distribution, `|V*|` distribution, shortest-path distance from `ENTRY` to nearest vulnerable node, fraction of nodes that are vulnerable. Use these to set:

- `K` (successor slots) = the smallest value covering at least 99% of nodes. Expect 2 for most nodes and larger values only for `switch`.
- `max_steps` = e.g. `min(200, 4 * |V|)` per graph.

Nodes whose out-degree exceeds `K` (large `switch` statements): either drop those graphs if rare, or rewrite the switch as a chain of binary dispatch nodes. Decide from the statistics.

### Exit criteria

- `graphs_train.jsonl`, `graphs_val.jsonl`, `graphs_test.jsonl`, and `stats.json` exist.
- Manual spot-check of 20 random graphs rendered with Graphviz next to their source, confirming edges and flaw-node placement look right.
- Alignment unit tests pass (Section 16).

---

## 9. Phase 3 — Node Feature Engineering

`soren/data/features.py` exposes `featurize(graph, tier) -> np.ndarray[num_nodes, F_tier]`. Features are precomputed and stored in the graph record.

### 9.1 Tier S — structural (matches the PDF)

| Feature | Encoding |
|---|---|
| Node kind | One-hot over: `ENTRY`, `EXIT`, `ASSIGN`, `CALL`, `BRANCH` (if), `LOOP` (for/while/do), `SWITCH`, `RETURN`, `JUMP` (break/continue/goto), `DECL`, `OTHER` |
| In-degree | `min(deg, 4) / 4` |
| Out-degree | `min(deg, 4) / 4` |
| Depth from entry | `depth / max_depth_in_graph` |
| In loop | 0/1 |
| Is loop header | 0/1 |
| Relative line position | `line / num_lines` |

Kind is assigned by priority when a line carries several constructs: control structure > return > jump > assignment > call > declaration > other. A line such as `x = foo(y);` is `ASSIGN` with the separate `has_call` flag set in Tier L.

### 9.2 Tier L — lexical additions (default)

| Feature | Detection |
|---|---|
| `has_call` | Any non-operator call on the line |
| `dangerous_call` | Callee in a configurable list: `memcpy`, `memmove`, `memset`, `strcpy`, `strncpy`, `strcat`, `strncat`, `sprintf`, `snprintf`, `vsprintf`, `gets`, `read`, `recv`, `fread`, `copy_from_user`, `copy_to_user`, `alloca`, … |
| `alloc_call` | `malloc`, `calloc`, `realloc`, `kmalloc`, `kzalloc`, `vmalloc`, `new`, … |
| `free_call` | `free`, `kfree`, `vfree`, `delete`, … |
| `array_subscript` | Joern `<operator>.indirectIndexAccess` on the line |
| `pointer_deref` | `<operator>.indirection` or `<operator>.indirectFieldAccess` |
| `pointer_arith` | Addition/subtraction where an operand is a pointer-typed identifier, or regex fallback |
| `arith_op` | `+ - * / % << >>` operators |
| `comparison` | Relational operator present (possible bounds check) |
| `uses_sizeof` | `sizeof` present |
| `has_cast` | `<operator>.cast` |
| `length_like_ident` | Identifier matching `len|size|count|num|idx|index|offset|off|pos` |
| `token_count` | `min(tokens, 40) / 40` |

Prefer Joern operator names over regexes where both are available; they are more reliable. Keep the lists in `configs/data.yaml` so they can be edited without touching code.

### 9.3 Tier E — embeddings (stretch)

Frozen CodeBERT `[CLS]` embedding of each statement's text, reduced to 32 dimensions with PCA fitted on the training split only.

### 9.4 Leakage checklist

- No feature may be derived from the post-fix function or the diff.
- No feature may depend on node ID if node IDs were assigned in an order that correlates with the label.
- Any normalisation statistics (PCA, means) are fitted on the training split only.

---

## 10. Phase 4 — The RL Environment

`soren/env/cfg_nav_env.py` — `class CFGNavEnv(gymnasium.Env)`.

### 10.1 Episode lifecycle

- `reset(seed, options)`: sample a graph from the split's sampler (uniform, or curriculum-weighted), place the agent at `ENTRY`, clear the path stack and visit counts. `options={"graph_id": ...}` forces a specific graph for evaluation and visualisation.
- `step(action)`: apply the action, return `(obs, reward, terminated, truncated, info)`.
- `action_masks() -> np.ndarray[bool]`: required by `MaskablePPO`.

### 10.2 Action space

`Discrete(K + 2)` with `K = 6` (confirm from `stats.json`):

| Index | Action | Valid when |
|---|---|---|
| `0 … K-1` | `MOVE_i`: move to the i-th successor (canonical order) | `i < out_degree(current)` |
| `K` | `BACKTRACK`: pop the path stack, return to the previous node | Path stack non-empty |
| `K + 1` | `DECLARE`: declare the current node vulnerable | Current node is not `ENTRY` or `EXIT` |

Semantics:

- `MOVE_i` pushes the current node on the path stack, then moves.
- `BACKTRACK` pops one node. It is how the agent leaves a dead end (for example after reaching `EXIT` or a `return`) and explores a sibling branch.
- Moves follow CFG edges forward only. Because of `BACKTRACK`, every node reachable from `ENTRY` remains reachable by the agent.

### 10.3 Observation space

A single flat `Box(float32)` vector, concatenating:

1. **Current node features** — `F` values (Tier S or L).
2. **Current node episode features** — visit count for this node (clipped, scaled), already-declared-here flag.
3. **Successor slots** — `K` slots, each: `[valid, node features (F), visited_before, is_back_edge]`. Empty slots are zero-filled.
4. **Backtrack target summary** — `[valid, visited_count, kind one-hot]` of the node on top of the stack.
5. **Episode context** — `step / max_steps`, fraction of nodes visited, stack depth (scaled), `log(|V|)` (scaled), declares remaining (scaled).

With Tier L (`F ≈ 30`) and `K = 6` the vector is roughly 250 values. A plain MLP handles this easily.

Design notes:

- The environment is partially observable: the agent sees one hop. The `visited` flags and context features give it enough memory to avoid loops without a recurrent policy. (`sb3-contrib` has no combined recurrent + maskable PPO, so in-observation memory is the pragmatic route.)
- Do **not** include node IDs or absolute line numbers in the observation.

### 10.4 Reward function

All values live in `configs/env.yaml`. Defaults:

| Event | Reward | Terminal? |
|---|---|---|
| Any `MOVE` or `BACKTRACK` | `-0.01` | No |
| `MOVE` onto an already-visited node | additional `-0.02` | No |
| `DECLARE` on a node in `V*` | `+1.0` | Yes |
| `DECLARE` on a node not in `V*` | `-0.5` | Yes if declare budget exhausted, else no |
| Step limit reached without a correct declare | `-0.5` | Truncated |

- **Declare budget** (`max_declares`, default 1): number of wrong declarations allowed before the episode ends. Setting it to 3 or 5 gives a natural Top-k evaluation. Train with 1 first.
- **Optional potential-based shaping** (off by default; an ablation): `r += γ·Φ(s') − Φ(s)` with `Φ(s) = −β · d(s)`, where `d` is the shortest-path distance from the current node to the nearest node in `V*`, and `β ≈ 0.05`. Potential-based shaping preserves the optimal policy. It uses the label, so it is strictly a **training-time** signal and must be disabled during evaluation reward reporting. Report cumulative reward using the unshaped reward only.
- Scale check: the worst-case accumulated step cost (`max_steps × 0.01`) should stay comparable to, and not far exceed, the `+1.0` terminal reward. Otherwise "declare immediately and take the loss" becomes optimal. With `max_steps = 200` the accumulated cost reaches `−2.0`, so either cap `max_steps` lower for small graphs (`4·|V|`) or reduce the step penalty to `−0.005`. Verify with the degenerate-policy test in Section 16.

### 10.5 `info` dictionary

Returned every step for the evaluation harness and the visualizer: `graph_id`, `current_node`, `action_name`, `visited_unique`, `is_success`, `declared_node`, `first_hit_step` (step at which any `V*` node was first visited, or `None`).

### 10.6 Wrappers (`soren/env/wrappers.py`)

- **`CurriculumSampler`**: sampling weight over graphs by node count; starts with graphs of ≤ 20 nodes and widens the cap on a schedule or when rolling success exceeds a threshold.
- **`TraceRecorder`**: records `(step, node, action, reward, mask, action_probs)` per episode to JSON for the visualizer.

### Exit criteria

- `gymnasium.utils.env_checker.check_env` passes.
- A scripted oracle policy (shortest path to `V*`, then declare) achieves 100% success, validating that every kept graph is solvable.
- A random masked policy runs 10,000 episodes without an exception.

---

## 11. Phase 5 — Baselines

### 11.1 The comparison problem

Random Walk, BFS, and DFS define a *visiting order*. They have no rule for *declaring*. Two protocols resolve this; report both.

**Protocol A — Search efficiency (oracle stop).**
A baseline succeeds the moment it first visits any node in `V*`; the metric is the number of nodes inspected before that happens. The RL agent is measured the hard way: it must both reach **and** correctly declare. This is deliberately favourable to the baselines; if the agent still inspects fewer nodes, the result is robust.

**Protocol B — Localization (declare-based).**
Every method must issue a declaration. Baselines are paired with a shared stopping rule: traverse in the baseline's order and declare at the first node whose score `s(v)` exceeds a threshold `τ`, where `τ` is tuned on the validation split. Two scorers:

- `heuristic`: a fixed weighted sum of Tier L flags (dangerous call, array subscript, pointer arithmetic).
- `classifier`: the supervised node classifier from 11.3.

This isolates the question "does learned *navigation* help, given the same per-node information?".

### 11.2 Baseline implementations (`soren/baselines/`)

Common interface:

```python
class Searcher(Protocol):
    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult: ...
```

`EpisodeResult` holds the visit order, the declared node (if any), success flag, nodes inspected, raw actions taken, and cumulative reward under the environment's reward config.

| Baseline | Behaviour |
|---|---|
| `RandomWalk` | Uniform choice among valid `MOVE`/`BACKTRACK` actions through the real environment. Average over many seeds. |
| `BFS` | Breadth-first from `ENTRY`, successors in canonical order. |
| `DFS` | Depth-first from `ENTRY`, successors in canonical order, backtracking at dead ends. Maps exactly onto the environment's actions. |
| `RandomOrder` | Visit nodes in a uniformly random permutation. Analytic expectation with one target is `(|V| + 1) / 2`; a useful sanity reference. |
| `LineOrder` | Visit nodes top to bottom by line number. A trivial "read the function" baseline. |
| `HeuristicFirst` | Best-first search ordered by the heuristic score. A strong non-learned baseline. |

### 11.3 Supervised node classifier (non-navigating reference)

A small MLP (or gradient-boosted trees) trained on Tier L node features to predict membership in `V*`, with class weighting for the heavy imbalance. At test time, rank all nodes in a function and report Top-1/Top-3/Top-5 accuracy.

This is not a traversal method, but it answers an examiner's obvious question: *"what if you just classify every node?"*. It sees the whole graph at once, so it is an upper-bound-style reference on the information content of the features, not a like-for-like competitor.

### 11.4 Step accounting

Report two step counts so the comparison is honest:

- **Nodes inspected** — number of distinct nodes visited. Comparable across all methods, including BFS (which "teleports" between frontier nodes). **Primary metric.**
- **Actions taken** — raw environment actions including backtracks and revisits. Comparable between the RL agent, DFS, and Random Walk.

---

## 12. Phase 6 — Agent Training

### 12.1 PPO (primary) — `soren/agents/ppo.py`, `scripts/train.py`

- `MaskablePPO("MlpPolicy", env, ...)` with the environment exposing `action_masks()`.
- Vectorised with `SubprocVecEnv` (8–16 environments).
- `VecNormalize` for rewards only (observations are already scaled to roughly `[0, 1]`).

Starting hyperparameters (`configs/ppo.yaml`):

| Parameter | Value |
|---|---|
| Policy network | MLP `[256, 256]`, separate actor/critic heads, `tanh` |
| `n_envs` | 16 |
| `n_steps` | 512 |
| `batch_size` | 512 |
| `n_epochs` | 10 |
| `learning_rate` | `3e-4`, linear decay |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 (raise to 0.02–0.05 if the policy collapses to immediate `DECLARE`) |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| Total timesteps | 2–5 M to start |

### 12.2 Training loop details

- **Evaluation callback:** every N updates, run the deterministic policy over the whole validation split; log success rate, mean nodes inspected, mean reward. Save the best checkpoint by validation success rate (tie-break on steps).
- **Curriculum:** enable `CurriculumSampler` if training on the full size range stalls.
- **Seeds:** 5 seeds per reported configuration. Report mean ± std across seeds.
- **Logged diagnostics:** action distribution (fraction of `MOVE`/`BACKTRACK`/`DECLARE`), mean episode length, fraction of episodes ending by wrong declare vs. timeout, entropy.

### 12.3 Failure modes to watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| Declares at the first legal node every time | Step cost too high relative to wrong-declare penalty | Lower step penalty, raise wrong-declare penalty, raise `ent_coef` |
| Never declares; always times out | Wrong-declare penalty dominates | Lower it, or add shaping, or start with the curriculum on tiny graphs |
| Train success high, validation near baseline | Memorising training graphs | More data (widen CWEs), smaller network, dropout, feature noise, stronger de-duplication |
| Oscillates between two nodes | Missing revisit penalty or visit features | Check observation includes visit flags |

### 12.4 Masked DQN (secondary) — `soren/agents/masked_dqn.py`

Stock SB3 `DQN` does not support action masking. Implement a compact DQN (about 200 lines of PyTorch):

- Q-network: same MLP body, `K + 2` outputs.
- Masking in **both** places: action selection (set invalid Q-values to `−inf` before `argmax`; sample ε-greedy only among valid actions) and the bootstrap target (max over valid next actions, so the replay buffer must store the next-state mask).
- Double DQN target, Huber loss, target network with periodic hard update, uniform replay buffer (size ~200k), ε annealed 1.0 → 0.05.

Do this only after PPO produces a credible result. If time is short, drop DQN and state in the report that PPO was chosen for native masking support; the PDF says "PPO **or** DQN".

### 12.5 Hyperparameter search

Small, targeted sweep on the validation split only (never test): step penalty `{0.005, 0.01, 0.02}`, wrong-declare penalty `{0.25, 0.5, 1.0}`, `ent_coef` `{0.005, 0.01, 0.03}`, network width `{128, 256}`. A manual grid is enough; Optuna is optional.

---

## 13. Phase 7 — Evaluation and Ablations

### 13.1 Harness (`soren/eval/runner.py`, `scripts/evaluate.py`)

One entry point that takes any `Searcher` (RL policies are wrapped to implement the same interface), a split, and a config, and writes one row per `(method, seed, graph)` to a Parquet file. All tables and plots are derived from that file, so every number in the report is reproducible from a single artefact.

The **test split is touched once**, after all design and hyperparameter choices are frozen.

### 13.2 Metrics (`soren/eval/metrics.py`)

| Metric | Definition |
|---|---|
| **Localization accuracy** | Fraction of episodes whose (first) declared node is in `V*`. With declare budget `k`: Top-k accuracy. |
| **Success rate** | Fraction of episodes that end with a correct declaration within the step budget. (Protocol A for baselines: fraction that reach `V*` within the budget.) |
| **Average search steps** | Mean nodes inspected, and mean actions taken, over (a) successful episodes and (b) all episodes with failures counted at `max_steps`. Report both. |
| **Normalised search cost** | Nodes inspected ÷ `|V|`. Makes graphs of different sizes comparable. |
| **Cumulative reward** | Mean undiscounted, unshaped episode return under the default reward config. |
| **Path efficiency** | Agent's actions ÷ shortest-path length from `ENTRY` to nearest `V*` node (1.0 is optimal). |

### 13.3 Breakdowns

- By graph size bucket: ≤ 20, 21–50, 51–100, > 100 nodes.
- By CWE.
- By distance from `ENTRY` to nearest vulnerable node.
- By `|V*|` (single vs. multiple vulnerable statements).

### 13.4 Statistics (`soren/eval/stats.py`)

- 95% bootstrap confidence intervals over test graphs (resample graphs, 10,000 resamples).
- Paired comparison between the agent and each baseline on the same graphs: Wilcoxon signed-rank on nodes inspected; McNemar on success.
- Mean ± std across the 5 training seeds.

### 13.5 Ablations

| Ablation | Question answered |
|---|---|
| Tier S vs. Tier L (vs. Tier E) | How much do lexical features matter beyond pure structure? |
| With vs. without `BACKTRACK` | Is backtracking actually used and useful? |
| With vs. without potential shaping | Does shaping speed up learning or change final performance? |
| With vs. without revisit penalty | |
| Declare budget 1 / 3 / 5 | Top-k behaviour |
| PPO vs. masked DQN | Algorithm comparison |
| Train on CWE-119, test on CWE-125 (and reverse) | Cross-CWE transfer |
| Shuffled-label control | Train with `V*` replaced by a random node. Performance must fall to chance; if it does not, there is leakage. |

### 13.6 Figures for the report

1. Learning curves (validation success and nodes inspected vs. timesteps), mean ± std over seeds.
2. Bar chart: success rate per method, with CIs.
3. Bar chart or box plot: normalised search cost per method.
4. Line plot: success rate vs. graph size bucket, one line per method.
5. Cumulative "found within N nodes inspected" curve per method (the most informative single figure).
6. Action-distribution histogram of the trained agent.
7. Two or three qualitative traversal screenshots from the visualizer: one clean success, one failure.

---

## 14. Phase 8 — Traversal Visualizer

The PDF calls out interpretability and real-time visualisation as a selling point, so this is a required deliverable, not a stretch.

### 14.1 Trace format (`soren/viz/trace.py`)

```json
{
  "graph_id": "bigvul_000123",
  "method": "ppo_tierL_seed0",
  "steps": [
    {"t": 0, "node": 0, "action": "MOVE_0", "reward": -0.01,
     "mask": [true, false, "..."], "probs": [0.81, 0.0, "..."], "value": 0.42}
  ],
  "outcome": {"success": true, "declared_node": 5, "return": 0.93}
}
```

Baselines emit the same format (without `probs`/`value`), so the app can replay any method.

### 14.2 App (`soren/viz/app.py`, Streamlit)

Layout:

- **Sidebar:** split, graph picker (filter by CWE, size, outcome), method picker, "live" vs. "replay" mode.
- **Left pane — CFG:** Graphviz rendering, top-to-bottom. Colour coding: current node (highlighted), visited nodes (shaded by visit count), path stack (bold outline), declared node (green if correct, red if wrong), ground-truth nodes (dashed outline, behind a "show ground truth" toggle so the demo can hide the answer until the end).
- **Right pane — source:** the function with line numbers; current line highlighted; visited lines tinted; flaw lines marked when the toggle is on.
- **Bottom:** step slider, play/pause with a speed control, step table (action, reward, running return), bar chart of the policy's action probabilities at the current step, value estimate.
- **Compare mode:** two methods side by side on the same graph (for example PPO vs. DFS).

Modes:

- **Replay:** load a saved trace JSON.
- **Live:** load a checkpoint, step the environment on demand, and render after each action. This is the "real time" behaviour from the PDF.

Also allow pasting an arbitrary C function in live mode *only if* Joern is available on the demo machine; otherwise restrict live mode to pre-processed graphs.

### Exit criteria

- Any test graph can be replayed for any method.
- Live mode steps a trained checkpoint without lag on graphs up to about 150 nodes.

---

## 15. Phase 9 — Report and Demo

- **README:** setup, how to reproduce the data pipeline, how to train, how to evaluate, how to launch the visualizer.
- **Report sections:** introduction and motivation; related work (GNN/Transformer detectors, statement-level localizers, RL on graphs); MDP formulation (state, action, reward, transition, termination); dataset and preprocessing with the attrition table; baselines and evaluation protocols; results; ablations; limitations; future work.
- **Limitations to state openly:** label noise in BigVul (changed lines are a proxy for the vulnerable statement); small dataset after filtering; intra-procedural only; baselines evaluated under an oracle stop in Protocol A; Tier L features are hand-designed for memory-safety CWEs.
- **Demo script (about 5 minutes):** show a function; run DFS in the visualizer; run the agent on the same function with ground truth hidden; reveal ground truth; show one failure case and explain it; show the headline results chart.

---

## 16. Testing Strategy

| Area | Tests |
|---|---|
| **CFG builder** | Hand-written C snippets (straight line, if/else, while, for, switch, early return, goto, nested loops) with expected line-level edges asserted exactly. |
| **Label alignment** | Flaw-line text is contained in the mapped node's text; off-by-one detection; multi-line statement maps to its first-line node; samples with unalignable labels are rejected, not silently kept. |
| **Features** | Each lexical flag fires on a positive example and not on a negative one; feature matrix has no NaN and stays within `[0, 1]`. |
| **Split** | No shared `commit_id` and no shared body hash across splits. |
| **Environment — API** | `check_env` passes; observation shape and dtype are constant across graphs. |
| **Environment — masks** | For every reachable state in a set of small graphs: mask is true exactly for valid actions; taking a masked action raises. |
| **Environment — dynamics** | `MOVE` then `BACKTRACK` returns to the same node with the same stack; `DECLARE` on `V*` terminates with the configured reward; step limit truncates. |
| **Environment — determinism** | Same seed and same action sequence produce identical trajectories. |
| **Reward sanity** | Oracle policy return > any degenerate policy return ("declare immediately", "never declare") on every graph in a sample. This guards the reward scale. |
| **Baselines** | BFS/DFS visit orders match hand-computed orders on small graphs; `RandomOrder` empirical mean matches `(|V| + 1) / 2` on a line graph. |
| **Leakage** | Shuffled-label control (13.5) drops to chance. Permuting node IDs within a graph does not change the agent's success rate. |
| **End-to-end smoke** | 20k-timestep training on 50 synthetic graphs finishes and improves over the random policy. Runs in CI in under a few minutes. |

---

## 17. Timeline and Work Split

Assumes about 8 weeks and two people. Adjust the calendar to the actual course deadline; the ordering and dependencies are what matter. The two tracks run in parallel from week 1, joined by the `GraphRecord` schema.

| Week | Track A — Data and program analysis | Track B — RL and evaluation |
|---|---|---|
| 1 | Repo setup. Download BigVul. Filtering script, attrition table, EDA. Install Joern and parse a first batch. | Agree `GraphRecord` schema. Synthetic CFG generator. Environment skeleton with masks. |
| 2 | Joern export script. Line-level CFG collapse. CFG unit tests. | Environment complete, `check_env` passes. Oracle and random policies. PPO learns on synthetic graphs. |
| 3 | Label alignment and verification. Multi-line statement handling. Graph stats. Spot-check 20 graphs. | BFS/DFS/RandomWalk/RandomOrder/LineOrder baselines. Evaluation harness and metrics. |
| 4 | Tier S and Tier L features. De-dupe and split. **Deliver processed dataset.** | First PPO runs on real graphs (Tier S, then Tier L). Trace recorder. |
| 5 | Heuristic scorer and supervised node classifier. Visualizer: static graph + source panes. | Reward and hyperparameter tuning. Curriculum if needed. Multi-seed training. |
| 6 | Visualizer: replay, live mode, compare mode. | Ablations (features, backtrack, shaping, declare budget). Masked DQN if on schedule. |
| 7 | Cross-CWE experiment support. README and pipeline reproducibility pass. | Freeze configuration. **Single test-set evaluation.** Statistics, figures. |
| 8 | Report writing, demo rehearsal, buffer. | Report writing, demo rehearsal, buffer. |

### Milestones

- **M1 (end of week 2):** PPO beats DFS on synthetic graphs. Joern produces line-level CFGs for a sample batch.
- **M2 (end of week 4):** processed BigVul dataset delivered; first end-to-end PPO run on real data.
- **M3 (end of week 6):** tuned agent, all baselines, ablations, working visualizer.
- **M4 (end of week 8):** final results, report, demo.

---

## 18. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Joern setup or parsing eats weeks | Medium | High | Phase 0 synthetic sandbox keeps Track B unblocked. Fallback: build a simpler statement-level CFG with `tree-sitter-c` (handle sequence, if, loops, switch, return, break/continue; treat goto conservatively). |
| Too few usable samples after filtering | Medium | High | Decision gate in 7.4: widen to the memory-safety CWE family. Report per-CWE results on the pooled set. |
| Agent does not beat baselines on real data | Medium | High | Tier L features; curriculum; shaping; multi-declare budget. Even a negative result is reportable if the ablations explain it (for example, Tier S ≈ chance, Tier L > DFS). |
| Label noise (changed lines ≠ vulnerable statement) | High | Medium | Text-verified alignment; drop non-code lines; report Top-k; discuss in limitations. Optional: manually validate 50 test samples and report agreement. |
| Positional shortcut (flaw lines cluster at a typical depth) | Medium | Medium | `LineOrder` baseline and the relative-position EDA expose it. Ablate the relative-position feature. |
| Data leakage via duplicates or node ordering | Medium | High | Commit-level split, body-hash assertion, canonical successor order, shuffled-label and node-permutation controls. |
| Reward mis-scaling yields a degenerate policy | High | Medium | Reward sanity test in Section 16; action-distribution logging; documented fixes in 12.3. |
| Overfitting to a few large projects (Linux, Chromium dominate BigVul) | Medium | Medium | Report per-project breakdown; optional cross-project split as an extra experiment. |
| DQN implementation takes too long | Medium | Low | It is explicitly optional (12.4). |

---

## 19. Stretch Goals

In rough priority order:

1. **GNN-encoded observations.** Run a small GNN over the whole CFG once per episode and append the current node's embedding to the observation. Tests whether global context helps the navigator.
2. **Tier E features** (CodeBERT statement embeddings).
3. **Data-flow edges.** Add reaching-definition edges from Joern's PDG as extra move options, turning the CFG into a CFG+DFG.
4. **Bidirectional movement.** Add predecessor slots so the agent can walk backwards from a suspicious sink to its source.
5. **Detection head.** Include non-vulnerable functions and a `DECLARE_NONE` action, turning localization into joint detection and localization.
6. **Additional datasets.** Evaluate zero-shot on a second dataset with statement-level labels (for example D2A or a Devign-derived subset with line labels).
7. **Attention-based neighbour selection.** Replace the fixed `K` slots with a pointer-style policy that scores an arbitrary number of successors.

---

## 20. Definition of Done

- [ ] `scripts/01`–`05` reproduce `data/processed/` from the raw BigVul CSV with one documented command sequence.
- [ ] Attrition table, graph statistics, and EDA plots are saved and referenced in the report.
- [ ] `CFGNavEnv` passes `check_env` and the full test suite in Section 16.
- [ ] Random Walk, BFS, DFS (required), plus RandomOrder, LineOrder, and HeuristicFirst (recommended) run through the shared evaluation harness.
- [ ] A PPO agent is trained with 5 seeds; best checkpoints selected on validation only.
- [ ] Test-set results report localization accuracy, success rate, average search steps (nodes inspected and actions), and cumulative reward, with 95% CIs, under both Protocol A and Protocol B.
- [ ] At least the Tier S vs. Tier L, backtrack, and shuffled-label ablations are complete.
- [ ] The Streamlit visualizer replays saved traces and steps a live checkpoint, with CFG and source panes in sync.
- [ ] README lets a third party install, train on a small subset, evaluate, and launch the visualizer.
- [ ] Report and demo are complete, including an honest limitations section.
