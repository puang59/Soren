#!/usr/bin/env python3
"""Generate the report figures from saved results.

Example:
    python scripts/make_figures.py

Reads ``experiments/test`` (the test-set evaluation), ``experiments/ablations`` and the
validation histories under ``runs/final``, and writes PNG and PDF files to
``experiments/figures``. Every number drawn comes from a file on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soren.eval.metrics import breakdown
from soren.eval.stats import bootstrap_ci, per_graph

# Categorical slots in fixed order (validated for colour-vision separation on this surface);
# baselines and references wear neutral ink so the agent is the one coloured series.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, MUTED, SURFACE, GRID = "#898781", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"

LABELS = {
    "ppo": "PPO agent",
    "dfs+heuristic": "DFS",
    "bfs+heuristic": "BFS",
    "line_order+heuristic": "Line order",
    "heuristic_first+heuristic": "Heuristic-first",
    "random_order+heuristic": "Random order",
    "random_walk+heuristic": "Random walk",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out / f"{name}{suffix}", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.png")


def success_with_declarations(frame: pd.DataFrame, summary: dict, out: Path) -> None:
    """Test success of every method that has to declare, with 95% intervals."""
    table = per_graph(frame[frame["method"].isin(LABELS)])
    rows = []
    for method, group in table.groupby("method"):
        mean, low, high = bootstrap_ci(group["success"].to_numpy())
        rows.append((LABELS[method], method == "ppo", mean, low, high))
    rows.sort(key=lambda row: row[2])
    names = [row[0] for row in rows]
    means = np.array([row[2] for row in rows])
    lows = means - np.array([row[3] for row in rows])
    highs = np.array([row[4] for row in rows]) - means
    colours = [BLUE if row[1] else NEUTRAL for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(names, means, height=0.55, color=colours)
    ax.errorbar(means, names, xerr=[lows, highs], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    chance = summary["ranking_top_k"]["random"]["top1"]
    ax.axvline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(chance, len(names) - 0.42, f" random guess {chance:.2f}", color=MUTED, va="bottom")
    for y, (mean, high) in enumerate(zip(means, highs, strict=True)):
        ax.text(mean + high + 0.008, y, f"{mean:.2f}", va="center", color=INK)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xlabel("share of test functions localized (one declaration); bars show 95% intervals")
    ax.set_title("Success when every method must declare")
    ax.grid(axis="y", visible=False)
    save(fig, out, "test_success")


def success_by_budget(frame: pd.DataFrame, summary: dict, out: Path) -> None:
    """Success against the number of declarations allowed, with ranking references."""
    budgets = [1, 3, 5]
    success = frame.groupby("method")["success"].mean()
    ranking = summary["ranking_top_k"]
    series = [
        ("PPO agent", BLUE, "o", [success[m] for m in ("ppo", "ppo_top3", "ppo_top5")]),
        ("Heuristic ranking", ORANGE, "s", [ranking["heuristic"][f"top{k}"] for k in budgets]),
        ("Classifier ranking", AQUA, "^", [ranking["classifier"][f"top{k}"] for k in budgets]),
        ("Random ranking", NEUTRAL, "D", [ranking["random"][f"top{k}"] for k in budgets]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, colour, marker, values in series:
        ax.plot(budgets, values, color=colour, linewidth=2, marker=marker, markersize=7, label=name)
    # Direct labels at the line ends, nudged apart where two lines finish together.
    ends = sorted((values[-1], name) for name, _, _, values in series)
    position = -1.0
    for value, name in ends:
        position = max(value, position + 0.04)
        ax.text(5.12, position, f"{name} {value:.2f}", va="center", color=INK)
    ax.set_xticks(budgets)
    ax.set_xlim(0.7, 7.6)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("declarations allowed per function")
    ax.set_ylabel("share of test functions localized")
    ax.set_title("Test success by number of declarations allowed")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "success_by_budget")


def success_by_size(frame: pd.DataFrame, out: Path) -> None:
    """Agent against the strongest declaring baseline, by graph size."""
    order = ["<=20", "21-50", "51-100", ">100"]
    table = breakdown(frame[frame["method"].isin(["ppo", "dfs+heuristic"])], "size_bucket")
    pivot = table.pivot(index="size_bucket", columns="method", values="success_rate").reindex(order)
    counts = (
        frame[frame["method"] == "dfs+heuristic"]
        .assign(
            size_bucket=lambda t: pd.cut(
                t["num_nodes"], [0, 20, 50, 100, 10_000], labels=order
            ).astype(str)
        )["size_bucket"]
        .value_counts()
    )
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.34
    for offset, method, colour, name in (
        (-width / 2 - 0.01, "ppo", BLUE, "PPO agent"),
        (width / 2 + 0.01, "dfs+heuristic", NEUTRAL, "DFS + heuristic stop"),
    ):
        values = pivot[method].to_numpy()
        ax.bar(x + offset, values, width=width, color=colour, label=name)
        for xi, value in zip(x + offset, values, strict=True):
            ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", color=INK)
    ax.set_xticks(x, [f"{bucket} nodes\n(n = {counts.get(bucket, 0)})" for bucket in order])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("share of test functions localized")
    ax.set_title("Success falls as functions grow")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, out, "success_by_size")


def learning_curves(runs: Path, out: Path) -> None:
    """Validation success during training, mean and spread over seeds."""
    histories = [
        pd.DataFrame(json.loads(path.read_text()))
        for path in sorted(runs.glob("final/base/seed*/val_history.json"))
    ]
    if not histories:
        print("no validation histories under runs/final; skipping learning curves")
        return
    steps = histories[0]["timesteps"].to_numpy()[: min(len(h) for h in histories)]
    values = np.stack([h["success_rate"].to_numpy()[: len(steps)] for h in histories])
    mean, std = values.mean(axis=0), values.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.fill_between(steps / 1e6, mean - std, mean + std, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(steps / 1e6, mean, color=BLUE, linewidth=2)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("training timesteps (millions)")
    ax.set_ylabel("validation success")
    ax.set_title(f"Validation success plateaus early (mean and spread, {len(histories)} seeds)")
    ax.grid(axis="x", visible=False)
    save(fig, out, "learning_curves")


def ablations(path: Path, out: Path) -> None:
    """Validation success of every single-declaration ablation arm."""
    names = {
        "tier_l": "Default (lexical features)",
        "tier_s": "Structural features only",
        "tier_e": "Plus statement embeddings",
        "no_rel_line": "No relative line position",
        "no_backtrack": "No backtracking",
        "no_revisit_penalty": "No revisit penalty",
        "shaping": "Potential-based shaping",
    }
    table = pd.read_csv(path)
    table = table[table["arm"].isin(names)].sort_values("success_rate")
    labels = [names[arm] for arm in table["arm"]]
    colours = [BLUE if arm == "tier_l" else NEUTRAL for arm in table["arm"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(labels, table["success_rate"], height=0.55, color=colours)
    ax.errorbar(
        table["success_rate"], labels, xerr=table["success_std"], fmt="none",
        ecolor=INK, elinewidth=1, capsize=2.5,
    )  # fmt: skip
    for y, (mean, std) in enumerate(zip(table["success_rate"], table["success_std"], strict=True)):
        ax.text(mean + std + 0.006, y, f"{mean:.2f}", va="center", color=INK)
    ax.axvline(0.136, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(0.136, len(labels) - 0.42, " random guess 0.14", color=MUTED, va="bottom")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.set_xlabel("validation success, one declaration (mean and spread over 3 seeds)")
    ax.set_title("Ablations: validation success, one declaration")
    ax.grid(axis="y", visible=False)
    save(fig, out, "ablations")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="experiments/figures")
    args = parser.parse_args(argv)
    experiments, out = Path(args.experiments), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    frame = pd.read_parquet(experiments / "test" / "results.parquet")
    summary = json.loads((experiments / "test" / "summary.json").read_text())
    success_with_declarations(frame, summary, out)
    success_by_budget(frame, summary, out)
    success_by_size(frame, out)
    learning_curves(Path(args.runs), out)
    ablations(experiments / "ablations" / "ablations.csv", out)


if __name__ == "__main__":
    main()
