"""Traversal visualizer: replay how a method walked a function's control flow graph.

Run with:
    streamlit run soren/viz/app.py -- --graphs data/synthetic/val.jsonl --traces runs/demo/traces

``--traces`` is a directory written by ``scripts/make_traces.py``: one sub-directory per
method, one JSON file per graph. The paths can also be set in the sidebar or through the
``SOREN_GRAPHS`` and ``SOREN_TRACES`` environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from soren.data.schema import GraphRecord, read_jsonl
from soren.viz.render import (
    action_labels,
    cfg_dot,
    outcome_text,
    replay_state,
    source_html,
    step_table,
)
from soren.viz.trace import Trace

NO_COMPARISON = "(none)"
LEGEND = (
    "Orange: current node. Blue: visited, darker with more visits. Bold outline: on the path "
    "stack. Green or red: declared, correct or wrong. Purple double outline: ground truth. "
    "Dashed edge: loop back edge."
)


def default_paths() -> tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--graphs", default=os.environ.get("SOREN_GRAPHS", "data/synthetic/val.jsonl")
    )
    parser.add_argument("--traces", default=os.environ.get("SOREN_TRACES", "runs/demo/traces"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.graphs, args.traces


@st.cache_data(show_spinner=False)
def load_graphs(path: str, modified: float) -> dict[str, GraphRecord]:
    return {graph.sample_id: graph for graph in read_jsonl(path)}


@st.cache_data(show_spinner=False)
def load_trace_index(directory: str, modified: float) -> pd.DataFrame:
    """One row per stored trace: method, graph and outcome."""
    rows = []
    for path in sorted(Path(directory).glob("*/*.json")):
        trace = Trace.load(path)
        rows.append(
            {
                "method": path.parent.name,
                "graph_id": trace.graph_id,
                "success": bool(trace.outcome.get("success")),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=["method", "graph_id", "success", "path"])


def sidebar(
    graphs: dict[str, GraphRecord], index: pd.DataFrame
) -> tuple[str, str, bool, str | None] | None:
    """Method and graph pickers with filters.

    Returns ``(method, graph_id, show_truth, compare_with)``; ``compare_with`` is a second
    method to show side by side, or ``None``.
    """
    st.sidebar.header("Episode")
    method = st.sidebar.selectbox("Method", sorted(index["method"].unique()), key="method")
    traces = index[index["method"] == method]

    sizes = [graphs[g].num_nodes for g in traces["graph_id"] if g in graphs]
    if not sizes:
        st.sidebar.warning("None of this method's traces match the loaded graphs.")
        return None
    cwes = sorted({graphs[g].cwe or "(none)" for g in traces["graph_id"] if g in graphs})
    chosen_cwes = st.sidebar.multiselect("CWE", cwes, default=cwes, key="cwes")
    low, high = min(sizes), max(sizes)
    size_range = (low, high)
    if low < high:
        size_range = st.sidebar.slider("Graph size (nodes)", low, high, (low, high), key="sizes")
    outcome = st.sidebar.radio(
        "Outcome", ["all", "success", "failure"], horizontal=True, key="outcome"
    )

    def keep(row: pd.Series) -> bool:
        graph = graphs.get(row["graph_id"])
        if graph is None or (graph.cwe or "(none)") not in chosen_cwes:
            return False
        if not size_range[0] <= graph.num_nodes <= size_range[1]:
            return False
        return outcome == "all" or row["success"] == (outcome == "success")

    matching = traces[traces.apply(keep, axis=1)]
    st.sidebar.caption(f"{len(matching)} of {len(traces)} episodes match")
    if matching.empty:
        return None
    graph_id = st.sidebar.selectbox(
        "Graph",
        matching["graph_id"].tolist(),
        format_func=lambda g: f"{g} · {graphs[g].num_nodes} nodes",
        key="graph",
    )
    show_truth = st.sidebar.toggle("Show ground truth", value=False, key="truth")

    others = index[(index["graph_id"] == graph_id) & (index["method"] != method)]
    compare_with = None
    if not others.empty:
        choice = st.sidebar.selectbox(
            "Compare with", [NO_COMPARISON, *sorted(others["method"])], key="compare"
        )
        compare_with = None if choice == NO_COMPARISON else choice
    return method, graph_id, show_truth, compare_with


def playback(total: int, episode_key: str) -> int:
    """Step slider with play/pause over ``total`` steps; returns the current step."""
    if st.session_state.get("episode") != episode_key:
        st.session_state["episode"] = episode_key
        st.session_state["step"] = 0
        st.session_state["playing"] = False
    if total == 0:
        return 0

    # A pending advance from the previous play tick is applied before the slider is built.
    if st.session_state.pop("advance", False):
        st.session_state["step"] = min(st.session_state["step"] + 1, total)

    controls = st.columns([1, 1, 1, 2, 6])
    if controls[0].button("⏮", help="Back to the start", key="first"):
        st.session_state.update(step=0, playing=False)
    if controls[1].button("◀", help="One step back", key="prev"):
        st.session_state.update(step=max(st.session_state["step"] - 1, 0), playing=False)
    if controls[2].button("▶", help="One step forward", key="next"):
        st.session_state.update(step=min(st.session_state["step"] + 1, total), playing=False)
    label = "Pause" if st.session_state["playing"] else "Play"
    if controls[3].button(label, key="play"):
        st.session_state["playing"] = not st.session_state["playing"]
        if st.session_state["playing"] and st.session_state["step"] >= total:
            st.session_state["step"] = 0
    speed = controls[4].select_slider(
        "Speed (steps per second)", options=[1, 2, 4, 8], value=2, key="speed"
    )
    st.slider("Step", 0, total, key="step")
    st.session_state["delay"] = 1.0 / speed
    return int(st.session_state["step"])


def progress_line(graph: GraphRecord, trace: Trace, step: int, show_truth: bool) -> None:
    """Outcome banner once the episode is over, otherwise where the replay stands."""
    total = len(trace.steps)
    if step >= total:
        report = st.success if trace.outcome.get("success") else st.error
        report(outcome_text(graph, trace))
    else:
        st.caption(
            f"Step {step} of {total}. Ground truth is {'shown' if show_truth else 'hidden'}."
        )


def policy_panel(graph: GraphRecord, trace: Trace, step: int) -> None:
    upcoming = trace.steps[step] if step < len(trace.steps) else None
    if upcoming is not None and upcoming.probs is not None:
        labels = action_labels(graph, upcoming.node, len(upcoming.probs))
        chart = pd.DataFrame({"action": labels, "probability": upcoming.probs})
        if upcoming.mask is not None:
            chart = chart[pd.Series(upcoming.mask)]  # hide actions that were not available
        st.caption(f"Policy at this step (value estimate {upcoming.value:.3f})")
        st.bar_chart(chart, x="action", y="probability", horizontal=True)
    elif upcoming is not None:
        st.caption("This method has no action probabilities to show.")


def single_view(graph: GraphRecord, trace: Trace, step: int, show_truth: bool) -> None:
    progress_line(graph, trace, step, show_truth)
    left, right = st.columns(2)
    with left:
        st.subheader("Control flow graph")
        st.graphviz_chart(cfg_dot(graph, trace, step, show_truth), width="stretch")
        st.caption(LEGEND)
    with right:
        st.subheader("Source")
        st.markdown(source_html(graph, trace, step, show_truth), unsafe_allow_html=True)

    st.subheader("Steps")
    table_column, policy_column = st.columns([3, 2])
    with table_column:
        st.dataframe(step_table(graph, trace, step), hide_index=True, width="stretch")
    with policy_column:
        policy_panel(graph, trace, step)


def compare_view(graph: GraphRecord, traces: list[Trace], step: int, show_truth: bool) -> None:
    """Two methods on the same graph, side by side, advancing together."""
    for column, trace in zip(st.columns(2), traces, strict=True):
        own_step = min(step, len(trace.steps))
        state = replay_state(trace, own_step)
        with column:
            st.subheader(trace.method)
            counts = st.columns(2)
            counts[0].metric("Nodes inspected", f"{len(state.visits)} of {graph.num_nodes}")
            counts[1].metric("Actions", f"{own_step} of {len(trace.steps)}")
            progress_line(graph, trace, own_step, show_truth)
            st.graphviz_chart(cfg_dot(graph, trace, own_step, show_truth), width="stretch")
            with st.expander("Source"):
                st.markdown(source_html(graph, trace, own_step, show_truth), unsafe_allow_html=True)
            with st.expander("Steps"):
                st.dataframe(step_table(graph, trace, own_step), hide_index=True, width="stretch")
                policy_panel(graph, trace, own_step)
    st.caption(LEGEND)


def main() -> None:
    st.set_page_config(page_title="Soren traversal visualizer", layout="wide")
    st.title("Soren traversal visualizer")

    graphs_default, traces_default = default_paths()
    st.sidebar.header("Data")
    graphs_path = st.sidebar.text_input("Graphs (JSONL)", graphs_default, key="graphs_path")
    traces_dir = st.sidebar.text_input("Traces directory", traces_default, key="traces_dir")
    if not Path(graphs_path).is_file():
        st.info(f"Graph file not found: `{graphs_path}`. Set the path in the sidebar.")
        return
    if not Path(traces_dir).is_dir():
        st.info(
            f"Traces directory not found: `{traces_dir}`. "
            "Create traces with `scripts/make_traces.py`."
        )
        return

    graphs = load_graphs(graphs_path, Path(graphs_path).stat().st_mtime)
    index = load_trace_index(traces_dir, Path(traces_dir).stat().st_mtime)
    if index.empty:
        st.info(f"No traces in `{traces_dir}`. Expected `<method>/<graph_id>.json` files.")
        return

    selection = sidebar(graphs, index)
    if selection is None:
        st.info("No episode matches the filters.")
        return
    method, graph_id, show_truth, compare_with = selection
    graph = graphs[graph_id]

    def load(name: str) -> Trace:
        row = index[(index["method"] == name) & (index["graph_id"] == graph_id)].iloc[0]
        return Trace.load(row["path"])

    trace = load(method)
    other = load(compare_with) if compare_with else None
    # In compare mode one control drives both episodes; the shorter one waits at its end.
    total = max(len(trace.steps), len(other.steps)) if other else len(trace.steps)
    step = playback(total, f"{method}/{compare_with}/{graph_id}")

    if other is not None:
        compare_view(graph, [trace, other], step, show_truth)
    else:
        single_view(graph, trace, step, show_truth)

    if st.session_state.get("playing"):
        if step >= total:
            st.session_state["playing"] = False
        else:
            time.sleep(st.session_state.get("delay", 0.5))
            st.session_state["advance"] = True
        st.rerun()


if __name__ == "__main__":
    main()
