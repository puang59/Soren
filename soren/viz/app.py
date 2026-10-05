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
from soren.viz.render import action_labels, cfg_dot, outcome_text, source_html, step_table
from soren.viz.trace import Trace


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


def sidebar(graphs: dict[str, GraphRecord], index: pd.DataFrame) -> tuple[str, str, bool] | None:
    """Method and graph pickers with filters; returns ``(method, graph_id, show_truth)``."""
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
    return method, graph_id, show_truth


def playback(trace: Trace, episode_key: str) -> int:
    """Step slider with play/pause; returns the current step."""
    total = len(trace.steps)
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
    method, graph_id, show_truth = selection
    graph = graphs[graph_id]
    row = index[(index["method"] == method) & (index["graph_id"] == graph_id)].iloc[0]
    trace = Trace.load(row["path"])

    step = playback(trace, f"{method}/{graph_id}")
    total = len(trace.steps)
    if step >= total:
        report = st.success if trace.outcome.get("success") else st.error
        report(outcome_text(graph, trace))
    else:
        st.caption(
            f"Step {step} of {total}. Ground truth is {'shown' if show_truth else 'hidden'}."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Control flow graph")
        st.graphviz_chart(cfg_dot(graph, trace, step, show_truth), width="stretch")
        st.caption(
            "Orange: current node. Blue: visited, darker with more visits. Bold outline: on the "
            "path stack. Green or red: declared, correct or wrong. Purple double outline: ground "
            "truth. Dashed edge: loop back edge."
        )
    with right:
        st.subheader("Source")
        st.markdown(source_html(graph, trace, step, show_truth), unsafe_allow_html=True)

    st.subheader("Steps")
    table_column, policy_column = st.columns([3, 2])
    with table_column:
        st.dataframe(step_table(graph, trace, step), hide_index=True, width="stretch")
    with policy_column:
        upcoming = trace.steps[step] if step < total else None
        if upcoming is not None and upcoming.probs is not None:
            labels = action_labels(graph, upcoming.node, len(upcoming.probs))
            chart = pd.DataFrame({"action": labels, "probability": upcoming.probs})
            if upcoming.mask is not None:
                chart = chart[pd.Series(upcoming.mask)]  # hide actions that were not available
            st.caption(f"Policy at this step (value estimate {upcoming.value:.3f})")
            st.bar_chart(chart, x="action", y="probability", horizontal=True)
        elif upcoming is not None:
            st.caption("This method has no action probabilities to show.")

    if st.session_state.get("playing"):
        if step >= total:
            st.session_state["playing"] = False
        else:
            time.sleep(st.session_state.get("delay", 0.5))
            st.session_state["advance"] = True
        st.rerun()


if __name__ == "__main__":
    main()
