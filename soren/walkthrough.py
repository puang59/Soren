"""Walk one function through Soren and print every calculation along the way.

    python -m soren.walkthrough                        # a test function the agent localizes
    python -m soren.walkthrough --graph bigvul_048191  # a particular function
    python -m soren.walkthrough --part 5               # only part 5
    python -m soren.walkthrough --synthetic            # no dataset or checkpoint needed

Parts:

    1  the function                       6  the whole episode, step by step
    2  its control flow graph             7  the return
    3  the features of one statement      8  how PPO learns from this episode
    4  the observation vector             9  the evaluation statistics
    5  one decision of the network

Part 9 reads the per-episode results the final evaluation saved (``--results``) and shows how
the success rate, its confidence interval and the significance tests are computed from them.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from soren.agents.buffer import compute_gae
from soren.agents.policy import ActorCritic
from soren.agents.ppo import PPOConfig, adam_snapshot, explain_adam, explain_clip, ppo_loss
from soren.data.features import explain_node, feature_dim
from soren.data.graph_utils import bfs_depths, find_back_edges, strongly_connected_components
from soren.data.schema import NODE_KINDS, GraphRecord, read_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import (
    CONTEXT_DIM,
    SIZE_REF,
    STACK_CLIP,
    VISIT_CLIP,
    CFGNavEnv,
    EnvConfig,
)
from soren.env.rewards import RewardConfig, explain_reward, reward_terms
from soren.eval.stats import bootstrap_ci, mcnemar_success, per_graph, wilcoxon_nodes_inspected
from soren.mathlog import log, num
from soren.nn import Adam, clip_grad_norm

DEFAULT_GRAPHS = "data/processed/graphs_test.jsonl"
DEFAULT_CHECKPOINT = "runs/v2/final/base/seed1/best_model.npz"
DEFAULT_RESULTS = "experiments/test/results.parquet"


@dataclass
class Step:
    t: int
    node: int
    obs: np.ndarray
    mask: np.ndarray
    probs: np.ndarray
    value: float
    action: int
    action_name: str
    terms: list[tuple[str, float]]
    reward_text: str
    next_node: int


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    final_obs: np.ndarray | None = None
    terminated: bool = False

    @property
    def rewards(self) -> list[float]:
        return [sum(v for _, v in s.terms) for s in self.steps]


def run_episode(policy: ActorCritic, env: CFGNavEnv) -> Episode:
    """One deterministic episode, keeping everything the explanation needs."""
    obs, info = env.reset(options={"graph_index": 0})
    episode = Episode()
    done = False
    while not done:
        mask = env.action_masks()
        probs = policy.action_probs(obs, mask)
        value = policy.value(obs)
        node = env.node
        action, _ = policy.predict(obs, mask, deterministic=True)
        next_obs, _, terminated, truncated, info = env.step(action)
        event = env.last_event
        episode.steps.append(
            Step(
                t=len(episode.steps),
                node=node,
                obs=obs,
                mask=mask,
                probs=probs,
                value=value,
                action=action,
                action_name=info["action_name"],
                terms=reward_terms(event, env.reward_model.config),
                reward_text=explain_reward(event, env.reward_model.config),
                next_node=info["current_node"],
            )
        )
        obs, done = next_obs, terminated or truncated
        episode.terminated = terminated
    episode.info, episode.final_obs = info, obs
    return episode


def line(graph: GraphRecord, node: int) -> str:
    n = graph.nodes[node]
    return n.kind if n.kind in ("ENTRY", "EXIT") else f"line {n.line}"


def short(text: str, width: int = 48) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def detail_step(episode: Episode) -> Step:
    """The first step with a real choice: at least two valid actions."""
    return next((s for s in episode.steps if s.mask.sum() >= 2), episode.steps[0])


# ----------------------------------------------------------------------------- parts


def part_function(graph: GraphRecord) -> None:
    log.heading("1. The function")
    log.text(
        f"{graph.sample_id} · {graph.project or 'synthetic'} · {graph.cwe or 'no CWE'}"
        f" · {len(graph.source_lines)} lines"
    )
    log.write()
    for number, text in enumerate(graph.source_lines, start=1):
        log.write(f"   {number:>3} │ {text.rstrip()[:90]}")
    log.note(
        "",
        "The agent sees none of this text directly: it sees the graph built from it (part 2)",
        "and a vector of numbers per statement (part 3).",
    )


def part_graph(graph: GraphRecord) -> None:
    log.heading("2. Its control flow graph")
    log.text(
        "Joern parses the C code into a graph with one node per expression;",
        "soren/data/cfg_builder.py merges those into one node per source line (a statement)",
        "and keeps an edge u → v whenever control can pass from line u to line v.",
    )
    loops = set().union(*(c for c in strongly_connected_components(graph.num_nodes, graph.edges)
                          if len(c) > 1))  # fmt: skip
    rows = [
        [n.id, n.line if n.kind not in ("ENTRY", "EXIT") else "", n.kind, n.depth,
         "yes" if n.id in loops else "", short(n.code, 44)]
        for n in graph.nodes
    ]  # fmt: skip
    log.write()
    log.table(["node", "line", "kind", "depth", "loop", "code"], rows)
    log.write()
    edges = ", ".join(f"{u}→{v}" for u, v in graph.edges[:24])
    more = " …" if len(graph.edges) > 24 else ""
    log.text(f"{graph.num_nodes} nodes, {len(graph.edges)} edges: {edges}{more}")

    log.section("three graph searches (soren/data/graph_utils.py)")
    depths = bfs_depths(graph.num_nodes, graph.edges, graph.entry)
    log.formula(
        "depth by BFS",
        "depth(ENTRY) = 0;  depth(v) = depth(u) + 1 for the first u that reaches v",
        None,
        "deepest node at depth " + str(max(depths)),
    )
    back = find_back_edges(graph.num_nodes, graph.edges, graph.entry)
    log.formula(
        "back edges by DFS",
        "an edge u → v where v is still on the DFS path (grey) closes a loop",
        None,
        ", ".join(f"{u}→{v}" for u, v in back) or "none: the function has no loops",
    )
    components = [sorted(c) for c in strongly_connected_components(graph.num_nodes, graph.edges)]
    cyclic = [c for c in components if len(c) > 1]
    log.formula(
        "loops by Kosaraju",
        "nodes in a strongly connected component of size > 1 lie on a cycle",
        f"{len(components)} components; with a cycle: {cyclic or 'none'}",
    )
    matches = (
        depths == [n.depth for n in graph.nodes]
        and back == sorted(graph.back_edges)
        and loops == {n.id for n in graph.nodes if n.in_loop}
    )
    log.note(("✓ the same as" if matches else "✗ differs from") + " what the record stores")


def part_features(graph: GraphRecord, node: int) -> None:
    log.heading(f"3. The features of one statement: node {node} ({line(graph, node)})")
    log.text(f"code: {short(graph.nodes[node].code, 80)}")
    log.text(
        "Every statement becomes 30 numbers in [0, 1] (soren/data/features.py): 17 describe",
        "its place in the graph, 13 flag what the code does. No label is ever read here.",
    )
    log.write()
    rows = explain_node(graph, node)
    log.table(["feature", "value", "how it is computed"], [[n, v, w] for n, w, v in rows])
    log.note("(the 10 kind_* columns for other node kinds are 0 and not listed)")


def part_observation(graph: GraphRecord, env_config: EnvConfig, step: Step, env: CFGNavEnv):
    f = feature_dim(env_config.feature_tier)
    k = env_config.k
    slot = f + 3
    obs = step.obs
    where = line(graph, step.node)
    log.heading(f"4. The observation vector at step {step.t} (the agent is at {where})")
    log.text(
        f"The policy reads one vector of {len(obs)} numbers. It always has this length,",
        "whatever the size of the function, so one network handles every function.",
    )
    pos = f + 2 + k * slot
    segments = [
        ("current node features", 0, f),
        ("visits, declared flag", f, f + 2),
        (f"{k} successor slots × {slot}", f + 2, pos),
        ("backtrack target", pos, pos + 2 + len(NODE_KINDS)),
        ("episode context", pos + 2 + len(NODE_KINDS), len(obs)),
    ]
    log.write()
    log.table(
        ["segment", "positions", "size", "non-zero values"],
        [[name, f"{a}–{b - 1}", b - a, int(np.count_nonzero(obs[a:b]))] for name, a, b in segments],
    )
    visits = int(round(obs[f] * VISIT_CLIP))
    log.section("a few of the values")
    log.formula(
        "visits here", f"min(visits, {VISIT_CLIP}) / {VISIT_CLIP}", None, f"{obs[f]:.3f}"
        f"  ({visits} visit{'s' if visits != 1 else ''})"
    )  # fmt: skip
    successors = graph.successors(step.node)
    for i, succ in enumerate(successors[:k]):
        base = f + 2 + i * slot
        log.formula(
            f"slot {i} → node {succ}",
            "[valid, its 30 features, visited before, back edge]",
            f"valid = {obs[base]:.0f}, visited = {obs[base + 1 + f]:.0f}, "
            f"back edge = {obs[base + 2 + f]:.0f}  ({line(graph, succ)})",
        )
    if len(successors) < k:
        log.note(f"slots {len(successors)}–{k - 1} are all zero: this node has no more successors")
    c = len(obs) - CONTEXT_DIM
    n = graph.num_nodes
    log.formula("time used", "t / max_steps", f"{step.t} / {env.max_steps}", f"{obs[c]:.3f}")
    seen = round(obs[c + 1] * n)
    log.formula("explored", "unique nodes visited / |V|", f"{seen} / {n}", f"{obs[c + 1]:.3f}")
    log.formula("path depth", f"min(stack size, {STACK_CLIP}) / {STACK_CLIP}", None,
                f"{obs[c + 2]:.3f}")  # fmt: skip
    log.formula(
        "function size",
        f"min(log |V| / log {SIZE_REF}, 1)",
        f"log {n} / log {SIZE_REF} = {math.log(n):.3f} / {math.log(SIZE_REF):.3f}",
        f"{obs[c + 3]:.3f}",
    )
    log.formula("declarations left", "left / allowed", None, f"{obs[c + 4]:.3f}")


def part_decision(graph: GraphRecord, policy: ActorCritic, env: CFGNavEnv, step: Step) -> None:
    log.heading(f"5. One decision of the network at step {step.t}")
    x = step.obs.astype(np.float64)[None]
    actor, critic = policy.actor, policy.critic
    actor.forward(x)
    sizes = actor.sizes
    log.text(
        f"actor: {' → '.join(map(str, sizes))} with {actor.activation}; "
        f"{actor.num_params():,} weights, all trained by soren/agents/ppo.py"
    )
    log.write()
    pre = x
    for index, linear in enumerate(actor.linears):
        z = actor.activations[2 * index]
        last = index == len(actor.linears) - 1
        out = z if last else actor.activations[2 * index + 1]
        name = "logits" if last else f"hidden {index + 1}"
        formula = "z = h·W + b" if last else f"h{index + 1} = tanh(h{index}·W + b)"
        log.formula(
            f"layer {index + 1} ({name})",
            formula,
            f"[1×{pre.shape[1]}]·[{linear.W.shape[0]}×{linear.W.shape[1]}] + [{linear.b.size}]",
            _preview(out[0]),
        )
        pre = out

    logits = actor.activations[-1][0]
    valid = np.flatnonzero(step.mask)
    log.section("from logits to probabilities")
    log.table(
        ["action", "meaning", "logit z", "valid", "masked z"],
        [[a, env.action_name(a) + _target(graph, step.node, a, env), float(logits[a]),
          "yes" if step.mask[a] else "no", f"{logits[a]:+.3f}" if step.mask[a] else "−10⁸"]
         for a in range(len(logits))],
    )  # fmt: skip
    top = logits[valid].max()
    exps = np.exp(logits[valid] - top)
    total = exps.sum()
    log.formula(
        "softmax",
        "p_i = exp(z_i − max z) / Σ_j exp(z_j − max z)",
        f"Σ_j exp(z_j − {top:.3f}) = " + " + ".join(f"{e:.3f}" for e in exps) + f" = {total:.3f}",
    )
    for a, e in zip(valid, exps, strict=True):
        log.formula(
            f"  p({env.action_name(a)})", "", f"{e:.3f} / {total:.3f}", f"{step.probs[a]:.3f}"
        )
    log.formula(
        "chosen action",
        "a = argmax_i p_i   (evaluation; training samples a ~ p)",
        None,
        f"{step.action_name}{_target(graph, step.node, step.action, env)}",
    )
    value = critic.forward(x)[0, 0]
    log.formula(
        "critic",
        f"V(s) = {' → '.join(map(str, critic.sizes))} network",
        None,
        f"{value:+.4f}  (the return the agent expects from here on, in normalised units)",
    )


def _preview(values: np.ndarray, count: int = 6) -> str:
    shown = ", ".join(f"{v:+.3f}" for v in values[:count])
    return f"[{shown}, …]" if len(values) > count else f"[{shown}]"


def _target(graph: GraphRecord, node: int, action: int, env: CFGNavEnv) -> str:
    successors = graph.successors(node)
    if action < len(successors):
        return f" → node {successors[action]} ({line(graph, successors[action])})"
    return ""


def part_episode(graph: GraphRecord, episode: Episode) -> None:
    log.heading("6. The whole episode, step by step")
    log.text(
        "At every step: the agent is at a node, the network gives a probability to each valid",
        "action, the most probable one is taken, and the reward function scores it",
        "(soren/env/rewards.py).",
    )
    log.write()
    running = 0.0
    rows = []
    for s in episode.steps:
        running += sum(v for _, v in s.terms)
        rows.append(
            [s.t, line(graph, s.node), s.action_name, float(s.probs[s.action]), s.value,
             s.reward_text, f"{running:+.3f}"]
        )  # fmt: skip
    log.table(["t", "at", "action", "π(a|s)", "V(s)", "reward", "total"], rows)
    info = episode.info
    truth = ", ".join(line(graph, v) for v in graph.vuln_nodes) or "unknown"
    declared = info["declared_node"]
    log.write()
    log.text(
        f"ended: {info['end_reason']} after {info['steps']} actions, "
        f"{info['visited_unique']} nodes inspected"
    )
    log.text(
        f"declared: {line(graph, declared) if declared is not None else 'nothing'};"
        f" ground truth (hidden from the agent): {truth}"
    )


def part_return(episode: Episode, gamma: float) -> None:
    log.heading("7. The return")
    rewards = episode.rewards
    shown = rewards if len(rewards) <= 10 else [*rewards[:4], None, *rewards[-4:]]
    text = " + ".join("…" if r is None else num(r, 2) for r in shown)
    log.formula("return", "G = Σ_t r_t", text, f"{sum(rewards):+.3f}")
    discounted = sum(gamma**t * r for t, r in enumerate(rewards))
    log.formula(
        "discounted return",
        "G₀ = Σ_t γᵗ·r_t",
        f"γ = {gamma}: "
        + " + ".join(f"{gamma}^{t}×{num(r, 2)}" for t, r in enumerate(rewards[:3]))
        + (" + …" if len(rewards) > 3 else ""),
        f"{discounted:+.4f}",
    )  # fmt: skip
    log.note(
        "Discounting makes a reward now worth more than the same reward later, so shorter",
        "routes to the flaw score higher. This is the quantity PPO tries to maximise.",
    )


def part_learning(policy: ActorCritic, episode: Episode, cfg: PPOConfig) -> None:
    log.heading("8. How PPO learns from this episode")
    log.text(
        "Training does this for 8,192 steps at a time; here the same calculation runs on this",
        "one episode. Raw rewards are used so the numbers stay readable (training divides",
        "them by a running standard deviation first). Nothing is saved.",
    )
    steps = episode.steps
    rewards = np.array(episode.rewards)[:, None]
    values = np.array([s.value for s in steps])[:, None]
    dones = np.zeros_like(rewards, dtype=bool)
    dones[-1] = episode.terminated
    last = np.array([policy.value(episode.final_obs)])
    adv, returns = compute_gae(rewards, values, dones, last, cfg.gamma, cfg.gae_lambda, True)

    log.section("advantages for every step")
    rows = [[s.t, s.action_name, float(rewards[i, 0]), float(values[i, 0]), float(adv[i, 0]),
             float(returns[i, 0])] for i, s in enumerate(steps)]  # fmt: skip
    log.table(["t", "action", "r_t", "V(s_t)", "A_t", "R_t"], rows)
    log.note("A_t > 0: the action did better than the critic expected.")

    batch = {
        "obs": np.stack([s.obs for s in steps]),
        "masks": np.stack([s.mask for s in steps]),
        "actions": np.array([s.action for s in steps]),
        "advantages": adv[:, 0],
        "returns": returns[:, 0],
    }
    before = policy.log_probs(batch["obs"], batch["masks"])
    rows_index = np.arange(len(steps))
    batch["old_log_probs"] = before[rows_index, batch["actions"]]
    optimizer = Adam(policy.params, policy.grads, cfg.learning_rate, eps=cfg.adam_eps)

    ppo_loss(policy, batch, cfg.clip_range, cfg.ent_coef, cfg.vf_coef, explain="PPO loss")
    norm, scale = clip_grad_norm(policy.grads, cfg.max_grad_norm)
    explain_clip(norm, scale, cfg.max_grad_norm)
    snapshot = adam_snapshot(optimizer)
    optimizer.step()
    explain_adam(optimizer, snapshot, cfg.learning_rate)
    after_one = policy.log_probs(batch["obs"], batch["masks"])[rows_index, batch["actions"]]
    for _ in range(29):
        ppo_loss(policy, batch, cfg.clip_range, cfg.ent_coef, cfg.vf_coef)
        clip_grad_norm(policy.grads, cfg.max_grad_norm)
        optimizer.step()
    after = policy.log_probs(batch["obs"], batch["masks"])[rows_index, batch["actions"]]

    log.section("what 30 such updates did to the policy")
    old = np.exp(batch["old_log_probs"])
    normalised = adv[:, 0]
    if len(normalised) > 1:
        normalised = (normalised - normalised.mean()) / (normalised.std() + 1e-8)
    rows = []
    for i, s in enumerate(steps):
        ratio = math.exp(after[i] - batch["old_log_probs"][i])
        probs = [float(old[i]), math.exp(after_one[i]), math.exp(after[i])]
        choices = int(s.mask.sum())
        rows.append([s.t, s.action_name, choices, float(normalised[i]), *probs, f"{ratio:.3f}"])
    header = ["t", "action", "choices", "Â_t", "π before", "after 1", "after 30", "ratio r"]
    log.table(header, rows)
    log.note(
        "Â is the advantage after normalising within the batch, so it compares each action",
        "with the others in the batch: Â > 0 makes the action likelier, Â < 0 less likely.",
        "A step with a single valid action keeps π = 1: there is nothing to choose.",
        f"Once r leaves [{1 - cfg.clip_range:.1f}, {1 + cfg.clip_range:.1f}] in the direction "
        "Â asks, clipping removes that sample's gradient;",
        "r can still drift a little further because Adam keeps momentum from earlier steps.",
    )


def part_statistics(results_path: Path) -> None:
    log.heading("9. The evaluation statistics")
    if not results_path.is_file():
        log.text(f"{results_path} not found: run scripts/final_evaluation.py first.")
        return
    frame = pd.read_parquet(results_path)
    declaring = frame[frame["method"].str.contains(r"\+heuristic") | (frame["method"] == "ppo")]
    log.text(
        f"From {results_path}: {frame['graph_id'].nunique()} test functions, "
        f"{frame['method'].nunique()} methods.",
        "The unit is the function: a method with several seeds is first averaged per function",
        "(soren/eval/stats.py). The agent is compared with DFS that declares by the heuristic.",
    )
    ppo = per_graph(declaring)
    ppo = ppo[ppo["method"] == "ppo"]
    log.formula(
        "success rate",
        "S = (1/G)·Σ_g (share of seeds that localize g)",
        f"{ppo['success'].sum():.1f} / {len(ppo)}",
        f"{ppo['success'].mean():.4f}",
    )
    bootstrap_ci(ppo["success"].to_numpy(), explain="PPO success rate")
    mcnemar_success(declaring, "ppo", "dfs+heuristic", explain=True)
    wilcoxon_nodes_inspected(declaring, "ppo", "dfs+heuristic", explain=True)


# ------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default=DEFAULT_GRAPHS)
    parser.add_argument("--graph", help="sample id; by default a small function the agent solves")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="per-episode results (part 9)")
    parser.add_argument("--node", type=int, help="node whose features part 3 explains")
    parser.add_argument("--part", type=int, action="append", help="show only these parts")
    parser.add_argument("--synthetic", action="store_true", help="generated graphs")
    args = parser.parse_args(argv)
    log.enabled = True

    env_config, reward_config, cfg = EnvConfig(), RewardConfig(), PPOConfig()
    if args.synthetic or not Path(args.graphs).is_file():
        if not args.synthetic:
            print(f"{args.graphs} not found; using synthetic graphs", file=sys.stderr)
        graphs = generate_dataset(20, seed=7, cfg=SyntheticConfig(min_nodes=10, max_nodes=16))
    else:
        graphs = read_jsonl(args.graphs)

    probe = CFGNavEnv(graphs[:1], env_config, reward_config)
    if Path(args.checkpoint).is_file():
        load = lambda: ActorCritic.load(args.checkpoint)  # noqa: E731
    else:
        print(f"{args.checkpoint} not found; using an untrained network", file=sys.stderr)
        shape = (probe.observation_space.shape[0], probe.action_space.n)
        load = lambda: ActorCritic(*shape, seed=0)  # noqa: E731
    policy = load()

    by_id = {g.sample_id: g for g in graphs}
    if args.graph:
        if args.graph not in by_id:
            raise SystemExit(f"no graph {args.graph!r} in {args.graphs}")
        graph = by_id[args.graph]
        episode = run_episode(policy, CFGNavEnv([graph], env_config, reward_config))
    else:
        graph, episode = choose(graphs, policy, env_config, reward_config)
    env = CFGNavEnv([graph], env_config, reward_config)
    env.reset(options={"graph_index": 0})
    step = detail_step(episode)
    node = args.node if args.node is not None else _feature_node(graph, episode)

    parts = set(args.part or range(1, 10))
    if 1 in parts:
        part_function(graph)
    if 2 in parts:
        part_graph(graph)
    if 3 in parts:
        part_features(graph, node)
    if 4 in parts:
        part_observation(graph, env_config, step, env)
    if 5 in parts:
        part_decision(graph, policy, env, step)
    if 6 in parts:
        part_episode(graph, episode)
    if 7 in parts:
        part_return(episode, cfg.gamma)
    if 8 in parts:
        part_learning(load(), episode, cfg)  # a fresh copy: the update is not kept
    if 9 in parts:
        part_statistics(Path(args.results))
    log.write()


def choose(graphs, policy, env_config, reward_config) -> tuple[GraphRecord, Episode]:
    """A function that fits on screen and that the agent localizes in a few steps."""
    small = [g for g in graphs if 8 <= g.num_nodes <= 24 and len(g.source_lines) <= 32]
    fallback = None
    for graph in small or graphs:
        episode = run_episode(policy, CFGNavEnv([graph], env_config, reward_config))
        if fallback is None:
            fallback = (graph, episode)
        if episode.info["is_success"] and 3 <= len(episode.steps) <= 12:
            return graph, episode
    assert fallback is not None
    return fallback


def _feature_node(graph: GraphRecord, episode: Episode) -> int:
    declared = episode.info.get("declared_node")
    if declared is not None:
        return declared
    return next(n.id for n in graph.nodes if n.kind not in ("ENTRY", "EXIT"))


if __name__ == "__main__":
    main()
