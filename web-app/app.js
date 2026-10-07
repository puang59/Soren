// Soren web visualizer. Plain DOM and SVG, no dependencies: it reads window.SOREN_DATA
// (written by scripts/export_webapp.py) and replays each method's saved trace. Served by
// `python -m soren.serve` it can also send a pasted function to be analysed.

const data = window.SOREN_DATA;
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

const ui = {
  fn: $("function"), method: $("method"), compare: $("compare"), truth: $("truth"),
  first: $("first"), prev: $("prev"), next: $("next"), play: $("play"), speed: $("speed"),
  step: $("step"), stepLabel: $("step-label"), panels: $("panels"),
  paste: $("paste"), pasteOpen: $("paste-open"), pasteCode: $("paste-code"), pasteRun: $("paste-run"),
  pasteSample: $("paste-sample"), pasteClose: $("paste-close"), pasteNote: $("paste-note"),
  pasteStatus: $("paste-status"),
};

const state = { step: 0, timer: null };

// ---------------------------------------------------------------- replay

// Where the walk stands after `step` actions: position, visit counts, path stack, declarations.
function replay(trace, step) {
  const s = {
    current: trace.start, visits: new Map([[trace.start, 1]]), stack: [],
    declared: [], taken: new Set(), ret: 0,
  };
  for (const act of trace.steps.slice(0, step)) {
    s.ret += act.reward;
    if (act.action === "DECLARE") { s.declared.push(act.node); continue; }
    const moved = act.next !== act.node;
    if (act.action === "BACKTRACK") s.stack.pop();
    else if (act.action.startsWith("MOVE_") && moved) {
      s.stack.push(act.node);
      s.taken.add(`${act.node}>${act.next}`);
    }
    if (moved) s.visits.set(act.next, (s.visits.get(act.next) || 0) + 1);
    s.current = act.next;
  }
  return s;
}

// ---------------------------------------------------------------- graph layout

const NODE_W = 132, NODE_H = 24, GAP_X = 10, GAP_Y = 28, PAD = 12;

// Layered layout: a node's layer is its distance from the entry, ordered by source line.
function layout(episode) {
  const depth = (n) => (n.depth < 0 ? 0 : n.depth);
  const layers = new Map();
  for (const node of episode.nodes) {
    const key = depth(node);
    if (!layers.has(key)) layers.set(key, []);
    layers.get(key).push(node);
  }
  const keys = [...layers.keys()].sort((a, b) => a - b);
  const widest = Math.max(...keys.map((k) => layers.get(k).length));
  const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X;
  const pos = new Map();
  keys.forEach((key, row) => {
    const nodes = layers.get(key).sort((a, b) => a.line - b.line || a.id - b.id);
    const rowWidth = nodes.length * NODE_W + (nodes.length - 1) * GAP_X;
    nodes.forEach((node, i) => {
      pos.set(node.id, {
        x: (width - rowWidth) / 2 + i * (NODE_W + GAP_X),
        y: PAD + row * (NODE_H + GAP_Y),
      });
    });
  });
  return { pos, width, height: PAD * 2 + keys.length * NODE_H + (keys.length - 1) * GAP_Y };
}

function edgePath(a, b) {
  const x1 = a.x + NODE_W / 2, x2 = b.x + NODE_W / 2;
  if (b.y > a.y) {
    const y1 = a.y + NODE_H, y2 = b.y, mid = (y1 + y2) / 2;
    return `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`;
  }
  // Upward or sideways (a loop back, or a jump within a layer): swing out to the side.
  const side = x2 >= x1 ? 1 : -1;
  const sx = a.x + (side > 0 ? NODE_W : 0), tx = b.x + (side > 0 ? NODE_W : 0);
  const bend = Math.max(sx, tx) * (side > 0 ? 1 : 0) + (side > 0 ? 36 : -36) + (side > 0 ? 0 : Math.min(sx, tx));
  const y1 = a.y + NODE_H / 2, y2 = b.y + NODE_H / 2;
  return `M${sx},${y1} C${bend},${y1} ${bend},${y2} ${tx},${y2}`;
}

function el(name, attrs = {}, text) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function nodeLabel(node) {
  if (node.kind === "ENTRY" || node.kind === "EXIT") return node.kind;
  const text = `${node.line}: ${node.code.replace(/\s+/g, " ")}`;
  return text.length > 19 ? text.slice(0, 18) + "…" : text;
}

function buildGraph(svg, episode) {
  const { pos, width, height } = layout(episode);
  svg.replaceChildren();
  // Drawn at natural size and scaled down by CSS when wider than its pane.
  svg.setAttribute("viewBox", `-22 0 ${width + 44} ${height}`);
  svg.style.width = `${width + 44}px`;
  const defs = el("defs");
  for (const [id, colour] of [["arrow", "#9a9a9a"], ["arrow-taken", "#000"]]) {
    const marker = el("marker", { id, viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" });
    marker.append(el("path", { d: "M0,0 L8,4 L0,8 z", fill: colour }));
    defs.append(marker);
  }
  svg.append(defs);
  const back = new Set(episode.back_edges.map(([u, v]) => `${u}>${v}`));
  const edges = new Map();
  for (const [u, v] of episode.edges) {
    const key = `${u}>${v}`;
    const path = el("path", { d: edgePath(pos.get(u), pos.get(v)), class: back.has(key) ? "edge back" : "edge", "marker-end": "url(#arrow)" });
    edges.set(key, path);
    svg.append(path);
  }
  const nodes = new Map();
  for (const node of episode.nodes) {
    const p = pos.get(node.id);
    const group = el("g", { class: "node", transform: `translate(${p.x},${p.y})` });
    group.append(el("title", {}, node.code || node.kind));
    group.append(el("rect", { width: NODE_W, height: NODE_H }));
    group.append(el("text", { x: 7, y: 16 }, nodeLabel(node)));
    nodes.set(node.id, group);
    svg.append(group);
  }
  return { nodes, edges };
}

// ---------------------------------------------------------------- panels

function buildPanel(episode, methodKey) {
  const root = $("panel-template").content.firstElementChild.cloneNode(true);
  const trace = episode.traces[methodKey];
  root.querySelector("h2").textContent = data.methods[methodKey];
  const info = data.method_info[methodKey];
  root.querySelector(".rule").textContent = {
    policy: "Learned policy: chooses each move and decides for itself when to declare.",
    threshold: `Fixed rule, no learning: declares at the first statement whose suspicion score reaches ${info.threshold}. It is never told the answer.`,
    oracle: "Fixed rule, no learning: stopped automatically on reaching a vulnerable statement, so it cannot be wrong.",
  }[info.rule];
  const graph = buildGraph(root.querySelector("svg"), episode);

  const lineNodes = new Map();
  for (const node of episode.nodes) {
    if (node.kind !== "ENTRY" && node.kind !== "EXIT") lineNodes.set(node.id, node.line);
  }
  lineNodes.set(episode.entry, episode.nodes[episode.entry].line);
  const source = root.querySelector(".source");
  const lines = episode.source.map((text, i) => {
    const li = document.createElement("li");
    const no = document.createElement("span");
    no.className = "no";
    no.textContent = i + 1;
    const code = document.createElement("span");
    code.textContent = text.replace(/\t/g, "    ") || " ";
    li.append(no, code);
    source.append(li);
    return li;
  });
  return { root, episode, trace, graph, lines, lineNodes, info };
}

function renderPanel(panel, step, showTruth) {
  const { root, episode, trace, graph, lines, lineNodes, info } = panel;
  const at = Math.min(step, trace.steps.length);
  const done = at === trace.steps.length;
  const s = replay(trace, at);
  const vulnerable = new Set(episode.vulnerable);
  const declared = new Set(s.declared);
  const stack = new Set(s.stack);
  // A pasted function has no ground truth, so a declaration is neither right nor wrong.
  const verdict = (id) => (episode.unlabelled ? "declared" : vulnerable.has(id) ? "ok" : "bad");

  for (const [id, group] of graph.nodes) {
    const classes = ["node"];
    if (s.visits.has(id)) classes.push("visited");
    if (id === s.current && !declared.has(id)) classes.push("current");
    if (declared.has(id)) classes.push(verdict(id));
    if (stack.has(id)) classes.push("stack");
    if (showTruth && vulnerable.has(id)) classes.push("truth");
    group.setAttribute("class", classes.join(" "));
  }
  for (const [key, path] of graph.edges) {
    const taken = s.taken.has(key);
    path.classList.toggle("taken", taken);
    path.setAttribute("marker-end", taken ? "url(#arrow-taken)" : "url(#arrow)");
  }

  const mark = new Map();
  const set = (id, cls) => { if (lineNodes.has(id)) mark.set(lineNodes.get(id), cls); };
  for (const id of s.visits.keys()) set(id, "visited");
  if (!declared.has(s.current)) set(s.current, "current");
  for (const id of declared) set(id, verdict(id));
  const truthLines = new Set(showTruth ? episode.vulnerable.map((id) => lineNodes.get(id)) : []);
  lines.forEach((li, i) => {
    li.className = [mark.get(i + 1) || "", truthLines.has(i + 1) ? "truth" : ""].join(" ").trim();
  });
  const currentLine = lines[(lineNodes.get(s.current) || 1) - 1];
  if (currentLine && state.timer) currentLine.scrollIntoView({ block: "nearest" });

  const score = info.rule === "threshold" ? ` · suspicion score here ${episode.scores[s.current].toFixed(2)}` : "";
  // The return depends on whether the declaration was right, which is unknown for pasted code.
  const ret = episode.unlabelled ? "" : ` · return ${s.ret.toFixed(2)}`;
  root.querySelector(".status").textContent =
    `step ${at} of ${trace.steps.length} · ${s.visits.size} of ${episode.nodes.length} nodes inspected${ret}${score}`;

  const outcome = root.querySelector(".outcome");
  outcome.hidden = !done;
  if (done && episode.unlabelled) {
    const last = trace.steps[trace.steps.length - 1];
    const said = last && last.action === "DECLARE";
    outcome.className = `outcome ${said ? "declared" : "bad"}`;
    outcome.textContent = said
      ? `Declared line ${episode.nodes[last.node].line} after ${trace.steps.length} actions. There is no ground truth for pasted code, so this is the method's guess, not a verified finding.`
      : "Did not declare any statement.";
  } else if (done) {
    outcome.className = `outcome ${trace.success ? "ok" : "bad"}`;
    const reason = { correct: "", wrong_declare: " (declared the wrong statement)", timeout: " (ran out of steps)", exhausted: " (nothing passed its threshold)", dead_end: " (dead end)" }[trace.end_reason] || "";
    outcome.textContent = trace.success
      ? `✔ Found the vulnerable statement in ${trace.steps.length} actions`
      : `✘ Missed${reason}`;
  }

  const upcoming = done ? null : trace.steps[at];
  const policy = root.querySelector(".policy");
  policy.hidden = !(upcoming && upcoming.options);
  if (upcoming && upcoming.options) {
    const best = Math.max(...upcoming.options.map((o) => o.p));
    const list = root.querySelector(".options");
    list.replaceChildren(...[...upcoming.options].sort((a, b) => b.p - a.p).map((option) => {
      const li = document.createElement("li");
      if (option.p === best) li.className = "chosen";
      const bar = document.createElement("span");
      bar.className = "bar";
      bar.style.width = `${Math.max(option.p * 100, 0.5)}%`;
      const name = document.createElement("span");
      name.textContent = option.label;
      const pct = document.createElement("span");
      pct.textContent = `${Math.round(option.p * 100)}%`;
      li.append(name, pct, bar);
      return li;
    }));
    root.querySelector(".value").textContent = `Value estimate: ${upcoming.value.toFixed(2)}`;
  }

  const lineOf = (id) => {
    const node = episode.nodes[id];
    return node.kind === "ENTRY" || node.kind === "EXIT" ? node.kind.toLowerCase() : node.line;
  };
  root.querySelector("tbody").replaceChildren(...trace.steps.slice(0, at).map((act, i) => {
    const tr = document.createElement("tr");
    const unknown = episode.unlabelled && act.action === "DECLARE";
    const cells = [i + 1, lineOf(act.node), act.action.replace(/_\d+$/, "").toLowerCase(), act.action === "DECLARE" ? "" : lineOf(act.next), unknown ? "n/a" : act.reward.toFixed(2)];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }).reverse());
}

// ---------------------------------------------------------------- wiring

let panels = [];

function totalSteps() {
  return Math.max(...panels.map((p) => p.trace.steps.length));
}

// The CVE, the fixing commit and its diff: shown with the ground truth, never before.
function renderWhy(episode) {
  const why = $("why");
  ui.truth.disabled = Boolean(episode.unlabelled);
  why.hidden = !ui.truth.checked || ui.truth.disabled;
  if (why.hidden || why.dataset.id === episode.id) return;
  why.dataset.id = episode.id;

  const meta = why.querySelector(".why-meta");
  meta.replaceChildren();
  const link = (text, url) => {
    if (!url) return document.createTextNode(text);
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = text;
    return a;
  };
  const parts = [];
  if (episode.cve) parts.push(link(episode.cve, episode.cve_url));
  parts.push(document.createTextNode(`${episode.cwe}${episode.cwe_name ? `: ${episode.cwe_name}` : ""}`));
  if (episode.commit) parts.push(link(`fixing commit ${episode.commit.slice(0, 10)}`, episode.commit_url));
  parts.forEach((part, i) => {
    if (i) meta.append(" · ");
    meta.append(part);
  });

  const message = why.querySelector(".why-message");
  message.textContent = episode.commit_message
    ? `Commit message:\n${episode.commit_message}`
    : "The dataset has no commit message for this fix.";

  const sign = { removed: "−", added: "+", context: " ", gap: "" };
  why.querySelector(".diff").replaceChildren(...episode.fix.map((row) => {
    const li = document.createElement("li");
    li.className = row.kind;
    if (row.kind === "gap") { li.textContent = "⋯"; return li; }
    const no = document.createElement("span");
    no.className = "no";
    no.textContent = row.line ?? "";
    const mark = document.createElement("span");
    mark.className = "sign";
    mark.textContent = sign[row.kind];
    const code = document.createElement("span");
    code.textContent = row.text.replace(/\t/g, "    ") || " ";
    li.append(no, mark, code);
    return li;
  }));
}

function render() {
  if (panels.length) renderWhy(panels[0].episode);
  for (const panel of panels) renderPanel(panel, state.step, ui.truth.checked && !ui.truth.disabled);
  ui.step.value = state.step;
  ui.stepLabel.textContent = `${state.step} / ${totalSteps()}`;
}

function stop() {
  clearInterval(state.timer);
  state.timer = null;
  ui.play.textContent = "Play";
  ui.play.setAttribute("aria-pressed", "false");
}

function setStep(step) {
  state.step = Math.max(0, Math.min(step, totalSteps()));
  if (state.step === totalSteps()) stop();
  render();
}

function play() {
  if (state.timer) return stop();
  if (state.step >= totalSteps()) state.step = 0;
  ui.play.textContent = "Pause";
  ui.play.setAttribute("aria-pressed", "true");
  render();
  state.timer = setInterval(() => setStep(state.step + 1), Number(ui.speed.value));
}

function rebuild() {
  stop();
  const episode = data.episodes[ui.fn.selectedIndex];
  // Not every method has a trace for every function: an oracle stop needs a ground truth.
  for (const select of [ui.method, ui.compare]) {
    for (const o of select.options) o.disabled = Boolean(o.value) && !(o.value in episode.traces);
    if (select.selectedOptions[0].disabled) select.value = select === ui.method ? Object.keys(episode.traces)[0] : "";
  }
  const methods = [ui.method.value];
  if (ui.compare.value && ui.compare.value !== ui.method.value) methods.push(ui.compare.value);
  panels = methods.map((key) => buildPanel(episode, key));
  ui.panels.classList.toggle("two", panels.length === 2);
  ui.panels.replaceChildren(...panels.map((p) => p.root));
  state.step = 0;
  ui.step.max = totalSteps();
  render();
}

function option(value, label) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  return o;
}

function episodeOption(episode, first) {
  const verdict = episode.unlabelled ? "✎" : episode.traces[first].success ? "✔" : "✘";
  return option(episode.id, `${verdict} ${episode.id} · ${episode.project} · ${episode.nodes.length} nodes`);
}

// ---------------------------------------------------------------- your own function

const SAMPLE = `int copy_name(char *dst, const char *src, int len)
{
    char buf[64];
    int i;
    if (len < 0) {
        return -1;
    }
    for (i = 0; i < len; i++) {
        buf[i] = src[i];
    }
    memcpy(dst, buf, len);
    dst[len] = '\\0';
    return len;
}
`;

let live = false;

function pasteStatus(text, error = false) {
  ui.pasteStatus.textContent = text;
  ui.pasteStatus.classList.toggle("error", error);
}

function togglePaste(open) {
  ui.paste.hidden = !open;
  ui.pasteOpen.setAttribute("aria-expanded", String(open));
  if (open && live) ui.pasteCode.focus();
}

async function analyse() {
  const code = ui.pasteCode.value;
  if (!code.trim()) return pasteStatus("Paste a C or C++ function first.", true);
  ui.pasteRun.disabled = true;
  pasteStatus("Parsing with Joern and running the methods. This takes about ten seconds…");
  try {
    const response = await fetch("api/analyse", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code }),
    });
    const body = await response.json();
    if (!response.ok) return pasteStatus(body.error || "The analysis failed.", true);
    data.episodes.push(body);
    ui.fn.append(episodeOption(body));
    ui.fn.selectedIndex = data.episodes.length - 1;
    pasteStatus("");
    togglePaste(false);
    rebuild();
    ui.panels.scrollIntoView({ block: "start" });
  } catch {
    pasteStatus("Could not reach the server. Is `python -m soren.serve` still running?", true);
  } finally {
    ui.pasteRun.disabled = false;
  }
}

// Live analysis exists only when the page is served by soren.serve with Joern available.
async function initPaste() {
  let note = "This needs the local server. Start it with `python -m soren.serve` and open the address it prints.";
  try {
    const status = await (await fetch("api/status")).json();
    live = Boolean(status.live);
    note = live
      ? "Paste one complete C or C++ function. It is parsed into a control flow graph, then walked by the trained agent and the declaring baselines. There is no ground truth for your code: a declaration is the method's guess, and on the test set the agent's guess is right about one time in five."
      : "The server is running but Joern was not found, so pasted code cannot be parsed. Set JOERN_HOME or pass --joern-home.";
  } catch { /* opened from disk or a static host */ }
  ui.pasteNote.replaceChildren(...note.split("`").map((part, i) => {
    if (i % 2 === 0) return document.createTextNode(part);
    const code = document.createElement("code");
    code.textContent = part;
    return code;
  }));
  for (const control of [ui.pasteCode, ui.pasteRun, ui.pasteSample]) control.disabled = !live;
  ui.pasteOpen.addEventListener("click", () => togglePaste(ui.paste.hidden));
  ui.pasteClose.addEventListener("click", () => togglePaste(false));
  ui.pasteSample.addEventListener("click", () => { ui.pasteCode.value = SAMPLE; pasteStatus(""); });
  ui.pasteRun.addEventListener("click", analyse);
}

function init() {
  if (!data || !data.episodes.length) {
    ui.panels.textContent = "No data found. Run: python scripts/export_webapp.py";
    return;
  }
  const first = Object.keys(data.methods)[0];
  ui.fn.append(...data.episodes.map((e) => episodeOption(e, first)));
  for (const [key, name] of Object.entries(data.methods)) ui.method.append(option(key, name));
  ui.compare.append(option("", "(none)"));
  for (const [key, name] of Object.entries(data.methods)) ui.compare.append(option(key, name));

  ui.fn.addEventListener("change", rebuild);
  ui.method.addEventListener("change", rebuild);
  ui.compare.addEventListener("change", rebuild);
  ui.truth.addEventListener("change", render);
  ui.first.addEventListener("click", () => { stop(); setStep(0); });
  ui.prev.addEventListener("click", () => { stop(); setStep(state.step - 1); });
  ui.next.addEventListener("click", () => { stop(); setStep(state.step + 1); });
  ui.play.addEventListener("click", play);
  ui.speed.addEventListener("change", () => { if (state.timer) { stop(); play(); } });
  ui.step.addEventListener("input", () => { stop(); setStep(Number(ui.step.value)); });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("select, input, textarea")) return;
    if (event.key === "ArrowRight") { stop(); setStep(state.step + 1); }
    else if (event.key === "ArrowLeft") { stop(); setStep(state.step - 1); }
    else if (event.key === "Home") { stop(); setStep(0); }
    else if (event.key === " ") { event.preventDefault(); play(); }
  });

  // Start on a function the first method gets right, so the first thing shown is a success.
  const start = data.episodes.findIndex((e) => e.traces[first].success && e.traces[first].steps.length >= 4);
  ui.fn.selectedIndex = Math.max(start, 0);
  rebuild();
  initPaste();
}

init();
