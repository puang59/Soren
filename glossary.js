// Hover cards for the technical notes. Reads the glossary embedded by build.py, underlines the
// first use of each term in every paragraph, list item, table cell and formula, and shows its
// explanation on hover, keyboard focus or tap.
(() => {
  const entries = JSON.parse(document.getElementById("glossary").textContent);
  const content = document.getElementById("content");
  const squash = (text) => text.replace(/\s+/g, " ");

  const exact = new Map(), loose = new Map(), patterns = [];
  for (const entry of entries) {
    for (const raw of entry.terms) {
      const term = squash(raw);
      (entry.cs ? exact : loose).set(entry.cs ? term : term.toLowerCase(), entry);
      patterns.push(term);
    }
  }
  patterns.sort((a, b) => b.length - a.length);
  const source = patterns
    .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/ /g, "\\s+"))
    .join("|");
  const finder = new RegExp(`(?<![\\p{L}\\p{N}_])(?:${source})(?![\\p{L}\\p{N}_])`, "giu");

  const SKIP = "a, h1, h2, nav, .mermaid, .diagram, .term, .code-lang";
  const BLOCK = "p, li, td, th, pre, h3, blockquote";
  const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
  const texts = [];
  while (walker.nextNode()) texts.push(walker.currentNode);

  const seen = new Map();
  for (const node of texts) {
    const parent = node.parentElement;
    if (!parent || parent.closest(SKIP)) continue;
    const pre = parent.closest("pre");
    // Highlighted code keeps its own colours; plain blocks are the formulas.
    if (pre && pre.querySelector(".code-lang")) continue;
    const block = parent.closest(BLOCK) || parent;
    if (!seen.has(block)) seen.set(block, new Set());
    const used = seen.get(block);

    const text = node.nodeValue;
    const pieces = [];
    let last = 0;
    for (const match of text.matchAll(finder)) {
      const key = squash(match[0]);
      const entry = exact.get(key) || loose.get(key.toLowerCase());
      if (!entry || used.has(entry) || (entry.formula && !pre)) continue;
      used.add(entry);
      if (match.index > last) pieces.push(text.slice(last, match.index));
      const span = document.createElement("span");
      span.className = "term";
      span.tabIndex = 0;
      span.textContent = match[0];
      span.entry = entry;
      pieces.push(span);
      last = match.index + match[0].length;
    }
    if (!pieces.length) continue;
    if (last < text.length) pieces.push(text.slice(last));
    node.replaceWith(...pieces);
  }

  const card = document.createElement("div");
  card.id = "tip";
  card.setAttribute("role", "tooltip");
  card.hidden = true;
  document.body.append(card);
  let shown = null;

  function show(term) {
    shown = term;
    const title = document.createElement("strong");
    title.textContent = term.entry.title;
    card.replaceChildren(title, ...term.entry.tip.split("\n").map((line) => {
      const p = document.createElement("p");
      p.textContent = line;
      return p;
    }));
    card.hidden = false;
    term.setAttribute("aria-describedby", "tip");
    const box = term.getBoundingClientRect();
    const gap = 8, width = card.offsetWidth, height = card.offsetHeight;
    const left = Math.max(gap, Math.min(box.left, window.innerWidth - width - gap));
    const below = box.bottom + gap;
    const top = below + height > window.innerHeight && box.top - gap - height > 0 ? box.top - gap - height : below;
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  }

  function hide() {
    if (shown) shown.removeAttribute("aria-describedby");
    shown = null;
    card.hidden = true;
  }

  const termOf = (event) => (event.target instanceof Element ? event.target.closest(".term") : null);
  content.addEventListener("mouseover", (event) => { const term = termOf(event); if (term && term !== shown) show(term); });
  content.addEventListener("mouseout", (event) => { if (termOf(event) && !termOf({ target: event.relatedTarget })) hide(); });
  content.addEventListener("focusin", (event) => { const term = termOf(event); if (term) show(term); });
  content.addEventListener("focusout", hide);
  // Tap toggles on touch screens, where there is no hover.
  document.addEventListener("click", (event) => {
    const term = termOf(event);
    if (!term) return hide();
    if (window.matchMedia("(hover: none)").matches && term === shown) hide(); else show(term);
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") hide(); });
  window.addEventListener("scroll", hide, { passive: true });
})();
