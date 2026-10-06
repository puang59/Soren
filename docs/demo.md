# Demo script (about five minutes)

The demo runs offline from saved traces: it needs no training, no Joern and no network.

## Before the demo

Open the web visualizer; it is a static page with the episodes already exported:

```bash
open web-app/index.html
```

As a fallback that needs no browser, the same walk-through works in a terminal with
`python -m soren.demo --step` and `python -m soren.demo --failure --step`.

Pick two functions in the sidebar ahead of time, both under about 30 nodes so the graph fits
on screen, and note their ids:

- **A success:** in the Function list, `✔` marks the functions the agent gets right.
- **A failure:** ideally one where the agent declares a statement with a dangerous call that
  is not the flaw.

With the exported checkpoint (`runs/final/base/seed1`), functions that fit these descriptions
include `bigvul_005074` and `bigvul_048191` (successes) and `bigvul_015735` and
`bigvul_031552` (failures). Look at them before the demo and keep the two that read best.

Keep "Show ground truth" off, and have `experiments/figures/test_success.png` open in
another window.

## Script

**1. The problem (30 seconds).** Show the source pane of the success function. "This function
has a known vulnerability. Detectors say *that* it is vulnerable. We want to know *where*,
and we treat finding it as walking the control flow graph."

**2. A baseline (45 seconds).** Method "Depth-first search". Press Play. "Depth-first search follows the
first branch as far as it goes, then backs up. It has no idea what it is looking for; under
the oracle stop it simply halts when it stands on the flaw." Point at the node count.

**3. The agent, answer hidden (60 seconds).** Method "PPO agent", same function. Step through with
the ▶ button. At each step point to "What the agent is weighing": "These are the agent's options:
the successors, backtrack, or declare. It sees only the current statement and one step
ahead." Stop when it declares.

**4. Reveal (20 seconds).** Switch on "Show ground truth". The declared node turns green and
the purple outline lands on it.

**5. Side by side (40 seconds).** Set "Compare with" to "Depth-first search". Drag the step slider. "Same
function, same controls: the agent commits after a handful of statements, DFS is still
exploring."

**6. A failure (60 seconds).** Switch to the failure function, ground truth off. Step to the
declaration, then reveal. "It declared this call, which looks dangerous, but the fix changed
a different line. This is the typical failure: a statement can look risky on its own and be
fine, and the flawed one can look ordinary. Whether it is a flaw depends on what happened
earlier, which the agent cannot see."

**7. The numbers (45 seconds).** Show `test_success.png`. "On 127 held-out functions the agent
finds the flaw 21% of the time with one guess; a random guess is 12.5%. Reading the function
in order and stopping at the first suspicious statement gets 28%. So the agent learned
something, but not more than a simple rule. Our classifier experiments show why: the
per-statement features, even with CodeBERT embeddings, top out around 23%."

**8. Close (20 seconds).** "The pipeline and the environment work, and on synthetic graphs
the agent reaches 98%. On real code the bottleneck is what each statement's features can say.
The next step is giving statements context: data flow, or a graph encoder."

## If something goes wrong

- **The page is blank or says "No data found":** run `python scripts/export_webapp.py`, or
  serve the folder with `python -m http.server -d web-app 8000`.
- **No browser at all:** use the terminal demo, `python -m soren.demo --step`.
- **A graph is too large to read:** pick a function with fewer nodes; the count is in the list.
- **No time for the live walk-through:** show steps 4, 6 and 7 only.
