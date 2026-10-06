#!/usr/bin/env bash
# Regenerate snippets.jsonl from src/ with Joern. Run after changing a snippet or
# joern/export_cfg.sc. Needs JOERN_HOME, or Joern under .tools/joern-cli.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
JOERN_HOME="${JOERN_HOME:-$ROOT/.tools/joern-cli}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
"$JOERN_HOME/joern-parse" "$HERE/src" --language c -o "$WORK/cpg.bin" >/dev/null 2>&1
"$JOERN_HOME/joern" --script "$ROOT/joern/export_cfg.sc" --param cpgFile="$WORK/cpg.bin" \
  --param outFile="$WORK/out.jsonl" >/dev/null 2>&1
# Sort by file name so the fixture is stable across runs.
python3 -c "
import json, sys
rows = sorted((json.loads(l) for l in open('$WORK/out.jsonl')), key=lambda r: (r['file'], r['line']))
open('$HERE/snippets.jsonl', 'w').write(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
print(len(rows), 'methods')
"
