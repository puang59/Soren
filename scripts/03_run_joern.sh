#!/usr/bin/env bash
# Parse every source batch with Joern and export its control flow graphs.
#
#   scripts/03_run_joern.sh [SRC_DIR] [OUT_DIR]
#
# SRC_DIR holds batch_XXXX directories written by scripts/02_write_sources.py (default
# data/interim/src). For each batch this writes OUT_DIR/batch_XXXX.jsonl (default
# data/interim/joern), one JSON object per function.
#
# Batches whose output already exists are skipped, so an interrupted run can be restarted.
# A batch that fails is recorded in OUT_DIR/failed_batches.txt and the run carries on.
#
# Environment:
#   JOERN_HOME  directory holding joern and joern-parse (default .tools/joern-cli, else PATH)
#   JOBS        batches processed in parallel (default 2; each one starts a JVM)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-$ROOT/data/interim/src}"
OUT_DIR="${2:-$ROOT/data/interim/joern}"
JOBS="${JOBS:-2}"
SCRIPT="$ROOT/joern/export_cfg.sc"

if [ -n "${JOERN_HOME:-}" ]; then
  :
elif [ -x "$ROOT/.tools/joern-cli/joern" ]; then
  JOERN_HOME="$ROOT/.tools/joern-cli"
elif command -v joern >/dev/null 2>&1; then
  JOERN_HOME="$(dirname "$(command -v joern)")"
else
  echo "joern not found: set JOERN_HOME or put joern on PATH" >&2
  exit 1
fi

[ -d "$SRC_DIR" ] || { echo "source directory not found: $SRC_DIR" >&2; exit 1; }
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
mkdir -p "$OUT_DIR/logs"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
: > "$OUT_DIR/failed_batches.txt"

run_batch() {
  local batch_dir="$1" name out work
  name="$(basename "$batch_dir")"
  out="$OUT_DIR/$name.jsonl"
  if [ -s "$out" ]; then
    echo "$name: already exported, skipping"
    return 0
  fi
  # Joern keeps a workspace in the current directory, so every batch gets its own.
  work="$(mktemp -d "$OUT_DIR/.work_$name.XXXXXX")"
  if (
    cd "$work" &&
      "$JOERN_HOME/joern-parse" "$batch_dir" --language c -o "$work/cpg.bin" &&
      "$JOERN_HOME/joern" --script "$SCRIPT" --param cpgFile="$work/cpg.bin" \
        --param outFile="$work/out.jsonl"
  ) >"$OUT_DIR/logs/$name.log" 2>&1 && [ -s "$work/out.jsonl" ]; then
    mv "$work/out.jsonl" "$out"
    echo "$name: $(wc -l <"$out" | tr -d ' ') functions"
  else
    echo "$name" >>"$OUT_DIR/failed_batches.txt"
    echo "$name: FAILED, see $OUT_DIR/logs/$name.log" >&2
  fi
  rm -rf "$work"
}
export -f run_batch
export OUT_DIR JOERN_HOME SCRIPT

find "$SRC_DIR" -maxdepth 1 -type d -name 'batch_*' | sort |
  xargs -P "$JOBS" -I{} bash -c 'run_batch "$@"' _ {}

failed="$(wc -l <"$OUT_DIR/failed_batches.txt" | tr -d ' ')"
total="$(cat "$OUT_DIR"/batch_*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
echo "exported $total functions to $OUT_DIR; $failed batch(es) failed"
[ "$failed" -eq 0 ]
