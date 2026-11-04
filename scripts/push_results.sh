#!/usr/bin/env bash
# Every 10 minutes, force-push results (no checkpoints, no question cache) to the `results-live` branch.
# Does nothing without GH_TOKEN. The token is read by git's credential helper, so it stays out of URLs and logs.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${GH_TOKEN:-}" ] || exit 0
REMOTE="https://github.com/meta-lucky-builds/sleep.git"
WORK="$ROOT/.results-repo"
PY="$ROOT/.venv/bin/python"

push_once() {
  cd "$ROOT" || return 1
  for dir in runs/e[0-9]*; do
    [ -d "$dir" ] || continue
    "$PY" experiments/analyze.py --results "$dir" --json "$dir/analysis.json" > "$dir/analysis.txt" 2>&1 || true
  done
  rm -rf "$WORK"
  mkdir -p "$WORK"
  find runs logs -type f \( -name 'results.jsonl' -o -name '*.json' -o -name '*.txt' -o -name '*.log' \) \
    -not -path '*/checkpoints/*' -not -path '*/_cache/*' -size -50M -print0 | xargs -0 -r cp --parents -t "$WORK"
  cd "$WORK" || return 1
  git init -q
  git add -A
  git -c user.name="sleep-gpu" -c user.email="sleep-gpu@users.noreply.github.com" commit -q -m "results $(date -u '+%Y-%m-%d %H:%M')" || return 1
  git -c credential.helper= \
    -c 'credential.helper=!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f' \
    push -q -f "$REMOTE" HEAD:refs/heads/results-live > /dev/null 2>&1
}

while true; do
  if push_once; then echo "[$(date '+%H:%M:%S')] results pushed"; else echo "[$(date '+%H:%M:%S')] results push FAILED (check the token)"; fi
  sleep 600
done
