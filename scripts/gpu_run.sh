#!/usr/bin/env bash
# One command for a rented multi-GPU machine: environment, GPU smoke test, E0b, E1, results archive.
# Safe to run again: every stage skips what is already finished. It keeps running if the terminal closes.
#
# Overrides: PER_GPU (cells per GPU, default 2), THREADS (CPU threads per cell, default 4), STAGE (default e1).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

if [ "${1:-}" != "--inner" ]; then
  setsid nohup bash "${BASH_SOURCE[0]}" --inner > "$LOGS/pipeline.log" 2>&1 < /dev/null &
  echo "Started in the background. Closing this terminal will not stop it."
  echo "Progress: tail -f $LOGS/pipeline.log"
  echo "Press Ctrl+C to stop watching; the run continues."
  sleep 2
  exec tail -f "$LOGS/pipeline.log"
fi

PER_GPU="${PER_GPU:-2}"
THREADS="${THREADS:-4}"
STAGE="${STAGE:-e1}"
MODEL="Qwen/Qwen2.5-0.5B"
VENV="$ROOT/.venv"
export HF_HOME="$ROOT/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

say() { echo "[$(date '+%H:%M:%S')] $*"; }
stop() { say "STOPPED: $*"; say "Send me this log: $LOGS/pipeline.log"; exit 1; }
cd "$ROOT" || stop "no repo"

# ---- environment. A private venv with the versions CI tested, whatever the template ships. ----
if [ ! -f "$VENV/.ready" ]; then
  say "Setting up the environment"
  python3 -m venv "$VENV" || { pip install virtualenv && python3 -m virtualenv "$VENV"; } || stop "cannot create a venv"
  "$VENV/bin/pip" install -q --upgrade pip || stop "pip upgrade failed"
  "$VENV/bin/pip" install -q "torch==2.15.0" "transformers==5.22.1" "datasets>=2.19" || stop "install failed"
  "$VENV/bin/pip" install -q -e ".[capability]" || stop "package install failed"
  touch "$VENV/.ready"
fi
PY="$VENV/bin/python"

GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')" || stop "torch does not import"
[ "$GPUS" -ge 1 ] || stop "torch sees no GPU"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h "$ROOT" | tail -1
say "$GPUS GPUs, $PER_GPU cells per GPU, $THREADS threads per cell, git $(git rev-parse --short HEAD)"

# ---- GPU smoke test: fp16, SDPA and the real tokenizer on a GPU, for the first time. ----
SMOKE="$ROOT/runs/smoke/smoke_Qwen__Qwen2.5-0.5B.json"
if ! grep -q '"passed": true, "question_sets"' "$SMOKE" 2>/dev/null; then
  say "GPU smoke test"
  CUDA_VISIBLE_DEVICES=0 "$PY" experiments/run_e0.py --config configs/e0.json --out runs/smoke --device cuda \
    --models "$MODEL" --smoke > "$LOGS/smoke.log" 2>&1 || { tail -30 "$LOGS/smoke.log"; stop "smoke test failed, nothing else was started"; }
  say "Smoke test passed"
else
  say "Smoke test already passed"
fi

# ---- E0b on the last GPU, alongside E1: one tuning run, then one confirmation of its winner. ----
run_e0b() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS"
  if [ ! -f runs/e0b/e0b_tune.json ]; then
    "$PY" experiments/run_e0b.py --config configs/e0b.json --out runs/e0b --device cuda > "$LOGS/e0b_tune.log" 2>&1 \
      || { say "E0b tuning failed; E1 is unaffected. See logs/e0b_tune.log"; return 1; }
  fi
  say "E0b tuning finished"
  if [ -f runs/e0b/winner.txt ] && [ ! -f runs/e0b/e0b_confirm.json ]; then
    "$PY" experiments/run_e0b.py --config configs/e0b.json --out runs/e0b --device cuda \
      --confirm "$(cat runs/e0b/winner.txt)" --tune-output runs/e0b/e0b_tune.json > "$LOGS/e0b_confirm.log" 2>&1 \
      || { say "E0b confirmation failed. See logs/e0b_confirm.log"; return 1; }
  fi
  say "E0b finished"
}

LAST=$((GPUS - 1))
run_e0b "$LAST" &
E0B_PID=$!

# ---- E1: every slot but the one E0b holds. ----
SLOTS=""
for gpu in $(seq 0 "$LAST"); do
  for _ in $(seq 1 "$PER_GPU"); do SLOTS="$SLOTS,$gpu"; done
done
SLOTS="${SLOTS#,}"
if [ "$GPUS" -gt 1 ]; then SLOTS="${SLOTS%,"$LAST"}"; fi

say "E1 starts on slots $SLOTS"
"$PY" scripts/schedule.py --config "configs/$STAGE.json" --out "runs/$STAGE" --logs "$LOGS/$STAGE" --slots "$SLOTS" --threads "$THREADS"
E1_CODE=$?
say "E1 scheduler exited with $E1_CODE. Waiting for E0b."
wait "$E0B_PID"

# ---- results: everything but checkpoints and the question cache. ----
"$PY" experiments/analyze.py --results "runs/$STAGE" --json "runs/$STAGE/analysis.json" > "runs/$STAGE/analysis.txt" 2>&1 \
  || say "analyze.py failed; the raw results are still in the archive"
"$VENV/bin/pip" freeze > runs/pip-freeze.txt
find runs logs -type f \( -name 'results.jsonl' -o -name '*.json' -o -name '*.txt' -o -name '*.log' \) \
  -not -path '*/checkpoints/*' -not -path '*/_cache/*' > runs/archive-list.txt
tar -czf results.tar.gz -T runs/archive-list.txt || stop "could not write the archive"

ls -la results.tar.gz
if [ "$E1_CODE" -eq 0 ]; then
  say "ALL DONE. Download $ROOT/results.tar.gz and keep the instance running until I've read it."
else
  say "FINISHED WITH FAILURES. Download $ROOT/results.tar.gz, and running this script again will retry the failed cells."
fi
