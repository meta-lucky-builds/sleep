#!/usr/bin/env bash
# One command for a rented GPU machine: environment, GPU smoke test, then by default E1b, pick_setting and E3x.
# Safe to run again: every stage skips what is already finished. It keeps running if the terminal closes.
#
# Overrides: PER_GPU (cells per GPU, default from the smallest GPU's VRAM), THREADS (CPU threads per cell, default from
# the CPU count), STAGE (default e1b+e3x, the chain; a config name such as e1 runs that one stage alone).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

if [ "${1:-}" != "--inner" ]; then
  setsid nohup bash "${BASH_SOURCE[0]}" --inner >> "$LOGS/pipeline.log" 2>&1 < /dev/null &
  echo "Started in the background. Closing this terminal will not stop it."
  echo "Progress: tail -f $LOGS/pipeline.log"
  echo "Press Ctrl+C to stop watching; the run continues."
  sleep 2
  exec tail -f "$LOGS/pipeline.log"
fi

STAGE="${STAGE:-e1b+e3x}"
MODEL="Qwen/Qwen2.5-0.5B"
VENV="$ROOT/.venv"
E0B_RESULT="results/e0b/e0b_confirm.json"
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

# ---- A real allocation and matmul on the GPU: is_available alone can pass on a wheel the driver cannot run. ----
cuda_works() {
  "$PY" -c '
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda, "torch.cuda.is_available()", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.ones(64, 64, device="cuda:0")
assert float((x @ x).sum()) > 0, "matmul gave a wrong result"
print("allocation and matmul on cuda:0 passed", flush=True)
'
}
if ! cuda_works; then
  say "This torch cannot use the GPU, likely a wheel newer than the driver. Trying torch 2.14.1 from the cu130 index."
  "$VENV/bin/pip" install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130 || stop "the fallback torch 2.14.1 (cu130) did not install"
  say "Installed torch 2.14.1 (cu130). CI tested 2.15.0; each result records the versions used."
  cuda_works || stop "CUDA does not work with torch 2.15.0 or with 2.14.1 (cu130). Check nvidia-smi's driver and CUDA version, then send this log."
fi

GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')" || stop "torch does not import"
[ "$GPUS" -ge 1 ] || stop "torch sees no GPU"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h "$ROOT" | tail -1

# ---- cells per GPU and threads per cell: computed, and the environment wins. ----
if [ -n "${PER_GPU:-}" ]; then
  say "PER_GPU=$PER_GPU, from the environment"
else
  VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)"
  [ -n "$VRAM_MIB" ] || stop "nvidia-smi reports no GPU memory; set PER_GPU"
  VRAM_GB=$(((VRAM_MIB + 512) / 1024)) # Rounded: a 16 GB card reports a little under 16384 MiB.
  if [ "$VRAM_GB" -lt 16 ]; then PER_GPU=2
  elif [ "$VRAM_GB" -lt 32 ]; then PER_GPU=3
  elif [ "$VRAM_GB" -lt 64 ]; then PER_GPU=5
  else PER_GPU=6
  fi
  say "PER_GPU=$PER_GPU, computed: the smallest GPU has $VRAM_GB GB (under 16: 2, up to 31: 3, 32-63: 5, 64 or more: 6)"
fi
TOTAL_SLOTS=$((GPUS * PER_GPU))
if [ -n "${THREADS:-}" ]; then
  say "THREADS=$THREADS, from the environment"
else
  CPUS="$(nproc)"
  THREADS=$((CPUS / TOTAL_SLOTS))
  if [ "$THREADS" -lt 2 ]; then THREADS=2; fi
  say "THREADS=$THREADS, computed: max(2, $CPUS CPUs / $TOTAL_SLOTS slots)"
fi
say "Stage $STAGE, $GPUS GPUs, $PER_GPU cells per GPU, $THREADS threads per cell, git $(git rev-parse --short HEAD)"

# ---- archives: everything but checkpoints and the question cache. Written aside and moved, so one is always whole. ----
write_archive() {
  local name="$1" list="runs/.$1.list"
  mkdir -p runs
  find runs logs -type f \( -name 'results.jsonl' -o -name '*.json' -o -name '*.txt' -o -name '*.log' \) \
    -not -path '*/checkpoints/*' -not -path '*/_cache/*' > "$list"
  # Logs grow while tar reads them. That is exit code 1, and is expected; 2 is a real failure.
  tar -czf "$name.tar.gz.tmp" --warning=no-file-changed --warning=no-file-shrank --ignore-failed-read -T "$list"
  if [ "$?" -gt 1 ]; then rm -f "$name.tar.gz.tmp"; return 1; fi
  mv "$name.tar.gz.tmp" "$name.tar.gz"
}

( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed "s/^/$(date '+%H:%M:%S'), /" >> "$LOGS/gpu_util.log"
    sleep 60
  done ) &
GPU_LOG_PID=$!
( while true; do
    sleep 600
    write_archive partial || echo "[$(date '+%H:%M:%S')] could not refresh partial.tar.gz"
  done ) &
PARTIAL_PID=$!
trap 'kill "$GPU_LOG_PID" "$PARTIAL_PID" 2> /dev/null' EXIT

# Results go to the results-live branch every 10 minutes if GH_TOKEN is set.
if [ -n "${GH_TOKEN:-}" ] && ! pgrep -f push_results.sh > /dev/null; then
  nohup bash scripts/push_results.sh >> "$LOGS/push.log" 2>&1 < /dev/null &
  say "Pushing results to the results-live branch every 10 minutes"
fi

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

# ---- E0b on the last GPU, alongside the stage: one tuning run, then one confirmation of its winner. ----
run_e0b() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS"
  if [ ! -f runs/e0b/e0b_tune.json ]; then
    "$PY" experiments/run_e0b.py --config configs/e0b.json --out runs/e0b --device cuda > "$LOGS/e0b_tune.log" 2>&1 \
      || { say "E0b tuning failed; the stage is unaffected. See logs/e0b_tune.log"; return 1; }
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
ALL_SLOTS="" # Round-robin, 0,1,...,N,0,1,...: fewer cells than slots then spread over every GPU.
for _ in $(seq 1 "$PER_GPU"); do
  for gpu in $(seq 0 "$LAST"); do ALL_SLOTS="$ALL_SLOTS,$gpu"; done
done
ALL_SLOTS="${ALL_SLOTS#,}"
SLOTS="$ALL_SLOTS"

# A2.3 allows one test, and it has been run if its result is committed. A fresh machine must not run it again.
E0B_PID=""
if [ -f "$E0B_RESULT" ] || git cat-file -e "HEAD:$E0B_RESULT" 2> /dev/null; then
  say "E0b skipped: $E0B_RESULT is committed, and A2.3 allows one test. Every slot goes to the stage."
else
  run_e0b "$LAST" &
  E0B_PID=$!
  # Every slot but the one E0b holds.
  if [ "$GPUS" -gt 1 ]; then SLOTS="${SLOTS%,"$LAST"}"; fi
fi

# ---- A stage: run its cells, analyse, archive. Skipped if a previous run finished it. ----
mkdir -p runs
"$VENV/bin/pip" freeze > runs/pip-freeze.txt 2> /dev/null || true

run_stage() {
  local name="$1" config="$2" code=0
  if [ -f "runs/.done-$name" ]; then
    say "$name already finished; skipping"
    return 0
  fi
  say "$name starts on slots $SLOTS"
  "$PY" scripts/schedule.py --config "$config" --out "runs/$name" --logs "$LOGS/$name" --slots "$SLOTS" --threads "$THREADS"
  code=$?
  say "$name scheduler exited with $code"
  "$PY" experiments/analyze.py --results "runs/$name" --json "runs/$name/analysis.json" > "runs/$name/analysis.txt" 2>&1 \
    || say "analyze.py failed; the raw results are still in the archive"
  write_archive results || say "could not write results.tar.gz"
  if [ "$code" -eq 0 ]; then touch "runs/.done-$name"; fi
  return "$code"
}

RUN_CODE=0
if [ "$STAGE" = "e1b+e3x" ]; then
  run_stage e1b configs/e1b.json || RUN_CODE=$?
  # E1b's failures alone do not stop the chain: pick_setting does, if any cell it expects is missing.
  E3X_CONFIG="runs/e3x.config.json"
  PICKED=1
  if [ -f "$E3X_CONFIG" ]; then
    say "Keeping $E3X_CONFIG from the earlier pick, so that E3x's cells still match it"
  elif "$PY" scripts/pick_setting.py --results runs/e1b --config configs/e1b.json \
      --template configs/e3x.template.json --out "$E3X_CONFIG" 2>&1 | tee runs/pick_setting.txt; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    say "pick_setting chose E3x's setting"
  else
    PICKED=0
    RUN_CODE=1
    say "pick_setting failed, so E3x was not started. See above."
  fi
  if [ "$PICKED" -eq 1 ]; then
    # The same question set, already built: no need to spend the first cell's time on it again.
    mkdir -p runs/e3x/_cache
    cp -n runs/e1b/_cache/capability-*.json runs/e3x/_cache/ 2> /dev/null || true
    run_stage e3x "$E3X_CONFIG" || RUN_CODE=$?
  fi
else
  run_stage "$STAGE" "configs/$STAGE.json" || RUN_CODE=$?
fi
if [ -n "$E0B_PID" ]; then
  say "Waiting for E0b."
  wait "$E0B_PID"
fi

write_archive results || stop "could not write the archive"
ls -la results.tar.gz
if [ "$RUN_CODE" -eq 0 ]; then
  say "ALL DONE. Download $ROOT/results.tar.gz and keep the instance running until I've read it."
else
  say "FINISHED WITH FAILURES. Download $ROOT/results.tar.gz, and running this script again will retry the failed cells."
fi

# ---- With a token, wait for the next stage: configs/next_stage.txt on the working branch names a config. ----
[ -n "${GH_TOKEN:-}" ] || exit 0
say "Watching for the next stage. Billing continues until you destroy this instance."
while true; do
  sleep 300
  git fetch -q --depth 1 origin refs/heads/feat/sleep-framework 2>/dev/null || continue
  NEXT="$(git show FETCH_HEAD:configs/next_stage.txt 2>/dev/null | tr -d '[:space:]')"
  [ -n "$NEXT" ] && [ ! -f "runs/.done-$NEXT" ] || continue
  git checkout -q --detach FETCH_HEAD || continue
  say "Stage $NEXT at git $(git rev-parse --short HEAD)"
  if "$PY" scripts/schedule.py --config "configs/$NEXT.json" --out "runs/$NEXT" --logs "$LOGS/$NEXT" --slots "$ALL_SLOTS" --threads "$THREADS"; then
    say "Stage $NEXT finished"
  else
    say "Stage $NEXT finished with failures"
  fi
  touch "runs/.done-$NEXT"
done
