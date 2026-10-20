#!/usr/bin/env bash
# run_seeds.sh
# Run configs × trajectories with a global seed, keeping MAX_JOBS parallel runs.

set -euo pipefail

############################################
# User parameters (override via env)
############################################

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

# Where your configs live
CONFIG_DIR="${CONFIG_DIR:-deimmunization_pipeline/inputs/fmc63}"

# Default config list (matches your screenshot)
CONFIG_FILES=(
  "${CONFIG_DIR}/fmc63_deimmunize_caliby.yaml"
)

# How many trajectories per config
TRAJ_PER_CONFIG="${TRAJ_PER_CONFIG:-20}"

# GPU-limited parallelism
MAX_JOBS="${MAX_JOBS:-4}"

# Optional: set to reproduce runs exactly
# If empty, we generate one randomly and derive all trajectory seeds from it.
GLOBAL_SEED="${GLOBAL_SEED:-}"

# Logs
LOG_ROOT="${LOG_ROOT:-run_logs}"
RUN_TAG="${RUN_TAG:-deimmune_fmc63_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}}"

# Start staggering to avoid all jobs initializing at once
START_GAP="${START_GAP:-20}"  # seconds

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

############################################
# Setup / validation
############################################

# Validate config files exist
for cfg in "${CONFIG_FILES[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Error: config file '$cfg' not found." >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

# Pick a global seed if not provided
if [[ -z "${GLOBAL_SEED}" ]]; then
  GLOBAL_SEED="$("$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.randbelow(2**31-1) + 1)
PY
)"
fi

echo "Run tag:              $RUN_TAG"
echo "Log directory:        $LOG_DIR"
echo "Python binary:        $PYTHON_BIN"
echo "GPU ID:               $GPU_ID"
echo "Configs:              ${#CONFIG_FILES[@]}"
printf "  - %s\n" "${CONFIG_FILES[@]}"
echo "Traj per config:      $TRAJ_PER_CONFIG"
echo "Max concurrent jobs:  $MAX_JOBS"
echo "GLOBAL_SEED:          $GLOBAL_SEED"
echo "START_GAP (sec):      $START_GAP"
echo "OMP_NUM_THREADS:      $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS:      $MKL_NUM_THREADS"
echo

echo "$GLOBAL_SEED" > "${LOG_DIR}/GLOBAL_SEED.txt"

############################################
# Build a manifest: config, traj_idx, seed, log_file
############################################

MANIFEST="${LOG_DIR}/manifest.tsv"
"$PYTHON_BIN" - "$GLOBAL_SEED" "$TRAJ_PER_CONFIG" "$LOG_DIR" "${CONFIG_FILES[@]}" > "$MANIFEST" <<'PY'
import os, sys, random
from pathlib import Path

base_seed = int(sys.argv[1])
n_traj    = int(sys.argv[2])
log_dir   = sys.argv[3]
configs   = sys.argv[4:]

rng = random.Random(base_seed)

print("config\ttraj_idx\tseed\tlog_file")
for cfg in configs:
  tag = Path(cfg).stem
  cfg_log_dir = Path(log_dir) / tag
  cfg_log_dir.mkdir(parents=True, exist_ok=True)

  for t in range(1, n_traj + 1):
    seed = rng.randrange(1, 2**31 - 1)
    log_file = cfg_log_dir / f"traj_{t:03d}_seed_{seed}.log"
    print(f"{cfg}\t{t}\t{seed}\t{log_file}")
PY

echo "Wrote manifest: $MANIFEST"
echo

############################################
# Launch with xargs -P (global concurrency cap)
############################################

RATE_LOCK="${LOG_DIR}/.start_rate.lock"
LAST_START="${LOG_DIR}/.last_start_ts"

# Skip header, pass 4 fields into each worker: config traj seed log_file
tail -n +2 "$MANIFEST" | xargs -r -n4 -P "$MAX_JOBS" bash -c '
  set -euo pipefail
  cfg="$1"
  traj="$2"
  seed="$3"
  log_file="$4"

  # ---- Global start-time staggering (one start every ~START_GAP seconds) ----
  {
    flock -x 9

    now=$(date +%s)
    last=0
    if [[ -f "'"$LAST_START"'" ]]; then
      last=$(cat "'"$LAST_START"'" 2>/dev/null || echo 0)
    fi

    wait=$(( (last + '"$START_GAP"' ) - now ))
    if (( wait > 0 )); then
      echo "[INFO] ($traj) waiting ${wait}s to stagger start..." >&2
      sleep "$wait"
    fi

    date +%s > "'"$LAST_START"'"
  } 9>"'"$RATE_LOCK"'"
  # ------------------------------------------------------------------------

  mkdir -p "$(dirname "$log_file")"

  echo "[INFO] Launching cfg=$(basename "$cfg") traj=$traj seed=$seed -> $log_file"

  CUDA_VISIBLE_DEVICES="'"$GPU_ID"'" \
  TQDM_MININTERVAL=100 TQDM_MAXINTERVAL=100 \
  "'"$PYTHON_BIN"'" run_provada.py \
    --config "$cfg" \
    --seed "$seed" \
    >"$log_file" 2>&1
' bash

echo
echo "All jobs dispatched. Logs + manifest are in: $LOG_DIR"
