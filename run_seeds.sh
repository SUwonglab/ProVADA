#!/usr/bin/env bash
# run_seeds.sh
# Run many seeds in parallel on a single GPU using xargs -P.

set -eo pipefail

############################################
# User parameters
############################################

CONFIG_FILE="inputs/tevp/tevp_deimmune_baseline.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

START_SEED="${START_SEED:-5001}"
END_SEED="${END_SEED:-5005}"      

MAX_JOBS="${MAX_JOBS:-5}"      # concurrent jobs (GPU-limited)
LOG_DIR="${LOG_DIR:-run_logs}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

############################################
# Setup
############################################

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file '$CONFIG_FILE' not found." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Using config:          $CONFIG_FILE"
echo "Python binary:         $PYTHON_BIN"
echo "GPU ID:                $GPU_ID"
echo "Seed range:            $START_SEED .. $END_SEED"
echo "Max concurrent jobs:   $MAX_JOBS"
echo "OMP_NUM_THREADS:       $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS:       $MKL_NUM_THREADS"
echo "Log directory:         $LOG_DIR"
echo

############################################
# Launch with xargs -P
############################################

START_GAP=20
RATE_LOCK="${LOG_DIR}/.start_rate.lock"
LAST_START="${LOG_DIR}/.last_start_ts"

seq "$START_SEED" "$END_SEED" | xargs -n1 -P "$MAX_JOBS" -I{} bash -c '
    set -euo pipefail

    seed="$1"
    seed_str=$(printf "%03d" "$seed")
    log_file="'"$LOG_DIR"'/seed_${seed_str}.log"

    # ---- Global start-time staggering (one start every ~20s) ----
    {
        flock -x 9

        now=$(date +%s)
        last=0
        if [[ -f "'"$LAST_START"'" ]]; then
            last=$(cat "'"$LAST_START"'" 2>/dev/null || echo 0)
        fi

        wait=$(( (last + '"$START_GAP"' ) - now ))
        if (( wait > 0 )); then
            echo "[INFO] Seed $seed waiting ${wait}s to stagger start..." >&2
            sleep "$wait"
        fi

        date +%s > "'"$LAST_START"'"
    } 9>"'"$RATE_LOCK"'"
    # ------------------------------------------------------------

    echo "[INFO] Launching seed $seed -> $log_file"

    CUDA_VISIBLE_DEVICES="'"$GPU_ID"'" \
    TQDM_MININTERVAL=100 TQDM_MAXINTERVAL=100 \
    "'"$PYTHON_BIN"'" run_provada.py \
        --config "'"$CONFIG_FILE"'" \
        --seed "$seed" \
        >"$log_file" 2>&1
' bash {}

