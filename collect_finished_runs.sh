#!/usr/bin/env bash
# Collect completed deimmunization runs (reached iteration 99) into a dated folder and zip it.

set -euo pipefail

RESULTS_DIR="$(dirname "$0")/results"
DATE=$(date +%Y%m%d)
OUTPUT_DIR="$(dirname "$0")/fmc63_deimmunize_caliby_${DATE}"
MAX_ITER=99

# Read iteration by its CSV header, including quoted fields and either schema.
is_complete() {
    python3 - "$1" "$MAX_ITER" <<'PY'
import csv
import sys

target = int(sys.argv[2])
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    if "iteration" not in (reader.fieldnames or []):
        sys.exit(1)
    for row in reader:
        try:
            iteration = float(row.get("iteration", ""))
        except (TypeError, ValueError):
            continue
        if iteration == target:
            sys.exit(0)
sys.exit(1)
PY
}

echo "Scanning results in: $RESULTS_DIR"
echo "Output folder: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

found=0
for run_dir in "$RESULTS_DIR"/*/; do
    traj="$run_dir/trajectory.csv"
    if [[ ! -f "$traj" ]]; then
        continue
    fi
    if is_complete "$traj"; then
        run_name=$(basename "$run_dir")
        echo "  [DONE] $run_name"
        cp -R "${run_dir%/}" "$OUTPUT_DIR/"
        (( found++ )) || true
    else
        echo "  [SKIP] $(basename "$run_dir")"
    fi
done

if [[ $found -eq 0 ]]; then
    echo "No completed runs found."
    rmdir "$OUTPUT_DIR"
    exit 0
fi

echo ""
echo "Copied $found completed run(s) to $OUTPUT_DIR"

ZIP_FILE="${OUTPUT_DIR}.zip"
zip -r "$ZIP_FILE" "$OUTPUT_DIR"
echo "Created zip: $ZIP_FILE"
