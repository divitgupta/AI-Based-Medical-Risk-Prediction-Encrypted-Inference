#!/usr/bin/env bash
# Train + evaluate the three models with 5-fold CV, compile all three to FHE
# with the Phase-2 cost studies and build the Elastic Learner, on each
# dataset the web app can switch between:
#   cleveland         our 303 patients                      → results/
#   base1025_patient  base paper's file, split by patient   → results/base1025_patient/
#   base1025_row      base paper's file, split by row       → results/base1025_row/
# then summarise the base paper comparison (results/base_paper.json).
# Add --quick for a smoke run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/aimlel-venv/bin/python"
[ -x "$PY" ] || { echo "✗ $PY not found — run scripts/setup_fhe_env.sh first"; exit 1; }
for DS in cleveland base1025_patient base1025_row; do
  "$PY" -u -m healthguard.experiment --dataset "$DS" "$@"
  "$PY" -u -m healthguard.elastic --dataset "$DS"
done
"$PY" -u -m healthguard.base_paper
