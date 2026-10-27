#!/usr/bin/env bash
# Train + evaluate the three models with 5-fold CV, compile all three to FHE
# with the Phase-2 cost studies, build the Elastic Learner (stacking) and its
# leaderboard, then rerun the pipeline on the base paper's own dataset. Writes everything the web app shows into results/.
# Add --quick for a smoke run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/aimlel-venv/bin/python"
[ -x "$PY" ] || { echo "✗ $PY not found — run scripts/setup_fhe_env.sh first"; exit 1; }
"$PY" -u -m healthguard.experiment "$@"
"$PY" -u -m healthguard.elastic
"$PY" -u -m healthguard.base_paper "$@"
