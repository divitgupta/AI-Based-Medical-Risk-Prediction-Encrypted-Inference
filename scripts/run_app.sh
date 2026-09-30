#!/usr/bin/env bash
# Serve the dashboard on http://localhost:5001 (loads results/, never trains).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/aimlel-venv/bin/python"
[ -x "$PY" ] || { echo "✗ $PY not found — run scripts/setup_fhe_env.sh first"; exit 1; }
[ -f results/metrics.json ] || { echo "✗ No results yet — run scripts/run_experiments.sh first"; exit 1; }
exec "$PY" -u -m webapp.server
