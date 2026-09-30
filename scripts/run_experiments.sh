#!/usr/bin/env bash
# Train + evaluate the three models with 5-fold CV and compile the FHE circuits.
# Writes everything the web app shows into results/.  Add --quick for a smoke run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/aimlel-venv/bin/python"
[ -x "$PY" ] || { echo "✗ $PY not found — run scripts/setup_fhe_env.sh first"; exit 1; }
exec "$PY" -u -m healthguard.experiment "$@"
