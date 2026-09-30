#!/usr/bin/env bash
# Re-run the full training + FHE benchmark pipeline (~3 minutes)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="$HOME/aimlel-venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "⚠  FHE interpreter not found — FHE benchmarks will be skipped."
    echo "   Run tools/setup_fhe_env.sh first."
    PY="./venv/bin/python"
fi

exec "$PY" main.py
