#!/usr/bin/env bash
# Launch the demo dashboard on http://localhost:5001
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="$HOME/aimlel-venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "⚠  FHE interpreter not found — running with the project venv."
    echo "   Encrypted inference will be disabled. Run tools/setup_fhe_env.sh first."
    PY="./venv/bin/python"
fi

echo "▶ Starting HealthGuard AI on http://localhost:5001 …"
exec "$PY" app.py
