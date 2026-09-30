#!/usr/bin/env bash
# Concrete ML calls the system linker without quoting its arguments, so FHE
# compilation fails when the interpreter lives under a path containing spaces
# (this project is in ".../3rd year/AIML EL"). This mirrors the project venv to
# ~/aimlel-venv, a space-free path, and checks that a circuit compiles.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$HOME/aimlel-venv"
[ -d "$PROJECT/venv" ] || { echo "✗ $PROJECT/venv missing: python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt"; exit 1; }
echo "▶ Mirroring venv → $DST"
rm -rf "$DST" && cp -a "$PROJECT/venv" "$DST"
"$DST/bin/python" -m pip install -q pytest
"$DST/bin/python" - <<'PY'
import numpy as np
from concrete.ml.sklearn import LogisticRegression
X = np.random.randn(60, 5); y = (X[:, 0] > 0).astype(int)
m = LogisticRegression(n_bits=8).fit(X, y); m.compile(X)
m.predict(X[:2], fhe="execute")
print("✓ FHE compile + encrypted execution OK")
PY
