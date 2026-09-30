#!/usr/bin/env bash
#
# Concrete ML shells out to the system linker to build each FHE circuit, and
# that command is assembled without quoting. Because this project lives under
# "/Users/.../3rd year/AIML EL", the linker sees "-L /Users/.../3rd" and fails
# with:  ld: library 'ConcretelangRuntime' not found
#
# Fix: keep the interpreter itself on a path with no spaces. This script mirrors
# the project venv to ~/aimlel-venv, which is what run_demo.sh / run_pipeline.sh
# use. The project directory can stay exactly where it is.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_VENV="$PROJECT_DIR/venv"
DST_VENV="$HOME/aimlel-venv"

if [ ! -d "$SRC_VENV" ]; then
    echo "✗ No venv found at $SRC_VENV — create it first:"
    echo "    python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "▶ Mirroring $SRC_VENV → $DST_VENV (~1.5 GB, takes a minute) …"
rm -rf "$DST_VENV"
cp -a "$SRC_VENV" "$DST_VENV"

echo "▶ Verifying that FHE compilation works …"
"$DST_VENV/bin/python" - <<'PY'
import numpy as np, time
from concrete.ml.sklearn import LogisticRegression
X = np.random.randn(60, 5); y = (X[:, 0] > 0).astype(int)
m = LogisticRegression(n_bits=8, random_state=42); m.fit(X, y)
t0 = time.time(); m.compile(X)
print(f"  ✓ circuit compiled in {time.time()-t0:.2f}s")
t0 = time.time(); m.predict(X[:3], fhe="execute")
print(f"  ✓ real homomorphic execution OK ({(time.time()-t0)/3*1000:.0f} ms/sample)")
PY

echo
echo "✅ Done. Use this interpreter for anything that touches FHE:"
echo "     $DST_VENV/bin/python"
