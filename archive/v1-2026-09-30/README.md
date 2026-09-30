# Archive — v1 (Phase-0 implementation, 2026-09-30)

This directory contains the original Phase-0 implementation, preserved unchanged.
The active project is the Phase-1 rebuild at the repository root.

Known issues that motivated the rebuild (documented in the root DEMO.md):

- Heart Disease labels were used as-is, but in this Kaggle CSV `target = 1`
  means *no* disease, so reported precision/recall were "no-disease" metrics.
- Breast Cancer recall was benign recall (`1 = benign` in the CSV).
- MLP early stopping restored the last epoch, not the best one (shallow copy).
- No 5-fold CV for Heart Disease; MLP was never FHE-compiled.

To run it as a fallback (uses the space-free interpreter):

    cd "archive/v1-2026-09-30" && ~/aimlel-venv/bin/python app.py   # http://localhost:5001
