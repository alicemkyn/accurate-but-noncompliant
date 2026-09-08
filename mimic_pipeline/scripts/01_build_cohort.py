#!/usr/bin/env python
"""STEP 1 — build the AKI cohort and cache it to OUT_DIR (gitignored).

    export MIMIC_DIR=/path/to/mimiciv/hosp
    export AUDITNESY_OUT=./auditnesy_mimic_out
    python scripts/01_build_cohort.py

Writes: OUT_DIR/aki_cohort.parquet  (patient-level; NEVER commit -- PhysioNet DUA).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.cohort import build_cohort
from auditnesy_mimic.config import load_config
from auditnesy_mimic.io_utils import ensure_out

if __name__ == "__main__":
    cfg = load_config()
    out = ensure_out(cfg.out_dir)
    df, feat_cols = build_cohort(cfg)
    path = out / "aki_cohort.parquet"
    df.to_parquet(path, index=False)
    (out / "feature_cols.txt").write_text("\n".join(feat_cols))
    print(f"[done] cohort -> {path}  ({len(df)} rows, features: {feat_cols})")
