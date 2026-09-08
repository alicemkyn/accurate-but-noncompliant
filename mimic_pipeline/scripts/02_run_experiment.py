#!/usr/bin/env python
"""STEP 2 — run the AuditNeSy suite (20 seeds, bootstrap CIs, negative controls).

    export AUDITNESY_OUT=./auditnesy_mimic_out
    python scripts/02_run_experiment.py

Reads the cached cohort from STEP 1, runs T1/T2/T3 + T2s probe + NC-A'..NC-D',
writes OUT_DIR/audit_summary_mimic.json and per_seed_results.json.

IMPORTANT (falsifiability): thresholds are frozen in config.yaml at tag
prereg-v1.0. Do not edit them after seeing results without logging a deviation.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.config import load_config
from auditnesy_mimic.experiment import run

if __name__ == "__main__":
    cfg = load_config()
    cohort_path = cfg.out_dir / "aki_cohort.parquet"
    if not cohort_path.exists():
        sys.exit(f"Cohort not found at {cohort_path}. Run scripts/01_build_cohort.py first.")
    df = pd.read_parquet(cohort_path)
    feat_cols = (cfg.out_dir / "feature_cols.txt").read_text().splitlines()
    print(f"[run] cohort {len(df)} rows, {df['subject_id'].nunique()} patients, "
          f"{cfg.experiment['seeds'].__len__()} seeds")
    run(cfg, df, feat_cols)
