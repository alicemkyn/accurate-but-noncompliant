#!/usr/bin/env python
"""Run the existing AuditNeSy harness on the eICU cohort (reuse, don't reimplement).
Uses hospital-based groups as the distribution-shift axis (a genuine multi-center
shift, unlike MIMIC's weak era shift).

    export EICU_OUT=./eicu_out           # dir with aki_cohort.parquet from build_eicu_cohort.py
    python run_eicu.py

Imports the mimic_pipeline package read-only; writes eICU results to EICU_OUT.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(__file__).resolve().parent.parent / "mimic_pipeline"
sys.path.insert(0, str(MAIN))
from auditnesy_mimic.config import load_config      # noqa: E402
from auditnesy_mimic import experiment              # noqa: E402

OUT = Path(os.environ.get("EICU_OUT", "./eicu_out"))


def main():
    cfg = load_config(MAIN / "config.yaml")
    cfg.raw["data"]["out_dir"] = str(OUT)
    cfg.raw["dataset_name"] = "eICU-CRD (AKI)"
    df = pd.read_parquet(OUT / "aki_cohort.parquet")
    feat_cols = (OUT / "feature_cols.txt").read_text().splitlines()

    # multi-center shift: split hospitals into train vs held-out, encode as the
    # column the harness already understands (anchor_year_group), so run_shift_eval
    # works unchanged.
    hosp = df["hospitalid"].dropna().unique()
    rng = np.random.default_rng(0)
    rng.shuffle(hosp)
    train_h = set(hosp[: len(hosp) // 2])
    df["anchor_year_group"] = np.where(df["hospitalid"].isin(train_h), "hosp_train", "hosp_eval")
    cfg.raw["shift"] = {"axis": "hospitalid", "train_groups": ["hosp_train"],
                        "eval_groups": ["hosp_eval"]}

    print(f"[eICU] {len(df)} stays, {df['subject_id'].nunique()} patients, "
          f"{df['hospitalid'].nunique()} hospitals; shift = held-out hospitals")
    experiment.run(cfg, df, feat_cols)
    print("Compare eICU results to mimic_pipeline/results/audit_summary_mimic_final_v1.3.json")


if __name__ == "__main__":
    main()
