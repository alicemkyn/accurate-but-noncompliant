#!/usr/bin/env python
"""STEP 0 (eICU) — list the lab names + units present, so we can map the 8 analytes
the harness expects. Run before building the cohort.

    export EICU_DIR=/path/to/eicu-crd            # folder with lab.csv[.gz], patient.csv[.gz], diagnosis.csv[.gz]
    python 00_check_eicu_labnames.py
"""
import os
from pathlib import Path

import pandas as pd

EICU_DIR = os.environ.get("EICU_DIR", "./eicu")

# candidate eICU labname substrings for each analyte we need (VERIFY against output)
TARGETS = {
    "creatinine": "creatinine",
    "glucose": "glucose",
    "bun": "bun",
    "potassium": "potassium",
    "sodium": "sodium",
    "hemoglobin": "hgb",       # eICU often uses 'Hgb'
    "wbc": "wbc",              # e.g. 'WBC x 1000'
    "bicarbonate": "bicarbonate",
}


def _read(name):
    for ext in (".csv.gz", ".csv"):
        p = Path(EICU_DIR) / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.csv[.gz] not found in {EICU_DIR}")


if __name__ == "__main__":
    lab = pd.read_csv(_read("lab"), usecols=["labname", "labmeasurenamesystem"],
                      low_memory=False)
    counts = lab.value_counts(["labname", "labmeasurenamesystem"]).reset_index(name="n")
    print(f"distinct labname rows: {lab['labname'].nunique()}\n")
    for key, sub in TARGETS.items():
        hits = counts[counts["labname"].str.contains(sub, case=False, na=False)]
        print(f"--- {key.upper()} (search '{sub}') ---")
        print(hits.head(8).to_string(index=False) if len(hits) else "  (no match — adjust substring)")
        print()
    print("Copy the exact labname strings + units into build_eicu_cohort.py LABNAME map.")
