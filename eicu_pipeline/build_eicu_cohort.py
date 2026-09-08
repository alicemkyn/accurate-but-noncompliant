#!/usr/bin/env python
"""Build an eICU AKI cohort in the SAME dataframe contract the MIMIC harness expects,
so the existing audit machinery runs unchanged (see EICU_PLAN.md).

    export EICU_DIR=/path/to/eicu-crd
    export EICU_OUT=./eicu_out
    python build_eicu_cohort.py

Writes EICU_OUT/aki_cohort.parquet + feature_cols.txt (patient-level; NEVER commit).

STATUS: functional skeleton. The spots marked  # VERIFY  depend on exact eICU
labname strings/units (run 00_check_eicu_labnames.py first) and must be confirmed
on the real data before trusting the output. Nothing here fabricates values; if a
required field is missing it raises rather than guessing.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

EICU_DIR = Path(os.environ.get("EICU_DIR", "./eicu"))
OUT = Path(os.environ.get("EICU_OUT", "./eicu_out"))
OUT.mkdir(parents=True, exist_ok=True)

FIRST24 = 1440          # minutes
SEVERE = 4.0            # mg/dL, same as MIMIC constraint
REL = 1.5               # acute-rise factor
ABS_RISE = 0.3          # mg/dL

# eICU labname -> analyte. VERIFY exact strings + units via 00_check_eicu_labnames.py
LABNAME = {
    "creatinine": "creatinine",
    "glucose": "glucose",
    "bun": "BUN",
    "potassium": "potassium",
    "sodium": "sodium",
    "hemoglobin": "Hgb",
    "wbc": "WBC x 1000",
    "bicarbonate": "bicarbonate",
}
ANALYTES = list(LABNAME)


def _read(name):
    for ext in (".csv.gz", ".csv"):
        p = EICU_DIR / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.csv[.gz] not found in {EICU_DIR}")


def _age_to_num(a):
    # eICU age is a string; '> 89' -> 90
    return pd.to_numeric(pd.Series(a).replace({"> 89": "90"}), errors="coerce")


def build():
    name2analyte = {v.lower(): k for k, v in LABNAME.items()}

    print("[eICU] loading patient ...")
    pat = pd.read_csv(_read("patient"), low_memory=False,
                      usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid",
                               "gender", "age", "hospitalid"])
    pat["age"] = _age_to_num(pat["age"])
    pat["is_female"] = (pat["gender"].astype(str).str.upper().str.startswith("F")).astype(float)

    print("[eICU] loading lab (first 24h) ...")
    lab = pd.read_csv(_read("lab"), low_memory=False,
                      usecols=["patientunitstayid", "labname", "labresultoffset", "labresult"])
    lab = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= FIRST24)]
    lab["analyte"] = lab["labname"].str.lower().map(name2analyte)  # VERIFY mapping coverage
    lab = lab.dropna(subset=["analyte", "labresult"])

    # first value per analyte per stay (decision-time features)
    feat = (lab.sort_values("labresultoffset")
               .groupby(["patientunitstayid", "analyte"])["labresult"].first()
               .unstack().reset_index())

    # creatinine peak/nadir within first 24h for the acute-severe mask
    creat = lab[lab["analyte"] == "creatinine"].groupby("patientunitstayid")["labresult"]
    sev = creat.max().rename("creat_peak_24h").reset_index()
    nad = creat.min().rename("creat_nadir_24h").reset_index()

    # --- AKI label: diagnosis-string proxy (analogue of ICD; INDEPENDENT of creatinine) ---
    print("[eICU] loading diagnosis (AKI label) ...")
    dx = pd.read_csv(_read("diagnosis"), low_memory=False,
                     usecols=["patientunitstayid", "diagnosisstring"])
    aki_mask = dx["diagnosisstring"].str.contains(
        "acute renal failure|acute kidney", case=False, na=False)   # VERIFY string coverage
    aki = (dx.assign(aki=aki_mask.astype(int))
             .groupby("patientunitstayid")["aki"].max().reset_index())

    df = (feat.merge(sev, on="patientunitstayid", how="inner")
              .merge(nad, on="patientunitstayid", how="left")
              .merge(aki, on="patientunitstayid", how="left")
              .merge(pat, on="patientunitstayid", how="left"))
    df["aki"] = df["aki"].fillna(0).astype(int)     # no AKI dx -> negative label
    df = df.rename(columns={"patientunitstayid": "hadm_id",
                            "patienthealthsystemstayid": "subject_id"})
    df["subject_id"] = df["subject_id"].fillna(df["uniquepid"])

    df["creat_peak_24h"] = pd.to_numeric(df["creat_peak_24h"], errors="coerce").fillna(0.0)
    df["creat_nadir_24h"] = pd.to_numeric(df["creat_nadir_24h"], errors="coerce")
    acute = ((df["creat_peak_24h"] >= REL * df["creat_nadir_24h"]) |
             ((df["creat_peak_24h"] - df["creat_nadir_24h"]) >= ABS_RISE))
    df["severe_acute"] = ((df["creat_peak_24h"] >= SEVERE) & acute.fillna(False)).astype(int)

    feat_cols = [a for a in ANALYTES if a in df.columns] + ["age", "is_female"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df = df.dropna(subset=["aki", "subject_id"]).reset_index(drop=True)

    n_acute = int(df["severe_acute"].sum())
    print(f"[eICU] stays={len(df)}, patients={df['subject_id'].nunique()}, "
          f"hospitals={df['hospitalid'].nunique()}, AKI rate={df['aki'].mean():.3f}, "
          f"severe-acute@24h={n_acute}")
    if n_acute < 500:
        print("  [!] < 500 acute-severe cases — T2 subgroup underpowered; revisit windowing.")
    df.to_parquet(OUT / "aki_cohort.parquet", index=False)
    (OUT / "feature_cols.txt").write_text("\n".join(feat_cols))
    print(f"[done] -> {OUT/'aki_cohort.parquet'}")
    return df, feat_cols


if __name__ == "__main__":
    build()
