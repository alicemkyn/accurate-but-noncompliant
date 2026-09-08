"""Build the AKI cohort from MIMIC-IV (hosp module).

Design decisions (addressing the review):

  #1 label-constraint circularity  -> the LABEL is administrative AKI
     (ICD-9 584* / ICD-10 N17*, from diagnoses_icd), which is *independent* of
     the creatinine trajectory the T2 hard constraint uses. The KDIGO
     creatinine rule is therefore a genuine, separately-sourced clinical
     constraint that a model can actually violate -- not the label rule in
     disguise. (A kdigo_creatinine label option exists but is flagged CIRCULAR.)

  #4 patient leakage               -> we carry subject_id so the experiment can
     split by patient (GroupShuffleSplit), never by admission.

  #5 baseline / windowing          -> features use ONLY the first
     `feature_window_hours` of the admission (no look-ahead). The severe-creatinine
     constraint uses the max creatinine within `constraint_window_days`.

The severe-creatinine value (`creat_severe`) is kept SEPARATE from the model
features so the T2 constraint is evaluated on a clinically-defined quantity, not
on a leaked feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import read_table
from .itemids import DEFAULT_ITEMID

# labevents chunk size (rows) for streaming the large table
_CHUNK = 2_000_000


def _load_labs(mimic_dir, itemid: dict[str, int]) -> pd.DataFrame:
    keep = set(itemid.values())
    id2name = {v: k for k, v in itemid.items()}
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    parts = []
    reader = pd.read_csv(read_table(mimic_dir, "labevents"), usecols=usecols,
                         chunksize=_CHUNK, parse_dates=["charttime"])
    for i, ch in enumerate(reader):
        ch = ch[ch["itemid"].isin(keep)].dropna(subset=["valuenum", "hadm_id"])
        parts.append(ch)
        print(f"    labevents chunk {i + 1}: kept {len(ch)} rows")
    labs = pd.concat(parts, ignore_index=True)
    labs["lab"] = labs["itemid"].map(id2name)
    return labs


def _icd_aki_labels(mimic_dir, icd9_prefixes, icd10_prefixes) -> pd.DataFrame:
    """Administrative AKI label per admission (independent of creatinine)."""
    dx = pd.read_csv(read_table(mimic_dir, "diagnoses_icd"),
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    code = dx["icd_code"].astype(str).str.strip().str.upper()
    is9 = (dx["icd_version"] == 9) & code.str.startswith(tuple(p.upper() for p in icd9_prefixes))
    is10 = (dx["icd_version"] == 10) & code.str.startswith(tuple(p.upper() for p in icd10_prefixes))
    dx["aki"] = (is9 | is10).astype(int)
    lab = dx.groupby(["subject_id", "hadm_id"])["aki"].max().reset_index()
    return lab


def _kdigo_creatinine_labels(labs, admit, abs_rise, rel_factor) -> pd.DataFrame:
    """Alternative CIRCULAR label (creatinine-derived). Off by default."""
    creat = labs[labs["lab"] == "creatinine"].merge(
        admit[["hadm_id", "admittime"]], on="hadm_id", how="inner")
    rows = []
    for hadm, g in creat.sort_values("charttime").groupby("hadm_id"):
        vals = g["valuenum"].to_numpy()
        times = g["charttime"].to_numpy()
        baseline = np.nanmin(vals)
        aki = int(np.nanmax(vals) >= rel_factor * baseline)
        if not aki:
            for j in range(len(vals)):
                m = (times >= times[j]) & (times <= times[j] + np.timedelta64(48, "h"))
                if m.sum() >= 2 and (np.nanmax(vals[m]) - vals[j]) >= abs_rise:
                    aki = 1
                    break
        rows.append((hadm, aki))
    return pd.DataFrame(rows, columns=["hadm_id", "aki"])


def build_cohort(cfg) -> tuple[pd.DataFrame, list[str]]:
    task = cfg.task
    itemid = DEFAULT_ITEMID
    print("[cohort] loading admissions / patients ...")
    adm = pd.read_csv(read_table(cfg.mimic_hosp_dir, "admissions"),
                      usecols=["subject_id", "hadm_id", "admittime", "race", "insurance"],
                      parse_dates=["admittime"])
    pat = pd.read_csv(read_table(cfg.mimic_hosp_dir, "patients"),
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year_group"])
    adm = adm.merge(pat, on="subject_id", how="left")

    print("[cohort] streaming labevents ...")
    labs = _load_labs(cfg.mimic_hosp_dir, itemid).merge(
        adm[["hadm_id", "admittime"]], on="hadm_id", how="inner")
    labs["hrs"] = (labs["charttime"] - labs["admittime"]) / np.timedelta64(1, "h")

    # --- features: first value per lab within the first feature_window_hours ---
    fw = float(task["feature_window_hours"])
    early = labs[(labs["hrs"] >= 0) & (labs["hrs"] <= fw)]
    feat = (early.sort_values("charttime")
                 .groupby(["hadm_id", "lab"])["valuenum"].first()
                 .unstack().reset_index())

    # --- creatinine within the DECISION window (== feature window) for the T2 constraint ---
    # prereg v1.2: a per-decision hard constraint must be a function of the evidence
    # available AT DECISION TIME. We therefore compute severity/acuteness from the SAME
    # first-`fw`-hours window as the features -- never from later (post-decision)
    # creatinine, which the model has not seen and cannot be held accountable for.
    creat_dec = early[early["lab"] == "creatinine"]
    dec_grp = creat_dec.groupby("hadm_id")["valuenum"]
    severe = dec_grp.max().rename("creat_peak_24h").reset_index()
    baseline = dec_grp.min().rename("creat_nadir_24h").reset_index()

    # --- label ---
    if task["label_source"] == "icd":
        lab_df = _icd_aki_labels(cfg.mimic_hosp_dir, task["icd9_prefixes"], task["icd10_prefixes"])
        lab_df = lab_df[["hadm_id", "aki"]]
    elif task["label_source"] == "kdigo_creatinine":
        print("    WARNING: kdigo_creatinine label is CIRCULAR with the T2 constraint.")
        lab_df = _kdigo_creatinine_labels(
            labs, adm, cfg.constraints["kdigo_abs_rise_mgdl"], cfg.constraints["kdigo_rel_rise_factor"])
    else:
        raise ValueError(f"unknown label_source: {task['label_source']}")

    df = (lab_df.merge(feat, on="hadm_id", how="inner")
                .merge(severe, on="hadm_id", how="left")
                .merge(baseline, on="hadm_id", how="left")
                .merge(adm, on="hadm_id", how="left"))
    df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["is_female"] = (df["gender"] == "F").astype(float)
    df["creat_peak_24h"] = pd.to_numeric(df["creat_peak_24h"], errors="coerce").fillna(0.0)
    df["creat_nadir_24h"] = pd.to_numeric(df["creat_nadir_24h"], errors="coerce")

    # --- T2 hard-constraint applicability (prereg v1.2) ---
    # Fire ONLY on ACUTE severe AKI that is evident WITHIN THE DECISION WINDOW:
    #   peak creatinine >= severe threshold, AND an acute rise within the same
    #   window (>= rel-factor over the nadir, OR >= KDIGO absolute rise). This
    #   (a) excludes stable chronic kidney disease (high but non-rising creatinine),
    #   and (b) uses only decision-time information, so the model is never held
    #   accountable for severity that manifested after it acted.
    thr = cfg.constraints["creatinine_severe_mgdl"]
    rel = cfg.constraints["kdigo_rel_rise_factor"]
    abs_rise = cfg.constraints["kdigo_abs_rise_mgdl"]
    acute = ((df["creat_peak_24h"] >= rel * df["creat_nadir_24h"]) |
             ((df["creat_peak_24h"] - df["creat_nadir_24h"]) >= abs_rise))
    df["severe_acute"] = ((df["creat_peak_24h"] >= thr) & acute.fillna(False)).astype(int)

    feat_cols = [c for c in itemid.keys() if c in df.columns] + ["age", "is_female"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df = df.dropna(subset=["aki"]).reset_index(drop=True)
    n_abs = int((df["creat_peak_24h"] >= thr).sum())
    n_acute = int(df["severe_acute"].sum())
    print(f"[cohort] admissions={len(df)}, patients={df['subject_id'].nunique()}, "
          f"AKI rate={df['aki'].mean():.3f}, "
          f"severe-creat@24h absolute(>= {thr})={n_abs}, "
          f"severe-ACUTE@24h (T2 constraint applies)={n_acute}")
    return df, feat_cols
