"""KDIGO urine-output arm: quantify how much the creatinine-only AKI definition
under-counts by omitting the KDIGO urine-output criterion.

Reviewer point (clinical): the KDIGO definition of AKI has TWO limbs --
serum-creatinine rise AND oliguria (urine output < 0.5 mL/kg/h). Our label and
T2 constraint use the creatinine limb only, so they *under-count* oliguric AKI
that has no creatinine rise. Rather than defend this in prose ("our gap is a
lower bound"), we MEASURE it: among admissions with an ICU stay, how many
additional AKI cases does the urine limb capture beyond creatinine?

This is a measurement arm (no model re-run). Output is aggregate-only (counts,
rates) -- DUA-safe, committed as evidence. All intermediate patient-level frames
stay in memory / local only.

Method (documented simplifications, stated in the paper):
  * Urine sources: outputevents itemids 226559 (Foley), 226627 (OR Urine),
    226631 (PACU Urine). Irrigant-mixed sources are excluded.
  * Window: first 24 h from ICU intime (icustays.intime).
  * Weight: median inputevents.patientweight per stay (kg).
  * Oliguria: first-24h split into four fixed 6-h blocks; block rate =
    block_mL / (6 * weight_kg); a stay is oliguric if ANY block with data has
    rate < 0.5 mL/kg/h. (Fixed blocks, not a rolling window -- a defensible
    approximation that avoids false positives from sparse hourly charting.)
  * hadm-level: a hadm is urine-oliguric if ANY of its ICU stays is oliguric.

Env:
  URINE_ICU_DIR   path to the MIMIC-IV `icu` module (default: MIMIC_DIR/../icu)
  AUDITNESY_OUT   cohort output dir holding aki_cohort.parquet (as in the main run)
  MIMIC_DIR       (used only to derive URINE_ICU_DIR default)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

URINE_ITEMIDS = [226559, 226627, 226631]  # Foley, OR Urine, PACU Urine
BLOCK_HOURS = 6
N_BLOCKS = 4  # 24 h / 6 h
OLIGURIA_THRESHOLD = 0.5  # mL/kg/h (KDIGO stage-1 urine limb)


def _icu_dir() -> Path:
    if os.environ.get("URINE_ICU_DIR"):
        return Path(os.environ["URINE_ICU_DIR"])
    mimic = os.environ.get("MIMIC_DIR")
    if not mimic:
        raise SystemExit("set URINE_ICU_DIR (or MIMIC_DIR so ../icu can be derived)")
    return Path(mimic).resolve().parent / "icu"


def _cohort_path() -> Path:
    out = os.environ.get("AUDITNESY_OUT")
    if not out:
        raise SystemExit("set AUDITNESY_OUT to the cohort output dir (holds aki_cohort.parquet)")
    p = Path(out) / "aki_cohort.parquet"
    if not p.exists():
        raise SystemExit(f"aki_cohort.parquet not found in {out}; run 01_build_cohort.py first")
    return p


def load_stays(icu: Path) -> pd.DataFrame:
    stays = pd.read_csv(icu / "icustays.csv.gz",
                        usecols=["stay_id", "hadm_id", "intime"],
                        parse_dates=["intime"])
    return stays


def load_weights(icu: Path) -> pd.DataFrame:
    # inputevents.patientweight is present on every row; median per stay (kg).
    w = pd.read_csv(icu / "inputevents.csv.gz",
                    usecols=["stay_id", "patientweight"])
    w = w.dropna(subset=["stay_id", "patientweight"])
    w = w[(w["patientweight"] > 20) & (w["patientweight"] < 400)]  # plausibility
    return w.groupby("stay_id")["patientweight"].median().rename("weight_kg").reset_index()


def load_urine(icu: Path) -> pd.DataFrame:
    uo = pd.read_csv(icu / "outputevents.csv.gz",
                     usecols=["stay_id", "charttime", "itemid", "value"],
                     parse_dates=["charttime"])
    uo = uo[uo["itemid"].isin(URINE_ITEMIDS)].copy()
    uo = uo.dropna(subset=["stay_id", "charttime", "value"])
    uo = uo[(uo["value"] > 0) & (uo["value"] < 5000)]  # drop errors/outliers (mL)
    return uo[["stay_id", "charttime", "value"]]


def compute_oliguria(uo: pd.DataFrame, stays: pd.DataFrame,
                     weights: pd.DataFrame) -> pd.DataFrame:
    """Per-stay oliguria flags over first 24 h, using fixed 6-h blocks.

    Two operationalizations, reported side by side:
      * sensitive    : ANY charted 6-h block < 0.5 mL/kg/h (matches the KDIGO
                       stage-1 urine limb literally, but is inflated by sparse
                       charting and transient/post-op dips -> UPPER bound).
      * conservative : full 24-h charting (all 4 blocks have data, so a low value
                       reflects genuine low output rather than a charting gap)
                       AND >= 2 low blocks (>= 12 h low) -> specific LOWER bound.
    """
    df = uo.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
    df["hrs"] = (df["charttime"] - df["intime"]) / np.timedelta64(1, "h")
    df = df[(df["hrs"] >= 0) & (df["hrs"] < N_BLOCKS * BLOCK_HOURS)]
    df["block"] = (df["hrs"] // BLOCK_HOURS).astype(int)

    block_ml = (df.groupby(["stay_id", "block"])["value"].sum()
                  .rename("block_ml").reset_index())
    block_ml = block_ml.merge(weights, on="stay_id", how="inner")  # need weight
    block_ml["rate"] = block_ml["block_ml"] / (BLOCK_HOURS * block_ml["weight_kg"])
    block_ml["low"] = block_ml["rate"] < OLIGURIA_THRESHOLD

    oli = (block_ml.groupby("stay_id")
                   .agg(n_blocks=("block", "nunique"),
                        n_low=("low", "sum"),
                        min_rate=("rate", "min"))
                   .reset_index())
    oli["oliguric_sensitive"] = (oli["n_low"] >= 1).astype(int)
    oli["oliguric_conservative"] = ((oli["n_blocks"] == N_BLOCKS) &
                                    (oli["n_low"] >= 2)).astype(int)
    return oli


def main() -> None:
    icu = _icu_dir()
    cohort_p = _cohort_path()
    print(f"[urine-arm] ICU module : {icu}")
    print(f"[urine-arm] cohort     : {cohort_p}")

    print("[urine-arm] loading icustays / weights / urine ...")
    stays = load_stays(icu)
    weights = load_weights(icu)
    uo = load_urine(icu)
    print(f"           ICU stays={len(stays):,}  stays w/ weight={len(weights):,}  "
          f"urine rows={len(uo):,}")

    oli = compute_oliguria(uo, stays, weights)
    print(f"[urine-arm] stays with computable urine limb: {len(oli):,}")

    # map stay-level oliguria to hadm level (any ICU stay oliguric -> hadm oliguric)
    stay2hadm = stays[["stay_id", "hadm_id"]].dropna()
    oli_h = (oli.merge(stay2hadm, on="stay_id", how="inner")
                .groupby("hadm_id")[["oliguric_sensitive", "oliguric_conservative"]]
                .max().reset_index())
    oli_h["hadm_id"] = oli_h["hadm_id"].astype("int64")

    # join to the audited cohort (creatinine-based aki + severe_acute)
    coh = pd.read_parquet(cohort_p, columns=["hadm_id", "aki", "severe_acute"])
    coh["hadm_id"] = coh["hadm_id"].astype("int64")
    m = coh.merge(oli_h, on="hadm_id", how="inner")  # ICU subset w/ computable limb
    n_cohort = len(coh)
    n_icu = len(m)
    aki_creat = m["aki"] == 1
    creat_pos = int(aki_creat.sum())

    def arm(flag_col: str) -> dict:
        aki_urine = m[flag_col] == 1
        add_aki = int((aki_urine & ~aki_creat).sum())
        return {
            "urine_oliguric_positive": int(aki_urine.sum()),
            "aki_both_limbs": int((aki_urine & aki_creat).sum()),
            "additional_aki_from_urine_only": add_aki,
            "undercount_fraction_of_creatinine_positives":
                round(add_aki / creat_pos, 4) if creat_pos else None,
            "additional_aki_as_fraction_of_icu_subset":
                round(add_aki / n_icu, 4) if n_icu else None,
            "oliguric_rate_in_icu_subset": round(int(aki_urine.sum()) / n_icu, 4),
        }

    summary = {
        "description": "KDIGO urine-output arm: additional AKI captured beyond "
                       "the creatinine-only definition (ICU subset).",
        "method": {
            "urine_itemids": URINE_ITEMIDS,
            "window_hours": N_BLOCKS * BLOCK_HOURS,
            "block_hours": BLOCK_HOURS,
            "oliguria_threshold_ml_kg_h": OLIGURIA_THRESHOLD,
            "weight_source": "inputevents.patientweight (median per stay)",
            "rules": {
                "sensitive": "any charted 6-h block < 0.5 mL/kg/h (UPPER bound; "
                             "inflated by sparse charting + transient dips)",
                "conservative": "full 24-h charting (all 4 blocks) AND >= 2 low "
                                "blocks (>= 12 h low) (specific LOWER bound)",
            },
            "caveats": [
                "ICU intime-anchored window vs hosp admittime-anchored creatinine label",
                "fixed 6-h blocks (not rolling)",
                "urine limb computable only for admissions with an ICU stay",
            ],
        },
        "counts": {
            "cohort_admissions_total": n_cohort,
            "admissions_with_computable_urine_limb": n_icu,
            "creatinine_aki_positive": creat_pos,
        },
        "sensitive": arm("oliguric_sensitive"),
        "conservative": arm("oliguric_conservative"),
    }
    cons = summary["conservative"]
    summary["interpretation"] = (
        "The creatinine-only AKI definition omits the urine limb entirely. "
        "Under the conservative (>=12 h sustained, fully-charted) rule, the KDIGO "
        f"urine criterion flags {cons['additional_aki_from_urine_only']} additional "
        "AKI cases in the ICU subset with no qualifying creatinine rise "
        f"(+{round(100*cons['undercount_fraction_of_creatinine_positives'],1)}% over "
        "creatinine-positives). Even this lower-bound estimate shows the "
        "administrative/creatinine label materially under-counts AKI, so the "
        "label-validity gap reported in the paper is conservative."
    )

    out_p = Path(__file__).resolve().parent.parent / "results" / "urine_output_arm.json"
    out_p.parent.mkdir(exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== URINE-OUTPUT ARM (aggregate, DUA-safe) =====")
    print("counts:", json.dumps(summary["counts"], indent=2))
    print("sensitive (upper):", json.dumps(summary["sensitive"], indent=2))
    print("conservative (lower):", json.dumps(summary["conservative"], indent=2))
    print(f"\n[urine-arm] wrote {out_p}")


if __name__ == "__main__":
    main()
