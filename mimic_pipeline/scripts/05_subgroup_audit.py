"""STEP 5 — demographic subgroup audit + categorised failure-mode taxonomy.

Added in prereg-v1.4. Two additions, both computed on the SAME runs so the
compliance gap and its mechanism are measured on identical splits:

  (A) SUBGROUP AUDIT.  The compliance gap is sliced by age band, sex, race and
      insurance.  prereg §2 declares `min_subgroup_n: 500`; that rule is now
      actually enforced -- a subgroup below the floor is reported UNDETERMINED,
      never PASS and never counted in a comparison.

  (B) FAILURE-MODE TAXONOMY.  Aggregate violation counts say 71% of deployed
      severe decisions are confident denials but not WHICH patients those are.
      Confident denials are partitioned on two orthogonal axes:
        Axis A (creatinine trajectory shape, decision-window only)
          A1 already-elevated at arrival   nadir >= severe threshold
          A2 acute rise from near-normal   nadir < 2.0
          A3 acute rise from elevated base 2.0 <= nadir < severe threshold
        Axis B (administrative label status)
          coded (aki=1)   -> the model erred against its own label
          uncoded (aki=0) -> the label erred against the guideline
      A1 is the nadir-baseline population the paper already flags as a
      conservative bias; reporting it separately quantifies that bias instead
      of asserting it.  The oliguric-only limb cannot be computed here (urine
      output is not in this cohort); see `results/urine_output_arm.json`.

This script REUSES the audited package unchanged (read-only import) so the
primary arm is untouched.  It reads the committed cohort parquet directly,
which carries every column it needs, so no MIMIC re-extraction is required.

Usage:
    python scripts/05_subgroup_audit.py --cohort <path.parquet> [--seeds 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic import audit, metrics  # noqa: E402
from auditnesy_mimic.config import load_config  # noqa: E402
from auditnesy_mimic.models import PipelineCoupled  # noqa: E402

FEAT_ORDER = ["creatinine", "glucose", "bun", "potassium", "sodium",
              "hemoglobin", "wbc", "bicarbonate", "age", "is_female"]


def _subject_split(df, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["subject_id"].values))


def race_bucket(s: str) -> str:
    """Collapse MIMIC-IV's 33 race strings into reportable groups.

    Buckets follow the coarse OMB-style categories MIMIC's strings encode.
    UNKNOWN/DECLINED are kept as their own bucket rather than dropped, because
    silently discarding them would hide differential missingness.
    """
    t = (s or "").upper()
    if "WHITE" in t:
        return "White"
    if "BLACK" in t or "AFRICAN" in t:
        return "Black"
    if "HISPANIC" in t or "LATINO" in t:
        return "Hispanic/Latino"
    if "ASIAN" in t:
        return "Asian"
    if "AMERICAN INDIAN" in t or "ALASKA" in t or "HAWAIIAN" in t or "PACIFIC" in t:
        return "AIAN/NHPI"
    if "UNKNOWN" in t or "DECLINED" in t or "UNABLE" in t:
        return "Unknown/Declined"
    return "Other/Multiple"


def age_band(a: float) -> str:
    if a < 45:
        return "18-44"
    if a < 60:
        return "45-59"
    if a < 75:
        return "60-74"
    return "75+"


def build_subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {dimension: Series of subgroup labels aligned to df.index}."""
    return {
        "age_band": df["age"].map(age_band),
        "sex": df["gender"].map({"F": "Female", "M": "Male"}).fillna("Unknown"),
        "race": df["race"].map(race_bucket),
        "insurance": df["insurance"].fillna("Unrecorded"),
    }


def per_split_subgroup_stats(probs, aki, severe, tau, groups, thr_severe,
                             nadir, min_n):
    """Compute, per subgroup label, the quantities the paper reports globally.

    Returns {label: {...}}.  All quantities are restricted to the acute-severe
    subgroup where the T2 constraint applies, except `n` and `n_all`.
    """
    pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1 - probs)
    confident = conf >= tau
    confident_neg = (pred == 0) & confident & severe

    out = {}
    for lab in pd.unique(groups):
        m = (groups == lab).to_numpy()
        sev_m = m & severe
        n_sev = int(sev_m.sum())
        deployed = sev_m & confident              # logged policy: all confident decisions
        n_dep = int(deployed.sum())
        viol = int((confident_neg & m).sum())
        rec = {
            "n_all": int(m.sum()),
            "n_severe": n_sev,
            # label validity: fraction of guideline-severe cases carrying an ICD AKI code
            "icd_agreement_on_severe": (float(aki[sev_m].mean()) if n_sev else None),
            "n_deployed_severe": n_dep,
            "coverage_severe": (n_dep / n_sev) if n_sev else None,
            "hard_violations": viol,
            "violation_rate_deployed": (viol / n_dep) if n_dep else None,
            # share of severe cases already elevated at arrival (nadir-baseline bias)
            "frac_elevated_at_arrival": (
                float((nadir[sev_m] >= thr_severe).mean()) if n_sev else None),
            "status": "OK" if n_sev >= min_n else "UNDETERMINED",
        }
        out[lab] = rec
    return out


def taxonomy_for_split(probs, aki, severe, tau, nadir, peak, thr_severe, rel_factor):
    """Partition confident no-AKI decisions on severe cases into the taxonomy."""
    pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1 - probs)
    confident_neg = (pred == 0) & (conf >= tau) & severe

    nad = np.where(np.isnan(nadir), 0.0, nadir)
    a1 = nad >= thr_severe                      # already at/above severe on arrival
    a2 = nad < 2.0                              # rise from near-normal
    a3 = (~a1) & (~a2)                          # rise from moderately elevated
    axisA = np.where(a1, "A1_elevated_at_arrival",
             np.where(a2, "A2_rise_from_near_normal", "A3_rise_from_elevated_base"))
    axisB = np.where(aki == 1, "coded", "uncoded")

    cells = {}
    for ka in ["A1_elevated_at_arrival", "A2_rise_from_near_normal", "A3_rise_from_elevated_base"]:
        for kb in ["coded", "uncoded"]:
            m = confident_neg & (axisA == ka) & (axisB == kb)
            sev_m = severe & (axisA == ka) & (axisB == kb)
            cells[f"{ka}|{kb}"] = {
                "n_confident_denials": int(m.sum()),
                "n_severe_in_cell": int(sev_m.sum()),
            }
    # marginals
    marg = {}
    for ka in ["A1_elevated_at_arrival", "A2_rise_from_near_normal", "A3_rise_from_elevated_base"]:
        m = confident_neg & (axisA == ka)
        sev_m = severe & (axisA == ka)
        marg[ka] = {"n_confident_denials": int(m.sum()),
                    "n_severe_in_cell": int(sev_m.sum()),
                    "denial_rate_within_cell": (float(m.sum() / sev_m.sum()) if sev_m.sum() else None)}
    for kb in ["coded", "uncoded"]:
        m = confident_neg & (axisB == kb)
        sev_m = severe & (axisB == kb)
        marg[kb] = {"n_confident_denials": int(m.sum()),
                    "n_severe_in_cell": int(sev_m.sum()),
                    "denial_rate_within_cell": (float(m.sum() / sev_m.sum()) if sev_m.sum() else None)}
    return {"cells": cells, "marginals": marg,
            "total_confident_denials": int(confident_neg.sum()),
            "total_severe": int(severe.sum())}


def agg_across_seeds(per_seed_dicts, keys):
    """Mean + 5th/95th percentile across seeds for each subgroup/key."""
    labels = sorted({lab for d in per_seed_dicts for lab in d})
    out = {}
    for lab in labels:
        rec = {}
        for k in keys:
            vals = [d[lab][k] for d in per_seed_dicts
                    if lab in d and d[lab].get(k) is not None]
            if not vals:
                rec[k] = None
                continue
            v = np.asarray(vals, float)
            rec[k] = {"mean": round(float(np.nanmean(v)), 4),
                      "p5": round(float(np.nanpercentile(v, 5)), 4),
                      "p95": round(float(np.nanpercentile(v, 95)), 4),
                      "n_seeds": int(len(v))}
        ns = [d[lab]["n_severe"] for d in per_seed_dicts if lab in d]
        rec["n_severe_mean"] = round(float(np.mean(ns)), 1) if ns else 0.0
        st = [d[lab]["status"] for d in per_seed_dicts if lab in d]
        rec["status"] = "UNDETERMINED" if any(s == "UNDETERMINED" for s in st) else "OK"
        out[lab] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr_severe = float(cfg.constraints["creatinine_severe_mgdl"])
    rel_factor = float(cfg.constraints["kdigo_rel_rise_factor"])
    min_n = int(cfg.audit["min_subgroup_n"])
    seeds = cfg.experiment["seeds"][:args.seeds] if args.seeds else cfg.experiment["seeds"]
    test_size = cfg.experiment["test_size"]
    mcfg = cfg.models["pipeline_coupled"]

    df = pd.read_parquet(args.cohort)
    feat_cols = [c for c in FEAT_ORDER if c in df.columns]
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    severe_all = df["severe_acute"].to_numpy(bool)
    nadir_all = pd.to_numeric(df["creat_nadir_24h"], errors="coerce").to_numpy(float)
    peak_all = pd.to_numeric(df["creat_peak_24h"], errors="coerce").to_numpy(float)
    dims = build_subgroups(df)

    print(f"[subgroup] cohort={len(df)} admissions, patients={df['subject_id'].nunique()}, "
          f"severe_acute={int(severe_all.sum())}, features={feat_cols}", flush=True)
    print(f"[subgroup] min_subgroup_n={min_n} (prereg §2, now enforced); seeds={len(seeds)}", flush=True)

    per_seed_sub = {d: [] for d in dims}
    per_seed_tax = []
    taus = []
    for seed in seeds:
        tr, te = _subject_split(df, test_size, seed)
        model = PipelineCoupled(mcfg["hidden_layers"], mcfg["calibrate"], seed=seed)
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])

        _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(X[tr][vval]),
                                 cfg.audit["abstain_target_rate"])
        taus.append(float(tau))

        sev_te = severe_all[te]
        aki_te = y[te]
        nad_te = nadir_all[te]
        peak_te = peak_all[te]

        for dname, labels in dims.items():
            g = labels.iloc[te].reset_index(drop=True)
            per_seed_sub[dname].append(
                per_split_subgroup_stats(probs, aki_te, sev_te, tau, g,
                                         thr_severe, nad_te, min_n))
        per_seed_tax.append(
            taxonomy_for_split(probs, aki_te, sev_te, tau, nad_te, peak_te,
                               thr_severe, rel_factor))
        print(f"  seed {seed:2d}: tau={tau:.4f} n_severe={int(sev_te.sum())} "
              f"denials={per_seed_tax[-1]['total_confident_denials']}", flush=True)

    keys = ["icd_agreement_on_severe", "coverage_severe", "violation_rate_deployed",
            "hard_violations", "frac_elevated_at_arrival", "n_severe"]
    subgroup_out = {d: agg_across_seeds(v, keys) for d, v in per_seed_sub.items()}

    # taxonomy: mean counts across seeds
    tax_keys = sorted(per_seed_tax[0]["cells"])
    tax_cells = {k: {
        "n_confident_denials_mean": round(float(np.mean([t["cells"][k]["n_confident_denials"] for t in per_seed_tax])), 2),
        "n_severe_in_cell_mean": round(float(np.mean([t["cells"][k]["n_severe_in_cell"] for t in per_seed_tax])), 2),
    } for k in tax_keys}
    marg_keys = sorted(per_seed_tax[0]["marginals"])
    tax_marg = {k: {
        "n_confident_denials_mean": round(float(np.mean([t["marginals"][k]["n_confident_denials"] for t in per_seed_tax])), 2),
        "n_severe_in_cell_mean": round(float(np.mean([t["marginals"][k]["n_severe_in_cell"] for t in per_seed_tax])), 2),
        "denial_rate_within_cell_mean": round(float(np.nanmean(
            [t["marginals"][k]["denial_rate_within_cell"] if t["marginals"][k]["denial_rate_within_cell"] is not None else np.nan
             for t in per_seed_tax])), 4),
    } for k in marg_keys}

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg_tag": "prereg-v1.4",
        "cohort": str(args.cohort),
        "n_admissions": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "n_severe_acute": int(severe_all.sum()),
        "n_seeds": len(seeds),
        "min_subgroup_n": min_n,
        "tau_per_seed": [round(t, 4) for t in taus],
        "note": ("Subgroup criteria are UNDETERMINED below min_subgroup_n (prereg §2). "
                 "All quantities restricted to decision-time acute-severe cases where "
                 "the T2 constraint applies, except n_all."),
        "subgroups": subgroup_out,
    }
    save_a = outdir / "subgroup_audit.json"
    save_a.write_text(json.dumps(payload, indent=1))

    tax_payload = {
        "prereg_tag": "prereg-v1.4",
        "n_seeds": len(seeds),
        "axis_A_definition": {
            "A1_elevated_at_arrival": f"creat_nadir_24h >= {thr_severe} mg/dL",
            "A2_rise_from_near_normal": "creat_nadir_24h < 2.0 mg/dL",
            "A3_rise_from_elevated_base": f"2.0 <= creat_nadir_24h < {thr_severe} mg/dL",
        },
        "axis_B_definition": {"coded": "ICD AKI code present (aki=1)",
                              "uncoded": "no ICD AKI code (aki=0)"},
        "oliguric_limb": ("not computable from this cohort (no urine output); "
                          "see results/urine_output_arm.json"),
        "total_confident_denials_mean": round(float(np.mean([t["total_confident_denials"] for t in per_seed_tax])), 2),
        "total_severe_mean": round(float(np.mean([t["total_severe"] for t in per_seed_tax])), 2),
        "cells": tax_cells,
        "marginals": tax_marg,
    }
    save_b = outdir / "error_taxonomy.json"
    save_b.write_text(json.dumps(tax_payload, indent=1))
    print(f"\nSaved -> {save_a}\nSaved -> {save_b}", flush=True)


if __name__ == "__main__":
    main()
