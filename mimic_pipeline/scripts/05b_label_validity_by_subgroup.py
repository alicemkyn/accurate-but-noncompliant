"""STEP 5b — label validity by subgroup, on the FULL cohort (prereg-v1.4).

WHY THIS IS SEPARATE FROM 05.  The headline label-validity quantity -- the
fraction of guideline-severe cases that carry an ICD AKI code -- is a property of
the DATA, not of any model: no training, no split, no seed. Computing it on the
full cohort therefore uses every severe case (n=5,954) instead of a ~1,770-case
test split, so subgroups that fall below the pre-registered `min_subgroup_n`
floor per split are adequately powered here.

This matters for honesty in both directions. The per-split model quantities in
05 remain UNDETERMINED for the smaller subgroups, exactly as prereg §2 requires,
and we do not launder them through this script. What this script establishes is
the narrower, model-free claim: that the ICD label's agreement with the
guideline differs by subgroup. Uncertainty is a patient-level cluster bootstrap
(resampling `subject_id`, not admissions) so repeated admissions do not inflate
precision.

Usage:
    python scripts/05b_label_validity_by_subgroup.py --cohort <path.parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.config import load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_sg = import_module("05_subgroup_audit")
build_subgroups, age_band, race_bucket = _sg.build_subgroups, _sg.age_band, _sg.race_bucket


def cluster_bootstrap_ci(values, clusters, B=2000, seed=0, stat=np.mean):
    """Patient-level cluster bootstrap: resample subject_ids with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        out[b] = stat(values[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr_abs = float(cfg.constraints["creatinine_severe_mgdl"])
    df = pd.read_parquet(args.cohort)
    sev = df["severe_acute"].to_numpy(bool)
    dsev = df.loc[sev].reset_index(drop=True)
    aki = dsev["aki"].to_numpy(float)
    subj = dsev["subject_id"].to_numpy()
    nadir = pd.to_numeric(dsev["creat_nadir_24h"], errors="coerce").to_numpy(float)
    dims = build_subgroups(dsev)

    overall_lo, overall_hi = cluster_bootstrap_ci(aki, subj, B=args.B, seed=0)
    print(f"[label-validity] severe cases n={len(dsev)} (patients={dsev['subject_id'].nunique()})", flush=True)
    print(f"[label-validity] overall ICD agreement = {aki.mean():.4f} "
          f"[{overall_lo:.4f}, {overall_hi:.4f}]", flush=True)

    out = {}
    for dname, labels in dims.items():
        rec = {}
        for i, lab in enumerate(sorted(pd.unique(labels))):
            m = (labels == lab).to_numpy()
            n = int(m.sum())
            if n == 0:
                continue
            if n >= 30:
                lo, hi = cluster_bootstrap_ci(aki[m], subj[m], B=args.B, seed=100 + i)
            else:
                lo = hi = None
            rec[lab] = {
                "n_severe_cases": n,
                "n_patients": int(pd.unique(subj[m]).size),
                "icd_agreement": round(float(aki[m].mean()), 4),
                "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                "frac_elevated_at_arrival": round(float(np.nanmean(
                    np.nan_to_num(nadir[m], nan=0.0) >= thr_abs)), 4),
                "note": None if n >= 30 else "n<30: no interval reported",
            }
        out[dname] = rec
        print(f"\n== {dname.upper()}", flush=True)
        for lab, r in sorted(rec.items(), key=lambda kv: -kv[1]["n_severe_cases"]):
            ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r["ci95"] else "(no CI)"
            print(f"   {lab[:24]:24s} n={r['n_severe_cases']:5d} pts={r['n_patients']:5d} "
                  f"agree={r['icd_agreement']:.4f} {ci}", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg_tag": "prereg-v1.4",
        "quantity": ("fraction of decision-time acute-severe cases carrying an ICD AKI code "
                     "(model-free: no training, no split, no seed)"),
        "severity_definition": f"creat_peak_24h >= {thr_abs} mg/dL within the 24h decision window",
        "n_severe_cases": int(len(dsev)),
        "n_patients": int(dsev["subject_id"].nunique()),
        "uncertainty": f"patient-level cluster bootstrap on subject_id, B={args.B}, 95% percentile interval",
        "overall_icd_agreement": round(float(aki.mean()), 4),
        "overall_ci95": [round(overall_lo, 4), round(overall_hi, 4)],
        "caveat": ("this script reports the model-free label-validity quantity only; the "
                   "model-dependent per-split quantities remain UNDETERMINED below "
                   "min_subgroup_n as required by prereg §2 and are NOT re-reported here"),
        "subgroups": out,
    }
    (outdir / "label_validity_by_subgroup.json").write_text(json.dumps(payload, indent=1))
    print(f"\nSaved -> {outdir/'label_validity_by_subgroup.json'}", flush=True)


if __name__ == "__main__":
    main()
