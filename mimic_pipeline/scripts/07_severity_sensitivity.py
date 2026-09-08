"""STEP 7 — severity-definition sensitivity arm (prereg-v1.4).

MOTIVATION, PRE-DECLARED BEFORE RUNNING (see preregistration.md deviations log).
KDIGO Stage 3 AKI is a DISJUNCTION: serum creatinine >= 3.0x baseline, OR an
absolute rise to >= 4.0 mg/dL, OR initiation of renal replacement therapy (RRT),
OR (in patients <18y) eGFR < 35.  The primary arm (prereg-v1.3) operationalises
only the ABSOLUTE limb (`creatinine_severe_mgdl: 4.0`).  That is a proper subset
of the guideline, so the primary arm's severe-case set is narrower than KDIGO
Stage 3, and the paper should not describe it as "the KDIGO definition" without
qualification.

This arm therefore widens the severity definition to the limbs that ARE
computable from the released cohort and asks whether the headline finding is an
artefact of the narrow definition:

  computable here  : absolute limb (>= 4.0 mg/dL)  [primary]
                     relative limb (peak >= 3.0x within-window nadir)
  NOT computable   : RRT initiation -- requires MIMIC-IV `procedureevents` /
                     dialysis items, which are NOT part of the released cohort
                     derivative.  This is recorded as an explicit limitation,
                     not silently omitted.  In an ICU population RRT is the
                     least ambiguous marker of severe AKI, so its absence makes
                     BOTH arms conservative: severe cases treated with dialysis
                     but never crossing 4.0 mg/dL (or 3.0x) are missed entirely.
  NOT computable   : paediatric eGFR limb -- MIMIC-IV is adult-only.

DIRECTION OF THE PRE-DECLARED EXPECTATION.  Widening the severity set adds cases
whose creatinine trajectory is proportionally steep but absolutely lower.  Such
patients are LESS likely to carry an ICD AKI code (coding tracks absolute
severity), so we expect the label-validity gap to persist or widen, and the
violation rate to move little.  If instead the violation rate collapsed under the
wider definition, that would indicate the finding IS an artefact of the narrow
absolute threshold, and we would report it as such.

The primary arm is UNCHANGED: v1.3 remains the registered primary specification
and this is reported as a sensitivity analysis, so no gated threshold is altered
after seeing results.

CAVEAT ON THE BASELINE PROXY.  The relative limb uses the within-24h nadir as a
proxy for pre-morbid baseline creatinine, inheriting exactly the conservative
bias the primary arm already discloses: patients who arrive already injured have
an elevated nadir, so their true ratio is understated and they are under-counted.
This arm does not fix that; it quantifies the effect of the limb choice only.

NOT CLINICALLY REVIEWED.  This is a reading of the published KDIGO criteria by
the authors, not a clinician-ratified operationalisation.  No nephrologist
reviewed these rules.  That limitation is stated in the paper.

Usage:
    python scripts/07_severity_sensitivity.py --cohort <path.parquet> [--seeds 20]
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


def severity_masks(df, thr_abs, rel_factor_stage3=3.0):
    """Return {arm_name: bool mask} for each computable severity definition."""
    peak = pd.to_numeric(df["creat_peak_24h"], errors="coerce").to_numpy(float)
    nadir = pd.to_numeric(df["creat_nadir_24h"], errors="coerce").to_numpy(float)
    absolute = df["severe_acute"].to_numpy(bool)          # as released (>= thr_abs)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = peak / nadir
    relative = np.nan_to_num(ratio, nan=0.0, posinf=0.0) >= rel_factor_stage3
    return {
        "absolute_only_PRIMARY_v13": absolute,
        "relative_only_3x": relative,
        "absolute_or_relative_EXTENDED": absolute | relative,
    }, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr_abs = float(cfg.constraints["creatinine_severe_mgdl"])
    seeds = cfg.experiment["seeds"][:args.seeds] if args.seeds else cfg.experiment["seeds"]
    beta = cfg.audit["coverage_floor_beta"]
    vmax = cfg.audit["violation_rate_max"]

    df = pd.read_parquet(args.cohort)
    feat_cols = [c for c in FEAT_ORDER if c in df.columns]
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    masks, ratio = severity_masks(df, thr_abs)

    cohort_level = {k: {
        "n_cases_cohort": int(m.sum()),
        "icd_agreement_cohort": round(float(y[m].mean()), 4) if m.sum() else None,
    } for k, m in masks.items()}
    newly = masks["relative_only_3x"] & ~masks["absolute_only_PRIMARY_v13"]
    cohort_level["newly_captured_by_relative_limb"] = {
        "n_cases_cohort": int(newly.sum()),
        "icd_agreement_cohort": round(float(y[newly].mean()), 4) if newly.sum() else None,
    }
    print("[severity] cohort-level severity sets:", flush=True)
    for k, v in cohort_level.items():
        print(f"   {k:38s} n={v['n_cases_cohort']:6d} icd_agreement={v['icd_agreement_cohort']}", flush=True)

    per_arm = {}
    for arm, mask in masks.items():
        rows = []
        for seed in seeds:
            tr, te = _subject_split(df, cfg.experiment["test_size"], seed)
            model = PipelineCoupled(cfg.models["pipeline_coupled"]["hidden_layers"],
                                    cfg.models["pipeline_coupled"]["calibrate"], seed=seed)
            model.fit(X[tr], y[tr])
            _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
            tau = metrics.select_tau(model.predict_proba(X[tr][vval]),
                                     cfg.audit["abstain_target_rate"])
            t2 = audit.run_T2(model, X[te], mask[te], tau, enforce=False)
            probs = model.predict_proba(X[te])
            pred = (probs >= 0.5).astype(int)
            sev_te = mask[te]
            rows.append({
                "seed": seed,
                "n_severe": int(sev_te.sum()),
                "icd_agreement_on_severe": (float(y[te][sev_te].mean()) if sev_te.sum() else None),
                "hard_violations": t2.get("hard_violations"),
                "coverage_severe": t2.get("coverage_severe"),
                "violation_rate_deployed": t2.get("violation_rate_deployed"),
            })
            print(f"  [{arm}] seed {seed:2d}: n_sev={rows[-1]['n_severe']:5d} "
                  f"viol={rows[-1]['hard_violations']} "
                  f"vr={rows[-1]['violation_rate_deployed']}", flush=True)

        def arr(k):
            return np.array([r[k] for r in rows if r[k] is not None], float)

        cov, vr = arr("coverage_severe"), arr("violation_rate_deployed")
        cov_p5 = float(np.nanpercentile(cov, 5)) if cov.size else None
        vr_p95 = float(np.nanpercentile(vr, 95)) if vr.size else None
        per_arm[arm] = {
            "n_severe_mean": round(float(arr("n_severe").mean()), 1),
            "icd_agreement_on_severe_mean": round(float(arr("icd_agreement_on_severe").mean()), 4),
            "hard_violations_mean": round(float(arr("hard_violations").mean()), 2),
            "violation_rate_deployed_mean": round(float(vr.mean()), 4),
            "violation_rate_deployed_p95": round(vr_p95, 4) if vr_p95 is not None else None,
            "coverage_severe_mean": round(float(cov.mean()), 4),
            "coverage_severe_p5": round(cov_p5, 4) if cov_p5 is not None else None,
            "certificate_pass_logged": (bool(cov_p5 >= beta and vr_p95 <= vmax)
                                        if cov_p5 is not None else None),
            "per_seed": rows,
        }

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg_tag": "prereg-v1.4",
        "arm_type": "SENSITIVITY (primary specification prereg-v1.3 unchanged)",
        "n_seeds": len(seeds),
        "absolute_threshold_mgdl": thr_abs,
        "relative_factor_stage3": 3.0,
        "not_computable": {
            "rrt_initiation": ("requires MIMIC-IV procedureevents / dialysis items, which are "
                               "not part of the released cohort derivative; in an ICU population "
                               "RRT is the least ambiguous marker of severe AKI, so BOTH arms "
                               "remain conservative"),
            "paediatric_egfr_limb": "MIMIC-IV is adult-only",
        },
        "baseline_proxy_caveat": ("the relative limb uses the within-24h nadir as a proxy for "
                                 "pre-morbid baseline, inheriting the same conservative bias the "
                                 "primary arm discloses"),
        "clinical_review": "NOT clinically reviewed; authors' reading of published KDIGO criteria",
        "cohort_level": cohort_level,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in per_arm.items()},
    }
    (outdir / "severity_sensitivity_v14.json").write_text(json.dumps(payload, indent=1))
    (outdir / "severity_sensitivity_per_seed_v14.json").write_text(
        json.dumps({k: v["per_seed"] for k, v in per_arm.items()}, indent=1))
    print(f"\nSaved -> {outdir/'severity_sensitivity_v14.json'}", flush=True)
    for k, v in payload["arms"].items():
        print(f"  {k:38s} n_sev={v['n_severe_mean']:7.1f} icd_agree={v['icd_agreement_on_severe_mean']:.4f} "
              f"vr={v['violation_rate_deployed_mean']:.4f} cert={v['certificate_pass_logged']}", flush=True)


if __name__ == "__main__":
    main()
