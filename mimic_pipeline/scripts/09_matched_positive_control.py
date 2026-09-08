"""STEP 9 -- matched positive control: SAME cohort, SAME splits, only the label changes.

WHY THIS EXISTS.  The paper's causal claim is that the ICD labelling convention,
not the model, produces the compliance failure. The evidence for it was a
positive control that retrained on a guideline-derived (KDIGO) label -- but that
arm was built from a separately constructed cohort of 364,566 admissions versus
378,591 for the ICD arm, so a reviewer can object that cohort composition, not
the label, moved the result. That objection is correct and was open.

This script closes it. It intersects the two cohorts on `hadm_id`, keeps only
admissions present in BOTH, and then runs the identical pipeline twice on that
one population -- same rows, same features, same patient-level splits, same
seeds, same tau-selection procedure, same audit configuration -- varying nothing
but the training target. Everything reported is a PAIRED comparison across the
20 matched splits (Wilcoxon signed-rank, exact for n=20).

Two data facts established before running, which the script re-verifies and
records rather than assuming:

  * On the intersection the severe-case mask is IDENTICAL between the two
    cohort builds (5,954 = 5,954), so the audited subpopulation does not move.
  * Nine of the ten features are bit-identical on the intersection; `wbc`
    differs on ~4.5% of rows. The script therefore takes ALL features from the
    ICD build for both arms, so the feature matrix is identical by construction
    and `wbc` cannot contribute to any difference. Only `y` differs.

PRE-REGISTRATION STATUS.  Not pre-registered: added in response to review, after
the unmatched positive control was known. Labelled EXPLORATORY. It can only
qualify or strengthen the existing claim; no registered threshold is changed.
If the effect vanishes on the matched cohort, that is the finding and the paper
must say so.

Usage:
    python scripts/09_matched_positive_control.py \
        --icd-cohort audit_mimic_out/aki_cohort.parquet \
        --kdigo-cohort audit_mimic_kdigo/aki_cohort.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auditnesy_mimic import audit, metrics  # noqa: E402
from auditnesy_mimic.config import load_config  # noqa: E402
from auditnesy_mimic.models import PipelineCoupled  # noqa: E402

from importlib import import_module  # noqa: E402

_bl = import_module("06_baselines")
_subject_split, _auroc, FEAT_ORDER = _bl._subject_split, _bl._auroc, _bl.FEAT_ORDER
# Reuse 06's own factory rather than constructing the model here, so the arm is
# the SAME primary model the paper reports -- not a re-specified lookalike.
_MAKE_PRIMARY = _bl.ARMS["pipeline_coupled_mlp"]


def build_matched(icd_path: str, kdigo_path: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return the intersection frame (features from the ICD build) + the KDIGO label."""
    icd = pd.read_parquet(icd_path)
    kdg = pd.read_parquet(kdigo_path)
    inter = np.intersect1d(icd["hadm_id"].to_numpy(), kdg["hadm_id"].to_numpy())

    ii = icd[icd["hadm_id"].isin(inter)].sort_values("hadm_id").reset_index(drop=True)
    kk = kdg[kdg["hadm_id"].isin(inter)].sort_values("hadm_id").reset_index(drop=True)
    assert (ii["hadm_id"].to_numpy() == kk["hadm_id"].to_numpy()).all(), "join misaligned"

    # verify -- do not assume -- that the audited subpopulation is the same
    sev_i = ii["severe_acute"].to_numpy(bool)
    sev_k = kk["severe_acute"].to_numpy(bool)
    prov = {
        "n_icd_cohort": int(len(icd)),
        "n_kdigo_cohort": int(len(kdg)),
        "n_matched": int(len(ii)),
        "dropped_from_icd_arm": int(len(icd) - len(ii)),
        "dropped_from_kdigo_arm": int(len(kdg) - len(kk)),
        "severe_mask_identical": bool((sev_i == sev_k).all()),
        "n_severe_matched": int(sev_i.sum()),
        "prevalence_icd_label": round(float(ii["aki"].mean()), 4),
        "prevalence_kdigo_label": round(float(kk["aki"].mean()), 4),
        "features_taken_from": "ICD build for BOTH arms, so the design matrix is identical",
    }
    feat_diff = {}
    for f in FEAT_ORDER:
        a = pd.to_numeric(ii[f], errors="coerce").to_numpy(float)
        b = pd.to_numeric(kk[f], errors="coerce").to_numpy(float)
        n_diff = int((~np.isclose(a, b, equal_nan=True)).sum())
        if n_diff:
            feat_diff[f] = {"rows_differing": n_diff,
                            "share": round(n_diff / len(a), 4)}
    prov["feature_differences_between_builds"] = feat_diff or "none"
    return ii, kk["aki"].to_numpy(int), prov


def run_target(label_name, df, X, y, severe, subj, cfg, seeds):
    """Identical pipeline; `y` is the only thing that varies between calls."""
    audit_cfg = dict(cfg.audit)
    audit_cfg["bootstrap_B"] = cfg.experiment["bootstrap_B"]
    per_seed = []
    for seed in seeds:
        tr, te = _subject_split(df, cfg.experiment["test_size"], seed)
        model = _MAKE_PRIMARY(seed, cfg.models["pipeline_coupled"])
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])
        _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(X[tr][vval]),
                                 cfg.audit["abstain_target_rate"])
        t2 = audit.run_T2(model, X[te], severe[te], tau, enforce=False)
        t3 = audit.run_T3(probs, y[te], subj[te], tau, audit_cfg, seed=seed)
        per_seed.append({
            "seed": seed,
            "auroc_own_target": _auroc(y[te], probs),
            "tau": float(tau),
            "hard_violations": t2.get("hard_violations"),
            "n_severe": t2.get("n_severe"),
            "coverage_severe": t2.get("coverage_severe"),
            "violation_rate_deployed": t2.get("violation_rate_deployed"),
            "ece_point": t3["ece_point"],
            "ece_ci_upper": t3["ece_ci"][1],
        })
        print(f"  [{label_name}] seed {seed:2d}: AUROC={per_seed[-1]['auroc_own_target']:.4f} "
              f"viol={t2.get('hard_violations')} "
              f"vrate={t2.get('violation_rate_deployed')} "
              f"cov={t2.get('coverage_severe')}", flush=True)

    def arr(k):
        return np.array([p[k] for p in per_seed if p[k] is not None], float)

    beta = cfg.audit["coverage_floor_beta"]
    vmax = cfg.audit["violation_rate_max"]
    cov_p5 = float(np.nanpercentile(arr("coverage_severe"), 5))
    vr_p95 = float(np.nanpercentile(arr("violation_rate_deployed"), 95))
    return {
        "target": label_name,
        "n_seeds": len(seeds),
        "auroc_own_target_mean": round(float(arr("auroc_own_target").mean()), 4),
        "hard_violations_mean": round(float(arr("hard_violations").mean()), 3),
        "violation_rate_deployed_mean": round(float(arr("violation_rate_deployed").mean()), 4),
        "coverage_severe_mean": round(float(arr("coverage_severe").mean()), 4),
        "ece_point_mean": round(float(arr("ece_point").mean()), 4),
        "certificate": {
            "coverage_severe_p5": round(cov_p5, 4),
            "violation_rate_deployed_p95": round(vr_p95, 4),
            "certificate_pass": bool(cov_p5 >= beta and vr_p95 <= vmax),
        },
        "per_seed": per_seed,
    }


def paired(a, b, label):
    """Two-sided Wilcoxon on matched seeds; degenerate cases reported, not hidden."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return {"comparison": label, "median_difference": 0.0, "p_value": None,
                "note": "all per-seed differences are exactly zero; test degenerate"}
    st, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"comparison": label,
            "median_difference": round(float(np.median(d)), 4),
            "statistic": float(st), "p_value": float(p), "n_pairs": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd-cohort", required=True)
    ap.add_argument("--kdigo-cohort", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # the registered seed list, exactly as 06 reads it -- not a re-derived range
    seeds = cfg.experiment["seeds"][:args.seeds] if args.seeds else cfg.experiment["seeds"]

    df, y_kdigo, prov = build_matched(args.icd_cohort, args.kdigo_cohort)
    print(f"[matched] intersection n={prov['n_matched']:,} "
          f"(dropped {prov['dropped_from_icd_arm']:,} from ICD build, "
          f"{prov['dropped_from_kdigo_arm']:,} from KDIGO build)", flush=True)
    print(f"[matched] severe mask identical: {prov['severe_mask_identical']} "
          f"(n_severe={prov['n_severe_matched']:,})", flush=True)
    print(f"[matched] prevalence: ICD {prov['prevalence_icd_label']} "
          f"vs KDIGO {prov['prevalence_kdigo_label']}", flush=True)
    print(f"[matched] feature differences between builds: "
          f"{prov['feature_differences_between_builds']}", flush=True)

    X = df[FEAT_ORDER].to_numpy(float)
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    severe = df["severe_acute"].to_numpy(bool)
    subj = df["subject_id"].to_numpy()
    y_icd = df["aki"].to_numpy(int)

    icd_res = run_target("ICD label", df, X, y_icd, severe, subj, cfg, seeds)
    kdg_res = run_target("KDIGO label", df, X, y_kdigo, severe, subj, cfg, seeds)

    def col(res, k):
        return [p[k] for p in res["per_seed"]]

    tests = [
        paired(col(icd_res, "violation_rate_deployed"), col(kdg_res, "violation_rate_deployed"),
               "deployed violation rate: ICD - KDIGO"),
        paired(col(icd_res, "hard_violations"), col(kdg_res, "hard_violations"),
               "hard violations: ICD - KDIGO"),
        paired(col(icd_res, "coverage_severe"), col(kdg_res, "coverage_severe"),
               "severe coverage: ICD - KDIGO"),
        paired(col(icd_res, "auroc_own_target"), col(kdg_res, "auroc_own_target"),
               "AUROC against own target: ICD - KDIGO"),
    ]
    for t in tests:
        print(f"[paired] {t['comparison']:44s} median={t['median_difference']:+.4f} "
              f"p={t.get('p_value')}", flush=True)

    # ---------- how much of the positive control's effect is arithmetic? ----------
    # A T2 violation is a confident NO-AKI decision on a constraint-applicable case, so a
    # model fitted to a target that already says "AKI" on those cases cannot violate much
    # regardless of what it learns. The KDIGO target is creatinine-derived and so is the
    # constraint; cohort.py flags that label CIRCULAR for exactly this reason. This block
    # measures the overlap and bounds the share of the effect that is base-rate arithmetic
    # rather than learned behaviour, so the control is reported with its ceiling attached.
    # both targets live on the SAME matched rows, so the comparison is exact
    _pos_i = float(y_icd[severe].mean())
    _pos_k = float(y_kdigo[severe].mean())
    _n_sev = int(severe.sum())
    _implied = (1.0 - _pos_i) - (1.0 - _pos_k)
    _observed = (icd_res["violation_rate_deployed_mean"]
                 - kdg_res["violation_rate_deployed_mean"])
    ceiling = {
        "why_this_exists": (
            "The KDIGO target and the T2 constraint are both derived from the creatinine "
            "trajectory; cohort.py flags this label CIRCULAR. Reporting the control without "
            "this bound would overstate it."),
        "constraint_applicable_cases": _n_sev,
        "note_on_matching": ("both targets are evaluated on the same matched rows, so the two "
                             "positive rates below are computed on an identical denominator"),
        "target_positive_rate_among_constraint_applicable": {
            "icd_label": round(_pos_i, 4), "kdigo_label": round(_pos_k, 4)},
        "target_says_no_aki_there": {
            "icd_label": round(1.0 - _pos_i, 4), "kdigo_label": round(1.0 - _pos_k, 4)},
        "coincidence_of_kdigo_target_and_constraint": round(_pos_k, 4),
        "violation_drop_implied_by_target_base_rates": round(_implied, 4),
        "violation_drop_observed": round(_observed, 4),
        "observed_over_implied": round(_observed / _implied, 3) if _implied else None,
        "share_attributable_to_base_rate": round(_implied / _observed, 3) if _observed else None,
        "reading": (
            f"On the constraint-applicable subgroup the KDIGO target is already positive for "
            f"{_pos_k:.1%} of cases against {_pos_i:.1%} for the ICD target, so a model fitted to "
            f"it starts near-compliant by construction. Base rates alone imply a violation-rate "
            f"drop of {_implied:.3f}; the observed drop is {_observed:.3f}. Roughly "
            f"{_implied / _observed:.0%} of the effect is therefore target arithmetic and the "
            f"remainder is learned. This is a valid POSITIVE CONTROL -- it shows the constraint "
            f"is triggerable and the harness responds -- but it is NOT clean causal evidence "
            f"that a guideline-faithful target fixes compliance, and it should not be promoted "
            f"to the paper's primary causal claim."),
    }
    print(f"\n[ceiling] KDIGO target already positive on {_pos_k:.1%} of constraint-applicable "
          f"cases (ICD: {_pos_i:.1%})", flush=True)
    print(f"[ceiling] implied drop {_implied:.4f} vs observed {_observed:.4f} "
          f"-> {_implied / _observed:.0%} of the effect is base-rate arithmetic", flush=True)

    payload = {
        "prereg_tag": "prereg-v1.5",
        "analysis_status": ("EXPLORATORY -- not pre-registered; added in response to review to "
                           "test whether the positive control's effect survives when cohort "
                           "composition is held fixed. Same rows, same features, same splits, "
                           "same seeds; the training target is the only thing that varies. "
                           "No registered threshold was changed."),
        "matching_provenance": prov,
        "effect_ceiling": ceiling,
        "arms": {"icd_label": {k: v for k, v in icd_res.items() if k != "per_seed"},
                 "kdigo_label": {k: v for k, v in kdg_res.items() if k != "per_seed"}},
        "paired_tests": tests,
        "per_seed": {"icd_label": icd_res["per_seed"], "kdigo_label": kdg_res["per_seed"]},
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "matched_positive_control.json").write_text(json.dumps(payload, indent=1))
    print(f"\nSaved -> {outdir/'matched_positive_control.json'}", flush=True)


if __name__ == "__main__":
    main()
