"""Build claim_verification_w3phiai.csv: every number the W3PHIAI-27 paper quotes,
traced to a results file and a JSON key path, read programmatically (never typed).

Columns: claim_text, value, source_file, json_key_path, status.
`status` is "read" when the key resolved, "MISSING" when it did not (the row is kept
so the gap is visible rather than silently dropped).

    python scripts/build_claim_verification_w3phiai.py --out ../claim_verification_w3phiai.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # mimic_pipeline/
REPO = ROOT.parent

SOURCES = {
    "vr": ROOT / "results" / "violation_rate_by_renal_history.json",
    "adj": ROOT / "results" / "clinician_adjudication.json",
    "abs": ROOT / "results" / "adjudication_by_stratum.json",
    "eicu": REPO / "eicu_pipeline" / "results" / "heldout_hospital_certificate.json",
}


def _get(obj, path):
    """Resolve 'a.b[2].c' against nested dicts/lists."""
    for tok in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if tok.startswith("["):
            obj = obj[int(tok[1:-1])]
        else:
            obj = obj[tok]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "claim_verification_w3phiai.csv"))
    ap.add_argument("--eicu", default=None, help="override path to heldout_hospital_certificate.json")
    args = ap.parse_args()
    if args.eicu:
        SOURCES["eicu"] = Path(args.eicu)

    data = {k: (json.loads(p.read_text()) if p.exists() else None) for k, p in SOURCES.items()}

    claims = []   # (claim_text, source_key, json_key_path)
    S = claims.append

    # ---- restricted certificate (results/violation_rate_by_renal_history.json) ----
    S(("Restricted (non-dialysis) cohort: deployed violation rate p95 across seeds", "vr", "restricted_certificate.violation_rate.p95"))
    S(("Restricted cohort: deployed violation rate mean across seeds", "vr", "restricted_certificate.violation_rate.mean"))
    S(("Restricted cohort: coverage on severe cases p5 across seeds", "vr", "restricted_certificate.coverage.p5"))
    S(("Restricted cohort: coverage on severe cases mean across seeds", "vr", "restricted_certificate.coverage.mean"))
    S(("Restricted cohort: severe cases per split, min", "vr", "restricted_certificate.n_severe_per_split.min"))
    S(("Restricted cohort: severe cases per split, mean", "vr", "restricted_certificate.n_severe_per_split.mean"))
    S(("Restricted cohort: severe cases per split, max", "vr", "restricted_certificate.n_severe_per_split.max"))
    S(("Restricted cohort: certificate passes", "vr", "restricted_certificate.certificate_pass"))
    S(("Restricted cohort: certificate fails on limb", "vr", "restricted_certificate.fails_on"))
    S(("Restricted cohort: coverage floor beta", "vr", "restricted_certificate.beta"))
    S(("Restricted cohort: violation cap", "vr", "restricted_certificate.cap"))
    S(("Headline: violation rate, surviving (non-dialysis) cases, n_deployed-weighted", "vr", "headline_comparison.violation_rate_surviving_cases"))
    S(("Headline: violation rate, prior-renal-replacement stratum", "vr", "headline_comparison.violation_rate_prior_renal_replacement"))
    S(("Headline: mean severe cases per split, prior-renal-replacement stratum", "vr", "headline_comparison.n_severe_mean_prior_renal_replacement"))
    S(("Headline: mean severe cases per split, surviving strata", "vr", "headline_comparison.n_severe_mean_surviving"))
    S(("Headline: mean deployed decisions per split, surviving strata", "vr", "headline_comparison.n_deployed_mean_surviving"))

    # ---- NEW (Task B): subject-clustered CIs ----
    for stat, label in (("violation_rate_deployed", "deployed violation rate"),
                        ("coverage_severe", "coverage on severe cases"),
                        ("icd_agreement_on_severe", "ICD-label agreement on severe cases")):
        base = f"restricted_cohort_cluster_ci.per_seed_median.{stat}"
        S((f"Restricted cohort, {label}: median across seeds of per-seed point estimate", "vr", base + ".point_median"))
        S((f"Restricted cohort, {label}: median across seeds of subject-clustered 95% CI lower bound", "vr", base + ".lo_median"))
        S((f"Restricted cohort, {label}: median across seeds of subject-clustered 95% CI upper bound", "vr", base + ".hi_median"))
        base = f"restricted_cohort_cluster_ci.pooled_across_seed_test_decisions.{stat}"
        S((f"Restricted cohort, {label}: pooled-across-seed-test-decisions estimate (descriptive)", "vr", base + ".point"))
        S((f"Restricted cohort, {label}: pooled subject-clustered 95% CI lower (descriptive)", "vr", base + ".lo"))
        S((f"Restricted cohort, {label}: pooled subject-clustered 95% CI upper (descriptive)", "vr", base + ".hi"))
    S(("Cluster bootstrap: number of resamples B", "vr", "restricted_cohort_cluster_ci.B"))
    S(("Cluster bootstrap: RNG seed", "vr", "restricted_cohort_cluster_ci.seed"))
    S(("Cluster bootstrap: cluster variable", "vr", "restricted_cohort_cluster_ci.cluster_variable"))
    S(("Cluster bootstrap: stacked severe decisions across 20 seed test sets", "vr", "restricted_cohort_cluster_ci.pooled_across_seed_test_decisions.n_decisions_stacked"))
    S(("Cluster bootstrap: unique subjects with a severe surviving case across seeds", "vr", "restricted_cohort_cluster_ci.pooled_across_seed_test_decisions.n_unique_subjects_with_severe_surviving_case"))
    S(("Cluster bootstrap: subjects appearing in more than one seed's test set", "vr", "restricted_cohort_cluster_ci.pooled_across_seed_test_decisions.n_subjects_in_more_than_one_seed_test_set"))

    # ---- eICU held-out-hospital certificate ----
    S(("eICU held-out hospitals: AUROC mean across seeds", "eicu", "auroc_heldout.mean"))
    S(("eICU held-out hospitals: AUROC min across seeds", "eicu", "auroc_heldout.min"))
    S(("eICU held-out hospitals: AUROC max across seeds", "eicu", "auroc_heldout.max"))
    S(("eICU held-out: coverage on severe cases, mean (logged policy)", "eicu", "certificate.logged_policy.coverage_severe_mean"))
    S(("eICU held-out: coverage on severe cases, p5 (logged policy)", "eicu", "certificate.logged_policy.coverage_severe_p5"))
    S(("eICU held-out: deployed violation rate, mean (logged policy)", "eicu", "certificate.logged_policy.violation_rate_deployed_mean"))
    S(("eICU held-out: deployed violation rate, p95 (logged policy)", "eicu", "certificate.logged_policy.violation_rate_deployed_p95"))
    S(("eICU held-out: certificate passes (logged policy)", "eicu", "certificate.logged_policy.certificate_pass"))
    S(("eICU held-out: certificate fails on limb (logged policy)", "eicu", "certificate.logged_policy.fails_on"))
    S(("eICU partition: training hospitals", "eicu", "hospital_partition.n_train_hospitals"))
    S(("eICU partition: held-out hospitals", "eicu", "hospital_partition.n_heldout_hospitals"))
    S(("eICU partition: training stays", "eicu", "hospital_partition.n_train_stays"))
    S(("eICU partition: held-out stays", "eicu", "hospital_partition.n_heldout_stays"))
    S(("eICU partition: held-out severe cases", "eicu", "hospital_partition.n_heldout_severe"))
    S(("eICU partition: RNG seed", "eicu", "hospital_partition.rng_seed"))
    S(("eICU per-site: held-out sites with severe cases", "eicu", "per_site.n_sites_with_severe_cases"))
    S(("eICU per-site: sites clearing the registered subgroup floor", "eicu", "per_site.n_sites_determinate"))
    S(("eICU per-site: registered subgroup floor", "eicu", "per_site.subgroup_floor"))
    S(("eICU: number of seeds", "eicu", "n_seeds"))

    # ---- clinician adjudication ----
    S(("Adjudication: cases decided (non-blank)", "adj", "n_cases_adjudicated"))
    S(("Adjudication: cases left blank by every reviewer", "adj", "n_cases_left_blank_by_every_reviewer"))
    S(("Adjudication: number of reviewers", "adj", "n_reviewers"))
    for cell in ("A_severe_coded", "B_severe_uncoded", "C_near_threshold", "D_clearly_not_severe"):
        S((f"Adjudication cell {cell}: decided cases", "adj", f"per_cell.{cell}.n"))
        S((f"Adjudication cell {cell}: clinician-vs-rule agreement", "adj", f"per_cell.{cell}.clinician_vs_rule_agreement"))
        S((f"Adjudication cell {cell}: clinician-vs-code agreement", "adj", f"per_cell.{cell}.clinician_vs_code_agreement"))
        S((f"Adjudication cell {cell}: agreeing decided cases (from cross-tab)", "abs", f"per_cell.{cell}.n_agree_with_rule"))
        S((f"Adjudication cell {cell}: blank cases (from cross-tab)", "abs", f"per_cell.{cell}.n_blank"))
    S(("Adjudication: Cohen's kappa, clinician vs rule", "adj", "overall.clinician_vs_rule_kappa"))
    S(("Adjudication: Cohen's kappa, clinician vs administrative code", "adj", "overall.clinician_vs_code_kappa"))
    S(("Adjudication: declared floor, cell-B agreement", "adj", "declared_thresholds.min_agreement_cell_B"))
    S(("Adjudication: declared floor, overall kappa", "adj", "declared_thresholds.min_kappa_overall"))

    # ---- NEW (Task A): adjudication x stratum ----
    S(("Cross-tab: cases", "abs", "n_cases"))
    S(("Cross-tab: decided", "abs", "n_decided"))
    S(("Cross-tab: blank", "abs", "n_blank"))
    for st in ("prior_renal_replacement", "no_prior_renal_replacement", "prior_status_unobservable"):
        for fld, lab in (("n_cases", "cases"), ("n_decided", "decided"), ("n_blank", "blank"),
                         ("n_agree_with_rule", "agree with rule"),
                         ("agreement_on_decided", "agreement on decided"), ("blank_share", "blank share")):
            S((f"Cross-tab, renal history {st}: {lab}", "abs", f"per_renal_history.{st}.{fld}"))
    for tr in ("A1_elevated_at_arrival", "A2_rise_from_near_normal", "A3_rise_from_elevated_base"):
        for fld, lab in (("n_cases", "cases"), ("n_decided", "decided"), ("n_blank", "blank"),
                         ("n_agree_with_rule", "agree with rule"),
                         ("agreement_on_decided", "agreement on decided"), ("blank_share", "blank share")):
            S((f"Cross-tab, trajectory {tr}: {lab}", "abs", f"per_trajectory.{tr}.{fld}"))
    abs_d = data["abs"] or {}
    for tbl in ("cell_x_renal_history", "cell_x_trajectory"):
        for k in (abs_d.get(tbl) or {}):
            for fld, lab in (("n_cases", "cases"), ("n_decided", "decided"),
                             ("n_agree_with_rule", "agree with rule"),
                             ("agreement_on_decided", "agreement on decided")):
                S((f"Cross-tab {tbl} {k}: {lab}", "abs", f"{tbl}.{k}.{fld}"))
    S(("Cross-tab: number of small cells (n_decided < 10)", "abs", "small_cell_warning"))

    rows = []
    for text, src, path in claims:
        d = data[src]
        rel = SOURCES[src]
        try:
            rel = rel.relative_to(REPO)
        except ValueError:
            pass
        if d is None:
            rows.append((text, "", str(rel), path, "MISSING (source file absent)"))
            continue
        try:
            v = _get(d, path)
        except (KeyError, IndexError, TypeError):
            rows.append((text, "", str(rel), path, "MISSING (key not found)"))
            continue
        if isinstance(v, list):
            v, path = len(v), path + " (len)"
        rows.append((text, "" if v is None else v, str(rel), path, "read"))

    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["claim_text", "value", "source_file", "json_key_path", "status"])
        w.writerows(rows)
    n_missing = sum(r[4] != "read" for r in rows)
    print(f"{len(rows)} claims written -> {out}; missing: {n_missing}")
    for r in rows:
        if r[4] != "read":
            print("  MISSING:", r[0], "|", r[3])


if __name__ == "__main__":
    main()
