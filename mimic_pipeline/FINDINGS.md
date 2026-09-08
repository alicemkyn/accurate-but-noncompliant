# AuditNeSy / MIMIC-IV — Final Findings (prereg-v1.3)

Frozen result: `results/audit_summary_mimic_final_v1.3.json` (supersedes v1.2.1).
Criteria frozen at tag `prereg-v1.3`; the v1.3 compliance-coverage certificate was
declared before its results.

## Compliance–coverage certificate (v1.3): neither policy passes T2

The coverage floor (β = 0.50) closes the abstention-gaming loophole and yields the
sharpest, gaming-resistant result:

| Policy | Coverage (deployed frac.) | Deployed violation rate | Certificate | Fails on |
|---|---|---|---|---|
| Logged | 0.78 (p5 0.78) | 0.71 (p95 0.74) | FAIL | violation |
| Enforced | 0.23 (p5 0.20) | 0.00 | FAIL | coverage |

**No deployment policy of the ICD-trained model is both compliant (low deployed
violation) and useful (adequate coverage) on the guideline-severe subgroup.** The logged
policy confidently denies AKI on 71% of deployed severe decisions; the enforced policy
reaches zero violations only by deferring 77% of the subgroup, deploying just 23% — below
the floor. The earlier "enforced arm = 3/3" is thereby exposed as a mass-abstention pass;
the certificate makes the audit robust to this Goodhart move. This is the paper's
strongest, least-gameable statement of the accuracy-vs-guideline gap.

**Model-capacity robustness (GBM arm):** the gap is not a weak-model artifact.
Swapping the MLP core for a gradient-boosted core (sklearn HistGradientBoosting,
isotonic-calibrated) gives a slightly HIGHER AUROC (0.882 vs 0.880) yet the SAME
verdict: T2 logged violations mean 1029, logged coverage 0.81 / deployed-violation
0.716 (FAIL), enforced coverage 0.23 (FAIL), probe creatinine-sensitivity 0.079
(even lower than MLP's 0.083). A stronger model fits the biased ICD label even
more faithfully, so it violates the guideline at least as much. Result file:
`results/audit_summary_mimic_gbm_v1.3.json`.

**Threshold sensitivity (M4):** the verdict is robust to the pre-registered
thresholds. Sweeping β ∈ {0.30,0.40,0.50,0.60,0.70} × cap ∈ {0.02,0.05,0.10},
**0/30 grid points pass** (`results/threshold_sensitivity_v1.3.json`). Logged
coverage-p5 = 0.78 / violation-p95 = 0.74 (would need cap > 0.74 to pass);
enforced coverage-p5 = 0.20 / violation-p95 = 0.00 (would need β < 0.20 to pass).
Both boundaries are clinically untenable, so "neither policy passes" is not an
artifact of choosing β = 0.50.

---

## Prior finding record (prereg-v1.2.1)

Frozen result: `results/audit_summary_mimic_final_v1.2.1.json`. Criteria frozen at tag
`prereg-v1.2` (analysis arms at `prereg-v1.2.1`); no criteria changed after results.

## Headline numbers

- Cohort: 378,591 admissions / 167,147 patients; AKI rate (ICD) 18.3%.
- AUROC 0.880 [0.878, 0.882] over 20 subject-level splits.
- **T1 Reproducibility:** 20/20 PASS (NC-D′ flags a genuine reproducibility failure).
- **T3 Safe fallback:** ECE CI-upper median 0.004 (< 0.10) — clean PASS.
- **T2 Constraint compliance (primary / rules LOGGED):** FAIL — mean 1012 confident
  no-AKI decisions on ~1875 decision-time acute-severe cases per split (0/20 seeds pass).
- **T2 (rules ENFORCED / escalate):** PASS by construction, but escalates 57% of
  acute-severe cases to deferral — the safety cost quantifies the logged model's gap.
- **Concept-semantics probe:** creatinine sensitivity 0.083 (model barely depends on the
  constrained feature).
- **Diagnostic — AKI rate within acute-severe subgroup: 0.37.**
- Negative controls: NC-A′→T3, NC-B′→T2, NC-D′→T1 all flag correctly; NC-C′ (era shift)
  does not flag (task exhibits little exploitable era shift; ECE degradation ≤0.01).

## The finding: an accuracy-vs-guideline-compliance gap driven by label validity

Only **37%** of patients who meet the KDIGO clinical definition of severe acute AKI at
decision time (first-24h creatinine ≥ 4.0 mg/dL with an acute rise) carry an ICD AKI code.
KDIGO defines these as AKI by construction; the administrative label disagrees on ~63% of
them. ICD codes are known to under-capture AKI relative to KDIGO lab criteria
(~40–60% sensitivity). The model, trained on ICD labels, faithfully learned to under-call
AKI — hence AUROC 0.880 against ICD yet confident no-AKI decisions on the majority of
guideline-defined severe cases, which T2 flags. The low probe sensitivity (0.083) is
consistent: 24h creatinine weakly predicts the ICD label because that label omits most
high-creatinine AKI.

**Accountability takeaway:** a model that looks accurate against its administrative
ground truth can systematically violate a guideline-grounded safety constraint. Accuracy
alone hides this; AuditNeSy's T2 axis (with the semantics probe) surfaces it. A caveat we
report transparently: the KDIGO 0.3 mg/dL absolute-rise criterion, applied on an already-
severe baseline, admits some chronic-kidney-disease fluctuation, so 0.37 reflects both ICD
under-coding and residual CKD; we did not tune the constraint further to separate them
(the constraint is frozen at v1.2).

## Paper framing (two-task structure)

1. **Pima (diabetes):** clean 3/3 pass — the suite *can* pass when label and constraint
   agree (existing demonstration).
2. **MIMIC-IV (AKI):** real-world scale; the suite catches a subtle, important failure —
   an accuracy-vs-guideline gap rooted in ICD label validity — that accuracy-only
   evaluation would miss.
3. Negative controls on both establish diagnosticity.

This answers the single-dataset objection and demonstrates the protocol in
both regimes (a genuine pass and a genuine, informative failure), and it makes the
reasoning-shortcut / semantics concern concrete and measured.

## Version sequence (transparency, for the paper appendix)

| Tag | T2 constraint | T2 result | Why superseded |
|---|---|---|---|
| v1.0 | absolute creatinine ≥ 4.0 (7-day) | FAIL (mean 4350) | conflated AKI with chronic CKD |
| v1.1 | acute-severe, 7-day window | FAIL (mean 2379) | temporal leakage (used post-decision creatinine) |
| v1.2 | acute-severe, 24h decision window, defer-aware | FAIL (mean 1012) | **final**; residual gap is the finding |
| v1.2.1 | + enforced/logged arms, shortcut diagnostic | — | analysis addendum (no criteria change) |

Each correction was a validity fix derivable on principle, and each was flagged by
AuditNeSy's own T2 axis — itself evidence for the framework's constraint-specification-risk
thesis.
