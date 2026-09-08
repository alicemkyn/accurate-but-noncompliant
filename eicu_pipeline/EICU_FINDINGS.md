# eICU results (branch `eicu-experiment`) — multi-center replication

Frozen aggregate: `results/audit_summary_eicu_v1.3.json` (DUA-safe; no patient rows).
Same harness, same frozen prereg (`prereg-v1.3`); only the cohort is eICU-specific
(diagnosis-string AKI label, hospital-based shift). MLP core.

## Cohort
- 169,825 unit stays / 147,832 patients / **205 hospitals** (genuinely multi-center).
- AKI rate (diagnosis-string label) 10.9%; acute-severe@24h = 4,142 (well-powered, ≥500).
- aki_rate_in_severe_acute = 0.43 (cf. MIMIC 0.37): the diagnosis-string label under-captures
  KDIGO-severe AKI at least as much as MIMIC's ICD label.

## The finding replicates — and is stronger

| Quantity | MIMIC | eICU |
|---|---|---|
| AUROC | 0.880 | 0.852 |
| T1 / T3 | pass / pass | pass / pass (ECE CI-upper 0.006) |
| Deployed-violation rate (severe) | 0.716 | **0.928** |
| Compliance–coverage certificate | neither passes | **neither passes** |
| — logged | fails (violation) | fails (coverage 0.39 and violation 0.93) |
| — enforced | fails (coverage 0.23) | fails (coverage 0.03) |
| probe creatinine-sensitivity | 0.083 | 0.153 |

Same accuracy-vs-label-validity failure mode on a different EHR system across 205 hospitals:
the model confidently denies AKI on 93% of deployed guideline-severe cases, and no deployment
policy is both compliant and adequately covering. This kills the single-institution and
"MIMIC quirk" critiques.

## Bonus: the T3 shift axis works on a real multi-center shift
Training on half the hospitals and evaluating on held-out hospitals, T3 shift **passes**:
degradation 3/3 (ECE stable, even improves), abstention-direction 3/3 (defers more on unseen
hospitals). On MIMIC the era-shift was weak (direction 0/3); the hospital shift is the genuine
distribution shift the axis was designed for.

## Honest caveat: NC-A did not flag on eICU
The uncalibrated negative control (fixed 3× probability sharpening) reached ECE CI-upper 0.088,
just **below** the 0.10 bound, so it did NOT trip T3 on eICU (it did on MIMIC, 0.102). The base
eICU model is better calibrated, so the fixed sharpening is sub-threshold here. NC-A is an
illustrative control, not a registered pass criterion; its severity is dataset-dependent. Report
this transparently: on eICU a stronger control is needed to exercise T3. NC-B (T2), NC-C
(shift), NC-D (T1) all flag correctly.

## Decision gate: PASS -> worth integrating
Cohort sane, finding interpretable and stronger, shift axis validated. Integrate as a
multi-center validation subsection; report the NC-A caveat honestly.

## Notes for integration
- Fix the dataset string in the summary writer (currently inherits the MIMIC label; corrected
  in the saved results file).
- eICU AKI label = diagnosis-string proxy (not ICD); state this explicitly.
- Ethics/data statement must add eICU-CRD (PhysioNet credentialed, separate DUA).

## C-lite follow-up (mechanism replication + NC-A severity)

**Probe curve replicates the mechanism (both EHRs).** Full-model predicted risk is
non-monotonic in creatinine on both datasets: it peaks (MIMIC cr=2.0, risk 0.51; eICU
cr=2.75, risk 0.27) then DECLINES through the guideline-severe zone. The wrong-direction
use of creatinine at the severe end replicates across two independent EHR systems — a
mechanism replication, stronger than a matching violation rate. Caveat: on MIMIC the full
model crosses BELOW the constraint-blind model in the severe zone (cr>=4.5); on eICU it does
not, because eICU's lower AKI base rate (10.9%) gives a lower blind-model baseline (~0.10 vs
0.19). The wrong-direction decline is present in both; the relative crossing differs by base
rate. Files: results/probe_curve_{mimic,eicu}.json.

**NC-A severity as a portable measure (k*).** NC-A stretches probabilities by a factor k;
we sweep k and report k*, the minimum severity whose ECE CI-upper reaches the 0.10 bound
(same measurement as the pre-registered criterion). Over 20 seeds:
- MIMIC-IV: k* median 3.125 (range 3.0-3.25) -- the registered k=3.0 sits right at the bound.
- eICU: k* median ~8.0 (range 7.25-8.0); on 15/20 seeds k=8 still did not breach the bound.
The better-calibrated eICU model needs ~2.5x more perturbation for the same calibration
damage. This makes NC-A a comparable measurement rather than a binary flag: a fixed-severity
synthetic control is not portable across datasets, since the same perturbation produces
different calibration damage depending on the base predictive distribution. The pass bound
(0.10) is unchanged; k* characterizes the control instrument, not the criterion, and is
reported for both datasets (a comparison, not a rescue). File: results/ncA_kstar.json.

Small methodological point for the Discussion: synthetic negative controls should be reported
with the severity required to trigger them, not as binary pass/fail flags.
