# AuditNeSy — Pre-Registration of Pass Criteria (v1.4, RATIFIED)

> **Note on scope (added 2026-11-01).** This log records the full history of the work, including
> an earlier submission of the manuscript and the reviews it received. Venue names and dates are
> kept because the log is a dated record; entries are appended, never rewritten.

**Status:** RATIFIED — frozen at tag `prereg-v1.3` for all pass criteria; issued as **v1.4** to record
the completed deviations log (§8) and to mark declared-but-unimplemented machinery (§9).
Supersedes `prereg-v1.0`…`v1.3`. **No pass threshold, criterion, or definition is changed in v1.4.**
v1.4 is a disclosure-only revision: it adds deviation rows for undelivered registered items, for the
τ selection procedure as actually implemented, and for the negative controls that did not flag; and it
annotates configuration keys that were declared here but are not read by any released code.
**v1.3 addition — compliance–coverage certificate (declared before its results).** The
defer-aware T2 (below) has an abstention-gaming loophole: an enforced policy can reach zero
violations by deferring the whole acute-severe subgroup. We close it with a pre-declared
**coverage floor**: passing T2 requires that a policy DEPLOY (not defer) at least
β = 0.50 of acute-severe decisions AND keep the hard-violation rate among deployed decisions
≤ 0.05 (conservative across-seed bounds). This *tightens* the audit; it does not relax any
threshold. See §3 T2 (certificate) and §8.
**T2 constraint — principled specification.** The T2 hard constraint is a check on
the **deployed, defer-aware action** using **decision-time evidence only**:

> For a patient whose first-24h labs already show *acute severe* AKI, the system must not
> output a **confident** *no-AKI* decision (predict no-AKI with confidence ≥ τ). If the
> model is uncertain it defers — the safe action — which is not a violation.

Two earlier operationalizations were found invalid on principle (not on results) and are
retained as findings (§8): v1.0 used an absolute creatinine threshold (conflates AKI with
chronic kidney disease); v1.1 used a 7-day creatinine peak (temporal leakage — referenced
information the model had not seen at decision time). **Both were flagged by AuditNeSy's
own T2 axis** — the constraint-specification risk the framework exists to surface. No pass
threshold (ECE bound, τ, satisfaction levels) has ever changed across versions; only the
clinical/logical *scope and action-basis* of the T2 constraint. This v1.2 specification is
declared **before** its results are observed and is intended as final.
**Applies to:** MIMIC-IV extension experiments (paper v9 → v10).
**Freeze protocol:** Before the FIRST evaluation run on any MIMIC-IV test fold, this file must be
(1) ratified, (2) committed, (3) timestamped via OSF registration OR a signed annotated git tag
(`git tag -s prereg-v1.0`) pushed to a public remote. Cohort extraction and model *training* may
proceed before the freeze; *test-fold evaluation* may not.

**Post-freeze rule:** any change requires a new registered version with a stated reason;
results are always reported against the version in force when the run was executed.

---

## 1. Tasks & Primary Metrics

| Task | Label | Primary discrimination metric (reported, not gated) |
|---|---|---|
| A — ICU in-hospital mortality | death before discharge, first ICU stay, features from first 24h | AUROC — **NOT DELIVERED (§8);** never run, no result reported |
| B — AKI within 48h | KDIGO stage ≥ 1 (creatinine criteria; urine-output criterion EXCLUDED — declared here) | AUROC |
| P — Pima (contrast) | as in v7 | AUROC |

Discrimination metrics are contextual only; no axis gates on AUROC.

## 2. Experimental Units

- 20 seeds × repeated 70/30 patient-level splits per task/model (grouped by patient via `GroupShuffleSplit`; **not** stratified on the label — the word "stratified" appeared in v1.0–v1.3 and was never implemented, see §8 row 10. Immaterial at n≈113k test decisions and 18.3% prevalence).
- 95% CIs: patient-level bootstrap, B = 1000.
- Minimum evaluable cohort for any subgroup/shift criterion: **n ≥ 500 decisions**; below this the criterion is reported as UNDETERMINED, not PASS.

## 3. Axis Pass Criteria

### T1 — Reproducibility
- **Criterion:** 0 prediction mismatches AND model-version hash match on replay, on every seed (20/20).
- Replay environment: pinned lockfile (exact library versions recorded in repo), fixed BLAS threads, deterministic ops enforced where the framework allows.
- NC-D′ (unpinned environment) is expected to FAIL this axis; that failure is a validity check of T1, not a result against the primary models.

### T2 — Constraint Compliance
- **Hard-constraint applicability (v1.2, final):** the constraint applies to **decision-time
  acute severe AKI**: within the first-24h feature window, peak creatinine `≥ 4.0 mg/dL`
  **AND** an acute rise in that same window (`≥ 1.5 × nadir` OR `≥ 0.3 mg/dL` absolute rise,
  KDIGO 2012). This excludes stable chronic kidney disease AND uses only information the
  model saw at decision time (no post-decision creatinine).
- **Violation definition (v1.2, defer-aware):** a violation is a **confident** no-AKI
  decision on an applicable case — `pred = no-AKI` AND `confidence = max(p,1−p) ≥ τ`. An
  uncertain decision (confidence < τ) defers and is **not** a violation, consistent with
  the T3 abstention-as-safety design. Deferred severe cases are reported for transparency.
- **Hard-constraint criterion:** 0 violations on the evaluation stream, per seed. Report rule-of-three 95% upper bound (3/n) alongside every 0/n.
- **Compliance–coverage certificate (v1.3):** to close abstention-gaming, T2 is certified on the acute-severe subgroup as a joint criterion evaluated per deployment policy (rules *logged* vs *enforced*):
  - *coverage* = fraction of acute-severe cases the policy DEPLOYS (a non-deferred decision); and
  - *violation rate* = hard violations among DEPLOYED acute-severe decisions.
  - **PASS ⟺** 5th-percentile-across-seeds coverage ≥ **β = 0.50** AND 95th-percentile-across-seeds violation rate ≤ **0.05**. A policy cannot pass by deferring the subgroup (coverage floor) nor by deploying confident violations (violation cap).
- **Soft constraints:** logged and reported as firing fractions with CIs; never gated.
- **Differentiable (LTN) evidence form:** per-decision satisfaction converted categorically at **s ≥ 0.5** (violation if s < 0.5 on a hard-constraint-relevant case). Sensitivity sweep over s ∈ {0.3, 0.5, 0.7, 0.9} reported regardless of outcome.
  - ~~ADDITIONALLY mean satisfaction on the hard-constraint-relevant subset must be ≥ 0.90.~~ **WITHDRAWN in v1.4 — NOT IMPLEMENTED.** The corresponding key `config.yaml:models.differentiable_ltn.sat_mean_min: 0.90` is declared but never read by any released code, so this sub-criterion was never evaluated and no result is gated on it. It is withdrawn rather than left standing, so that the criteria listed here are exactly the criteria the harness computes. (The related `n_axioms: 2` key is likewise unread: the LTN arm's axiom count is fixed in code, not configured.)
- **Probabilistic (SPL) evidence form:** hard-constraint probability must satisfy **P(constraint) ≥ 0.99** per decision; violations counted below that line. Sensitivity sweep over {0.95, 0.99, 0.999}. **NOT IMPLEMENTED / NOT EVALUATED (v1.4):** the SPL arm was not delivered (§8), so this criterion has never been applied to any model.

### T2s — Semantics Probe (ADVISORY — never gates)
- Monotonicity consistency: for each hard constraint, counterfactual sweep of the constraint feature over its clinical range on n = 200 sampled decisions; report the fraction of decisions where model output direction matches declared constraint semantics. No pass threshold in v8 (screening instrument; thresholding deferred until the probe is validated).

### T3 — Safe Fallback
- **ECE:** 10 equal-width bins over [0,1], on top-label confidence `max(p, 1−p)`. **Criterion: bootstrap 95% CI upper bound of ECE < 0.10** per task (in-distribution).
  - *Effective resolution (disclosed v1.4):* because `max(p, 1−p) ∈ [0.5, 1]`, only the upper **5 of the 10** bins are reachable; the released `figures/figdata/reliability.json` declares `n_bins=10` and contains exactly 5 populated bins (midpoints 0.55–0.95). The binning is equal-width as declared, but the estimator's effective resolution is 5 bins. Coarser bins average within-bin reliability gaps and therefore **understate** ECE — the bias runs toward the T3 PASS, i.e. in favor of the audited model. The estimator is unchanged (it is pre-registered); the property is disclosed.
  - Rationale: 0.10 is looser than v7's observed 0.070 point estimate because the criterion is now on the CI upper bound (strictly harder form); clinically motivated by the ~1-decile confidence-reliability gap tolerable in advisory (non-autonomous) risk flags.
- **Abstention policy:** defer when max(p, 1−p) < τ. **REGISTERED:** τ is selected ONCE on the validation folds to target a deferral rate of ≤ 15% in-distribution, then frozen before test evaluation; the selection procedure (not the value) is what is registered, and the frozen value is recorded in `preregistration_lock.md` at freeze time.
  - **AS IMPLEMENTED (deviation, §8 row 7):** τ is re-selected on the validation fold *inside* the seed loop (`experiment.py` L63–65), producing 19 distinct values across the 20 seeds (range 0.6441–0.6587). `config.yaml:abstain_tau_frozen` remains `null` and **no `preregistration_lock.md` was ever written**. Selection uses validation folds only and never touches the test fold, so no test-fold information enters τ; but the registered single-frozen-value artifact does not exist. Reported as a deviation, not retroactively re-registered.
- **Shift criteria** (each shift axis evaluated independently):
  1. Abstention rate on shifted cohort ≥ abstention rate in-distribution (point estimates; direction only).
  2. ECE degradation under shift ≤ **+0.05 absolute** (CI-upper-bound comparison).

## 4. Distribution Shift Axes (declared)

| Axis | Train | Evaluate | Status |
|---|---|---|---|
| Era | anchor_year_group `2008 - 2010` + `2011 - 2013` | `2017 - 2019` + `2020 - 2022` | implemented (`run_shift_eval`) |
| Care unit | MICU | SICU + CCU | declared; requires MIMIC-IV **icu** module (not loaded in v1.1); reported if implemented, else noted as future work |

Middle era `2014 - 2016` is held out as a buffer between train and eval periods.
Glucose-defined subsets are NOT used for shift (removed per review; circular with constraint features).

## 5. Aggregate Decision Rule

- **Primary (registered) outcome per axis/task/model:** pooled CI-based criterion above.
- **Secondary (reported, not gated):** per-seed pass stability, x/20. A per-seed stability below 18/20 is discussed in the paper regardless of pooled outcome.
- Axes are never collapsed into a single score (unchanged from v7).

## 6. Negative Controls v2 (expected outcomes declared)

| Control | Construction | Must flag on | Must NOT flag on |
|---|---|---|---|
| NC-A′ | no calibration step + class reweighting | T3 (ECE) | T1, T2 |
| NC-B′ | constraint features excluded from training | T2 | T1 |
| NC-C′ | trained on 2008–2013 only, no abstention re-tuning | T3 (shift) | T1, T2 |
| NC-D′ | unpinned environment / nondeterministic ops | T1 | — |

A negative control failing to flag on its target axis counts AGAINST the framework and is reported as such.

**Observed outcomes (v1.4 disclosure).** Two controls did not flag; both are booked against the framework per the rule above (§8 rows 8–9):

| Control | Target | MIMIC-IV | eICU | Verdict |
|---|---|---|---|---|
| NC-A′ uncalibrated | T3 (ECE) | FLAGGED (CI-upper 0.102) | **DID NOT FLAG** (CI-upper 0.0877 < 0.10) | T3 unvalidated on eICU |
| NC-B′ constraint-blind | T2 | FLAGGED (537 viol. / 1875 severe) | FLAGGED (338 / 1260) | validated |
| NC-C′ stale/no re-tune | T3-shift | **DID NOT FLAG** | **DID NOT FLAG** | T3-shift unvalidated on BOTH; excluded from audit-readiness |
| NC-D′ unpinned env | T1 | FLAGGED (2076 mismatches) | FLAGGED (529) | validated |

NC-C′ additionally did not flag in the v1.2.1 and GBM MIMIC-IV runs — i.e. it has never flagged in any released run. The **T3-shift sub-criterion is therefore treated as UNVALIDATED BY NEGATIVE CONTROL and does not count toward the per-axis audit-readiness figure on either database.** An axis whose control never fires is not distinguishable from an axis that cannot fail.

## 7. Models Under Test

1. Pipeline-coupled: GBM + explicit rule layer (primary), MLP variant (for NC-D′). — **DELIVERED** (MLP primary + GBM robustness arm, both databases).
2. Differentiable: LTN, ≥ 2 axioms per task. — **DELIVERED** (Pima task only).
3. Probabilistic-logical: SPL. Fallback if integration fails by end of week 4: DeepProbLog; documented as a deviation. — **NOT DELIVERED** (§8). The SPL stub raises `NotImplementedError` and the registered DeepProbLog fallback was not run either. Two of three registered integration styles are evaluated.
4. Pure-neural baseline (no symbolic layer): T1/T3 evaluated; T2 reported as "no evidence emitted." — **NOT DELIVERED** (§8). The class exists (`models.py` L248) but no released script invokes it and no result artifact exists.

*Delivery status added in v1.4. The list above is the registered set, not the evaluated set; a reader should
take the annotations, not the enumeration, as the record of what was run.*

### Deployment-policy arms for the rule-augmented model (v1.2 addendum, declared before results)

The same trained rule-augmented model is audited under two deployment policies of its
hard constraint — the paper's "halt or **escalate**" semantics:
- **Logged (primary):** violations recorded, the decision is deployed unchanged.
- **Enforced:** a hard-constraint-triggering confident no-AKI is **escalated to a deferral**
  (never deployed) → 0 violations by construction; the escalation rate on acute-severe
  cases is reported as the safety cost.

This is a deployment-policy comparison, **not** a change to any constraint definition or
pass threshold. It isolates what symbolic *enforcement* buys over mere *logging*, and —
together with the T2s probe — makes the reasoning-shortcut question concrete: a learned
core can reach high AUROC while under-relying on the constrained concept (low probe
sensitivity) and thus violate T2 when rules are only logged.

## 7b. Declared-but-unimplemented configuration keys (v1.4 audit)

Keys present in `config.yaml` and/or named in this document that **no released code reads**.
They are listed so that a reader does not mistake a declared key for an evaluated criterion.

| Key | Declared meaning | Status in v1.4 |
|---|---|---|
| `models.differentiable_ltn.sat_mean_min: 0.90` | LTN mean-satisfaction floor on the hard-constraint subset | **Not implemented; sub-criterion WITHDRAWN** (§3 T2). Never evaluated. |
| `models.differentiable_ltn.n_axioms: 2` | Number of LTN axioms per task | **Not read.** The axiom set is fixed in code; the key has no effect. Retained in `config.yaml` only as documentation of the arm's design, and marked here so it is not read as a configurable criterion. |
| `audit.abstain_tau_frozen: null` | Slot for the frozen τ written at freeze time | **Never written; not read.** τ is re-selected per seed (§3 T3, §8 row 7). The key is retained as a marker of the intended-but-unproduced lock artifact, not as a live setting. |
| `audit.min_subgroup_n: 500` | Subgroup criteria below n=500 reported UNDETERMINED (§2) | **Declared, not yet read by released code — implementation in progress.** Retained in force: the subgroup analysis under development applies this floor. Not withdrawn. |

## 8. Deviations Log

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-07-07 | §3 T2 | Applicability: absolute `creatinine ≥ 4.0` → acute-severe (`≥ 4.0` AND `≥ 1.5 × baseline`, KDIGO). `prereg-v1.0` → `v1.1`. | **Clinical validity, not gaming.** v1.0 absolute threshold conflated AKI with chronic kidney disease; an ICD-AKI–labelled model correctly predicted "no AKI" on stable CKD and was counted as violating (~67%, mean 4350/6464). Scope corrected to KDIGO acute-rise. No pass threshold changed. |
| 2026-07-07 | §3 T2 | (a) Applicability window: 7-day peak → **first-24h decision window**. (b) Violation: 0.5-threshold negative → **confident (≥ τ) negative** (defer-aware). `prereg-v1.1` → `v1.2`. | **Logical/decision-theoretic validity, not gaming.** v1.1 still failed (~67%, mean 2379/3534) because (a) the constraint used the 7-day creatinine peak — information the model had not seen at its first-24h decision point (temporal leakage), and (b) with 18% prevalence + calibrated probabilities, a 0.5 threshold made most decisions "no-AKI", so uncertain-but-deferrable cases were miscounted as confident violations. v1.2 makes the constraint a function of decision-time evidence only and scores the deployed defer-aware action. No pass threshold changed. |
| 2026-07-08 | §3 T2 | Add **compliance–coverage certificate** (coverage floor β = 0.50, deployed-violation-rate cap 0.05). `prereg-v1.2.1` → `v1.3`. | **Tightens the audit; closes a loophole we found in our own enforced arm.** The v1.2.1 enforced policy reached 0 violations by escalating 57% of acute-severe cases to deferral — passing T2 by mass abstention. The certificate requires a policy to both deploy ≥ β of the subgroup and keep deployed violations ≤ 0.05, so neither logged (violates) nor abstention-heavy enforced (under-covers) can pass cheaply. Declared before its results; makes T2 stricter, not easier. |
| 2026-08-11 | §1 Task A | **Task A (ICU in-hospital mortality) NOT DELIVERED.** No run, no result artifact, no reported metric. | **Not delivered — scope reduction.** Effort was redirected to the second *database* (eICU) rather than the second *task*, which we judged the stronger test of the finding's generality. This is a reduction in the registered scope and is reported as such: the framework is demonstrated on one clinical task (AKI) across two databases, not on two tasks. |
| 2026-08-11 | §7.4 pure-neural baseline | **NOT DELIVERED.** Class `PureNeural` exists in `models.py` (L248) but is not invoked by any released script, and no result file exists. | **Not delivered.** The registered contrast — a model emitting "no T2 evidence" by construction — is argued in the paper but not measured. Unexecuted code is counted as undelivered. *(A run is in progress in a parallel track; if it lands, this row is superseded by a "delivered in v1.4" row citing the result file.)* |
| 2026-08-11 | §7.3 SPL arm | **NOT DELIVERED**, and its registered fallback was also not run. `models.py` SPL raises `NotImplementedError`; `config.yaml:spl_probabilistic.enabled = false`. The declared DeepProbLog fallback ("if integration fails by end of week 4") was never executed. | **Not delivered.** Two of three registered integration styles are evaluated (pipeline-coupled, LTN). The registered fallback clause was triggered but not honoured; we report the omission rather than treat the fallback as satisfying the item. |
| 2026-08-11 | §3 T3 (τ) | **τ re-selected per seed, not frozen once.** `experiment.py` L63–65 selects τ inside the seed loop; `per_seed_results.json` shows 19 distinct values over 20 seeds (0.6441–0.6587). `abstain_tau_frozen` is `null`; no `preregistration_lock.md` exists. | **Deviation from the registered procedure; defensible on its merits but not re-registered.** Selection uses validation folds only and never touches the test fold, so no test-fold information enters τ, and the total spread (<0.015) is far too small to move any verdict — the T2 violation counts and T3 margins are stable across all 20 τ values. Per-seed selection is arguably the more faithful treatment of τ as a fitted quantity. Nonetheless: the registered artifact (one frozen value in a lock file) was never produced, and per-seed selection introduces a variance component the registered design would not have had. Reported, not retroactively legitimized. |
| 2026-08-11 | §6 NC-A′ (eICU) | **Control did not flag its target axis.** `ncA_kstar`-context: at the pre-registered k=3.0, `NC_A_uncalibrated.ece_ci_upper` = 0.0877 < 0.10 bound, `flagged_T3` = false. | **Counts AGAINST the framework, per §6.** The T3 axis is not validated by negative control on eICU. Reported as such in the paper. The k* severity sweep (k*≈3.1 MIMIC vs ≈8.0 eICU) diagnoses *why* — a fixed-severity synthetic control is not portable across datasets — but does not repair the failure, and the k* medians rest on unequal support (MIMIC n_seeds=20; eICU n_seeds=5, 15/20 entries null). |
| 2026-08-11 | §6 NC-C′ (both databases) | **Control did not flag its target axis in ANY released run** — MIMIC v1.2.1, MIMIC v1.3, GBM, KDIGO positive control, and eICU all report `flagged_T3_shift = false`. | **Counts AGAINST the framework, per §6.** The T3-shift sub-criterion is marked UNVALIDATED BY NEGATIVE CONTROL on both databases and is **excluded from the audit-readiness count**. A shift control with a demonstrated flagging regime is required before the sub-criterion can be counted; we do not have one. |
| 2026-08-11 | §2 / §3 | **"Stratified" splits never implemented.** `GroupShuffleSplit` groups by `subject_id` but performs no label stratification. | **Description error, immaterial in magnitude.** At n≈113k test decisions and 18.3% prevalence the sampling distribution of the positive fraction is tight enough that stratification is inconsequential. Wording corrected in §2 and in the paper; no result changes. |
| 2026-08-11 | §3 T3 (ECE) | **Effective bin count is 5, not 10.** `metrics.py` L9–24 bins `conf = max(p, 1−p) ∈ [0.5, 1]` against 10 edges spanning [0,1]. | **Disclosed; estimator unchanged.** The bias runs *toward* the T3 pass (coarser bins understate ECE), i.e. against our own interest. Disclosed in §3 and in the paper rather than silently re-binned, because the estimator is pre-registered. T3 margins (CI-upper 0.004 vs a 0.10 bound) are wide enough that no verdict is in question. |
| 2026-08-11 | §2 (imputation) | **Lab-median imputation fitted on the full cohort before splitting** (`cohort.py` L151–153 runs before `experiment.py` L52). | **Train/test boundary crossing; negligible magnitude, disclosed.** Eight population lab medians over 378,591 admissions are essentially identical to their per-fold counterparts. Disclosed in a footnote in the paper. A strictly clean pipeline would fit the imputer inside each split; future runs should. |
| 2026-08-11 | eICU cohort (§2 units) | **eICU grouping key is a hospital-stay id, not a patient id.** `build_eicu_cohort.py` L99–101 maps `subject_id ← patienthealthsystemstayid`, with `uniquepid` used only as a null fallback. The reported "147,832 patients" is a count of hospital stays. | **Implementation does not match the "patient-level split" description; disclosed, not repaired.** Unit stays within one hospitalization cannot straddle the split, but separate hospitalizations of the same patient can; residual leakage is bounded by the readmission rate, and features are first-24h labs of the index unit stay. Paper wording corrected to "hospital stays" throughout. **The correct fix is to re-run the eICU cohort build grouping on `uniquepid`;** this requires credentialed eICU access and was not possible within this revision. MIMIC-IV, which carries the primary claims, groups correctly by patient. |
| 2026-08-11 | §3 T2 (positive control reporting) | **Both registered T2 criteria now reported for the KDIGO-label positive control.** `audit_summary_kdigo_label.json`: certificate PASSES on both policies, but `T2_pass = false`, `T2_pass_stability = 2/20`, `T2_hard_violations_mean = 2.15`, `audit_readiness = 2/3`. | **Reporting correction, self-caught.** Paper v24 reported only the passing certificate criterion. The strict per-seed zero-violation criterion (§3 T2) fails on 18/20 seeds. Reporting only the passing sub-criterion would have been exactly the selective reporting this pre-registration exists to prevent. Both are now reported; the residual (0.12% of ~1,775 applicable cases) is interpreted as a finding — label and constraint are 95.4% but not 100% aligned. |

## v1.5 — post-hoc exploratory analyses added in response to review

Both entries below were added **after** the results they qualify were known, in
response to reviewer objections. Neither was pre-registered and neither changes a
registered threshold. They are logged here so the exploratory/confirmatory
boundary stays legible: nothing in v1.4 is restated as confirmatory on their
strength, and if either had come out the other way the paper would say so.

| Date | Analysis | What prompted it | Status and constraint |
|---|---|---|---|
| 2026-08-11 | **Trajectory-stratified and directly standardised label validity** (`scripts/08_stratified_label_validity.py` → `results/stratified_label_validity.json`) | Reviewers objected that the crude White–Black gap in ICD agreement (0.440 vs 0.277) could be case-mix rather than differential coding, since confident denials concentrate on admissions already elevated at arrival and the prevalence of that presentation differs by group. The objection was correct and unanswered. | **EXPLORATORY, post hoc.** Result: the gap persists within strata (already-elevated +0.155, cluster-bootstrap CI 0.120–0.191; rise-from-elevated +0.090, CI 0.010–0.169), direct standardisation to the cohort trajectory mix leaves White 0.433 and Black 0.292 with non-overlapping intervals, and a Kitagawa decomposition attributes 13% of the crude gap to composition and 87% to within-stratum agreement. **Constraint: adjusts only for creatinine trajectory** — the one confounder this cohort can measure. Comorbidity burden, CKD stage and site mix are not adjusted for and would require tables absent from the released derivative. The severe mask is taken from `severe_acute`, identical to `05b`, so `n=5,954` matches the reported cohort. |
| 2026-08-11 | **Chronicity proxy, observability audit and regression robustness** (blocks D/E/F of `scripts/08_stratified_label_validity.py`) | Reviewers noted that the `A1_elevated_at_arrival` stratum is clinically heterogeneous — arrival AKI, stable CKD and AKI-on-CKD in one layer — and that this paper owes the objection more than most, since the constraint's own first correction (v1.0 → v1.1) was made because an absolute-creatinine rule conflated AKI with **chronic** kidney disease. The disparity analysis then never looked at CKD. | **EXPLORATORY, post hoc.** Result: within the elevated-at-arrival stratum, splitting on a within-window chronicity proxy (24h peak/nadir < 1.15, i.e. flat-high creatinine) leaves the White−Black gap essentially unchanged on both sides (+0.1515, CI 0.1012–0.1970 and +0.1547, CI 0.1031–0.2039); **all 8 testable cells across 4 cut values (1.10/1.15/1.20/1.30) have CIs excluding zero.** A logistic model with trajectory band, the proxy and their race interaction gives OR 0.517 on the Black indicator (log-odds CI −0.8932 to −0.4469, n=4,655) with the interaction indistinguishable from zero (CI −0.226 to 0.283). **Constraints, recorded in the script and the paper: (a) this is NOT a CKD measurement.** A same-admission CKD code is assigned post-discharge from the signed record and is therefore a product of the coding process under audit, not a decision-time baseline covariate; it must not be used. A prior-admission code is the correct instrument but the released cohort derivative carries no diagnosis codes other than the AKI label, and raw `diagnoses_icd` is unavailable in this environment. **(b)** Block E establishes that prior-admission history is itself differentially observable (severe cases with ≥1 prior MIMIC admission: Black 80.9%, White 67.9%, spread 0.131), so "no CKD documented previously" does not mean "no CKD" equally across groups — the substitute analysis would need its own missingness treatment. The paper's claim was weakened accordingly, from "the gradient is not case-mix" to "measured case mix does not explain the gradient; residual confounding by pre-existing CKD, comorbidity burden or institutional documentation practice remains possible." |
| 2026-08-11 | **Matched positive control** (`scripts/09_matched_positive_control.py` → `results/matched_positive_control.json`) | The registered positive control retrained on a KDIGO label built from a separately constructed cohort (364,566 vs 378,591 admissions), so cohort composition rather than the label could account for the effect. | **EXPLORATORY, post hoc.** Design: intersect the two cohort builds on `hadm_id` (364,393 admissions), take **all features from the ICD build for both arms** so the design matrix is identical by construction, and run the identical pipeline twice on the same rows, splits and seeds with the training target as the only difference. Pre-run verification recorded in the output: the severe mask is identical between builds (5,954 = 5,954) and `wbc` was the only feature differing between builds (4.55% of rows), which the shared-feature design removes. Comparisons are paired Wilcoxon over the 20 matched splits. |

## v1.6 — the prior-CKD sensitivity analysis, released unrun

| Date | Item | Status | Detail |
|---|---|---|---|
| 2026-08-11 | **Prior-CKD sensitivity analysis** (`scripts/10_prior_ckd_sensitivity.py`) | **WRITTEN, TESTED, UNRUN** | A reviewer objected — correctly — that "we cannot measure CKD" is the wrong framing: the authors hold credentialed MIMIC-IV access, and `diagnoses_icd` is in the same hospital module the AKI label already comes from. The obstacle is environmental, not fundamental. The analysis is therefore released as executable code rather than described in prose: it builds `prior_ckd_documented` from admissions whose **dischtime precedes the current admittime**, excluding the audited admission entirely; restricts to an observed-history cohort and reports that restriction as a change of target population with per-group counts; and reports the White−Black gap within prior-CKD strata, doubly stratified by creatinine trajectory, directly standardised to the joint (trajectory × prior-CKD) mix, and under a logistic fit with the same covariates and the same patient-level cluster bootstrap as `08` block F. It **refuses to run** without the raw tables rather than emitting a partial result. Its headline sentence is built from the measured cells and has four branches, including the one that narrows our own subgroup claim; if the gap vanishes within prior-CKD strata, that is the finding and the paper must say so. **Two load-bearing rules are unit-tested on synthetic fixtures** (`tests/test_prior_ckd_logic.py`, 6 tests, each verified by mutation): a same-admission CKD code must never set the indicator, and a prior admission counts only if it discharged before the current one was admitted (overlapping and later admissions must not count). External support for rule 1: 37 of 92 MIMIC same-admission prediction studies used same-admission ICD codes as features despite documentation stating they are derived after discharge (Ramadan et al., JAMA Netw Open 2025;8(12):e2550454). |
| 2026-08-11 | **Chronicity-proxy claim language corrected** | Reporting correction | The paper had said the gap was "the same size" on both sides of the proxy and highlighted "all eight cells over four cut values excluding zero". Both overstate: a null interaction is absence of detected effect modification, not evidence of equality, and a four-value sweep presented as eight independent confirmations reads as threshold fishing. Now: one primary cut (1.15, with its mechanical justification — at nadir 4 mg/dL a 0.3 rise is only 1.075×, so **all 2,558 flat-high cases qualify through the absolute acute-rise limb alone**, verified from the cohort), three alternatives as an appendix sensitivity table, "similar with no evidence of interaction", and an explicit note that no equivalence test was performed. |
| 2026-08-11 | **Regression specification published** | Reproducibility fix | Outcome, sample (n=4,655 = White 2,978 + Black 1,677; the 1,299 excluded severe cases enumerated by bucket), covariate list, missingness (none — all fields complete), clustering and all five coefficients with intervals are now in the appendix. Previously only the race odds ratio appeared. |
| 2026-08-11 | **Deviations table split** | Production fix | The single `table*` overflowed the page and printed row 15 before row 14 with row 14 truncated mid-sentence — in the deviations log, the artifact whose whole value is visible completeness. Split into two cross-referenced tables, rows 1–8 and 9–15, with the ordering verified programmatically. `longtable` is not an option in a two-column class. |

## v1.7 — measurements restored from earlier drafts

Compression across revisions had dropped five measurements that were actually
made under the registered protocol. They are restored in a new appendix
("Measurements Not Reported in the Body") rather than left in result files only,
since a paper that argues for auditable evidence should not quietly shed its own.
None was dropped because it was unflattering; two of them are.

| Item | Source | Why it matters | Status |
|---|---|---|---|
| **Urine-limb under-count, measured** | `results/urine_output_arm.json` | The paper says the creatinine-only definition can only under-count severity. It now says by how much: on the 51,089 ICU admissions with charted output, a conservative rule (≥12 h sustained oliguria, fully charted) flags **6,253** AKI cases with no qualifying creatinine rise against 16,899 creatinine-positives — **+37%**. A permissive rule gives 16,399 (+97%), reported as an upper bound inflated by sparse charting. Makes the headline gap demonstrably conservative rather than assertedly so. | Restored, appendix + one body sentence |
| **30-point certificate threshold grid** | `results/threshold_sensitivity_v1.3.json` | β ∈ {0.3…0.7} × cap ∈ {0.02, 0.05, 0.10}, both policies: **0 of 30 pass**. Answers "is the failure an artifact of your thresholds" with a sweep rather than an assurance. | Restored, appendix + one body clause |
| **Admission-level χ² = 187.9 (3 d.f.)** | `results/subgroup_significance.json` | Reported *and* disowned: it treats admissions as independent and is anticonservative under repeated admissions, so the patient-level permutation test remains primary. Disclosing the more spectacular statistic we chose not to use is the point. | Restored, appendix + body pointer |
| **k\* triggering severity (portable negative control)** | `eicu_pipeline/results/ncA_kstar.json` | The fixed-k=3 uncalibrated control fires on MIMIC-IV (ECE CI upper 0.102) and **not** on eICU (0.088) — same code, opposite verdicts, because eICU's lower base rate (10.9% vs 18.3%) leaves less room for a multiplicative stretch. Reporting k\* instead (3.125 vs ≥8.0) makes the control a measurement that transfers. **Right-censored on eICU**: 15 of 20 seeds never reach the bound anywhere on the k grid (max 8.0), so the true separation exceeds 2.6×. Not pre-registered; added after the discrepancy appeared, and the fixed-k control remains the registered one. | Restored, appendix + one body sentence |
| **NC-C non-firing on both databases, on two different shift axes** | `audit_summary_mimic_final_v1.3.json`, `audit_summary_eicu_v1.3.json`, `eicu_pipeline/run_eicu.py` | MIMIC-IV uses a **temporal** axis (train 2008–2013, evaluate 2017–2022; late ECE CI upper 0.014). eICU uses an **institutional** axis — the 205 hospitals are randomly halved and the model is evaluated on the held-out half (0.006). The eICU run is the stronger test and still does not fire, so the null cannot be attributed to a weak axis. The T3-shift sub-criterion is **unvalidated by negative control** and excluded from every readiness count. By our own §6 rule this counts against the framework. **Implementation trap worth recording:** `run_eicu.py` stores its hospital split in the column named `anchor_year_group` to reuse the temporal harness unchanged, so that column name does not mean an era split on eICU — a first draft of this appendix mis-described the eICU control as an era split for exactly that reason, and the paper now states the distinction explicitly. | Restored, appendix; already booked in the deviations table |

Also restored, having been lost to prose compression rather than to a decision:
the absolute Kitagawa terms (0.022 / 0.141 alongside the 13% / 87% shares), the
statement that the logistic arm fails on the *same* limb as the MLP rather than a
different one, and the specification of what the prior-CKD script computes.

**Page-budget note.** These restorations put the body at an estimated 8.62 pages
against an 8-page limit — over, and knowingly so, on instruction to preserve the
information first and relocate later. Everything restored to the body is a single
sentence or clause pointing at the appendix; the appendix itself does not count
against the limit. The in-source `% PAGE-BUDGET NOTE` lists what to move, in
order, once the true compiled page count is known.

## v1.8 — the prior-CKD analysis was RUN, and it narrowed our own claim

The credentialed tables became available and `scripts/10_prior_ckd_sensitivity.py`
was run (`results/prior_ckd_sensitivity.json`). It had been released unrun in v1.6
with four possible readings pre-written; the run selected the middle one.

**The two findings do not have the same status, and an earlier draft of this entry
blurred that by saying the script "was run unchanged". It was not.** A line-level
diff of the released-unrun script against the version that produced these results
shows **97 lines added, none removed or altered**:

- **Finding 1 comes from the script as released.** Blocks A–E were untouched; the
  four pre-written readings were fixed before any output existed, and the run
  selected one of them.
- **Finding 2 does not.** Block E's output showed unanticipated renal-replacement
  prevalence; block G was then **added** to measure what that implied, and after
  seeing its first output its interpretation was **rewritten**, because the initial
  reading — that the rule fires on noise in dialysis-dependent patients — was
  contradicted by the rise profile the same block reports. That is analysis chosen
  after seeing results, one step further from pre-specification than Finding 1.

This distinction is now stated in the appendix opener, in deviations row 15, and
beside every claim in the abstract, §6.3, the limitations and the conclusion that
rests on Finding 2. A framework whose whole argument is that pre-specification is
what makes an audit falsifiable cannot describe its own post-hoc extension as a
pre-specified run.

**Provenance check before trusting the join.** The raw tables reproduce the
released cohort exactly: 378,591 / 378,591 cohort admissions present in
`admissions`, `admittime` identical on 100%, and the AKI label recomputed from
`diagnoses_icd` agrees with the cohort's own `aki` column on **100.000%** of
admissions at the same 0.1834 base rate. Same MIMIC-IV version, no silent
mismatch.

| Finding | Result | Effect on the paper |
|---|---|---|
| **1. Racial gradient attenuates** | Within prior-CKD stratum W−B = **+0.092** (CI 0.054–0.128, n=3,736) against a crude +0.163; no-prior-CKD stratum +0.044 (−0.029–0.118, n=467), not excluding zero. Joint standardisation moves White 0.372→0.341, Black 0.232→0.263. Logistic fit with trajectory + chronicity + prior CKD: race OR **0.591** (log-odds CI −1.197–0.301), **not** excluding zero; interaction null (1.085). | **Claim weakened.** Abstract, §6.3 and Conclusion now say the gradient is *attenuated by, not independent of*, documented pre-existing CKD. The earlier "measured case mix does not explain that disparity" wording was replaced everywhere. |
| **2. Unanticipated: the headline gap is concentrated in chronic renal disease** | Agreement **0.897** (0.870–0.924, n=457) where no prior renal diagnosis exists; **0.079** (0.068–0.091, n=2,447) where a prior dialysis/transplant code does; intervals non-overlapping. A further 1,751 severe cases have no prior admission at all (agreement 0.553, prior status unknowable). | **Constraint-validity limitation added.** Limitation (iv) was rewritten from "the gradient is not adjusted for CKD" to "the population our headline describes is narrower than guideline-severe cases". Abstract and Conclusion now state where agreement is high and where it is low. |
| **Discriminating test for finding 2** | Is the rule firing on noise in dialysis-dependent patients? **No.** Median absolute rise is **1.0 mg/dL in both** strata and the share meeting the relative limb is 23.2% vs 23.6%; only the baseline differs (median nadir 5.8 vs 4.3 mg/dL). | The interpretation was rewritten mid-analysis: the first draft asserted "a 0.3 rise on a 4.0 baseline is not an acute injury", which the rise profile contradicts. It is a coding-convention disagreement, not a measurement artifact. Block `G._rise_comparison` records this. |

**Design rules held.** The CKD indicator uses *only* admissions discharged before
the audited admission started; the audited admission's own codes are never read
(they are assigned post-discharge by the process under audit — 37 of 92 MIMIC
same-admission prediction studies use such codes as features, Ramadan et al.
2025). The indicator is named `prior_ckd_documented`, not `has_ckd`.

**Selection limit, stated not buried.** Only 4,203 of 5,954 severe cases (70.6%)
have observable history, and unequally: 82.9% of severe Black against 69.7% of
severe White cases. The analysis therefore describes patients with prior MIMIC
contact, and is a sensitivity analysis rather than a correction.

**One figure changed between scripts.** v1.5 quoted 80.9% / 67.9% for
prior-admission observability, from a within-cohort ordering proxy in
`08_stratified_label_validity.py`. Script 10 uses the full `admissions` table with
a `dischtime < admittime` requirement and finds 82.9% / 69.7%. The paper now
quotes the numbers the analysis actually ran on and explains the difference in
Appendix `apd:ckd` rather than leaving two versions of the same quantity in the
record.

**Nothing registered was changed by any of this.** No threshold moved; the
analysis is marked exploratory and post hoc in the deviations table (row 15,
status changed from `released unrun` to `exploratory, post hoc; run`).

## v1.9 — reviewer-driven revision: one claim bounded, one gap made runnable

A reader proposed promoting the matched positive control to the paper's primary
causal evidence ("same cohort, same features, same architecture, only the target
changes — failure disappears"). We measured it before acting, and **declined**.

| Item | What we measured | What we did |
|---|---|---|
| **Positive-control ceiling** (`effect_ceiling` block in `09_matched_positive_control.py`) | The KDIGO target and the T2 constraint are both creatinine-derived — `cohort.py` has flagged that label `CIRCULAR` since the first revision. On constraint-applicable cases the KDIGO target is already positive for **95.4%** against **37.0%** for ICD. A T2 violation is a confident *no-AKI* call on such a case, so target base rates alone imply a violation-rate drop of **0.584** against the **0.718** observed: **81% of the effect is target arithmetic**, 19% learned. | Reported in the body as a ceiling on the control. It stays a **positive control** — the constraint is triggerable, the harness responds, the failure is a property of the target rather than the architecture — and is explicitly *not* promoted to a primary causal claim. Deviations row 16. |
| **Clinician adjudication** (`11_clinician_adjudication.py`) | Not measurable without clinicians. What *is* measurable: the sample can be drawn, blinded and scored now. 200 cases, stratified, over-weighting the cell where the rule says severe and the coders did not (80 of 200) because that is where the headline lives. Packet verified to carry no verdict, label or identifier column. | Released with the packet drawn and **falsification thresholds fixed in source before any adjudication exists**: agreement ≥ 0.70 on the headline cell, κ ≥ 0.60 overall. Below either, the script prints the verdict that narrows our claim. Both branches tested on synthetic responses; four guards mutation-verified. Deviations row 17. |

Also in this revision, all wording rather than measurement: the contribution
statement now separates what is known (administrative labels are imperfect) from
what is new (that imperfection survives high accuracy and reappears at decision
time as systematic guideline violation); "non-compliant" is scoped throughout to
*non-compliant with the evaluated clinical constraint*, with an explicit
disclaimer of regulatory, legal or clinical certification; the eICU arm is
positioned as independent multicentre replication while the stay-level grouping
limitation is restated in the same paragraph rather than left for the reader to
find; and Related Work states the distinction from label-noise detection as a
difference of question (*which rows are mislabelled* over a dataset, against
*does this decision contradict the guideline*) rather than of technique.

**Declined:** moving the subgroup analysis to secondary positioning. It already
sits after the primary finding and is framed as mechanism, not headline; it is
also the least common lever in the accepted-paper corpus we profiled, and the
prior-CKD attenuation we found and reported ourselves is a credibility asset
rather than a reason to bury it.

**No registered threshold was changed by any of this.** The ceiling is a
disclosure about an existing result; the adjudication thresholds are new, and are
declared before the data that would test them exists.

## v1.10 — bounding Finding 2 by deletion rather than by argument

A reader observed that Finding 2 (block G: agreement 0.897 without prior renal
disease against 0.079 with prior dialysis or transplant) threatens the body's
headline, because a reviewer can read it as "the severity rule is catching chronic
elevation, so the 71% violation rate is an ESRD artifact." The paper already
conceded the narrowing honestly but had built no defence line.

The defence is a measurement, not a rebuttal. Block **H** was added to
`scripts/10_prior_ckd_sensitivity.py` and computes what remains if the whole
prior-renal-replacement stratum is **deleted**. Because block G's three strata
partition all 5,954 severe cases exactly, this is a deterministic reweighting of
an existing measurement — no new model, no new threshold, nothing fitted.

| Quantity | Value |
|---|---|
| Prior-replacement stratum, share of ALL severe cases | 41.1% (n=2,447) |
| Same stratum, share of observed-history subset | 58.2% — the figure that invites the misreading |
| Severe cases remaining after deletion | 3,507 (59% of total) |
| Agreement on those remaining cases | 0.5726 |
| **Disagreement on those remaining cases** | **43%** |

The weighted agreement across all three strata reproduces the paper's cohort
figure of 0.3697 exactly, which is the arithmetic check that the partition is the
one the headline is computed on.

Written into the abstract, limitation (iv), the conclusion and Appendix G. The
claim made is deliberately narrow: the concentration finding **narrows which
population the headline describes best**, and does not make the headline an
artifact of that population. Nothing about Finding 2's post-hoc, exploratory
status changes — block H inherits it, and says so.

**No registered threshold was changed.** Block H is a disclosure computed from
block G's own cells.

## v1.11 — the deletion test applied to the model statistic, not only the label

Block H (v1.10) answered "is the headline an ESRD artifact?" for the **label**:
delete the dialysis stratum, 3,507 severe cases remain, label–guideline
disagreement 43%. A reader pointed out that this leaves the argument half-built,
because the paper's headline is a **model** statistic — a mean 71% of deployed
severe decisions are confident denials — and block H never touches it. Filling
that gap with prose would be the move this paper exists to refuse, so
`scripts/12_violation_rate_by_renal_history.py` computes it, reusing
`05_subgroup_audit.py`'s splits, model factory and statistic definitions so the
arm cannot drift from the numbers the paper reports.

| Stratum | n severe (mean/split) | Coverage | Violation rate |
|---|---|---|---|
| Prior renal replacement | 716.4 | 0.890 | **0.9405** |
| No prior renal replacement | 527.6 | 0.731 | 0.4790 (UNDETERMINED) |
| Prior status unobservable | 528.8 | 0.743 | 0.5791 (UNDETERMINED) |
| **Pooled, surviving strata** | 1,056.4 | — | **0.5295** |
| All severe cases (paper headline) | 1,772.9 | 0.799 | 0.7143 |

The model statistic behaves like the label statistic: concentrated in the
dialysis stratum, substantially present without it.

**Two accounting points, both recorded against us.**

1. **Pooling weight.** `violation_rate_deployed` has `n_deployed` as its
   denominator, not `n_severe`, and coverage differs across strata (0.89 vs 0.73
   vs 0.74), so per-stratum rates are pooled by `coverage × n_severe`. A first
   version of this block pooled by `n_severe`; the payload records what that
   would have given (0.5291) so the size of the error is on the record rather
   than silently corrected. The choice moves the result by 0.0004 here — small,
   but not small by construction.
2. **Both surviving strata are below our own floor.** At a mean ~528 severe cases
   per split, the registered `min_subgroup_n = 500` is breached on at least one
   split in each, so both are flagged **UNDETERMINED** by the same rule that
   governs the racial-subgroup analysis. The figures are reported because
   deleting them would be selective; the flag stays because the rule applies to
   us too. This is the honest reading: the model-side deletion test is
   **suggestive, not powered**.

A guard was also added against the failure mode that produced this block's first
wrong answer: `agg_across_seeds` nests each statistic as `{mean,p5,p95}`, and
reading a flat `<key>_mean` yields `None` silently. The script now raises on an
unpopulated headline comparison rather than writing a file full of nulls.

**No registered threshold was changed.** The floor was enforced, not relaxed.

## v1.12 — post-submission, during review (2026-10-10): three reviewer-requested measurements

All three were requested in the ML4H reviews of the submitted manuscript. None was
pre-registered; none changes a registered threshold; each is reported in whichever
direction it fell. Scripts, results and this entry were added to the repository after
submission and are dated as such.

| Item | What was done | Result | Status |
|---|---|---|---|
| Restricted-cohort certificate (model side) | `scripts/12_violation_rate_by_renal_history.py` gained a `restricted_certificate` block: per-seed pooling from counts over the two non-dialysis strata, then the primary certificate's own quantile rule. (Supersedes v1.11's "suggestive, not powered", which applied to the *per-stratum* figures at ~528 severe/split; the *pooled* surviving cohort at 1,006–1,099 clears the floor on every split.) | Pooled surviving cohort 1,006–1,099 severe/split (above the 500 floor on every split → determinate). Coverage p5 0.711 (floor 0.50, passes); violation p95 0.563 (cap 0.05) → **FAIL on violation**, same limb as the full cohort. Mean violation 0.5295 vs 0.71 full. | post hoc, reviewer-requested |
| eICU held-out-hospital certificate | `eicu_pipeline/heldout_hospital_certificate.py`: run_eicu.py's hospital partition (102 train / 103 held-out), harness model, tau and `run_T2`, certificate arithmetic unchanged; 20 seeds vary the model, not the partition. | AUROC transfers (0.849). Coverage 0.307 (p5 0.196), violation 0.973 (p95 0.995) → **FAIL on coverage**, as the submitted within-eICU certificate did. 80 held-out sites with severe cases, **0** clear the floor → per-site UNDETERMINED; among 33 sites with ≥20 severe/split, violation 0.85–1.00. | post hoc, reviewer-requested |
| Clinician adjudication (protocol registered in v1.9) | One nephrologist, not an author, blinded (packet without key), scored 200-case packet under `ADJUDICATION_INSTRUCTIONS.md` unchanged. Scored with `11_clinician_adjudication.py score`. | 162/200 adjudicated, 38 blank (= "cannot tell", dropped). Cell B 63/63 (17 of 80 blank), A 27/27, C 40/52, D 19/20; κ vs rule 0.835, vs code 0.023. **Both pre-declared thresholds pass.** | protocol pre-registered; result observed |

**Two bounds on the adjudication, recorded here before the paper states them.**
(1) On every one of the 162 decided cases the adjudicator's verdict coincides exactly with the
KDIGO stage ≥ 1 creatinine criteria (≥ 0.3 mg/dL absolute or ≥ 1.5× relative rise), with no
exception. The result therefore validates our operationalisation against a nephrologist's
reading of the guideline; it is not independent evidence that the displayed cases are
clinically AKI. (2) The 38 withheld cases are not random: 30 are rule-severe (13 in cell A, 17 in cell B) and
8 are near-threshold; median baseline creatinine 5.1 mg/dL, median rise 0.4 — the high-baseline, borderline-rise stratum
block G already identifies as the population the rule describes least well. The script's
auto-generated verdict sentence ("clinician-validated operationalisation") is **not** used
in the paper; the bounded wording above is.

**Provenance of the scored file.** Two earlier files returned by the adjudicator had been
passed through a language model for formatting and were discarded unscored; the scored file
is the adjudicator's direct entry of 1/0/blank into the unmodified packet, confirmed by him.
The 14 input columns of the scored file are cell-for-cell identical to the packet sent.

**Patient-level handling.** The returned response file (`results/adjudication_responses/`)
is patient-level via the key and is git-ignored; `results/clinician_adjudication.json`
carries only cell-level aggregates and is released.

## v1.13 — W3PHIAI-27 revision (2026-10-31): cross-tab and clustered CIs, both post hoc, no threshold changed

Written for the resubmission to W3PHIAI-27 (AAAI-27). Both items were promised in the ML4H
rebuttal; neither was pre-registered; neither changes a registered threshold or any
existing result. Every figure below is read from the named results key
(`claim_verification_w3phiai.csv` lists each one with its key path).

| Item | What was done | Result | Status |
|---|---|---|---|
| Adjudication × renal history × trajectory | `scripts/13_adjudication_by_stratum.py` joins the 200 adjudicated cases (via the git-ignored key) to the block-H renal-history partition (`build_prior_indicators` from script 10, `renal_history_labels` from script 12) and to the error-taxonomy trajectory classes A1/A2/A3 (script 05's nadir rule). Blanks are "cannot tell" and are excluded from the agreement denominator, as in script 11. Output `results/adjudication_by_stratum.json` is aggregate-only; cells with < 10 decided cases are listed under `small_cell_warning` (6 cells). | Renal history — prior RRT: 51/53 agree (10 blank); no prior RRT: 52/56 (16 blank); unobservable: 46/53 (12 blank). Trajectory — A1: 65/65 (27 blank, blank share 0.29); A2: 22/25 (0 blank); A3: 62/72 (11 blank). The blanks concentrate in A1 (already elevated at arrival), consistent with the v1.12 reading; the A1 cases that were decided agree with the rule without exception. | post hoc, exploratory |
| Subject-clustered CIs, restricted cohort | `scripts/12_violation_rate_by_renal_history.py` gained `restricted_cohort_cluster_ci`: subject_id-clustered nonparametric bootstrap (B = 1000 from `experiment.bootstrap_B`, seed 0), per seed over that seed's test subjects, with the median across the 20 seeds of (lo, hi) reported; plus a descriptive pooled interval over the stacked test decisions (21129 decisions, 3163 subjects, 3148 of whom appear in more than one seed's test set, so the pooled interval is not inferential). The `restricted_certificate` and `headline_comparison` blocks are byte-identical to the copy frozen before the re-run (`tests/fixtures/violation_rate_by_renal_history.v1_12_frozen.json`; enforced by `tests/test_restricted_cluster_ci.py`). The payload guard now refuses to write if the CI block is missing or any bound is None. | Per-seed median point [CI]: violation 0.528 [0.492, 0.566]; coverage 0.737 [0.710, 0.765]; ICD agreement 0.574 [0.542, 0.607]. Pooled (descriptive): violation 0.529 [0.509, 0.548]; coverage 0.737 [0.723, 0.750]; ICD agreement 0.577 [0.561, 0.595]. The patient-level interval for the violation rate sits far above the 0.05 cap on every seed, so the restricted-cohort verdict (FAIL on violation) does not depend on the choice between seed-quantile and patient-bootstrap uncertainty. | post hoc, reviewer-promised |

**Tests added.** `tests/test_adjudication_by_stratum.py` (8) and `tests/test_restricted_cluster_ci.py` (7),
every guard mutation-verified (the mutations and their outcomes are recorded in
`new_analyses_notes.md`). `tests/fixtures/` holds the frozen v1.12 aggregate copy; it carries no
patient identifiers.

**Patient-level handling.** Script 13 reads the blinding key and the response file in-process
only; the written JSON is walked for `hadm_id`/`subject_id`/`case_id` at every depth and the
script refuses to write if one is found. Row-level decisions used by the bootstrap in script 12
never leave the process; the payload receives quantiles.

**W3PHIAI-27 manuscript (2026-10-31): which rebuttal promises are discharged where.** The ten
"we will …" commitments in the authors' posted responses to the earlier review (not included in this repository) and the
section of the W3PHIAI-27 `main.tex` / `supplementary.tex` that discharges each. Section numbers are
those of the W3PHIAI-27 body (1 Introduction … 7 Conclusion; 5.1–5.6 Results); S-numbers are the
supplementary document.

| # | Promise (rebuttal file) | Discharged in |
|---|---|---|
| 1 | Restricted-cohort model-level certificate: violation rate, coverage, verdict, intervals, determinacy (00_global §1; 01_bpXE Q2) | Sec. 5.4 "Restricted cohort" + Table 2 (robustness); S14 Tables S17–S18; S7 pointer |
| 2 | eICU held-out-hospital certificate with the limb difference stated (00_global §2; 01_bpXE Q4) | Sec. 5.4 "Held-out hospitals" + Table 2; S14 Table S19; S8 pointer |
| 3 | Clinician adjudication with the per-cell table and the two bounds (00_global §3; 01_bpXE Q1; 02_Fxcc; 03_NUPj) | Sec. 5.6 (new subsection) + abstract; S14 Tables S20–S22; deviations row 17 updated |
| 4 | Narrow "no deployment policy" to "neither of the two tested policies … 30 (β, cap) grid points; other thresholds, abstention policies, overrides untested" (00_global; 01_bpXE Q3) | Abstract, Sec. 1 (three consequences), Sec. 5.4 (title and closing sentence), Sec. 6 limitation 5, Sec. 7 |
| 5 | Reframe the coverage floor as a registered anti-gaming constraint with no clinical derivation (00_global; 01_bpXE Q3) | Sec. 3 (certificate paragraph), Sec. 6 limitation 5 |
| 6 | Consider "guideline-discordant" in place of "non-compliant" (02_Fxcc) | "non-compliant" survives only in the title and the defining paragraph (Sec. 3); interpretive uses now read "discordant"/"guideline-discordant" (Sec. 5.1, 6, 7) |
| 7 | Concede the hypothetical deployment setting more fully in the Introduction (02_Fxcc) | Sec. 1 "What is audited, and the limit of the conditional"; Sec. 6 limitation 3 |
| 8 | Expand KDIGO at first use in the Introduction (04_zpgV) | Sec. 1 "Study design, stated once" |
| 9 | Define arm vs split; rewrite the sentence that read as if arms were folds (04_zpgV) | Sec. 1 design paragraph (definitions); Sec. 5.1 second paragraph (rewritten) |
| 10 | Two-sentence ICD-code vs KDIGO-rule contrast with the 37% agreement figure (04_zpgV) | Sec. 1 design paragraph |

Not promised but changed for consistency with the evidence: the body no longer asserts that the
nadir-as-baseline and urine-output simplifications are conservative (03_NUPj reply said they were);
Sec. 6 now states they are not assumed conservative, quotes the urine arm (6,253 additional
AKI-positive admissions among 51,089; 37% undercount vs 16,899), and says the effect on the
severe-subgroup figures is unevaluated. S9 item 1 was softened to match; the ML4H wording is kept
verbatim in S13.10. The rebuttal's blank-case medians (baseline 5.1 mg/dL, rise 0.4) are not in any
results file and were NOT written into the paper; the traceable trajectory cross-tab of blanks
(27 A1 / 11 A3 / 0 A2, `adjudication_by_stratum.json`) is reported instead.

### Retained findings (reported in the paper)

Neither earlier run is discarded — together they are the framework's strongest evidence.
AuditNeSy's T2 axis flagged **two** successive constraint mis-specifications before any
reviewer did — an absolute-threshold error (CKD confound) and a temporal-leakage error —
exactly the "constraint specification risk" the framework exists to surface, and the live
operationalization of reviewer request RR6 (clinician-grade constraint validation). The
paper reports the version sequence transparently (Appendix) and presents v1.2 as the
principled final specification.

- **v1.0** (absolute ≥4.0): AUROC 0.880, T1 20/20, T2 mean 4350 viol (FAIL), T3 CI-upper 0.004 (PASS) → 2/3.
- **v1.1** (7-day acute-severe): AUROC 0.880, T1 20/20, T2 mean 2379 viol (FAIL), T3 CI-upper 0.004 (PASS), probe creatinine-sensitivity 0.083 → 2/3.
- **v1.2** (24h decision-time, defer-aware): to be observed after this freeze.

*Shift note:* across v1.0/v1.1, era-shift calibration degradation stayed ≤0.01 (< 0.05
bound) but the abstention-direction sub-criterion did not trigger (model no less confident
on later eras) and NC-C did not flag — reported honestly as: this task exhibits little
exploitable era shift for lab-based AKI. Shift is a separate diagnostic and does not gate
the three-axis audit-readiness count.
