# AuditNeSy — Pre-Registration of Pass Criteria (v1.3, RATIFIED)

**Status:** RATIFIED 2026-07-08 — frozen at tag `prereg-v1.3`. Supersedes `prereg-v1.0`…`v1.2.1`.
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
| A — ICU in-hospital mortality | death before discharge, first ICU stay, features from first 24h | AUROC |
| B — AKI within 48h | KDIGO stage ≥ 1 (creatinine criteria; urine-output criterion EXCLUDED — declared here) | AUROC |
| P — Pima (contrast) | as in v7 | AUROC |

Discrimination metrics are contextual only; no axis gates on AUROC.

## 2. Experimental Units

- 20 seeds × repeated stratified 70/30 patient-level splits per task/model.
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
- **Differentiable (LTN) evidence form:** per-decision satisfaction converted categorically at **s ≥ 0.5** (violation if s < 0.5 on a hard-constraint-relevant case); ADDITIONALLY mean satisfaction on the hard-constraint-relevant subset must be **≥ 0.90**. Sensitivity sweep over s ∈ {0.3, 0.5, 0.7, 0.9} reported regardless of outcome.
- **Probabilistic (SPL) evidence form:** hard-constraint probability must satisfy **P(constraint) ≥ 0.99** per decision; violations counted below that line. Sensitivity sweep over {0.95, 0.99, 0.999}.

### T2s — Semantics Probe (ADVISORY — never gates)
- Monotonicity consistency: for each hard constraint, counterfactual sweep of the constraint feature over its clinical range on n = 200 sampled decisions; report the fraction of decisions where model output direction matches declared constraint semantics. No pass threshold in v8 (screening instrument; thresholding deferred until the probe is validated).

### T3 — Safe Fallback
- **ECE:** 10-bin, equal-width. **Criterion: bootstrap 95% CI upper bound of ECE < 0.10** per task (in-distribution).
  - Rationale: 0.10 is looser than v7's observed 0.070 point estimate because the criterion is now on the CI upper bound (strictly harder form); clinically motivated by the ~1-decile confidence-reliability gap tolerable in advisory (non-autonomous) risk flags.
- **Abstention policy:** defer when max(p, 1−p) < τ. **τ is selected ONCE on the validation folds to target a deferral rate of ≤ 15% in-distribution, then frozen** before test evaluation. The selection procedure (not the value) is what is registered; the frozen value is recorded in `preregistration_lock.md` at freeze time.
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

## 7. Models Under Test

1. Pipeline-coupled: GBM + explicit rule layer (primary), MLP variant (for NC-D′).
2. Differentiable: LTN, ≥ 2 axioms per task.
3. Probabilistic-logical: SPL. Fallback if integration fails by end of week 4: DeepProbLog; documented as a deviation.
4. Pure-neural baseline (no symbolic layer): T1/T3 evaluated; T2 reported as "no evidence emitted."

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

## 8. Deviations Log

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-07-07 | §3 T2 | Applicability: absolute `creatinine ≥ 4.0` → acute-severe (`≥ 4.0` AND `≥ 1.5 × baseline`, KDIGO). `prereg-v1.0` → `v1.1`. | **Clinical validity, not gaming.** v1.0 absolute threshold conflated AKI with chronic kidney disease; an ICD-AKI–labelled model correctly predicted "no AKI" on stable CKD and was counted as violating (~67%, mean 4350/6464). Scope corrected to KDIGO acute-rise. No pass threshold changed. |
| 2026-07-07 | §3 T2 | (a) Applicability window: 7-day peak → **first-24h decision window**. (b) Violation: 0.5-threshold negative → **confident (≥ τ) negative** (defer-aware). `prereg-v1.1` → `v1.2`. | **Logical/decision-theoretic validity, not gaming.** v1.1 still failed (~67%, mean 2379/3534) because (a) the constraint used the 7-day creatinine peak — information the model had not seen at its first-24h decision point (temporal leakage), and (b) with 18% prevalence + calibrated probabilities, a 0.5 threshold made most decisions "no-AKI", so uncertain-but-deferrable cases were miscounted as confident violations. v1.2 makes the constraint a function of decision-time evidence only and scores the deployed defer-aware action. No pass threshold changed. |
| 2026-07-08 | §3 T2 | Add **compliance–coverage certificate** (coverage floor β = 0.50, deployed-violation-rate cap 0.05). `prereg-v1.2.1` → `v1.3`. | **Tightens the audit; closes a loophole we found in our own enforced arm.** The v1.2.1 enforced policy reached 0 violations by escalating 57% of acute-severe cases to deferral — passing T2 by mass abstention. The certificate requires a policy to both deploy ≥ β of the subgroup and keep deployed violations ≤ 0.05, so neither logged (violates) nor abstention-heavy enforced (under-covers) can pass cheaply. Declared before its results; makes T2 stricter, not easier. |

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
