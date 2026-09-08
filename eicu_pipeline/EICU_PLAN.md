# eICU side-experiment (isolated on branch `eicu-experiment`)

Goal: replicate the MIMIC-IV AKI accuracy-vs-label-validity finding on the **eICU
Collaborative Research Database** (Philips; ~200k unit stays, 208 hospitals,
multi-center) to (a) answer the single-institution objection and (b) obtain a real
cross-hospital distribution shift for T3. **Integrate into the paper ONLY if it
works cleanly.** The `main` branch (`mimic_pipeline/`) is untouched.

## Principle: reuse the harness, rewrite only the cohort
The audit machinery is dataset-agnostic. We do **not** re-implement T1/T2/T3, the
compliance–coverage certificate, negative controls, or the experiment loop. We
only write an **eICU cohort adapter** that produces the same dataframe contract
the harness expects, then point the existing harness at it.

### Dataframe contract the harness needs (per MIMIC cohort)
Columns required by `auditnesy_mimic` (see `mimic_pipeline/auditnesy_mimic/cohort.py`):
- `subject_id`  — patient id for GroupShuffleSplit (eICU: `patienthealthsystemstayid` or `uniquepid`)
- `hadm_id`     — encounter id (eICU: `patientunitstayid`)
- feature cols  — first-24h labs + demographics (same names: creatinine, glucose, bun,
  potassium, sodium, hemoglobin, wbc, bicarbonate, age, is_female)
- `creat_peak_24h`, `creat_nadir_24h` — for the acute-severe mask
- `severe_acute` (0/1) — decision-time acute-severe AKI (peak≥4.0 AND acute rise), 24h window
- `aki` (0/1)   — label, INDEPENDENT of the constraint feature
- `anchor_year_group` — for era shift (eICU: no era; use **hospitalid** for cross-hospital shift instead)

## eICU schema notes (differs from MIMIC-IV; the real work)
- IDs: `patientunitstayid` (stay), `patienthealthsystemstayid`/`uniquepid` (patient).
- Time: **offset in minutes** from unit admission (`labresultoffset`, etc.), not timestamps.
  First-24h window = offset ∈ [0, 1440].
- Labs: `lab` table, `labname` (e.g. 'creatinine', 'glucose', 'BUN', 'potassium',
  'sodium', 'Hgb', 'WBC x 1000', 'bicarbonate'), value in `labresult`. VERIFY names/units.
- Demographics: `patient` table (`age` — note '> 89' strings; `gender`).
- **Label risk:** eICU has no clean ICD-9/10 like MIMIC `diagnoses_icd`. AKI label options:
  1. `diagnosis` table string match (e.g. 'acute renal failure'/'AKI') — messier proxy;
  2. APACHE / `apacheApsVar` creatinine-based — but that reintroduces creatinine circularity;
  3. KDIGO-from-labs as label — CIRCULAR with the T2 constraint (avoid, as in MIMIC).
  **Preferred:** diagnosis-string AKI label (analogue of ICD) so the accuracy-vs-label-validity
  story stays intact. Confirm the string label under-captures KDIGO (the whole point).
- **Shift axis:** replace `anchor_year_group` with `hospitalid` groups (train on a subset of
  hospitals, evaluate on held-out hospitals) — a genuine multi-center shift.

## Steps
0. PhysioNet: confirm eICU-CRD credentialed access + sign the eICU DUA (separate from MIMIC).
1. `00_check_eicu_labnames.py` — list `labname` values + units; confirm the 8 analytes.
2. `build_eicu_cohort.py` — assemble the contract dataframe above (this is the main effort).
3. Reuse the harness: point `mimic_pipeline` config at the eICU cohort parquet, run
   `02_run_experiment.py` (or a thin `run_eicu.py` that imports `auditnesy_mimic.experiment`).
4. Add a **hospital-shift** run (train hospitals vs held-out hospitals) for T3.
5. Compare to MIMIC: does the gap replicate? (deployed-violation rate, coverage certificate,
   probe sensitivity, AKI-rate-in-severe-acute.)

## Decision gate (before integrating)
Integrate into the paper only if BOTH hold:
- cohort is clinically sane (AKI rate plausible; ≥500 acute-severe cases), AND
- the finding is interpretable (either the gap replicates → strong multi-center claim, or a
  clean, explainable difference → still reportable). If eICU labeling muddies the story, keep
  it as an appendix robustness note or drop it — do NOT weaken the clean MIMIC narrative.

## Guardrails (same as main)
- No patient-level data committed (eICU parquet/CSV gitignored).
- No fabricated numbers. Pre-registration: reuse frozen thresholds; if a new eICU-specific
  choice is needed, declare it before running eICU test folds.
