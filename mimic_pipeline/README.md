# AuditNeSy — MIMIC-IV pipeline

Large-scale, real-world validation of the AuditNeSy falsifiable accountability
suite on **acute kidney injury (AKI) risk prediction** from MIMIC-IV (hosp module).

This package is the MIMIC-IV arm of the paper. It implements the audit
against the **ratified pre-registration** (`../preregistration.md`; the
`prereg-v1.0 … v1.3` tags live in the authors' identifying repository and are
not part of this anonymised package).

> **Data access & DUA.** MIMIC-IV requires PhysioNet credentialing + a signed
> DUA. **No patient-level data or derived artifacts (cohort caches, decision
> traces) may be committed or shared.** The output directory is gitignored and
> the example decision trace in the paper is *synthetic*.

## What changed vs. the earlier `auditnesy_mimic.py`

| Review issue | Fix (where) |
|---|---|
| **#1 label–constraint circularity** | Label is administrative AKI (ICD-9 `584*` / ICD-10 `N17*`), **independent** of the creatinine the T2 constraint uses (`cohort.py`). The hard constraint (prereg v1.2) fires only on **decision-time acute severe AKI** (peak ≥ 4.0 AND acute rise, both within the first-24h window), and a violation is a **confident** no-AKI decision (`pred=0` AND confidence ≥ τ); uncertain cases defer (safe), consistent with T3. This excludes stable CKD and never penalizes the model for post-decision severity it did not see. |
| **#2 shift = constraint feature** | Shift axis is `anchor_year_group` (era). The T3 shift component is a dedicated **train-early / evaluate-late** experiment (`experiment.run_shift_eval`), not a spurious later-era subset of an in-distribution random split. |
| **#3 single split / point ECE** | 20 seeds × subject-level splits; T3 gates on the **bootstrap 95% CI upper bound** of ECE, not a point estimate (`experiment.py`, `metrics.bootstrap_ci`). |
| **#4 patient leakage** | `GroupShuffleSplit` on `subject_id` (`experiment.py`). |
| **#5 KDIGO baseline / look-ahead** | Features use only the first 24h; severe-creatinine uses a 7-day window; both windowed against `admittime` (`cohort.py`). |
| **#6 one integration style** | Pipeline-coupled MLP + calibration, differentiable LTN (torch, optional), pure-neural baseline; SPL stub with DeepProbLog fallback (`models.py`). |
| NC-B circularity | Negative controls are **naturally deficient** models, not forced violations (`negative_controls.py`). |

## Layout

```
mimic_pipeline/
├── config.yaml                 # all gated thresholds (mirrors prereg-v1.0)
├── Makefile                    # make check | cohort | run | test
├── requirements.txt
├── auditnesy_mimic/            # the package (one concern per module)
│   ├── config.py               # typed config + env overrides
│   ├── io_utils.py             # table locator, hashing, JSON
│   ├── itemids.py              # STEP 0: verify lab itemids
│   ├── cohort.py               # build AKI cohort (ICD label, windowed features)
│   ├── metrics.py              # ECE, subject-level bootstrap CI, abstention
│   ├── models.py               # pipeline-coupled / LTN / pure-neural
│   ├── audit.py                # T1, T2, T3, T2s concept-semantics probe
│   ├── negative_controls.py    # NC-A' / NC-B' / NC-C' / NC-D'
│   └── experiment.py           # 20-seed orchestration + CI-gated pass
├── scripts/
│   ├── 00_check_itemids.py     # run FIRST
│   ├── 01_build_cohort.py
│   └── 02_run_experiment.py
└── tests/
    └── test_metrics.py         # runs without MIMIC access
```

## Run order

```bash
pip install -r requirements.txt
export MIMIC_DIR=/path/to/mimiciv/hosp
export AUDITNESY_OUT=./auditnesy_mimic_out

python scripts/00_check_itemids.py     # confirm creatinine itemid, then edit itemids.py if needed
python scripts/01_build_cohort.py      # caches aki_cohort.parquet (gitignored)
python scripts/02_run_experiment.py    # writes audit_summary_mimic.json
```

Or: `make all`. Tests need no data: `make test`.

## Outputs

- `audit_summary_mimic.json` — per-axis aggregate: AUROC, T1/T2/T3 pass (CI-gated),
  per-seed stability (e.g. `17/20`), and the four negative-control results.
- `per_seed_results.json` — full per-seed evidence (for the paper's CI tables).

## Pre-registration discipline

`config.yaml` thresholds are frozen at tag `prereg-v1.0`. Changing a gated
threshold after observing results requires a new registered prereg version and a
row in `../preregistration.md` §8 (Deviations Log). T3 borderline results are
reported, not suppressed — surfacing calibration fragility is the point of the
framework.
