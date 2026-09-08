"""Tests for `10_prior_ckd_sensitivity.py`, which ships UNRUN.

The script needs credentialed MIMIC-IV raw tables and could not be executed where
it was written. Releasing unrun code with a claim that it implements a specific
design is only honest if the design is verified some other way, so these tests
exercise its two load-bearing rules on synthetic fixtures that need no MIMIC
access:

  1. A CKD code on the CURRENT admission must never set the indicator. That code
     is assigned post-discharge by the coding process under audit; using it would
     condition one product of that process on another.
  2. A prior admission counts only if it DISCHARGED before the current admission
     was admitted. An overlapping or later admission must not count.

Plus a guard that the headline sentence is derived from the measured cells rather
than written, and a check that the indicator is not named "has CKD".

    cd mimic_pipeline && python -m pytest tests/ -q
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_ckd = import_module("10_prior_ckd_sensitivity")
SCRIPT = ROOT / "scripts" / "10_prior_ckd_sensitivity.py"


def _write_fixture(tmp_path, dx_rows, adm_rows):
    pd.DataFrame(adm_rows).to_csv(tmp_path / "admissions.csv", index=False)
    pd.DataFrame(dx_rows).to_csv(tmp_path / "diagnoses_icd.csv", index=False)
    return tmp_path


def test_same_admission_ckd_code_does_not_set_the_indicator(tmp_path):
    """The core rule: a CKD code on the audited admission is invisible to the indicator."""
    hosp = _write_fixture(
        tmp_path,
        dx_rows=[
            # CKD coded on the CURRENT admission only
            {"subject_id": 1, "hadm_id": 100, "icd_code": "N18.3", "icd_version": 10},
        ],
        adm_rows=[
            {"subject_id": 1, "hadm_id": 100,
             "admittime": "2150-01-10 08:00:00", "dischtime": "2150-01-20 08:00:00"},
        ])
    cohort = pd.DataFrame({"subject_id": [1], "hadm_id": [100]})
    out = _ckd.build_prior_indicators(cohort, hosp)
    assert bool(out["_prior_ckd"].iloc[0]) is False, (
        "a same-admission CKD code must not set prior_ckd_documented")
    assert bool(out["_prior_obs"].iloc[0]) is False


def test_prior_admission_counts_only_if_discharged_before_current_admit(tmp_path):
    """Timing rule: earlier discharge counts; overlapping and later admissions do not."""
    hosp = _write_fixture(
        tmp_path,
        dx_rows=[
            {"subject_id": 1, "hadm_id": 1, "icd_code": "5859", "icd_version": 9},   # earlier
            {"subject_id": 2, "hadm_id": 3, "icd_code": "N18.4", "icd_version": 10},  # overlapping
            {"subject_id": 3, "hadm_id": 5, "icd_code": "N18.4", "icd_version": 10},  # later
        ],
        adm_rows=[
            # subject 1: prior admission discharged BEFORE current -> counts
            {"subject_id": 1, "hadm_id": 1,
             "admittime": "2149-01-01 00:00:00", "dischtime": "2149-01-05 00:00:00"},
            {"subject_id": 1, "hadm_id": 2,
             "admittime": "2150-01-10 00:00:00", "dischtime": "2150-01-20 00:00:00"},
            # subject 2: other admission OVERLAPS the current one -> must not count
            {"subject_id": 2, "hadm_id": 3,
             "admittime": "2150-01-05 00:00:00", "dischtime": "2150-01-15 00:00:00"},
            {"subject_id": 2, "hadm_id": 4,
             "admittime": "2150-01-10 00:00:00", "dischtime": "2150-01-20 00:00:00"},
            # subject 3: other admission is LATER -> must not count
            {"subject_id": 3, "hadm_id": 5,
             "admittime": "2151-01-01 00:00:00", "dischtime": "2151-01-05 00:00:00"},
            {"subject_id": 3, "hadm_id": 6,
             "admittime": "2150-01-10 00:00:00", "dischtime": "2150-01-20 00:00:00"},
        ])
    cohort = pd.DataFrame({"subject_id": [1, 2, 3], "hadm_id": [2, 4, 6]})
    out = _ckd.build_prior_indicators(cohort, hosp).set_index("hadm_id")
    assert bool(out.loc[2, "_prior_ckd"]) is True, "an earlier discharged admission must count"
    assert bool(out.loc[2, "_prior_obs"]) is True
    assert bool(out.loc[4, "_prior_ckd"]) is False, "an overlapping admission must not count"
    assert bool(out.loc[4, "_prior_obs"]) is False
    assert bool(out.loc[6, "_prior_ckd"]) is False, "a later admission must not count"
    assert bool(out.loc[6, "_prior_obs"]) is False


def test_code_matching_respects_icd_version_and_dotless_form(tmp_path):
    """585.x is ICD-9 and N18.x is ICD-10; the same digits under the wrong version must not hit."""
    hosp = _write_fixture(
        tmp_path,
        dx_rows=[
            {"subject_id": 1, "hadm_id": 1, "icd_code": "585.9", "icd_version": 9},   # hit
            {"subject_id": 2, "hadm_id": 3, "icd_code": "N183", "icd_version": 10},   # hit, dotless
            {"subject_id": 3, "hadm_id": 5, "icd_code": "585", "icd_version": 10},    # wrong version
            {"subject_id": 4, "hadm_id": 7, "icd_code": "5849", "icd_version": 9},    # AKI, not CKD
        ],
        adm_rows=[
            {"subject_id": s, "hadm_id": h,
             "admittime": "2149-01-01 00:00:00", "dischtime": "2149-01-05 00:00:00"}
            for s, h in [(1, 1), (2, 3), (3, 5), (4, 7)]
        ] + [
            {"subject_id": s, "hadm_id": h,
             "admittime": "2150-01-10 00:00:00", "dischtime": "2150-01-20 00:00:00"}
            for s, h in [(1, 2), (2, 4), (3, 6), (4, 8)]
        ])
    cohort = pd.DataFrame({"subject_id": [1, 2, 3, 4], "hadm_id": [2, 4, 6, 8]})
    out = _ckd.build_prior_indicators(cohort, hosp).set_index("hadm_id")
    assert bool(out.loc[2, "_prior_ckd"]) is True, "585.9 under ICD-9 must match"
    assert bool(out.loc[4, "_prior_ckd"]) is True, "dotless N183 under ICD-10 must match"
    assert bool(out.loc[6, "_prior_ckd"]) is False, "585 under ICD-10 must not match"
    assert bool(out.loc[8, "_prior_ckd"]) is False, "an AKI code must not match CKD"


def test_indicator_is_not_named_has_ckd():
    """Naming matters: absence of a prior code is not absence of disease."""
    src = SCRIPT.read_text()
    assert "prior MIMIC hospitalisation" in src
    for banned in ('"has_ckd"', "'has_ckd'", '"no_ckd_patient"'):
        assert banned not in src, f"indicator must not be named {banned}"


def test_headline_reading_is_derived_from_the_cells():
    """The summary sentence must be built from measured verdicts, not written inline."""
    src = SCRIPT.read_text()
    i = src.find('"reading": reading')
    assert i != -1, "the payload must carry a `reading` built in code"
    # all four branches must exist, so the script can report a negative result too
    for branch in ("persists in BOTH", "part, but not all", "no longer distinguishable",
                   "UNDETERMINED"):
        assert branch in src, f"missing branch for outcome: {branch}"


def test_script_refuses_to_run_without_raw_tables():
    """Absent MIMIC access it must fail loudly, not silently produce a partial result."""
    src = SCRIPT.read_text()
    assert "raise SystemExit" in src
    assert "credentialed raw tables" in src
