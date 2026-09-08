"""Guards on what the chronicity proxy is allowed to claim.

The proxy in `08_stratified_label_validity.py` block D exists because the correct
stratifier -- a CKD diagnosis code from a PRIOR admission -- is not computable
from the released cohort derivative. Two failure modes would be worse than not
running the analysis at all, and both are the kind of drift that creeps in when a
result file is edited by hand:

  1. The proxy being described, in the results file or the script, as a CKD
     measurement. It is a creatinine-shape heuristic; a paper that audits label
     validity must not itself relabel a proxy as the thing it proxies.
  2. A same-admission CKD code being used as the stratifier. MIMIC-IV
     `diagnoses_icd` rows are assigned after discharge from the signed record, so
     a same-admission code is produced by the very coding process under audit.

These run without MIMIC access; the block-D assertions skip (not pass) when the
result file is absent, so a missing file cannot look like a green suite.

    cd mimic_pipeline && python -m pytest tests/ -q
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "08_stratified_label_validity.py"
RESULT = ROOT / "results" / "stratified_label_validity.json"


def _payload():
    if not RESULT.exists():
        pytest.skip(f"{RESULT.name} not present; run script 08 first")
    return json.loads(RESULT.read_text())


def test_script_does_not_read_a_same_admission_ckd_code():
    """The stratifier must not be a CKD code from the audited admission."""
    src = SCRIPT.read_text()
    # a CKD code family appearing OUTSIDE a comment line would mean it is being used
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
            continue
        for token in ("585", "N18", "icd_code", "diagnoses_icd"):
            assert token not in line, (
                f"script 08 appears to read a CKD/diagnosis code in executable code: {stripped}")


def test_results_file_declares_the_proxy_is_not_a_ckd_measurement():
    d = _payload()["D_chronicity_proxy_within_A1"]
    assert d["is_a_ckd_measurement"] is False, (
        "block D must declare explicitly that the proxy is not a CKD measurement")
    assert "proxy" in d["rationale"].lower()
    assert "not" in d["rationale"].lower() and "code" in d["rationale"].lower(), (
        "the rationale must say why a CKD code was not used")


def test_uncomputable_ckd_analysis_is_specified_not_merely_named():
    """A limitation is only useful if the missing analysis is reproducible from it."""
    nc = _payload()["not_computable"]["ckd_stratification_by_prior_diagnosis_code"]
    for token in ("diagnoses_icd", "dischtime", "admittime", "585", "N18"):
        assert token in nc, f"the uncomputable analysis is under-specified: missing {token}"
    assert "EXCLUDING the current admission" in nc


def test_chronicity_verdict_is_not_a_single_cut():
    """A verdict resting on one arbitrary threshold is not a verdict."""
    d = _payload()["D_chronicity_proxy_within_A1"]
    assert len(d["cut_sweep"]) >= 3, "the proxy cut must be swept, not asserted once"
    assert d["primary_cut"] in d["cut_sweep"]
    tested = [cells["_white_minus_black"]
              for entry in d["by_cut"].values() for cells in entry.values()
              if "_white_minus_black" in cells]
    assert tested, "no testable cell carried a gap estimate"
    # every reported gap must carry an interval, not just a point
    for g in tested:
        assert "ci95" in g and len(g["ci95"]) == 2
        assert g["ci95"][0] <= g["point"] <= g["ci95"][1], (
            f"point estimate outside its own interval: {g}")


def test_observability_check_is_measured_not_asserted():
    """Block E's verdict must follow from the shares it reports beside it."""
    e = _payload()["E_prior_admission_observability"]
    shares = [v["share_with_prior_admission"] for k, v in e.items() if not k.startswith("_")]
    assert shares, "block E reported no per-group shares"
    spread = max(shares) - min(shares)
    v = e["_verdict"]
    assert abs(v["max_minus_min_share"] - spread) < 1e-9, (
        f"recorded spread {v['max_minus_min_share']} != computed {spread:.4f}")
    assert v["differentially_observable"] == (spread > 0.05), (
        "the verdict contradicts the shares in the same file")


def test_regression_is_not_presented_as_causal():
    f = _payload()["F_regression_robustness"]
    assert f["is_causal_adjustment"] is False
    assert "cluster bootstrap" in f["inference"], (
        "regression inference must use the same clustering as the rest of the script")
