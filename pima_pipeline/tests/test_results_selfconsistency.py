"""Self-consistency guards: a results file must not contradict its own numbers.

Two defects of exactly this shape were found in review and are guarded here:

 1. `pima_audit.json` described the reference model as "isotonic-calibrated"
    while the script ran it with no calibration wrapper -- a durable file
    recording the wrong provenance for the model whose ECE it reports.
 2. `pima_styles.json` asserted "Both arms ... pass T1 and T3" while the same
    file recorded an LTN ECE of 0.111 against a 0.10 bound.

Both were narrative strings written independently of the measurements. These
tests check the invariant that matters: every verdict word in a results file
must be recomputable from the numbers beside it.

    cd pima_pipeline && python -m pytest tests/ -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent / "mimic_pipeline"))

_spec = importlib.util.spec_from_file_location("pima_audit", _ROOT / "scripts" / "10_pima_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

AUDIT = _ROOT / "results" / "pima_audit.json"
STYLES = _ROOT / "results" / "pima_styles.json"


def _load(p: Path):
    if not p.exists():  # pragma: no cover
        pytest.skip(f"{p.name} not present; run the scripts first")
    return json.loads(p.read_text())


def test_audit_records_the_calibration_actually_used():
    d = _load(AUDIT)
    recorded = d["protocol"]["calibration"]
    expected = pa.CALIBRATE or "none"
    assert recorded == expected, (
        f"results file records calibration={recorded!r} but the script runs {expected!r}")
    text = d["protocol"]["model"].lower()
    if pa.CALIBRATE is None:
        assert "calibrated" not in text.replace("no calibration", ""), (
            f"model string claims calibration that did not happen: {d['protocol']['model']!r}")


def test_audit_negative_control_does_not_claim_a_calibrated_baseline():
    d = _load(AUDIT)
    mech = d["negative_controls"]["NC_A_uncalibrated"]["mechanism"].lower()
    if pa.CALIBRATE is None:
        assert "the calibrated probabilities" not in mech, (
            "NC-A mechanism describes calibrated probabilities, but the reference "
            "model carries no calibration wrapper")


def test_audit_t3_verdict_follows_from_its_own_ece():
    d = _load(AUDIT)
    ece_val = d["reference_split"]["ece"]
    bound = d["protocol"]["ece_bound"]
    assert d["axes_reference_split"]["T3_pass"] == (ece_val < bound), (
        f"T3 verdict {d['axes_reference_split']['T3_pass']} contradicts ECE {ece_val} "
        f"against bound {bound}")


def test_styles_arm_verdicts_follow_from_their_own_ece():
    d = _load(STYLES)
    bound = d["ece_bound"]
    for name, arm in d["arms"].items():
        assert arm["T3_pass"] == (arm["ece"] < bound), (
            f"{name}: T3_pass={arm['T3_pass']} contradicts ECE {arm['ece']} vs bound {bound}")


def test_styles_summary_sentence_matches_the_measured_verdicts():
    """The regression that was actually shipped: a summary contradicting the data."""
    d = _load(STYLES)
    reading = d["reading"].lower()
    failing = [n for n, a in d["arms"].items() if not a["T3_pass"]]
    if failing:
        assert "does not" in reading or "neither" in reading, (
            f"arms {failing} fail T3, but the summary reads as an unqualified pass: "
            f"{d['reading']!r}")
        assert "both arms" not in reading or "both pass t1" in reading, (
            "summary generalises over both arms while one of them fails an axis")
    else:
        assert "does not" not in reading, (
            "summary reports a T3 failure that the measurements do not show")


def test_styles_summary_is_not_a_hardcoded_literal():
    """Guard the mechanism, not just the current output."""
    src = (_ROOT / "scripts" / "11_pima_styles.py").read_text()
    i = src.find('"reading":')
    assert i != -1, "reading key disappeared from the styles script"
    line = src[i:src.find("\n", i)]
    assert "_reading(" in line, (
        f"the summary must be derived from measurements, not written inline: {line.strip()}")
