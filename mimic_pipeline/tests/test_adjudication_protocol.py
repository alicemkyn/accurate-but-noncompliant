"""Tests for the clinician-adjudication protocol (`11_clinician_adjudication.py`).

The adjudication itself needs clinicians and has not been run. Shipping the
protocol unrun is only defensible if the parts that CAN be checked are checked,
so these tests establish, on synthetic responses:

  * the packet cannot leak the answer (no verdict, label or identifier column);
  * the failing branch is reachable -- a run where clinicians side with the
    coders rather than the rule produces the verdict that narrows our claim;
  * the passing branch is reachable and does not fire on the failing data;
  * the declared falsification thresholds are literals in the source, not
    values computed from data the script has seen.

Run: cd mimic_pipeline && python -m pytest tests/test_adjudication_protocol.py -q
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "11_clinician_adjudication.py"
PACKET = ROOT / "results" / "adjudication_packet.csv"
KEY = ROOT / "results" / "adjudication_key.json"

FORBIDDEN = {"aki", "severe_acute", "cell", "hadm_id", "subject_id",
             "rule_severe", "administrative_code", "prediction", "model_pred"}


def _load_key():
    if not KEY.exists():
        pytest.skip("adjudication key absent; run `11_clinician_adjudication.py sample` first")
    return json.loads(KEY.read_text())


def test_packet_does_not_leak_the_answer():
    if not PACKET.exists():
        pytest.skip("packet absent; run the sample step first")
    cols = set(next(csv.reader(PACKET.open())))
    assert not (cols & FORBIDDEN), f"packet leaks: {sorted(cols & FORBIDDEN)}"
    assert "clinician_aki" in cols, "packet must have a column for the adjudication"


def test_key_is_not_shipped_inside_the_packet():
    """The unblinding key must be a separate file, never a packet column."""
    key = _load_key()
    assert "cases" in key and key["cases"], "key carries no case mapping"
    assert "do not send" in key["status"].lower()


def test_sampling_oversamples_the_headline_cell():
    """Cell B (rule-severe, uncoded) is where the paper's claim lives."""
    key = _load_key()
    prov = key["provenance"]
    assert prov["B_severe_uncoded"]["drawn"] >= prov["A_severe_coded"]["drawn"], (
        "the headline cell must not be under-sampled relative to the agreeing cell")


def _write_responses(tmp_path, key, side_with):
    """Synthesise one reviewer who always sides with `side_with` ('rule' or 'code')."""
    p = tmp_path / f"reviewer_{side_with}.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "clinician_aki"])
        for c in key["cases"]:
            v = int(c["rule_severe"]) if side_with == "rule" else int(c["administrative_code"])
            w.writerow([c["case_id"], v])
    return p


def _score(tmp_path, *responses):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "score", "--key", str(KEY),
         "--responses", *[str(x) for x in responses], "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads((out / "clinician_adjudication.json").read_text())


def test_passing_branch_is_reachable(tmp_path):
    key = _load_key()
    res = _score(tmp_path, _write_responses(tmp_path, key, "rule"))
    assert "endorse the rule" in res["verdict"], res["verdict"]
    assert res["per_cell"]["B_severe_uncoded"]["clinician_vs_rule_agreement"] == 1.0


def test_failing_branch_is_reachable_and_narrows_our_claim(tmp_path):
    """If clinicians side with the coders, the script must say so."""
    key = _load_key()
    res = _score(tmp_path, _write_responses(tmp_path, key, "code"))
    assert "FAILS its declared validation" in res["verdict"], res["verdict"]
    assert "narrowed" in res["verdict"] or "conditional" in res["verdict"]
    assert res["per_cell"]["B_severe_uncoded"]["clinician_vs_rule_agreement"] < \
        res["declared_thresholds"]["min_agreement_cell_B"]


def test_pooled_figures_never_appear_without_the_per_cell_caveat(tmp_path):
    key = _load_key()
    res = _score(tmp_path, _write_responses(tmp_path, key, "rule"))
    assert "composition was chosen by us" in res["overall"]["note"]


def test_packet_and_key_are_patient_level_and_must_not_be_released():
    """Both files carry patient-level rows; the release bundle must exclude them.

    The packet is 200 rows of demographics and labs, and the key maps them to
    MIMIC identifiers. Neither may travel in the code release, whatever the
    availability statement promises -- so the statement promises the seed and the
    stratum targets instead, which regenerate the packet for anyone credentialed.
    """
    key = _load_key()
    assert "sample_seed" in key and "cell_targets" in key, (
        "the packet must be regenerable from recorded parameters, since it cannot ship")
    assert {"hadm_id", "subject_id"} <= set(key["cases"][0]), (
        "key is expected to carry identifiers -- that is why it is withheld")
    tex = ROOT.parent / "AuditNeSy_v26.tex"
    if not tex.exists():
        pytest.skip("paper source not in this checkout")
    # Collapse whitespace before matching: LaTeX line-wraps sentences, so an
    # exact-with-newline probe fails the moment a nearby edit re-flows the
    # paragraph -- which is a test breaking on formatting, not on meaning.
    s = re.sub(r"\s+", " ", tex.read_text(errors="ignore"))
    claims = ["are \\emph{not} released", "not itself released"]
    assert any(c in s for c in claims), (
        "the paper must state that the packet and key are withheld; looked for "
        f"{claims}")


def test_falsification_thresholds_are_declared_literals():
    """They must be fixed in source, not derived from anything the script saw."""
    src = SCRIPT.read_text()
    i = src.find("FALSIFICATION = {")
    assert i != -1, "declared thresholds disappeared from the script"
    line = src[i:src.find("\n", i)]
    assert "0.70" in line and "0.60" in line, line
    for bad in ("np.", "df[", "mean()", "quantile"):
        assert bad not in line, f"thresholds must be literals, found {bad!r}: {line}"
