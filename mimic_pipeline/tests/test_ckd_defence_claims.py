"""Guards on the leave-one-stratum-out defence (block H of the prior-CKD script).

Finding 2 -- the label-guideline gap concentrates in patients with prior renal
replacement -- invites the objection that the headline is an artifact of that
population. Block H answers it by DELETION: remove the stratum, report what is
left. That is only defensible while three properties hold, and each has a test:

  1. the three strata partition all severe cases exactly, so the deletion is a
     reweighting of an existing measurement rather than a new analysis;
  2. the reweighted total reproduces the paper's own cohort agreement, which is
     the arithmetic check that it is the same population;
  3. the paper states the bound in the BODY, not only in the appendix -- a
     defence a reviewer will not reach is not a defence.

A fourth test keeps the block-H strata distinct from the nested pair used for the
block G reading: conflating them once put a partition claim and a non-partition
claim in adjacent paragraphs of the paper.

Run: cd mimic_pipeline && python -m pytest tests/test_ckd_defence_claims.py -q
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "results" / "prior_ckd_sensitivity.json"
PAPER = ROOT.parent / "AuditNeSy_v26.tex"
SCRIPT = ROOT / "scripts" / "10_prior_ckd_sensitivity.py"


def _payload():
    if not PAYLOAD.exists():
        pytest.skip("prior_ckd_sensitivity.json not in this checkout")
    return json.loads(PAYLOAD.read_text())


def _H():
    d = _payload()
    assert "H_headline_without_the_rrt_stratum" in d, (
        "block H is missing from the results file; the deletion defence has no source")
    return d["H_headline_without_the_rrt_stratum"]


def _paper():
    if not PAPER.exists():
        pytest.skip("paper source not in this checkout")
    return re.sub(r"\s+", " ", PAPER.read_text(errors="ignore"))


def test_strata_partition_all_severe_cases():
    """Deletion is only a reweighting if the strata cover every severe case once."""
    d = _payload()
    H = _H()
    assert H["strata_partition_all_severe_cases"] is True
    assert H["n_severe_all"] == d["n_severe_cases"], (
        "block H's denominator must be the full severe cohort")
    g = d["G_renal_replacement_stratification"]
    cells = [g["strata"]["prior_renal_replacement"]["n"],
             g["strata"]["no_prior_renal_replacement"]["n"],
             g["no_observed_history"]["n"]]
    assert sum(cells) == H["n_severe_all"], f"strata sum {sum(cells)} != {H['n_severe_all']}"


def test_partition_strata_are_not_the_nested_reading_strata():
    """The gradient contrast is nested; the deletion triple is disjoint. Keep them apart."""
    d = _payload()
    g = d["G_renal_replacement_stratification"]["strata"]
    inner = g["no_prior_renal_disease_at_all"]["n"]
    outer = g["no_prior_renal_replacement"]["n"]
    assert inner < outer, (
        "the reading contrast is expected to be nested (no-disease inside "
        f"no-replacement); got {inner} and {outer}")
    assert _H().get("strata_are_disjoint_not_the_nested_reading_strata") is True, (
        "block H must declare that its triple is the disjoint one")
    s = _paper()
    assert "nested" in s, "the paper must say the gradient contrast is nested"
    assert "disjoint" in s, "the paper must say the deletion triple is disjoint"


def test_reweighted_agreement_reproduces_the_cohort_figure():
    """If the reweighting is faithful, the all-strata mean must equal the headline."""
    d = _payload()
    H = _H()
    g = d["G_renal_replacement_stratification"]
    cells = [(g["strata"]["prior_renal_replacement"]["n"],
              g["strata"]["prior_renal_replacement"]["agreement"]),
             (g["strata"]["no_prior_renal_replacement"]["n"],
              g["strata"]["no_prior_renal_replacement"]["agreement"]),
             (g["no_observed_history"]["n"], g["no_observed_history"]["agreement"])]
    recomputed = sum(n * a for n, a in cells) / sum(n for n, _ in cells)
    assert abs(recomputed - H["agreement_all_severe"]) < 5e-4
    kept = cells[1:]
    kept_agr = sum(n * a for n, a in kept) / sum(n for n, _ in kept)
    assert abs(kept_agr - H["excluding_prior_rrt"]["agreement"]) < 5e-4


def test_two_denominators_are_both_reported():
    """The 58% and 41% figures use different denominators; hiding either misleads."""
    H = _H()
    a = H["prior_rrt_share_of_all_severe"]
    b = H["prior_rrt_share_of_observed_history"]
    assert a < b, "the all-severe share must be the smaller of the two"
    reading = H["reading"]
    assert f"{100*a:.1f}%" in reading and f"{100*b:.0f}%" in reading, (
        "the reading must name both denominators, since quoting only the larger "
        f"overstates the concentration: {reading}")


def test_defence_is_stated_in_the_body_not_only_the_appendix():
    """A reviewer reads the abstract and the limitations; the bound must be there."""
    s = _paper()
    body = s[: s.find(r"\appendix")]
    for probe in ("3{,}507", "43\\%"):
        assert probe in body, f"the surviving-cases bound is missing from the body: {probe}"
    assert "artifact of that population" in body or "artifact of it" in body, (
        "the body must state what the concentration finding does NOT imply")


def test_block_h_claims_no_new_modelling():
    """Block H must stay a reweighting: no fitting, no new threshold in its code."""
    if not SCRIPT.exists():
        pytest.skip("script not in this checkout")
    src = SCRIPT.read_text(errors="ignore")
    i = src.find("robustness = {")
    assert i != -1, "block H disappeared from the script"
    j = src.find("payload = {", i)
    block = src[src.rfind("# ----------", 0, i):j]
    for banned in (".fit(", "LogisticRegression", "bootstrap", "np.random", "threshold ="):
        assert banned not in block, (
            f"block H must be a deterministic reweighting; found {banned!r}")


def test_block_h_points_at_the_model_side_arm():
    """Block H bounds the LABEL only; it must not be read as bounding the headline."""
    H = _H()
    note = H.get("what_this_does_not_cover", "")
    assert "12_violation_rate_by_renal_history" in note, (
        "block H must point at the arm that applies the same deletion to the model "
        f"statistic: {note!r}")


def test_post_hoc_status_is_not_laundered_by_the_defence():
    """Block H inherits Finding 2's exploratory status and must not obscure it."""
    s = _paper()
    i = s.find("Leave the stratum out entirely")
    assert i != -1, "the deletion paragraph disappeared from the appendix"
    window = s[i:i + 1600]
    assert "reweighting" in window, "the paragraph must say what kind of analysis it is"
    assert "no new model" in window or "introducing no new model" in window, (
        "the paragraph must state that nothing new was fitted")
