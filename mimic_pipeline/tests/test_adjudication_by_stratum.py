"""Guards on the adjudication cross-tabulation (`13_adjudication_by_stratum.py`, v1.13).

Every test here was mutation-verified: the script was deliberately broken in the way
the test names, the test failed, and the script was restored byte-identical. The
mutations are recorded in `new_analyses_notes.md`.

  1. strata are a partition of the 200 adjudicated cases;
  2. decided + blank = n in every cell of every table;
  3. a blank is "cannot tell", never a disagreement (mutation: score blank as 0);
  4. the written JSON carries no identifier key at any depth;
  5. the trajectory classes reproduce script 05's taxonomy rule exactly;
  6. per-cell counts agree with results/clinician_adjudication.json.

Run: cd mimic_pipeline && python -m pytest tests/test_adjudication_by_stratum.py -q
"""
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "results" / "adjudication_by_stratum.json"
ADJ = ROOT / "results" / "clinician_adjudication.json"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

TABLES = ("per_renal_history", "per_trajectory", "per_cell", "cell_x_renal_history",
          "cell_x_trajectory", "renal_history_x_trajectory")
FORBIDDEN = {"hadm_id", "subject_id", "case_id"}


def _payload():
    if not PAYLOAD.exists():
        pytest.skip("adjudication_by_stratum.json not in this checkout")
    return json.loads(PAYLOAD.read_text())


def _mod():
    return import_module("13_adjudication_by_stratum")


# 1 ---------------------------------------------------------------------------
def test_strata_partition_the_adjudicated_cases():
    d = _payload()
    assert d["n_cases"] == 200
    for tbl in TABLES:
        total = sum(v["n_cases"] for v in d[tbl].values())
        assert total == d["n_cases"], f"{tbl}: cell n_cases sum to {total}, not {d['n_cases']}"
    assert sum(v["n_decided"] for v in d["per_renal_history"].values()) == d["n_decided"]
    assert sum(v["n_blank"] for v in d["per_renal_history"].values()) == d["n_blank"]


# 2 ---------------------------------------------------------------------------
def test_decided_plus_blank_equals_n_in_every_cell():
    d = _payload()
    for tbl in TABLES:
        for k, v in d[tbl].items():
            assert v["n_decided"] + v["n_blank"] == v["n_cases"], f"{tbl}:{k}: {v}"
            assert 0 <= v["n_agree_with_rule"] <= v["n_decided"], f"{tbl}:{k}: {v}"
            if v["n_decided"]:
                assert abs(v["agreement_on_decided"] - v["n_agree_with_rule"] / v["n_decided"]) < 1e-3, (
                    f"{tbl}:{k}: rate {v['agreement_on_decided']} is not agree/decided")
            else:
                assert v["agreement_on_decided"] is None
            assert abs(v["blank_share"] - v["n_blank"] / v["n_cases"]) < 1e-3
    assert d["n_decided"] + d["n_blank"] == d["n_cases"]


# 3 ---------------------------------------------------------------------------
def test_blank_is_never_scored_as_disagreement():
    """Synthetic: two answered cases agree with the rule, one case is blank.

    The blank must leave the agreement at 1.0 and the decided count at 2. Scoring the
    blank as 0 (the mutation) drags the rate to 2/3 or inflates n_decided to 3.
    """
    m = _mod()
    cases = pd.DataFrame({
        "case_id": [1, 2, 3], "cell": ["B", "B", "B"], "rule_severe": [True, True, True],
        "decided": [True, True, False], "clinician": [1.0, 1.0, np.nan],
    })
    cases["rule"] = cases["rule_severe"].astype(int)
    out = m.summarise(m.score_cases(cases), "cell")["B"]
    assert out["n_cases"] == 3 and out["n_decided"] == 2 and out["n_blank"] == 1, out
    assert out["n_agree_with_rule"] == 2, out
    assert out["agreement_on_decided"] == 1.0, f"a blank was counted against the rule: {out}"
    # a cell of blanks only has no rate at all, not a rate of zero
    only_blank = m.summarise(m.score_cases(cases.iloc[[2]]), "cell")["B"]
    assert only_blank["agreement_on_decided"] is None, only_blank
    assert only_blank["n_agree_with_rule"] == 0


def test_blank_handling_is_stated_in_the_payload():
    d = _payload()
    assert "cannot tell" in d["blank_handling"] and "never" in d["blank_handling"]


# 4 ---------------------------------------------------------------------------
def test_no_identifier_key_at_any_depth():
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                assert k not in FORBIDDEN, f"identifier key at {path}{k}"
                walk(v, path + k + ".")
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
    walk(_payload())
    # and the script's own refusal guard still exists
    src = (ROOT / "scripts" / "13_adjudication_by_stratum.py").read_text(errors="ignore")
    assert "FORBIDDEN_KEYS" in src and "refusing to write" in src


# 5 ---------------------------------------------------------------------------
def test_trajectory_classes_match_script_05_taxonomy_rule():
    """Per-row labels must reproduce `taxonomy_for_split`'s axis-A marginals."""
    m = _mod()
    sg = import_module("05_subgroup_audit")
    rng = np.random.default_rng(0)
    n, thr = 2000, 4.0
    nadir = rng.uniform(0.2, 6.0, n)
    nadir[rng.random(n) < 0.05] = np.nan
    severe = np.ones(n, bool)                        # every row counted
    aki = rng.integers(0, 2, n)
    probs = np.full(n, 0.1)                           # every row a confident denial
    tax = sg.taxonomy_for_split(probs, aki, severe, 0.5, nadir, nadir, thr, 1.5)
    labels = m.trajectory_class(nadir, thr)
    for cls in m.TRAJECTORY_ORDER:
        assert int((labels == cls).sum()) == tax["marginals"][cls]["n_severe_in_cell"], cls
    assert (labels == "A1_elevated_at_arrival").sum() > 0 and (labels == "A3_rise_from_elevated_base").sum() > 0


# 6 ---------------------------------------------------------------------------
def test_per_cell_counts_agree_with_the_scored_adjudication():
    if not ADJ.exists():
        pytest.skip("clinician_adjudication.json not in this checkout")
    d, a = _payload(), json.loads(ADJ.read_text())
    assert d["n_decided"] == a["n_cases_adjudicated"]
    assert d["n_blank"] == a["n_cases_left_blank_by_every_reviewer"]
    for cell, rec in a["per_cell"].items():
        c = d["per_cell"][cell]
        assert c["n_decided"] == rec["n"], cell
        assert abs(c["agreement_on_decided"] - rec["clinician_vs_rule_agreement"]) < 1e-3, cell


def test_small_cells_are_flagged():
    d = _payload()
    flagged = set(d["small_cell_warning"])
    for tbl in TABLES:
        if tbl == "per_cell":
            continue
        for k, v in d[tbl].items():
            if v["n_decided"] < 10:
                assert f"{tbl}:{k}" in flagged, f"small cell {tbl}:{k} not in small_cell_warning"
