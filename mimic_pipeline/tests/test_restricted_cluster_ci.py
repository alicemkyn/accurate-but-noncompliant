"""Guards on the subject-clustered CIs of the restricted cohort (v1.13).

`12_violation_rate_by_renal_history.py` gained a `restricted_cohort_cluster_ci` block
for the W3PHIAI-27 revision. The reviewer-facing promise was "patient-clustered
intervals will be added"; the project-facing promise is that adding them changed
nothing else. Both are tested:

  1. every interval contains its point estimate and has lo <= hi;
  2. the method string names the cluster variable (subject) and the resampling unit;
  3. the pre-existing `restricted_certificate` and `headline_comparison` blocks are
     identical to the copy frozen BEFORE the script was re-run
     (tests/fixtures/violation_rate_by_renal_history.v1_12_frozen.json);
  4. the payload guard refuses a missing block or a None bound;
  5. B and seed are recorded and B equals the configured bootstrap_B.

All mutation-verified; see new_analyses_notes.md.

Run: cd mimic_pipeline && python -m pytest tests/test_restricted_cluster_ci.py -q
"""
import copy
import json
import re
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "results" / "violation_rate_by_renal_history.json"
FROZEN = ROOT / "tests" / "fixtures" / "violation_rate_by_renal_history.v1_12_frozen.json"
SCRIPT = ROOT / "scripts" / "12_violation_rate_by_renal_history.py"
CONFIG = ROOT / "config.yaml"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

STATS = ("violation_rate_deployed", "coverage_severe", "icd_agreement_on_severe")


def _payload():
    if not PAYLOAD.exists():
        pytest.skip("violation_rate_by_renal_history.json not in this checkout")
    return json.loads(PAYLOAD.read_text())


def _block():
    d = _payload()
    assert "restricted_cohort_cluster_ci" in d, "the cluster-CI block is missing"
    return d["restricted_cohort_cluster_ci"]


def _mod():
    return import_module("12_violation_rate_by_renal_history")


def _synthetic_rows(n_seeds=3, seed=1):
    rng = np.random.default_rng(seed)
    rows, pooled = [], []
    for s in range(n_seeds):
        n = 600
        subj = rng.integers(0, 300, n)
        dep = (rng.random(n) < 0.7).astype(float)
        rows.append({"seed": s, "test_subjects": np.arange(400), "subject": subj,
                     "deployed": dep, "violation": dep * (rng.random(n) < 0.5),
                     "aki": (rng.random(n) < 0.6).astype(float)})
        pooled.append({"n_severe": n, "n_deployed": dep.sum()})
    return rows, pooled


# 1 ---------------------------------------------------------------------------
def test_every_interval_is_ordered_and_contains_its_point():
    b = _block()
    for k in STATS:
        m = b["per_seed_median"][k]
        assert m["lo_median"] <= m["hi_median"], (k, m)
        assert m["lo_median"] <= m["point_median"] <= m["hi_median"], (k, m)
        p = b["pooled_across_seed_test_decisions"][k]
        assert p["lo"] <= p["point"] <= p["hi"], (k, p)
        for rec in b["per_seed"]:
            assert rec[k]["lo"] <= rec[k]["point"] <= rec[k]["hi"], (k, rec["seed"], rec[k])
    assert len(b["per_seed"]) == b["per_seed_median"][STATS[0]]["n_seeds"]


def test_compute_cluster_cis_brackets_the_point_on_synthetic_data():
    """Function-level: the interval must bracket the point, with lo <= hi, on fresh data."""
    m = _mod()
    out = m.compute_cluster_cis(*_synthetic_rows(), B=300, seed=0)
    for k in STATS:
        for rec in out["per_seed"]:
            assert rec[k]["lo"] <= rec[k]["point"] <= rec[k]["hi"], (k, rec)
            assert rec[k]["hi"] - rec[k]["lo"] > 0, "degenerate interval"
        p = out["pooled_across_seed_test_decisions"][k]
        assert p["lo"] <= p["point"] <= p["hi"], (k, p)


# 2 ---------------------------------------------------------------------------
def test_method_string_names_subject_clustering():
    b = _block()
    s = b["method"].lower()
    assert "subject" in s and "cluster" in s and "bootstrap" in s, b["method"]
    assert b["cluster_variable"] == "subject_id"
    assert "descriptive" in b["pooled_across_seed_test_decisions"]["interpretation"].lower()
    assert "several seeds" in b["method"] or "several seeds" in \
        b["pooled_across_seed_test_decisions"]["interpretation"]


# 3 ---------------------------------------------------------------------------
def test_pre_existing_blocks_are_unchanged_from_the_frozen_v1_12_copy():
    if not FROZEN.exists():
        pytest.skip("frozen copy not in this checkout")
    new, old = _payload(), json.loads(FROZEN.read_text())
    for blk in ("restricted_certificate", "headline_comparison", "per_stratum"):
        assert new[blk] == old[blk], f"{blk} changed when the cluster-CI block was added"
    # the frozen copy must itself be free of the new block (it predates it) and of identifiers
    assert "restricted_cohort_cluster_ci" not in old
    assert not re.search(r'"(hadm_id|subject_id|case_id)"', FROZEN.read_text())


# 4 ---------------------------------------------------------------------------
def test_guard_refuses_missing_block_or_none_bound():
    m = _mod()
    good = {"restricted_cohort_cluster_ci": _block()}
    assert m.missing_cluster_ci_keys(good) == []
    assert m.missing_cluster_ci_keys({}) == ["restricted_cohort_cluster_ci"]
    bad = copy.deepcopy(good)
    bad["restricted_cohort_cluster_ci"]["pooled_across_seed_test_decisions"][STATS[0]]["lo"] = None
    assert any("pooled" in k and STATS[0] in k for k in m.missing_cluster_ci_keys(bad))
    bad = copy.deepcopy(good)
    bad["restricted_cohort_cluster_ci"]["per_seed"][3][STATS[1]]["hi"] = None
    assert any("per_seed[" in k and STATS[1] in k for k in m.missing_cluster_ci_keys(bad))
    bad = copy.deepcopy(good)
    bad["restricted_cohort_cluster_ci"]["per_seed_median"][STATS[2]]["lo_median"] = None
    assert any("per_seed_median" in k and STATS[2] in k for k in m.missing_cluster_ci_keys(bad))


def test_guard_is_wired_to_the_refusal_path():
    src = SCRIPT.read_text(errors="ignore")
    i = src.find("missing += missing_cluster_ci_keys(payload)")
    assert i != -1, "the cluster-CI guard is not called before the write"
    window = src[i:i + 800]
    assert "raise SystemExit" in window and "write_text" not in src[:i].split("def main")[-1], (
        "the guard must precede the write and refuse, not warn")


# 5 ---------------------------------------------------------------------------
def test_b_and_seed_recorded_and_match_config():
    b = _block()
    assert isinstance(b["B"], int) and b["B"] >= 1000
    assert isinstance(b["seed"], int)
    assert b["ci_level"] == 0.95
    if CONFIG.exists():
        m = re.search(r"bootstrap_B:\s*(\d+)", CONFIG.read_text(errors="ignore"))
        if m:
            assert b["B"] == int(m.group(1)), "B must come from config when configured"
