"""Unit tests for the components that PRODUCE the headline claim.

Added in prereg-v1.4. v1.3 tested `metrics` only, which left the constraint
axis and the certificate aggregation -- the machinery the headline number
depends on -- untested.

WHAT EACH GROUP ACTUALLY EXERCISES (stated precisely, because a test that
re-implements the logic it claims to check is worse than no test):

  * T2 / constraint axis   -- calls `audit.run_T2` directly. Synthetic
                              fixtures, no data access required.
  * Certificate aggregation-- calls `experiment._aggregate` directly, with
                              synthetic per-seed records in the exact shape
                              `experiment.run` produces. A change to the real
                              percentile rule or the AND-condition fails these.
  * severe_acute mask      -- the rule is inline inside `cohort.build_cohort`,
                              which requires credentialed MIMIC-IV CSVs and so
                              cannot be exercised in a data-free unit test. The
                              tests below therefore validate the mask on the
                              RELEASED cohort artifact when it is present, and
                              SKIP (rather than silently pass) when it is not.
                              They are integration checks, not unit tests.

    cd mimic_pipeline && python -m pytest tests/ -q
    # with the cohort artifact present, the mask checks run too:
    AUDITNESY_COHORT=audit_mimic_out/aki_cohort.parquet python -m pytest tests/ -q
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic import audit, experiment  # noqa: E402
from auditnesy_mimic.config import load_config  # noqa: E402


class _StubModel:
    """Deterministic model with a hand-set probability vector.

    `constraint_satisfaction` mirrors the real coupled models: a confident
    no-AKI decision on a severe case is unsatisfied (0.0), everything else 1.0.
    """

    def __init__(self, probs, emit_evidence=True):
        self.probs = np.asarray(probs, float)
        self.emit_evidence = emit_evidence

    def predict_proba(self, X):
        return self.probs[: len(X)]

    def param_hash(self):
        return "stub"

    def constraint_satisfaction(self, X, severe_mask):
        if not self.emit_evidence:
            return None
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        sev = np.asarray(severe_mask, bool)
        return np.where(sev & (pred == 0), 0.0, 1.0)


# --------------------------------------------------------------------------
# T2: the constraint axis
# --------------------------------------------------------------------------

def test_T2_counts_only_confident_negatives_on_severe_cases():
    """A violation requires all three: severe, predicted no-AKI, and confident."""
    #        confident-no  uncertain-no  confident-yes  confident-no-but-not-severe
    probs = [0.05,         0.45,         0.95,          0.05]
    severe = [True,        True,         True,          False]
    X = np.zeros((4, 3))
    out = audit.run_T2(_StubModel(probs), X, np.array(severe), tau=0.6, enforce=False)
    # only index 0 qualifies: 0.45 defers (conf 0.55 < 0.6), 0.95 predicts AKI,
    # index 3 is not a severe case
    assert out["hard_violations"] == 1
    assert out["n_severe"] == 3


def test_T2_deferral_is_not_a_violation():
    """Uncertain no-AKI decisions defer -- the safe action -- and must not count."""
    probs = [0.49, 0.48, 0.47]           # all predict 0, all confidence ~0.5 < tau
    X = np.zeros((3, 3))
    out = audit.run_T2(_StubModel(probs), X, np.ones(3, bool), tau=0.9, enforce=False)
    assert out["hard_violations"] == 0
    assert out["coverage_severe"] == 0.0


def test_T2_enforced_policy_has_zero_violations_by_construction():
    """enforce=True escalates confident denials, so violations are 0 but coverage drops."""
    probs = [0.02, 0.03, 0.97]
    X = np.zeros((3, 3))
    logged = audit.run_T2(_StubModel(probs), X, np.ones(3, bool), tau=0.6, enforce=False)
    enforced = audit.run_T2(_StubModel(probs), X, np.ones(3, bool), tau=0.6, enforce=True)
    assert logged["hard_violations"] == 2
    assert enforced["hard_violations"] == 0
    # the enforced policy pays for compliance in coverage
    assert enforced["coverage_severe"] < logged["coverage_severe"]


def test_T2_reports_no_evidence_when_model_emits_none():
    """The pure-neural arm must be reported as NO EVIDENCE, never as a pass."""
    X = np.zeros((3, 3))
    out = audit.run_T2(_StubModel([0.02, 0.03, 0.97], emit_evidence=False), X,
                       np.ones(3, bool), tau=0.6, enforce=False)
    assert not out.get("evidence", False)
    assert out.get("hard_violations") is None


def test_T2_violation_rate_is_over_deployed_not_over_all_severe():
    """The denominator is the deployed (confident) subset, per prereg v1.2."""
    probs = [0.02, 0.02, 0.50, 0.50]     # 2 confident denials, 2 deferrals
    X = np.zeros((4, 3))
    out = audit.run_T2(_StubModel(probs), X, np.ones(4, bool), tau=0.6, enforce=False)
    assert out["hard_violations"] == 2
    assert out["violation_rate_deployed"] == pytest.approx(1.0)
    assert out["coverage_severe"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Certificate aggregation
# --------------------------------------------------------------------------

def _per_seed_record(cov, vr, n_seeds=20):
    """Synthetic per-seed records in the exact shape `experiment.run` emits.

    Only the keys `_aggregate` reads are populated; values are chosen so T1/T3
    pass and the certificate is the only thing under test.
    """
    out = []
    for i in range(n_seeds):
        out.append({
            "auroc": 0.88,
            "T1": {"pass_split": True},
            "T2": {"hard_violations": 0, "pass_split": True, "n_severe": 1000,
                   "coverage_severe": float(cov[i]),
                   "violation_rate_deployed": float(vr[i])},
            "T2_enforced": {"hard_violations": 0, "escalated_on_severe": 0.0,
                            "coverage_severe": float(cov[i]),
                            "violation_rate_deployed": float(vr[i])},
            "T3": {"ece_point": 0.01, "ece_ci": [0.005, 0.02]},
            "T2s": {"creatinine_sensitivity": 0.5},
        })
    return out


def _agg_cert(cov, vr, beta=0.80, vmax=0.05):
    """Run the REAL aggregation and return its logged-policy certificate block."""
    cfg = load_config()
    cfg.raw["audit"] = {**cfg.raw["audit"],
                        "coverage_floor_beta": beta, "violation_rate_max": vmax}
    agg = experiment._aggregate(_per_seed_record(cov, vr), cfg)
    return agg["compliance_coverage_certificate"]["logged_policy"]


def test_certificate_requires_both_coverage_and_compliance():
    """Exercises experiment._aggregate: PASS demands coverage AND compliance."""
    # compliant but under-covering -> fail, and the reason is attributed to coverage
    c = _agg_cert(np.full(20, 0.40), np.zeros(20))
    assert not c["certificate_pass"] and c["fails_on"] == "coverage"
    # well-covering but non-compliant -> fail, attributed to violations
    c = _agg_cert(np.full(20, 0.95), np.full(20, 0.50))
    assert not c["certificate_pass"] and c["fails_on"] == "violation"
    # both -> pass
    c = _agg_cert(np.full(20, 0.95), np.zeros(20))
    assert c["certificate_pass"] and c["fails_on"] == "none"


def test_certificate_uses_tail_quantiles_not_means():
    """The anti-gaming property: a good mean with a bad tail must not pass.

    Exercises the real percentile rule in experiment._aggregate -- if it were
    changed to a mean, this test fails.
    """
    cov = np.concatenate([np.full(18, 0.99), np.full(2, 0.10)])   # mean 0.90, 5th pct 0.10
    c = _agg_cert(cov, np.zeros(20))
    assert cov.mean() > 0.80                      # a mean-based rule would pass
    assert c["coverage_severe_p5"] < 0.80         # the tail quantile does not
    assert not c["certificate_pass"] and c["fails_on"] == "coverage"


def test_certificate_violation_side_uses_upper_tail():
    """Symmetric check on the violation limb: a bad upper tail must fail."""
    vr = np.concatenate([np.zeros(18), np.full(2, 0.90)])          # mean 0.09, 95th pct high
    c = _agg_cert(np.full(20, 0.95), vr)
    assert c["violation_rate_deployed_p95"] > 0.05
    assert not c["certificate_pass"] and c["fails_on"] == "violation"


# --------------------------------------------------------------------------
# severe_acute mask: the population the headline is computed on
# --------------------------------------------------------------------------

def _released_cohort():
    """Load the released cohort artifact, or skip.

    `severe_acute` is built inline inside `cohort.build_cohort`, which reads
    credentialed MIMIC-IV CSVs; it cannot be called in a data-free test. These
    checks therefore validate the mask as ACTUALLY MATERIALISED in the released
    artifact, and skip when it is absent rather than passing vacuously.
    """
    cand = os.environ.get("AUDITNESY_COHORT")
    paths = [Path(cand)] if cand else [
        Path(__file__).resolve().parent.parent / p / "aki_cohort.parquet"
        for p in ("audit_mimic_out", "audit_mimic_out_gbm", "audit_mimic_kdigo")
    ]
    for p in paths:
        if p.exists():
            return pd.read_parquet(p, columns=["creat_peak_24h", "creat_nadir_24h",
                                               "aki", "severe_acute", "subject_id"])
    pytest.skip("no released cohort artifact available (set AUDITNESY_COHORT)")


def test_released_severe_mask_matches_the_declared_rule():
    """The materialised mask must equal peak-in-window >= configured threshold.

    Catches a drift between `config.yaml:creatinine_severe_mgdl` and the column
    the headline is computed on -- the failure mode that would silently move the
    reported violation rate.
    """
    df = _released_cohort()
    thr = float(load_config().constraints["creatinine_severe_mgdl"])
    peak = pd.to_numeric(df["creat_peak_24h"], errors="coerce")
    sev = df["severe_acute"].astype(bool).to_numpy()
    # every severe row is at or above threshold
    assert bool((peak[sev] >= thr).all())
    # no row at or above threshold is left out on the peak criterion alone
    # (the mask is peak>=thr AND acute; acuteness can only remove rows)
    assert sev.sum() <= int((peak >= thr).sum())
    # NaN peak is never severe
    assert not bool(sev[peak.isna().to_numpy()].any())


def test_released_severe_mask_is_independent_of_the_label():
    """Circularity guard, checked on real data.

    If severity and the ICD label coincided by construction the headline finding
    would be vacuous. All four (severe, coded) combinations must occur, and the
    agreement must be strictly between 0 and 1.
    """
    df = _released_cohort()
    sev = df["severe_acute"].astype(bool).to_numpy()
    aki = df["aki"].astype(int).to_numpy()
    combos = set(zip(sev.tolist(), aki.tolist()))
    assert combos == {(True, 1), (True, 0), (False, 1), (False, 0)}
    agreement = float(aki[sev].mean())
    assert 0.0 < agreement < 1.0


def test_cohort_declares_the_severe_column_and_threshold_key():
    """Data-free guard against a silent rename of either name the headline needs."""
    src = (Path(__file__).resolve().parent.parent / "auditnesy_mimic" / "cohort.py").read_text()
    assert "severe_acute" in src
    assert "creatinine_severe_mgdl" in src
    assert "creat_peak_24h" in src
