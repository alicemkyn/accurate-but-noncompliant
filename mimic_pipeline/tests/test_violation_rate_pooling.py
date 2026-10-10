"""Guards on the model-side deletion test (`12_violation_rate_by_renal_history.py`).

Two real defects produced wrong answers here before these tests existed, and each
has a test:

  1. per-stratum violation rates were pooled by `n_severe`, but
     `violation_rate_deployed` has `n_deployed` as its denominator, so the correct
     weight is `coverage_severe * n_severe`;
  2. the pooling code read a flat `"<key>_mean"` while `agg_across_seeds` nests
     each statistic as `{"mean","p5","p95"}`, which yields `None` silently and
     wrote a payload of nulls that still looked like a result.

Plus the reporting rule the project applies to itself: both surviving strata sit
below the registered subgroup floor, so they must be reported AND flagged, never
one without the other.

Run: cd mimic_pipeline && python -m pytest tests/test_violation_rate_pooling.py -q
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "results" / "violation_rate_by_renal_history.json"
SCRIPT = ROOT / "scripts" / "12_violation_rate_by_renal_history.py"
CONFIG = ROOT / "config.yaml"
PAPER = ROOT.parent / "AuditNeSy_v26.tex"

SURVIVING = ("no_prior_renal_replacement", "prior_status_unobservable")


def _payload():
    if not PAYLOAD.exists():
        pytest.skip("violation_rate_by_renal_history.json not in this checkout")
    return json.loads(PAYLOAD.read_text())


def _src():
    if not SCRIPT.exists():
        pytest.skip("script not in this checkout")
    return SCRIPT.read_text(errors="ignore")


def test_pooling_uses_deployed_decisions_as_the_weight():
    """Recompute the pooled rate from the per-stratum cells and demand a match."""
    d = _payload()
    h = d["headline_comparison"]
    num = den = 0.0
    for name in SURVIVING:
        c = d["per_stratum"][name]
        vr = c["violation_rate_deployed"]["mean"]
        n_dep = c["n_severe_mean"] * c["coverage_severe"]["mean"]
        num += vr * n_dep
        den += n_dep
    assert den > 0, "no surviving stratum had deployed decisions"
    reported = h["violation_rate_surviving_cases"]
    # tolerance must be TIGHTER than the gap between the two candidate weights,
    # or substituting the wrong one passes silently -- which it did before this
    # line was tightened. The n_severe-weighted alternative is recorded in the
    # payload, so the discriminating tolerance is available rather than guessed.
    num_sev = den_sev = 0.0
    for name in SURVIVING:
        c = d["per_stratum"][name]
        num_sev += c["violation_rate_deployed"]["mean"] * c["n_severe_mean"]
        den_sev += c["n_severe_mean"]
    wrong = num_sev / den_sev
    correct = num / den
    tol = max(abs(correct - wrong) / 4, 1e-6)
    assert abs(reported - correct) < tol, (
        f"pooled rate {reported} does not equal the n_deployed-weighted mean "
        f"{correct:.6f} (n_severe-weighted would be {wrong:.6f})")
    assert abs(reported - wrong) >= tol or abs(correct - wrong) < 1e-9, (
        f"reported rate {reported} matches the n_severe weight {wrong:.6f}, "
        "not the n_deployed weight")


def test_the_wrong_weight_is_recorded_not_hidden():
    """The n_severe-weighted figure stays in the payload so the error is on record."""
    h = _payload()["headline_comparison"]
    assert h.get("same_pooling_with_n_severe_weight") is not None
    assert h.get("weight_choice_matters_by") is not None
    assert "n_deployed" in h["pooling_weight"], (
        "the payload must state which weight is used and why")


def test_coverage_differs_so_the_weight_choice_is_not_academic():
    """If coverage were equal across strata the weight would not matter; it is not."""
    cov = _payload()["headline_comparison"]["coverage_severe_mean_by_stratum"]
    vals = [v for v in cov.values() if v is not None]
    assert len(vals) == 3, f"expected three strata, got {cov}"
    assert max(vals) - min(vals) > 0.01, (
        "coverage is near-identical across strata; the pooling-weight justification "
        f"in the payload would then be misleading: {cov}")


def test_nested_statistics_are_read_through_a_helper():
    """A flat '<key>_mean' read on agg_across_seeds output silently yields None."""
    src = _src()
    for bad in ('"violation_rate_deployed_mean"', '"coverage_severe_mean"'):
        assert bad not in src, (
            f"{bad} is not a key agg_across_seeds produces; it nests per-key stats")
    assert "def _mean(" in src, "the nested-statistic accessor disappeared"


def test_script_refuses_to_write_an_unpopulated_comparison():
    """An all-null payload is a failure, not a result."""
    src = _src()
    assert "unpopulated" in src, "the emptiness guard disappeared from the script"
    i = src.find("missing = [k for k in")
    assert i != -1, "the missing-key check disappeared"
    window = src[i:i + 600]
    assert "raise SystemExit" in window, "the script must refuse, not warn"


def test_underpowered_strata_are_reported_and_flagged():
    """Our own subgroup floor applies to our own defence."""
    d = _payload()
    floor = 500
    if CONFIG.exists():
        m = re.search(r"min_subgroup_n:\s*(\d+)", CONFIG.read_text(errors="ignore"))
        if m:
            floor = int(m.group(1))
    for name in SURVIVING:
        c = d["per_stratum"][name]
        if c["n_severe_mean"] < floor * 1.2:
            assert c["status"] == "UNDETERMINED", (
                f"{name} is near/below the registered floor of {floor} but is not flagged")
            assert c["violation_rate_deployed"] is not None, (
                f"{name} is flagged but its figure was deleted; report and flag, not one")


def test_paper_states_the_underpowering_where_it_states_the_result():
    """The flag must travel with the number, not sit in a distant appendix line."""
    if not PAPER.exists():
        pytest.skip("paper source not in this checkout")
    s = re.sub(r"\s+", " ", PAPER.read_text(errors="ignore"))
    # the figure is set in math mode, so match on the prose that follows it
    i = s.find("pooled over the two surviving")
    assert i != -1, "the model-side deletion result disappeared from the paper"
    d = _payload()["headline_comparison"]
    assert f"{d['violation_rate_surviving_cases']:.2f}" in s[max(0, i - 60):i], (
        "the pooled figure quoted in the paper does not match the result file")
    window = s[i:i + 1500]
    assert "undetermined" in window.lower(), (
        "the paragraph reporting the pooled rate must also carry the UNDETERMINED flag")
    assert "n_{\\text{deployed}}" in window or "deployed decisions" in window, (
        "the paragraph must say which denominator the pooling uses")
