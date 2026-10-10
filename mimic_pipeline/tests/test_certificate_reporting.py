"""The certificate is decided on tails; anything reporting it must say which statistic.

`experiment.py` decides the certificate as `cov_p5 >= beta and vr_p95 <= vmax` --
5th-percentile coverage and 95th-percentile violation rate, never the means. A
table captioned "conservative across-seed bounds" that pairs the p5 coverage
(0.78) with the MEAN violation rate (0.71) reads as one statistic and is two,
and the number a reviewer would check against the cap (0.74) is the one missing.

These tests keep every certificate figure in the paper traceable to a named
statistic in the result file, and are mutation-verified: swapping a tail figure
for its mean, or dropping the statistic label, fails the corresponding test.

Run: cd mimic_pipeline && python -m pytest tests/test_certificate_reporting.py -q
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "audit_summary_mimic_final_v1.3.json"
PAPER = ROOT.parent / "AuditNeSy_v26.tex"
EXPERIMENT = ROOT / "auditnesy_mimic" / "experiment.py"


def _cc():
    if not SUMMARY.exists():
        pytest.skip("audit summary not in this checkout")
    return json.loads(SUMMARY.read_text())["aggregate"]["compliance_coverage_certificate"]


def _paper():
    if not PAPER.exists():
        pytest.skip("paper source not in this checkout")
    return re.sub(r"\s+", " ", PAPER.read_text(errors="ignore"))


def test_certificate_is_decided_on_tails_not_means():
    """Guard the premise: if the rule ever moved to means, these tests are wrong."""
    if not EXPERIMENT.exists():
        pytest.skip("experiment.py not in this checkout")
    src = EXPERIMENT.read_text(errors="ignore")
    i = src.find('"certificate_pass"')
    assert i != -1, "certificate_pass disappeared from experiment.py"
    line = src[i:src.find("\n", i)]
    assert "cov_p5" in line and "vr_p95" in line, (
        f"the certificate must be decided on the conservative tails: {line.strip()}")
    for banned in ("cov_mean", "vr_mean"):
        assert banned not in line, f"certificate decided on a mean: {line.strip()}"


def test_table_reports_both_tail_and_mean_for_each_quantity():
    """The table must not silently mix a percentile column with a mean column."""
    s = _paper()
    # `{tab:cc}` also appears in body cross-references; anchor on the float itself
    i = s.find("{tab:cc} {\\caption{Compliance")
    assert i != -1, "the certificate table disappeared"
    block = s[i:i + 1400]
    # `"5th pct" in "95th pct"` is True, so a plain substring test for the
    # coverage label passes on the violation label alone. Bound the match.
    assert re.search(r"(?<!9)5th pct", block), (
        "the coverage tail column must be labelled as the 5th percentile")
    assert "95th pct" in block, (
        "the violation tail column must be labelled as the 95th percentile")
    assert block.count("mean") >= 1, "the means must be labelled where they appear"
    cc = _cc()
    L = cc["logged_policy"]
    for v in (L["coverage_severe_p5"], L["coverage_severe_mean"],
              L["violation_rate_deployed_p95"], L["violation_rate_deployed_mean"]):
        assert f"{v:.2f}" in block, f"{v:.2f} missing from the certificate table"
    # a two-column-per-quantity layout is what makes tail and mean separable at
    # all; a single unlabelled column reads as one statistic and is two
    assert "multicolumn{2}{c}{\\textbf{Coverage}}" in block, (
        "coverage must span a tail column and a mean column, not one ambiguous column")
    assert re.search(r"begin\{tabular\}\{@\{\}l?c{5}", block), (
        "the table must carry five numeric columns (2 coverage, 2 violation, verdict); "
        "collapsing them reintroduces the tail/mean ambiguity")


def test_the_number_the_cap_is_tested_against_is_reported():
    """0.74 (p95) is the figure that fails the 0.05 cap; 0.71 (mean) is not."""
    s = _paper()
    cc = _cc()
    p95 = cc["logged_policy"]["violation_rate_deployed_p95"]
    assert f"{100*p95:.0f}\\%" in s or f"{p95:.4f}" in s or f"{p95:.2f}" in s, (
        f"the violation rate actually tested against the cap ({p95}) is never reported")


def test_body_prose_labels_which_statistic_each_figure_is():
    """A bare percentage in the certificate sentence is ambiguous between two values."""
    s = _paper()
    i = s.find("neither policy passes")
    assert i != -1, "the certificate sentence disappeared from the body"
    window = s[i:i + 900]
    assert "percentile" in window, (
        f"the certificate sentence must name the statistic it quotes: {window[:300]}")
    assert "mean" in window, "the sentence quotes a mean; it must say so"


def test_reported_figures_match_the_result_file():
    """Every certificate figure in the paper must round-trip to the source."""
    s = _paper()
    cc = _cc()
    for policy in ("logged_policy", "enforced_policy"):
        p = cc[policy]
        for key in ("coverage_severe_p5", "coverage_severe_mean",
                    "violation_rate_deployed_p95", "violation_rate_deployed_mean"):
            v = p[key]
            assert f"{v:.2f}" in s, f"{policy}.{key} = {v:.2f} not found in the paper"
