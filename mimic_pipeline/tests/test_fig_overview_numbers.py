"""Every number in Figure 1 must be parsed from a committed file, not typed in.

The script's docstring once claimed this while two values -- the hospital count
and the number of deviations-log rows -- were literals in the source. The claim
was the defect: a figure that asserts provenance it does not have is the same
failure mode the results-self-consistency guards exist to catch, one level up.

These tests are mutation-verified (see figures/README or the project README):
reintroducing either literal, or breaking either reader's source file, makes the
correspondingly named test fail.

Run: cd mimic_pipeline && python -m pytest tests/test_fig_overview_numbers.py -q
"""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "figures" / "make_fig_overview.py"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="figure script not in this checkout")


def _module():
    spec = importlib.util.spec_from_file_location("fig_overview", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _src() -> str:
    return SCRIPT.read_text(errors="ignore")


def test_hospital_count_is_derived_not_written():
    """The panel string must interpolate the reader, never a literal count."""
    src = _src()
    line = next((l for l in src.splitlines() if "hospitals" in l and "f\"" in l), None)
    assert line is not None, "no f-string panel line mentioning hospitals"
    assert "{n_hosp_eicu}" in line, f"hospital count must be interpolated, got: {line.strip()}"
    assert not re.search(r"\b20[0-9]\s+hospitals", line), (
        f"a literal hospital count is present: {line.strip()}")


def test_deviations_count_is_derived_not_written():
    src = _src()
    idx = src.find("Deviations log")
    assert idx != -1, "the deviations box disappeared from the figure"
    window = src[idx:idx + 220]
    assert "{n_dev}" in window, f"deviations count must be interpolated: {window!r}"
    assert not re.search(r"\b1[0-9] numbered entries", window), (
        f"a literal deviations count is present: {window!r}")


def test_hospital_reader_agrees_with_its_source_file():
    """The reader must return exactly what EICU_FINDINGS.md states, unmerged."""
    mod = _module()
    findings = ROOT / "eicu_pipeline" / "EICU_FINDINGS.md"
    if not findings.exists():
        pytest.skip("EICU_FINDINGS.md not in this checkout")
    m = re.search(r"\*\*(\d[\d,]*)\s+hospitals\*\*", findings.read_text(errors="ignore"))
    assert m, "the source file no longer states a bolded hospital count"
    assert mod.n_eicu_hospitals() == int(m.group(1).replace(",", ""))


def test_deviations_reader_counts_the_paper_tables():
    """The count must equal the distinct row numbers in the two log tables."""
    mod = _module()
    tex = ROOT / "AuditNeSy_v26.tex"
    if not tex.exists():
        pytest.skip("paper source not in this checkout")
    s = tex.read_text(errors="ignore")
    rows = set()
    for label in ("{tab:deviations}", "{tab:deviations2}"):
        k = s.find(label + "\n")
        start = s.rfind(r"\begin{table*}", 0, k)
        end = s.find(r"\end{table*}", k)
        rows |= {int(x) for x in re.findall(r"\n\s*(\d+)\s*&", s[start:end])}
    assert mod.n_deviation_rows() == len(rows) > 0


def test_readers_fail_loudly_rather_than_guessing():
    """Both readers raise when their source is unparseable -- no silent fallback."""
    src = _src()
    for fn in ("n_eicu_hospitals", "n_deviation_rows"):
        i = src.find(f"def {fn}(")
        assert i != -1, f"{fn} disappeared"
        body = src[i:src.find("\ndef ", i + 1)]
        assert "raise RuntimeError" in body, f"{fn} must raise, not default"
        assert not re.search(r"return\s+\d+\s*$", body, re.M), (
            f"{fn} has a hardcoded return value")


def test_deviations_captions_match_their_own_rows():
    """A log that claims completeness must not misstate its own row range.

    The part-2 caption read "rows 9--15" while the table carried rows 9--17: two
    deviations existed but were not announced by the table that contains them.
    In a paper whose argument is that declared deviations make an audit
    falsifiable, an undercounting caption is the worst possible place for an
    off-by-two.
    """
    tex = ROOT / "AuditNeSy_v26.tex"
    if not tex.exists():
        pytest.skip("paper source not in this checkout")
    s = tex.read_text(errors="ignore")
    for label in ("{tab:deviations}", "{tab:deviations2}"):
        k = s.find(label + "\n")
        assert k != -1, f"{label} missing"
        start = s.rfind(r"\begin{table*}", 0, k)
        end = s.find(r"\end{table*}", k)
        block = s[start:end]
        rows = sorted(int(x) for x in re.findall(r"\n\s*(\d+)\s*&", block))
        claimed = re.search(r"rows (\d+)--(\d+)", block)
        assert claimed, f"{label} caption states no row range"
        lo, hi = int(claimed.group(1)), int(claimed.group(2))
        assert (lo, hi) == (rows[0], rows[-1]), (
            f"{label} caption claims rows {lo}--{hi} but the table carries "
            f"{rows[0]}--{rows[-1]}")


def test_cohort_sizes_are_not_attributed_to_the_wrong_database():
    """MIMIC's admission count must never be paired with the multi-site hospital count.

    MIMIC-IV is one hospital; the 205 others are eICU, a separate cohort with its
    own denominator. An earlier figure panel read "379k admissions, 167k patients,
    206 hospitals", which asserts a 206-site MIMIC cohort that does not exist.
    """
    src = _src()
    for line in src.splitlines():
        if "hospital" in line and ("n_adm" in line or "n_pat" in line):
            assert "one hospital" in line, (
                f"MIMIC cohort size and a multi-site hospital count share a clause: {line.strip()}")


def test_figure_numbers_match_the_paper_body():
    """The figure and the prose must quote the same values (rule 1.7: one number per claim)."""
    mod = _module()
    tex = ROOT / "AuditNeSy_v26.tex"
    if not tex.exists():
        pytest.skip("paper source not in this checkout")
    s = re.sub(r"\s+", " ", tex.read_text(errors="ignore"))
    assert f"{mod.n_eicu_hospitals()} hospitals" in s, (
        "the figure's eICU hospital count does not appear in the paper body")
