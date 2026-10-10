"""The deviations log must be contiguous: no version number may be skipped.

Two log entries (v1.10, v1.11) were written and then lost when a later edit was made
against a stale copy of the file and overwrote them. A paper whose argument rests on
a complete pre-registration log cannot have holes in that log. This test fails if a
version heading is missing from the sequence, and if any entry's number is cited in
a table but has no heading.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
LOG = ROOT / "preregistration_v14.md"


def _versions(text):
    heads = re.findall(r"^## v1\.(\d+)\b", text, re.M)
    return [int(x) for x in heads]


def test_log_exists():
    assert LOG.exists(), "preregistration_v14.md missing from repository root"


def test_version_headings_are_contiguous_and_increasing():
    v = _versions(LOG.read_text(encoding="utf-8", errors="ignore"))
    assert v, "no '## v1.N' headings found"
    assert v == sorted(v), f"version headings out of order: {v}"
    expected = list(range(v[0], v[-1] + 1))
    missing = sorted(set(expected) - set(v))
    assert not missing, f"version numbers skipped in the log: v1.{', v1.'.join(map(str, missing))}"


def test_every_version_cited_in_text_has_a_heading():
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    heads = set(_versions(text))
    cited = {int(x) for x in re.findall(r"\bv1\.(\d+)\b", text)}
    # versions below the first heading live in preregistration.md, not here
    cited = {c for c in cited if c >= min(heads)}
    dangling = sorted(cited - heads)
    assert not dangling, f"versions cited but without a heading: v1.{', v1.'.join(map(str, dangling))}"


def test_readme_version_range_matches_log():
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README not in this checkout")
    v = _versions(LOG.read_text(encoding="utf-8", errors="ignore"))
    r = readme.read_text(encoding="utf-8", errors="ignore")
    # only lines that describe THIS log; preregistration.md's own range (v1.0 … v1.3)
    # is a different document and is allowed to end where it ends
    lines = [ln for ln in r.splitlines() if "preregistration_v14" in ln]
    m = []
    for ln in lines:
        m += re.findall(r"v1\.(\d+)\s*[–\-…]+\s*v1\.(\d+)", ln)
    assert m, "README does not state preregistration_v14.md's version range"
    for lo, hi in m:
        assert int(hi) == v[-1], f"README says the log ends at v1.{hi}; it ends at v1.{v[-1]}"
