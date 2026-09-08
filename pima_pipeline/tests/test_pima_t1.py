"""Non-vacuity tests for the T1 (replay reproducibility) check on both Pima arms.

Motivation: an earlier version of `11_pima_styles.py` reported
`"T1_hash_match": true` for the differentiable arm without ever comparing two
hashes -- a verdict asserted rather than measured. These tests exist so that a
regression to an unearned pass fails loudly.

The important property is NOT that the hash matches on a rerun (a function that
returns a constant would satisfy that). It is that the hash DISCRIMINATES: it
must match for an identical refit and differ when the fit changes. Both
directions are asserted below.

Run from the pipeline root:

    cd pima_pipeline && python -m pytest tests/ -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent / "mimic_pipeline"))

_spec = importlib.util.spec_from_file_location("pima_audit", _ROOT / "scripts" / "10_pima_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

DATA = _ROOT / "data" / "pima_diabetes.csv"


@pytest.fixture(scope="module")
def split():
    if not DATA.exists():  # pragma: no cover
        pytest.skip(f"dataset not present at {DATA}")
    df = pa.load_raw(DATA)
    X, y, tr, te, filled = pa.split_and_impute(df, pa.REFERENCE_SEED)
    return X, y, tr, te, filled


def test_discrete_arm_hash_matches_on_identical_refit(split):
    X, y, tr, _, _ = split
    h1 = pa.param_hash(pa.fit_model(X, y, tr, pa.REFERENCE_SEED))
    h2 = pa.param_hash(pa.fit_model(X, y, tr, pa.REFERENCE_SEED))
    assert h1 == h2, "identical refit must reproduce the parameter hash"


def test_discrete_arm_hash_differs_on_different_fit(split):
    """Non-vacuity: the hash must CHANGE when the fit changes.

    Without this, a `param_hash` returning a constant would satisfy the test
    above and every T1 verdict built on it would be meaningless.
    """
    X, y, tr, _, _ = split
    h1 = pa.param_hash(pa.fit_model(X, y, tr, pa.REFERENCE_SEED))
    h2 = pa.param_hash(pa.fit_model(X, y, tr[: len(tr) // 2], pa.REFERENCE_SEED))
    assert h1 != h2, "a materially different fit must not share the parameter hash"


def test_ltn_arm_hash_is_measured_not_asserted(split):
    """The differentiable arm's T1 must discriminate in both directions."""
    pytest.importorskip("torch")
    from auditnesy_mimic.models import DifferentiableLTN

    X, y, tr, _, filled = split
    sev = (filled["Glucose"].to_numpy(float) >= pa.C3_GLUCOSE)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    def fit(seed, epochs=30):
        m = DifferentiableLTN(n_features=Xz.shape[1], sat_threshold=pa.SAT_THRESHOLD,
                              epochs=epochs, seed=seed)
        m.fit(Xz[tr], y[tr], sev[tr])
        return m.param_hash()

    same = fit(pa.REFERENCE_SEED)
    again = fit(pa.REFERENCE_SEED)
    other_seed = fit(pa.REFERENCE_SEED + 1)
    other_epochs = fit(pa.REFERENCE_SEED, epochs=31)

    assert same == again, "identical refit must reproduce the LTN parameter hash"
    assert same != other_seed, "a different seed must change the LTN parameter hash"
    assert same != other_epochs, "a different epoch count must change the LTN parameter hash"


def test_styles_script_does_not_hardcode_a_t1_pass():
    """Guard the exact regression: a literal boolean in the T1 payload."""
    src = (_ROOT / "scripts" / "11_pima_styles.py").read_text()
    i = src.find('"T1_hash_match"')
    assert i != -1, "T1_hash_match key disappeared from the styles script"
    while i != -1:
        line = src[i:src.find("\n", i)]
        assert "True" not in line and "False" not in line, (
            f"T1_hash_match must be a measured value, not a literal: {line.strip()}")
        i = src.find('"T1_hash_match"', i + 1)
