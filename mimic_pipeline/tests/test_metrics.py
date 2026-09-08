"""Unit tests for metrics that run WITHOUT MIMIC access (synthetic data).

    cd mimic_pipeline && python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic import metrics


def test_ece_perfect_calibration_is_low():
    rng = np.random.default_rng(0)
    # probs equal to true class probability -> well calibrated
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert metrics.ece(p, y, n_bins=10) < 0.05


def test_ece_overconfident_is_high():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.2, 0.8, 5000)
    y = (rng.uniform(0, 1, 5000) < true_p).astype(int)
    # push probabilities to extremes -> overconfident -> high ECE
    over = np.clip((true_p - 0.5) * 5 + 0.5, 1e-3, 1 - 1e-3)
    assert metrics.ece(over, y, n_bins=10) > metrics.ece(true_p, y, n_bins=10)


def test_abstain_mask_threshold():
    p = np.array([0.5, 0.6, 0.9, 0.05])
    # confidence = max(p,1-p) = [0.5, 0.6, 0.9, 0.95]; tau=0.65 -> defer first two
    m = metrics.abstain_mask(p, tau=0.65)
    assert m.tolist() == [True, True, False, False]


def test_select_tau_respects_target():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 2000)
    tau = metrics.select_tau(p, target_rate=0.15)
    conf = np.maximum(p, 1 - p)
    assert (conf < tau).mean() <= 0.15 + 1e-6


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    groups = rng.integers(0, 300, 1000)  # patients with multiple admissions
    pt, lo, hi = metrics.bootstrap_ci(
        lambda a, b: metrics.ece(a, b, 10), p, y, groups=groups, B=200, seed=0)
    assert lo <= pt <= hi
    assert hi - lo > 0
