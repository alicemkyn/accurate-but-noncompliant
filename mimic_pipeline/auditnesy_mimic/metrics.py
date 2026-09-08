"""Calibration, abstention, and subject-level bootstrap CIs (RR2)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width, confidence = max(p, 1-p))."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5).astype(int) == labels).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(conf)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def abstain_mask(probs: np.ndarray, tau: float) -> np.ndarray:
    """Defer when max-class confidence falls below tau."""
    conf = np.maximum(probs, 1 - probs)
    return conf < tau


def select_tau(val_probs: np.ndarray, target_rate: float) -> float:
    """Pick the largest tau whose in-distribution deferral rate <= target_rate.

    Selected ONCE on validation, then frozen (recorded in preregistration_lock).
    """
    conf = np.maximum(val_probs, 1 - val_probs)
    # candidate taus = sorted unique confidences; choose the largest tau whose
    # deferral rate still respects the target (monotone: higher tau => more deferral)
    best = float(np.min(conf))
    for tau in np.unique(np.round(conf, 4)):
        if (conf < tau).mean() <= target_rate:
            best = float(tau)
        else:
            break
    return best


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    probs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) with a subject-level (grouped) bootstrap.

    Resampling by `groups` (subject_id) rather than by row keeps the CI honest
    when a patient contributes multiple admissions.
    """
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    point = metric_fn(probs, labels)

    if groups is None:
        idx = np.arange(len(probs))
        stats = np.empty(B)
        for b in range(B):
            s = rng.integers(0, len(idx), len(idx))
            stats[b] = metric_fn(probs[s], labels[s])
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        by_group = {g: np.where(groups == g)[0] for g in uniq}
        stats = np.empty(B)
        for b in range(B):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([by_group[g] for g in chosen])
            stats[b] = metric_fn(probs[sel], labels[sel])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi
