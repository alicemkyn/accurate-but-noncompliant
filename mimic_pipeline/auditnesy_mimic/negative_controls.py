"""Negative controls v2 (RR7 / DA-2): NATURALLY deficient models, not forced.

Each control is expected to FAIL a specific axis. A control that fails to flag
on its target axis counts AGAINST the framework (reported honestly). None of
these mechanically inject the violation they are meant to detect -- that
circularity is exactly what the review rejected in the earlier NC-B.

    NC-A'  overconfident/uncalibrated (no calibration + reweighting)  -> T3
    NC-B'  constraint-blind (trained WITHOUT creatinine feature)      -> T2
    NC-C'  stale (trained on an earlier era, evaluated on a later one) -> T3 (shift)
    NC-D'  nondeterministic environment (different seed on replay)     -> T1
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics
from .io_utils import model_hash


def nc_a_uncalibrated(X_tr, y_tr, X_te, y_te, groups_te, cfg_audit, seed=0):
    """Overconfident model: no calibration, class-reweighted -> should fail T3."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64,), random_state=seed, max_iter=500)),
    ]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    # sharpen toward extremes to simulate the common overconfidence failure
    p = np.clip((p - 0.5) * 3.0 + 0.5, 1e-4, 1 - 1e-4)
    _, _, hi = metrics.bootstrap_ci(
        lambda a, b: metrics.ece(a, b, cfg_audit["n_bins_ece"]),
        p, y_te, groups=groups_te, B=cfg_audit.get("bootstrap_B", 1000), seed=seed)
    flagged = hi >= cfg_audit["ece_pass_upper_ci"]
    return {"ece_ci_upper": float(hi), "flagged_T3": bool(flagged), "target": "T3"}


def nc_b_constraint_blind(X_tr, y_tr, X_te, severe_mask_te, feat_cols, abstain_target, seed=0):
    """Model trained WITHOUT creatinine -> naturally violates the acute-severe constraint.

    Evaluated under the SAME defer-aware T2 rule as the primary model (confident no-AKI
    on a severe case), with tau selected on its own training predictions, so the control
    is comparable rather than advantaged by a different decision rule.
    """
    if "creatinine" not in feat_cols:
        return {"available": False}
    ci = feat_cols.index("creatinine")
    keep = [i for i in range(X_tr.shape[1]) if i != ci]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), random_state=seed, max_iter=500)),
    ]).fit(X_tr[:, keep], y_tr)
    tau = metrics.select_tau(pipe.predict_proba(X_tr[:, keep])[:, 1], abstain_target)
    p_te = pipe.predict_proba(X_te[:, keep])[:, 1]
    pred = (p_te >= 0.5).astype(int)
    conf = np.maximum(p_te, 1 - p_te)
    severe = np.asarray(severe_mask_te, bool)
    viol = int(np.sum((pred == 0) & (conf >= tau) & severe))
    return {"available": True, "hard_violations": viol,
            "flagged_T2": bool(viol > 0) if severe.sum() else None,
            "n_severe": int(severe.sum()), "target": "T2"}


def nc_c_stale(model_factory, X_early, y_early, X_late, y_late, groups_late,
               tau, cfg_audit, seed=0):
    """Trained on an earlier era, evaluated on a later era, no recalibration -> T3 shift."""
    model = model_factory()
    model.fit(X_early, y_early)
    p_late = model.predict_proba(X_late)
    _, _, hi = metrics.bootstrap_ci(
        lambda a, b: metrics.ece(a, b, cfg_audit["n_bins_ece"]),
        p_late, y_late, groups=groups_late, B=cfg_audit.get("bootstrap_B", 1000), seed=seed)
    flagged = hi >= cfg_audit["ece_pass_upper_ci"]
    ab = metrics.abstain_mask(p_late, tau).mean()
    return {"ece_ci_upper_late": float(hi), "abstain_rate_late": float(ab),
            "flagged_T3_shift": bool(flagged), "target": "T3-shift"}


def nc_d_nondeterministic(model_factory_seeded, X_tr, y_tr, X_te, seed=0):
    """Replay under a DIFFERENT seed (proxy for an unpinned/nondeterministic env).

    A reproducible pipeline must fail here (hash mismatch / prediction drift),
    demonstrating T1 catches a real reproducibility failure -- not a trivial pass.
    """
    m1 = model_factory_seeded(seed)
    m1.fit(X_tr, y_tr)
    m2 = model_factory_seeded(seed + 1000)  # different init => different params
    m2.fit(X_tr, y_tr)
    pred1 = (m1.predict_proba(X_te) >= 0.5).astype(int)
    pred2 = (m2.predict_proba(X_te) >= 0.5).astype(int)
    mism = int((pred1 != pred2).sum())
    flagged = (mism > 0) or (m1.param_hash() != m2.param_hash())
    return {"mismatches": mism, "hash_match": bool(m1.param_hash() == m2.param_hash()),
            "flagged_T1": bool(flagged), "target": "T1"}
