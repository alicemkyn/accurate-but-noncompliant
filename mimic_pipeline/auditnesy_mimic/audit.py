"""The three AuditNeSy axes + the concept-semantics probe, on a single split.

Each function returns a plain dict of measured quantities. Pass/fail decisions
that require CIs across seeds are made in experiment.py, not here -- these
functions report the raw evidence per split.
"""
from __future__ import annotations

import numpy as np

from . import metrics


def run_T1(model_factory, X_tr, y_tr, X_te, model):
    """Reproducibility: retrain identically, compare predictions + param hash.

    A genuine reproducibility failure is exercised by NC-D' (nondeterministic
    environment) in negative_controls.py -- so T1 passing here is not vacuous.
    """
    p1 = model.predict_proba(X_te)
    pred1 = (p1 >= 0.5).astype(int)
    h1 = model.param_hash()

    m2 = model_factory()
    m2.fit(X_tr, y_tr)
    pred2 = (m2.predict_proba(X_te) >= 0.5).astype(int)
    h2 = m2.param_hash()

    mism = int((pred1 != pred2).sum())
    return {
        "mismatches": mism,
        "n": int(len(pred1)),
        "hash_match": bool(h1 == h2),
        "pass_split": bool(mism == 0 and h1 == h2),
    }


def run_T2(model, X_te, severe_mask, tau, enforce=False):
    """Constraint compliance on the DEPLOYED, defer-aware action (prereg v1.2).

    `severe_mask` (bool) marks decision-time acute-severe AKI cases. A violation is a
    *confident* no-AKI decision on such a case: pred == 0 AND confidence >= tau. If the
    model is uncertain (confidence < tau) it defers -- the safe action -- which is NOT a
    violation, consistent with the T3 abstention-as-safety design.

    Two deployment policies of the same rule-augmented model (paper: hard constraints
    "halt or escalate"):
      * enforce=False (rules LOGGED): the confident-no-AKI decision is deployed as-is and
        counted as a violation.
      * enforce=True (rules ENFORCED): the hard constraint ESCALATES such a decision to a
        deferral, so it is never deployed -- 0 violations by construction; the count of
        escalated decisions is reported as the safety cost of enforcement.
    """
    sat = model.constraint_satisfaction(X_te, severe_mask)
    severe = np.asarray(severe_mask, bool)
    n_severe = int(severe.sum())
    if sat is None:  # pure-neural baseline: no symbolic evidence
        return {"evidence": False, "enforced": enforce, "n_severe": n_severe,
                "hard_violations": None, "pass_split": None}
    p = model.predict_proba(X_te)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    confident_neg = (pred == 0) & (conf >= tau) & severe
    # DEPLOYED severe decisions (coverage) and violations, per policy. This lets a
    # compliance-COVERAGE certificate close the abstention-gaming loophole: a policy
    # cannot "pass" by deferring the whole severe subgroup, because coverage is floored.
    if enforce:
        # escalate confident no-AKI to deferral; only confident-AKI is deployed
        deployed = severe & (conf >= tau) & (pred == 1)
        viol = 0
        escalated = int(confident_neg.sum())
    else:
        # deploy every confident decision (AKI or no-AKI); no-AKI ones are violations
        deployed = severe & (conf >= tau)
        viol = int(confident_neg.sum())
        escalated = 0
    n_deployed = int(deployed.sum())
    deferred = (conf < tau) & severe
    if enforce:
        deferred = deferred | (confident_neg)  # escalated cases are also deferred
    coverage = (n_deployed / n_severe) if n_severe else float("nan")
    viol_rate = (viol / n_deployed) if n_deployed else 0.0
    sat = np.asarray(sat, float)
    rule_of_three_upper = 3.0 / n_severe if (viol == 0 and n_severe > 0) else None
    return {
        "evidence": True,
        "enforced": enforce,
        "n_severe": n_severe,
        "hard_violations": viol,
        "escalated_on_severe": escalated,
        "deferred_on_severe": int(deferred.sum()),
        "n_deployed_severe": n_deployed,
        "coverage_severe": coverage,
        "violation_rate_deployed": viol_rate,
        "mean_satisfaction_severe": float(sat[severe].mean()) if n_severe else float("nan"),
        "rule_of_three_upper": rule_of_three_upper,
        "pass_split": bool(viol == 0),
    }


def run_T3(probs, labels, groups, tau, cfg_audit, seed=0):
    """Safe fallback (calibration component): ECE + bootstrap CI + abstention rate.

    Measured on the in-distribution random split. The distribution-SHIFT component
    of T3 is a distinct train-early / evaluate-late experiment (experiment.run_shift_eval),
    since the shift criterion is meaningless on an in-distribution random split.
    """
    n_bins = cfg_audit["n_bins_ece"]
    ece_pt, ece_lo, ece_hi = metrics.bootstrap_ci(
        lambda p, y: metrics.ece(p, y, n_bins), probs, labels,
        groups=groups, B=cfg_audit_B(cfg_audit), seed=seed)
    ab = metrics.abstain_mask(probs, tau)
    return {
        "ece_point": ece_pt,
        "ece_ci": [ece_lo, ece_hi],
        "tau": tau,
        "abstain_rate_overall": float(ab.mean()),
        "pass_split_pointwise": bool(ece_pt < cfg_audit["ece_pass_upper_ci"]),
    }


def cfg_audit_B(cfg_audit):
    # bootstrap B lives under experiment; audit dict may not carry it -> default
    return int(cfg_audit.get("bootstrap_B", 1000))


def run_T2s_probe(model_factory, X_tr, y_tr, X_te, feat_cols, cfg_audit):
    """Concept-semantics probe (advisory): does the model DEPEND on creatinine?

    Counterfactually sweeps the creatinine feature low->high and measures the
    mean change in predicted AKI risk. A genuinely constraint-dependent model
    has high sensitivity; a constraint-blind model near zero. This dissociates
    'passes T2 (compliance)' from 'decision depends on the constrained concept'
    -- the reasoning-shortcut screen T2 alone cannot provide.
    """
    if "creatinine" not in feat_cols:
        return {"available": False}
    ci = feat_cols.index("creatinine")
    lo_val, hi_val = cfg_audit["probe_sweep_creatinine"]
    n = min(cfg_audit["probe_n_samples"], len(X_te))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_te), size=n, replace=False)

    model = model_factory()
    model.fit(X_tr, y_tr)
    Xhi = X_te[idx].copy(); Xhi[:, ci] = hi_val
    Xlo = X_te[idx].copy(); Xlo[:, ci] = lo_val
    sens = float(np.mean(model.predict_proba(Xhi) - model.predict_proba(Xlo)))
    return {"available": True, "creatinine_sensitivity": sens}
