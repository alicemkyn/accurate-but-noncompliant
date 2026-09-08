"""NeSy integration styles (RR/S2: three styles + pure-neural baseline).

Every model exposes a common interface so the audit harness stays model-agnostic:

    fit(X, y)                         -> self
    predict_proba(X)                  -> np.ndarray  (P(AKI))
    param_hash()                      -> str         (for T1)
    constraint_satisfaction(X, sev)   -> np.ndarray | None
        per-decision satisfaction score for the hard constraint, or None if the
        model emits no symbolic evidence (pure-neural baseline).

`sev` is the clinically-defined severe-creatinine value per row (kept OUT of X).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import model_hash


# ----------------------------------------------------------------------
# 1. Pipeline-coupled: neural core (MLP) + calibration + explicit rule layer
# ----------------------------------------------------------------------
class PipelineCoupled:
    """Neural probability estimate + a post-hoc, logged symbolic constraint layer.

    The rule layer does NOT silently overwrite predictions; violations are logged
    for T2 (matching the paper: hard constraints halt/escalate, they are audited).
    A calibration wrapper is used so T3 measures calibration rather than the
    probability distortion introduced by class reweighting.
    """

    def __init__(self, hidden_layers=(32, 16), calibrate="isotonic", seed=0):
        self.hidden_layers = tuple(hidden_layers)
        self.calibrate = calibrate
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=self.hidden_layers,
                                  random_state=self.seed, max_iter=500,
                                  early_stopping=True, n_iter_no_change=15)),
        ])
        if self.calibrate:
            self.model_ = CalibratedClassifierCV(base, method=self.calibrate, cv=3)
        else:
            self.model_ = base
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, 1]

    def param_hash(self):
        # hash the underlying MLP coefficients across calibrated folds
        arrs = []
        est = self.model_
        cals = getattr(est, "calibrated_classifiers_", None)
        if cals:
            for cc in cals:
                pipe = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                mlp = pipe.named_steps["mlp"]
                arrs += [np.concatenate([w.ravel() for w in mlp.coefs_]),
                         np.concatenate([b.ravel() for b in mlp.intercepts_])]
        else:
            mlp = est.named_steps["mlp"]
            arrs += [np.concatenate([w.ravel() for w in mlp.coefs_]),
                     np.concatenate([b.ravel() for b in mlp.intercepts_])]
        return model_hash(*arrs)

    def constraint_satisfaction(self, X, severe_mask):
        """Discrete evidence: 1.0 if the decision respects the hard constraint.

        Constraint C: ACUTE severe creatinine => must NOT predict "no AKI".
        `severe_mask` marks the cases where the constraint applies (acute rise to a
        severe level; stable CKD is excluded upstream in cohort.py). Satisfaction is
        1.0 unless (pred==0 and severe_mask), i.e. a discrete check.
        """
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        severe_mask = np.asarray(severe_mask, bool)
        return np.where(severe_mask & (pred == 0), 0.0, 1.0)


# ----------------------------------------------------------------------
# 1b. Pipeline-coupled with a STRONGER core: gradient-boosted trees.
# ----------------------------------------------------------------------
class GradientBoosted:
    """Same pipeline-coupled design with a gradient-boosting core (sklearn
    HistGradientBoostingClassifier -- no extra dependency), isotonic-calibrated.

    Purpose: test whether the accuracy-vs-guideline gap is an artifact of a weak
    neural core. A stronger model is expected to fit the (biased) ICD label even
    more faithfully, so it should exhibit the SAME or a LARGER gap -- closing the
    "you just used a weak model" objection. Identical interface to PipelineCoupled,
    so the entire audit/certificate/negative-control harness runs unchanged.
    """

    def __init__(self, calibrate="isotonic", seed=0, max_depth=None,
                 learning_rate=0.1, max_iter=300):
        self.calibrate = calibrate
        self.seed = seed
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model_ = None

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        base = Pipeline([
            ("scaler", StandardScaler()),  # harmless for trees; keeps interface uniform
            ("gbm", HistGradientBoostingClassifier(
                random_state=self.seed, learning_rate=self.learning_rate,
                max_depth=self.max_depth, max_iter=self.max_iter,
                early_stopping=True, n_iter_no_change=15)),
        ])
        self.model_ = (CalibratedClassifierCV(base, method=self.calibrate, cv=3)
                       if self.calibrate else base)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, 1]

    def param_hash(self):
        # HGB exposes no simple coefficient array; hash the deterministic pickled
        # state of the fitted estimator. Two identical fits (same seed, same data,
        # pinned versions) produce identical bytes -> T1 replay is meaningful.
        import hashlib
        import pickle
        return hashlib.sha256(pickle.dumps(self.model_)).hexdigest()[:16]

    def constraint_satisfaction(self, X, severe_mask):
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        severe_mask = np.asarray(severe_mask, bool)
        return np.where(severe_mask & (pred == 0), 0.0, 1.0)


# ----------------------------------------------------------------------
# 2. Differentiable (LTN-style): fuzzy satisfaction axiom trained jointly
# ----------------------------------------------------------------------
class DifferentiableLTN:
    """Minimal Logic Tensor Network-style model (requires torch).

    Trains an MLP with BCE + a fuzzy first-order axiom implemented as a
    satisfaction loss:  severe(creat) -> AKI  (product/Reichenbach implication).
    Emits a SOFT per-decision satisfaction score, illustrating the
    differentiability-verifiability trade-off (Table IV / Table V of the paper).
    """

    def __init__(self, n_features, sat_threshold=0.5, epochs=200, seed=0, hidden=32):
        self.n_features = n_features
        self.sat_threshold = sat_threshold
        self.epochs = epochs
        self.seed = seed
        self.hidden = hidden
        self.net_ = None
        self._torch = None

    def _lazy_torch(self):
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DifferentiableLTN requires PyTorch. Install with "
                "`pip install torch`, or disable models.differentiable_ltn in config.yaml."
            ) from e
        return torch

    def fit(self, X, y, severe_mask):
        torch = self._lazy_torch()
        self._torch = torch
        torch.manual_seed(self.seed)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        # antecedent degree: 1.0 where the acute-severe constraint applies
        severe_deg = torch.tensor(np.asarray(severe_mask, dtype=float), dtype=torch.float32)

        self.net_ = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=1e-2)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            logit = self.net_(Xt).squeeze(1)
            p = torch.sigmoid(logit)
            # Reichenbach implication  severe -> AKI :  1 - severe_deg + severe_deg * p
            sat = 1 - severe_deg + severe_deg * p
            axiom_loss = (1 - sat).mean()          # maximize satisfaction
            loss = bce(logit, yt) + 0.5 * axiom_loss
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        torch = self._torch
        with torch.no_grad():
            logit = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)
            return torch.sigmoid(logit).numpy()

    def param_hash(self):
        arrs = [p.detach().numpy().ravel() for p in self.net_.parameters()]
        return model_hash(*arrs)

    def constraint_satisfaction(self, X, severe_mask):
        """SOFT evidence: Reichenbach satisfaction of  severe_acute -> AKI  per decision.

        Antecedent is the (hard) applicability mask; the consequent P(AKI) stays
        soft, so satisfaction is a continuous score on the constrained subset --
        the differentiability-verifiability point of Table IV/V.
        """
        p = self.predict_proba(X)
        a = np.asarray(severe_mask, dtype=float)
        return 1 - a + a * p


# ----------------------------------------------------------------------
# 3. Probabilistic-logical (SPL) — stub; documented fallback DeepProbLog
# ----------------------------------------------------------------------
class SemanticProbabilisticLayer:  # pragma: no cover
    """Placeholder for the SPL integration style (third evidence form).

    Not implemented in v1.0. The plan's fallback is DeepProbLog. Left as an
    explicit stub so the three-style Table IV can be completed in a follow-up
    without changing the harness interface.
    """

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "SPL style is a stub (models.spl_probabilistic.enabled=false). "
            "Registered as pre-registration §7.3; fallback is DeepProbLog."
        )


# ----------------------------------------------------------------------
# 4. Pure-neural baseline (S1): emits NO symbolic/T2 evidence by construction
# ----------------------------------------------------------------------
class PureNeural:
    def __init__(self, hidden_layers=(32, 16), calibrate="isotonic", seed=0):
        self._inner = PipelineCoupled(hidden_layers, calibrate, seed)

    def fit(self, X, y):
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._inner.predict_proba(X)

    def param_hash(self):
        return self._inner.param_hash()

    def constraint_satisfaction(self, X, severe_mask):
        return None  # no symbolic layer -> T2 has no evidence to audit
