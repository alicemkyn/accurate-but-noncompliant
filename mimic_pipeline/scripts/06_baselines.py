"""STEP 6 — additional model arms + paired significance tests (prereg-v1.4).

Discharges two pre-registered commitments that v1.3 left undelivered:

  * S1 PURE-NEURAL BASELINE (prereg §7).  An identical MLP with no symbolic
    coupling.  `constraint_satisfaction` returns None by construction, so T2 is
    reported as "no evidence emitted" rather than as a pass -- which is the
    point of the arm: a system can be accurate and produce NO auditable
    constraint evidence at all, and the framework must say so rather than
    silently score it.
  * LOGISTIC REGRESSION.  A transparent linear arm, so the reported
    accuracy-vs-compliance tension cannot be attributed to model family or
    capacity.  Constraint evidence is emitted exactly as in the pipeline-coupled
    model, so its T2 numbers are directly comparable to the primary arm.

It also runs the paired significance tests the paper currently lacks.  Because
every arm is evaluated on the SAME 20 patient-level splits, seeds are matched
and a paired test is the correct choice: we use a two-sided Wilcoxon signed-rank
test on the per-seed differences (no normality assumption, exact for n=20),
with Holm-Bonferroni correction across the family of comparisons.

Usage:
    python scripts/06_baselines.py --cohort <path.parquet> [--seeds 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic import audit, metrics  # noqa: E402
from auditnesy_mimic.config import load_config  # noqa: E402
from auditnesy_mimic.models import PipelineCoupled, GradientBoosted, PureNeural  # noqa: E402

FEAT_ORDER = ["creatinine", "glucose", "bun", "potassium", "sodium",
              "hemoglobin", "wbc", "bicarbonate", "age", "is_female"]


class LogisticArm:
    """Transparent linear arm; emits constraint evidence like PipelineCoupled."""

    def __init__(self, seed=0, calibrate="isotonic"):
        self.seed = seed
        self.calibrate = calibrate
        self.pipe = None

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        base = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, random_state=self.seed))])
        if self.calibrate == "isotonic":
            self.pipe = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            self.pipe = base
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)[:, 1]

    def param_hash(self):
        import hashlib
        try:
            cs = self.pipe.calibrated_classifiers_
            w = np.concatenate([c.estimator.named_steps["lr"].coef_.ravel() for c in cs])
        except AttributeError:
            w = self.pipe.named_steps["lr"].coef_.ravel()
        return hashlib.sha256(np.round(w, 6).tobytes()).hexdigest()[:16]

    def constraint_satisfaction(self, X, severe_mask):
        """Same symbolic check the pipeline-coupled model emits."""
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        sev = np.asarray(severe_mask, bool)
        return np.where(sev & (pred == 0), 0.0, 1.0)


def _subject_split(df, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["subject_id"].values))


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


ARMS = {
    "pipeline_coupled_mlp": lambda seed, mcfg: PipelineCoupled(
        mcfg["hidden_layers"], mcfg["calibrate"], seed=seed),
    "gbm": lambda seed, mcfg: GradientBoosted(calibrate=mcfg["calibrate"], seed=seed),
    "pure_neural": lambda seed, mcfg: PureNeural(
        mcfg["hidden_layers"], mcfg["calibrate"], seed=seed),
    "logistic_regression": lambda seed, mcfg: LogisticArm(seed=seed,
                                                          calibrate=mcfg["calibrate"]),
}


def run_arm(name, factory, df, X, y, severe, subj, cfg, seeds, feat_cols):
    audit_cfg = dict(cfg.audit)
    audit_cfg["bootstrap_B"] = cfg.experiment["bootstrap_B"]
    per_seed = []
    for seed in seeds:
        tr, te = _subject_split(df, cfg.experiment["test_size"], seed)
        model = factory(seed, cfg.models["pipeline_coupled"])
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])
        _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(X[tr][vval]),
                                 cfg.audit["abstain_target_rate"])
        t2 = audit.run_T2(model, X[te], severe[te], tau, enforce=False)
        t3 = audit.run_T3(probs, y[te], subj[te], tau, audit_cfg, seed=seed)
        per_seed.append({
            "seed": seed, "auroc": _auroc(y[te], probs), "tau": float(tau),
            "T2_evidence": bool(t2.get("evidence")),
            "hard_violations": t2.get("hard_violations"),
            "n_severe": t2.get("n_severe"),
            "coverage_severe": t2.get("coverage_severe"),
            "violation_rate_deployed": t2.get("violation_rate_deployed"),
            "ece_point": t3["ece_point"], "ece_ci_upper": t3["ece_ci"][1],
        })
        print(f"  [{name}] seed {seed:2d}: AUROC={per_seed[-1]['auroc']:.4f} "
              f"viol={t2.get('hard_violations')} ECEhi={t3['ece_ci'][1]:.4f}", flush=True)

    def arr(k):
        return np.array([ps[k] for ps in per_seed
                         if ps[k] is not None], float)

    ece_hi = arr("ece_ci_upper")
    thr = cfg.audit["ece_pass_upper_ci"]
    beta = cfg.audit["coverage_floor_beta"]
    vmax = cfg.audit["violation_rate_max"]
    has_ev = all(ps["T2_evidence"] for ps in per_seed)
    if has_ev:
        cov = arr("coverage_severe"); vr = arr("violation_rate_deployed")
        cov_p5 = float(np.nanpercentile(cov, 5)); vr_p95 = float(np.nanpercentile(vr, 95))
        cert = {"coverage_severe_p5": round(cov_p5, 4),
                "violation_rate_deployed_p95": round(vr_p95, 4),
                "certificate_pass": bool(cov_p5 >= beta and vr_p95 <= vmax)}
        t2_summary = {"evidence": True,
                      "hard_violations_mean": round(float(arr("hard_violations").mean()), 3),
                      "T2_pass": bool(arr("hard_violations").sum() == 0)}
    else:
        cert = {"certificate_pass": None,
                "note": "no constraint evidence emitted; certificate not computable"}
        t2_summary = {"evidence": False, "hard_violations_mean": None,
                      "T2_pass": None,
                      "note": "T2 reported as NO EVIDENCE EMITTED (prereg §7 S1)"}
    return {
        "arm": name,
        "n_seeds": len(seeds),
        "auroc_mean": round(float(arr("auroc").mean()), 4),
        "auroc_range": [round(float(arr("auroc").min()), 4),
                        round(float(arr("auroc").max()), 4)],
        "T2": t2_summary,
        "T3_ece_ci_upper_median": round(float(np.nanmedian(ece_hi)), 4),
        "T3_pass": bool(float(np.nanmedian(ece_hi)) < thr),
        "compliance_coverage_certificate": cert,
        "per_seed": per_seed,
    }


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return {labels[i]: {"p_raw": float(pvals[i]), "p_holm": float(adj[i])}
            for i in range(m)}


def paired_tests(arms):
    """Two-sided Wilcoxon signed-rank on per-seed differences + Holm correction."""
    from scipy.stats import wilcoxon

    def series(name, key):
        return np.array([ps[key] for ps in arms[name]["per_seed"]], float)

    comps, labels, pvals = [], [], []

    def add(label, a, b, key, note):
        x, y = series(a, key), series(b, key)
        n = min(len(x), len(y))
        d = x[:n] - y[:n]
        if np.allclose(d, 0):
            comps.append({"comparison": label, "key": key, "note": note,
                          "median_difference": 0.0, "p_raw": 1.0,
                          "detail": "all per-seed differences identically zero"})
            labels.append(label); pvals.append(1.0)
            return
        stat, p = wilcoxon(x[:n], y[:n], alternative="two-sided")
        comps.append({"comparison": label, "key": key, "note": note,
                      "mean_a": round(float(np.mean(x[:n])), 4),
                      "mean_b": round(float(np.mean(y[:n])), 4),
                      "median_difference": round(float(np.median(d)), 4),
                      "wilcoxon_statistic": float(stat), "p_raw": float(p),
                      "n_pairs": int(n)})
        labels.append(label); pvals.append(float(p))

    add("mlp_vs_gbm_auroc", "pipeline_coupled_mlp", "gbm", "auroc",
        "does the accuracy differ between model families?")
    add("mlp_vs_gbm_violation_rate", "pipeline_coupled_mlp", "gbm", "violation_rate_deployed",
        "does the compliance failure differ between model families?")
    add("mlp_vs_logistic_auroc", "pipeline_coupled_mlp", "logistic_regression", "auroc",
        "does a linear arm discriminate differently?")
    add("mlp_vs_logistic_violation_rate", "pipeline_coupled_mlp", "logistic_regression",
        "violation_rate_deployed",
        "is the compliance failure present in a transparent linear model too?")
    add("mlp_vs_pure_neural_auroc", "pipeline_coupled_mlp", "pure_neural", "auroc",
        "does removing symbolic coupling change accuracy?")

    corrected = holm(np.array(pvals), labels)
    for c in comps:
        c.update(corrected[c["comparison"]])
    return {"test": "two-sided Wilcoxon signed-rank on per-seed paired differences",
            "correction": "Holm-Bonferroni across the 5 comparisons",
            "rationale": ("all arms are evaluated on the SAME 20 patient-level splits, "
                          "so seeds are matched; Wilcoxon avoids a normality assumption"),
            "comparisons": comps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.experiment["seeds"][:args.seeds] if args.seeds else cfg.experiment["seeds"]
    df = pd.read_parquet(args.cohort)
    feat_cols = [c for c in FEAT_ORDER if c in df.columns]
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    severe = df["severe_acute"].to_numpy(bool)
    subj = df["subject_id"].to_numpy()

    print(f"[baselines] cohort={len(df)} severe={int(severe.sum())} seeds={len(seeds)}", flush=True)
    arms = {}
    for name, factory in ARMS.items():
        print(f"[baselines] running arm: {name}", flush=True)
        arms[name] = run_arm(name, factory, df, X, y, severe, subj, cfg, seeds, feat_cols)

    tests = paired_tests(arms)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in arms.items()}
    (outdir / "model_arms_v14.json").write_text(json.dumps(
        {"prereg_tag": "prereg-v1.4", "n_seeds": len(seeds),
         "n_arms": len(arms), "arms": slim,
         "significance_tests": tests}, indent=1))
    (outdir / "model_arms_per_seed_v14.json").write_text(json.dumps(
        {k: v["per_seed"] for k, v in arms.items()}, indent=1))
    print(f"\nSaved -> {outdir/'model_arms_v14.json'}", flush=True)
    for c in tests["comparisons"]:
        print(f"  {c['comparison']}: p_raw={c['p_raw']:.4g} p_holm={c['p_holm']:.4g}", flush=True)


if __name__ == "__main__":
    main()
