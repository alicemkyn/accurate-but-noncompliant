"""Orchestration: 20 seeds x subject-level splits, bootstrap CIs, CI-gated pass.

Applies the RATIFIED pre-registration (prereg-v1.0):
  * split by subject_id (GroupShuffleSplit) -- no patient leakage (#4)
  * T2 constraint applies only to ACUTE severe AKI (severe_acute mask; #1 clinical fix)
  * T3 calibration gated on the bootstrap 95% CI UPPER bound of ECE (#3/RR3)
  * T3 distribution-SHIFT is a distinct train-early / evaluate-late experiment (#2 fix),
    not a spurious later-era subset of an in-distribution random split
  * tau selected once on validation to hit the deferral target, then frozen
  * per-seed pass stability reported (a model passing 17/20 is a finding)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from . import audit, metrics, negative_controls as nc
from .io_utils import ensure_out, save_json
from .models import GradientBoosted, PipelineCoupled


def _subject_split(df, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["subject_id"].values))
    return tr, te


def _arrays(df, feat_cols):
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    severe = df["severe_acute"].to_numpy(bool)   # #1: acute-severe applicability mask
    subj = df["subject_id"].to_numpy()
    return X, y, severe, subj


def _pipeline_factory(seed, cfg):
    m = cfg.models["pipeline_coupled"]
    core = m.get("core", "mlp")
    if core == "gbm":
        return lambda: GradientBoosted(calibrate=m["calibrate"], seed=seed)
    return lambda: PipelineCoupled(m["hidden_layers"], m["calibrate"], seed=seed)


def run(cfg, df, feat_cols) -> dict:
    out = ensure_out(cfg.out_dir)
    exp = cfg.experiment
    audit_cfg = dict(cfg.audit); audit_cfg["bootstrap_B"] = exp["bootstrap_B"]
    X, y, severe, subj = _arrays(df, feat_cols)

    per_seed = []
    for seed in exp["seeds"]:
        tr, te = _subject_split(df, exp["test_size"], seed)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        severe_te = severe[te]
        subj_te = subj[te]

        factory = _pipeline_factory(seed, cfg)
        model = factory()
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)

        # tau frozen from a validation slice of TRAIN (never the test fold)
        _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(Xtr[vval]), cfg.audit["abstain_target_rate"])

        t1 = audit.run_T1(factory, Xtr, ytr, Xte, model)
        t2 = audit.run_T2(model, Xte, severe_te, tau, enforce=False)   # rules logged
        t2e = audit.run_T2(model, Xte, severe_te, tau, enforce=True)   # rules enforced (escalate)
        t3 = audit.run_T3(probs, yte, subj_te, tau, audit_cfg, seed=seed)
        probe = audit.run_T2s_probe(factory, Xtr, ytr, Xte, feat_cols, cfg.audit)

        per_seed.append({
            "seed": seed, "n_test": int(len(te)),
            "auroc": _auroc(yte, probs), "tau": tau,
            "T1": t1, "T2": t2, "T2_enforced": t2e, "T3": t3, "T2s": probe,
        })
        print(f"  seed {seed:2d}: AUROC={per_seed[-1]['auroc']:.3f} "
              f"T1={'P' if t1['pass_split'] else 'F'} "
              f"T2viol(logged)={t2.get('hard_violations')}/{t2.get('n_severe')} "
              f"T2(enforced)=0(esc {t2e.get('escalated_on_severe')}) "
              f"ECE={t3['ece_point']:.3f} CIhi={t3['ece_ci'][1]:.3f}")

    agg = _aggregate(per_seed, cfg)
    shift = run_shift_eval(cfg, df, feat_cols)              # #2: proper era-shift eval

    # negative controls (once, on the seed-0 split)
    tr0, te0 = _subject_split(df, exp["test_size"], 0)
    nc_results = _run_negative_controls(cfg, df, feat_cols, X, y, severe, subj, tr0, te0, audit_cfg)

    # diagnostic: AKI rate within the acute-severe subgroup distinguishes a model
    # reasoning-shortcut (rate high, model still says no-AKI) from ICD label noise
    # (rate not high -> the administrative code misses AKI that the rule flags).
    aki_rate_severe = float(df.loc[severe, "aki"].mean()) if severe.sum() else float("nan")

    summary = {
        "prereg_tag": cfg.prereg_tag,
        "model_core": cfg.models["pipeline_coupled"].get("core", "mlp"),
        "dataset": f"MIMIC-IV hosp / {cfg.task['name']} (label_source={cfg.task['label_source']})",
        "n_admissions": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "aki_rate": round(float(y.mean()), 4),
        "n_severe_acute": int(severe.sum()),
        "aki_rate_in_severe_acute": round(aki_rate_severe, 4),
        "n_seeds": len(exp["seeds"]),
        "aggregate": agg,
        "shift_evaluation": shift,
        "negative_controls": nc_results,
    }
    save_json(out / "audit_summary_mimic.json", summary)
    save_json(out / "per_seed_results.json", per_seed)
    print(f"\nSaved -> {out / 'audit_summary_mimic.json'}")
    return summary


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return round(float(roc_auc_score(y, p)), 4)
    except ValueError:
        return float("nan")


def _aggregate(per_seed, cfg):
    """Across-seed means + CI-gated pass decisions (RR2/RR3)."""
    ece_pts = np.array([ps["T3"]["ece_point"] for ps in per_seed], float)
    ece_hi = np.array([ps["T3"]["ece_ci"][1] for ps in per_seed], float)
    t1_pass = np.array([ps["T1"]["pass_split"] for ps in per_seed])
    t2_viol = np.array([(ps["T2"]["hard_violations"] or 0) for ps in per_seed])
    t2_pass = np.array([bool(ps["T2"]["pass_split"]) for ps in per_seed
                        if ps["T2"]["pass_split"] is not None])
    auroc = np.array([ps["auroc"] for ps in per_seed], float)
    probe_sens = np.array([ps["T2s"].get("creatinine_sensitivity", np.nan) for ps in per_seed], float)

    thr = cfg.audit["ece_pass_upper_ci"]
    t3_ci_upper_median = float(np.nanmedian(ece_hi))
    t1_ok = bool(t1_pass.all())
    t3_ok = bool(t3_ci_upper_median < thr)

    # enforced-arm (same model, rules ESCALATE): T2 = 0 by construction; report the cost
    t2e_viol = np.array([(ps["T2_enforced"]["hard_violations"] or 0) for ps in per_seed])
    esc = np.array([ps["T2_enforced"]["escalated_on_severe"] for ps in per_seed], float)
    n_sev = np.array([max(ps["T2"]["n_severe"], 1) for ps in per_seed], float)
    t2e_ok = bool(t2e_viol.sum() == 0)

    # --- compliance-coverage certificate (closes abstention-gaming) ---
    beta = cfg.audit["coverage_floor_beta"]
    vmax = cfg.audit["violation_rate_max"]

    def _cert(key):
        cov = np.array([ps[key]["coverage_severe"] for ps in per_seed], float)
        vr = np.array([ps[key]["violation_rate_deployed"] for ps in per_seed], float)
        cov_p5 = float(np.nanpercentile(cov, 5))     # conservative coverage across seeds
        vr_p95 = float(np.nanpercentile(vr, 95))     # conservative violation rate across seeds
        return {
            "coverage_severe_mean": round(float(np.nanmean(cov)), 4),
            "coverage_severe_p5": round(cov_p5, 4),
            "violation_rate_deployed_mean": round(float(np.nanmean(vr)), 4),
            "violation_rate_deployed_p95": round(vr_p95, 4),
            # PASS requires BOTH: enough deployed coverage AND low violation among deployed
            "certificate_pass": bool(cov_p5 >= beta and vr_p95 <= vmax),
            "fails_on": ("coverage" if cov_p5 < beta else ("violation" if vr_p95 > vmax else "none")),
        }

    cc_certificate = {
        "coverage_floor_beta": beta,
        "violation_rate_max": vmax,
        "logged_policy": _cert("T2"),
        "enforced_policy": _cert("T2_enforced"),
    }

    return {
        "arm": "primary = learned neural core, rules LOGGED",
        "auroc_mean": round(float(np.nanmean(auroc)), 4),
        "auroc_range": [round(float(np.nanmin(auroc)), 4), round(float(np.nanmax(auroc)), 4)],
        "T1_pass_stability": f"{int(t1_pass.sum())}/{len(t1_pass)}",
        "T1_pass": t1_ok,
        "T2_hard_violations_mean": round(float(t2_viol.mean()), 3),
        "T2_pass_stability": f"{int(t2_pass.sum())}/{len(t2_pass)}" if len(t2_pass) else "n/a",
        "T2_pass": bool(t2_viol.sum() == 0),
        "T3_ece_point_mean": round(float(np.nanmean(ece_pts)), 4),
        "T3_ece_ci_upper_median": round(t3_ci_upper_median, 4),
        "T3_pass": t3_ok,
        "T3_pass_borderline": bool(t3_ci_upper_median >= thr * 0.8),
        "probe_creatinine_sensitivity_mean": round(float(np.nanmean(probe_sens)), 4),
        "audit_readiness": f"{int(t1_ok) + int(bool(t2_viol.sum() == 0)) + int(t3_ok)}/3",
        "enforced_arm": {
            "policy": "same model, hard constraint ESCALATES confident no-AKI to a deferral",
            "T2_hard_violations_mean": round(float(t2e_viol.mean()), 3),
            "T2_pass": t2e_ok,
            "escalated_on_severe_mean": round(float(esc.mean()), 1),
            "escalation_rate_on_severe": round(float((esc / n_sev).mean()), 4),
            "audit_readiness": f"{int(t1_ok) + int(t2e_ok) + int(t3_ok)}/3",
        },
        "compliance_coverage_certificate": cc_certificate,
    }


def run_shift_eval(cfg, df, feat_cols, n_shift_seeds=3) -> dict:
    """T3 distribution-SHIFT: train on early eras, evaluate on late eras.

    Reports, per seed: ECE CI-upper in-distribution (early held-out) vs shifted
    (late), the degradation, and abstention rates. Pass criteria (prereg):
      - ECE degradation (shift - id, CI-upper) <= shift_ece_degradation_max
      - abstention rate on shifted cohort >= in-distribution (DIRECTIONAL)
    """
    exp = cfg.experiment
    B = exp["bootstrap_B"]
    yg = df["anchor_year_group"].astype(str)
    early = df[yg.isin(cfg.shift["train_groups"])].reset_index(drop=True)
    late = df[yg.isin(cfg.shift["eval_groups"])].reset_index(drop=True)
    if len(early) == 0 or len(late) == 0:
        return {"available": False}

    Xl = late[feat_cols].to_numpy(float); yl = late["aki"].to_numpy(int)
    subjl = late["subject_id"].to_numpy()
    Xe = early[feat_cols].to_numpy(float); ye = early["aki"].to_numpy(int)
    subje = early["subject_id"].to_numpy()
    n_bins = cfg.audit["n_bins_ece"]
    ece_fn = lambda a, b: metrics.ece(a, b, n_bins)

    results = []
    for seed in exp["seeds"][:n_shift_seeds]:
        tr, te = _subject_split(early, exp["test_size"], seed)
        model = _pipeline_factory(seed, cfg)()
        model.fit(Xe[tr], ye[tr])
        _, vval = _subject_split(early.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(Xe[tr][vval]), cfg.audit["abstain_target_rate"])

        p_id, p_sh = model.predict_proba(Xe[te]), model.predict_proba(Xl)
        _, _, ece_id_hi = metrics.bootstrap_ci(ece_fn, p_id, ye[te], subje[te], B=B, seed=seed)
        _, _, ece_sh_hi = metrics.bootstrap_ci(ece_fn, p_sh, yl, subjl, B=B, seed=seed)
        ab_id = float(metrics.abstain_mask(p_id, tau).mean())
        ab_sh = float(metrics.abstain_mask(p_sh, tau).mean())
        degr = ece_sh_hi - ece_id_hi
        results.append({
            "seed": seed,
            "ece_id_ci_upper": round(float(ece_id_hi), 4),
            "ece_shift_ci_upper": round(float(ece_sh_hi), 4),
            "ece_degradation": round(float(degr), 4),
            "abstain_id": round(ab_id, 4), "abstain_shift": round(ab_sh, 4),
            "degradation_ok": bool(degr <= cfg.audit["shift_ece_degradation_max"]),
            "direction_ok": bool(ab_sh >= ab_id),
        })

    degr_ok = sum(r["degradation_ok"] for r in results)
    dir_ok = sum(r["direction_ok"] for r in results)
    n = len(results)
    return {
        "available": True, "n_seeds": n,
        "degradation_pass_stability": f"{degr_ok}/{n}",
        "direction_pass_stability": f"{dir_ok}/{n}",
        "T3_shift_pass": bool(degr_ok == n and dir_ok == n),
        "per_seed": results,
    }


def _run_negative_controls(cfg, df, feat_cols, X, y, severe, subj, tr, te, audit_cfg):
    factory0 = _pipeline_factory(0, cfg)
    tau0 = 0.65  # nominal for the shift diagnostic
    yg = df["anchor_year_group"].astype(str)
    era_train = yg.isin(cfg.shift["train_groups"]).to_numpy()
    era_eval = yg.isin(cfg.shift["eval_groups"]).to_numpy()

    return {
        "NC_A_uncalibrated": nc.nc_a_uncalibrated(
            X[tr], y[tr], X[te], y[te], subj[te], audit_cfg, seed=0),
        "NC_B_constraint_blind": nc.nc_b_constraint_blind(
            X[tr], y[tr], X[te], severe[te], feat_cols,
            cfg.audit["abstain_target_rate"], seed=0),
        "NC_C_stale": nc.nc_c_stale(
            factory0, X[era_train], y[era_train], X[era_eval], y[era_eval],
            subj[era_eval], tau0, audit_cfg, seed=0)
        if era_train.sum() and era_eval.sum() else {"available": False},
        "NC_D_nondeterministic": nc.nc_d_nondeterministic(
            lambda s: PipelineCoupled(cfg.models["pipeline_coupled"]["hidden_layers"],
                                      cfg.models["pipeline_coupled"]["calibrate"], seed=s),
            X[tr], y[tr], X[te], seed=0),
    }
