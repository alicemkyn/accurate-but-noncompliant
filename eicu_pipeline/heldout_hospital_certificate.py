#!/usr/bin/env python
"""Held-out-hospital compliance-coverage certificate on eICU (post hoc, added after the pre-registered study).

WHY THIS EXISTS.  The pre-registered study used the held-out-hospital split on eICU only as
the axis of a negative control (NC-C), which reports calibration and abstention. An
external reader asked for the CENTRAL outcomes on that split -- severe-subgroup violation
rate, coverage, and the certificate verdict -- and for per-site results on adequately
sized hospitals. None of that was computed before submission. This script computes it
and says so in its own output.

WHAT IT REUSES, UNCHANGED.  The hospital split is run_eicu.py's (rng seed 0, hospitals
shuffled and halved -> `hosp_train` / `hosp_eval`). The model factory, tau selection,
constraint evaluation (`audit.run_T2`) and certificate arithmetic (cov_p5 >= beta AND
vr_p95 <= cap over seeds) are the harness's own. Nothing is re-specified here.

WHAT IS DIFFERENT FROM THE PRIMARY CERTIFICATE, AND WHY IT MATTERS.  The primary audit
splits by PATIENT within one population; here the model is trained on one set of
hospitals and every evaluated decision is made at a hospital the model never saw.
Seeds vary the patient-level split inside the training hospitals (and hence the model),
not the hospital partition, so the across-seed quantiles describe model variability on
a FIXED held-out hospital set. That is a transportability test of the certificate
verdict, not a resampling of hospitals.

PER-SITE.  Each held-out hospital is reported separately, pooled over seeds from counts.
The registered subgroup floor (min_subgroup_n) applies: a site with fewer severe stays
than the floor on a split is UNDETERMINED, not PASS or FAIL. eICU hospitals are small;
expect most sites to be UNDETERMINED and say so rather than drop them.

    export EICU_OUT=/path/with/aki_cohort.parquet
    python heldout_hospital_certificate.py [--seeds N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(__file__).resolve().parent.parent / "mimic_pipeline"
sys.path.insert(0, str(MAIN))
from auditnesy_mimic.config import load_config          # noqa: E402
from auditnesy_mimic import audit, metrics              # noqa: E402
from auditnesy_mimic.experiment import _subject_split, _pipeline_factory, _auroc  # noqa: E402

OUT = Path(os.environ.get("EICU_OUT", "./eicu_out"))


def _hospital_split(df, rng_seed=0):
    """Byte-for-byte the partition run_eicu.py builds."""
    hosp = df["hospitalid"].dropna().unique()
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(hosp)
    train_h = set(hosp[: len(hosp) // 2])
    return train_h, set(hosp) - train_h


def _cert(cov, vr, beta, vmax):
    cov = np.asarray(cov, float); vr = np.asarray(vr, float)
    cov_p5 = float(np.nanpercentile(cov, 5)); vr_p95 = float(np.nanpercentile(vr, 95))
    return {
        "coverage_severe_mean": round(float(np.nanmean(cov)), 4),
        "coverage_severe_p5": round(cov_p5, 4),
        "coverage_severe_p95": round(float(np.nanpercentile(cov, 95)), 4),
        "violation_rate_deployed_mean": round(float(np.nanmean(vr)), 4),
        "violation_rate_deployed_p5": round(float(np.nanpercentile(vr, 5)), 4),
        "violation_rate_deployed_p95": round(vr_p95, 4),
        "certificate_pass": bool(cov_p5 >= beta and vr_p95 <= vmax),
        "fails_on": ("coverage" if cov_p5 < beta else ("violation" if vr_p95 > vmax else "none")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N registered seeds")
    args = ap.parse_args()

    cfg = load_config(MAIN / "config.yaml")
    exp, aud = cfg.experiment, cfg.audit
    beta, vmax, min_n = float(aud["coverage_floor_beta"]), float(aud["violation_rate_max"]), int(aud["min_subgroup_n"])
    seeds = exp["seeds"][: args.seeds] if args.seeds else exp["seeds"]

    df = pd.read_parquet(OUT / "aki_cohort.parquet")
    feat_cols = (OUT / "feature_cols.txt").read_text().splitlines()
    train_h, eval_h = _hospital_split(df)
    tr_df = df[df["hospitalid"].isin(train_h)].reset_index(drop=True)
    ev_df = df[df["hospitalid"].isin(eval_h)].reset_index(drop=True)

    X_tr_all = tr_df[feat_cols].to_numpy(float); y_tr_all = tr_df["aki"].to_numpy(int)
    X_ev = ev_df[feat_cols].to_numpy(float); y_ev = ev_df["aki"].to_numpy(int)
    sev_ev = ev_df["severe_acute"].to_numpy(bool)
    hosp_ev = ev_df["hospitalid"].to_numpy()

    print(f"[eICU held-out] train hospitals {len(train_h)} ({len(tr_df):,} stays) | "
          f"held-out hospitals {len(eval_h)} ({len(ev_df):,} stays, {int(sev_ev.sum()):,} severe)", flush=True)

    per_seed, per_site_rows = [], []
    for seed in seeds:
        # patient-level split INSIDE the training hospitals, as run_shift_eval does;
        # the held-out hospitals are never touched before evaluation
        tr, _ = _subject_split(tr_df, exp["test_size"], seed)
        model = _pipeline_factory(seed, cfg)()
        model.fit(X_tr_all[tr], y_tr_all[tr])
        _, vval = _subject_split(tr_df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(X_tr_all[tr][vval]), aud["abstain_target_rate"])

        t2 = audit.run_T2(model, X_ev, sev_ev, tau, enforce=False)
        t2e = audit.run_T2(model, X_ev, sev_ev, tau, enforce=True)
        per_seed.append({"seed": seed, "tau": round(float(tau), 4),
                         "auroc_heldout": _auroc(y_ev, model.predict_proba(X_ev)),
                         "T2": t2, "T2_enforced": t2e})
        print(f"  seed {seed:2d}: AUROC(held-out)={per_seed[-1]['auroc_heldout']:.3f} "
              f"viol={t2['hard_violations']}/{t2['n_deployed_severe']} deployed of {t2['n_severe']} severe "
              f"cov={t2['coverage_severe']:.3f} vr={t2['violation_rate_deployed']:.3f}", flush=True)

        # per-site: same model, same tau, severe mask restricted to one hospital
        for h in sorted(eval_h):
            m = (hosp_ev == h)
            sev_h = sev_ev & m
            if sev_h.sum() == 0:
                continue
            r = audit.run_T2(model, X_ev[m], sev_ev[m], tau, enforce=False)
            per_site_rows.append({"seed": seed, "hospitalid": int(h), "n_severe": int(r["n_severe"]),
                                  "n_deployed": int(r.get("n_deployed_severe") or 0),
                                  "hard_violations": int(r.get("hard_violations") or 0)})

    # ----- held-out certificate, identical arithmetic to experiment._aggregate._cert -----
    logged = _cert([p["T2"]["coverage_severe"] for p in per_seed],
                   [p["T2"]["violation_rate_deployed"] for p in per_seed], beta, vmax)
    enforced = _cert([p["T2_enforced"]["coverage_severe"] for p in per_seed],
                     [p["T2_enforced"]["violation_rate_deployed"] for p in per_seed], beta, vmax)

    # ----- per-site, pooled over seeds from counts -----
    ps = pd.DataFrame(per_site_rows)
    sites = []
    for h, g in ps.groupby("hospitalid"):
        n_sev_min = int(g["n_severe"].min())
        dep, viol = g["n_deployed"].sum(), g["hard_violations"].sum()
        sites.append({
            "hospitalid": int(h),
            "n_severe_per_seed": int(round(g["n_severe"].mean())),
            "n_seeds": int(len(g)),
            "coverage_severe": round(float(dep / g["n_severe"].sum()), 4) if g["n_severe"].sum() else None,
            "violation_rate_deployed": round(float(viol / dep), 4) if dep else None,
            # the registered floor applies per split; a site is determinate only if it
            # clears the floor on EVERY split it appears in
            "status": "OK" if n_sev_min >= min_n else "UNDETERMINED",
        })
    sites.sort(key=lambda s: -s["n_severe_per_seed"])
    n_ok = sum(s["status"] == "OK" for s in sites)

    payload = {
        "prereg_tag": cfg.prereg_tag,
        "analysis_status": ("Post hoc, added after the pre-registered study; not pre-registered. The hospital "
                            "partition, model factory, tau rule, constraint and certificate arithmetic "
                            "are the released harness's own and were not changed."),
        "hospital_partition": {"rng_seed": 0, "n_train_hospitals": len(train_h), "n_heldout_hospitals": len(eval_h),
                               "n_train_stays": int(len(tr_df)), "n_heldout_stays": int(len(ev_df)),
                               "n_heldout_severe": int(sev_ev.sum())},
        "what_seeds_vary": ("the patient-level training split inside the training hospitals, hence the "
                            "model; NOT the hospital partition. Quantiles describe model variability on a "
                            "fixed held-out hospital set."),
        "n_seeds": len(seeds),
        "auroc_heldout": {"mean": round(float(np.mean([p["auroc_heldout"] for p in per_seed])), 4),
                          "min": round(float(np.min([p["auroc_heldout"] for p in per_seed])), 4),
                          "max": round(float(np.max([p["auroc_heldout"] for p in per_seed])), 4)},
        "certificate": {"coverage_floor_beta": beta, "violation_rate_max": vmax,
                        "logged_policy": logged, "enforced_policy": enforced},
        "per_site": {"subgroup_floor": min_n, "n_sites_with_severe_cases": len(sites),
                     "n_sites_determinate": n_ok,
                     "note": ("eICU hospitals are small relative to the registered floor; a site below the "
                              "floor on any split is UNDETERMINED and is reported, not dropped."),
                     "sites": sites},
        "per_seed": per_seed,
    }
    # guards: a result that exists only in a log is not a result
    for k in ("certificate_pass", "violation_rate_deployed_p95", "coverage_severe_p5"):
        if payload["certificate"]["logged_policy"].get(k) is None:
            raise SystemExit(f"certificate field unpopulated: {k}")

    out = OUT / "heldout_hospital_certificate.json"
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\n[held-out certificate, logged]   cov p5={logged['coverage_severe_p5']} (floor {beta}) | "
          f"vr p95={logged['violation_rate_deployed_p95']} (cap {vmax}) -> "
          f"{'PASS' if logged['certificate_pass'] else 'FAIL on ' + logged['fails_on']}")
    print(f"[held-out certificate, enforced] cov p5={enforced['coverage_severe_p5']} | "
          f"vr p95={enforced['violation_rate_deployed_p95']} -> "
          f"{'PASS' if enforced['certificate_pass'] else 'FAIL on ' + enforced['fails_on']}")
    print(f"[per-site] {len(sites)} held-out sites with severe cases; {n_ok} clear the floor of {min_n} on every split")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
