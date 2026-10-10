"""STEP 12 -- the symmetric half of the deletion test (prereg-v1.11).

WHY THIS EXISTS.  Block H of `10_prior_ckd_sensitivity.py` answers the reviewer
objection "your headline is an artifact of dialysis-dependent patients" for the
LABEL side: delete the prior-renal-replacement stratum and the administrative
label still disagrees with the guideline on 43% of the 3,507 cases that remain.

But the paper's headline is not a label statistic -- it is a MODEL statistic: a
mean 71% of deployed severe decisions are confident denials. Block H does not
touch that number, and a reader who follows the deletion argument will
immediately ask what the deployed violation rate is on the surviving cases.
Answering it with prose would be exactly the move this paper exists to refuse,
so it is computed here.

WHAT THIS DOES.  Reuses `05_subgroup_audit.py`'s own per-split machinery (same
model factory, same splits, same tau selection, same statistic definitions) with
one new subgroup dimension: prior renal history, built by
`10_prior_ckd_sensitivity.py`'s `build_prior_indicators` from admissions
discharged BEFORE the audited one. Nothing about the audit changes; only the
grouping variable is new.

STATUS: exploratory, post hoc, selection-limited -- it inherits every caveat of
blocks G and H, including that prior history is differentially observable.  No
registered threshold is touched.

    python scripts/12_violation_rate_by_renal_history.py \
        --cohort audit_mimic_out/aki_cohort.parquet \
        --mimic-hosp ../mimic_raw/hosp --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auditnesy_mimic import metrics  # noqa: E402
from auditnesy_mimic.config import load_config  # noqa: E402
from auditnesy_mimic.models import PipelineCoupled  # noqa: E402

_sg = import_module("05_subgroup_audit")
_ckd = import_module("10_prior_ckd_sensitivity")

# reuse 05's definitions verbatim -- re-specifying them here would let this arm
# drift away from the numbers the paper reports
_subject_split = _sg._subject_split
per_split_subgroup_stats = _sg.per_split_subgroup_stats
agg_across_seeds = _sg.agg_across_seeds
build_prior_indicators = _ckd.build_prior_indicators

STRATUM_ORDER = ["prior_renal_replacement", "no_prior_renal_replacement",
                 "prior_status_unobservable"]


def renal_history_labels(df: pd.DataFrame) -> pd.Series:
    """Three mutually exclusive strata covering every admission exactly once.

    These are the BLOCK H strata (a partition), not the block G reading strata
    (which are nested: "no renal disease at all" sits inside "no prior renal
    replacement"). Keeping the two triples distinct is deliberate -- conflating
    them is how a partition claim and a non-partition claim ended up in adjacent
    paragraphs of an earlier draft.
    """
    obs = df["_prior_obs"].astype(bool).to_numpy()
    esrd = df["_prior_esrd"].astype(bool).to_numpy()
    out = np.where(~obs, "prior_status_unobservable",
                   np.where(esrd, "prior_renal_replacement",
                            "no_prior_renal_replacement"))
    return pd.Series(out, index=df.index, name="renal_history")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--mimic-hosp", default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hosp = Path(args.mimic_hosp) if args.mimic_hosp else Path(cfg.mimic_hosp_dir)
    if not hosp.exists():
        raise SystemExit(
            f"MIMIC-IV hosp directory not found: {hosp}\n"
            "This arm needs the credentialed diagnoses_icd and admissions tables to "
            "build the prior-history indicator. It refuses rather than guessing.")

    df = pd.read_parquet(args.cohort)
    df = build_prior_indicators(df, hosp)
    labels = renal_history_labels(df)

    # read exactly the keys 05 reads, from the same places
    mcfg = cfg.models["pipeline_coupled"]
    thr_severe = float(cfg.constraints["creatinine_severe_mgdl"])
    min_n = int(cfg.audit["min_subgroup_n"])
    seeds = cfg.experiment["seeds"]
    test_size = cfg.experiment["test_size"]

    feat_cols = [c for c in _sg.FEAT_ORDER if c in df.columns]
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    severe_all = df["severe_acute"].astype(bool).to_numpy()
    nadir_all = pd.to_numeric(df["creat_nadir_24h"], errors="coerce").to_numpy()

    per_seed = []
    for seed in seeds:
        tr, te = _subject_split(df, test_size, seed)
        model = PipelineCoupled(mcfg["hidden_layers"], mcfg["calibrate"], seed=seed)
        model.fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])

        _, vval = _subject_split(df.iloc[tr].reset_index(drop=True), 0.25, seed + 7)
        tau = metrics.select_tau(model.predict_proba(X[tr][vval]),
                                 cfg.audit["abstain_target_rate"])

        g = labels.iloc[te].reset_index(drop=True)
        per_seed.append(per_split_subgroup_stats(
            probs, y[te], severe_all[te], tau, g, thr_severe, nadir_all[te], min_n))
        print(f"  seed {seed:2d}: tau={tau:.4f} n_severe={int(severe_all[te].sum())}",
              flush=True)

    keys = ["icd_agreement_on_severe", "coverage_severe", "violation_rate_deployed",
            "hard_violations", "n_severe"]
    strata = agg_across_seeds(per_seed, keys)

    # The UNDETERMINED flag is set when ANY split falls below min_subgroup_n, but the
    # aggregate only records mean/p5/p95 of n_severe, from which a reader cannot
    # verify the breach (a 5th percentile of 506 does not show a split below 500).
    # Record the per-split minimum and the number of breaching splits explicitly.
    for lab, rec in strata.items():
        ns = [d[lab]["n_severe"] for d in per_seed if lab in d and d[lab].get("n_severe") is not None]
        if ns:
            rec["n_severe_min_over_splits"] = int(min(ns))
            rec["n_severe_max_over_splits"] = int(max(ns))
            rec["n_splits_below_floor"] = int(sum(n < min_n for n in ns))
            rec["subgroup_floor"] = int(min_n)

    # ---------- exact per-seed pooling on the SURVIVING cohort, and its certificate ----------
    # A reviewer asked for the violation rate, coverage and certificate verdict on the
    # restricted (non-dialysis) cohort. The mean is recoverable from the per-stratum
    # aggregates, but the certificate is decided on TAIL quantiles across seeds
    # (cov_p5 >= beta and vr_p95 <= cap, exactly as experiment.py decides it), and tails
    # of a pooled quantity cannot be read off per-stratum tails. So pool per seed, from
    # counts, then take the quantiles -- the same arithmetic as the primary certificate.
    beta = float(cfg.audit["coverage_floor_beta"])
    vmax = float(cfg.audit["violation_rate_max"])
    surviving_strata = ["no_prior_renal_replacement", "prior_status_unobservable"]
    pooled_rows = []
    for d in per_seed:
        n_sev = n_dep = viol = 0.0
        for s in surviving_strata:
            c = d.get(s)
            if not c or c.get("n_severe") is None:
                continue
            ns = float(c["n_severe"])
            cov = c.get("coverage_severe")
            hv = c.get("hard_violations")
            if cov is None or hv is None:
                continue
            n_sev += ns
            n_dep += ns * float(cov)
            viol += float(hv)
        if n_sev > 0 and n_dep > 0:
            pooled_rows.append({"n_severe": n_sev, "n_deployed": n_dep,
                                "coverage": n_dep / n_sev, "violation_rate": viol / n_dep})
    if not pooled_rows:
        raise SystemExit("could not pool the surviving cohort per seed: no complete rows")
    cov_arr = np.array([r["coverage"] for r in pooled_rows])
    vr_arr = np.array([r["violation_rate"] for r in pooled_rows])
    nsev_arr = np.array([r["n_severe"] for r in pooled_rows])
    cov_p5, vr_p95 = float(np.percentile(cov_arr, 5)), float(np.percentile(vr_arr, 95))
    restricted_certificate = {
        "rule": "coverage_p5 >= beta AND violation_p95 <= cap, identical to experiment.py",
        "beta": beta, "cap": vmax,
        "n_seeds": int(len(pooled_rows)),
        "n_severe_per_split": {"mean": round(float(nsev_arr.mean()), 1),
                               "min": round(float(nsev_arr.min()), 1),
                               "max": round(float(nsev_arr.max()), 1)},
        "coverage": {"mean": round(float(cov_arr.mean()), 4), "p5": round(cov_p5, 4),
                     "p95": round(float(np.percentile(cov_arr, 95)), 4)},
        "violation_rate": {"mean": round(float(vr_arr.mean()), 4),
                           "p5": round(float(np.percentile(vr_arr, 5)), 4),
                           "p95": round(vr_p95, 4)},
        "coverage_limb_passes": bool(cov_p5 >= beta),
        "violation_limb_passes": bool(vr_p95 <= vmax),
        "certificate_pass": bool(cov_p5 >= beta and vr_p95 <= vmax),
        "fails_on": ("none" if (cov_p5 >= beta and vr_p95 <= vmax)
                     else "violation" if cov_p5 >= beta else
                     "coverage" if vr_p95 <= vmax else "both"),
        # the pooled surviving cohort is ~1,056 severe cases per split, far above the
        # registered subgroup floor, so the POOLED estimate is determinate even though the
        # two strata composing it individually sit near the floor
        "pooled_cohort_above_subgroup_floor": bool(nsev_arr.min() >= int(cfg.audit["min_subgroup_n"])),
        "subgroup_floor": int(cfg.audit["min_subgroup_n"]),
        "per_seed": [{k: round(v, 4) for k, v in r.items()} for r in pooled_rows],
    }
    print(f"\n[restricted certificate] coverage p5={cov_p5:.4f} (floor {beta}), "
          f"violation p95={vr_p95:.4f} (cap {vmax}) -> "
          f"{'PASS' if restricted_certificate['certificate_pass'] else 'FAIL on ' + restricted_certificate['fails_on']}",
          flush=True)

    # The headline comparison: does deleting the dialysis stratum move the 71%?
    #
    # POOLING WEIGHT.  `violation_rate_deployed` is violations / n_DEPLOYED, not
    # violations / n_severe -- `coverage_severe` is n_deployed / n_severe. Pooling the
    # per-stratum rates with n_severe weights is therefore wrong unless coverage happens
    # to be identical across strata, which nothing here guarantees. The correct weight is
    # n_deployed = coverage_severe * n_severe, which this script already collects.
    # `agg_across_seeds` returns each key as a nested {"mean","p5","p95","n_seeds"}
    # dict, and adds only two scalars (n_severe_mean, status). Reading a flat
    # "<key>_mean" silently yields None for every rate, which is how an earlier
    # version of this block produced an all-null headline comparison.
    def _mean(cell, key):
        v = cell.get(key)
        if isinstance(v, dict):
            return v.get("mean")
        if v is None:
            return None
        raise TypeError(f"unexpected shape for {key}: {type(v).__name__} -- "
                        "agg_across_seeds is expected to nest per-key statistics")

    def _cell(name):
        c = strata.get(name, {})
        n_sev = c.get("n_severe_mean")
        cov = _mean(c, "coverage_severe")
        n_dep = (n_sev * cov) if (n_sev is not None and cov is not None) else None
        return _mean(c, "violation_rate_deployed"), n_sev, cov, n_dep

    vr_rrt, n_rrt, cov_rrt, dep_rrt = _cell("prior_renal_replacement")
    vr_kept, n_kept, cov_kept, dep_kept = _cell("no_prior_renal_replacement")
    vr_unk, n_unk, cov_unk, dep_unk = _cell("prior_status_unobservable")

    surviving = [(v, d) for v, d in ((vr_kept, dep_kept), (vr_unk, dep_unk))
                 if v is not None and d]
    vr_surviving = (sum(v * d for v, d in surviving) / sum(d for _, d in surviving)
                    if surviving else None)
    # what the same pooling would have given with the wrong weight, so the size of the
    # error is on the record rather than silently corrected
    surv_sev = [(v, n) for v, n in ((vr_kept, n_kept), (vr_unk, n_unk))
                if v is not None and n]
    vr_surviving_wrong_weight = (
        sum(v * n for v, n in surv_sev) / sum(n for _, n in surv_sev) if surv_sev else None)

    payload = {
        "prereg_tag": "prereg-v1.11",
        "analysis_status": (
            "EXPLORATORY, post hoc, selection-limited. Inherits every caveat of blocks G "
            "and H of 10_prior_ckd_sensitivity.py, including differential observability of "
            "prior history. No registered threshold is changed; only the grouping variable "
            "is new."),
        "what_this_answers": (
            "Block H showed the LABEL disagreement survives deleting the "
            "prior-renal-replacement stratum. This arm asks the same question of the "
            "MODEL statistic the headline actually reports: the deployed violation rate."),
        "strata_are_a_partition": True,
        "stratum_definition": (
            "prior_renal_replacement: a dialysis or transplant status code in an admission "
            "discharged before the audited one. no_prior_renal_replacement: earlier "
            "admission exists, no such code. prior_status_unobservable: no earlier MIMIC "
            "admission, so prior status cannot be read. These three cover every admission "
            "exactly once and are NOT the nested triple used for the block G reading."),
        "reused_from_05_subgroup_audit": [
            "_subject_split", "per_split_subgroup_stats", "agg_across_seeds"],
        "per_stratum": strata,
        "restricted_certificate": restricted_certificate,
        "headline_comparison": {
            "pooling_weight": (
                "n_deployed = coverage_severe * n_severe. `violation_rate_deployed` has "
                "n_deployed as its denominator, so pooling per-stratum rates by n_severe "
                "would only be exact if coverage were identical across strata."),
            "violation_rate_prior_renal_replacement": vr_rrt,
            "violation_rate_surviving_cases": (round(vr_surviving, 4)
                                               if vr_surviving is not None else None),
            "n_severe_mean_prior_renal_replacement": n_rrt,
            "n_severe_mean_surviving": (round(sum(n for n in (n_kept, n_unk) if n), 1)
                                        if (n_kept or n_unk) else None),
            "n_deployed_mean_prior_renal_replacement": (round(dep_rrt, 1)
                                                        if dep_rrt else None),
            "n_deployed_mean_surviving": (round(sum(d for _, d in surviving), 1)
                                          if surviving else None),
            "coverage_severe_mean_by_stratum": {
                "prior_renal_replacement": cov_rrt,
                "no_prior_renal_replacement": cov_kept,
                "prior_status_unobservable": cov_unk},
            "same_pooling_with_n_severe_weight": (
                round(vr_surviving_wrong_weight, 4)
                if vr_surviving_wrong_weight is not None else None),
            "weight_choice_matters_by": (
                round(abs(vr_surviving - vr_surviving_wrong_weight), 4)
                if (vr_surviving is not None and vr_surviving_wrong_weight is not None)
                else None),
        },
    }
    # refuse to write a headline comparison that is silently all-null: this arm
    # exists to produce one number, so an unpopulated payload is a failure, not a
    # result
    hc = payload["headline_comparison"]
    missing = [k for k in ("violation_rate_prior_renal_replacement",
                           "violation_rate_surviving_cases",
                           "n_deployed_mean_surviving") if hc.get(k) is None]
    # the certificate block was once computed and printed but never attached to the
    # payload -- a result that exists only in a log is not a result
    if "certificate_pass" not in payload.get("restricted_certificate", {}):
        missing.append("restricted_certificate.certificate_pass")
    if missing:
        raise SystemExit(
            "headline comparison is unpopulated: " + ", ".join(missing) +
            "\nPer-stratum keys present: "
            + ", ".join(sorted(strata.get(STRATUM_ORDER[0], {}))) +
            "\nThis usually means agg_across_seeds changed its return shape.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "violation_rate_by_renal_history.json").write_text(json.dumps(payload, indent=1))
    print(f"\n[headline] violation rate: prior-RRT {vr_rrt}, surviving "
          f"{payload['headline_comparison']['violation_rate_surviving_cases']}", flush=True)
    print(f"Saved -> {outdir/'violation_rate_by_renal_history.json'}", flush=True)


if __name__ == "__main__":
    main()
