"""STEP 8 -- is the racial gap in label validity case-mix confounding? (prereg-v1.5)

WHY THIS EXISTS.  Step 05b established that among guideline-severe cases the
fraction carrying an ICD AKI code differs by race (White 0.44, Black 0.28).
Reviewers correctly objected that this is not yet evidence about CODING
PRACTICE, because the severe population is not homogeneous: step 05's error
taxonomy shows confident denials concentrate in admissions that were ALREADY
ELEVATED at arrival (denial rate 0.65) versus an acute rise from near-normal
(0.26), and the prevalence of that presentation differs by race for
epidemiological reasons (CKD burden). The observed gap could therefore be
case-mix, not differential coding.

This script answers the objection the only way it can be answered: by asking
whether the gap SURVIVES INSIDE a trajectory stratum. Three analyses, in
increasing strength:

  (A) Stratified agreement.  Agreement by race WITHIN each creatinine-trajectory
      stratum (the same A1/A2/A3 axis as the error taxonomy). If the White-Black
      gap persists inside strata, case-mix does not explain it.

  (B) Direct standardisation.  Each race group's agreement re-weighted to the
      trajectory distribution of the WHOLE severe cohort. This is the single
      number that answers "what would this group's agreement be if it had the
      same case-mix as everyone else?" Reported with a patient-level cluster
      bootstrap.

  (C) Decomposition.  How much of the crude White-Black difference is
      attributable to differing trajectory mix versus differing within-stratum
      agreement.

PRE-REGISTRATION STATUS.  This analysis was NOT pre-registered: it was added in
response to review, after the crude gap was known. It is therefore labelled
EXPLORATORY throughout and in its output file. It does not change any registered
threshold, and it can only weaken or qualify the crude claim -- never strengthen
it beyond what the data support. The direction of the result is not chosen in
advance; whatever comes out is what the paper will say.

Usage:
    python scripts/08_stratified_label_validity.py --cohort <path.parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auditnesy_mimic.config import load_config  # noqa: E402
from importlib import import_module  # noqa: E402

_sg = import_module("05_subgroup_audit")
race_bucket = _sg.race_bucket

# the trajectory axis, identical to the error taxonomy in 05 (prereg-v1.4)
TRAJ_BANDS = [
    ("A1_elevated_at_arrival", lambda nadir: nadir >= 4.0),
    ("A3_rise_from_elevated_base", lambda nadir: (nadir >= 2.0) & (nadir < 4.0)),
    ("A2_rise_from_near_normal", lambda nadir: nadir < 2.0),
]

MIN_CELL_N = 30   # below this a stratum-by-race cell is reported but marked thin

# ---------------------------------------------------------------------------
# The chronicity axis (added v1.5, block D).
#
# WHY A PROXY AND NOT A CKD CODE.  Reviewers noted that `A1_elevated_at_arrival`
# is clinically heterogeneous: it mixes a patient arriving with AKI, a patient
# with long-standing CKD, and AKI-on-CKD. That objection is correct, and it is
# sharper for this paper than for most, because the constraint's own first
# pre-registered correction was made precisely because an absolute-creatinine
# rule conflated AKI with CKD.
#
# The obvious fix -- stratify on a CKD diagnosis code -- has two problems here,
# one fatal and one disqualifying:
#
#  1. A CKD code on the CURRENT admission is NOT a decision-time baseline
#     covariate. MIMIC-IV `diagnoses_icd` rows are assigned after discharge from
#     the signed record, so a same-admission CKD code is produced by the very
#     coding process under audit. Conditioning the coding outcome on another
#     product of the same process would not estimate what it appears to.
#  2. A code from a PRIOR admission is defensible in principle, but the released
#     cohort derivative carries no diagnosis codes other than the AKI label, and
#     the raw `diagnoses_icd` table is not available in this environment. It
#     cannot be computed here, and we do not pretend otherwise.
#
# What IS computable from the cohort is a within-window chronicity proxy: among
# cases already elevated at arrival, the ratio of 24h peak to 24h nadir
# separates a flat-high creatinine (consistent with stable chronic impairment)
# from a rising one (consistent with an acute insult on an elevated baseline).
# It is a weak proxy for CKD and we report it as one; the cut is a round number
# chosen before any agreement was computed on either side of it, and block D
# reports the sensitivity of the verdict to moving it.
CHRONICITY_RATIO_CUT = 1.15
CHRONICITY_CUT_SWEEP = (1.10, 1.15, 1.20, 1.30)


def trajectory_band(nadir: np.ndarray) -> np.ndarray:
    out = np.full(len(nadir), "unclassified", dtype=object)
    for name, pred in TRAJ_BANDS:
        out[pred(nadir) & (out == "unclassified")] = name
    return out


def _cluster_index_draws(clusters, B=400, seed=0):
    """Yield resampled row indices, resampling whole patients with replacement.

    Same resampling scheme as `cluster_bootstrap` below, exposed as indices so a
    statistic that cannot be written as a closure over one array (e.g. refitting a
    model) uses identical clustering rather than a second, different assumption.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([idx_by_cluster[c] for c in pick])


def cluster_bootstrap(values, clusters, B=2000, seed=0, stat=np.mean):
    """Patient-level cluster bootstrap on an arbitrary statistic."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        out[b] = stat(idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr = float(cfg.constraints["creatinine_severe_mgdl"])

    df = pd.read_parquet(args.cohort)
    # Use the SAME severe mask as 05b (`severe_acute`), not a bare peak threshold:
    # `severe_acute` additionally requires the acute component, so a bare
    # `peak >= thr` mask selects a ~3x larger and materially different population
    # and would not be comparable to the paper's n=5,954.
    if "severe_acute" not in df.columns:
        raise SystemExit("cohort lacks `severe_acute`; rebuild it with auditnesy_mimic.cohort")
    severe = df["severe_acute"].to_numpy(bool)
    nadir = pd.to_numeric(df["creat_nadir_24h"], errors="coerce")

    sev = df.loc[severe].copy()
    sev["_race"] = [race_bucket(r) for r in sev["race"].astype(str)]
    sev["_traj"] = trajectory_band(nadir[severe].fillna(0.0).to_numpy())
    sev["_coded"] = pd.to_numeric(sev["aki"], errors="coerce").fillna(0).astype(int).to_numpy()
    sev["_subj"] = sev["subject_id"].to_numpy()

    print(f"[strat] severe cases n={len(sev)}  threshold={thr} mg/dL", flush=True)
    print("[strat] trajectory mix overall:", flush=True)
    for t, n in sev["_traj"].value_counts().items():
        print(f"          {t:28s} n={n:5d} ({100*n/len(sev):4.1f}%)", flush=True)

    # ---------- (A) agreement within trajectory stratum, by race ----------
    focus = ["White", "Black", "Hispanic/Latino", "Asian"]
    strat = {}
    for t in sev["_traj"].unique():
        blk = sev[sev["_traj"] == t]
        cells = {}
        for g in focus:
            c = blk[blk["_race"] == g]
            if len(c) == 0:
                continue
            vals = c["_coded"].to_numpy(float)
            entry = {"n": int(len(c)), "agreement": round(float(vals.mean()), 4),
                     "thin_cell": bool(len(c) < MIN_CELL_N)}
            if len(c) >= MIN_CELL_N:
                draws = cluster_bootstrap(
                    vals, c["_subj"].to_numpy(), B=args.B, seed=0,
                    stat=lambda idx, v=vals: v[idx].mean())
                entry["ci95"] = [round(float(np.percentile(draws, 2.5)), 4),
                                 round(float(np.percentile(draws, 97.5)), 4)]
            cells[g] = entry
        # within-stratum White - Black gap with a cluster bootstrap on the DIFFERENCE
        if {"White", "Black"} <= set(cells) and not (cells["White"]["thin_cell"] or cells["Black"]["thin_cell"]):
            sub = blk[blk["_race"].isin(["White", "Black"])]
            v = sub["_coded"].to_numpy(float)
            is_w = (sub["_race"] == "White").to_numpy()
            draws = cluster_bootstrap(
                v, sub["_subj"].to_numpy(), B=args.B, seed=1,
                stat=lambda idx, v=v, w=is_w: (v[idx][w[idx]].mean() if w[idx].any() else np.nan)
                                              - (v[idx][~w[idx]].mean() if (~w[idx]).any() else np.nan))
            draws = draws[~np.isnan(draws)]
            cells["_white_minus_black"] = {
                "point": round(float(cells["White"]["agreement"] - cells["Black"]["agreement"]), 4),
                "ci95": [round(float(np.percentile(draws, 2.5)), 4),
                         round(float(np.percentile(draws, 97.5)), 4)],
                "excludes_zero": bool(np.percentile(draws, 2.5) > 0 or np.percentile(draws, 97.5) < 0),
            }
        strat[t] = cells
        line = "  ".join(f"{g}={cells[g]['agreement']:.3f}(n={cells[g]['n']})"
                         for g in focus if g in cells)
        print(f"[strat] {t:28s} {line}", flush=True)

    # ---------- (B) direct standardisation to the whole-cohort trajectory mix ----------
    overall_mix = sev["_traj"].value_counts(normalize=True).to_dict()
    std = {}
    for g in focus:
        c = sev[sev["_race"] == g]
        if len(c) == 0:
            continue

        def _std(idx, cc=c, mix=overall_mix):
            sub = cc.iloc[idx]
            num = den = 0.0
            for t, w in mix.items():
                cell = sub[sub["_traj"] == t]
                if len(cell) == 0:
                    continue                 # stratum unrepresented in this draw
                num += w * cell["_coded"].mean()
                den += w
            return num / den if den > 0 else np.nan

        point = _std(np.arange(len(c)))
        draws = cluster_bootstrap(None, c["_subj"].to_numpy(), B=args.B, seed=2, stat=_std)
        draws = draws[~np.isnan(draws)]
        std[g] = {"crude": round(float(c["_coded"].mean()), 4),
                  "standardised": round(float(point), 4),
                  "ci95": [round(float(np.percentile(draws, 2.5)), 4),
                           round(float(np.percentile(draws, 97.5)), 4)],
                  "n": int(len(c))}
        print(f"[std]   {g:18s} crude={std[g]['crude']:.4f} -> standardised={std[g]['standardised']:.4f} "
              f"CI{std[g]['ci95']}", flush=True)

    # ---------- (C) decomposition of the crude White-Black difference ----------
    decomp = None
    if {"White", "Black"} <= set(std):
        W = sev[sev["_race"] == "White"]
        B_ = sev[sev["_race"] == "Black"]
        traj = list(overall_mix)
        pW = {t: (W["_traj"] == t).mean() for t in traj}
        pB = {t: (B_["_traj"] == t).mean() for t in traj}
        aW = {t: (W.loc[W["_traj"] == t, "_coded"].mean() if (W["_traj"] == t).any() else np.nan) for t in traj}
        aB = {t: (B_.loc[B_["_traj"] == t, "_coded"].mean() if (B_["_traj"] == t).any() else np.nan) for t in traj}
        crude_gap = float(W["_coded"].mean() - B_["_coded"].mean())
        # Kitagawa decomposition: composition (mix) term + rate (within-stratum) term
        comp = float(np.nansum([(pW[t] - pB[t]) * (aW[t] + aB[t]) / 2 for t in traj]))
        rate = float(np.nansum([(aW[t] - aB[t]) * (pW[t] + pB[t]) / 2 for t in traj]))
        decomp = {
            "crude_gap": round(crude_gap, 4),
            "composition_term_case_mix": round(comp, 4),
            "rate_term_within_stratum": round(rate, 4),
            "residual": round(crude_gap - comp - rate, 6),
            "share_attributable_to_case_mix": round(comp / crude_gap, 4) if crude_gap else None,
            "trajectory_mix_white": {t: round(float(pW[t]), 4) for t in traj},
            "trajectory_mix_black": {t: round(float(pB[t]), 4) for t in traj},
        }
        print(f"[decomp] crude gap {crude_gap:.4f} = case-mix {comp:.4f} + within-stratum {rate:.4f}",
              flush=True)
        print(f"[decomp] share attributable to case-mix: "
              f"{100*decomp['share_attributable_to_case_mix']:.1f}%", flush=True)

    # ---------- (D) chronicity proxy within the elevated-at-arrival stratum ----------
    # Does the gap survive inside a stratum that also holds chronicity roughly fixed?
    A1 = sev[sev["_traj"] == "A1_elevated_at_arrival"].copy()
    peak_A1 = pd.to_numeric(A1["creat_peak_24h"], errors="coerce")
    nadir_A1 = pd.to_numeric(A1["creat_nadir_24h"], errors="coerce")
    A1["_ratio"] = (peak_A1 / nadir_A1.replace(0, np.nan)).to_numpy()

    chron = {}
    for cut in CHRONICITY_CUT_SWEEP:
        bands = {"flat_high_chronic_like": A1["_ratio"] < cut,
                 "rising_acute_on_elevated": A1["_ratio"] >= cut}
        entry = {}
        for bname, mask in bands.items():
            blk = A1.loc[mask]
            cells = {"n": int(len(blk)),
                     "agreement": round(float(blk["_coded"].mean()), 4) if len(blk) else None}
            for g in ("White", "Black"):
                cg = blk[blk["_race"] == g]
                cells[g] = {"n": int(len(cg)),
                            "agreement": round(float(cg["_coded"].mean()), 4) if len(cg) else None,
                            "thin_cell": bool(len(cg) < MIN_CELL_N)}
            W = blk[blk["_race"] == "White"]
            B_ = blk[blk["_race"] == "Black"]
            if len(W) >= MIN_CELL_N and len(B_) >= MIN_CELL_N:
                pair = pd.concat([W, B_])
                is_w = (pair["_race"] == "White").to_numpy()
                coded = pair["_coded"].to_numpy(float)

                def _gap(idx, is_w=is_w, coded=coded):
                    w, b = coded[idx][is_w[idx]], coded[idx][~is_w[idx]]
                    if len(w) == 0 or len(b) == 0:
                        return np.nan
                    return float(w.mean() - b.mean())

                draws = cluster_bootstrap(None, pair["_subj"].to_numpy(),
                                          B=args.B, seed=3, stat=_gap)
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                cells["_white_minus_black"] = {
                    "point": round(float(W["_coded"].mean() - B_["_coded"].mean()), 4),
                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
            entry[bname] = cells
        chron[f"ratio_cut_{cut}"] = entry

    prim = chron[f"ratio_cut_{CHRONICITY_RATIO_CUT}"]
    print(f"\n[chron] within A1 (elevated at arrival), chronicity proxy at "
          f"peak/nadir < {CHRONICITY_RATIO_CUT}:", flush=True)
    for bname, cells in prim.items():
        g = cells.get("_white_minus_black")
        print(f"          {bname:26s} n={cells['n']:5d} agree={cells['agreement']}"
              + (f"  W-B gap={g['point']:+.4f} CI{g['ci95']} excl0={g['excludes_zero']}"
                 if g else "  (thin cell, not tested)"), flush=True)
    gaps = [chron[f"ratio_cut_{c}"][b].get("_white_minus_black")
            for c in CHRONICITY_CUT_SWEEP for b in prim]
    gaps = [g for g in gaps if g]
    print(f"[chron] cut sweep {CHRONICITY_CUT_SWEEP}: "
          f"{sum(g['excludes_zero'] for g in gaps)}/{len(gaps)} testable cells have a "
          f"CI excluding zero", flush=True)

    # ---------- (E) is the prior-admission record differentially available? ----------
    # A reviewer's proposed fix -- stratify on CKD documented in a PRIOR MIMIC
    # admission -- would itself be confounded if prior-admission history is
    # differentially observable by race. That is checkable from the cohort even
    # though the CKD codes are not, so we check it and record the answer as a
    # constraint on the analysis we could not run.
    full = pd.read_parquet(args.cohort)[["subject_id", "hadm_id", "admittime", "race"]]
    full = full.sort_values(["subject_id", "admittime"])
    full["_n_prior"] = full.groupby("subject_id").cumcount()
    prior = full.set_index("hadm_id")["_n_prior"].reindex(sev["hadm_id"]).to_numpy()
    sev["_n_prior"] = prior
    avail = {}
    for g in ("White", "Black", "Hispanic/Latino", "Asian"):
        d = sev[sev["_race"] == g]
        if not len(d):
            continue
        avail[g] = {"n_severe": int(len(d)),
                    "share_with_prior_admission": round(float((d["_n_prior"] > 0).mean()), 4),
                    "median_n_prior_admissions": float(np.median(d["_n_prior"]))}
    spread = [v["share_with_prior_admission"] for v in avail.values()]
    avail["_verdict"] = {
        "max_minus_min_share": round(float(max(spread) - min(spread)), 4),
        "differentially_observable": bool(max(spread) - min(spread) > 0.05),
        "interpretation": (
            "Prior-admission history is differentially observable across race buckets, so "
            "'no CKD documented in a prior MIMIC admission' does not mean 'no CKD' equally "
            "across groups. A prior-code stratification would therefore carry its own "
            "differential-measurement problem, in the direction of the observed gap. This is "
            "recorded as a limitation of the analysis we could not run, not as a reason it "
            "would be uninformative."
            if max(spread) - min(spread) > 0.05 else
            "Prior-admission availability is similar across race buckets.")}
    print(f"\n[avail] prior-admission observability by race "
          f"(spread {avail['_verdict']['max_minus_min_share']:.3f}, "
          f"differential={avail['_verdict']['differentially_observable']}):", flush=True)
    for g, v in avail.items():
        if g.startswith("_"):
            continue
        print(f"          {g:16s} n={v['n_severe']:5d} "
              f"has prior admission {100*v['share_with_prior_admission']:5.1f}%", flush=True)

    # ---------- (F) regression robustness: one model, both adjustments at once ----------
    # A different functional form for the same question: holding trajectory band AND the
    # chronicity proxy fixed simultaneously, is race still associated with carrying a code?
    # Reported as a CONDITIONAL ASSOCIATION, never as a causal adjustment. Inference is the
    # same patient-level cluster bootstrap used everywhere else in this script, not a
    # model-based standard error, so the clustering assumption does not change between blocks.
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    reg_df = sev[sev["_race"].isin(["White", "Black"])].copy()
    peak_r = pd.to_numeric(reg_df["creat_peak_24h"], errors="coerce")
    nadir_r = pd.to_numeric(reg_df["creat_nadir_24h"], errors="coerce")
    ratio_r = (peak_r / nadir_r.replace(0, np.nan)).fillna(1.0).to_numpy()
    X = np.column_stack([
        (reg_df["_race"] == "Black").to_numpy(float),                       # 0 race
        (reg_df["_traj"] == "A3_rise_from_elevated_base").to_numpy(float),  # 1 traj A3
        (reg_df["_traj"] == "A2_rise_from_near_normal").to_numpy(float),    # 2 traj A2
        (ratio_r < CHRONICITY_RATIO_CUT).astype(float),                     # 3 chronic-like
    ])
    X = np.column_stack([X, X[:, 0] * X[:, 3]])                             # 4 race x chronic
    y_reg = reg_df["_coded"].to_numpy(int)
    names = ["race_black", "traj_A3", "traj_A2", "chronic_like", "race_black_x_chronic_like"]

    def _fit_coefs(idx):
        Xi, yi = X[idx], y_reg[idx]
        if len(np.unique(yi)) < 2:
            return np.full(X.shape[1], np.nan)
        m = LogisticRegression(max_iter=2000, C=1e6)
        m.fit(Xi, yi)
        return m.coef_.ravel()

    point = _fit_coefs(np.arange(len(y_reg)))
    draws = np.vstack([
        _fit_coefs(idx) for idx in _cluster_index_draws(
            reg_df["_subj"].to_numpy(), B=min(args.B, 400), seed=4)])
    reg = {"model": "logistic, outcome = ICD AKI code present, guideline-severe cases only",
           "groups": "White and Black buckets only", "n": int(len(y_reg)),
           "inference": f"patient-level cluster bootstrap, B={min(args.B, 400)}",
           "is_causal_adjustment": False, "coefficients": {}}
    for j, nm in enumerate(names):
        lo, hi = np.nanpercentile(draws[:, j], [2.5, 97.5])
        reg["coefficients"][nm] = {
            "log_odds": round(float(point[j]), 4),
            "odds_ratio": round(float(np.exp(point[j])), 4),
            "ci95_log_odds": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}
    rb = reg["coefficients"]["race_black"]
    print(f"\n[reg] race_black adjusted for trajectory + chronicity proxy: "
          f"OR={rb['odds_ratio']:.3f} log-odds CI{rb['ci95_log_odds']} "
          f"excl0={rb['excludes_zero']}", flush=True)
    ix = reg["coefficients"]["race_black_x_chronic_like"]
    print(f"[reg] race x chronic-like interaction: log-odds {ix['log_odds']:+.4f} "
          f"CI{ix['ci95_log_odds']} excl0={ix['excludes_zero']}", flush=True)

    payload = {
        "prereg_tag": "prereg-v1.5",
        "analysis_status": "EXPLORATORY -- not pre-registered; added in response to review "
                           "after the crude gap was known. Reported to test whether the crude "
                           "racial gap in label validity survives adjustment for creatinine "
                           "trajectory (case-mix). No registered threshold was changed.",
        "severe_threshold_mgdl": thr,
        "n_severe_cases": int(len(sev)),
        "trajectory_axis": {name: ("nadir >= 4.0" if name == "A1_elevated_at_arrival"
                                   else "2.0 <= nadir < 4.0" if name == "A3_rise_from_elevated_base"
                                   else "nadir < 2.0") for name, _ in TRAJ_BANDS},
        "min_cell_n": MIN_CELL_N,
        "bootstrap_B": args.B,
        "overall_trajectory_mix": {k: round(float(v), 4) for k, v in overall_mix.items()},
        "A_agreement_within_trajectory": strat,
        "B_direct_standardisation": std,
        "C_kitagawa_decomposition": decomp,
        "D_chronicity_proxy_within_A1": {
            "rationale": (
                "A1_elevated_at_arrival mixes arrival AKI, stable CKD and AKI-on-CKD. A CKD "
                "diagnosis code cannot serve as the stratifier here: a same-admission code is "
                "assigned post-discharge and is a product of the coding process under audit, and "
                "prior-admission codes are absent from the released cohort derivative (the raw "
                "diagnoses_icd table is not available in this environment). This is a within-window "
                "proxy: peak/nadir ratio over the first 24h, flat-high being consistent with "
                "stable chronic impairment."),
            "primary_cut": CHRONICITY_RATIO_CUT,
            "cut_sweep": list(CHRONICITY_CUT_SWEEP),
            "is_a_ckd_measurement": False,
            "by_cut": chron},
        "E_prior_admission_observability": avail,
        "F_regression_robustness": reg,
        "not_computable": {
            "ckd_stratification_by_prior_diagnosis_code": (
                "REQUIRES raw MIMIC-IV diagnoses_icd joined to admissions on dischtime < "
                "current admittime, restricted to ICD-9 585.x / ICD-10 N18.x, EXCLUDING the "
                "current admission. Not computable here: the released cohort derivative carries "
                "no diagnosis codes but the AKI label, and credentialed raw-table access is "
                "absent from this environment. Blocks D and E bound what the missing analysis "
                "could show; they do not substitute for it."),
            "comorbidity_and_site_mix": (
                "Charlson/Elixhauser components and hospital identity are not in the cohort "
                "derivative; no adjustment for either is attempted."),
        },
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "stratified_label_validity.json").write_text(json.dumps(payload, indent=1))
    print(f"\nSaved -> {outdir/'stratified_label_validity.json'}", flush=True)


if __name__ == "__main__":
    main()
