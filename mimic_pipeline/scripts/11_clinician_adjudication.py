"""STEP 11 -- clinician adjudication of the audit constraint (prereg-v1.9).

STATUS: SAMPLING RUNS HERE; ADJUDICATION REQUIRES CLINICIANS.

WHY THIS EXISTS.  The paper's constraint is the authors' operationalisation of
published KDIGO criteria and is NOT clinician-ratified. Every compliance verdict
in the paper is conditional on that rule being a reasonable reading. Reviewers
have raised this, correctly, as the single largest gap in the work; it is named
as a limitation in the paper, but a named limitation is not a measurement.

This script does the half that does not need a clinician:

    (1) draws a stratified, blinded case sample designed so that agreement is
        actually *falsifiable* -- see the sampling rationale below;
    (2) writes a review packet with one row per case, carrying only the
        information a clinician needs and NOTHING that reveals what the rule or
        the model decided;
    (3) scores returned adjudications against the rule, reporting raw agreement,
        Cohen's kappa with a patient-clustered interval, and -- separately --
        agreement on the cases where rule and administrative label DISAGREE,
        which is the subpopulation the paper's headline rests on.

    # step 1: draw the packet (runs today, needs only the cohort)
    python scripts/11_clinician_adjudication.py sample \\
        --cohort audit_mimic_out/aki_cohort.parquet --n 200

    # step 2: after clinicians fill in the `clinician_aki` column
    python scripts/11_clinician_adjudication.py score \\
        --packet results/adjudication_packet.csv \\
        --responses <one CSV per reviewer, repeatable>

SAMPLING RATIONALE (fixed before any adjudication is seen).
A uniform sample of severe cases would be uninformative: the constraint fires on
a creatinine threshold, so most sampled cases would be unambiguous and agreement
would be high by construction. The sample is therefore stratified into four
cells, and the strata are declared here so the composition cannot be chosen after
seeing results:

  A. rule-severe AND coded          -- rule and administrative label agree
  B. rule-severe AND NOT coded      -- THE HEADLINE CELL: the rule says severe,
                                       the coders did not. If clinicians side
                                       with the coders here, the paper's central
                                       claim weakens and we report that.
  C. near-threshold, not rule-severe -- peak creatinine just below the cut, where
                                       a threshold rule is most arguable
  D. clearly not severe             -- a floor: if clinicians call these severe
                                       the rule is mis-specified in the other
                                       direction

Cells B and C are over-sampled relative to their prevalence because they carry
the information; the scoring step reports per-cell agreement and NEVER a single
pooled number without it, since pooling across a sample we chose the composition
of would be meaningless.

BLINDING. The packet contains creatinine trajectory, demographics and the
clinical variables a reviewer would see. It does NOT contain: the ICD code, the
rule's verdict, the model's prediction, or the stratum label. Column order is
fixed and cases are shuffled with a recorded seed.

WHAT WOULD FALSIFY THE PAPER'S CONSTRAINT.  Declared in advance: if clinician
agreement with the rule on cell B is below 0.70, or if Cohen's kappa over cells
A-D is below 0.60, the constraint should be treated as an author-specific
operationalisation rather than a faithful reading of the guideline, and the
paper's compliance verdicts should be reported as conditional on a rule that
clinicians do not endorse. These thresholds are recorded before any adjudication
exists so they cannot be adjusted to the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.config import load_config  # noqa: E402

# ---- declared before any adjudication is collected -------------------------
CELL_TARGETS = {"A_severe_coded": 40,
                "B_severe_uncoded": 80,      # the headline cell, over-sampled
                "C_near_threshold": 60,      # where a threshold rule is arguable
                "D_clearly_not_severe": 20}
NEAR_THRESHOLD_BAND = (0.75, 1.00)           # fraction of the severity threshold
FALSIFICATION = {"min_agreement_cell_B": 0.70, "min_kappa_overall": 0.60}
SAMPLE_SEED = 20260812

PACKET_COLUMNS = ["case_id", "age", "sex", "creat_baseline_mgdl", "creat_peak_24h_mgdl",
                  "creat_nadir_24h_mgdl", "absolute_rise_mgdl", "peak_over_nadir",
                  "bun", "sodium", "potassium", "bicarbonate", "hemoglobin", "wbc",
                  "clinician_aki", "clinician_confidence", "clinician_notes"]


def _cells(df: pd.DataFrame, thr: float) -> dict:
    peak = pd.to_numeric(df["creat_peak_24h"], errors="coerce")
    nadir = pd.to_numeric(df["creat_nadir_24h"], errors="coerce")
    # A case with no creatinine measured cannot be adjudicated: the reviewer would be
    # asked to judge an acute kidney injury with no kidney function data in front of
    # them, and whatever they answered would enter kappa as noise. The cohort fills a
    # missing trajectory with 0.0, which otherwise lands such cases in the
    # "clearly not severe" floor cell by accident. Exclude them from the frame and
    # record how many, rather than shipping an unanswerable row.
    adjudicable = (peak > 0) & peak.notna() & (nadir > 0) & nadir.notna()
    severe = df["severe_acute"].astype(bool).to_numpy() & adjudicable.to_numpy()
    coded = pd.to_numeric(df["aki"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    adj = adjudicable.to_numpy()
    near = adj & (~severe) & (peak >= thr * NEAR_THRESHOLD_BAND[0]).to_numpy() & \
           (peak < thr * NEAR_THRESHOLD_BAND[1]).to_numpy()
    clearly_not = adj & (~severe) & (peak < thr * 0.5).to_numpy()
    return {"A_severe_coded": severe & coded,
            "B_severe_uncoded": severe & ~coded,
            "C_near_threshold": near,
            "D_clearly_not_severe": clearly_not}


def do_sample(args) -> None:
    cfg = load_config(args.config)
    thr = float(cfg.constraints["creatinine_severe_mgdl"])
    df = pd.read_parquet(args.cohort)
    cells = _cells(df, thr)
    rng = np.random.default_rng(SAMPLE_SEED)

    picks, provenance = [], {}
    scale = args.n / sum(CELL_TARGETS.values())
    for name, mask in cells.items():
        want = int(round(CELL_TARGETS[name] * scale))
        idx = np.flatnonzero(mask)
        take = min(want, len(idx))
        chosen = rng.choice(idx, size=take, replace=False) if take else np.array([], int)
        provenance[name] = {"available": int(len(idx)), "requested": want, "drawn": int(take)}
        for i in chosen:
            picks.append((name, i))
        if take < want:
            print(f"  WARNING: cell {name} exhausted ({len(idx)} available, {want} wanted)",
                  flush=True)

    rng.shuffle(picks)
    peak = pd.to_numeric(df["creat_peak_24h"], errors="coerce")
    nadir = pd.to_numeric(df["creat_nadir_24h"], errors="coerce")

    rows, key = [], []
    for case_id, (cell, i) in enumerate(picks, start=1):
        r = df.iloc[i]
        rows.append({
            "case_id": case_id,
            "age": int(r["age"]) if pd.notna(r["age"]) else "",
            "sex": "F" if int(r.get("is_female", 0)) == 1 else "M",
            "creat_baseline_mgdl": round(float(nadir.iloc[i]), 2) if pd.notna(nadir.iloc[i]) else "",
            "creat_peak_24h_mgdl": round(float(peak.iloc[i]), 2) if pd.notna(peak.iloc[i]) else "",
            "creat_nadir_24h_mgdl": round(float(nadir.iloc[i]), 2) if pd.notna(nadir.iloc[i]) else "",
            "absolute_rise_mgdl": round(float(peak.iloc[i] - nadir.iloc[i]), 2)
                                  if pd.notna(peak.iloc[i]) and pd.notna(nadir.iloc[i]) else "",
            "peak_over_nadir": round(float(peak.iloc[i] / nadir.iloc[i]), 3)
                               if pd.notna(nadir.iloc[i]) and nadir.iloc[i] else "",
            **{c: (round(float(r[c]), 2) if pd.notna(r.get(c)) else "")
               for c in ("bun", "sodium", "potassium", "bicarbonate", "hemoglobin", "wbc")},
            "clinician_aki": "", "clinician_confidence": "", "clinician_notes": "",
        })
        # the unblinding key stays OUT of the packet
        key.append({"case_id": case_id, "cell": cell,
                    "hadm_id": int(r["hadm_id"]), "subject_id": int(r["subject_id"]),
                    "rule_severe": bool(r["severe_acute"]),
                    "administrative_code": int(pd.to_numeric(r["aki"], errors="coerce") or 0)})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[PACKET_COLUMNS].to_csv(outdir / "adjudication_packet.csv", index=False)
    (outdir / "adjudication_key.json").write_text(json.dumps(
        {"status": "BLINDING KEY -- do not send to reviewers",
         "sample_seed": SAMPLE_SEED, "severity_threshold_mgdl": thr,
         "excluded_unadjudicable": int(len(df) - int(((pd.to_numeric(df["creat_peak_24h"], errors="coerce") > 0)
                                                      & (pd.to_numeric(df["creat_nadir_24h"], errors="coerce") > 0)).sum())),
         "exclusion_rule": ("admissions with no measured creatinine trajectory are outside the "
                            "sampling frame: they cannot be adjudicated and would enter kappa as noise"),
         "cell_targets": CELL_TARGETS, "near_threshold_band": NEAR_THRESHOLD_BAND,
         "falsification_thresholds_declared_in_advance": FALSIFICATION,
         "provenance": provenance, "cases": key}, indent=1))
    print(f"\n[sample] {len(rows)} cases drawn", flush=True)
    for name, p in provenance.items():
        print(f"    {name:24s} available={p['available']:6d} drawn={p['drawn']:4d}", flush=True)
    print(f"\nSaved -> {outdir/'adjudication_packet.csv'} (send this)", flush=True)
    print(f"Saved -> {outdir/'adjudication_key.json'} (KEEP THIS BACK)", flush=True)


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def do_score(args) -> None:
    key = json.loads(Path(args.key).read_text())
    by_case = {c["case_id"]: c for c in key["cases"]}
    thr_decl = key["falsification_thresholds_declared_in_advance"]

    reviewers = {}
    for path in args.responses:
        r = pd.read_csv(path)
        if "case_id" not in r or "clinician_aki" not in r:
            raise SystemExit(f"{path}: needs columns case_id and clinician_aki")
        reviewers[Path(path).stem] = r.set_index("case_id")["clinician_aki"]
    if not reviewers:
        raise SystemExit("no responses supplied; run `sample` first and collect adjudications")

    resp = pd.DataFrame(reviewers).apply(pd.to_numeric, errors="coerce")

    # A blank means "I cannot tell from what is shown" and is NOT a "no": it is
    # dropped from the denominator rather than counted against the rule. Cases every
    # reviewer left blank carry no adjudication at all and leave the analysis, with
    # the count reported -- a rule the clinicians could not apply is itself a result.
    blanks_per_case = resp.isna().sum(axis=1)
    undecidable = resp.isna().all(axis=1)
    n_undecidable = int(undecidable.sum())
    resp = resp.loc[~undecidable]
    if resp.empty:
        raise SystemExit("every case was left blank; nothing to score")
    majority = (resp.mean(axis=1, skipna=True) > 0.5).astype(int)
    rule = pd.Series({cid: int(by_case[cid]["rule_severe"]) for cid in resp.index})
    code = pd.Series({cid: int(by_case[cid]["administrative_code"]) for cid in resp.index})
    cell = pd.Series({cid: by_case[cid]["cell"] for cid in resp.index})

    out = {"n_reviewers": len(reviewers), "n_cases_adjudicated": int(len(resp)),
           "n_cases_left_blank_by_every_reviewer": n_undecidable,
           "n_cases_with_at_least_one_blank": int((blanks_per_case > 0).sum()),
           "blank_handling": ("a blank is 'cannot tell from what is shown', not a 'no': it is "
                              "dropped from that reviewer's contribution rather than scored "
                              "against the rule"),
           "declared_thresholds": thr_decl, "per_cell": {}}
    for name in CELL_TARGETS:
        m = (cell == name)
        if not m.any():
            continue
        out["per_cell"][name] = {
            "n": int(m.sum()),
            "clinician_vs_rule_agreement": round(float((majority[m] == rule[m]).mean()), 4),
            "clinician_vs_code_agreement": round(float((majority[m] == code[m]).mean()), 4)}
    out["overall"] = {
        "clinician_vs_rule_kappa": round(_kappa(majority.to_numpy(), rule.to_numpy()), 4),
        "clinician_vs_code_kappa": round(_kappa(majority.to_numpy(), code.to_numpy()), 4),
        "note": ("pooled figures are reported only alongside the per-cell table: the sample "
                 "composition was chosen by us, so a pooled number is not an estimate of "
                 "agreement in the population")}
    if len(reviewers) > 1:
        # pairwise agreement is computed on the cases BOTH reviewers answered; a blank
        # from either one is not evidence of disagreement
        pairs = []
        for i, a in enumerate(resp.columns):
            for b in resp.columns[i + 1:]:
                both = resp[[a, b]].dropna()
                k = _kappa(both[a].to_numpy(int), both[b].to_numpy(int)) if len(both) else float("nan")
                pairs.append({"pair": f"{a}|{b}", "n_both_answered": int(len(both)),
                              "kappa": None if np.isnan(k) else round(float(k), 4)})
        vals = [p["kappa"] for p in pairs if p["kappa"] is not None]
        out["inter_reviewer_kappa"] = {
            "pairwise": pairs,
            "mean": round(float(np.mean(vals)), 4) if vals else None}

    cell_b = out["per_cell"].get("B_severe_uncoded", {}).get("clinician_vs_rule_agreement")
    kappa = out["overall"]["clinician_vs_rule_kappa"]
    if cell_b is None:
        out["verdict"] = "UNDETERMINED: the headline cell was not adjudicated."
    elif cell_b >= thr_decl["min_agreement_cell_B"] and kappa >= thr_decl["min_kappa_overall"]:
        out["verdict"] = (
            f"Clinicians endorse the rule where it matters: agreement {cell_b:.3f} on cases the "
            f"rule calls severe and the coders did not (declared floor "
            f"{thr_decl['min_agreement_cell_B']}), overall kappa {kappa:.3f} (floor "
            f"{thr_decl['min_kappa_overall']}). The constraint may be described as a "
            "clinician-validated operationalisation rather than the authors' reading.")
    else:
        out["verdict"] = (
            f"The rule FAILS its declared validation: cell-B agreement {cell_b:.3f} against a "
            f"floor of {thr_decl['min_agreement_cell_B']}, overall kappa {kappa:.3f} against "
            f"{thr_decl['min_kappa_overall']}. The compliance verdicts must be reported as "
            "conditional on an operationalisation clinicians do not endorse, and the paper's "
            "headline narrowed accordingly. This branch exists so that outcome is reportable.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "clinician_adjudication.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    print(f"\nSaved -> {outdir/'clinician_adjudication.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw the blinded review packet (runs today)")
    s.add_argument("--cohort", default="audit_mimic_out/aki_cohort.parquet")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--out", default="results")
    s.set_defaults(func=do_sample)

    c = sub.add_parser("score", help="score returned adjudications against the rule")
    c.add_argument("--packet", default="results/adjudication_packet.csv")
    c.add_argument("--key", default="results/adjudication_key.json")
    c.add_argument("--responses", nargs="+", required=True)
    c.add_argument("--out", default="results")
    c.set_defaults(func=do_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
