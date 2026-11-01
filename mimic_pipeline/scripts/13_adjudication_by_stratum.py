"""STEP 13 -- adjudication cross-tabulated by renal history and trajectory (v1.13).

WHY THIS EXISTS.  The clinician adjudication (`11_clinician_adjudication.py score`)
reports agreement per SAMPLING cell (A-D). A reviewer asked whether that agreement
-- and, as importantly, the 38 "cannot tell" blanks -- concentrate in the
populations the paper's own sensitivity analyses single out: patients with prior
renal replacement (block H strata of `10_prior_ckd_sensitivity.py`) and patients
already severely elevated at arrival (trajectory class A1 of the error taxonomy in
`05_subgroup_audit.py`). The v1.12 log entry answered this with two medians typed
from a notebook; this script replaces that with a measured table.

WHAT THIS DOES.  Joins the 200 adjudicated cases (via the blinding key, which never
leaves this process) to

  (i)   the renal-history stratum: `build_prior_indicators` from script 10 and
        `renal_history_labels` from script 12, imported, on the same raw tables;
  (ii)  the trajectory class A1/A2/A3, with the SAME nadir rule `taxonomy_for_split`
        in script 05 applies (`trajectory_class` below; a test asserts the two agree);
  (iii) the packet sampling cell A/B/C/D from the key,

and writes, per stratum and per cell x stratum: n_cases, n_decided, n_blank,
n_agree_with_rule, agreement_on_decided and the blank share. A blank is "cannot
tell from what is shown" and is DROPPED from the agreement denominator, never
scored as disagreement -- the same rule script 11 applies.

OUTPUT IS AGGREGATE ONLY. No hadm_id, subject_id or case_id reaches the result
file; a test walks the written JSON to enforce this.

STATUS: post hoc, exploratory; no registered threshold is touched. Cells with fewer
than 10 decided cases are listed under `small_cell_warning` and must not be quoted
as rates without that flag.

    python scripts/13_adjudication_by_stratum.py \\
        --cohort audit_mimic_out/aki_cohort.parquet --mimic-hosp ../mimic_raw/hosp
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

from auditnesy_mimic.config import load_config  # noqa: E402

_sg = import_module("05_subgroup_audit")
_ckd = import_module("10_prior_ckd_sensitivity")
_vr = import_module("12_violation_rate_by_renal_history")

build_prior_indicators = _ckd.build_prior_indicators
renal_history_labels = _vr.renal_history_labels
STRATUM_ORDER = list(_vr.STRATUM_ORDER)
TRAJECTORY_ORDER = ["A1_elevated_at_arrival", "A2_rise_from_near_normal",
                    "A3_rise_from_elevated_base"]
CELL_ORDER = ["A_severe_coded", "B_severe_uncoded", "C_near_threshold", "D_clearly_not_severe"]
SMALL_CELL_N_DECIDED = 10
FORBIDDEN_KEYS = ("hadm_id", "subject_id", "case_id")


def trajectory_class(nadir, thr_severe: float) -> np.ndarray:
    """Axis A of the error taxonomy, row-wise, with script 05's exact rule.

    `taxonomy_for_split` fills a missing nadir with 0.0, then: A1 if nadir >= severe
    threshold, A2 if nadir < 2.0, A3 otherwise. Reproduced here because 05 only
    returns cell counts, not per-row labels.
    """
    nad = np.where(np.isnan(np.asarray(nadir, float)), 0.0, np.asarray(nadir, float))
    a1 = nad >= thr_severe
    a2 = nad < 2.0
    return np.where(a1, TRAJECTORY_ORDER[0],
                    np.where(a2, TRAJECTORY_ORDER[1], TRAJECTORY_ORDER[2]))


def load_adjudications(key_path: Path, response_paths) -> pd.DataFrame:
    """One row per adjudicated case: cell, rule verdict, majority verdict (NaN = blank).

    Mirrors `11_clinician_adjudication.do_score`: responses are coerced to numeric,
    a case every reviewer left blank is `decided=False`, otherwise the majority of the
    non-blank answers is the clinician verdict.
    """
    key = json.loads(key_path.read_text())
    cases = pd.DataFrame(key["cases"])
    reviewers = {}
    for p in response_paths:
        r = pd.read_csv(p)
        if "case_id" not in r or "clinician_aki" not in r:
            raise SystemExit(f"{p}: needs columns case_id and clinician_aki")
        reviewers[Path(p).stem] = r.set_index("case_id")["clinician_aki"]
    if not reviewers:
        raise SystemExit("no adjudication responses found")
    resp = pd.DataFrame(reviewers).apply(pd.to_numeric, errors="coerce")
    resp = resp.reindex(cases["case_id"])
    decided = ~resp.isna().all(axis=1)
    majority = (resp.mean(axis=1, skipna=True) > 0.5).astype(float)
    majority[~decided] = np.nan
    cases["decided"] = decided.to_numpy()
    cases["clinician"] = majority.to_numpy()
    cases["rule"] = cases["rule_severe"].astype(int)
    return cases


def score_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Attach `agree` (NaN when blank). Blank is NEVER 0: it is absent from the mean."""
    agree = np.where(cases["decided"].to_numpy(),
                     (cases["clinician"] == cases["rule"]).astype(float), np.nan)
    return cases.assign(agree=agree)


def summarise(cases: pd.DataFrame, by) -> dict:
    """Aggregate counts per group; `by` is a column name or list of names."""
    out = {}
    for g, d in cases.groupby(by, sort=False, dropna=False):
        name = g if isinstance(g, str) else "|".join(map(str, g))
        n = int(len(d))
        n_dec = int(d["decided"].sum())
        agree = d["agree"].to_numpy(float)
        n_agree = int(np.nansum(agree))
        # the rate is the mean over the ANSWERED cases only; a blank is NaN here and so
        # drops out of the mean rather than pulling it towards zero
        rate = float(np.nanmean(agree)) if np.isfinite(agree).any() else None
        out[name] = {
            "n_cases": n,
            "n_decided": n_dec,
            "n_blank": n - n_dec,
            "n_agree_with_rule": n_agree,
            "agreement_on_decided": (round(rate, 4) if rate is not None else None),
            "blank_share": round((n - n_dec) / n, 4) if n else None,
        }
    return out


def _ordered(d: dict, order) -> dict:
    return {k: d[k] for k in order if k in d} | {k: v for k, v in d.items() if k not in order}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="audit_mimic_out/aki_cohort.parquet")
    ap.add_argument("--mimic-hosp", default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--key", default="results/adjudication_key.json")
    ap.add_argument("--responses-dir", default="results/adjudication_responses")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr_severe = float(cfg.constraints["creatinine_severe_mgdl"])
    hosp = Path(args.mimic_hosp) if args.mimic_hosp else Path(cfg.mimic_hosp_dir)
    if not hosp.exists():
        raise SystemExit(f"MIMIC-IV hosp directory not found: {hosp}; refusing to guess strata")
    key_path = Path(args.key)
    if not key_path.exists():
        raise SystemExit(f"{key_path} not present: this script needs the blinding key (kept out of git)")
    response_paths = sorted(Path(args.responses_dir).glob("*.csv"))

    cases = score_cases(load_adjudications(key_path, response_paths))

    # strata from the cohort: restrict the raw-table join to the adjudicated admissions
    df = pd.read_parquet(args.cohort)
    if not df["hadm_id"].is_unique:
        raise SystemExit("cohort hadm_id is not unique; the join would be ambiguous")
    sub = df[df["hadm_id"].isin(cases["hadm_id"])].copy()
    sub = build_prior_indicators(sub, hosp)
    sub["renal_history"] = renal_history_labels(sub).to_numpy()
    sub["trajectory"] = trajectory_class(
        pd.to_numeric(sub["creat_nadir_24h"], errors="coerce").to_numpy(), thr_severe)
    joined = cases.merge(sub[["hadm_id", "renal_history", "trajectory"]], on="hadm_id",
                         how="left", validate="one_to_one")
    n_unjoined = int(joined["renal_history"].isna().sum())
    if n_unjoined:
        raise SystemExit(f"{n_unjoined} adjudicated cases did not join to the cohort; "
                         "refusing to write a partial table")

    per_renal = _ordered(summarise(joined, "renal_history"), STRATUM_ORDER)
    per_traj = _ordered(summarise(joined, "trajectory"), TRAJECTORY_ORDER)
    per_cell = _ordered(summarise(joined, "cell"), CELL_ORDER)
    from itertools import product
    cell_x_renal = _ordered(summarise(joined, ["cell", "renal_history"]),
                            ["|".join(t) for t in product(CELL_ORDER, STRATUM_ORDER)])
    cell_x_traj = _ordered(summarise(joined, ["cell", "trajectory"]),
                           ["|".join(t) for t in product(CELL_ORDER, TRAJECTORY_ORDER)])
    renal_x_traj = _ordered(summarise(joined, ["renal_history", "trajectory"]),
                            ["|".join(t) for t in product(STRATUM_ORDER, TRAJECTORY_ORDER)])
    # combinations with no sampled case are absent from the tables above; name them so
    # an absent row is read as "zero cases", not as an omission
    empty_combinations = {
        "cell_x_renal_history": [k for k in ("|".join(t) for t in product(CELL_ORDER, STRATUM_ORDER))
                                 if k not in cell_x_renal],
        "cell_x_trajectory": [k for k in ("|".join(t) for t in product(CELL_ORDER, TRAJECTORY_ORDER))
                              if k not in cell_x_traj],
        "renal_history_x_trajectory": [k for k in ("|".join(t) for t in product(STRATUM_ORDER, TRAJECTORY_ORDER))
                                       if k not in renal_x_traj],
    }

    small = sorted(
        [f"{tbl}:{k}" for tbl, d in (("per_renal_history", per_renal),
                                     ("per_trajectory", per_traj),
                                     ("cell_x_renal_history", cell_x_renal),
                                     ("cell_x_trajectory", cell_x_traj),
                                     ("renal_history_x_trajectory", renal_x_traj))
         for k, v in d.items() if v["n_decided"] < SMALL_CELL_N_DECIDED])

    total = summarise(joined.assign(_all="all"), "_all")["all"]
    payload = {
        "revision": "v1.13 (W3PHIAI-27); post hoc, exploratory; no registered threshold changed",
        "n_cases": total["n_cases"],
        "n_decided": total["n_decided"],
        "n_blank": total["n_blank"],
        "n_reviewers": int(len(response_paths)),
        "blank_handling": ("a blank is 'cannot tell from what is shown', not a 'no': blanks "
                           "are excluded from the agreement denominator and never counted "
                           "as disagreement (same rule as 11_clinician_adjudication.py)"),
        "agreement_definition": "clinician majority verdict == rule_severe from the blinding key",
        "stratum_definition": _vr.renal_history_labels.__doc__.strip().splitlines()[0]
                              + " (block H partition from 12_violation_rate_by_renal_history.py)",
        "trajectory_definition": {
            TRAJECTORY_ORDER[0]: f"creat_nadir_24h >= {thr_severe} mg/dL (missing nadir -> 0.0)",
            TRAJECTORY_ORDER[1]: "creat_nadir_24h < 2.0 mg/dL",
            TRAJECTORY_ORDER[2]: f"2.0 <= creat_nadir_24h < {thr_severe} mg/dL",
            "source": "05_subgroup_audit.taxonomy_for_split, axis A"},
        "small_cell_rule": f"n_decided < {SMALL_CELL_N_DECIDED}",
        "small_cell_warning": small,
        "empty_combinations_zero_cases": empty_combinations,
        "per_renal_history": per_renal,
        "per_trajectory": per_traj,
        "per_cell": per_cell,
        "cell_x_renal_history": cell_x_renal,
        "cell_x_trajectory": cell_x_traj,
        "renal_history_x_trajectory": renal_x_traj,
    }

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in FORBIDDEN_KEYS:
                    raise SystemExit(f"identifier key {k!r} in payload; refusing to write")
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(payload)
    if payload["n_cases"] != len(cases):
        raise SystemExit("row count changed during the join; refusing to write")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "adjudication_by_stratum.json").write_text(json.dumps(payload, indent=1))

    def _table(title, d):
        print(f"\n{title}")
        print(f"  {'group':48s} {'n':>4s} {'dec':>4s} {'blank':>5s} {'agree':>5s} {'agr_rate':>8s} {'blank_sh':>8s}")
        for k, v in d.items():
            ar = "-" if v["agreement_on_decided"] is None else f"{v['agreement_on_decided']:.3f}"
            print(f"  {k:48s} {v['n_cases']:4d} {v['n_decided']:4d} {v['n_blank']:5d} "
                  f"{v['n_agree_with_rule']:5d} {ar:>8s} {v['blank_share']:8.3f}")
    _table("[renal history]", per_renal)
    _table("[trajectory]", per_traj)
    _table("[cell x renal history]", cell_x_renal)
    _table("[cell x trajectory]", cell_x_traj)
    print(f"\nsmall cells (n_decided < {SMALL_CELL_N_DECIDED}): {len(small)}")
    print(f"Saved -> {outdir/'adjudication_by_stratum.json'}", flush=True)


if __name__ == "__main__":
    main()
