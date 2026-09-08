"""STEP 12 -- Pima concept-semantics probe (reconstruction).

The paper claims the probe "separates them cleanly": sweeping glucose from 70
to 200 mg/dL, mean predicted risk rises in the genuine model but stays flat in
the glucose-blind control, so the genuine model's constraint compliance is a
consequence of DEPENDENCE on the constrained concept rather than an accident.

The probe holds every other feature at its test-split value and replaces
glucose with each grid point in turn, averaging the predicted risk over the
test split. Sensitivity is the risk difference between the top and bottom of
the sweep -- for the glucose-blind model it is exactly zero by construction,
which is the point of the comparison.

Same reference split, same model construction and same imputation as
`10_pima_audit.py`; this script only adds the sweep.

Usage:
    python scripts/12_pima_probe.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "mimic_pipeline"))
_spec = importlib.util.spec_from_file_location("pima_audit", _HERE.parent / "10_pima_audit.py")
_pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pa)

GRID_LO, GRID_HI, GRID_N = 70.0, 200.0, 27


def sweep(model, X, te, gi, grid, keep=None):
    out = []
    for g in grid:
        Xs = X[te].copy()
        Xs[:, gi] = g
        Xq = Xs if keep is None else Xs[:, keep]
        out.append(float(model.predict_proba(Xq)[:, 1].mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_HERE.parents[1] / "data" / "pima_diabetes.csv"))
    ap.add_argument("--out", default=str(_HERE.parents[1] / "results"))
    args = ap.parse_args()

    df = _pa.load_raw(Path(args.data))
    seed = _pa.REFERENCE_SEED
    X, y, tr, te, filled = _pa.split_and_impute(df, seed)
    gi = _pa.FEATURES.index("Glucose")
    grid = np.linspace(GRID_LO, GRID_HI, GRID_N)

    genuine = _pa.fit_model(X, y, tr, seed)
    curve_g = sweep(genuine, X, te, gi, grid)

    keep = [i for i, f in enumerate(_pa.FEATURES) if f != "Glucose"]
    blind = _pa.fit_model(X[:, keep], y, tr, seed)
    curve_b = sweep(blind, X, te, gi, grid, keep=keep)

    payload = {
        "reconstruction_notice": (
            "Reconstructed from the paper's written description; the original probe code was "
            "never committed. Where these numbers differ from earlier drafts, the paper is "
            "corrected to match this output."),
        "reference_seed": seed,
        "grid_mgdl": [GRID_LO, GRID_HI], "grid_points": GRID_N,
        "genuine_model": {
            "risk_at_low": round(curve_g[0], 4), "risk_at_high": round(curve_g[-1], 4),
            "sensitivity": round(curve_g[-1] - curve_g[0], 4),
            "monotone_nondecreasing": bool(all(b >= a - 1e-9 for a, b in zip(curve_g, curve_g[1:]))),
        },
        "glucose_blind_control": {
            "risk_at_low": round(curve_b[0], 4), "risk_at_high": round(curve_b[-1], 4),
            "sensitivity": round(curve_b[-1] - curve_b[0], 4),
            "note": "zero by construction: the feature is absent from the model",
        },
        "curve_genuine": [round(v, 4) for v in curve_g],
        "curve_blind": [round(v, 4) for v in curve_b],
    }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pima_probe.json").write_text(json.dumps(payload, indent=1))
    g, b = payload["genuine_model"], payload["glucose_blind_control"]
    print(f"genuine      : {g['risk_at_low']} -> {g['risk_at_high']}  sensitivity={g['sensitivity']} "
          f"monotone={g['monotone_nondecreasing']}", flush=True)
    print(f"glucose-blind: {b['risk_at_low']} -> {b['risk_at_high']}  sensitivity={b['sensitivity']}", flush=True)
    print(f"\nSaved -> {outdir/'pima_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
