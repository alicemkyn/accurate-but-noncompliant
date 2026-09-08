#!/usr/bin/env python
"""M4 — sensitivity of the compliance-coverage certificate to (beta, cap).

Post-processing only: reads the per-seed coverage / deployed-violation-rate
already computed by 02_run_experiment.py (OUT_DIR/per_seed_results.json). No
re-training. Sweeps the coverage floor beta and the deployed-violation cap and
reports, for each grid point, whether each deployment policy PASSES.

Purpose: show that the headline verdict ("neither policy passes") is not an
artifact of the specific pre-registered thresholds (beta=0.50, cap=0.05), i.e.
close the reviewer's "why 0.5?" concern (DA-2).

    export AUDITNESY_OUT=./auditnesy_mimic_out
    python scripts/03_threshold_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.config import load_config
from auditnesy_mimic.io_utils import save_json

BETAS = [0.30, 0.40, 0.50, 0.60, 0.70]
CAPS = [0.02, 0.05, 0.10]
POLICIES = {"logged": "T2", "enforced": "T2_enforced"}


def _series(per_seed, key):
    cov = np.array([ps[key]["coverage_severe"] for ps in per_seed], float)
    vr = np.array([ps[key]["violation_rate_deployed"] for ps in per_seed], float)
    return cov, vr


def main():
    cfg = load_config()
    import json
    path = cfg.out_dir / "per_seed_results.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run scripts/02_run_experiment.py first.")
    per_seed = json.loads(path.read_text())

    grid = {}
    print(f"{'policy':9} {'beta':>5} {'cap':>5} {'cov_p5':>7} {'vr_p95':>7}  verdict")
    print("-" * 52)
    for pname, key in POLICIES.items():
        cov, vr = _series(per_seed, key)
        cov_p5 = float(np.nanpercentile(cov, 5))
        vr_p95 = float(np.nanpercentile(vr, 95))
        for beta in BETAS:
            for cap in CAPS:
                passed = bool(cov_p5 >= beta and vr_p95 <= cap)
                grid[f"{pname}|beta={beta}|cap={cap}"] = {
                    "coverage_p5": round(cov_p5, 4),
                    "violation_p95": round(vr_p95, 4),
                    "pass": passed,
                }
                print(f"{pname:9} {beta:5.2f} {cap:5.2f} {cov_p5:7.3f} {vr_p95:7.3f}  "
                      f"{'PASS' if passed else 'fail'}")

    any_pass = any(v["pass"] for v in grid.values())
    n_pass = sum(v["pass"] for v in grid.values())
    summary = {
        "betas": BETAS, "caps": CAPS,
        "grid": grid,
        "any_policy_passes_anywhere": any_pass,
        "n_passing_gridpoints": n_pass,
        "n_gridpoints": len(grid),
        "interpretation": (
            "Verdict robust: no deployment policy passes anywhere on the grid."
            if not any_pass else
            "Some grid points pass -- verdict is threshold-sensitive; report boundary."
        ),
    }
    out = cfg.out_dir / "threshold_sensitivity.json"
    save_json(out, summary)
    print("-" * 52)
    print(summary["interpretation"])
    print(f"passing gridpoints: {n_pass}/{len(grid)}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
