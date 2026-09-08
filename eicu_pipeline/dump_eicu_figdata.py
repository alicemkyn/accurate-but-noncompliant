#!/usr/bin/env python
"""C-lite: for BOTH datasets (MIMIC-IV, eICU) dump
  (1) the concept-semantics probe curve  -> mechanism replication (side-by-side figure)
  (2) the NC-A severity sweep k*          -> min stretch factor to breach the 0.10 ECE bound

Standalone; imports the harness read-only; writes aggregate JSON to results/
(no patient-level rows). Reuses the cached cohort parquets (no re-extraction).

    python dump_eicu_figdata.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(__file__).resolve().parent.parent / "mimic_pipeline"
sys.path.insert(0, str(MAIN))
from auditnesy_mimic.config import load_config          # noqa: E402
from auditnesy_mimic.models import PipelineCoupled        # noqa: E402
from auditnesy_mimic import metrics                       # noqa: E402
from sklearn.model_selection import GroupShuffleSplit     # noqa: E402
from sklearn.neural_network import MLPClassifier          # noqa: E402
from sklearn.pipeline import Pipeline                     # noqa: E402
from sklearn.preprocessing import StandardScaler          # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
CFG = load_config(MAIN / "config.yaml")

COHORTS = {
    "mimic": MAIN / "audit_mimic_out",
    "eicu": Path("/Volumes/Extreme SSD/Datasets/eicu-crd/eicu_out"),
}
ECE_BOUND = CFG.audit["ece_pass_upper_ci"]     # 0.10
N_BINS = CFG.audit["n_bins_ece"]
SWEEP = np.linspace(0.5, 6.0, 23)              # creatinine mg/dL (probe)
K_GRID = np.arange(1.0, 8.01, 0.25)            # NC-A stretch factors


def _load(d):
    df = pd.read_parquet(d / "aki_cohort.parquet")
    feat = (d / "feature_cols.txt").read_text().splitlines()
    return df, feat


def _split(df, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=CFG.experiment["test_size"], random_state=seed)
    return next(gss.split(df, groups=df["subject_id"].values))


def probe_curve(df, feat):
    X = df[feat].to_numpy(float); y = df["aki"].to_numpy(int)
    tr, te = _split(df, 0)
    ci = feat.index("creatinine")
    full = PipelineCoupled(CFG.models["pipeline_coupled"]["hidden_layers"],
                           CFG.models["pipeline_coupled"]["calibrate"], seed=0).fit(X[tr], y[tr])
    keep = [i for i in range(X.shape[1]) if i != ci]
    blind = Pipeline([("s", StandardScaler()),
                      ("m", MLPClassifier(hidden_layer_sizes=(32, 16), random_state=0,
                                          max_iter=500, early_stopping=True))]).fit(X[tr][:, keep], y[tr])
    n = min(CFG.audit["probe_n_samples"], len(te))
    idx = te[np.random.default_rng(0).choice(len(te), n, replace=False)]
    f, b = [], []
    for v in SWEEP:
        Xf = X[idx].copy(); Xf[:, ci] = v; f.append(float(full.predict_proba(Xf).mean()))
        Xb = X[idx].copy(); Xb[:, ci] = v; b.append(float(blind.predict_proba(Xb[:, keep])[:, 1].mean()))
    return {"creatinine_mgdl": SWEEP.tolist(), "full_model_mean_risk": f,
            "constraint_blind_mean_risk": b, "severe_threshold": CFG.constraints["creatinine_severe_mgdl"]}


def kstar(df, feat, n_seeds=20, B=200):
    """Per seed: min stretch factor k whose NC-A ECE CI-upper >= the 0.10 bound."""
    X = df[feat].to_numpy(float); y = df["aki"].to_numpy(int)
    ece_fn = lambda a, c: metrics.ece(a, c, N_BINS)
    ks = []
    for seed in CFG.experiment["seeds"][:n_seeds]:
        tr, te = _split(df, seed)
        m = PipelineCoupled(CFG.models["pipeline_coupled"]["hidden_layers"],
                            CFG.models["pipeline_coupled"]["calibrate"], seed=seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te]); yt = y[te]; grp = df["subject_id"].to_numpy()[te]
        k_hit = None
        for k in K_GRID:
            sharp = np.clip((p - 0.5) * k + 0.5, 1e-4, 1 - 1e-4)
            _, _, hi = metrics.bootstrap_ci(ece_fn, sharp, yt, groups=grp, B=B, seed=seed)
            if hi >= ECE_BOUND:
                k_hit = float(k); break
        ks.append(k_hit if k_hit is not None else float("nan"))
    ks = np.array(ks, float)
    return {"k_star_median": round(float(np.nanmedian(ks)), 3),
            "k_star_range": [round(float(np.nanmin(ks)), 3), round(float(np.nanmax(ks)), 3)],
            "k_star_per_seed": [None if np.isnan(v) else round(v, 3) for v in ks],
            "ece_bound": ECE_BOUND, "n_seeds": int((~np.isnan(ks)).sum())}


if __name__ == "__main__":
    out = {"ece_bound": ECE_BOUND, "datasets": {}}
    for name, d in COHORTS.items():
        if not (d / "aki_cohort.parquet").exists():
            print(f"[skip] {name}: cohort not found at {d}"); continue
        df, feat = _load(d)
        print(f"[{name}] probe curve ...")
        pc = probe_curve(df, feat)
        (OUT / f"probe_curve_{name}.json").write_text(json.dumps(pc, indent=2))
        print(f"[{name}] NC-A k* sweep ...")
        ks = kstar(df, feat)
        out["datasets"][name] = ks
        print(f"    k* median {ks['k_star_median']} range {ks['k_star_range']}")
    (OUT / "ncA_kstar.json").write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT} (probe_curve_*.json, ncA_kstar.json)")
