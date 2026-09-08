#!/usr/bin/env python
"""Standalone dump of the extra data needed for figures C (concept-semantics probe
curve) and D (reliability diagram). Does NOT modify the auditnesy_mimic package;
it only imports model/metric classes read-only, reuses the cached cohort, and
writes aggregate JSON to figures/figdata/ (no patient-level rows).

    export AUDITNESY_OUT=../mimic_pipeline/audit_mimic_out   # dir with aki_cohort.parquet
    cd figures && python dump_figure_data.py

Outputs:
    figdata/probe_curve.json       (Fig C: mean predicted risk vs swept creatinine)
    figdata/reliability.json       (Fig D: confidence-bin accuracy, calibrated vs uncalibrated)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# import the pipeline package read-only (no edits to it)
PKG_ROOT = Path(__file__).resolve().parent.parent / "mimic_pipeline"
sys.path.insert(0, str(PKG_ROOT))
from auditnesy_mimic.config import load_config          # noqa: E402
from auditnesy_mimic.models import PipelineCoupled        # noqa: E402
from sklearn.model_selection import GroupShuffleSplit     # noqa: E402
from sklearn.neural_network import MLPClassifier          # noqa: E402
from sklearn.pipeline import Pipeline                     # noqa: E402
from sklearn.preprocessing import StandardScaler          # noqa: E402

OUT = Path(__file__).resolve().parent / "figdata"
OUT.mkdir(exist_ok=True)
SEED = 0


def _load():
    cfg = load_config(PKG_ROOT / "config.yaml")
    out_dir = Path(os.environ.get("AUDITNESY_OUT", cfg.out_dir))
    if not out_dir.is_absolute():
        out_dir = PKG_ROOT / out_dir
    df = pd.read_parquet(out_dir / "aki_cohort.parquet")
    feat_cols = (out_dir / "feature_cols.txt").read_text().splitlines()
    return cfg, df, feat_cols


def _split(df, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["subject_id"].values))


def dump_probe_curve(cfg, df, feat_cols):
    """Fig C: sweep creatinine across a clinical range; record mean predicted risk
    for the full (creatinine-aware) model and a constraint-blind model."""
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    tr, te = _split(df, cfg.experiment["test_size"], SEED)
    ci = feat_cols.index("creatinine")

    full = PipelineCoupled(cfg.models["pipeline_coupled"]["hidden_layers"],
                           cfg.models["pipeline_coupled"]["calibrate"], seed=SEED).fit(X[tr], y[tr])
    keep = [i for i in range(X.shape[1]) if i != ci]
    blind = Pipeline([("s", StandardScaler()),
                      ("m", MLPClassifier(hidden_layer_sizes=(32, 16), random_state=SEED,
                                          max_iter=500, early_stopping=True))]).fit(X[tr][:, keep], y[tr])

    n = min(cfg.audit["probe_n_samples"], len(te))
    rng = np.random.default_rng(SEED)
    idx = te[rng.choice(len(te), size=n, replace=False)]
    sweep = np.linspace(0.5, 6.0, 23)  # mg/dL clinical range
    full_risk, blind_risk = [], []
    for v in sweep:
        Xf = X[idx].copy(); Xf[:, ci] = v
        full_risk.append(float(full.predict_proba(Xf).mean()))
        Xb = X[idx].copy(); Xb[:, ci] = v
        blind_risk.append(float(blind.predict_proba(Xb[:, keep])[:, 1].mean()))

    obj = {"creatinine_mgdl": sweep.tolist(),
           "full_model_mean_risk": full_risk,
           "constraint_blind_mean_risk": blind_risk,
           "severe_threshold": cfg.constraints["creatinine_severe_mgdl"],
           "n_sampled_decisions": n}
    (OUT / "probe_curve.json").write_text(json.dumps(obj, indent=2))
    print(f"[C] probe_curve.json  (full sens {full_risk[-1]-full_risk[0]:+.3f}, "
          f"blind sens {blind_risk[-1]-blind_risk[0]:+.3f})")


def _reliability(probs, labels, n_bins=10):
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5).astype(int) == labels).astype(int)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        rows.append({"bin_mid": float((edges[b] + edges[b + 1]) / 2),
                     "mean_confidence": float(conf[m].mean()),
                     "accuracy": float(correct[m].mean()),
                     "weight": float(m.mean())})
    return rows


def dump_reliability(cfg, df, feat_cols):
    """Fig D: reliability of the calibrated model vs an over-confident (NC-A) variant."""
    X = df[feat_cols].to_numpy(float)
    y = df["aki"].to_numpy(int)
    tr, te = _split(df, cfg.experiment["test_size"], SEED)
    model = PipelineCoupled(cfg.models["pipeline_coupled"]["hidden_layers"],
                            cfg.models["pipeline_coupled"]["calibrate"], seed=SEED).fit(X[tr], y[tr])
    p = model.predict_proba(X[te])
    p_over = np.clip((p - 0.5) * 3.0 + 0.5, 1e-4, 1 - 1e-4)   # NC-A sharpening
    obj = {"n_bins": cfg.audit["n_bins_ece"],
           "calibrated": _reliability(p, y[te], cfg.audit["n_bins_ece"]),
           "uncalibrated_ncA": _reliability(p_over, y[te], cfg.audit["n_bins_ece"])}
    (OUT / "reliability.json").write_text(json.dumps(obj, indent=2))
    print("[D] reliability.json")


if __name__ == "__main__":
    cfg, df, feat_cols = _load()
    print(f"[dump] cohort {len(df)} rows from cached parquet")
    dump_probe_curve(cfg, df, feat_cols)
    dump_reliability(cfg, df, feat_cols)
    print(f"Saved -> {OUT}")
