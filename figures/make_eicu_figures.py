#!/usr/bin/env python
"""Two-panel probe figure: mechanism replication across MIMIC-IV and eICU.
Reads aggregate probe_curve_{mimic,eicu}.json (no patient data). Standalone.

    cd figures && python make_eicu_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "eicu_pipeline" / "results"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "axes.titleweight": "bold", "axes.edgecolor": "0.25",
                     "lines.linewidth": 2.2})
C_LOG = "#c0392b"


def panel(ax, d, title):
    x = np.array(d["creatinine_mgdl"]); full = np.array(d["full_model_mean_risk"])
    blind = np.array(d["constraint_blind_mean_risk"]); sev = d["severe_threshold"]
    ymax = max(full.max(), blind.max()) * 1.28
    ax.axvspan(sev, x.max(), color=C_LOG, alpha=0.07, zorder=0)
    ax.axvline(sev, ls=":", color=C_LOG, lw=1.4)
    ax.plot(x, full, "-o", ms=4.5, color="#8e44ad", markeredgecolor="white",
            markeredgewidth=0.5, label="full model (creatinine-aware)", zorder=4)
    ax.plot(x, blind, "--s", ms=3.5, color="0.55", markeredgecolor="white",
            markeredgewidth=0.4, label="constraint-blind model", zorder=3)
    pk = int(np.argmax(full))
    ax.annotate("", xy=(x[-3], full[-3]), xytext=(x[pk], full[pk]),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.3, alpha=0.7))
    ax.text(x.max(), full.max() * 0.06, "guideline-severe\nzone (KDIGO)", color=C_LOG,
            fontsize=8, ha="right", va="bottom")
    ax.set_ylim(0, ymax); ax.set_xlabel("counterfactual creatinine (mg/dL)")
    ax.set_title(title, fontsize=11)


if __name__ == "__main__":
    dm = json.loads((RES / "probe_curve_mimic.json").read_text())
    de = json.loads((RES / "probe_curve_eicu.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=False)
    panel(axes[0], dm, "MIMIC-IV (1 hospital)")
    panel(axes[1], de, "eICU (205 hospitals)")
    axes[0].set_ylabel("mean predicted AKI risk")
    axes[0].legend(frameon=True, fontsize=8, loc="upper right")
    fig.suptitle("Concept-semantics probe: risk declines through the severe zone on both EHRs",
                 y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_twopanel.pdf")
    print(f"[eICU] fig_probe_twopanel.pdf -> {OUT}")
