#!/usr/bin/env python
"""Generate the four publication figures as vector PDFs. Reads only aggregate
result JSONs (no patient-level data). Standalone -- does not touch the pipeline.

Figures:
    A  fig_seed_distributions.pdf   20-seed box plots (rigor / no cherry-picking)
    B  fig_compliance_coverage.pdf  (coverage, violation) plane + admissible region
                                     + threshold-sweep robustness annotation
    C  fig_probe_curve.pdf          predicted risk vs swept creatinine
                                     (full vs constraint-blind model)
    D  fig_reliability.pdf          reliability diagram (calibrated vs NC-A)

A and B use committed aggregate results; C and D need figdata/ from
dump_figure_data.py (run that first for C/D).

    cd figures && python make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "mimic_pipeline" / "results"
FIGDATA = HERE / "figdata"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# publication styling: seaborn theme as the base aesthetic for every figure
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,  # editable text
    "axes.titleweight": "bold", "axes.edgecolor": "0.25", "axes.linewidth": 0.9,
    "lines.linewidth": 2.2, "lines.markersize": 6, "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
})
C_LOG, C_ENF, C_OK = "#c0392b", "#2c6fbb", "#27ae60"


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# ---------------------------------------------------------------- Figure A
def fig_A():
    per = _load(RESULTS / "per_seed_results.json")
    if per is None:
        print("[A] skip: per_seed_results.json not in results/ "
              "(copy it from the run's out dir; it is aggregate over seeds)")
        return
    labels = ["AUROC", "ECE", "Logged\nviolation rate", "Enforced\ncoverage"]
    keys = [lambda p: p["auroc"], lambda p: p["T3"]["ece_point"],
            lambda p: p["T2"]["violation_rate_deployed"],
            lambda p: p["T2_enforced"]["coverage_severe"]]
    data = [[k(ps) for ps in per] for k in keys]
    pal = ["#34495e", "#16a085", C_LOG, C_ENF]

    # one subpanel per metric, each auto-scaled -> ECE no longer squished by AUROC,
    # points no longer collapse onto a shared axis
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 3.0))
    rng = np.random.default_rng(0)
    for ax, lab, col, vals in zip(axes, labels, pal, data):
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5, showfliers=False,
                        medianprops=dict(color="white", lw=1.6),
                        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor(col); bp["boxes"][0].set_linewidth(1.5)
        xj = 1 + rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(xj, vals, s=34, color=col, alpha=0.85, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.set_title(lab, fontsize=11)
        ax.set_xticks([])
        m, s = np.mean(vals), np.std(vals)
        ax.set_xlabel(f"{m:.3f}\n$\\pm${s:.3f}", fontsize=9)
    axes[0].set_ylabel("value (per seed, n=20)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.80, wspace=0.55)
    fig.suptitle("Stability across 20 patient-level splits", y=0.99, fontsize=13, weight="bold")
    fig.savefig(OUT / "fig_seed_distributions.pdf")
    plt.close(fig); print("[A] fig_seed_distributions.pdf")


# ---------------------------------------------------------------- Figure B
def fig_B():
    s = _load(RESULTS / "audit_summary_mimic_final_v1.3.json")
    if s is None:
        print("[B] skip: audit_summary_mimic_final_v1.3.json not found"); return
    cc = s["aggregate"]["compliance_coverage_certificate"]
    log_pt = (cc["logged_policy"]["coverage_severe_mean"], cc["logged_policy"]["violation_rate_deployed_mean"])
    enf_pt = (cc["enforced_policy"]["coverage_severe_mean"], cc["enforced_policy"]["violation_rate_deployed_mean"])
    beta, cap = cc["coverage_floor_beta"], cc["violation_rate_max"]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # admissible region for the registered thresholds
    ax.add_patch(plt.Rectangle((beta, 0), 1 - beta, cap, color=C_OK, alpha=0.18, zorder=0))
    ax.text(beta + (1 - beta) / 2, cap + 0.03, "admissible (PASS)", color="#1e7a43",
            ha="center", fontsize=9, weight="bold")
    # threshold-sweep bands (beta in [0.3,0.7], cap in [0.02,0.10]) -> boundary never captures points
    ax.axvspan(0.30, 0.70, color="0.5", alpha=0.06, zorder=0)
    ax.axhspan(0.02, 0.10, color="0.5", alpha=0.06, zorder=0)
    ax.axvline(beta, ls="--", lw=1, color="0.4"); ax.axhline(cap, ls="--", lw=1, color="0.4")
    ax.text(beta + 0.01, 0.75, r"$\beta=0.5$ (swept 0.3–0.7)", fontsize=8, color="0.35")
    ax.text(0.55, cap + 0.005, "cap 0.05 (swept 0.02–0.10)", fontsize=8, color="0.35")

    ax.scatter(*log_pt, s=90, color=C_LOG, zorder=5, label="logged")
    ax.annotate("logged\n(fails: violation)", log_pt, textcoords="offset points",
                xytext=(-8, -34), color=C_LOG, fontsize=9, ha="center")
    ax.scatter(*enf_pt, s=90, marker="s", color=C_ENF, zorder=5, label="enforced")
    ax.annotate("enforced\n(fails: coverage)", enf_pt, textcoords="offset points",
                xytext=(38, 6), color=C_ENF, fontsize=9, ha="center")

    ax.set_xlim(0, 1); ax.set_ylim(0, 0.8)
    ax.set_xlabel("coverage (deployed fraction of acute-severe)")
    ax.set_ylabel("deployed violation rate")
    ax.set_title("Compliance–coverage certificate\n(neither policy passes; robust to thresholds)")
    fig.savefig(OUT / "fig_compliance_coverage.pdf")
    plt.close(fig); print("[B] fig_compliance_coverage.pdf")


# ---------------------------------------------------------------- Figure C
def fig_C():
    d = _load(FIGDATA / "probe_curve.json")
    if d is None:
        print("[C] skip: figdata/probe_curve.json missing -- run dump_figure_data.py first"); return
    x = np.array(d["creatinine_mgdl"])
    full = np.array(d["full_model_mean_risk"])
    blind = np.array(d["constraint_blind_mean_risk"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(x, full, "-o", ms=5, color="#8e44ad", markeredgecolor="white",
            markeredgewidth=0.6, label="full model (creatinine-aware)", zorder=4)
    ax.plot(x, blind, "--s", ms=4, color="0.55", markeredgecolor="white",
            markeredgewidth=0.5, label="constraint-blind model", zorder=3)
    # shade the guideline-severe zone (KDIGO AKI region)
    ax.axvspan(d["severe_threshold"], x.max(), color=C_LOG, alpha=0.07, zorder=0)
    ax.axvline(d["severe_threshold"], ls=":", color=C_LOG, lw=1.4)
    ymax = max(full.max(), blind.max()) * 1.30
    ax.set_ylim(0, ymax)
    sev = d["severe_threshold"]
    zc = (sev + x.max()) / 2   # horizontal centre of the shaded severe zone
    # severe-zone label: 2 lines, centred at the bottom of the shaded band (fits inside the zone)
    ax.text(zc, ymax * 0.04, "guideline-severe\nzone (KDIGO)", color=C_LOG,
            fontsize=8.5, ha="center", va="bottom", linespacing=1.15)
    # annotate the clinically-wrong crossing: inside the severe zone the
    # creatinine-aware model predicts LOWER risk than the constraint-blind model.
    # 4 short lines, centred in the zone so the whole label sits inside the band.
    below = [i for i in range(len(x)) if x[i] >= sev and full[i] < blind[i]]
    tgt = below[0] if below else len(x) - 4       # first severe-zone point where full < blind
    ax.annotate("in the severe zone,\nthe creatinine-aware\nmodel predicts LOWER\n"
                "risk than the blind model",
                xy=(x[tgt], full[tgt]), xytext=(zc, ymax * 0.97),
                fontsize=7.8, color="#8e44ad", ha="center", va="top", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2))
    ax.set_xlabel("counterfactual creatinine (mg/dL)")
    ax.set_ylabel("mean predicted AKI risk")
    ax.set_title("Concept-semantics probe:\ndoes the decision depend on creatinine?")
    ax.legend(frameon=True, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "fig_probe_curve.pdf")
    plt.close(fig); print("[C] fig_probe_curve.pdf")


# ---------------------------------------------------------------- Figure D
def fig_D():
    d = _load(FIGDATA / "reliability.json")
    if d is None:
        print("[D] skip: figdata/reliability.json missing -- run dump_figure_data.py first"); return
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0.5, 1], [0.5, 1], ls="--", color="0.5", lw=1, label="perfect calibration")
    for key, col, lab, mk in [("calibrated", C_OK, "calibrated (T3 pass)", "o"),
                              ("uncalibrated_ncA", C_LOG, "over-confident (NC-A)", "s")]:
        rows = d[key]
        cf = [r["mean_confidence"] for r in rows]
        ac = [r["accuracy"] for r in rows]
        ax.plot(cf, ac, marker=mk, ms=4, color=col, label=lab)
    ax.set_xlim(0.5, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig_reliability.pdf")
    plt.close(fig); print("[D] fig_reliability.pdf")


if __name__ == "__main__":
    fig_A(); fig_B(); fig_C(); fig_D()
    print(f"Figures -> {OUT}")
