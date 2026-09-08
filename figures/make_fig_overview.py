"""Figure 1 -- the schematic overview.

Every number in the figure is PARSED from a committed file; none is a literal in
this script. An earlier version claimed that while hardcoding two of them (the
hospital count and the deviations-row count), which is exactly the defect this
project's own results-self-consistency tests exist to catch, so both now have
readers below and `test_fig_overview_numbers.py` fails if either is reintroduced
as a literal.

Sources, one per number:
  mimic_pipeline/results/audit_summary_mimic_final_v1.3.json
      cohort sizes, per-axis verdicts, certificate thresholds and outcomes,
      negative-control count
  mimic_pipeline/results/matched_positive_control.json
      the positive control's base-rate ceiling
  eicu_pipeline/results/audit_summary_eicu_v1.3.json
      eICU cohort size (its own denominator, never merged with MIMIC's)
  eicu_pipeline/EICU_FINDINGS.md
      eICU hospital count (the eICU summary JSON does not record it; the count
      is printed by build_eicu_cohort.py and written up there)
  AuditNeSy_v26.tex
      number of rows in the deviations log, counted from the tables themselves

Run:

    cd figures && python make_fig_overview.py

Writes ../paper_figs/fig_overview.{pdf,png}. Needs no cohort access and no
credentialed data.
"""
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "mimic_pipeline" / "results"
OUT = ROOT / "paper_figs"
PAPER = ROOT / "AuditNeSy_v26.tex"
EICU_FINDINGS = ROOT / "eicu_pipeline" / "EICU_FINDINGS.md"


def n_eicu_hospitals() -> int:
    """eICU's hospital count -- the multi-site half of the study.

    Not in the eICU summary JSON: `build_eicu_cohort.py` prints it at cohort build
    and it is written up in EICU_FINDINGS.md, so that is the file of record.

    Returned SEPARATELY from the MIMIC cohort size on purpose. An earlier version
    returned a single 206 total and the panel read "379k admissions, 167k patients,
    206 hospitals" -- which says the MIMIC cohort spans 206 sites. It does not:
    MIMIC-IV is one hospital, and the 205 others are the eICU replication with its
    own cohort. Two databases, two denominators, stated separately.
    """
    m = re.search(r"\*\*(\d[\d,]*)\s+hospitals\*\*", EICU_FINDINGS.read_text(errors="ignore"))
    if not m:
        raise RuntimeError(f"no bolded hospital count found in {EICU_FINDINGS}")
    return int(m.group(1).replace(",", ""))


def n_deviation_rows() -> int:
    """Rows in the deviations log, counted from the paper's own tables.

    The log lives in the paper (two continuation tables), not in a result file, so
    the figure counts the rows rather than restating a number that would silently
    go stale the next time a deviation is added.
    """
    tex = PAPER.read_text(errors="ignore")
    rows: set[int] = set()
    for label in ("{tab:deviations}", "{tab:deviations2}"):
        k = tex.find(label + "\n")
        if k == -1:
            raise RuntimeError(f"deviations table {label} not found in {PAPER.name}")
        start = tex.rfind(r"\begin{table*}", 0, k)
        end = tex.find(r"\end{table*}", k)
        rows |= {int(x) for x in re.findall(r"\n\s*(\d+)\s*&", tex[start:end])}
    if not rows:
        raise RuntimeError("deviations tables parsed but no numbered rows found")
    return len(rows)

GREY, FOCAL, ALARM, PASS = "#6b7280", "#1f5fa8", "#b03a2e", "#2e7d32"
INPUT_FC, INPUT_EC = "#eceff1", "#9aa5ab"
MODEL_FC, MODEL_EC = "#fdf1dc", "#c99a3f"
PANEL_FC = "#e9f0f7"
RULE_FC, RULE_EC = "#fbeceb", "#b03a2e"


def _style():
    mpl.rcParams.update({
        "font.size": 8, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "axes.linewidth": 0.8,
    })


def main() -> None:
    S = json.loads((RES / "audit_summary_mimic_final_v1.3.json").read_text())
    M = json.loads((RES / "matched_positive_control.json").read_text())
    E = json.loads((ROOT / "eicu_pipeline" / "results" /
                    "audit_summary_eicu_v1.3.json").read_text())
    agg, cc = S["aggregate"], S["aggregate"]["compliance_coverage_certificate"]
    ce = M["effect_ceiling"]

    n_adm, n_pat, n_sev = S["n_admissions"], S["n_patients"], S["n_severe_acute"]
    agree = S["aki_rate_in_severe_acute"]
    beta, cap = cc["coverage_floor_beta"], cc["violation_rate_max"]
    vr_log = cc["logged_policy"]["violation_rate_deployed_mean"]
    cov_enf = cc["enforced_policy"]["coverage_severe_mean"]
    n_nc = len(S["negative_controls"])
    pc_share = round(100 * ce["share_attributable_to_base_rate"])
    n_hosp_eicu = n_eicu_hospitals()
    n_eicu_stays = E["n_admissions"]
    n_dev = n_deviation_rows()

    _style()
    fig = plt.figure(figsize=(7.16, 6.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def rbox(x, y, w, h, txt, fc="white", ec=GREY, tc="black", fs=6.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=0.9, zorder=3))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4, linespacing=1.35,
                fontweight="bold" if bold else "normal")

    def panel(x, y, w, h, fc=PANEL_FC, ec="none"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=0.8, zorder=1))

    def arrow(x1, y1, x2, y2, color="black", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=1.0, ls=ls, shrinkA=2, shrinkB=2, zorder=2))

    # ---- section 1 ----
    ax.text(2, 96.5, "1. What is audited: a model trained on one authority, judged against another",
            ha="left", va="center", fontsize=8, fontweight="bold")
    rbox(3, 80, 20, 9, "First 24 h\nlabs and vitals", fc=INPUT_FC, ec=INPUT_EC)
    rbox(34, 80, 20, 9, "AKI risk model", fc=MODEL_FC, ec=MODEL_EC, bold=True)
    rbox(65, 80, 22, 9, 'Confident\n"no AKI" decision', fc=MODEL_FC, ec=MODEL_EC)
    arrow(23.5, 84.5, 33.5, 84.5); arrow(54.5, 84.5, 64.5, 84.5)
    ax.text(28.5, 85.5, "(A) trained", ha="center", va="bottom", fontsize=6.3, color=ALARM)
    ax.text(59.5, 85.5, "(B) deployed", ha="center", va="bottom", fontsize=6.3, color=ALARM)

    rbox(34, 66, 20, 8, "Diagnosis code\nassigned at discharge", fc=INPUT_FC, ec=INPUT_EC, fs=6.2)
    arrow(44, 74.5, 44, 79.5, GREY, ls=(0, (2.5, 2)))
    ax.text(45.5, 76, "target", ha="left", va="center", fontsize=6.3, color=GREY)
    rbox(65, 66, 22, 8, "KDIGO rule fires:\nsevere acute AKI", fc=RULE_FC, ec=RULE_EC, fs=6.2)
    arrow(76, 74.5, 76, 79.5, ALARM)
    ax.text(77.5, 76, "(C) conflict", ha="left", va="center", fontsize=6.3, color=ALARM)

    panel(2, 53, 85, 10.5)
    ax.text(4.5, 58.2,
            f"MIMIC-IV, one hospital: {n_adm/1000:.0f}k admissions, {n_pat/1000:.0f}k patients.  "
            f"The two authorities agree on only {100*agree:.0f}% of the {n_sev:,} cases where the rule fires.\n"
            f"Replicated on eICU, a separate cohort of {n_eicu_stays/1000:.0f}k ICU stays across {n_hosp_eicu} hospitals with a different labelling system.",
            ha="left", va="center", fontsize=6.3, zorder=4, linespacing=1.5)
    ax.plot([2, 98], [50, 50], color="#cfd8dc", lw=0.8)

    # ---- section 2 ----
    ax.text(2, 47, "2. What the audit reports: three axes, never collapsed, plus a joint certificate",
            ha="left", va="center", fontsize=8, fontweight="bold")
    rbox(3, 30, 19, 10, "Model under\naudit", fc=MODEL_FC, ec=MODEL_EC, bold=True)
    panel(26, 22.5, 44, 19)
    ax.text(48, 40.0, "Per-axis verdicts", ha="center", va="center", fontsize=6.8,
            fontweight="bold", zorder=4)
    rows = [
        (36.0, "T1  Replay", f"{agg['T1_pass_stability']} seeds", "pass", PASS),
        (31.5, "T2  Constraint", f"mean {agg['T2_hard_violations_mean']:.0f} violations / split", "FAIL", ALARM),
        (27.0, "T3  Calibration", f"ECE {agg['T3_ece_point_mean']:.3f}", "pass", PASS),
    ]
    for yy, name, val, verdict, col in rows:
        ax.add_patch(FancyBboxPatch((28, yy - 1.6), 30, 3.2,
                                    boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc="white", ec="#c5cdd2", lw=0.7, zorder=3))
        ax.text(29.2, yy, name, ha="left", va="center", fontsize=6.2, zorder=4)
        ax.text(56.2, yy, val, ha="right", va="center", fontsize=6.0, color=GREY, zorder=4)
        ax.text(57.4, yy, verdict, ha="left", va="center", fontsize=6.2, color=col,
                fontweight="bold" if verdict == "FAIL" else "normal", zorder=4)
    ax.text(48, 24.0, f"audit readiness {agg['audit_readiness']}  \u2014  reported separately, never averaged",
            ha="center", va="center", fontsize=6.0, color=GREY, zorder=4)
    arrow(22.5, 35, 25.5, 35)
    ax.text(24, 36, "(D)", ha="center", va="bottom", fontsize=6.3, color=ALARM)

    panel(72, 22.5, 26, 19, fc="#f7f2ea", ec=MODEL_EC)
    ax.text(85, 40.0, "Joint certificate", ha="center", va="center", fontsize=6.8,
            fontweight="bold", zorder=4)
    ax.text(85, 36.6, f"coverage $\\geq$ {beta:.2f}  AND\nviolations $\\leq$ {cap:.2f}",
            ha="center", va="center", fontsize=6.1, zorder=4, linespacing=1.4)
    for yy, who, val, why in [(31.8, "as deployed", f"violation rate {vr_log:.2f}", "exceeds the cap"),
                              (26.2, "rule enforced", f"coverage {cov_enf:.2f}", "falls below the floor")]:
        ax.text(74.5, yy + 1.1, who, ha="left", va="center", fontsize=6.1, zorder=4)
        ax.text(74.5, yy - 0.9, f"{val} \u2014 {why}", ha="left", va="center",
                fontsize=5.8, color=ALARM, zorder=4)
    arrow(70.5, 32, 71.5, 32)
    ax.text(71, 33.2, "(E)", ha="center", va="bottom", fontsize=6.0, color=ALARM)

    # ---- section 3 ----
    ax.plot([2, 98], [18.5, 18.5], color="#cfd8dc", lw=0.8)
    ax.text(2, 15.5, "3. Why the verdict is evidence: every threshold fixed in advance, every control declared",
            ha="left", va="center", fontsize=8, fontweight="bold")
    items = [
        (3.0, 22, "Pre-registration", "all pass thresholds frozen\nbefore any test fold"),
        (27.5, 22, "Negative controls", f"{n_nc} declared with the outcome\neach must produce"),
        (52.0, 22, "Positive control", f"retrain on the guideline;\n{pc_share}% of the effect is arithmetic"),
        (76.5, 21.5, "Deviations log",
         f"{n_dev} numbered entries,\nincluding what we did not deliver"),
    ]
    for x, w, head, sub in items:
        ax.add_patch(FancyBboxPatch((x, 3.2), w, 8.4, boxstyle="round,pad=0.4,rounding_size=1.1",
                                    fc="white", ec="#c5cdd2", lw=0.8, zorder=3))
        ax.text(x + w / 2, 9.6, head, ha="center", va="center", fontsize=6.4,
                fontweight="bold", zorder=4)
        ax.text(x + w / 2, 6.0, sub, ha="center", va="center", fontsize=5.9,
                color=GREY, zorder=4, linespacing=1.35)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_overview.png", dpi=300, bbox_inches="tight")

    # non-vacuity: no two visible text boxes may overlap
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:30], b.get_text()[:30])
          for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    assert not ov, f"overlapping labels: {ov}"
    print(f"wrote {OUT/'fig_overview.pdf'} and .png; {len(texts)} labels, 0 overlaps")


if __name__ == "__main__":
    main()
