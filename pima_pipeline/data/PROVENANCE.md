# Pima Indians Diabetes — data provenance

**File:** `pima_diabetes.csv` (768 rows + header, 9 columns)
**SHA-256:** `b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac`
**MD5:** `b43dd020fa775d93fced49f959c227ed`

## Where it came from, and why not from UCI directly

The paper cites this dataset as "UCI Machine Learning Repository (Pima Indians
Diabetes)". The dataset was **withdrawn from the UCI repository** and is no
longer served there; the UCI "Diabetes" entry (dataset id 34) is a *different*
dataset (longitudinal insulin/glucose records for 70 patients), not this one.
We record that here rather than citing a URL that does not resolve to the data
actually used.

The file was therefore obtained from two independent public mirrors and
**cross-checked byte-for-byte**:

1. `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv` (headerless)
2. `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv` (with header)

After stripping the header and CRLF line endings, the two mirrors are
**identical row-for-row**. The committed copy is mirror (2) with line endings
normalised to LF.

## Integrity checks against the canonical description

| Property | Canonical | This file |
|---|---|---|
| Records | 768 | 768 |
| Positive (Outcome=1) | 268 | 268 |
| Negative (Outcome=0) | 500 | 500 |
| Columns | 9 (8 features + outcome) | 9 |

## Column semantics

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome`

`Glucose` is a 2-hour oral-glucose-tolerance-test plasma glucose concentration
(mg/dL). The audit constraints are phrased in terms of *fasting* glucose
thresholds from diabetes diagnostic guidance, so the constraint thresholds and
the measured quantity are not the same clinical variable. This is a limitation
of using this benchmark to illustrate the protocol, and the rules should be
read as a mechanical demonstration rather than as valid diabetes criteria.

**Status of this caveat in the paper:** as of this file's creation the paper
did NOT carry it. It is being added to the Pima appendix in the same revision
that restores the Pima tables; if you are reading this and the appendix does
not mention the OGTT-vs-fasting mismatch, that edit was lost and should be
re-applied. `scripts/10_pima_audit.py` also writes the caveat into
`results/pima_audit.json` under `clinical_validity_caveat`, so it travels with
the numbers regardless.

Physiologically impossible zeros in `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin` and `BMI` are treated as missing and median-imputed
by the pipeline (see `scripts/10_pima_audit.py`); zeros in `Pregnancies` are
genuine and are left alone.
