# Pima demonstration pipeline

Reconstruction of the paper's Demonstration I (Appendix "Pima Demonstration:
A Model That Passes"). The original harness was never committed — no code, no
results, and nothing in the repository history — so this directory was written
from the specification in the paper and is released so that every Pima number
is traceable to code anyone can re-run.

## Layout

```
data/     pima_diabetes.csv     the benchmark (768 records), with PROVENANCE.md
scripts/  10_pima_audit.py      three-axis audit + multi-seed + bootstrap + negative controls
          11_pima_styles.py     pipeline-coupled vs differentiable (LTN) integration styles
          12_pima_probe.py      concept-semantics probe (glucose sweep)
results/  pima_audit.json       everything behind tab:pima_stats and tab:pima_nc
          pima_styles.json      everything behind tab:pima_styles
          pima_probe.json       the probe sensitivity quoted in the appendix
tests/    test_pima_t1.py       non-vacuity tests for the T1 replay check
          test_results_selfconsistency.py
                                every verdict word in a results file must be
                                recomputable from the numbers beside it
```

Each of the ten tests was verified by mutation: a defect was introduced, the
named test was observed to fail, and the file was restored byte-for-byte
(checked with `diff -q`) before the next mutation. The table below is the
record — if you add a test here, add its mutation too, or the suite starts
carrying tests nobody has shown can fail.

| Test | Mutation that makes it fail |
|---|---|
| `test_pima_t1::..._hash_matches_on_identical_refit` | make `fit_model` shuffle its training rows, so a refit is not identical |
| `test_pima_t1::..._hash_differs_on_different_fit` | make `param_hash` return a constant |
| `test_pima_t1::..._ltn_arm_hash_is_measured_not_asserted` | make `DifferentiableLTN.param_hash` return a constant |
| `test_pima_t1::test_styles_script_does_not_hardcode_a_t1_pass` | write `"T1_hash_match": True` as a literal |
| `test_selfconsistency::..._records_the_calibration_actually_used` | write `"calibration": "isotonic"` into the results file |
| `test_selfconsistency::..._negative_control_does_not_claim_a_calibrated_baseline` | restore the "applied to the calibrated probabilities" wording |
| `test_selfconsistency::..._t3_verdict_follows_from_its_own_ece` | flip `axes_reference_split.T3_pass` against its own ECE |
| `test_selfconsistency::..._styles_arm_verdicts_follow_from_their_own_ece` | set the LTN arm's `T3_pass` to `true` against its ECE of 0.111 |
| `test_selfconsistency::..._summary_sentence_matches_the_measured_verdicts` | write "Both arms ... pass T1 and T3" into `reading` |
| `test_selfconsistency::..._summary_is_not_a_hardcoded_literal` | replace the `_reading(...)` call with an inline string |

One mutation (flipping the LTN `T3_pass`) fails two tests rather than one,
since the summary sentence is derived from that field.

## Caught defects worth recording

Three defects of the same shape were found in review of this reconstruction and
are recorded here rather than quietly fixed, since a paper arguing for auditable
evidence should show its own audit trail.

**A results file described the wrong model.** `pima_audit.json` recorded the
reference model as "isotonic-calibrated" while the script deliberately runs it
uncalibrated — the wrong provenance for the very model whose ECE and T3 verdict
that file reports. The string is now derived from the `CALIBRATE` constant, so
it cannot drift from what ran, and the file also carries an explicit
`protocol.calibration` field.

**A summary contradicted its own data.** `pima_styles.json` asserted "Both arms
... pass T1 and T3" while the same file recorded an LTN ECE of 0.111 against a
0.10 bound. The sentence is now built by `_reading()` from the measured
verdicts, and each arm carries an explicit `T3_pass` field.

**A verdict was asserted rather than measured** — described below.



The first version of `11_pima_styles.py` wrote `"T1_hash_match": true` for the
differentiable arm as a literal, without ever comparing two hashes: it computed
one hash and never refit. A durable results file therefore carried a
reproducibility verdict that had not been measured — precisely the failure mode
this paper argues against. The script now refits independently from the same
seed and compares, and `tests/test_pima_t1.py` guards it in both directions:
the hash must match an identical refit **and** must differ under a changed seed
or epoch count, so a `param_hash` that returned a constant would fail. A
source-level guard also fails if any `T1_hash_match` line is ever set to a
literal again. This was verified by mutation: reintroducing the literal makes
the guard fail, and restoring the file makes it pass.

## Reproducing

```bash
python scripts/10_pima_audit.py     # ~10 s
python scripts/11_pima_styles.py    # needs torch
python scripts/12_pima_probe.py
```

No credentialed access is required: the benchmark is public. The scripts import
`ece` and `bootstrap_ci` from `mimic_pipeline/auditnesy_mimic/metrics.py`, and
`11` imports `DifferentiableLTN` from that package's `models.py`, so both
demonstrations in the paper are judged by the same metric code.

## How the original configuration was recovered

Two reported quantities depend only on the split, not on the model: the
soft-constraint firing counts C1 = 94/231 and C2 = 137/231. Sweeping
`train_test_split` seeds 0–399 across {stratified, unstratified} and
{train-median, full-median} imputation, exactly four configurations reproduce
both counts — seeds 42 and 277 under stratification. Fitting candidate models
on each, **seed 42 with an uncalibrated logistic regression on standardised
features** reproduces the paper's headline triple to three decimals (AUROC
0.836, accuracy 0.745, ECE 0.070); seed 277 reaches only ≈0.80 AUROC. Seed 42
is therefore the reference split.

The multi-seed sweep is seeds 0–19, evaluated separately from the reference
split: with that range the mean hard-violation count is exactly 0.0500 (= 1/20)
and mean ECE 0.0481, matching the paper's 0.05 and 0.047.

## What reproduced, and what did not

| Quantity | Status |
|---|---|
| Reference AUROC / accuracy / ECE | reproduced exactly (0.836 / 0.745 / 0.070) |
| C1 / C2 firing counts | reproduced exactly (94 / 137 of 231) |
| Bootstrap ECE interval | reproduced ([0.041, 0.123] vs. drafted [0.040, 0.123]) |
| Multi-seed AUROC / accuracy | reproduced (0.839 / 0.764) |
| Negative control NC-B | reproduced (AUROC → 0.753, 8 violations) |
| Concept-semantics probe | reproduced (0.074 → 0.846, sensitivity 0.772) |
| Negative control NC-A magnitude | **differs**: ECE → 0.199, drafted 0.220. The sharpening factor was never recorded; this uses ×3 on the logit. Direction and verdict are unchanged (T3 flags it). |
| Abstention rates | **differ**: 0.186 / 0.306 vs. drafted 0.163 / 0.257. The covariate-shift definition was never recorded; this shifts glucose by one train-split SD. The shift-sensitivity conclusion is unchanged. |
| Differentiable (LTN) arm | **not reproduced**. No hyper-parameters were recorded, and the drafted four values (AUROC 0.831, acc 0.745, ECE 0.089, satisfaction 0.887) cannot be produced jointly at any epoch count from 100 to 1200. This is a fresh run at the implementation's defaults, and the paper says so. |

In every case where the reconstruction disagreed with the draft, **the paper was
corrected to match this code** — never the other way round.

## A caveat that belongs with the numbers

Pima's `Glucose` column is a two-hour OGTT value, while the 126 and 180 mg/dL
thresholds come from *fasting* glucose guidance. The rules illustrate the
protocol's mechanics on a public benchmark; they are not valid diabetes
criteria. Both `10_pima_audit.py` and `11_pima_styles.py` write this caveat
into their output JSON so it travels with the numbers.
