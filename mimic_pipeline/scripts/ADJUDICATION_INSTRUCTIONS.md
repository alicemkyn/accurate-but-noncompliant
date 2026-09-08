# Reviewer instructions — AKI case adjudication

**Time required:** about 60–90 minutes for 200 cases (roughly 20–25 seconds each).
**What we need from you:** one column filled in. Nothing else.

## What this is for

We are auditing a machine-learning risk model. The audit checks whether the
model's confident *no-AKI* outputs disagree with a guideline-derived rule that
**we** wrote by operationalising KDIGO creatinine criteria. Every conclusion in
our paper is conditional on that rule being a reasonable reading of the
guideline — and no clinician has checked it. That is what you are checking.

You are **not** reviewing the model, and you will not see its outputs. You are
answering one question per case, and we compare your answers to our rule.

## The question

For each row: **based on the information shown, would you call this acute kidney
injury?**

Fill in `clinician_aki`:

| Value | Meaning |
|---|---|
| `1` | Yes — I would call this AKI |
| `0` | No — I would not call this AKI |
| leave blank | I genuinely cannot tell from this information |

Blank is a legitimate answer and is treated as missing, not as `0`. Please do not
guess to fill a row.

Optionally also fill in:

- `clinician_confidence` — `high`, `medium` or `low`
- `clinician_notes` — free text, most useful when you think the case is
  ambiguous or when the data shown is insufficient

## What each column means

| Column | Meaning |
|---|---|
| `case_id` | Row identifier. Please do not reorder or delete rows. |
| `age`, `sex` | Patient age in years, M/F |
| `creat_baseline_mgdl` | Lowest creatinine in the first 24 h, used as the baseline |
| `creat_peak_24h_mgdl` | Highest creatinine in the first 24 h |
| `creat_nadir_24h_mgdl` | Lowest creatinine in the first 24 h (same as baseline; shown for convenience) |
| `absolute_rise_mgdl` | peak − nadir |
| `peak_over_nadir` | peak ÷ nadir |
| `bun`, `sodium`, `potassium`, `bicarbonate`, `hemoglobin`, `wbc` | First value in the first 24 h, standard units |

## What you are deliberately not being shown, and why

You will not see the diagnosis code the hospital assigned, our rule's verdict, or
the model's prediction. This is a blinded review: if you saw any of those, your
answers would no longer be independent evidence about the rule.

You will also not see the clinical narrative, imaging, urine output, medications
or fluid balance. **This is a real limitation and we want you to feel it.** The
audit rule itself only has access to these variables, so we are asking whether
*this information* supports calling AKI. If you find that insufficient for a
given case, leaving it blank and saying so in the notes is exactly the finding we
need — it tells us the rule is being asked to decide on inadequate evidence.

## Important: we are not looking for a particular answer

We committed in advance, in code, to what would count as your **disagreeing**
with us — and we have written the paper so that either outcome is publishable. If
your answers agree with our rule, we will describe the rule as
clinician-adjudicated. If they do not, we will report that our compliance
verdicts rest on an operationalisation clinicians do not endorse, and narrow the
paper's claims accordingly. The thresholds are fixed in the analysis script and
cannot be adjusted after seeing your answers.

Please do not try to infer what we hope for and do not discuss cases with the
other reviewers until all responses are returned. Independent disagreement
between reviewers is itself informative — we report inter-reviewer agreement
separately.

## How to return it

Fill in the CSV as-is and send it back. Please keep `case_id` intact — it is how
we match your answers to the cases. If your tool changes number formatting, that
is fine; only `case_id` and `clinician_aki` are parsed.

## Data handling

These are de-identified rows from a public research database (MIMIC-IV, PhysioNet
credentialed access). Please do not redistribute the file. Delete your copy once
you have returned it.

## Authorship and acknowledgement

Tell us which you prefer, and we will do it: co-authorship on the paper, a named
acknowledgement, or an anonymous acknowledgement. If you would rather not be
named at all, that is fine too.

## Questions we expect

**"Can I see more of the chart?"** — No, and that is intentional: the rule under
test only sees these variables. If that makes a case unanswerable, leave it blank
and note it.

**"Some of these are obviously not AKI."** — Yes. The sample deliberately
includes clear negatives as a floor. If clear negatives are called AKI, our rule
is mis-specified in the other direction, and we need to know.

**"The baseline is a 24-hour nadir, not a pre-admission value."** — Correct, and
it is a limitation we state in the paper. Please adjudicate on what is shown.

**"Is 200 enough?"** — It is powered for the comparison we pre-registered. If you
think the sample is composed wrongly, tell us; the stratification is documented
and we will report your objection.
