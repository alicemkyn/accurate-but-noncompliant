# W3PHIAI-27 revision — new analyses (v1.13): methods note

Date: 2026-10-31. Scope: the two analyses promised in the ML4H rebuttal, their tests,
and the claim-verification table. Both analyses are post hoc; no registered threshold
was changed; no pre-existing result value changed (verified byte-for-byte, see §B.3).

Trees edited identically: `NeSy/mimic_pipeline` (data-bearing, where everything was run) and
`latest/accurate-but-noncompliant-anon/mimic_pipeline` (code + aggregate results only; the
blinding key, packet and response CSVs remain absent there — checked after the copy).

## A. Adjudication × renal history × trajectory — `scripts/13_adjudication_by_stratum.py`

- Input: the 200-case blinding key (hadm_id/subject_id, cell, rule verdict) and the one
  returned response file, read in-process only; `results/adjudication_by_stratum.json` is
  aggregate-only and the script refuses to write if `hadm_id`, `subject_id` or `case_id`
  appears as a key at any depth.
- Renal-history stratum: `build_prior_indicators` imported from script 10 (same raw
  `admissions` / `diagnoses_icd` tables; prior admission must be discharged before the
  audited admittime) and `renal_history_labels` imported from script 12 (block-H partition:
  `prior_renal_replacement` / `no_prior_renal_replacement` / `prior_status_unobservable`).
- Trajectory class: script 05's `taxonomy_for_split` axis-A rule reproduced row-wise
  (`trajectory_class`: missing nadir → 0.0; A1 nadir ≥ 4.0; A2 nadir < 2.0; A3 otherwise).
  A test runs both on the same synthetic data and demands identical marginals.
- Blank handling identical to script 11: a blank is "cannot tell", excluded from the
  agreement denominator (NaN in a `nanmean`, never 0).
- Join: all 200 cases joined one-to-one to the cohort; the script would refuse on any
  unjoined case. Cross-checks against `results/clinician_adjudication.json`: decided 162,
  blank 38, per-cell n (27/63/52/20) and per-cell agreement all reproduced (tested).
- Tables written: per renal history, per trajectory, per cell, cell × renal history,
  cell × trajectory, renal history × trajectory; `small_cell_warning` (n_decided < 10) lists
  6 cells; `empty_combinations_zero_cases` names combinations with no sampled case so an
  absent row reads as zero, not as an omission.

Headline reading (all values in `results/adjudication_by_stratum.json`, key paths in
`claim_verification_w3phiai.csv`): agreement on decided cases is 0.962 / 0.929 / 0.868 across the
three renal-history strata (51/53, 52/56, 46/53) with blank shares 0.16 / 0.22 / 0.19; by
trajectory it is 1.000 (65/65) in A1 with the largest blank share (27/92 = 0.29), 0.880 (22/25)
in A2 with no blanks, 0.861 (62/72) in A3. The cell-C disagreements concentrate in A3 (10 of
12) and, within renal history, in `prior_status_unobservable` (7/14 agree).

## B. Subject-clustered CIs — `scripts/12_violation_rate_by_renal_history.py`

- New block `restricted_cohort_cluster_ci`, function `compute_cluster_cis`
  (module-level, imported by the tests). Cohort: severe cases in
  `no_prior_renal_replacement` + `prior_status_unobservable`.
- Row-level decisions are reconstructed inside the existing seed loop with the same
  definitions `per_split_subgroup_stats` uses (deployed = confident; violation =
  confident ∧ predicted no-AKI). The script asserts per seed that the row-level severe and
  deployed counts equal the count-level pooling already used by `restricted_certificate`
  (so the CI is for the same quantity the certificate reports).
- Resampling unit = subject_id. Per seed: all subjects of that seed's test set are drawn
  with replacement (as many as there are test subjects; subjects without a severe surviving
  case contribute a zero vector), the three statistics are recomputed from the resampled
  per-subject sums; 95% percentile interval; the median across the 20 seeds of (point, lo,
  hi) is reported, plus the full per-seed list (aggregates).
- Pooled: all 20 seeds' test decisions stacked (21,129 decisions, 3,163 subjects, 3,148 of
  whom appear in more than one seed's test set); clusters = subject across the stack;
  labelled DESCRIPTIVE in the payload and in the method string.
- B = 1000, read from `experiment.bootstrap_B` in `config.yaml` (the brief's "2000" fallback
  applies only when the key is absent; it is present). Seed 0. Both recorded in the payload.
- Guard: `missing_cluster_ci_keys(payload)` is appended to the existing refusal list; the
  script exits without writing if the block is missing or any lo/hi/point is None.
- Regression: `restricted_certificate`, `headline_comparison` and `per_stratum` are
  identical (dict equality) to `tests/fixtures/violation_rate_by_renal_history.v1_12_frozen.json`,
  a copy taken before the first re-run. The second full run (after comment-only edits)
  reproduced the output JSON byte-identically (`cmp`).

Result: per-seed median violation rate 0.528 [0.492, 0.566], coverage 0.737 [0.710, 0.765],
ICD agreement 0.574 [0.542, 0.607]; pooled (descriptive) violation 0.529 [0.509, 0.548],
coverage 0.737 [0.723, 0.750], ICD agreement 0.577 [0.561, 0.595]. The restricted-cohort
verdict (FAIL on violation, cap 0.05) is unchanged under either uncertainty reading.

## C. Claim-verification table — `claim_verification_w3phiai.csv`

Built by `scripts/build_claim_verification_w3phiai.py`; 206 rows, 0 missing. Every value is
resolved programmatically from a JSON key path in one of: `mimic_pipeline/results/
violation_rate_by_renal_history.json`, `.../clinician_adjudication.json`,
`.../adjudication_by_stratum.json`, `eicu_pipeline/results/heldout_hospital_certificate.json`.
Rows whose key cannot be resolved are kept with status `MISSING` rather than dropped (none
at present). Re-run the builder after any results change.

## D. Log and README

`preregistration_v14.md` gained a `## v1.13` entry in both trees (headings contiguous
v1.5…v1.13; every number in the entry was formatted from the results JSON by the script that
wrote it, not typed). `README.md` in both trees: the stated version range of the log was
updated from "v1.4 … v1.12" to "v1.4 … v1.13" — required by
`tests/test_deviations_log_integrity.py::test_readme_version_range_matches_log`. This README
edit is the one change outside the files the brief named.

## E. Deviations from the brief

1. B = 1000 (config value), not 2000 — the brief said "B from config if present".
2. The per-seed bootstrap draws from ALL test subjects of the seed (as the brief states),
   not only from subjects contributing a severe surviving case; the latter would fix the
   number of clusters per draw. The choice is stated in the method string.
3. README.md edited in both trees (version range), see §D.
4. Script 13 also emits `per_cell`, `renal_history_x_trajectory` and
   `empty_combinations_zero_cases`, beyond the tables the brief listed.
5. The frozen v1.12 copy lives in `tests/fixtures/` (aggregate JSON, no identifiers) so the
   regression test is runnable in the anonymous tree.
6. Nothing was committed in either tree (git structures are write-protected in this
   session); the working trees carry the changes.

## F. Mutation evidence

Procedure for every row: apply the mutation to the script (or, for file-level tests, to
the results JSON), regenerate where marked, run the single test, then restore from the
pristine copy and confirm `diff -q` silent and MD5 equal. Final state of both files
byte-identical to the versions that produced the shipped results.

### `tests/test_adjudication_by_stratum.py` (8 tests; script `13_adjudication_by_stratum.py`)

| Mutation | Regenerated JSON? | Test targeted | Outcome |
|---|---|---|---|
| blank scored as 0 (`np.nan` → `0.0` in `score_cases`) | no (function-level) | `test_blank_is_never_scored_as_disagreement` | 1 failed |
| same mutation, payload regenerated | yes | `test_decided_plus_blank_equals_n_in_every_cell` | 1 failed |
| drop `prior_status_unobservable` row from `per_renal_history` | yes | `test_strata_partition_the_adjudicated_cases` | 1 failed |
| `n_blank` miscounted (`n − n_dec − 1`) | yes | `test_decided_plus_blank_equals_n_in_every_cell` | 1 failed |
| identifier key added to payload AND guard removed | yes | `test_no_identifier_key_at_any_depth` | 1 failed |
| identifier key added, guard kept | attempted | — | script refused to write ("identifier key 'hadm_id' in payload; refusing to write"); stale good JSON untouched, test passed — the guard, not the test, caught it |
| trajectory A2 cut 2.0 → 1.5 | no | `test_trajectory_classes_match_script_05_taxonomy_rule` | 1 failed |
| `SMALL_CELL_N_DECIDED` 10 → 0 | yes | `test_small_cells_are_flagged` | 1 failed |
| agreement scored against the ICD code instead of the rule | yes | `test_per_cell_counts_agree_with_the_scored_adjudication` | 1 failed |

Restore: `diff -q` silent for script and JSON; MD5 `ac0d02810fbb3042ea8571cc2e87fb7c` before = after
(script at the time of verification; later edits to the script were none). 8 passed after restore.

### `tests/test_restricted_cluster_ci.py` (7 tests; script `12_violation_rate_by_renal_history.py`)

| Mutation | Where | Test targeted | Outcome |
|---|---|---|---|
| per-seed lo/hi swapped in `compute_cluster_cis` | script | `test_compute_cluster_cis_brackets_the_point_on_synthetic_data` | 1 failed |
| pooled point shifted +0.5 outside its interval | script | same | 1 failed |
| `missing_cluster_ci_keys` returns [] whenever the block exists | script | `test_guard_refuses_missing_block_or_none_bound` | 1 failed |
| guard result computed but not appended to `missing` | script | `test_guard_is_wired_to_the_refusal_path` | 1 failed |
| `per_seed_median.coverage_severe.lo_median` set above the point | JSON | `test_every_interval_is_ordered_and_contains_its_point` | 1 failed |
| `per_seed[5].violation_rate_deployed.hi` set below the point | JSON | same | 1 failed |
| method string: "subject" → "patient-row" | JSON | `test_method_string_names_subject_clustering` | 1 failed |
| `restricted_certificate.violation_rate.p95` 0.5626 → 0.5627 | JSON | `test_pre_existing_blocks_are_unchanged_from_the_frozen_v1_12_copy` | 1 failed |
| `headline_comparison.violation_rate_surviving_cases` → 0.53 | JSON | same | 1 failed |
| `B` → 500 | JSON | `test_b_and_seed_recorded_and_match_config` | 1 failed |
| whole `restricted_cohort_cluster_ci` block removed | JSON | `test_every_interval_is_ordered_and_contains_its_point` | 1 failed |

Restore: `diff -q` silent for script and JSON; MD5 script `def4882465dd139163b5c65d7c11df57`,
JSON `3ad544f68f956a70e4f1f545ca2361ca` before = after. (The script was subsequently edited
twice — a one-line comment relocation and a docstring paragraph, no logic — and re-run; the
re-run JSON is byte-identical to the one the mutations were verified against.)

## G. Test counts

- NeSy tree: 74 passed before this work → **89 passed, 0 failed, 0 skipped** after
  (+8 adjudication-by-stratum, +7 restricted-cluster-CI).
- Anonymous tree: **66 passed, 23 skipped, 0 failed**; all 15 new tests pass there. The 23
  skips are pre-existing and expected in that tree (paper source, cohort parquet, blinding key
  and packet are deliberately absent).
