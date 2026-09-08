"""AuditNeSy on MIMIC-IV: falsifiable accountability evaluation harness.

Package layout (each concern is a separate module):
    config              - typed config, loaded from config.yaml + env (mirrors prereg-v1.0)
    io_utils            - MIMIC table locator, output paths, hashing
    itemids             - verify lab itemids against d_labitems (STEP 0)
    cohort              - build AKI cohort; ICD label (independent of constraint feature)
    metrics             - ECE, subject-level bootstrap CI, abstention
    models              - pipeline-coupled MLP, differentiable LTN, pure-neural baseline
    audit               - T1 reproducibility, T2 constraint compliance, T3 safe fallback, T2s probe
    negative_controls   - NC-A' / NC-B' / NC-C' / NC-D' (natural deficiencies)
    experiment          - 20-seed x split orchestration + bootstrap aggregation + summary

Run order (see scripts/):
    00_check_itemids.py -> 01_build_cohort.py -> 02_run_experiment.py
"""

__version__ = "1.0.0"
__prereg__ = "prereg-v1.0"
