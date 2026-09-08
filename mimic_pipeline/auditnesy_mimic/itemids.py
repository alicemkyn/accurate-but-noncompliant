"""STEP 0 — verify MIMIC-IV lab itemids against d_labitems before anything else.

The itemids below are the commonly-used MIMIC-IV v2.2/v3.1 chemistry values, but
they MUST be confirmed on your local d_labitems (the AKI label and the T2
constraint both depend on the creatinine itemid being correct).
"""
from __future__ import annotations

import pandas as pd

from .io_utils import read_table

# Commonly-used MIMIC-IV hosp itemids (Blood/Chemistry). VERIFY LOCALLY.
DEFAULT_ITEMID = {
    "creatinine": 50912,
    "glucose": 50931,
    "bun": 51006,
    "potassium": 50971,
    "sodium": 50983,
    "hemoglobin": 51222,
    "wbc": 51301,
    "bicarbonate": 50882,
}

# label substrings used to search d_labitems for confirmation
SEARCH_TERMS = {
    "creatinine": "creatinine",
    "glucose": "glucose",
    "bun": "urea nitrogen",
    "potassium": "potassium",
    "sodium": "sodium",
    "hemoglobin": "hemoglobin",
    "wbc": "white blood",
    "bicarbonate": "bicarbonate",
}


def verify_itemids(mimic_dir) -> pd.DataFrame:
    """Return the d_labitems rows matching each target, preferring Blood fluid."""
    d = pd.read_csv(read_table(mimic_dir, "d_labitems"))
    rows = []
    for key, term in SEARCH_TERMS.items():
        hits = d[d["label"].str.contains(term, case=False, na=False)].copy()
        if "fluid" in hits.columns:
            hits["_blood"] = hits["fluid"].str.contains("blood", case=False, na=False)
            hits = hits.sort_values("_blood", ascending=False)
        for _, r in hits.iterrows():
            rows.append({
                "target": key,
                "default_itemid": DEFAULT_ITEMID.get(key),
                "found_itemid": r.get("itemid"),
                "label": r.get("label"),
                "fluid": r.get("fluid"),
                "category": r.get("category"),
                "matches_default": r.get("itemid") == DEFAULT_ITEMID.get(key),
            })
    return pd.DataFrame(rows)
