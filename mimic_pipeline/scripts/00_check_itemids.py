#!/usr/bin/env python
"""STEP 0 — verify MIMIC-IV lab itemids before building the cohort.

    export MIMIC_DIR=/path/to/mimiciv/hosp
    python scripts/00_check_itemids.py

Confirm the printed creatinine itemid matches DEFAULT_ITEMID in itemids.py
(it defines both the AKI constraint and, optionally, the label).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditnesy_mimic.config import load_config
from auditnesy_mimic.itemids import verify_itemids

if __name__ == "__main__":
    cfg = load_config()
    table = verify_itemids(cfg.mimic_hosp_dir)
    import pandas as pd
    pd.set_option("display.max_rows", None, "display.width", 160)
    print(table.to_string(index=False))
    # Warn only if a target has NO exact-match row on a Blood fluid. Extra
    # alternative itemids (whole-blood, seldom-populated 52xxx duplicates) are
    # expected and are NOT a problem -- the default is the canonical serum assay.
    missing = [t for t in table["target"].unique()
               if table[(table["target"] == t) & table["matches_default"]].empty]
    if missing:
        print(f"\n[!] No exact default match found for: {missing} — update itemids.py.")
    else:
        print("\n[ok] All 8 default itemids have a canonical Blood match — proceed to STEP 1.")
