"""File location, output directories, and deterministic hashing helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_table(mimic_dir: os.PathLike | str, name: str) -> Path:
    """Locate a MIMIC table that may be .csv or .csv.gz."""
    mimic_dir = Path(mimic_dir)
    for ext in (".csv.gz", ".csv"):
        p = mimic_dir / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {name}.csv[.gz] in {mimic_dir}. "
        f"Set MIMIC_DIR to your MIMIC-IV 'hosp' folder."
    )


def ensure_out(out_dir: os.PathLike | str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_hash(*arrays: np.ndarray) -> str:
    """Stable short hash over model parameter arrays (for T1)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_json(path: os.PathLike | str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
