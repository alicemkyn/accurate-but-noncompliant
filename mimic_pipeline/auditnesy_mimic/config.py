"""Typed configuration, loaded from config.yaml with environment overrides.

Keeping every gated threshold in one place (and mirroring the ratified
pre-registration) makes the falsifiability claim auditable: the reviewer can
diff config.yaml against tag prereg-v1.0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]

    # --- resolved convenience accessors ---
    @property
    def mimic_hosp_dir(self) -> Path:
        return Path(os.environ.get("MIMIC_DIR", self.raw["data"]["mimic_hosp_dir"]))

    @property
    def out_dir(self) -> Path:
        return Path(os.environ.get("AUDITNESY_OUT", self.raw["data"]["out_dir"]))

    @property
    def prereg_tag(self) -> str:
        return self.raw["prereg_tag"]

    # sub-sections returned as plain dicts (callers index by key)
    @property
    def task(self) -> dict:
        t = self.raw["task"]
        # env override so the KDIGO-label ablation (diagnosis->fix loop) can run
        # without editing the committed default (label_source stays 'icd').
        if os.environ.get("AUDITNESY_LABEL"):
            t = {**t, "label_source": os.environ["AUDITNESY_LABEL"]}
        return t
    @property
    def constraints(self) -> dict: return self.raw["constraints"]
    @property
    def audit(self) -> dict: return self.raw["audit"]
    @property
    def experiment(self) -> dict: return self.raw["experiment"]
    @property
    def shift(self) -> dict: return self.raw["shift"]
    @property
    def models(self) -> dict: return self.raw["models"]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
