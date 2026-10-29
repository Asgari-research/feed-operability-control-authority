#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core reduced-order calculations for the P10 feed-composition robustness study."""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import time
import subprocess
import traceback
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import warnings

# Keep numerical libraries from unexpectedly creating many threads when the
# user has explicitly asked for controlled parallelism via --n-jobs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch

from scipy.integrate import quad
from scipy.optimize import least_squares, brentq, minimize, minimize_scalar
from scipy.stats import median_abs_deviation, spearmanr
from joblib import Parallel, delayed

try:
    import openpyxl  # noqa: F401
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False


# ---------------------------------------------------------------------------
# Global constants and publication-independent defaults
# ---------------------------------------------------------------------------
R_J_MOL_K = 8.31446261815324
T_REF_K = 298.15
EPS = 1.0e-12
PRESSURE_FLOOR_BAR = 1.0e-8
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Molecular weights are only used when an input loading is mass-based.  The
# normal preferred input is mol/kg (numerically identical to mmol/g).
MOLECULAR_WEIGHT_G_MOL = {
    "CO2": 44.0095,
    "N2": 28.0134,
    "CH4": 16.0425,
    "H2": 2.01588,
    "CO": 28.0101,
    "O2": 31.9988,
    "H2O": 18.01528,
}

MATERIAL_ALIASES = {
    "calf20": "CALF-20",
    "calf_20": "CALF-20",
    "calf-20": "CALF-20",
    "calgaryframework20": "CALF-20",
    "mgmof74": "Mg-MOF-74",
    "mg_mof_74": "Mg-MOF-74",
    "mg-mof-74": "Mg-MOF-74",
    "mof74mg": "Mg-MOF-74",
    "utsa16": "UTSA-16",
    "utsa_16": "UTSA-16",
    "utsa-16": "UTSA-16",
}

GAS_ALIASES = {
    "co2": "CO2",
    "carbon_dioxide": "CO2",
    "carbondioxide": "CO2",
    "n2": "N2",
    "nitrogen": "N2",
    "ch4": "CH4",
    "methane": "CH4",
    "h2": "H2",
    "hydrogen": "H2",
    "co": "CO",
    "carbon_monoxide": "CO",
    "o2": "O2",
    "oxygen": "O2",
    "h2o": "H2O",
    "water": "H2O",
}

# Column aliases are intentionally broad: literature data often arrive with
# human-readable headers rather than a single database schema.
COLUMN_ALIASES = {
    "material": ["material", "adsorbent", "framework", "mof", "sample", "solid"],
    "material_form": ["material_form", "form", "sample_form", "particle_form", "shaping"],
    "adsorbate": ["adsorbate", "gas", "component", "species", "adsorptive"],
    "temperature_K": ["temperature_k", "temperature", "temp_k", "t_k", "temp"],
    "pressure_bar": ["pressure_bar", "pressure", "p_bar", "p", "equilibrium_pressure"],
    "loading_mol_kg": ["loading_mol_kg", "loading", "uptake", "q", "adsorbed_amount", "amount_adsorbed"],
    "uncertainty": ["uncertainty", "sigma", "std", "standard_deviation", "error", "loading_uncertainty"],
    "source_id": ["source_id", "dataset_id", "reference_id", "ref_id", "citation_key"],
    "doi": ["doi", "digital_object_identifier"],
    "table_figure_si_location": ["table_figure_si_location", "source_location", "location", "figure_table"],
    "extraction_method": ["extraction_method", "extraction", "method"],
    "allowed_use": ["allowed_use", "use", "role"],
    "notes": ["notes", "comment", "comments"],
    "kinetic_quantity": ["kinetic_quantity", "transport_quantity", "kinetic_metric"],
    "kinetic_value": ["kinetic_value", "transport_value", "diffusion_value", "rate_value"],
    "kinetic_units": ["kinetic_units", "transport_units", "rate_units"],
}

SUPPORTED_INPUT_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle"}


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
SAVE_MODES: Dict[str, Dict[str, Any]] = {
    "minimal": {
        "si_figures": False, "pdf": False, "svg": False,
        "panel_csv": False, "bootstrap_samples": False,
        "excel": False, "parquet": False, "dpi": 180,
    },
    "light": {
        "si_figures": True, "pdf": False, "svg": False,
        "panel_csv": False, "bootstrap_samples": False,
        "excel": False, "parquet": False, "dpi": 220,
    },
    "balanced": {
        "si_figures": True, "pdf": True, "svg": False,
        "panel_csv": True, "bootstrap_samples": False,
        "excel": True, "parquet": False, "dpi": 300,
    },
    "rich": {
        "si_figures": True, "pdf": True, "svg": True,
        "panel_csv": True, "bootstrap_samples": True,
        "excel": True, "parquet": True, "dpi": 350,
    },
    "archival": {
        "si_figures": True, "pdf": True, "svg": True,
        "panel_csv": True, "bootstrap_samples": True,
        "excel": True, "parquet": True, "dpi": 450,
    },
}

RAM_MODES: Dict[str, Dict[str, Any]] = {
    "ultra_light": {"batch": 8, "parallel_cap": 1, "float_dtype": "float32", "cache_models": False},
    "light":       {"batch": 16, "parallel_cap": 2, "float_dtype": "float32", "cache_models": False},
    "moderate":    {"batch": 32, "parallel_cap": 4, "float_dtype": "float64", "cache_models": True},
    "normal":      {"batch": 64, "parallel_cap": None, "float_dtype": "float64", "cache_models": True},
}

LEVELS: Dict[str, Dict[str, Any]] = {
    # Installation/preflight science smoke test only.  Do not use for manuscript claims.
    "quickcheck": {
        "models": ["langmuir", "dual_site_langmuir"], "n_starts": 3, "n_boot": 12,
        "composition_n": 25, "pressure_n": 9, "uncertainty_draws_atlas": 8,
        "bootstrap_models": "best", "loto": False, "loso": False,
    },
    # Rough retains model competition so the nearly single-site N2 reconstruction
    # is not forced into a structurally over-parameterized dual-site fit.
    "rough": {
        "models": ["langmuir", "dual_site_langmuir"], "n_starts": 4, "n_boot": 24,
        "composition_n": 31, "pressure_n": 17, "uncertainty_draws_atlas": 24,
        "bootstrap_models": "best", "loto": False, "loso": False,
    },
    "standard": {
        "models": ["langmuir", "dual_site_langmuir"], "n_starts": 8, "n_boot": 100,
        "composition_n": 61, "pressure_n": 27, "uncertainty_draws_atlas": 64,
        "bootstrap_models": "best", "loto": True, "loso": False,
    },
    "comprehensive": {
        "models": ["langmuir", "dual_site_langmuir", "sips", "toth"], "n_starts": 16, "n_boot": 300,
        "composition_n": 121, "pressure_n": 41, "uncertainty_draws_atlas": 320,
        "bootstrap_models": "retained", "loto": True, "loso": True,
    },
    "exhaustive": {
        "models": ["langmuir", "dual_site_langmuir", "sips", "toth"], "n_starts": 32, "n_boot": 800,
        "composition_n": 201, "pressure_n": 61, "uncertainty_draws_atlas": 1000,
        "bootstrap_models": "retained", "loto": True, "loso": True,
    },
}


@dataclass(frozen=True)
class EngineeringSettings:
    """Transparent assumptions for the CALF-20 feed-composition robustness and minimal adaptation layer."""
    nominal_temperature_K: float = 295.15
    adsorption_pressure_bar: float = 0.97
    nominal_feed_co2_fraction: float = 0.15
    # Exploratory numerical envelope.  The manuscript's primary evidence-supported
    # composition window is narrower (5--30% CO2) and is tracked separately.
    feed_co2_min: float = 0.02
    feed_co2_max: float = 0.40
    primary_evidence_feed_min: float = 0.05
    primary_evidence_feed_max: float = 0.30
    nominal_regeneration_pressure_bar: float = 0.05
    regeneration_pressure_floor_bar: float = 0.01
    nominal_adsorption_time_s: float = 120.0
    fixed_nonfeed_cycle_time_s: float = 180.0
    nominal_purity_anchor: float = 0.955
    nominal_recovery_anchor: float = 0.881
    target_purity: float = 0.95
    target_recovery: float = 0.88
    # Threshold sensitivity is a robustness analysis, not a second optimization
    # objective.  Baseline 95/88 remains the headline calibration-linked target.
    purity_threshold_sensitivity: Tuple[float, ...] = (0.90, 0.95, 0.97)
    recovery_threshold_sensitivity: Tuple[float, ...] = (0.85, 0.88, 0.90)
    vacuum_efficiency: float = 0.60
    bed_void_fraction: float = 0.38
    adsorbent_bulk_density_kg_m3: float = 650.0
    product_reference_co2_fraction: float = 0.95
    timing_min_scale: float = 0.30
    timing_max_scale: float = 1.60
    purge_max_fraction: float = 0.15
    modest_composition_extrapolation_fraction: float = 0.05
    class_probability_threshold: float = 0.80
    uncertainty_ci: float = 0.95
    vacuum_efficiency_range: Tuple[float, float] = (0.45, 0.75)
    bulk_density_range_kg_m3: Tuple[float, float] = (500.0, 800.0)
    front_factor_range: Tuple[float, float] = (0.80, 1.20)
    contamination_scale_range: Tuple[float, float] = (0.70, 1.30)


@dataclass
class RunConfig:
    input_folder: str
    output_folder: str
    save_mode: str
    ram_mode: str
    level: str
    n_jobs: int
    seed: int
    force_stage: Optional[str]
    restart: bool
    preflight_only: bool
    uncertainty_draws_override: Optional[int]
    engineering: EngineeringSettings
    css_mode: str = "validate_and_falsify"
    css_run_on_gate_fail: bool = False

    def scientific_hash_payload(self) -> Dict[str, Any]:
        """Fields whose change should invalidate scientific checkpoints."""
        return {
            "level": self.level,
            "seed": self.seed,
            "engineering": asdict(self.engineering),
            "pipeline_version": PIPELINE_VERSION,
            # Invalidate scientific checkpoints if the pipeline source itself changes,
            # even when the semantic version string is accidentally left unchanged.
            "pipeline_source_sha256": sha256_file(Path(__file__).resolve()),
            "uncertainty_draws_override": self.uncertainty_draws_override,
            "css_mode": self.css_mode,
            "css_run_on_gate_fail": self.css_run_on_gate_fail,
        }


PIPELINE_VERSION = "6.6.0"
MATCHED_DATA_VERSION = "4.0.0"
# Final analysis settings include continuous one-actuator effort, adaptive source sensitivity, and optional cyclic validation scaffolding.
# Checkpoints are configuration-hashed and may be resumed only when compatible with the current run.
LEGACY_SCIENCE_RELEASES = []
STAGES = [
    "00_audit", "01_harmonize", "02_fit_models", "03_uncertainty",
    "04_composition_evidence", "05_feed_composition_robustness", "06_robustness",
    "07_validation", "08_tables", "09_figures", "10_finalize",
]
SCIENCE_STAGES = tuple(STAGES[:8])


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_token(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("₂", "2").replace("₄", "4")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def normalize_gas(x: Any) -> str:
    t = normalize_token(x)
    return GAS_ALIASES.get(t, str(x).strip().upper())


def normalize_material(x: Any) -> str:
    t = normalize_token(x).replace("_", "")
    for alias, canonical in MATERIAL_ALIASES.items():
        if normalize_token(alias).replace("_", "") == t:
            return canonical
    raw = str(x).strip()
    return raw if raw else "UNKNOWN"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(obj: Any) -> str:
    txt = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, sort_keys=True, default=str)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def atomic_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8", newline="") as tmp:
        df.to_csv(tmp.name, index=index)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def atomic_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text, matching the JSON/CSV checkpoint helpers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8", newline="") as tmp:
        tmp.write(str(text))
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def safe_read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_table_multi(df: pd.DataFrame, stem: Path, save_opts: Mapping[str, Any]) -> None:
    """Always save CSV + pickle; optionally Parquet/Excel."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(df, stem.with_suffix(".csv"), index=False)
    atomic_pickle(df, stem.with_suffix(".pkl"))
    if save_opts.get("parquet") and HAVE_PYARROW:
        try:
            df.to_parquet(stem.with_suffix(".parquet"), index=False)
        except Exception:
            logging.getLogger("p10").exception("Could not save parquet: %s", stem)
    if save_opts.get("excel") and HAVE_OPENPYXL:
        try:
            df.to_excel(stem.with_suffix(".xlsx"), index=False)
        except Exception:
            logging.getLogger("p10").exception("Could not save Excel: %s", stem)


def save_figure(fig: plt.Figure, stem: Path, save_opts: Mapping[str, Any]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=int(save_opts["dpi"]), bbox_inches="tight")
    if save_opts.get("pdf"):
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if save_opts.get("svg"):
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def ensure_output_tree(root: Path) -> Dict[str, Path]:
    keys = {
        "root": root,
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "audit": root / "audit",
        "data": root / "data",
        "models": root / "models",
        "uncertainty": root / "uncertainty",
        "engineering": root / "engineering",
        "validation": root / "validation",
        "tables_main": root / "tables" / "main",
        "tables_si": root / "tables" / "SI",
        "table_fig_main": root / "tables" / "main" / "visual_previews",
        "table_fig_si": root / "tables" / "SI" / "visual_previews",
        "fig_main": root / "figures" / "main",
        "fig_si": root / "figures" / "SI",
        "source_main": root / "source_data" / "main",
        "source_si": root / "source_data" / "SI",
        "final": root / "final",
    }
    for p in keys.values():
        p.mkdir(parents=True, exist_ok=True)
    return keys


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("p10")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logfile = log_dir / "P10_run.log"
    fh = RotatingFileHandler(logfile, maxBytes=8_000_000, backupCount=4, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def package_versions() -> Dict[str, str]:
    names = ["numpy", "pandas", "scipy", "matplotlib", "joblib", "openpyxl", "pyarrow"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = "not installed"
    return out


def resolve_n_jobs(requested: int, ram_mode: str) -> int:
    cpu = os.cpu_count() or 1
    n = cpu if requested == -1 else max(1, int(requested))
    n = min(n, cpu)
    cap = RAM_MODES[ram_mode]["parallel_cap"]
    if cap is not None:
        n = min(n, int(cap))
    return max(1, n)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, dirs: Mapping[str, Path], config_hash: str, input_hash: str, logger: logging.Logger,
                 legacy_science_config_hashes: Optional[Sequence[str]] = None):
        self.dirs = dirs
        self.config_hash = config_hash
        self.input_hash = input_hash
        self.logger = logger
        self.legacy_science_config_hashes = set(legacy_science_config_hashes or [])
        self.state_path = dirs["checkpoints"] / "pipeline_state.json"
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                self.state = {}
        else:
            self.state = {}

    def _compatible(self, rec: Mapping[str, Any], stage: str) -> bool:
        if rec.get("input_hash") != self.input_hash:
            return False
        rec_hash = rec.get("config_hash")
        if rec_hash == self.config_hash:
            return True
        # Publication-only hotfixes may explicitly adopt known-compatible
        # scientific checkpoints.  Never apply this to tables/figures/finalize.
        return bool(stage in SCIENCE_STAGES and rec_hash in self.legacy_science_config_hashes)

    def is_done(self, stage: str) -> bool:
        rec = self.state.get(stage, {})
        payload = self.dirs["checkpoints"] / f"{stage}.pkl"
        return bool(rec.get("status") == "done" and self._compatible(rec, stage) and payload.exists())

    def load(self, stage: str) -> Any:
        if not self.is_done(stage):
            raise RuntimeError(f"Stage {stage} is not a compatible completed checkpoint.")
        rec = self.state.get(stage, {})
        if rec.get("config_hash") != self.config_hash:
            self.logger.info("Resume: adopting a scientifically compatible checkpoint for stage %s", stage)
            rec["config_hash"] = self.config_hash
            rec["compatibility_migrated_utc"] = utc_now_iso()
            rec["compatibility_note"] = "publication-table compatibility update; scientific stage payload unchanged"
            self.state[stage] = rec
            atomic_json(self.state, self.state_path)
        self.logger.info("Resume: loading completed stage %s", stage)
        return safe_read_pickle(self.dirs["checkpoints"] / f"{stage}.pkl")

    def save(self, stage: str, payload: Any, summary: Optional[Mapping[str, Any]] = None) -> None:
        atomic_pickle(payload, self.dirs["checkpoints"] / f"{stage}.pkl")
        self.state[stage] = {
            "status": "done",
            "finished_utc": utc_now_iso(),
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "summary": dict(summary or {}),
        }
        atomic_json(self.state, self.state_path)
        self.logger.info("Checkpoint saved: %s", stage)

    def invalidate_from(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose one of {STAGES}")
        idx = STAGES.index(stage)
        for s in STAGES[idx:]:
            self.state.pop(s, None)
            p = self.dirs["checkpoints"] / f"{s}.pkl"
            if p.exists():
                p.unlink()
        atomic_json(self.state, self.state_path)
        self.logger.warning("Invalidated checkpoints from %s onward", stage)


# ---------------------------------------------------------------------------
# Input discovery and audit
# ---------------------------------------------------------------------------
def list_input_files(input_folder: Path) -> List[Path]:
    if not input_folder.exists():
        return []
    files = []
    for p in input_folder.rglob("*"):
        if (p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS and not p.name.startswith("~$")
                and not p.name.endswith("_SCHEMA_TEMPLATE.csv")):
            files.append(p)
    return sorted(files)


def build_input_fingerprint(files: Sequence[Path], input_folder: Path) -> Tuple[pd.DataFrame, str]:
    rows = []
    seen_hashes: Dict[str, str] = {}
    for p in files:
        h = sha256_file(p)
        rel = str(p.relative_to(input_folder))
        duplicate_of = seen_hashes.get(h, "")
        if not duplicate_of:
            seen_hashes[h] = rel
        rows.append({
            "relative_path": rel,
            "bytes": p.stat().st_size,
            "sha256": h,
            "duplicate_content": bool(duplicate_of),
            "duplicate_of": duplicate_of,
            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        })
    df = pd.DataFrame(rows)
    fingerprint = stable_json_hash(df[["relative_path", "bytes", "sha256"]].to_dict("records") if len(df) else [])
    return df, fingerprint


def read_tabular_file(path: Path) -> Dict[str, pd.DataFrame]:
    """Return one or more named tables from a supported input file."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return {"data": pd.read_csv(path, low_memory=False)}
    if ext in {".tsv", ".txt"}:
        # Automatic delimiter detection handles tabs, commas and semicolons.
        return {"data": pd.read_csv(path, sep=None, engine="python", low_memory=False)}
    if ext in {".xlsx", ".xls"}:
        if not HAVE_OPENPYXL and ext == ".xlsx":
            raise RuntimeError("openpyxl is required to read .xlsx files")
        sheets = pd.read_excel(path, sheet_name=None)
        return {str(k): v for k, v in sheets.items()}
    if ext == ".parquet":
        return {"data": pd.read_parquet(path)}
    if ext in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return {"data": obj}
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items() if isinstance(v, pd.DataFrame)}
        raise TypeError(f"Pickle {path} does not contain a DataFrame or dict of DataFrames")
    raise ValueError(f"Unsupported extension: {ext}")


def audit_dataframe(df: pd.DataFrame, rel_path: str, subtable: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    record = {
        "relative_path": rel_path,
        "subtable": subtable,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": " | ".join(map(str, numeric_cols)),
        "categorical_or_other_columns": " | ".join(map(str, categorical_cols)),
        "column_names": " | ".join(map(str, df.columns)),
    }
    missing_rows = []
    for c in df.columns:
        nmiss = int(df[c].isna().sum())
        missing_rows.append({
            "relative_path": rel_path,
            "subtable": subtable,
            "column": str(c),
            "dtype": str(df[c].dtype),
            "missing_n": nmiss,
            "missing_pct": (100.0 * nmiss / len(df)) if len(df) else np.nan,
            "unique_n": int(df[c].nunique(dropna=True)),
        })
    return record, pd.DataFrame(missing_rows)


def stage_00_audit(config: RunConfig, dirs: Mapping[str, Path], logger: logging.Logger) -> Dict[str, Any]:
    input_folder = Path(config.input_folder)
    files = list_input_files(input_folder)
    fingerprint_df, input_hash = build_input_fingerprint(files, input_folder)
    save_table_multi(fingerprint_df, dirs["audit"] / "input_file_fingerprints", SAVE_MODES[config.save_mode])

    audit_records: List[Dict[str, Any]] = []
    missingness_parts: List[pd.DataFrame] = []
    read_errors: List[Dict[str, Any]] = []
    duplicate_hashes = set(fingerprint_df.loc[fingerprint_df.get("duplicate_content", False) == True, "sha256"].tolist()) if len(fingerprint_df) else set()
    first_for_hash: set[str] = set()

    if not files:
        logger.warning("No input files found in %s", input_folder)

    for p in files:
        h = sha256_file(p)
        if h in first_for_hash:
            logger.info("Skipping byte-identical duplicate input: %s", p.name)
            continue
        first_for_hash.add(h)
        rel = str(p.relative_to(input_folder))
        try:
            tables = read_tabular_file(p)
            for subtable, df in tables.items():
                rec, miss = audit_dataframe(df, rel, subtable)
                rec["sha256"] = h
                audit_records.append(rec)
                missingness_parts.append(miss)
        except Exception as exc:
            logger.exception("Could not audit input file %s", p)
            read_errors.append({"relative_path": rel, "error": repr(exc)})

    audit_df = pd.DataFrame(audit_records)
    missing_df = pd.concat(missingness_parts, ignore_index=True) if missingness_parts else pd.DataFrame()
    errors_df = pd.DataFrame(read_errors)
    save_table_multi(audit_df, dirs["audit"] / "table_audit", SAVE_MODES[config.save_mode])
    save_table_multi(missing_df, dirs["audit"] / "column_missingness", SAVE_MODES[config.save_mode])
    if len(errors_df):
        save_table_multi(errors_df, dirs["audit"] / "read_errors", SAVE_MODES[config.save_mode])

    return {
        "fingerprints": fingerprint_df,
        "audit": audit_df,
        "missingness": missing_df,
        "errors": errors_df,
        "input_hash": input_hash,
    }

# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------
def _find_alias_column(columns: Sequence[Any], canonical: str) -> Optional[Any]:
    norm_map = {normalize_token(c): c for c in columns}
    for alias in COLUMN_ALIASES.get(canonical, []):
        na = normalize_token(alias)
        if na in norm_map:
            return norm_map[na]
    # Fuzzy fallback: useful for headers such as "pressure (bar)".
    for nc, original in norm_map.items():
        for alias in COLUMN_ALIASES.get(canonical, []):
            na = normalize_token(alias)
            if nc.startswith(na + "_") or nc.endswith("_" + na):
                return original
    return None


def infer_metadata_from_text(text: str) -> Dict[str, Any]:
    """Infer material/gas/temperature only when strongly encoded in a name."""
    low = normalize_token(text)
    compact = low.replace("_", "")
    out: Dict[str, Any] = {}
    for alias, canonical in MATERIAL_ALIASES.items():
        if normalize_token(alias).replace("_", "") in compact:
            out["material"] = canonical
            break
    for alias, canonical in GAS_ALIASES.items():
        token = normalize_token(alias)
        # Require token boundaries for short gases to avoid accidental matches.
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", low):
            out["adsorbate"] = canonical
            break
    # Accept patterns such as 298K, 298_K, T298.
    m = re.search(r"(?:^|_)(?:t_?)?(2[5-9][0-9]|3[0-9][0-9]|4[0-5][0-9])_?k?(?:_|$)", low)
    if m:
        out["temperature_K"] = float(m.group(1))
    return out


def detect_pressure_unit(header: str) -> str:
    h = normalize_token(header)
    if "mpa" in h:
        return "MPa"
    if "kpa" in h:
        return "kPa"
    if "pa" in h and "kpa" not in h and "mpa" not in h:
        return "Pa"
    if "mbar" in h:
        return "mbar"
    if "atm" in h:
        return "atm"
    if "torr" in h or "mmhg" in h:
        return "torr"
    return "bar"


def pressure_to_bar(values: pd.Series, unit: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    factors = {"bar": 1.0, "kPa": 0.01, "MPa": 10.0, "Pa": 1.0e-5, "mbar": 1.0e-3, "atm": 1.01325, "torr": 1.0 / 750.061683}
    return x * factors.get(unit, 1.0)


def detect_temperature_unit(header: str) -> str:
    h = normalize_token(header)
    if "celsius" in h or h.endswith("_c") or "degc" in h:
        return "C"
    return "K"


def temperature_to_K(values: pd.Series, unit: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    return x + 273.15 if unit == "C" else x


def detect_loading_unit(header: str) -> str:
    h = normalize_token(header)
    if "mmol_g" in h or "mmol_per_g" in h:
        return "mmol/g"
    if "mmol_kg" in h or "mmol_per_kg" in h:
        return "mmol/kg"
    if "mol_g" in h or "mol_per_g" in h:
        return "mol/g"
    if "mol_kg" in h or "mol_per_kg" in h:
        return "mol/kg"
    if "mg_g" in h or "mg_per_g" in h:
        return "mg/g"
    if "g_kg" in h or "g_per_kg" in h:
        return "g/kg"
    if "wt" in h and ("pct" in h or "percent" in h):
        return "wt%"
    # Most adsorption CSVs use mmol/g or mol/kg, which are numerically equal.
    return "mol/kg"


def loading_to_mol_kg(values: pd.Series, unit: str, gases: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if unit in {"mol/kg", "mmol/g"}:
        return x
    if unit == "mmol/kg":
        return x / 1000.0
    if unit == "mol/g":
        return x * 1000.0
    if unit in {"mg/g", "g/kg"}:
        mw = gases.map(lambda g: MOLECULAR_WEIGHT_G_MOL.get(normalize_gas(g), np.nan)).astype(float)
        return x / mw
    if unit == "wt%":
        # wt% = g adsorbate per 100 g adsorbent = 10 g/kg adsorbent per wt%.
        mw = gases.map(lambda g: MOLECULAR_WEIGHT_G_MOL.get(normalize_gas(g), np.nan)).astype(float)
        return (10.0 * x) / mw
    return x


def canonicalize_long_table(df: pd.DataFrame, source_path: str, subtable: str) -> Optional[pd.DataFrame]:
    """Convert a recognizable long-format isotherm/provenance table."""
    cols = list(df.columns)
    p_col = _find_alias_column(cols, "pressure_bar")
    q_col = _find_alias_column(cols, "loading_mol_kg")
    if p_col is None or q_col is None:
        return None

    meta = infer_metadata_from_text(source_path + " " + subtable)
    out = pd.DataFrame(index=df.index)

    m_col = _find_alias_column(cols, "material")
    g_col = _find_alias_column(cols, "adsorbate")
    t_col = _find_alias_column(cols, "temperature_K")
    form_col = _find_alias_column(cols, "material_form")

    if m_col is not None:
        out["material"] = df[m_col].map(normalize_material)
    else:
        out["material"] = normalize_material(meta.get("material", "UNKNOWN"))

    if g_col is not None:
        out["adsorbate"] = df[g_col].map(normalize_gas)
    else:
        out["adsorbate"] = normalize_gas(meta.get("adsorbate", "UNKNOWN"))

    if t_col is not None:
        out["temperature_K"] = temperature_to_K(df[t_col], detect_temperature_unit(str(t_col)))
    else:
        out["temperature_K"] = float(meta.get("temperature_K", np.nan))

    out["pressure_bar"] = pressure_to_bar(df[p_col], detect_pressure_unit(str(p_col)))
    out["loading_mol_kg"] = loading_to_mol_kg(df[q_col], detect_loading_unit(str(q_col)), out["adsorbate"])
    out["material_form"] = df[form_col].astype(str) if form_col is not None else "unspecified"

    for canonical in ["uncertainty", "source_id", "doi", "table_figure_si_location", "extraction_method", "allowed_use", "notes", "kinetic_quantity", "kinetic_value", "kinetic_units"]:
        c = _find_alias_column(cols, canonical)
        if c is not None:
            out[canonical] = df[c]

    out["source_file"] = source_path
    out["source_subtable"] = subtable
    out["source_row"] = np.arange(len(out), dtype=int) + 2
    if "source_id" not in out:
        out["source_id"] = Path(source_path).stem
    return out


def _parse_measurement_column(name: str, common_meta: Mapping[str, Any]) -> Dict[str, Any]:
    meta = dict(common_meta)
    meta.update(infer_metadata_from_text(name))
    return meta


def canonicalize_wide_table(df: pd.DataFrame, source_path: str, subtable: str) -> Optional[pd.DataFrame]:
    """Handle simple wide isotherm layouts: one pressure column + q columns.

    Examples handled:
        pressure_bar, CO2_273K, CO2_298K, CO2_323K
        P_kPa, CALF20_CO2_298K, CALF20_N2_298K

    This intentionally avoids guessing when a measurement column has no gas or
    temperature information in either its name, sheet name or filename.
    """
    p_col = _find_alias_column(df.columns, "pressure_bar")
    if p_col is None:
        return None
    common_meta = infer_metadata_from_text(source_path + " " + subtable)
    parts = []
    for c in df.columns:
        if c == p_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        meta = _parse_measurement_column(str(c), common_meta)
        if "adsorbate" not in meta or "temperature_K" not in meta:
            continue
        tmp = pd.DataFrame({
            "material": normalize_material(meta.get("material", "UNKNOWN")),
            "material_form": "unspecified",
            "adsorbate": normalize_gas(meta["adsorbate"]),
            "temperature_K": float(meta["temperature_K"]),
            "pressure_bar": pressure_to_bar(df[p_col], detect_pressure_unit(str(p_col))),
            "loading_mol_kg": loading_to_mol_kg(df[c], detect_loading_unit(str(c)), pd.Series([meta["adsorbate"]] * len(df), index=df.index)),
            "source_id": Path(source_path).stem,
            "source_file": source_path,
            "source_subtable": subtable,
            "source_row": np.arange(len(df), dtype=int) + 2,
        })
        parts.append(tmp)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def harmonization_qc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reasons = []
    valid = pd.Series(True, index=df.index)
    checks = {
        "missing_material": df["material"].isna() | (df["material"].astype(str).str.len() == 0) | (df["material"] == "UNKNOWN"),
        "missing_gas": df["adsorbate"].isna() | (df["adsorbate"] == "UNKNOWN"),
        "invalid_temperature": ~np.isfinite(df["temperature_K"]) | (df["temperature_K"] < 150) | (df["temperature_K"] > 700),
        "invalid_pressure": ~np.isfinite(df["pressure_bar"]) | (df["pressure_bar"] < 0) | (df["pressure_bar"] > 5000),
        "invalid_loading": ~np.isfinite(df["loading_mol_kg"]) | (df["loading_mol_kg"] < -1e-10) | (df["loading_mol_kg"] > 200),
    }
    for name, mask in checks.items():
        valid &= ~mask
        for idx in df.index[mask]:
            reasons.append({"row_index": int(idx), "reason": name})
    rejected = df.loc[~valid].copy()
    reason_df = pd.DataFrame(reasons)
    if len(rejected) and len(reason_df):
        rejected = rejected.merge(reason_df.groupby("row_index")["reason"].apply(" | ".join), left_index=True, right_index=True, how="left")
    return df.loc[valid].copy(), rejected


def collapse_exact_observation_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove true repeated observations without merging independent literature.

    When DOI exists it is the preferred provenance key, so the same digitized
    observation copied into two files from the same paper is not double-weighted.
    If DOI is absent, source_id is used and distinct sources are retained.
    """
    work = df.copy()
    if "doi" in work.columns:
        doi = work["doi"].fillna("").astype(str).str.strip().str.lower()
    else:
        doi = pd.Series("", index=work.index)
    source = work["source_id"].fillna("").astype(str) if "source_id" in work.columns else pd.Series("", index=work.index)
    work["_provenance_key"] = np.where(doi.str.len() > 0, "doi:" + doi, "source:" + source)
    keys = ["material", "material_form", "adsorbate", "temperature_K", "pressure_bar", "loading_mol_kg", "_provenance_key"]
    keys = [k for k in keys if k in work.columns]
    dup = work.duplicated(keys, keep="first")
    duplicate_rows = work.loc[dup].drop(columns=["_provenance_key"], errors="ignore").copy()
    clean = work.loc[~dup].drop(columns=["_provenance_key"], errors="ignore").copy()
    return clean, duplicate_rows


def stage_01_harmonize(config: RunConfig, dirs: Mapping[str, Path], audit_payload: Mapping[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    input_folder = Path(config.input_folder)
    fingerprints: pd.DataFrame = audit_payload["fingerprints"]
    seen_hashes: set[str] = set()
    parts: List[pd.DataFrame] = []
    unrecognized: List[Dict[str, Any]] = []

    for rec in fingerprints.to_dict("records") if len(fingerprints) else []:
        if rec["sha256"] in seen_hashes:
            continue
        seen_hashes.add(rec["sha256"])
        path = input_folder / rec["relative_path"]
        try:
            tables = read_tabular_file(path)
        except Exception as exc:
            unrecognized.append({"relative_path": rec["relative_path"], "subtable": "", "reason": f"read_error: {exc!r}"})
            continue
        for subtable, raw in tables.items():
            long = canonicalize_long_table(raw, rec["relative_path"], subtable)
            if long is None:
                long = canonicalize_wide_table(raw, rec["relative_path"], subtable)
            if long is None:
                unrecognized.append({"relative_path": rec["relative_path"], "subtable": subtable, "reason": "no recognizable pressure/loading layout"})
            else:
                parts.append(long)

    if not parts:
        empty_cols = ["material", "material_form", "adsorbate", "temperature_K", "pressure_bar", "loading_mol_kg", "source_id", "source_file", "source_subtable", "source_row"]
        harmonized = pd.DataFrame(columns=empty_cols)
        rejected = pd.DataFrame()
        duplicates = pd.DataFrame()
    else:
        harmonized = pd.concat(parts, ignore_index=True, sort=False)
        # Clean types before QC.
        harmonized["material"] = harmonized["material"].map(normalize_material)
        harmonized["adsorbate"] = harmonized["adsorbate"].map(normalize_gas)
        harmonized["temperature_K"] = pd.to_numeric(harmonized["temperature_K"], errors="coerce")
        harmonized["pressure_bar"] = pd.to_numeric(harmonized["pressure_bar"], errors="coerce")
        harmonized["loading_mol_kg"] = pd.to_numeric(harmonized["loading_mol_kg"], errors="coerce")
        harmonized, rejected = harmonization_qc(harmonized)
        harmonized, duplicates = collapse_exact_observation_duplicates(harmonized)
        harmonized = harmonized.sort_values(["material", "adsorbate", "temperature_K", "pressure_bar", "source_id"], kind="stable").reset_index(drop=True)

    save_table_multi(harmonized, dirs["data"] / "harmonized_adsorption_data", SAVE_MODES[config.save_mode])
    if len(rejected):
        save_table_multi(rejected, dirs["data"] / "rejected_rows", SAVE_MODES[config.save_mode])
    if len(duplicates):
        save_table_multi(duplicates, dirs["data"] / "duplicate_observations_removed", SAVE_MODES[config.save_mode])
    unrec_df = pd.DataFrame(unrecognized)
    if len(unrec_df):
        save_table_multi(unrec_df, dirs["audit"] / "unrecognized_tables", SAVE_MODES[config.save_mode])

    # Compact data-domain summary becomes a useful SI source registry.
    if len(harmonized):
        domain = harmonized.groupby(["material", "material_form", "adsorbate"], dropna=False).agg(
            n_points=("loading_mol_kg", "size"),
            n_temperatures=("temperature_K", "nunique"),
            temperature_min_K=("temperature_K", "min"),
            temperature_max_K=("temperature_K", "max"),
            pressure_min_bar=("pressure_bar", "min"),
            pressure_max_bar=("pressure_bar", "max"),
            loading_max_mol_kg=("loading_mol_kg", "max"),
            n_sources=("source_id", "nunique"),
        ).reset_index()
    else:
        domain = pd.DataFrame()
    save_table_multi(domain, dirs["data"] / "data_domain_summary", SAVE_MODES[config.save_mode])

    logger.info("Harmonized %d valid adsorption rows from %d recognized table(s)", len(harmonized), len(parts))
    return {
        "data": harmonized,
        "rejected": rejected,
        "duplicates": duplicates,
        "unrecognized": unrec_df,
        "domain": domain,
    }


# ---------------------------------------------------------------------------
# Multi-temperature isotherm models
# ---------------------------------------------------------------------------
def _b_temperature(log_b_ref: float, qst_kj_mol: float, T: np.ndarray) -> np.ndarray:
    # qst is stored as a positive adsorption enthalpy magnitude.  For exothermic
    # adsorption, affinity increases as temperature falls.
    return np.exp(log_b_ref + (qst_kj_mol * 1000.0 / R_J_MOL_K) * (1.0 / T - 1.0 / T_REF_K))


def model_loading(model: str, p_bar: np.ndarray, T_K: np.ndarray, theta: Sequence[float]) -> np.ndarray:
    p = np.maximum(np.asarray(p_bar, dtype=float), 0.0)
    T = np.asarray(T_K, dtype=float)
    th = np.asarray(theta, dtype=float)
    if model == "langmuir":
        qsat = np.exp(th[0]); b = _b_temperature(th[1], th[2], T)
        return qsat * b * p / (1.0 + b * p)
    if model == "dual_site_langmuir":
        q1 = np.exp(th[0]); b1 = _b_temperature(th[1], th[2], T)
        q2 = np.exp(th[3]); b2 = _b_temperature(th[4], th[5], T)
        return q1 * b1 * p / (1.0 + b1 * p) + q2 * b2 * p / (1.0 + b2 * p)
    if model == "sips":
        qsat = np.exp(th[0]); b = _b_temperature(th[1], th[2], T); n = np.exp(th[3])
        z = np.maximum(b * p, 0.0) ** n
        return qsat * z / (1.0 + z)
    if model == "toth":
        qsat = np.exp(th[0]); b = _b_temperature(th[1], th[2], T); t = np.exp(th[3])
        z = np.maximum(b * p, 0.0)
        return qsat * z / np.power(1.0 + np.power(z, t), 1.0 / t)
    raise ValueError(model)


def model_param_spec(model: str, qmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    q0 = max(float(qmax), 0.1)
    if model == "langmuir":
        x0 = np.array([math.log(q0 * 1.15), math.log(1.0), 30.0])
        lo = np.array([math.log(max(q0 * 0.2, 1e-4)), -16.0, 0.1])
        hi = np.array([math.log(max(q0 * 20.0, 2.0)), 16.0, 120.0])
        names = ["log_qsat", "log_b_ref_bar_inv", "qst_kJ_mol"]
    elif model == "dual_site_langmuir":
        x0 = np.array([math.log(q0 * 0.55), math.log(5.0), 35.0, math.log(q0 * 0.65), math.log(0.3), 25.0])
        lo = np.array([math.log(max(q0 * 0.02, 1e-5)), -16.0, 0.1, math.log(max(q0 * 0.02, 1e-5)), -16.0, 0.1])
        hi = np.array([math.log(max(q0 * 20.0, 2.0)), 16.0, 120.0, math.log(max(q0 * 20.0, 2.0)), 16.0, 120.0])
        names = ["log_qsat_1", "log_b1_ref_bar_inv", "qst1_kJ_mol", "log_qsat_2", "log_b2_ref_bar_inv", "qst2_kJ_mol"]
    elif model in {"sips", "toth"}:
        x0 = np.array([math.log(q0 * 1.15), math.log(1.0), 30.0, math.log(1.0)])
        lo = np.array([math.log(max(q0 * 0.2, 1e-4)), -16.0, 0.1, math.log(0.25)])
        hi = np.array([math.log(max(q0 * 20.0, 2.0)), 16.0, 120.0, math.log(3.0)])
        names = ["log_qsat", "log_b_ref_bar_inv", "qst_kJ_mol", "shape_parameter_log"]
    else:
        raise ValueError(model)
    return x0, lo, hi, names


def canonicalize_theta(model: str, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).copy()
    if model == "dual_site_langmuir":
        # Label site 1 as the higher-affinity site at Tref to reduce arbitrary
        # site-swapping in bootstrap summaries.
        if theta[1] < theta[4]:
            theta = theta[[3, 4, 5, 0, 1, 2]]
    return theta


def observation_sigma(df: pd.DataFrame) -> np.ndarray:
    q = np.asarray(df["loading_mol_kg"], dtype=float)
    robust_scale = float(median_abs_deviation(q, nan_policy="omit", scale="normal")) if len(q) > 2 else 0.0
    abs_floor = max(0.005, 0.02 * robust_scale, 0.005 * max(np.nanmax(q), 1.0))
    sigma = np.maximum(abs_floor, 0.04 * np.maximum(q, abs_floor))
    if "uncertainty" in df.columns:
        u = pd.to_numeric(df["uncertainty"], errors="coerce").to_numpy(float)
        good = np.isfinite(u) & (u > 0)
        sigma[good] = np.maximum(u[good], abs_floor * 0.25)
    return sigma


def fit_one_model(df: pd.DataFrame, model: str, n_starts: int, seed: int) -> Dict[str, Any]:
    p = df["pressure_bar"].to_numpy(float)
    T = df["temperature_K"].to_numpy(float)
    q = df["loading_mol_kg"].to_numpy(float)
    sigma = observation_sigma(df)
    xbase, lo, hi, names = model_param_spec(model, np.nanmax(q))
    rng = np.random.default_rng(seed)
    attempts = []
    best = None

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = model_loading(model, p, T, theta)
        return (pred - q) / sigma

    starts = [xbase]
    for _ in range(max(0, n_starts - 1)):
        scale = 0.35 * (hi - lo)
        candidate = np.clip(xbase + rng.normal(0.0, 0.22, size=len(xbase)) * scale, lo + 1e-8, hi - 1e-8)
        starts.append(candidate)

    for j, start in enumerate(starts):
        try:
            res = least_squares(residual, start, bounds=(lo, hi), method="trf", max_nfev=8000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            theta = canonicalize_theta(model, res.x)
            r = residual(theta)
            sse_w = float(np.sum(r ** 2))
            attempts.append({"start": j, "success": bool(res.success), "cost": float(res.cost), "sse_weighted": sse_w, "nfev": int(res.nfev), "message": str(res.message)})
            if best is None or sse_w < best[0]:
                best = (sse_w, theta, res)
        except Exception as exc:
            attempts.append({"start": j, "success": False, "cost": np.nan, "sse_weighted": np.nan, "nfev": 0, "message": repr(exc)})

    if best is None:
        return {"success": False, "model": model, "attempts": attempts, "error": "all starts failed"}

    sse_w, theta, res = best
    pred = model_loading(model, p, T, theta)
    resid_raw = pred - q
    n = len(q); k = len(theta)
    # AICc is calculated from unweighted residual variance for within-dataset
    # model comparison.  Weighted SSE remains available for fit diagnostics.
    sse = max(float(np.sum(resid_raw ** 2)), 1e-30)
    aic = n * math.log(sse / max(n, 1)) + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1)) if n > k + 1 else np.inf
    rmse = math.sqrt(sse / max(n, 1))
    ss_tot = float(np.sum((q - np.mean(q)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.nan
    bound_hit = bool(np.any(np.isclose(theta, lo, rtol=0, atol=1e-5)) or np.any(np.isclose(theta, hi, rtol=0, atol=1e-5)))

    return {
        "success": True,
        "model": model,
        "theta": theta,
        "param_names": names,
        "lower_bounds": lo,
        "upper_bounds": hi,
        "n": n,
        "k": k,
        "sse": sse,
        "sse_weighted": sse_w,
        "rmse": rmse,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bound_hit": bound_hit,
        "attempts": attempts,
        "predicted": pred,
        "residuals": resid_raw,
    }


def fit_group_models(group: pd.DataFrame, models: Sequence[str], n_starts: int, seed: int) -> Dict[str, Any]:
    results = {}
    for i, model in enumerate(models):
        # More-complex models need a minimum amount of information.
        min_n = {"langmuir": 6, "dual_site_langmuir": 12, "sips": 8, "toth": 8}[model]
        if len(group) < min_n:
            results[model] = {"success": False, "model": model, "error": f"requires at least {min_n} rows"}
            continue
        results[model] = fit_one_model(group, model, n_starts=n_starts, seed=seed + 1009 * (i + 1))
    successful = {m: r for m, r in results.items() if r.get("success") and np.isfinite(r.get("aicc", np.inf))}
    if not successful:
        # If AICc is undefined because n is close to k, allow the successful
        # model with the smallest weighted SSE.
        successful = {m: r for m, r in results.items() if r.get("success")}
    if not successful:
        return {"models": results, "best_model": None, "retained_models": []}
    best_model = min(successful, key=lambda m: successful[m].get("aicc", np.inf) if np.isfinite(successful[m].get("aicc", np.inf)) else successful[m]["sse_weighted"])
    best_aicc = successful[best_model].get("aicc", np.inf)
    retained = []
    for m, r in successful.items():
        if np.isfinite(best_aicc) and np.isfinite(r.get("aicc", np.inf)):
            if r["aicc"] - best_aicc <= 4.0:
                retained.append(m)
        elif m == best_model:
            retained.append(m)
    if best_model not in retained:
        retained.insert(0, best_model)
    return {"models": results, "best_model": best_model, "retained_models": retained}


def group_key_to_id(material: str, adsorbate: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{material}__{adsorbate}").strip("_")


def stage_02_fit_models(config: RunConfig, dirs: Mapping[str, Path], harmonized_payload: Mapping[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    data: pd.DataFrame = harmonized_payload["data"]
    level = LEVELS[config.level]
    all_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    summary_rows = []
    attempts_rows = []
    residual_rows = []

    if data.empty:
        logger.warning("No harmonized adsorption data; model-fitting stage will be empty.")
        return {"fits": {}, "summary": pd.DataFrame(), "attempts": pd.DataFrame(), "residuals": pd.DataFrame()}

    grouped = list(data.groupby(["material", "adsorbate"], sort=True))
    logger.info("Fitting %d material/gas groups with models: %s", len(grouped), ", ".join(level["models"]))
    for gi, ((material, gas), g) in enumerate(grouped):
        # Temperature robustness needs at least two temperatures.  A one-T fit
        # is still saved for evidence plots, but its temperature enthalpy is not
        # treated as identified and the group will be flagged later.
        result = fit_group_models(g, level["models"], int(level["n_starts"]), config.seed + gi * 7919)
        result["material"] = material; result["adsorbate"] = gas
        result["n_temperatures"] = int(g["temperature_K"].nunique())
        result["domain"] = {
            "temperature_min_K": float(g["temperature_K"].min()),
            "temperature_max_K": float(g["temperature_K"].max()),
            "pressure_min_bar": float(max(g.loc[g["pressure_bar"] > 0, "pressure_bar"].min() if (g["pressure_bar"] > 0).any() else PRESSURE_FLOOR_BAR, PRESSURE_FLOOR_BAR)),
            "pressure_max_bar": float(g["pressure_bar"].max()),
        }
        all_results[(material, gas)] = result

        for model, r in result["models"].items():
            row = {
                "material": material, "adsorbate": gas, "model": model,
                "success": bool(r.get("success", False)), "best_model": model == result["best_model"],
                "retained_model": model in result["retained_models"], "n_temperatures": result["n_temperatures"],
            }
            for k in ["n", "k", "sse", "sse_weighted", "rmse", "r2", "aic", "aicc", "bound_hit", "error"]:
                row[k] = r.get(k, np.nan)
            if r.get("success"):
                for name, val in zip(r["param_names"], r["theta"]):
                    row[name] = float(val)
            summary_rows.append(row)
            for a in r.get("attempts", []):
                attempts_rows.append({"material": material, "adsorbate": gas, "model": model, **a})
            if r.get("success"):
                gg = g.reset_index(drop=True)
                for idx, (obs, pred, resid) in enumerate(zip(gg["loading_mol_kg"], r["predicted"], r["residuals"])):
                    residual_rows.append({
                        "material": material, "adsorbate": gas, "model": model,
                        "temperature_K": float(gg.loc[idx, "temperature_K"]),
                        "pressure_bar": float(gg.loc[idx, "pressure_bar"]),
                        "observed_mol_kg": float(obs), "predicted_mol_kg": float(pred), "residual_mol_kg": float(resid),
                        "source_id": gg.loc[idx, "source_id"] if "source_id" in gg else "",
                    })

    summary_df = pd.DataFrame(summary_rows)
    attempts_df = pd.DataFrame(attempts_rows)
    residual_df = pd.DataFrame(residual_rows)
    save_table_multi(summary_df, dirs["models"] / "isotherm_model_summary", SAVE_MODES[config.save_mode])
    save_table_multi(attempts_df, dirs["models"] / "fit_multistart_attempts", SAVE_MODES[config.save_mode])
    save_table_multi(residual_df, dirs["models"] / "fit_residuals", SAVE_MODES[config.save_mode])
    atomic_pickle(all_results, dirs["models"] / "fitted_models_complete.pkl")
    return {"fits": all_results, "summary": summary_df, "attempts": attempts_df, "residuals": residual_df}

# ---------------------------------------------------------------------------
# Bootstrap uncertainty and model-form ensemble
# ---------------------------------------------------------------------------
def stratified_resample(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample within temperature blocks so each bootstrap retains T coverage."""
    parts = []
    for _, block in group.groupby("temperature_K", sort=False):
        idx = rng.integers(0, len(block), size=len(block))
        parts.append(block.iloc[idx].copy())
    return pd.concat(parts, ignore_index=True)


def bootstrap_fit_worker(group: pd.DataFrame, model: str, n_starts: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    boot = stratified_resample(group, rng)
    r = fit_one_model(boot, model, n_starts=max(2, min(n_starts, 8)), seed=seed + 17)
    if not r.get("success"):
        return {"success": False, "seed": seed, "error": r.get("error", "fit failure")}
    return {
        "success": True,
        "seed": seed,
        "theta": np.asarray(r["theta"], dtype=float),
        "rmse": float(r["rmse"]),
        "bound_hit": bool(r.get("bound_hit", False)),
    }


def _bootstrap_checkpoint_path(dirs: Mapping[str, Path], material: str, gas: str, model: str) -> Path:
    return dirs["uncertainty"] / "bootstrap_chunks" / f"{group_key_to_id(material, gas)}__{model}.pkl"


def run_bootstrap_group(
    group: pd.DataFrame,
    material: str,
    gas: str,
    model: str,
    n_boot: int,
    n_starts: int,
    n_jobs: int,
    batch_size: int,
    base_seed: int,
    dirs: Mapping[str, Path],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Chunked bootstrap with an inner restart point after every batch."""
    path = _bootstrap_checkpoint_path(dirs, material, gas, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    if path.exists():
        try:
            old = safe_read_pickle(path)
            if isinstance(old, list):
                samples = old[:n_boot]
                logger.info("Bootstrap resume %s/%s/%s: %d/%d samples already present", material, gas, model, len(samples), n_boot)
        except Exception:
            logger.exception("Could not read partial bootstrap checkpoint %s; rebuilding it", path)
            samples = []

    while len(samples) < n_boot:
        start = len(samples)
        stop = min(n_boot, start + batch_size)
        seeds = [base_seed + i * 104729 for i in range(start, stop)]
        if n_jobs == 1:
            batch = [bootstrap_fit_worker(group, model, n_starts, s) for s in seeds]
        else:
            batch = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(bootstrap_fit_worker)(group, model, n_starts, s) for s in seeds
            )
        samples.extend(batch)
        atomic_pickle(samples, path)
        success_n = sum(bool(x.get("success")) for x in samples)
        logger.info("Bootstrap %s/%s/%s: %d/%d completed (%d successful)", material, gas, model, len(samples), n_boot, success_n)
        gc.collect()
    return samples


def bootstrap_summary_rows(material: str, gas: str, model: str, samples: Sequence[Mapping[str, Any]], param_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    ok = [s for s in samples if s.get("success")]
    rows: List[Dict[str, Any]] = []
    if not ok:
        return rows, pd.DataFrame()
    mat = np.vstack([s["theta"] for s in ok])
    for j, name in enumerate(param_names):
        vals = mat[:, j]
        rows.append({
            "material": material, "adsorbate": gas, "model": model, "parameter": name,
            "n_success": len(ok), "median": float(np.nanmedian(vals)),
            "q025": float(np.nanquantile(vals, 0.025)), "q975": float(np.nanquantile(vals, 0.975)),
            "mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
        })
    corr = pd.DataFrame(np.corrcoef(mat, rowvar=False), index=param_names, columns=param_names)
    corr.insert(0, "parameter", corr.index)
    corr.insert(0, "model", model)
    corr.insert(0, "adsorbate", gas)
    corr.insert(0, "material", material)
    return rows, corr.reset_index(drop=True)


def stage_03_uncertainty(config: RunConfig, dirs: Mapping[str, Path], harmonized_payload: Mapping[str, Any], fit_payload: Mapping[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    data: pd.DataFrame = harmonized_payload["data"]
    fits: Dict[Tuple[str, str], Dict[str, Any]] = fit_payload["fits"]
    level = LEVELS[config.level]
    n_jobs = resolve_n_jobs(config.n_jobs, config.ram_mode)
    batch_size = int(RAM_MODES[config.ram_mode]["batch"])
    n_boot = int(level["n_boot"])
    all_samples: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    corr_parts: List[pd.DataFrame] = []

    for gi, ((material, gas), fitset) in enumerate(sorted(fits.items())):
        best = fitset.get("best_model")
        if best is None:
            continue
        models = [best] if level["bootstrap_models"] == "best" else list(fitset.get("retained_models", [best]))
        group = data[(data["material"] == material) & (data["adsorbate"] == gas)].reset_index(drop=True)
        for mi, model in enumerate(models):
            central = fitset["models"].get(model, {})
            if not central.get("success"):
                continue
            samples = run_bootstrap_group(
                group, material, gas, model, n_boot=n_boot,
                n_starts=int(level["n_starts"]), n_jobs=n_jobs,
                batch_size=batch_size, base_seed=config.seed + 1_000_003 * (gi + 1) + 10_007 * (mi + 1),
                dirs=dirs, logger=logger,
            )
            all_samples[(material, gas, model)] = samples
            rows, corr = bootstrap_summary_rows(material, gas, model, samples, central["param_names"])
            summary_rows.extend(rows)
            if len(corr):
                corr_parts.append(corr)

    summary_df = pd.DataFrame(summary_rows)
    corr_df = pd.concat(corr_parts, ignore_index=True) if corr_parts else pd.DataFrame()
    save_table_multi(summary_df, dirs["uncertainty"] / "bootstrap_parameter_summary", SAVE_MODES[config.save_mode])
    if len(corr_df):
        save_table_multi(corr_df, dirs["uncertainty"] / "bootstrap_parameter_correlations", SAVE_MODES[config.save_mode])
    # Rich/archival modes also keep a convenient single-file bootstrap object.
    if SAVE_MODES[config.save_mode]["bootstrap_samples"]:
        atomic_pickle(all_samples, dirs["uncertainty"] / "bootstrap_samples_complete.pkl")
    success_rows = []
    for (m, g, model), samples in all_samples.items():
        ok = [s for s in samples if s.get("success")]
        success_rows.append({"material": m, "adsorbate": g, "model": model, "requested": len(samples), "successful": len(ok), "success_fraction": len(ok) / max(len(samples), 1)})
    success_df = pd.DataFrame(success_rows)
    save_table_multi(success_df, dirs["uncertainty"] / "bootstrap_success_rates", SAVE_MODES[config.save_mode])
    return {"samples": all_samples, "summary": summary_df, "correlations": corr_df, "success": success_df}


# ---------------------------------------------------------------------------
# Thermodynamic helpers: spreading pressure, IAST, inverse regeneration
# ---------------------------------------------------------------------------
def spreading_pressure(model: str, p_bar: float, T_K: float, theta: Sequence[float]) -> float:
    p = max(float(p_bar), 0.0)
    if p <= 0:
        return 0.0
    th = np.asarray(theta, dtype=float)
    if model == "langmuir":
        qsat = math.exp(th[0]); b = float(_b_temperature(th[1], th[2], np.array([T_K]))[0])
        return qsat * math.log1p(b * p)
    if model == "dual_site_langmuir":
        q1 = math.exp(th[0]); b1 = float(_b_temperature(th[1], th[2], np.array([T_K]))[0])
        q2 = math.exp(th[3]); b2 = float(_b_temperature(th[4], th[5], np.array([T_K]))[0])
        return q1 * math.log1p(b1 * p) + q2 * math.log1p(b2 * p)
    if model == "sips":
        qsat = math.exp(th[0]); b = float(_b_temperature(th[1], th[2], np.array([T_K]))[0]); n = math.exp(th[3])
        return (qsat / n) * math.log1p((b * p) ** n)
    # Toth: reliable numerical integral.  q/P is finite at P->0; use log-space
    # integration to avoid evaluating exactly at zero.
    p0 = max(p * 1e-12, 1e-14)
    def integrand_log(logp: float) -> float:
        pp = math.exp(logp)
        return float(model_loading(model, np.array([pp]), np.array([T_K]), th)[0])
    return float(quad(integrand_log, math.log(p0), math.log(p), epsabs=1e-9, epsrel=1e-7, limit=120)[0])


def pressure_at_spreading_pressure(model: str, target_pi: float, T_K: float, theta: Sequence[float]) -> float:
    if target_pi <= 0:
        return 1.0e-30
    # IAST can require fictitious pure-component pressures far below the
    # engineering pressure floor.  Use a numerical floor close to zero here;
    # PRESSURE_FLOOR_BAR is reserved for process/extrapolation decisions.
    lo = 1.0e-30
    hi = 1.0
    f = lambda p: spreading_pressure(model, p, T_K, theta) - target_pi
    flo = f(lo)
    if flo >= 0:
        return lo
    while f(hi) < 0 and hi < 1.0e6:
        hi *= 10.0
    if f(hi) < 0:
        raise ValueError("Could not bracket spreading-pressure inversion")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-9, maxiter=200))


def iast_binary(
    total_pressure_bar: float,
    y1: float,
    T_K: float,
    model1: str,
    theta1: Sequence[float],
    model2: str,
    theta2: Sequence[float],
) -> Dict[str, float]:
    """Binary IAST for two fitted pure-component isotherms.

    Returns component loadings, adsorbed mole fractions and separation factor.
    Failures are explicit; callers should not silently replace IAST with an
    unrelated selectivity metric.
    """
    P = float(total_pressure_bar)
    y1 = float(np.clip(y1, 1e-8, 1 - 1e-8)); y2 = 1.0 - y1

    def closure(pi: float) -> float:
        p01 = pressure_at_spreading_pressure(model1, pi, T_K, theta1)
        p02 = pressure_at_spreading_pressure(model2, pi, T_K, theta2)
        return y1 * P / p01 + y2 * P / p02 - 1.0

    lo = 1e-12
    hi = 1.0
    flo = closure(lo)
    fhi = closure(hi)
    while fhi > 0 and hi < 1e6:
        hi *= 2.0
        fhi = closure(hi)
    if flo * fhi > 0:
        raise ValueError("IAST spreading-pressure root not bracketed")
    pi = float(brentq(closure, lo, hi, xtol=1e-9, rtol=1e-8, maxiter=120))
    p01 = pressure_at_spreading_pressure(model1, pi, T_K, theta1)
    p02 = pressure_at_spreading_pressure(model2, pi, T_K, theta2)
    x1 = y1 * P / p01; x2 = y2 * P / p02
    s = x1 + x2
    x1 /= s; x2 /= s
    q01 = float(model_loading(model1, np.array([p01]), np.array([T_K]), theta1)[0])
    q02 = float(model_loading(model2, np.array([p02]), np.array([T_K]), theta2)[0])
    inv_qt = x1 / max(q01, EPS) + x2 / max(q02, EPS)
    qtot = 1.0 / max(inv_qt, EPS)
    q1 = x1 * qtot; q2 = x2 * qtot
    alpha = (x1 / max(x2, EPS)) / (y1 / y2)
    return {"q1_mol_kg": q1, "q2_mol_kg": q2, "x1": x1, "x2": x2, "alpha": alpha, "spreading_pressure": pi, "p0_1_bar": p01, "p0_2_bar": p02}


def inverse_regeneration_pressure(model: str, theta: Sequence[float], T_K: float, p_ads_bar: float, release_fraction: float, lower_bound_bar: float = PRESSURE_FLOOR_BAR) -> Tuple[float, bool]:
    qads = float(model_loading(model, np.array([p_ads_bar]), np.array([T_K]), theta)[0])
    target_q = (1.0 - release_fraction) * qads
    f = lambda p: float(model_loading(model, np.array([p]), np.array([T_K]), theta)[0]) - target_q
    flo = f(lower_bound_bar); fhi = f(p_ads_bar)
    if flo > 0 or fhi < 0:
        return np.nan, False
    try:
        return float(brentq(f, lower_bound_bar, p_ads_bar, xtol=1e-10, rtol=1e-9, maxiter=200)), True
    except Exception:
        return np.nan, False


def vacuum_work_proxy_kj_mol(T_K: float, p_low_bar: float, p_reference_bar: float = 1.0) -> float:
    """Ideal reversible pressure-ratio work proxy, not a pump duty prediction."""
    if p_low_bar <= 0 or p_reference_bar <= p_low_bar:
        return 0.0
    return (R_J_MOL_K * T_K * math.log(p_reference_bar / p_low_bar)) / 1000.0


def data_support_class(domain: Mapping[str, float], T_K: float, p_bar: float) -> str:
    """Label interpolation/extrapolation without pretending all extrapolation is equal."""
    tmin = domain["temperature_min_K"]; tmax = domain["temperature_max_K"]
    pmin = max(domain["pressure_min_bar"], PRESSURE_FLOOR_BAR); pmax = max(domain["pressure_max_bar"], pmin)
    t_out = max(tmin - T_K, T_K - tmax, 0.0)
    if p_bar < pmin:
        p_factor = pmin / max(p_bar, PRESSURE_FLOOR_BAR)
    elif p_bar > pmax:
        p_factor = p_bar / pmax
    else:
        p_factor = 1.0
    if t_out == 0 and p_factor <= 1.0:
        return "interpolation"
    if t_out <= 10.0 and p_factor <= 2.0:
        return "modest_extrapolation"
    if t_out <= 25.0 and p_factor <= 10.0:
        return "severe_extrapolation"
    return "unsupported"


def finite_difference_sensitivity(func, x: float, rel_step: float = 1e-3, abs_step: float = 1e-3) -> float:
    h = max(abs(x) * rel_step, abs_step)
    y0 = float(func(x))
    yp = float(func(x + h)); ym = float(func(x - h))
    deriv = (yp - ym) / (2.0 * h)
    return (x / y0) * deriv if abs(y0) > EPS else np.nan

# ---------------------------------------------------------------------------
# P10 composition-evidence parsing
# ---------------------------------------------------------------------------
# P10 needs a second harmonization layer in addition to unary isotherms.  The
# unary layer above intentionally stays generic and reusable; this section
# looks specifically for composition-dependent breakthrough traces and process
# anchors without forcing them into an equilibrium-isotherm schema.

BREAKTHROUGH_ALIASES = {
    "feed_co2_fraction": ["feed_co2_fraction", "feed_y_co2", "y_co2", "co2_feed_fraction", "co2_mole_fraction", "inlet_co2_fraction", "feed_co2", "co2_in"],
    "time_s": ["time_s", "time", "elapsed_time_s", "elapsed_s", "t_s", "seconds"],
    "outlet_co2_fraction": ["outlet_co2_fraction", "y_co2_out", "co2_out", "outlet_co2", "co2_mole_fraction_out", "co2_out_fraction"],
    "outlet_n2_fraction": ["outlet_n2_fraction", "y_n2_out", "n2_out", "outlet_n2"],
    "normalized_outlet": ["c_c0", "c_over_c0", "normalized_outlet", "normalized_concentration", "co2_c_c0"],
    "temperature_K": ["temperature_k", "temperature", "temp_k", "t_k"],
    "total_pressure_bar": ["total_pressure_bar", "pressure_bar", "total_pressure", "p_bar", "pressure"],
    "flow_mol_s": ["flow_mol_s", "molar_flow_mol_s", "total_flow_mol_s", "feed_flow_mol_s"],
    "bed_mass_g": ["bed_mass_g", "adsorbent_mass_g", "mass_adsorbent_g", "sample_mass_g"],
    "purity": ["purity", "co2_purity", "product_purity", "purity_co2"],
    "recovery": ["recovery", "co2_recovery", "capture_recovery", "recovery_co2"],
    "productivity": ["productivity", "co2_productivity"],
    "specific_energy": ["specific_energy", "energy", "specific_power", "specific_energy_mj_kg"],
    "regeneration_pressure_bar": ["regeneration_pressure_bar", "vacuum_pressure_bar", "desorption_pressure_bar", "p_low_bar", "p_reg_bar"],
    "adsorption_time_s": ["adsorption_time_s", "feed_time_s", "ads_time_s", "t_ads_s"],
}


def _find_any_column(columns: Sequence[Any], aliases: Sequence[str]) -> Optional[Any]:
    norm_map = {normalize_token(c): c for c in columns}
    for a in aliases:
        na = normalize_token(a)
        if na in norm_map:
            return norm_map[na]
    for nc, original in norm_map.items():
        for a in aliases:
            na = normalize_token(a)
            if nc.startswith(na + "_") or nc.endswith("_" + na):
                return original
    return None


def _fraction_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    finite = x[np.isfinite(x)]
    if len(finite) and np.nanmedian(np.abs(finite)) > 1.0 and np.nanmax(np.abs(finite)) <= 100.5:
        x = x / 100.0
    return x


def infer_feed_fraction_from_text(text: str) -> Optional[float]:
    """Infer the three CALF-20 anchor mixtures only when strongly encoded."""
    low = normalize_token(text)
    # Common forms: 5_95, 15_85, 30_70; co2_15; 15pct_co2.
    for pct in (5, 15, 30):
        other = 100 - pct
        patterns = [
            rf"(?:^|_){pct}_{other}(?:_|$)",
            rf"(?:^|_)co2_{pct}(?:_|$)",
            rf"(?:^|_){pct}_?pct_?co2(?:_|$)",
            rf"(?:^|_){pct}_?percent_?co2(?:_|$)",
        ]
        if any(re.search(pat, low) for pat in patterns):
            return pct / 100.0
    return None


def first_crossing_time(time_s: np.ndarray, z: np.ndarray, level: float) -> float:
    mask = np.isfinite(time_s) & np.isfinite(z)
    t = np.asarray(time_s[mask], dtype=float); v = np.asarray(z[mask], dtype=float)
    if len(t) < 2:
        return np.nan
    order = np.argsort(t); t = t[order]; v = v[order]
    idx = np.where(v >= level)[0]
    if not len(idx):
        return np.nan
    j = int(idx[0])
    if j == 0:
        return float(t[0])
    v0, v1 = float(v[j-1]), float(v[j]); t0, t1 = float(t[j-1]), float(t[j])
    if abs(v1-v0) < EPS:
        return float(t1)
    f = (level-v0)/(v1-v0)
    return float(t0 + np.clip(f, 0.0, 1.0)*(t1-t0))


def _iter_unique_input_tables(config: RunConfig, audit_payload: Mapping[str, Any], logger: logging.Logger):
    root = Path(config.input_folder)
    fps = audit_payload.get("fingerprints", pd.DataFrame())
    duplicate_paths = set()
    if len(fps) and "duplicate_content" in fps:
        duplicate_paths = set(fps.loc[fps["duplicate_content"].fillna(False), "relative_path"].astype(str))
    for p in list_input_files(root):
        rel = str(p.relative_to(root))
        if rel in duplicate_paths:
            continue
        try:
            for sheet, df in read_tabular_file(p).items():
                if isinstance(df, pd.DataFrame) and len(df):
                    yield p, rel, str(sheet), df
        except Exception:
            logger.exception("Could not read composition-evidence input %s", p)


def stage_04_composition_evidence(config: RunConfig, dirs: Mapping[str, Path], audit_payload: Mapping[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    trace_parts: List[pd.DataFrame] = []
    anchor_rows: List[Dict[str, Any]] = []
    metadata_anchor_parts: List[pd.DataFrame] = []
    process_parts: List[pd.DataFrame] = []

    for path, rel, sheet, df0 in _iter_unique_input_tables(config, audit_payload, logger):
        df = df0.copy()
        cols = list(df.columns)
        ycol = _find_any_column(cols, BREAKTHROUGH_ALIASES["feed_co2_fraction"])
        tcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["time_s"])
        ccol = _find_any_column(cols, BREAKTHROUGH_ALIASES["outlet_co2_fraction"])
        zcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["normalized_outlet"])
        inferred_y = infer_feed_fraction_from_text(f"{rel}_{sheet}")

        # Metadata-only composition anchors are scientifically useful even when
        # exact breakthrough coordinates cannot be redistributed.  They define
        # literature-tested feed compositions without fabricating a trace.
        # A table is treated as such an anchor only when it carries feed
        # composition plus explicit provenance/status-like metadata and does not
        # masquerade as a time-series trace or process-performance anchor.
        status_col = _find_any_column(cols, ["trace_status", "evidence_status", "support_status", "status"])
        fig_col = _find_any_column(cols, ["figure_location", "table_figure_si_location", "location", "source_location"])
        doi_col = _find_any_column(cols, ["doi"])
        sid_col = _find_any_column(cols, ["source_id", "source", "reference_id"])
        if ycol is not None and tcol is None and status_col is not None:
            meta = pd.DataFrame(index=df.index)
            meta["feed_co2_fraction"] = _fraction_series(df[ycol])
            meta["source_id"] = df[sid_col].astype(str) if sid_col is not None else f"{rel}::{sheet}"
            meta["evidence_status"] = df[status_col].astype(str)
            meta["source_location"] = df[fig_col].astype(str) if fig_col is not None else ""
            meta["doi"] = df[doi_col].astype(str) if doi_col is not None else ""
            Tmeta = _find_any_column(cols, BREAKTHROUGH_ALIASES["temperature_K"])
            Pmeta = _find_any_column(cols, BREAKTHROUGH_ALIASES["total_pressure_bar"])
            meta["temperature_K"] = temperature_to_K(df[Tmeta], detect_temperature_unit(str(Tmeta))) if Tmeta is not None else np.nan
            meta["total_pressure_bar"] = pressure_to_bar(df[Pmeta], detect_pressure_unit(str(Pmeta))) if Pmeta is not None else np.nan
            meta["source_file"] = rel
            meta["source_table"] = sheet
            meta = meta[np.isfinite(meta["feed_co2_fraction"])].copy()
            if len(meta):
                metadata_anchor_parts.append(meta)

        # Breakthrough traces: time plus either raw outlet composition or C/C0.
        if tcol is not None and (ccol is not None or zcol is not None):
            out = pd.DataFrame()
            out["time_s"] = pd.to_numeric(df[tcol], errors="coerce")
            if ycol is not None:
                out["feed_co2_fraction"] = _fraction_series(df[ycol])
            else:
                out["feed_co2_fraction"] = inferred_y
            if ccol is not None:
                out["outlet_co2_fraction"] = _fraction_series(df[ccol])
            else:
                out["outlet_co2_fraction"] = np.nan
            if zcol is not None:
                out["normalized_outlet"] = pd.to_numeric(df[zcol], errors="coerce")
            else:
                out["normalized_outlet"] = out["outlet_co2_fraction"] / out["feed_co2_fraction"].replace(0, np.nan)
            out["normalized_outlet"] = out["normalized_outlet"].clip(lower=-0.05, upper=1.5)
            Tcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["temperature_K"])
            Pcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["total_pressure_bar"])
            Fcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["flow_mol_s"])
            Mcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["bed_mass_g"])
            out["temperature_K"] = temperature_to_K(df[Tcol], detect_temperature_unit(str(Tcol))) if Tcol is not None else config.engineering.nominal_temperature_K
            out["total_pressure_bar"] = pressure_to_bar(df[Pcol], detect_pressure_unit(str(Pcol))) if Pcol is not None else config.engineering.adsorption_pressure_bar
            out["flow_mol_s"] = pd.to_numeric(df[Fcol], errors="coerce") if Fcol is not None else np.nan
            out["bed_mass_g"] = pd.to_numeric(df[Mcol], errors="coerce") if Mcol is not None else np.nan
            out["source_file"] = rel; out["source_table"] = sheet
            out["source_id"] = f"{rel}::{sheet}"
            out = out[np.isfinite(out["time_s"]) & np.isfinite(out["normalized_outlet"]) & np.isfinite(out["feed_co2_fraction"])].copy()
            if len(out):
                trace_parts.append(out)

        # Process anchors can be supplied as one-row or multi-row tables.
        pcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["purity"])
        rcol = _find_any_column(cols, BREAKTHROUGH_ALIASES["recovery"])
        if pcol is not None or rcol is not None:
            out = pd.DataFrame(index=df.index)
            if ycol is not None:
                out["feed_co2_fraction"] = _fraction_series(df[ycol])
            else:
                out["feed_co2_fraction"] = inferred_y if inferred_y is not None else config.engineering.nominal_feed_co2_fraction
            out["purity"] = _fraction_series(df[pcol]) if pcol is not None else np.nan
            out["recovery"] = _fraction_series(df[rcol]) if rcol is not None else np.nan
            for key in ["productivity", "specific_energy", "regeneration_pressure_bar", "adsorption_time_s"]:
                col = _find_any_column(cols, BREAKTHROUGH_ALIASES[key])
                out[key] = pd.to_numeric(df[col], errors="coerce") if col is not None else np.nan
            out["source_file"] = rel; out["source_table"] = sheet
            out["source_id"] = df[sid_col].astype(str).to_numpy() if sid_col is not None else f"{rel}::{sheet}"
            out["doi"] = df[doi_col].astype(str).to_numpy() if doi_col is not None else ""
            process_parts.append(out.reset_index(drop=True))

    traces = pd.concat(trace_parts, ignore_index=True) if trace_parts else pd.DataFrame(columns=[
        "time_s","feed_co2_fraction","outlet_co2_fraction","normalized_outlet","temperature_K","total_pressure_bar","flow_mol_s","bed_mass_g","source_file","source_table","source_id"])
    process = pd.concat(process_parts, ignore_index=True) if process_parts else pd.DataFrame(columns=[
        "feed_co2_fraction","purity","recovery","productivity","specific_energy","regeneration_pressure_bar","adsorption_time_s","source_file","source_table","source_id","doi"])
    composition_anchors = pd.concat(metadata_anchor_parts, ignore_index=True) if metadata_anchor_parts else pd.DataFrame(columns=[
        "feed_co2_fraction","source_id","evidence_status","source_location","doi","temperature_K","total_pressure_bar","source_file","source_table"])
    if len(composition_anchors):
        composition_anchors = composition_anchors.sort_values(["feed_co2_fraction","source_id"]).drop_duplicates().reset_index(drop=True)

    # Summarize each breakthrough trace by robust, reproducible front metrics.
    if len(traces):
        for (src, y), g in traces.groupby(["source_id", "feed_co2_fraction"], dropna=False):
            t = g["time_s"].to_numpy(float); z = g["normalized_outlet"].to_numpy(float)
            t10=first_crossing_time(t,z,0.10); t50=first_crossing_time(t,z,0.50); t90=first_crossing_time(t,z,0.90)
            flow=float(np.nanmedian(g["flow_mol_s"])) if np.isfinite(g["flow_mol_s"]).any() else np.nan
            mass_g=float(np.nanmedian(g["bed_mass_g"])) if np.isfinite(g["bed_mass_g"]).any() else np.nan
            # A simple breakthrough capacity proxy.  It becomes an absolute mol/kg
            # estimate only when molar flow and adsorbent mass are supplied.
            cap_proxy=float(y*t50) if np.isfinite(t50) else np.nan
            absolute=np.nan
            if np.isfinite(t50) and np.isfinite(flow) and np.isfinite(mass_g) and mass_g>0:
                absolute = y*flow*t50/(mass_g/1000.0)
            anchor_rows.append({
                "source_id":src,"feed_co2_fraction":float(y),"t10_s":t10,"t50_s":t50,"t90_s":t90,
                "front_width_s":t90-t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan,
                "dynamic_capacity_proxy_y_t50":cap_proxy,"breakthrough_capacity_mol_kg":absolute,
                "temperature_K":float(np.nanmedian(g["temperature_K"])),"total_pressure_bar":float(np.nanmedian(g["total_pressure_bar"])),
                "n_trace_points":len(g),
            })
    summary=pd.DataFrame(anchor_rows)
    save_table_multi(traces, dirs["data"] / "breakthrough_traces_harmonized", SAVE_MODES[config.save_mode])
    save_table_multi(summary, dirs["data"] / "breakthrough_anchor_summary", SAVE_MODES[config.save_mode])
    save_table_multi(composition_anchors, dirs["data"] / "composition_condition_anchors", SAVE_MODES[config.save_mode])
    save_table_multi(process, dirs["data"] / "process_anchor_data", SAVE_MODES[config.save_mode])
    return {"traces":traces,"summary":summary,"composition_anchors":composition_anchors,"process_anchors":process}


# ---------------------------------------------------------------------------
# P10 mixture thermodynamics and reduced-order process translation
# ---------------------------------------------------------------------------
def best_fit_descriptor(fits: Mapping[Tuple[str,str], Mapping[str,Any]], material: str, gas: str, model: Optional[str]=None) -> Optional[Tuple[str,np.ndarray,Mapping[str,float]]]:
    fs=fits.get((material,gas))
    if not fs:
        return None
    model=model or fs.get("best_model")
    if not model or not fs.get("models",{}).get(model,{}).get("success"):
        return None
    r=fs["models"][model]
    return model,np.asarray(r["theta"],dtype=float),fs["domain"]


def calf20_pair(fit_payload: Mapping[str,Any], model_co2: Optional[str]=None, model_n2: Optional[str]=None):
    fits=fit_payload["fits"]
    a=best_fit_descriptor(fits,"CALF-20","CO2",model_co2)
    b=best_fit_descriptor(fits,"CALF-20","N2",model_n2)
    if a is None or b is None:
        return None
    return a,b


def mixture_state(y: float, P: float, T: float, pair) -> Dict[str,float]:
    y=float(np.clip(y,1e-6,1-1e-6)); P=max(float(P),PRESSURE_FLOOR_BAR)
    (m1,th1,_),(m2,th2,_)=pair
    try:
        r=iast_binary(P,y,T,m1,th1,m2,th2)
        return {"q_co2":r["q1_mol_kg"],"q_n2":r["q2_mol_kg"],"x_co2":r["x1"],"alpha":r["alpha"],"iast_status":"ok"}
    except Exception:
        # Explicit ideal-partial-pressure fallback keeps diagnostics alive, but
        # is labelled so it can never be mistaken for successful IAST.
        q1=float(model_loading(m1,np.array([P*y]),np.array([T]),th1)[0])
        q2=float(model_loading(m2,np.array([P*(1-y)]),np.array([T]),th2)[0])
        x=q1/max(q1+q2,EPS)
        alpha=(x/max(1-x,EPS))/(y/max(1-y,EPS))
        return {"q_co2":q1,"q_n2":q2,"x_co2":x,"alpha":alpha,"iast_status":"partial_pressure_fallback"}


def composition_support_class(y: float, evidence_summary: pd.DataFrame, settings: EngineeringSettings, composition_anchors: Optional[pd.DataFrame]=None) -> str:
    """Classify evidence support without inventing missing experimental traces.

    Priority order:
    1) exact/near-exact machine-readable breakthrough trace;
    2) exact published composition condition represented only by provenance metadata;
    3) interpolation between published composition conditions inside 5--30% CO2;
    4) other states inside the declared 5--30% primary evidence envelope;
    5) exploratory 2--40% states outside that envelope.
    """
    y=float(y)
    trace_vals=np.sort(evidence_summary["feed_co2_fraction"].dropna().unique()) if len(evidence_summary) and "feed_co2_fraction" in evidence_summary else np.array([])
    if len(trace_vals):
        nearest=float(np.min(np.abs(trace_vals-y)))
        if nearest <= 0.0025:
            return "direct_breakthrough_trace_anchor"
    meta_vals=np.array([])
    if composition_anchors is not None and len(composition_anchors) and "feed_co2_fraction" in composition_anchors:
        meta_vals=np.sort(composition_anchors["feed_co2_fraction"].dropna().unique())
        if len(meta_vals) and float(np.min(np.abs(meta_vals-y))) <= 0.0025:
            return "published_composition_condition_anchor_no_trace"
    all_vals=np.sort(np.unique(np.r_[trace_vals,meta_vals])) if (len(trace_vals) or len(meta_vals)) else np.array([])
    if len(all_vals)>=2 and float(all_vals.min())-EPS <= y <= float(all_vals.max())+EPS:
        return "between_published_composition_conditions"
    if settings.primary_evidence_feed_min-EPS <= y <= settings.primary_evidence_feed_max+EPS:
        return "inside_primary_evidence_envelope_untraced"
    if settings.feed_co2_min-EPS <= y <= settings.feed_co2_max+EPS:
        return "exploratory_numerical_envelope"
    return "outside_declared_composition_domain"


def _anchor_front_factor_table(evidence_summary: pd.DataFrame, pair, settings: EngineeringSettings) -> pd.DataFrame:
    if evidence_summary.empty or evidence_summary["t50_s"].notna().sum()<2:
        return pd.DataFrame(columns=["feed_co2_fraction","front_factor","t50_s","q_co2_iast"])
    rows=[]
    for y,g in evidence_summary.groupby("feed_co2_fraction"):
        t50=float(np.nanmedian(g["t50_s"]))
        if not np.isfinite(t50) or y<=0:
            continue
        q=mixture_state(float(y),settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)["q_co2"]
        # For a fixed total flow and bed, y*t50/q is a useful relative front-
        # utilization observable.  It is normalized at the nominal 15% feed.
        rows.append({"feed_co2_fraction":float(y),"t50_s":t50,"q_co2_iast":q,"raw_front_factor":float(y)*t50/max(q,EPS)})
    out=pd.DataFrame(rows).sort_values("feed_co2_fraction") if rows else pd.DataFrame()
    if out.empty:
        return out
    y0=settings.nominal_feed_co2_fraction
    base=float(np.interp(y0,out["feed_co2_fraction"],out["raw_front_factor"]))
    out["front_factor"]=(out["raw_front_factor"]/max(base,EPS)).clip(0.5,1.5)
    return out


def front_factor(y: float, table: pd.DataFrame) -> float:
    if table.empty or len(table)<2:
        return 1.0
    xs=table["feed_co2_fraction"].to_numpy(float); ys=table["front_factor"].to_numpy(float)
    return float(np.clip(np.interp(y,xs,ys,left=ys[0],right=ys[-1]),0.5,1.5))


def release_state(y_feed: float, p_reg: float, pair, settings: EngineeringSettings, ff: float=1.0) -> Dict[str,float]:
    ads=mixture_state(y_feed,settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)
    reg=mixture_state(settings.product_reference_co2_fraction,p_reg,settings.nominal_temperature_K,pair)
    dco2=max(ads["q_co2"]-reg["q_co2"],0.0)*ff
    dn2=max(ads["q_n2"]-reg["q_n2"],0.0)*ff
    return {"q_ads_co2":ads["q_co2"],"q_ads_n2":ads["q_n2"],"q_reg_co2":reg["q_co2"],"q_reg_n2":reg["q_n2"],"dco2":dco2,"dn2":dn2,"alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"]}


def calibrate_nominal_surrogate(pair, front_table: pd.DataFrame, settings: EngineeringSettings, process_anchors: pd.DataFrame) -> Dict[str,float]:
    y0=settings.nominal_feed_co2_fraction; p0=settings.nominal_regeneration_pressure_bar
    purity=settings.nominal_purity_anchor; recovery=settings.nominal_recovery_anchor
    # If a user-supplied nominal process anchor is present, prefer its medians.
    if len(process_anchors):
        g=process_anchors[np.abs(process_anchors["feed_co2_fraction"]-y0)<=0.01]
        if len(g):
            pv=g["purity"].dropna(); rv=g["recovery"].dropna()
            if len(pv): purity=float(np.nanmedian(pv))
            if len(rv): recovery=float(np.nanmedian(rv))
    ff=front_factor(y0,front_table)
    rel=release_state(y0,p0,pair,settings,ff)
    F0=rel["dco2"]/max(recovery*y0,EPS)
    # Solve the single light-gas carryover scale needed to reproduce the nominal
    # purity.  This is a calibration parameter, not a new material property.
    target_n2=rel["dco2"]*(1.0/max(purity,EPS)-1.0)
    contam=target_n2/max(rel["dn2"],EPS) if rel["dn2"]>EPS else 0.0
    contam=float(np.clip(contam,0.0,100.0))
    return {"nominal_feed_mol_kg_cycle":float(F0),"light_gas_carryover_scale":contam,"purity_anchor":purity,"recovery_anchor":recovery,"nominal_dco2_mol_kg":rel["dco2"],"nominal_dn2_mol_kg":rel["dn2"],"front_factor_nominal":ff}


def vacuum_work_kJ_mol_gas(T: float, p_high: float, p_low: float, efficiency: float) -> float:
    if p_low<=0 or p_high<=p_low:
        return 0.0
    return R_J_MOL_K*T*math.log(p_high/p_low)/1000.0/max(efficiency,EPS)


def _process_metrics_from_release(y: float, p_reg: float, timing_scale: float, rel: Mapping[str,float], calibration: Mapping[str,float], settings: EngineeringSettings, *, contamination_multiplier: float=1.0, vacuum_efficiency: Optional[float]=None) -> Dict[str,float]:
    """Convert a precomputed equilibrium release state to process-facing metrics."""
    y=float(np.clip(y,1e-5,1-1e-5)); timing_scale=float(max(timing_scale,0.05))
    F=float(calibration["nominal_feed_mol_kg_cycle"])*timing_scale
    nco2_feed=max(y*F,EPS); nn2_feed=max((1-y)*F,0.0)
    co2_release_capacity=float(rel["dco2"])
    recovery_capacity_ratio=co2_release_capacity/max(nco2_feed,EPS)
    recovery_feed_limited=bool(co2_release_capacity >= nco2_feed*(1.0-1e-12))
    nco2=min(co2_release_capacity,nco2_feed)
    nn2=min(float(rel["dn2"])*float(calibration["light_gas_carryover_scale"])*contamination_multiplier,nn2_feed)
    purity=nco2/max(nco2+nn2,EPS); recovery=nco2/max(nco2_feed,EPS)
    cycle=settings.fixed_nonfeed_cycle_time_s+settings.nominal_adsorption_time_s*timing_scale
    productivity=nco2/max(cycle,EPS)*3600.0
    eta=vacuum_efficiency if vacuum_efficiency is not None else settings.vacuum_efficiency
    wgas=vacuum_work_kJ_mol_gas(settings.nominal_temperature_K,settings.adsorption_pressure_bar,p_reg,eta)
    specific=(wgas*(nco2+nn2)/nco2) if nco2 > 1.0e-8 else np.nan
    meets=purity>=settings.target_purity and recovery>=settings.target_recovery
    return {"feed_co2_fraction":y,"p_reg_bar":float(p_reg),"timing_scale":timing_scale,"purity_estimate":purity,"recovery_estimate":recovery,"productivity_mol_kg_h":productivity,"specific_vacuum_work_kJ_mol_CO2":specific,"co2_product_mol_kg_cycle":nco2,"n2_product_mol_kg_cycle":nn2,"co2_feed_mol_kg_cycle":nco2_feed,"co2_release_capacity_mol_kg_cycle":co2_release_capacity,"recovery_capacity_ratio_unbounded":recovery_capacity_ratio,"recovery_feed_limited":recovery_feed_limited,"mass_balance_recovery_violation":bool(recovery>1.0+1e-12),"front_factor":float(rel.get("front_factor",1.0)),"working_capacity_co2_mol_kg":float(rel["dco2"]),"working_capacity_n2_mol_kg":float(rel["dn2"]),"alpha_ads":float(rel["alpha_ads"]),"meets_targets":bool(meets),"iast_status_ads":rel["iast_status_ads"],"iast_status_reg":rel["iast_status_reg"]}


def process_surrogate(y: float, p_reg: float, timing_scale: float, pair, calibration: Mapping[str,float], front_table: pd.DataFrame, settings: EngineeringSettings, *, front_multiplier: float=1.0, contamination_multiplier: float=1.0, vacuum_efficiency: Optional[float]=None) -> Dict[str,float]:
    """Transparent reduced-order cyclic translation on a 1 kg adsorbent basis."""
    ff=front_factor(y,front_table)*front_multiplier
    rel=release_state(y,p_reg,pair,settings,ff); rel["front_factor"]=ff
    return _process_metrics_from_release(y,p_reg,timing_scale,rel,calibration,settings,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)


def _control_move_components(row: Mapping[str, Any], settings: EngineeringSettings) -> Dict[str, float]:
    """Return transparent, dimensionless control movements from the nominal state.

    The pressure and timing movements are normalized only by the declared normal
    operating ranges.  They are retained separately.  ``control_move_linf`` is
    used solely as a deterministic closest-to-nominal selector within a given
    intervention class; it is not presented as a scientific performance metric.
    """
    p0=max(settings.nominal_regeneration_pressure_bar, EPS)
    p=max(float(row.get("p_reg_bar", p0)), EPS)
    t=max(float(row.get("timing_scale", 1.0)), EPS)
    p_span=max(abs(math.log(max(settings.regeneration_pressure_floor_bar, EPS)/p0)), EPS)
    t_span=max(abs(math.log(max(settings.timing_min_scale, EPS))), abs(math.log(max(settings.timing_max_scale, EPS))), EPS)
    pmove=abs(math.log(p/p0))/p_span
    tmove=abs(math.log(t))/t_span
    return {
        "pressure_move_log": abs(math.log(p/p0)),
        "timing_move_log": abs(math.log(t)),
        "pressure_move_normalized": pmove,
        "timing_move_normalized": tmove,
        "control_move_linf": max(pmove, tmove),
        "control_move_l1": pmove+tmove,
    }


def _decorate_control_move(row: Mapping[str, Any], settings: EngineeringSettings) -> Dict[str, Any]:
    out=dict(row)
    out.update(_control_move_components(out, settings))
    return out


def _choose_best_feasible(rows: List[Dict[str,Any]], settings: EngineeringSettings) -> Optional[Dict[str,Any]]:
    """Choose the closest feasible state without a weighted adaptation index."""
    feasible=[_decorate_control_move(r, settings) for r in rows if r["meets_targets"]]
    if not feasible:
        return None
    feasible.sort(key=lambda r:(r["control_move_linf"], r["control_move_l1"], r["specific_vacuum_work_kJ_mol_CO2"], -r["productivity_mol_kg_h"]))
    return dict(feasible[0])

def solve_adaptation_levels(y: float, pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int, *, front_multiplier: float=1.0, contamination_multiplier: float=1.0, vacuum_efficiency: Optional[float]=None) -> pd.DataFrame:
    """Evaluate nested operating-intervention classes for one feed composition.

    Classes are intentionally defined by control authority rather than by a
    scalar score: 0=frozen; 1=either timing or regeneration pressure alone;
    2=pressure and timing together within declared normal bounds; 3=extended
    envelope diagnostic.  The extended class is not counted as normal process
    robustness in the headline interpretation.
    """
    p0=settings.nominal_regeneration_pressure_bar
    ff=front_factor(y,front_table)*front_multiplier
    release_cache: Dict[float,Dict[str,float]]={}
    def eval_state(p: float,t: float) -> Dict[str,float]:
        key=round(float(p),12)
        if key not in release_cache:
            rr=release_state(y,float(p),pair,settings,ff); rr["front_factor"]=ff; release_cache[key]=rr
        return _process_metrics_from_release(y,float(p),float(t),release_cache[key],calibration,settings,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
    fixed=_decorate_control_move(eval_state(p0,1.0), settings)
    rows=[{**fixed,"adaptation_level":0,"adaptation_name":"frozen_process","adaptation_variant":"frozen","success":fixed["meets_targets"]}]
    if fixed["meets_targets"]:
        for lev,name,var in [(1,"single_knob","none_needed"),(2,"coordinated_two_knob","none_needed"),(3,"extended_envelope_check","none_needed")]:
            rows.append({**fixed,"adaptation_level":lev,"adaptation_name":name,"adaptation_variant":var,"success":True})
        return pd.DataFrame(rows)
    timings=np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(15,pressure_n))
    pressures=np.geomspace(settings.regeneration_pressure_floor_bar,p0,max(11,pressure_n))
    timing_best=_choose_best_feasible([eval_state(p0,t) for t in timings],settings)
    pressure_best=_choose_best_feasible([eval_state(p,1.0) for p in pressures],settings)
    one=[]
    if timing_best: one.append({**timing_best,"adaptation_variant":"timing_only"})
    if pressure_best: one.append({**pressure_best,"adaptation_variant":"pressure_only"})
    one.sort(key=lambda r:(r["control_move_linf"],r["control_move_l1"],r["specific_vacuum_work_kJ_mol_CO2"]))
    best=one[0] if one else None
    rows.append({**(best or fixed),"adaptation_level":1,"adaptation_name":"single_knob","adaptation_variant":(best or {}).get("adaptation_variant","none_feasible"),"success":best is not None})
    nt=max(9,int(math.sqrt(max(pressure_n,9))*2)); npg=max(9,int(math.sqrt(max(pressure_n,9))*2))
    joint=[eval_state(p,t) for p in np.geomspace(settings.regeneration_pressure_floor_bar,p0,npg) for t in np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt)]
    best=_choose_best_feasible(joint,settings)
    rows.append({**(best or fixed),"adaptation_level":2,"adaptation_name":"coordinated_two_knob","adaptation_variant":"pressure_plus_timing" if best else "none_feasible","success":best is not None})
    pmin=max(PRESSURE_FLOOR_BAR,min(0.003,settings.regeneration_pressure_floor_bar*0.3))
    ext=[eval_state(p,t) for p in np.geomspace(pmin,p0,npg) for t in np.linspace(0.20,2.0,nt)]
    best=_choose_best_feasible(ext,settings)
    rows.append({**(best or fixed),"adaptation_level":3,"adaptation_name":"extended_envelope_check","adaptation_variant":"extended_pressure_timing" if best else "none_feasible","success":best is not None})
    return pd.DataFrame(rows)

def _contiguous_interval_containing_nominal(mask_df: pd.DataFrame, nominal_y: float) -> Tuple[float,float]:
    """Return the contiguous feed-fraction interval containing the nominal point."""
    if mask_df.empty:
        return (np.nan,np.nan)
    d=mask_df.sort_values("feed_co2_fraction").reset_index(drop=True)
    y=d["feed_co2_fraction"].to_numpy(float); ok=d["eligible"].to_numpy(bool)
    j=int(np.argmin(np.abs(y-nominal_y)))
    if not ok[j]:
        return (np.nan,np.nan)
    lo=j; hi=j
    while lo>0 and ok[lo-1]: lo-=1
    while hi<len(ok)-1 and ok[hi+1]: hi+=1
    return (float(y[lo]),float(y[hi]))


def summarize_intervention_windows(selected: pd.DataFrame, settings: EngineeringSettings) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows=[]
    labels={0:"frozen_safe",1:"single_knob_or_less",2:"coordinated_two_knob_or_less"}
    ymin=float(selected["feed_co2_fraction"].min()); ymax=float(selected["feed_co2_fraction"].max())
    for level,label in labels.items():
        tmp=selected[["feed_co2_fraction","success","adaptation_level"]].copy()
        tmp["eligible"]=tmp["success"].astype(bool) & (tmp["adaptation_level"]<=level)
        lo,hi=_contiguous_interval_containing_nominal(tmp[["feed_co2_fraction","eligible"]],settings.nominal_feed_co2_fraction)
        lower_status="unresolved"; upper_status="unresolved"
        if np.isfinite(lo):
            lower_status="left_censored_by_feed_domain_limit" if abs(lo-ymin)<=1e-12 else "interior_grid_transition"
        if np.isfinite(hi):
            upper_status="right_censored_by_feed_domain_limit" if abs(hi-ymax)<=1e-12 else "interior_grid_transition"
        rows.append({
            "intervention_ceiling":label,"max_adaptation_level":level,
            "lower_feed_co2_fraction":lo,"upper_feed_co2_fraction":hi,
            "window_width_fraction_observed":hi-lo if np.isfinite(lo) and np.isfinite(hi) else np.nan,
            "lower_shift_percentage_points":100*(lo-settings.nominal_feed_co2_fraction) if np.isfinite(lo) else np.nan,
            "upper_shift_percentage_points":100*(hi-settings.nominal_feed_co2_fraction) if np.isfinite(hi) else np.nan,
            "lower_boundary_status":lower_status,"upper_boundary_status":upper_status,
            "lower_boundary_resolved":lower_status=="interior_grid_transition",
            "upper_boundary_resolved":upper_status=="interior_grid_transition",
        })
    return pd.DataFrame(rows)


def intervention_boundary_diagnostics(selected: pd.DataFrame, windows: pd.DataFrame, settings: EngineeringSettings, fixed: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    """Describe each deterministic edge and the immediately adjacent outside state.

    Domain-censored edges are explicitly labelled so an explored-domain limit is
    never mistaken for a physical intervention boundary.
    """
    if selected.empty or windows.empty:
        return pd.DataFrame()
    d=selected.sort_values("feed_co2_fraction").reset_index(drop=True)
    f=(fixed.sort_values("feed_co2_fraction").reset_index(drop=True) if fixed is not None and len(fixed) else pd.DataFrame())
    rows=[]
    for _,w in windows.iterrows():
        for side,col,status_col in [("lower","lower_feed_co2_fraction","lower_boundary_status"),("upper","upper_feed_co2_fraction","upper_boundary_status")]:
            y=float(w[col]) if pd.notna(w[col]) else np.nan
            status=str(w.get(status_col,"unresolved"))
            if not np.isfinite(y):
                rows.append({"intervention_ceiling":w["intervention_ceiling"],"side":side,"feed_co2_fraction":np.nan,"boundary_status":status,"resolved":False})
                continue
            j=int(np.argmin(np.abs(d["feed_co2_fraction"].to_numpy(float)-y))); r=d.iloc[j]
            jo=j-1 if side=="lower" else j+1
            outside=d.iloc[jo] if 0<=jo<len(d) else None
            fr=None
            if outside is not None and len(f):
                fr=f.iloc[int(np.argmin(np.abs(f["feed_co2_fraction"].to_numpy(float)-float(outside["feed_co2_fraction"]))))]
            if fr is not None:
                pdef=max(settings.target_purity-float(fr.get("purity_estimate",np.nan)),0.0) if np.isfinite(fr.get("purity_estimate",np.nan)) else np.nan
                rdef=max(settings.target_recovery-float(fr.get("recovery_estimate",np.nan)),0.0) if np.isfinite(fr.get("recovery_estimate",np.nan)) else np.nan
                fail="both" if pdef>EPS and rdef>EPS else ("purity" if pdef>EPS else ("recovery" if rdef>EPS else "neither"))
            else:
                pdef=rdef=np.nan; fail="not_available"
            rows.append({
                "intervention_ceiling":w["intervention_ceiling"],"side":side,"feed_co2_fraction":float(r["feed_co2_fraction"]),
                "boundary_status":status,"resolved":status=="interior_grid_transition",
                "composition_support":r.get("composition_support",""),"inside_minimum_required_class":r.get("robustness_class",""),
                "inside_adaptation_level":r.get("adaptation_level",np.nan),"inside_adaptation_variant":r.get("adaptation_variant",""),
                "outside_feed_co2_fraction":float(outside["feed_co2_fraction"]) if outside is not None else np.nan,
                "outside_minimum_required_class":outside.get("robustness_class","") if outside is not None else "domain_limit",
                "outside_adaptation_level":outside.get("adaptation_level",np.nan) if outside is not None else np.nan,
                "outside_frozen_purity_deficit":pdef,"outside_frozen_recovery_deficit":rdef,
                "outside_frozen_limiting_target":fail,
                "p_reg_bar":r.get("p_reg_bar",np.nan),"timing_scale":r.get("timing_scale",np.nan),
                "purity_estimate":r.get("purity_estimate",np.nan),"recovery_estimate":r.get("recovery_estimate",np.nan),
            })
    return pd.DataFrame(rows)


def adaptation_recovery_decomposition(fixed: pd.DataFrame, selected: pd.DataFrame, settings: EngineeringSettings) -> pd.DataFrame:
    """Decompose off-design purity/recovery deficits into recovered and residual parts.

    This deliberately avoids a weighted scalar flexibility score.  Purity and
    recovery are kept as separate physical target deficits so the small CALF-20
    paper can distinguish process-recoverable loss from residual limitation
    without claiming a cross-material notion of "source flexibility."
    """
    if fixed.empty or selected.empty:
        return pd.DataFrame()
    f=fixed.set_index("feed_co2_fraction")
    s=selected.set_index("feed_co2_fraction")
    rows=[]
    for y in sorted(set(f.index).intersection(set(s.index))):
        rf=f.loc[y]; rs=s.loc[y]
        if isinstance(rf,pd.DataFrame): rf=rf.iloc[0]
        if isinstance(rs,pd.DataFrame): rs=rs.iloc[0]
        pdef_f=max(settings.target_purity-float(rf["purity_estimate"]),0.0)
        rdef_f=max(settings.target_recovery-float(rf["recovery_estimate"]),0.0)
        pdef_a=max(settings.target_purity-float(rs["purity_estimate"]),0.0) if np.isfinite(rs["purity_estimate"]) else pdef_f
        rdef_a=max(settings.target_recovery-float(rs["recovery_estimate"]),0.0) if np.isfinite(rs["recovery_estimate"]) else rdef_f
        rows.append({
            "feed_co2_fraction":float(y),
            "composition_support":rs.get("composition_support",""),
            "selected_adaptation_level":int(rs["adaptation_level"]) if np.isfinite(rs["adaptation_level"]) else np.nan,
            "purity_deficit_frozen":pdef_f,
            "purity_deficit_recovered":max(pdef_f-pdef_a,0.0),
            "purity_deficit_residual":pdef_a,
            "recovery_deficit_frozen":rdef_f,
            "recovery_deficit_recovered":max(rdef_f-rdef_a,0.0),
            "recovery_deficit_residual":rdef_a,
            "purity_gain_absolute":float(rs["purity_estimate"])-float(rf["purity_estimate"]) if np.isfinite(rs["purity_estimate"]) else np.nan,
            "recovery_gain_absolute":float(rs["recovery_estimate"])-float(rf["recovery_estimate"]) if np.isfinite(rs["recovery_estimate"]) else np.nan,
        })
    return pd.DataFrame(rows)


def threshold_sensitivity_recalculate(
    ys: Sequence[float],
    pair,
    calibration: Mapping[str,float],
    front_table: pd.DataFrame,
    settings: EngineeringSettings,
    pressure_n: int,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Re-run the minimal-adaptation search under alternate purity/recovery targets.

    This is intentionally recalculated rather than inferred from the baseline
    selected states, because a state that is suboptimal for 95/88 can become the
    closest feasible state under a different threshold pair.
    """
    selected_rows=[]; window_rows=[]
    purities=sorted(set(tuple(settings.purity_threshold_sensitivity)+(settings.target_purity,)))
    recoveries=sorted(set(tuple(settings.recovery_threshold_sensitivity)+(settings.target_recovery,)))
    for pt in purities:
        for rt in recoveries:
            s2=dataclasses.replace(settings,target_purity=float(pt),target_recovery=float(rt))
            per_y=[]
            for y in ys:
                a=_solve_adaptation_levels_grid_v130(float(y),pair,calibration,front_table,s2,pressure_n)
                normal=a[(a["success"]) & (a["adaptation_level"]<=2)].sort_values("adaptation_level")
                if len(normal):
                    row=normal.iloc[0].to_dict()
                    row["threshold_success"]=True
                else:
                    ext=a[(a["success"]) & (a["adaptation_level"]==3)]
                    row=(ext.iloc[0] if len(ext) else a.iloc[-1]).to_dict()
                    row["threshold_success"]=bool(len(ext))
                row["feed_co2_fraction"]=float(y)
                row["threshold_purity"]=float(pt)
                row["threshold_recovery"]=float(rt)
                per_y.append(row); selected_rows.append(row)
            sdf=pd.DataFrame(per_y)
            for lev,label in {0:"frozen_safe",1:"single_knob_or_less",2:"coordinated_two_knob_or_less"}.items():
                tmp=sdf[["feed_co2_fraction","threshold_success","adaptation_level"]].copy()
                tmp["eligible"]=tmp["threshold_success"].astype(bool)&(tmp["adaptation_level"]<=lev)
                lo,hi=_contiguous_interval_containing_nominal(tmp[["feed_co2_fraction","eligible"]],settings.nominal_feed_co2_fraction)
                window_rows.append({
                    "threshold_purity":float(pt),"threshold_recovery":float(rt),
                    "intervention_ceiling":label,"max_adaptation_level":lev,
                    "lower_feed_co2_fraction":lo,"upper_feed_co2_fraction":hi,
                    "window_width_fraction":hi-lo if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                })
    return pd.DataFrame(selected_rows),pd.DataFrame(window_rows)


def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    pair=calf20_pair(fit_payload)
    if pair is None:
        logger.warning("CALF-20 CO2 and N2 fits are required for the P10 feed-composition robustness analysis; process outputs will be empty.")
        empty=pd.DataFrame(); return {"intrinsic":empty,"fixed":empty,"adaptation":empty,"selected":empty,"pressure_atlas":empty,"windows":empty,"boundary_diagnostics":empty,"decomposition":empty,"calibration":{},"front_table":empty}
    settings=config.engineering; level=LEVELS[config.level]
    front_table=_anchor_front_factor_table(evidence_payload["summary"],pair,settings)
    calibration=calibrate_nominal_surrogate(pair,front_table,settings,evidence_payload["process_anchors"])
    atomic_json(calibration,dirs["engineering"] / "nominal_surrogate_calibration.json")
    save_table_multi(front_table,dirs["engineering"] / "breakthrough_front_factor_anchors",SAVE_MODES[config.save_mode])
    # Always include the primary literature composition anchors exactly, even
    # when the linspace grid would otherwise miss 5%, 15% or 30% CO2.
    ys=np.unique(np.r_[
        np.linspace(settings.feed_co2_min,settings.feed_co2_max,int(level["composition_n"])),
        settings.primary_evidence_feed_min,
        settings.nominal_feed_co2_fraction,
        settings.primary_evidence_feed_max,
    ]); ys.sort(); pressure_n=int(level["pressure_n"])
    base_p=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,max(11,pressure_n))
    npg=max(9,int(math.sqrt(max(pressure_n,9))*2)); joint_p=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,npg)
    pmin_ext=max(PRESSURE_FLOOR_BAR,min(0.003,settings.regeneration_pressure_floor_bar*0.3)); ext_p=np.geomspace(pmin_ext,settings.nominal_regeneration_pressure_bar,npg)
    diagnostic_p=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,pressure_n)
    p0=settings.nominal_regeneration_pressure_bar
    all_p=sorted(set(float(x) for x in np.r_[base_p,joint_p,ext_p,diagnostic_p,[p0]]))
    reg_cache={p:mixture_state(settings.product_reference_co2_fraction,p,settings.nominal_temperature_K,pair) for p in all_p}
    timings=np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(15,pressure_n)); nt=max(9,int(math.sqrt(max(pressure_n,9))*2)); joint_t=np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt); ext_t=np.linspace(0.20,2.0,nt)
    intrinsic_rows=[]; fixed_rows=[]; adaptation_parts=[]; selected_rows=[]; pa_rows=[]
    for i,y in enumerate(ys):
        y=float(y); ads=mixture_state(y,settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair); ff=front_factor(y,front_table)
        def rel_for(p):
            reg=reg_cache[float(p)]
            return {"dco2":max(ads["q_co2"]-reg["q_co2"],0.0)*ff,"dn2":max(ads["q_n2"]-reg["q_n2"],0.0)*ff,"alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"],"front_factor":ff}
        intrinsic_rows.append({"feed_co2_fraction":y,"q_co2_mol_kg":ads["q_co2"],"q_n2_mol_kg":ads["q_n2"],"adsorbed_x_co2":ads["x_co2"],"iast_selectivity_co2_n2":ads["alpha"],"working_capacity_co2_mol_kg":rel_for(p0)["dco2"],"working_capacity_n2_mol_kg":rel_for(p0)["dn2"],"front_factor":ff,"composition_support":composition_support_class(y,evidence_payload["summary"],settings,evidence_payload.get("composition_anchors")),"iast_status":ads["iast_status"]})
        fixed=_decorate_control_move(_process_metrics_from_release(y,p0,1.0,rel_for(p0),calibration,settings),settings)
        rows=[{**fixed,"adaptation_level":0,"adaptation_name":"frozen_process","adaptation_variant":"frozen","success":fixed["meets_targets"]}]
        if fixed["meets_targets"]:
            for lev,name in [(1,"single_knob"),(2,"coordinated_two_knob"),(3,"extended_envelope_check")]:
                rows.append({**fixed,"adaptation_level":lev,"adaptation_name":name,"adaptation_variant":"none_needed","success":True})
        else:
            timing_best=_choose_best_feasible([_process_metrics_from_release(y,p0,t,rel_for(p0),calibration,settings) for t in timings],settings)
            pressure_best=_choose_best_feasible([_process_metrics_from_release(y,float(p),1.0,rel_for(float(p)),calibration,settings) for p in base_p],settings)
            one=[]
            if timing_best: one.append({**timing_best,"adaptation_variant":"timing_only"})
            if pressure_best: one.append({**pressure_best,"adaptation_variant":"pressure_only"})
            one.sort(key=lambda r:(r["control_move_linf"],r["control_move_l1"],r["specific_vacuum_work_kJ_mol_CO2"]))
            best=one[0] if one else None
            rows.append({**(best or fixed),"adaptation_level":1,"adaptation_name":"single_knob","adaptation_variant":(best or {}).get("adaptation_variant","none_feasible"),"success":best is not None})
            cand=[_process_metrics_from_release(y,float(p),float(t),rel_for(float(p)),calibration,settings) for p in joint_p for t in joint_t]
            best=_choose_best_feasible(cand,settings)
            rows.append({**(best or fixed),"adaptation_level":2,"adaptation_name":"coordinated_two_knob","adaptation_variant":"pressure_plus_timing" if best else "none_feasible","success":best is not None})
            cand=[_process_metrics_from_release(y,float(p),float(t),rel_for(float(p)),calibration,settings) for p in ext_p for t in ext_t]
            best=_choose_best_feasible(cand,settings)
            rows.append({**(best or fixed),"adaptation_level":3,"adaptation_name":"extended_envelope_check","adaptation_variant":"extended_pressure_timing" if best else "none_feasible","success":best is not None})
        a=pd.DataFrame(rows); a["composition_support"]=composition_support_class(y,evidence_payload["summary"],settings,evidence_payload.get("composition_anchors")); adaptation_parts.append(a); fixed_rows.append(a[a["adaptation_level"]==0].iloc[0].to_dict())
        normal=a[(a["success"]) & (a["adaptation_level"]<=2)].sort_values("adaptation_level")
        if len(normal):
            best=normal.iloc[0].to_dict()
            best["robustness_class"]={0:"robust_frozen",1:"single_knob_recoverable",2:"coordinated_recoverable"}[int(best["adaptation_level"])]
        else:
            ext=a[(a["success"]) & (a["adaptation_level"]==3)]
            best=(ext.iloc[0] if len(ext) else a.iloc[-1]).to_dict()
            best["robustness_class"]="outside_declared_control_envelope" if len(ext) else "infeasible_in_reduced_order_model"
        selected_rows.append(best)
        for p in diagnostic_p:
            pa_rows.append(_process_metrics_from_release(y,float(p),1.0,rel_for(float(p)),calibration,settings) | {"composition_support":composition_support_class(y,evidence_payload["summary"],settings,evidence_payload.get("composition_anchors"))})
        if (i+1)%max(1,len(ys)//10)==0: logger.info("Feed-composition robustness analysis: %d/%d feed fractions",i+1,len(ys))
    intrinsic=pd.DataFrame(intrinsic_rows); fixed=pd.DataFrame(fixed_rows); adaptation=pd.concat(adaptation_parts,ignore_index=True); selected=pd.DataFrame(selected_rows); patlas=pd.DataFrame(pa_rows)
    windows=summarize_intervention_windows(selected,settings)
    boundary_diagnostics=intervention_boundary_diagnostics(selected,windows,settings,fixed)
    decomposition=adaptation_recovery_decomposition(fixed,selected,settings)
    for name,df in [("intrinsic_composition_response",intrinsic),("frozen_process_offdesign",fixed),("intervention_candidates",adaptation),("minimum_intervention_selected",selected),("nested_intervention_windows",windows),("intervention_boundary_diagnostics",boundary_diagnostics),("adaptation_recovery_decomposition",decomposition),("pressure_response_diagnostic",patlas)]:
        save_table_multi(df,dirs["engineering"] / name,SAVE_MODES[config.save_mode])
    return {"intrinsic":intrinsic,"fixed":fixed,"adaptation":adaptation,"selected":selected,"pressure_atlas":patlas,"windows":windows,"boundary_diagnostics":boundary_diagnostics,"decomposition":decomposition,"calibration":calibration,"front_table":front_table}


# ---------------------------------------------------------------------------
# Uncertainty-aware composition-robustness classification
# ---------------------------------------------------------------------------
def _successful_boot_theta(uncertainty_payload, material, gas, model) -> List[np.ndarray]:
    arr=[]
    for r in uncertainty_payload.get("samples",{}).get((material,gas,model),[]):
        if r.get("success"):
            arr.append(np.asarray(r["theta"],dtype=float))
    return arr


def _draw_pair(fit_payload, uncertainty_payload, rng: np.random.Generator, allow_model_form: bool):
    fits=fit_payload["fits"]; out=[]
    for gas in ["CO2","N2"]:
        fs=fits.get(("CALF-20",gas),{})
        models=list(fs.get("retained_models",[])) if allow_model_form else [fs.get("best_model")]
        models=[m for m in models if m and fs.get("models",{}).get(m,{}).get("success")]
        if not models: return None
        model=models[int(rng.integers(0,len(models)))]
        samples=_successful_boot_theta(uncertainty_payload,"CALF-20",gas,model)
        theta=samples[int(rng.integers(0,len(samples)))] if samples else np.asarray(fs["models"][model]["theta"],dtype=float)
        out.append((model,theta,fs["domain"]))
    return tuple(out)


def stage_06_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], uncertainty_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    if atlas_payload["intrinsic"].empty:
        return {"classification":pd.DataFrame(),"boundaries":pd.DataFrame(),"draw_boundaries":pd.DataFrame(),"threshold_sensitivity":pd.DataFrame(),"threshold_windows":pd.DataFrame()}
    settings=config.engineering; level=LEVELS[config.level]
    ys=atlas_payload["intrinsic"]["feed_co2_fraction"].to_numpy(float); n_draw=int(config.uncertainty_draws_override if config.uncertainty_draws_override is not None else level["uncertainty_draws_atlas"]); allow_model_form=level["bootstrap_models"]=="retained"
    class_order=["robust_frozen","single_knob_recoverable","coordinated_recoverable","outside_control_envelope"]
    per_y={float(y):{k:0 for k in class_order} for y in ys}; draw_bounds=[]
    partial_path=dirs["uncertainty"] / "feed_composition_robustness_draws_partial.pkl"
    partial_key=stable_json_hash({"scientific":config.scientific_hash_payload(),"n_draw":n_draw,"ys":ys.tolist(),"fit_summary":fit_payload.get("summary",pd.DataFrame()).to_dict("records")})
    start_draw=0
    if partial_path.exists():
        try:
            st=safe_read_pickle(partial_path)
            if st.get("key")==partial_key:
                start_draw=int(st.get("next_draw",0)); per_y=st.get("per_y",per_y); draw_bounds=st.get("draw_bounds",[])
                logger.info("Feed-composition robustness uncertainty resume: %d/%d draws already completed",start_draw,n_draw)
        except Exception:
            logger.exception("Could not read partial feed-composition robustness uncertainty checkpoint; rebuilding it")
    npg=max(7,min(13,int(level["pressure_n"]//2+3))); nt=max(7,min(13,int(level["pressure_n"]//2+3)))
    pgrid=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,npg); tgrid=np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt)
    batch_draws=max(1,int(RAM_MODES[config.ram_mode]["batch"]))
    for d in range(start_draw,n_draw):
        drng=np.random.default_rng(config.seed+6006+d*104729)
        pair=_draw_pair(fit_payload,uncertainty_payload,drng,allow_model_form)
        if pair is None: continue
        fmult=float(drng.uniform(*settings.front_factor_range)); cmult=float(drng.uniform(*settings.contamination_scale_range)); eta=float(drng.uniform(*settings.vacuum_efficiency_range))
        cal=calibrate_nominal_surrogate_conditioned(pair,atlas_payload["front_table"],settings,evidence_payload["process_anchors"],front_multiplier=fmult,contamination_multiplier=cmult)
        reg_cache={float(p):mixture_state(settings.product_reference_co2_fraction,float(p),settings.nominal_temperature_K,pair) for p in pgrid}
        reg_cache[float(settings.nominal_regeneration_pressure_bar)]=mixture_state(settings.product_reference_co2_fraction,settings.nominal_regeneration_pressure_bar,settings.nominal_temperature_K,pair)
        exact={k:[] for k in class_order}; nested_frozen=[]; nested_single=[]; nested_coord=[]
        for y in ys:
            ads=mixture_state(float(y),settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair); ff=front_factor(float(y),atlas_payload["front_table"])*fmult
            def rel_for(p):
                reg=reg_cache[float(p)]; return {"dco2":max(ads["q_co2"]-reg["q_co2"],0.0)*ff,"dn2":max(ads["q_n2"]-reg["q_n2"],0.0)*ff,"alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"],"front_factor":ff}
            p0=settings.nominal_regeneration_pressure_bar
            r0=_process_metrics_from_release(float(y),p0,1.0,rel_for(p0),cal,settings,contamination_multiplier=cmult,vacuum_efficiency=eta)
            if r0["meets_targets"]:
                cls="robust_frozen"
            else:
                timing_ok=any(_process_metrics_from_release(float(y),p0,float(t),rel_for(p0),cal,settings,contamination_multiplier=cmult,vacuum_efficiency=eta)["meets_targets"] for t in tgrid)
                pressure_ok=False
                if not timing_ok:
                    for p in pgrid:
                        if _process_metrics_from_release(float(y),float(p),1.0,rel_for(float(p)),cal,settings,contamination_multiplier=cmult,vacuum_efficiency=eta)["meets_targets"]:
                            pressure_ok=True; break
                if timing_ok or pressure_ok:
                    cls="single_knob_recoverable"
                else:
                    coordinated=False
                    for p in pgrid:
                        rel=rel_for(float(p))
                        for t in tgrid:
                            if _process_metrics_from_release(float(y),float(p),float(t),rel,cal,settings,contamination_multiplier=cmult,vacuum_efficiency=eta)["meets_targets"]:
                                coordinated=True; break
                        if coordinated: break
                    cls="coordinated_recoverable" if coordinated else "outside_control_envelope"
            per_y[float(y)][cls]+=1; exact[cls].append(float(y))
            if cls=="robust_frozen": nested_frozen.append(float(y))
            if cls in {"robust_frozen","single_knob_recoverable"}: nested_single.append(float(y))
            if cls in {"robust_frozen","single_knob_recoverable","coordinated_recoverable"}: nested_coord.append(float(y))
        def contiguous_bounds(vals):
            eligible=set(float(v) for v in vals)
            tmp=pd.DataFrame({"feed_co2_fraction":ys,"eligible":[float(v) in eligible for v in ys]})
            return _contiguous_interval_containing_nominal(tmp,settings.nominal_feed_co2_fraction)
        f_lo,f_hi=contiguous_bounds(nested_frozen); s_lo,s_hi=contiguous_bounds(nested_single); c_lo,c_hi=contiguous_bounds(nested_coord)
        draw_bounds.append({"draw":d,"frozen_lower_y":f_lo,"frozen_upper_y":f_hi,"single_or_less_lower_y":s_lo,"single_or_less_upper_y":s_hi,"coordinated_or_less_lower_y":c_lo,"coordinated_or_less_upper_y":c_hi})
        if (d+1)%batch_draws==0 or (d+1)==n_draw:
            atomic_pickle({"key":partial_key,"next_draw":d+1,"per_y":per_y,"draw_bounds":draw_bounds},partial_path)
        if (d+1)%max(1,n_draw//8)==0: logger.info("Feed-composition robustness uncertainty: %d/%d draws",d+1,n_draw)
    records=[]
    for y in ys:
        counts=per_y[float(y)]; total=max(sum(counts.values()),1); probs={k:counts[k]/total for k in class_order}; kmax=max(probs,key=probs.get); maxp=probs[kmax]; final=kmax if maxp>=settings.class_probability_threshold else "indeterminate"
        records.append({"feed_co2_fraction":float(y),"p_robust_frozen":probs["robust_frozen"],"p_single_knob_recoverable":probs["single_knob_recoverable"],"p_coordinated_recoverable":probs["coordinated_recoverable"],"p_outside_control_envelope":probs["outside_control_envelope"],"classification":final,"max_class_probability":maxp,"composition_support":composition_support_class(float(y),evidence_payload["summary"],settings,evidence_payload.get("composition_anchors"))})
    clsdf=pd.DataFrame(records); db=pd.DataFrame(draw_bounds); bound_rows=[]
    for col,label in [("frozen_lower_y","frozen_safe_lower"),("frozen_upper_y","frozen_safe_upper"),("single_or_less_lower_y","single_knob_lower"),("single_or_less_upper_y","single_knob_upper"),("coordinated_or_less_lower_y","coordinated_lower"),("coordinated_or_less_upper_y","coordinated_upper")]:
        vals=db[col].dropna().to_numpy(float) if len(db) else np.array([])
        med=float(np.nanmedian(vals)) if len(vals) else np.nan
        q025=float(np.nanquantile(vals,0.025)) if len(vals) else np.nan
        q975=float(np.nanquantile(vals,0.975)) if len(vals) else np.nan
        support=composition_support_class(med,evidence_payload["summary"],settings,evidence_payload.get("composition_anchors")) if np.isfinite(med) else "unresolved"
        entirely_primary=bool(np.isfinite(q025) and np.isfinite(q975) and q025>=settings.primary_evidence_feed_min-EPS and q975<=settings.primary_evidence_feed_max+EPS)
        min_valid=max(3,int(math.ceil(0.5*n_draw)))
        valid_fraction=float(len(vals)/n_draw) if n_draw else 0.0
        if len(vals) < min_valid:
            censor="insufficient_valid_draw_support"
        elif np.isfinite(med) and label.endswith("lower") and abs(med-float(np.min(ys)))<=1e-12:
            censor="left_censored_by_feed_domain_limit"
        elif np.isfinite(med) and label.endswith("upper") and abs(med-float(np.max(ys)))<=1e-12:
            censor="right_censored_by_feed_domain_limit"
        else:
            censor="interior_uncertainty_boundary" if np.isfinite(med) else "unresolved"
        bound_rows.append({"boundary":label,"median_feed_co2_fraction":med,"q025":q025,"q975":q975,"n_valid_draws":len(vals),"valid_draw_fraction":valid_fraction,"minimum_valid_draws_for_resolved_boundary":min_valid,"median_composition_support":support,"ci_entirely_inside_primary_evidence_envelope":entirely_primary,"boundary_status":censor,"boundary_resolved":censor=="interior_uncertainty_boundary"})
    boundaries=pd.DataFrame(bound_rows)
    save_table_multi(clsdf,dirs["uncertainty"] / "feed_composition_robustness_uncertainty_classification",SAVE_MODES[config.save_mode]); save_table_multi(boundaries,dirs["uncertainty"] / "feed_composition_robustness_intervention_boundaries",SAVE_MODES[config.save_mode])
    if SAVE_MODES[config.save_mode]["bootstrap_samples"]: save_table_multi(db,dirs["uncertainty"] / "feed_composition_robustness_draw_boundaries",SAVE_MODES[config.save_mode])
    pair0=calf20_pair(fit_payload)
    if pair0 is not None:
        threshold_settings=settings if config.level!="quickcheck" else dataclasses.replace(settings,purity_threshold_sensitivity=(settings.target_purity,),recovery_threshold_sensitivity=(settings.target_recovery,))
        threshold_sensitivity,threshold_windows=threshold_sensitivity_recalculate(
            ys,pair0,atlas_payload.get("calibration",{}),atlas_payload.get("front_table",pd.DataFrame()),
            threshold_settings,int(level["pressure_n"])
        )
    else:
        threshold_sensitivity,threshold_windows=pd.DataFrame(),pd.DataFrame()
    save_table_multi(threshold_sensitivity,dirs["uncertainty"] / "target_threshold_sensitivity_selected",SAVE_MODES[config.save_mode])
    save_table_multi(threshold_windows,dirs["uncertainty"] / "target_threshold_sensitivity_windows",SAVE_MODES[config.save_mode])
    return {"classification":clsdf,"boundaries":boundaries,"draw_boundaries":db,"threshold_sensitivity":threshold_sensitivity,"threshold_windows":threshold_windows}


# ---------------------------------------------------------------------------
# Validation and numerical QA
# ---------------------------------------------------------------------------
def breakthrough_validation(evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any]) -> pd.DataFrame:
    summary = evidence_payload.get("summary", pd.DataFrame())
    intrinsic = atlas_payload.get("intrinsic", pd.DataFrame())
    required = {"feed_co2_fraction", "t50_s"}
    if summary.empty or not required.issubset(summary.columns) or intrinsic.empty:
        return pd.DataFrame()
    s = summary.dropna(subset=["feed_co2_fraction", "t50_s"]).copy()
    if len(s) < 2:
        return pd.DataFrame()
    # Model t50 ~ K q_CO2(y)/y.  K is fitted only as a scale; the composition
    # dependence is therefore tested rather than fitted independently at each y.
    q=np.interp(s["feed_co2_fraction"],intrinsic["feed_co2_fraction"],intrinsic["q_co2_mol_kg"])
    x=q/np.maximum(s["feed_co2_fraction"].to_numpy(float),EPS); obs=s["t50_s"].to_numpy(float)
    K=float(np.dot(x,obs)/max(np.dot(x,x),EPS)); pred=K*x
    rows=[]
    for (_,r),pp in zip(s.iterrows(),pred):
        rows.append({"validation_type":"breakthrough_t50_global_scale","source_id":r["source_id"],"feed_co2_fraction":r["feed_co2_fraction"],"observed_t50_s":r["t50_s"],"predicted_t50_s":pp,"residual_s":pp-r["t50_s"]})
    # Leave-one-composition-out scale test.
    unique=sorted(s["feed_co2_fraction"].unique())
    for yhold in unique:
        train=s[s["feed_co2_fraction"]!=yhold]; test=s[s["feed_co2_fraction"]==yhold]
        if train.empty: continue
        qtr=np.interp(train["feed_co2_fraction"],intrinsic["feed_co2_fraction"],intrinsic["q_co2_mol_kg"]); xtr=qtr/np.maximum(train["feed_co2_fraction"].to_numpy(float),EPS); otr=train["t50_s"].to_numpy(float)
        kk=float(np.dot(xtr,otr)/max(np.dot(xtr,xtr),EPS))
        for _,r in test.iterrows():
            qt=float(np.interp(r["feed_co2_fraction"],intrinsic["feed_co2_fraction"],intrinsic["q_co2_mol_kg"])); pp=kk*qt/max(r["feed_co2_fraction"],EPS)
            rows.append({"validation_type":"leave_one_composition_out","source_id":r["source_id"],"feed_co2_fraction":r["feed_co2_fraction"],"observed_t50_s":r["t50_s"],"predicted_t50_s":pp,"residual_s":pp-r["t50_s"]})
    return pd.DataFrame(rows)


def blocked_validation(data: pd.DataFrame, fit_payload: Mapping[str,Any], level: Mapping[str,Any], seed: int) -> pd.DataFrame:
    rows=[]
    for gi,((material,gas),g) in enumerate(data.groupby(["material","adsorbate"],sort=True)):
        fs=fit_payload["fits"].get((material,gas),{}); model=fs.get("best_model")
        if not model: continue
        temps=sorted(g["temperature_K"].dropna().unique())
        if level.get("loto") and len(temps)>=3:
            for T in temps:
                tr=g[g["temperature_K"]!=T]; te=g[g["temperature_K"]==T]
                r=fit_one_model(tr,model,n_starts=max(4,min(int(level["n_starts"]),10)),seed=seed+gi*1009+int(T))
                if r.get("success"):
                    pr=model_loading(model,te["pressure_bar"].to_numpy(float),te["temperature_K"].to_numpy(float),r["theta"]); ob=te["loading_mol_kg"].to_numpy(float)
                    rmse=float(np.sqrt(np.mean((pr-ob)**2)))
                else: rmse=np.nan
                rows.append({"material":material,"adsorbate":gas,"validation_type":"leave_one_temperature_out","held_out":f"{T:g} K","n_train":len(tr),"n_test":len(te),"success":bool(r.get("success")),"rmse_mol_kg":rmse})
        if level.get("loso") and g["source_id"].nunique()>=2:
            for si,src in enumerate(list(g["source_id"].dropna().astype(str).unique())[:12]):
                tr=g[g["source_id"].astype(str)!=src]; te=g[g["source_id"].astype(str)==src]
                if len(tr)<8 or len(te)<2: continue
                r=fit_one_model(tr,model,n_starts=max(4,min(int(level["n_starts"]),10)),seed=seed+gi*2003+si*37)
                if r.get("success"):
                    pr=model_loading(model,te["pressure_bar"].to_numpy(float),te["temperature_K"].to_numpy(float),r["theta"]); ob=te["loading_mol_kg"].to_numpy(float); rmse=float(np.sqrt(np.mean((pr-ob)**2)))
                else: rmse=np.nan
                rows.append({"material":material,"adsorbate":gas,"validation_type":"leave_one_source_out","held_out":src,"n_train":len(tr),"n_test":len(te),"success":bool(r.get("success")),"rmse_mol_kg":rmse})
    return pd.DataFrame(rows)


def numerical_verification(atlas_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], settings: EngineeringSettings) -> pd.DataFrame:
    pair=calf20_pair(fit_payload)
    if pair is None: return pd.DataFrame()
    cal=atlas_payload["calibration"]; ft=atlas_payload["front_table"]; rows=[]
    for y in [settings.feed_co2_min,0.05,settings.nominal_feed_co2_fraction,0.30,settings.feed_co2_max]:
        for n in [11,21,41]:
            pressures=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,n)
            feasible=[]
            for p in pressures:
                r=process_surrogate(y,p,1.0,pair,cal,ft,settings); feasible.append((p,r["purity_estimate"],r["recovery_estimate"],r["meets_targets"]))
            ok=[x for x in feasible if x[3]]
            threshold=max(x[0] for x in ok) if ok else np.nan
            rows.append({"feed_co2_fraction":y,"pressure_grid_n":n,"largest_feasible_p_reg_bar":threshold,"n_feasible":len(ok)})
    return pd.DataFrame(rows)


def boundary_grid_convergence(atlas_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], settings: EngineeringSettings, resolutions: Sequence[int]=(9,17,33,49)) -> pd.DataFrame:
    """Recompute headline nested windows at several control-grid resolutions."""
    pair=calf20_pair(fit_payload)
    if pair is None or atlas_payload.get("intrinsic",pd.DataFrame()).empty:
        return pd.DataFrame()
    ys=atlas_payload["intrinsic"]["feed_co2_fraction"].to_numpy(float)
    cal=atlas_payload.get("calibration",{}); ft=atlas_payload.get("front_table",pd.DataFrame())
    rows=[]
    for n in resolutions:
        selected=[]
        for y in ys:
            a=_solve_adaptation_levels_grid_v130(float(y),pair,cal,ft,settings,n)
            normal=a[(a["success"]) & (a["adaptation_level"]<=2)].sort_values("adaptation_level")
            if len(normal):
                r=normal.iloc[0].to_dict(); r["success"]=True
            else:
                r={"feed_co2_fraction":float(y),"success":False,"adaptation_level":99}
            selected.append(r)
        w=summarize_intervention_windows(pd.DataFrame(selected),settings)
        for _,r in w.iterrows():
            rows.append({"control_grid_resolution":n,**r.to_dict()})
    out=pd.DataFrame(rows)
    if len(out):
        ref=out[out["control_grid_resolution"]==out["control_grid_resolution"].max()][["intervention_ceiling","lower_feed_co2_fraction","upper_feed_co2_fraction"]].set_index("intervention_ceiling")
        out["delta_lower_vs_finest"]=out.apply(lambda r:r["lower_feed_co2_fraction"]-ref.loc[r["intervention_ceiling"],"lower_feed_co2_fraction"] if r["intervention_ceiling"] in ref.index else np.nan,axis=1)
        out["delta_upper_vs_finest"]=out.apply(lambda r:r["upper_feed_co2_fraction"]-ref.loc[r["intervention_ceiling"],"upper_feed_co2_fraction"] if r["intervention_ceiling"] in ref.index else np.nan,axis=1)
    return out


def boundary_falsification_portfolio(atlas_payload: Mapping[str,Any], settings: EngineeringSettings) -> pd.DataFrame:
    """Select a small, non-optimizing set of states for future higher-fidelity boundary checks."""
    sel=atlas_payload.get("selected",pd.DataFrame()); win=atlas_payload.get("windows",pd.DataFrame())
    if sel.empty or win.empty:
        return pd.DataFrame()
    rows=[]; seen=set()
    def add(y,reason):
        if not np.isfinite(y): return
        idx=int((sel["feed_co2_fraction"]-float(y)).abs().argsort().iloc[0])
        r=sel.iloc[idx]
        key=round(float(r["feed_co2_fraction"]),8)
        if key in seen:
            return
        seen.add(key)
        rows.append({"feed_co2_fraction":float(r["feed_co2_fraction"]),"reason":reason,"composition_support":r.get("composition_support",""),"predicted_class":r.get("robustness_class",""),"adaptation_level":r.get("adaptation_level",np.nan),"adaptation_variant":r.get("adaptation_variant",""),"p_reg_bar":r.get("p_reg_bar",np.nan),"timing_scale":r.get("timing_scale",np.nan),"purity_estimate":r.get("purity_estimate",np.nan),"recovery_estimate":r.get("recovery_estimate",np.nan)})
    add(settings.nominal_feed_co2_fraction,"nominal calibration point")
    ys=np.sort(sel["feed_co2_fraction"].unique())
    for _,w in win.iterrows():
        label=w["intervention_ceiling"]
        for side,col,status_col in [("lower","lower_feed_co2_fraction","lower_boundary_status"),("upper","upper_feed_co2_fraction","upper_boundary_status")]:
            y=float(w[col]) if pd.notna(w[col]) else np.nan
            if not np.isfinite(y): continue
            j=int(np.argmin(np.abs(ys-y)))
            status=str(w.get(status_col,""))
            if "censored_by_feed_domain_limit" in status:
                add(ys[j],f"{label} {side} explored-domain limit; transition unresolved ({status})")
                continue
            add(ys[j],f"{label} {side} boundary")
            if side=="lower" and j>0: add(ys[j-1],f"just outside {label} lower boundary")
            if side=="upper" and j<len(ys)-1: add(ys[j+1],f"just outside {label} upper boundary")
    return pd.DataFrame(rows).sort_values("feed_co2_fraction").reset_index(drop=True)


def model_form_sensitivity(fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], settings: EngineeringSettings) -> pd.DataFrame:
    rows=[]; fits=fit_payload["fits"]
    fco=fits.get(("CALF-20","CO2"),{}); fn=fits.get(("CALF-20","N2"),{})
    def weights(fs):
        ms=[m for m in fs.get("retained_models",[]) if fs.get("models",{}).get(m,{}).get("success")]
        if not ms: return {}
        a=np.array([fs["models"][m].get("aicc",np.nan) for m in ms],float)
        if np.isfinite(a).any():
            amin=np.nanmin(a); w=np.exp(-0.5*np.where(np.isfinite(a),a-amin,50.0)); w=w/w.sum()
        else: w=np.full(len(ms),1/len(ms))
        return dict(zip(ms,w))
    wc,wn=weights(fco),weights(fn)
    for mc in fco.get("retained_models",[]):
        for mn in fn.get("retained_models",[]):
            pair=calf20_pair(fit_payload,mc,mn)
            if pair is None: continue
            ft=_anchor_front_factor_table(evidence_payload["summary"],pair,settings); cal=calibrate_nominal_surrogate(pair,ft,settings,evidence_payload["process_anchors"])
            for y in [0.05,0.15,0.30]:
                r=process_surrogate(y,settings.nominal_regeneration_pressure_bar,1.0,pair,cal,ft,settings)
                rows.append({"co2_model":mc,"n2_model":mn,"aicc_weight_co2":wc.get(mc,np.nan),"aicc_weight_n2":wn.get(mn,np.nan),"joint_model_weight":wc.get(mc,np.nan)*wn.get(mn,np.nan),"feed_co2_fraction":y,"purity_estimate":r["purity_estimate"],"recovery_estimate":r["recovery_estimate"],"specific_vacuum_work_kJ_mol_CO2":r["specific_vacuum_work_kJ_mol_CO2"]})
    return pd.DataFrame(rows)


def stage_07_validation(config: RunConfig, dirs: Mapping[str,Path], harmonized_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    blocked=blocked_validation(harmonized_payload["data"],fit_payload,LEVELS[config.level],config.seed)
    bt=breakthrough_validation(evidence_payload,atlas_payload)
    numerical=numerical_verification(atlas_payload,fit_payload,config.engineering)
    _resolutions={"quickcheck":(9,),"rough":(9,17),"standard":(9,17,33),"comprehensive":(9,17,33,49),"exhaustive":(9,17,33,49,61)}[config.level]
    boundary_grid=boundary_grid_convergence(atlas_payload,fit_payload,evidence_payload,config.engineering,resolutions=_resolutions)
    falsification=boundary_falsification_portfolio(atlas_payload,config.engineering)
    model_sens=model_form_sensitivity(fit_payload,evidence_payload,config.engineering)
    for name,df in [("blocked_validation",blocked),("breakthrough_validation",bt),("numerical_grid_verification",numerical),("boundary_grid_convergence",boundary_grid),("boundary_falsification_portfolio",falsification),("model_form_process_sensitivity",model_sens)]:
        save_table_multi(df,dirs["validation"] / name,SAVE_MODES[config.save_mode])
    return {"blocked":blocked,"breakthrough":bt,"numerical":numerical,"boundary_grid":boundary_grid,"falsification_portfolio":falsification,"model_sensitivity":model_sens}


# ===========================================================================
# Scientific-strengthening layer
# ===========================================================================
# These functions preserve the restart/checkpoint structure.
# They add: (i) calibration-conditioned uncertainty, (ii) continuous refinement
# of minimum-control moves, (iii) source/model sensitivity, (iv) independent
# pilot-data validation, (v) CSS-result ingestion hooks, and (vi) mechanism/
# control-authority diagnostics.  Optional data are NEVER merged into the
# nominal fitting dataset.


def resolve_engineering_settings_from_anchors(settings: EngineeringSettings, process_anchors: pd.DataFrame, logger: Optional[logging.Logger]=None) -> EngineeringSettings:
    """Use reported nominal cycle settings when supplied at the nominal composition.

    Purity/recovery are anchor-aware; this routine also adopts
    reported regeneration pressure and adsorption time if a parser-ready process
    anchor supplies them.  Blank fields leave the transparent engineering
    defaults unchanged.  This makes future literature completion data-active
    without requiring any code edit.
    """
    if process_anchors is None or process_anchors.empty:
        return settings
    g=process_anchors.copy()
    if "feed_co2_fraction" in g:
        g=g[np.abs(pd.to_numeric(g["feed_co2_fraction"],errors="coerce")-settings.nominal_feed_co2_fraction)<=0.01]
    if g.empty:
        return settings
    updates={}
    if "regeneration_pressure_bar" in g:
        x=pd.to_numeric(g["regeneration_pressure_bar"],errors="coerce").dropna()
        if len(x) and 0<float(np.nanmedian(x))<settings.adsorption_pressure_bar:
            updates["nominal_regeneration_pressure_bar"]=float(np.nanmedian(x))
    if "adsorption_time_s" in g:
        x=pd.to_numeric(g["adsorption_time_s"],errors="coerce").dropna()
        if len(x) and float(np.nanmedian(x))>0:
            updates["nominal_adsorption_time_s"]=float(np.nanmedian(x))
    if not updates:
        return settings
    # Never permit the declared floor to sit above a newly recovered nominal pressure.
    if "nominal_regeneration_pressure_bar" in updates and settings.regeneration_pressure_floor_bar>updates["nominal_regeneration_pressure_bar"]:
        updates["regeneration_pressure_floor_bar"]=max(PRESSURE_FLOOR_BAR,0.2*updates["nominal_regeneration_pressure_bar"])
    out=dataclasses.replace(settings,**updates)
    if logger:
        logger.info("Nominal operating settings updated from supplied process anchor: %s",updates)
    return out


def calibrate_nominal_surrogate_conditioned(pair, front_table: pd.DataFrame, settings: EngineeringSettings, process_anchors: pd.DataFrame, *, front_multiplier: float=1.0, contamination_multiplier: float=1.0) -> Dict[str,float]:
    """Calibrate each uncertainty realization back to the nominal experiment.

    Thermodynamic and engineering
    nuisance draws may change off-design behavior, but each realization is
    conditioned to reproduce the supplied nominal purity/recovery observation.
    Thus nominal uncertainty is not created merely by forgetting calibration.
    """
    y0=settings.nominal_feed_co2_fraction; p0=settings.nominal_regeneration_pressure_bar
    purity=settings.nominal_purity_anchor; recovery=settings.nominal_recovery_anchor
    if process_anchors is not None and len(process_anchors):
        g=process_anchors[np.abs(pd.to_numeric(process_anchors["feed_co2_fraction"],errors="coerce")-y0)<=0.01]
        if len(g):
            pv=pd.to_numeric(g.get("purity",pd.Series(dtype=float)),errors="coerce").dropna()
            rv=pd.to_numeric(g.get("recovery",pd.Series(dtype=float)),errors="coerce").dropna()
            if len(pv): purity=float(np.nanmedian(pv))
            if len(rv): recovery=float(np.nanmedian(rv))
    ff=front_factor(y0,front_table)*float(front_multiplier)
    rel=release_state(y0,p0,pair,settings,ff)
    F0=rel["dco2"]/max(recovery*y0,EPS)
    target_n2=rel["dco2"]*(1.0/max(purity,EPS)-1.0)
    denominator=max(rel["dn2"]*max(float(contamination_multiplier),EPS),EPS)
    contam=target_n2/denominator if rel["dn2"]>EPS else 0.0
    contam=float(np.clip(contam,0.0,100.0))
    return {"nominal_feed_mol_kg_cycle":float(F0),"light_gas_carryover_scale":contam,"purity_anchor":purity,"recovery_anchor":recovery,"nominal_dco2_mol_kg":rel["dco2"],"nominal_dn2_mol_kg":rel["dn2"],"front_factor_nominal":ff,"calibration_conditioned":True}


# Retain the grid solver as a convergence reference.
_solve_adaptation_levels_grid_v130 = solve_adaptation_levels


def _constraint_margins(row: Mapping[str,Any], settings: EngineeringSettings) -> Tuple[float,float]:
    return float(row["purity_estimate"]-settings.target_purity), float(row["recovery_estimate"]-settings.target_recovery)


def _refine_one_control(y: float, variant: str, pair, calibration, front_table, settings: EngineeringSettings, seed_row: Mapping[str,Any], *, front_multiplier=1.0, contamination_multiplier=1.0, vacuum_efficiency=None) -> Optional[Dict[str,Any]]:
    """Continuously refine one actuator around a feasible grid seed."""
    if variant=="pressure_only":
        lo,hi=math.log(settings.regeneration_pressure_floor_bar),math.log(settings.nominal_regeneration_pressure_bar)
        def make(x): return process_surrogate(y,math.exp(float(x)),1.0,pair,calibration,front_table,settings,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
        x0=math.log(max(float(seed_row.get("p_reg_bar",settings.nominal_regeneration_pressure_bar)),settings.regeneration_pressure_floor_bar))
    elif variant=="timing_only":
        lo,hi=math.log(settings.timing_min_scale),math.log(settings.timing_max_scale)
        def make(x): return process_surrogate(y,settings.nominal_regeneration_pressure_bar,math.exp(float(x)),pair,calibration,front_table,settings,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
        x0=math.log(max(float(seed_row.get("timing_scale",1.0)),EPS))
    else:
        return None
    def objective(xarr):
        x=float(np.atleast_1d(xarr)[0]); r=make(x); pm,rm=_constraint_margins(r,settings)
        move=_control_move_components(r,settings)["control_move_linf"]
        penalty=2.0e4*(max(-pm,0.0)**2+max(-rm,0.0)**2)
        return move+penalty
    starts=[x0,0.5*(lo+hi),lo,hi]
    candidates=[]
    for st in starts:
        try:
            res=minimize(lambda z: objective(z),x0=np.array([np.clip(st,lo,hi)]),method="SLSQP",bounds=[(lo,hi)],options={"maxiter":200,"ftol":1e-12})
            r=make(float(res.x[0]));
            if r["meets_targets"]: candidates.append(_decorate_control_move(r,settings))
        except Exception:
            pass
    if not candidates: return None
    candidates.sort(key=lambda r:(r["control_move_linf"],r["control_move_l1"],r["specific_vacuum_work_kJ_mol_CO2"]))
    return candidates[0]


def _refine_two_controls(y: float, pair, calibration, front_table, settings: EngineeringSettings, seed_row: Mapping[str,Any], *, front_multiplier=1.0, contamination_multiplier=1.0, vacuum_efficiency=None) -> Optional[Dict[str,Any]]:
    lp0=math.log(settings.nominal_regeneration_pressure_bar)
    bounds=[(math.log(settings.regeneration_pressure_floor_bar),lp0),(math.log(settings.timing_min_scale),math.log(settings.timing_max_scale))]
    def make(z):
        return process_surrogate(y,math.exp(float(z[0])),math.exp(float(z[1])),pair,calibration,front_table,settings,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
    def obj(z):
        r=make(z); mv=_control_move_components(r,settings)
        # smooth approximation to L-infinity, retaining L1 as a tiny tie breaker
        a,b=mv["pressure_move_normalized"],mv["timing_move_normalized"]
        return (a**8+b**8+1e-12)**(1/8)+1e-4*mv["control_move_l1"]
    cons=[{"type":"ineq","fun":lambda z: make(z)["purity_estimate"]-settings.target_purity},{"type":"ineq","fun":lambda z: make(z)["recovery_estimate"]-settings.target_recovery}]
    seeds=[np.array([math.log(max(float(seed_row.get("p_reg_bar",settings.nominal_regeneration_pressure_bar)),settings.regeneration_pressure_floor_bar)),math.log(max(float(seed_row.get("timing_scale",1.0)),EPS))]),np.array([lp0,0.0]),np.array([0.5*(bounds[0][0]+bounds[0][1]),0.0])]
    cand=[]
    for x0 in seeds:
        try:
            res=minimize(obj,x0=x0,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":350,"ftol":1e-11})
            r=make(res.x)
            if r["meets_targets"]: cand.append(_decorate_control_move(r,settings))
        except Exception:
            pass
    if not cand: return None
    cand.sort(key=lambda r:(r["control_move_linf"],r["control_move_l1"],r["specific_vacuum_work_kJ_mol_CO2"]))
    return cand[0]


def _select_one_actuator_policy(candidates: List[Dict[str,Any]], y: float, settings: EngineeringSettings, relative_tie_tolerance: float=0.05) -> Optional[Dict[str,Any]]:
    """Select a physically interpretable one-actuator policy among near-equivalent states.

    The intervention *class* is determined before this selector and is therefore
    unaffected.  This function only resolves which one-actuator solution to show
    when pressure-only and timing-only states have almost the same normalized
    control effort.  Within a 5% effort tie, pressure is preferred on the lean
    side (where purity is the dominant constraint) and timing on the rich side
    (where recovery dominates).  Outside that narrow tie set the mathematically
    closest-to-nominal solution remains selected.
    """
    if not candidates:
        return None
    cand=[dict(c) for c in candidates]
    for c in cand:
        mv=_control_move_components(c,settings)
        c.update(mv)
        c["control_effort_selector"] = float(mv["control_move_linf"] + 1.0e-3*mv["control_move_l1"])
    cand.sort(key=lambda r:(r["control_effort_selector"],r["specific_vacuum_work_kJ_mol_CO2"],-r["productivity_mol_kg_h"]))
    best=cand[0]; b=float(best["control_effort_selector"]); tol=max(1.0e-6,relative_tie_tolerance*max(b,1.0e-6))
    near=[c for c in cand if float(c["control_effort_selector"]) <= b+tol]
    preference = "pressure_only" if y < settings.nominal_feed_co2_fraction-1.0e-10 else ("timing_only" if y > settings.nominal_feed_co2_fraction+1.0e-10 else "")
    chosen=best
    if preference:
        pref=[c for c in near if str(c.get("adaptation_variant",""))==preference]
        if pref:
            pref.sort(key=lambda r:(r["control_effort_selector"],r["specific_vacuum_work_kJ_mol_CO2"],-r["productivity_mol_kg_h"]))
            chosen=pref[0]
    chosen=dict(chosen)
    chosen["policy_selection_rule"]="minimum normalized control effort; mechanism-side preference only within 5% near-tie"
    chosen["policy_preferred_actuator"] = preference or "none"
    chosen["policy_near_tie_count"] = int(len(near))
    chosen["policy_effort_best"] = b
    chosen["policy_effort_selected"] = float(chosen["control_effort_selector"])
    return chosen


def solve_adaptation_levels(y: float, pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int, *, front_multiplier: float=1.0, contamination_multiplier: float=1.0, vacuum_efficiency: Optional[float]=None) -> pd.DataFrame:
    """Grid-backed continuous minimum-control solver.

    A deterministic grid first proves/locates feasibility and remains the
    convergence fail-safe. Feasible one- and two-control states are then refined
    continuously. This removes plotting stair-steps without sacrificing the
    reproducible grid audit used in the SI.
    """
    base=_solve_adaptation_levels_grid_v130(y,pair,calibration,front_table,settings,pressure_n,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
    if base.empty: return base
    out=[]
    for _,row in base.iterrows():
        r=row.to_dict(); lev=int(r.get("adaptation_level",-1)); variant=str(r.get("adaptation_variant",""))
        refined=None
        if bool(r.get("success",False)) and lev==1 and variant in {"pressure_only","timing_only"}:
            one_candidates=[]
            for vv in ("pressure_only","timing_only"):
                rr=_refine_one_control(y,vv,pair,calibration,front_table,settings,r,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
                if rr is not None:
                    rr=dict(rr); rr["adaptation_variant"]=vv; one_candidates.append(rr)
            refined=_select_one_actuator_policy(one_candidates,y,settings)
        elif bool(r.get("success",False)) and lev==2 and variant=="pressure_plus_timing":
            refined=_refine_two_controls(y,pair,calibration,front_table,settings,r,front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,vacuum_efficiency=vacuum_efficiency)
        if refined is not None:
            r.update(refined); r["success"]=True; r["continuous_refined"]=True
        else:
            r["continuous_refined"]=False
        out.append(r)
    return pd.DataFrame(out)



def threshold_sensitivity_recalculate(ys: Sequence[float], pair, calibration: Mapping[str,float], front_table: pd.DataFrame, settings: EngineeringSettings, pressure_n: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Fast target sensitivity using one shared physical state atlas.

    Purity/recovery target changes do not change equilibrium states.  We therefore
    compute all candidate pressure/timing states once per composition and then
    reclassify that common atlas for every target pair.  This is scientifically
    equivalent to repeated grid searches and much cheaper on a single CPU.
    """
    ys=np.asarray(list(ys),float)
    purities=sorted(set(tuple(settings.purity_threshold_sensitivity)+(settings.target_purity,)))
    recoveries=sorted(set(tuple(settings.recovery_threshold_sensitivity)+(settings.target_recovery,)))
    p0=settings.nominal_regeneration_pressure_bar
    npg=max(9,int(math.sqrt(max(pressure_n,9))*2)); nt=max(9,int(math.sqrt(max(pressure_n,9))*2))
    pgrid=np.unique(np.r_[np.geomspace(settings.regeneration_pressure_floor_bar,p0,npg),p0])
    tgrid=np.unique(np.r_[np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(15,pressure_n)),1.0])
    reg_cache={float(p):mixture_state(settings.product_reference_co2_fraction,float(p),settings.nominal_temperature_K,pair) for p in pgrid}
    candidates_by_y={}
    for y in ys:
        ads=mixture_state(float(y),settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair); ff=front_factor(float(y),front_table)
        def rel_for(p):
            reg=reg_cache[float(p)]
            return {"dco2":max(ads["q_co2"]-reg["q_co2"],0.0)*ff,"dn2":max(ads["q_n2"]-reg["q_n2"],0.0)*ff,"alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"],"front_factor":ff}
        fixed=_decorate_control_move(_process_metrics_from_release(float(y),p0,1.0,rel_for(p0),calibration,settings),settings); fixed["adaptation_level"]=0; fixed["adaptation_variant"]="frozen"
        one=[]
        for t in tgrid:
            r=_decorate_control_move(_process_metrics_from_release(float(y),p0,float(t),rel_for(p0),calibration,settings),settings); r["adaptation_level"]=1; r["adaptation_variant"]="timing_only"; one.append(r)
        for p in pgrid:
            r=_decorate_control_move(_process_metrics_from_release(float(y),float(p),1.0,rel_for(float(p)),calibration,settings),settings); r["adaptation_level"]=1; r["adaptation_variant"]="pressure_only"; one.append(r)
        joint=[]
        for p in pgrid:
            rel=rel_for(float(p))
            for t in np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt):
                r=_decorate_control_move(_process_metrics_from_release(float(y),float(p),float(t),rel,calibration,settings),settings); r["adaptation_level"]=2; r["adaptation_variant"]="pressure_plus_timing"; joint.append(r)
        candidates_by_y[float(y)]=(fixed,one,joint)
    selected_rows=[]; window_rows=[]
    def best(rows,pt,rt):
        ok=[r for r in rows if r["purity_estimate"]>=pt and r["recovery_estimate"]>=rt]
        if not ok: return None
        ok.sort(key=lambda r:(r["control_move_linf"],r["control_move_l1"],r["specific_vacuum_work_kJ_mol_CO2"],-r["productivity_mol_kg_h"]))
        return dict(ok[0])
    for pt in purities:
        for rt in recoveries:
            per=[]
            for y in ys:
                fixed,one,joint=candidates_by_y[float(y)]
                if fixed["purity_estimate"]>=pt and fixed["recovery_estimate"]>=rt: row=dict(fixed); success=True
                else:
                    row=best(one,pt,rt)
                    if row is None: row=best(joint,pt,rt)
                    success=row is not None
                    if row is None: row=dict(fixed); row["adaptation_level"]=3; row["adaptation_variant"]="none_feasible"
                row.update({"feed_co2_fraction":float(y),"threshold_purity":float(pt),"threshold_recovery":float(rt),"threshold_success":bool(success)})
                per.append(row); selected_rows.append(row)
            sdf=pd.DataFrame(per)
            for lev,label in {0:"frozen_safe",1:"single_knob_or_less",2:"coordinated_two_knob_or_less"}.items():
                tmp=sdf[["feed_co2_fraction","threshold_success","adaptation_level"]].copy(); tmp["eligible"]=tmp["threshold_success"].astype(bool)&(pd.to_numeric(tmp["adaptation_level"],errors="coerce")<=lev)
                lo,hi=_contiguous_interval_containing_nominal(tmp[["feed_co2_fraction","eligible"]],settings.nominal_feed_co2_fraction)
                window_rows.append({"threshold_purity":float(pt),"threshold_recovery":float(rt),"intervention_ceiling":label,"max_adaptation_level":lev,"lower_feed_co2_fraction":lo,"upper_feed_co2_fraction":hi,"window_width_fraction":hi-lo if np.isfinite(lo) and np.isfinite(hi) else np.nan})
    return pd.DataFrame(selected_rows),pd.DataFrame(window_rows)


def mechanism_and_control_authority(atlas_payload: Mapping[str,Any], settings: EngineeringSettings) -> pd.DataFrame:
    fixed=atlas_payload.get("fixed",pd.DataFrame()).copy(); selected=atlas_payload.get("selected",pd.DataFrame()).copy()
    if fixed.empty: return pd.DataFrame()
    cols=["feed_co2_fraction","p_reg_bar","timing_scale","purity_estimate","recovery_estimate","working_capacity_co2_mol_kg","working_capacity_n2_mol_kg","alpha_ads"]
    d=fixed[[c for c in cols if c in fixed.columns]].copy()
    if not selected.empty:
        ss=selected[[c for c in ["feed_co2_fraction","adaptation_level","adaptation_name","adaptation_variant","success","p_reg_bar","timing_scale","control_move_linf"] if c in selected.columns]].copy()
        ss=ss.rename(columns={"p_reg_bar":"selected_p_reg_bar","timing_scale":"selected_timing_scale"})
        d=d.merge(ss,on="feed_co2_fraction",how="left")
    y=np.clip(d["feed_co2_fraction"].to_numpy(float),1e-8,1-1e-8)
    d["feed_co2_partial_pressure_bar"]=y*settings.adsorption_pressure_bar
    d["nominal_product_regen_co2_partial_pressure_bar"]=settings.product_reference_co2_fraction*settings.nominal_regeneration_pressure_bar
    d["thermodynamic_pressure_ratio"]=d["feed_co2_partial_pressure_bar"]/max(settings.product_reference_co2_fraction*settings.nominal_regeneration_pressure_bar,EPS)
    d["log_thermodynamic_driving_ratio"]=np.log(np.maximum(d["thermodynamic_pressure_ratio"],EPS))
    d["purity_margin"]=d["purity_estimate"]-settings.target_purity
    d["recovery_margin"]=d["recovery_estimate"]-settings.target_recovery
    d["allowed_n2_to_co2_product_ratio"]=1.0/settings.target_purity-1.0
    d["actual_n2_to_co2_product_ratio"]=(1.0-np.clip(d["purity_estimate"],EPS,1.0))/np.clip(d["purity_estimate"],EPS,1.0)
    def active(r):
        pm,rm=float(r["purity_margin"]),float(r["recovery_margin"])
        if pm>=0 and rm>=0: return "both_targets_met"
        if pm<0 and rm>=0: return "purity_limited"
        if rm<0 and pm>=0: return "recovery_limited"
        return "jointly_limited"
    d["active_constraint"]=d.apply(active,axis=1)
    return d


def _optional_data_root(config: RunConfig) -> Path:
    # Standard P10 data package layout: data_root/input_folder plus sibling
    # external_validation, external_equilibrium_sensitivity, css_validation.
    return Path(config.input_folder).resolve().parent


def _read_csvs_under(folder: Path, include_pattern: str="*.csv") -> pd.DataFrame:
    parts=[]
    if not folder.exists(): return pd.DataFrame()
    for f in sorted(folder.rglob(include_pattern)):
        if "DO_NOT_COPY_TO_INPUT" in f.name.upper(): continue
        try:
            d=pd.read_csv(f); d["__source_file__"]=str(f); parts.append(d)
        except Exception:
            continue
    return pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def audit_optional_literature_data(config: RunConfig, dirs: Mapping[str,Path], logger: Optional[logging.Logger]=None) -> Dict[str,Any]:
    """Audit the matched literature-data package without converting optional gaps into fake data.

    The checks are strict when the matched package identifies itself through
    DATA_PACKAGE_VERSION.json.  CSS validation remains optional by design.
    """
    root=_optional_data_root(config); rows=[]
    def add(check, observed, expected, status, detail=""):
        rows.append({"check":check,"observed":str(observed),"expected":str(expected),"status":status,"detail":detail})

    meta={}
    mpath=root/"DATA_PACKAGE_VERSION.json"
    if mpath.exists():
        try: meta=json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc: add("data_package_version_json","unreadable","valid JSON","FAIL",repr(exc))
    matched=str(meta.get("matched_pipeline_version",""))
    if meta:
        add("matched_pipeline_version", matched, MATCHED_DATA_VERSION,
            "PASS" if matched == MATCHED_DATA_VERSION else "FAIL",
            "Pipeline and packaged-data metadata must be compatible.")
    else:
        add("data_package_version_json","not supplied","optional for legacy packages","INFO","Legacy package mode; scientific parser remains usable.")

    # Provenance registries are expected in the matched release.
    for rel in ["provenance/catalog/source_registry.csv","provenance/catalog/numeric_fact_registry.csv","provenance/catalog/data_gap_register.csv"]:
        p=root/rel; add(rel,"present" if p.exists() else "missing","present in the packaged data","PASS" if p.exists() else ("FAIL" if meta else "WARN"))
    cross_files=sorted((root/"audit").glob("source_id_crosswalk_audit_v*.csv")) if (root/"audit").exists() else []
    cross=_safe_read_csv(cross_files[-1]) if cross_files else pd.DataFrame()
    if len(cross) and "status" in cross:
        nbad=int((cross["status"].astype(str)!="PASS").sum()); add("source_id_provenance_crosswalk",nbad,0,"PASS" if nbad==0 else "FAIL","All active parser/validation source IDs should resolve to provenance registry entries.")

    # External pilot campaign: only enforce campaign-specific invariants if rows are supplied.
    pilot=_read_csvs_under(root/"external_validation")
    req={"feed_co2_fraction","purity_fraction","recovery_fraction"}
    valid=pilot.dropna(subset=list(req)).copy() if (not pilot.empty and req.issubset(pilot.columns)) else pd.DataFrame()
    if len(valid):
        for c in req: valid[c]=pd.to_numeric(valid[c],errors="coerce")
        valid=valid.dropna(subset=list(req))
        add("external_pilot_numeric_rows",len(valid),">=1","PASS")
        add("external_pilot_fraction_bounds",int(((valid[list(req)]<0)|(valid[list(req)]>1)).to_numpy().sum()),0,"PASS" if (((valid[list(req)]>=0)&(valid[list(req)]<=1)).all().all()) else "FAIL")
        if "publication_doi" in valid.columns:
            dois=set(valid["publication_doi"].dropna().astype(str)); add("external_pilot_peer_reviewed_doi",";".join(sorted(dois)),"10.1016/j.cej.2026.174238","PASS" if "10.1016/j.cej.2026.174238" in dois else "WARN")
        # Campaign invariants for the audited Wilkins open SI table.
        is_wilkins=("source_id" in valid.columns and valid["source_id"].astype(str).str.contains("WILKINS",case=False,na=False).any())
        if is_wilkins:
            add("wilkins_pilot_total_rows",len(valid),75,"PASS" if len(valid)==75 else "FAIL")
            if "contactor_type" in valid:
                lam=int(valid["contactor_type"].astype(str).str.lower().eq("laminate").sum()); pac=int(valid["contactor_type"].astype(str).str.lower().eq("packed").sum())
                add("wilkins_laminate_rows",lam,61,"PASS" if lam==61 else "FAIL"); add("wilkins_packed_rows",pac,14,"PASS" if pac==14 else "FAIL")
            feeds=set(np.round(pd.to_numeric(valid["feed_co2_fraction"],errors="coerce").dropna().unique(),8)); add("wilkins_feed_fraction_set",sorted(feeds),[0.08,0.12,0.16],"PASS" if feeds=={0.08,0.12,0.16} else "FAIL")
            if "source_table" in valid:
                got=valid["source_table"].astype(str).value_counts().sort_index().to_dict(); exp={"S1":6,"S2":7,"S3-1":20,"S3-2":16,"S4":12,"S5":14}; add("wilkins_source_table_counts",got,exp,"PASS" if got==exp else "FAIL")
            steps=[c for c in ["adsorption_time_s","blowdown_time_s","evacuation_time_s","feed_pressurization_time_s","dead_space_time_s"] if c in valid]
            if len(steps)==5 and "cycle_time_s" in valid:
                cyc=pd.to_numeric(valid["cycle_time_s"],errors="coerce"); step=valid[steps].apply(pd.to_numeric,errors="coerce").sum(axis=1); delta=float((cyc-step).abs().max()); add("wilkins_cycle_time_step_sum_max_abs_delta_s",delta,0.0,"PASS" if delta<=1e-9 else "FAIL")
    else:
        add("external_pilot_numeric_rows",0,"optional but strongly recommended","WARN")

    ext=_read_csvs_under(root/"external_equilibrium_sensitivity")
    add("external_equilibrium_sensitivity_rows",len(ext),">=1 recommended","PASS" if len(ext) else "WARN")
    css_parts=[]
    for fd in [root/"css_validation",root/"higher_fidelity_validation"]:
        z=_read_csvs_under(fd); css_parts.append(len(z))
    add("high_fidelity_css_rows",sum(css_parts),"0 is permitted until new CSS calculations are supplied","INFO" if sum(css_parts)==0 else "PASS","No CSS rows are fabricated by this package.")

    audit=pd.DataFrame(rows); fail=int((audit["status"]=="FAIL").sum()) if len(audit) else 0
    summary={"status":"PASS" if fail==0 else "FAIL","n_checks":len(audit),"n_fail":fail,"n_warn":int((audit["status"]=="WARN").sum()) if len(audit) else 0,"data_root":str(root),"matched_data_version":meta.get("version","")}
    save_table_multi(audit,dirs["audit"]/"literature_data_audit",SAVE_MODES[config.save_mode]); atomic_json(summary,dirs["audit"]/"literature_data_audit_summary.json")
    if logger: logger.info("Literature-data audit %s: %d checks, %d failures, %d warnings",summary["status"],summary["n_checks"],summary["n_fail"],summary["n_warn"])
    return {"audit":audit,"summary":summary}


def external_pilot_validation(config: RunConfig, settings: EngineeringSettings, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    root=_optional_data_root(config)/"external_validation"
    d=_read_csvs_under(root)
    required={"feed_co2_fraction","purity_fraction","recovery_fraction"}
    if d.empty or not required.issubset(d.columns):
        return {"raw":pd.DataFrame(),"summary":pd.DataFrame(),"association":pd.DataFrame()}
    for c in ["feed_co2_fraction","purity_fraction","recovery_fraction","adsorption_time_s","evacuation_pressure_kPa_abs","adsorption_pressure_kPa_abs","adsorption_temperature_C","productivity_tpd_m3"]:
        if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["feed_co2_fraction","purity_fraction","recovery_fraction"]).copy()
    d["meets_95_88"]=(d["purity_fraction"]>=0.95)&(d["recovery_fraction"]>=0.88)
    d["meets_95_90"]=(d["purity_fraction"]>=0.95)&(d["recovery_fraction"]>=0.90)
    group_cols=["feed_co2_fraction"]+(["contactor_type"] if "contactor_type" in d else [])
    summary=d.groupby(group_cols,dropna=False).agg(n=("recovery_fraction","size"),purity_median=("purity_fraction","median"),purity_min=("purity_fraction","min"),purity_max=("purity_fraction","max"),recovery_median=("recovery_fraction","median"),recovery_min=("recovery_fraction","min"),recovery_max=("recovery_fraction","max"),fraction_meeting_95_88=("meets_95_88","mean"),fraction_meeting_95_90=("meets_95_90","mean")).reset_index()
    # Non-causal adjusted association: standardize available operating covariates
    # and report the feed-composition coefficient with bootstrap uncertainty.
    assoc=[]
    numeric_cov=[c for c in ["feed_co2_fraction","adsorption_time_s","evacuation_pressure_kPa_abs","adsorption_pressure_kPa_abs","adsorption_temperature_C"] if c in d.columns and pd.to_numeric(d[c],errors="coerce").notna().sum()>=8]
    work=d.dropna(subset=numeric_cov+["purity_fraction","recovery_fraction"]).copy()
    if len(work)>=12 and "feed_co2_fraction" in numeric_cov:
        Xnum=work[numeric_cov].to_numpy(float); mu=Xnum.mean(0); sd=Xnum.std(0); sd[sd<EPS]=1.0; Xn=(Xnum-mu)/sd
        Xparts=[np.ones((len(work),1)),Xn]; names=["intercept"]+numeric_cov
        if "contactor_type" in work.columns and work["contactor_type"].nunique()>1:
            dum=pd.get_dummies(work["contactor_type"].astype(str),drop_first=True,dtype=float); Xparts.append(dum.to_numpy(float)); names+=list(dum.columns)
        X=np.column_stack(Xparts)
        rng=np.random.default_rng(config.seed+7711)
        for outcome in ["purity_fraction","recovery_fraction"]:
            yy=work[outcome].to_numpy(float); ymu=yy.mean(); ysd=max(yy.std(),EPS); yz=(yy-ymu)/ysd
            beta=np.linalg.lstsq(X,yz,rcond=None)[0]; idx=names.index("feed_co2_fraction")
            boots=[]
            for _ in range(400):
                ii=rng.integers(0,len(work),len(work))
                try: boots.append(float(np.linalg.lstsq(X[ii],yz[ii],rcond=None)[0][idx]))
                except Exception: pass
            assoc.append({"outcome":outcome,"standardized_feed_composition_coefficient":float(beta[idx]),"bootstrap_q025":float(np.quantile(boots,.025)) if boots else np.nan,"bootstrap_q975":float(np.quantile(boots,.975)) if boots else np.nan,"n":len(work),"interpretation":"descriptive adjusted association; not causal and not a matched-cycle validation","covariates":";".join(names[1:])})
    if logger and len(d): logger.info("Independent pilot validation data detected: %d rows across feed compositions %s",len(d),sorted(d["feed_co2_fraction"].dropna().unique().tolist()))
    return {"raw":d,"summary":summary,"association":pd.DataFrame(assoc)}


def _fit_external_equilibrium_pair(df: pd.DataFrame, level: Mapping[str,Any], seed: int):
    if df.empty: return None, pd.DataFrame()
    fits={}; rows=[]
    for gas in ["CO2","N2"]:
        g=df[df["adsorbate"].astype(str).map(normalize_gas)==gas].copy()
        for c in ["temperature_K","pressure_bar","loading_mol_kg"]: g[c]=pd.to_numeric(g[c],errors="coerce")
        g=g.dropna(subset=["temperature_K","pressure_bar","loading_mol_kg"])
        if g.empty: return None,pd.DataFrame()
        fr=fit_group_models(g,level["models"],max(4,int(level["n_starts"]//2)),seed+(1 if gas=="CO2" else 2)*1709)
        fits[gas]=fr
        for m,r in fr["models"].items():
            if r.get("success"): rows.append({"gas":gas,"model":m,"aicc":r.get("aicc"),"rmse":r.get("rmse"),"r2":r.get("r2"),"retained":m in fr.get("retained_models",[])})
    pair=[]
    for gas in ["CO2","N2"]:
        m=fits[gas].get("best_model"); r=fits[gas].get("models",{}).get(m,{})
        g=df[df["adsorbate"].astype(str).map(normalize_gas)==gas]
        dom={"p_min":float(pd.to_numeric(g["pressure_bar"],errors="coerce").min()),"p_max":float(pd.to_numeric(g["pressure_bar"],errors="coerce").max()),"T_min":float(pd.to_numeric(g["temperature_K"],errors="coerce").min()),"T_max":float(pd.to_numeric(g["temperature_K"],errors="coerce").max())}
        if not m or not r.get("success"): return None,pd.DataFrame(rows)
        pair.append((m,np.asarray(r["theta"],float),dom))
    return tuple(pair),pd.DataFrame(rows)


def external_equilibrium_source_sensitivity(config: RunConfig, evidence_payload: Mapping[str,Any], settings: EngineeringSettings, fit_payload: Optional[Mapping[str,Any]]=None, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    folder=_optional_data_root(config)/"external_equilibrium_sensitivity"
    files=[f for f in sorted(folder.glob("*.csv")) if "scalar" not in f.name.lower() and "stress_policy" not in f.name.lower()]
    rows=[]; fitrows=[]
    for j,f in enumerate(files):
        try: d=pd.read_csv(f)
        except Exception: continue
        need={"adsorbate","temperature_K","pressure_bar","loading_mol_kg"}
        if not need.issubset(d.columns): continue
        pair,fr=_fit_external_equilibrium_pair(d,LEVELS[config.level],config.seed+8800+j*101)
        if len(fr): fr["source_file"]=f.name; fitrows.append(fr)
        if pair is None: continue
        ft=_anchor_front_factor_table(evidence_payload.get("summary",pd.DataFrame()),pair,settings)
        cal=calibrate_nominal_surrogate(pair,ft,settings,evidence_payload.get("process_anchors",pd.DataFrame()))
        ys=np.linspace(settings.feed_co2_min,settings.feed_co2_max,LEVELS[config.level]["composition_n"])
        selected=[]
        for y in ys:
            ad=_solve_adaptation_levels_grid_v130(float(y),pair,cal,ft,settings,LEVELS[config.level]["pressure_n"])
            ok=ad[ad["success"]==True]
            if len(ok): selected.append(ok.sort_values("adaptation_level").iloc[0].to_dict() | {"feed_co2_fraction":float(y)})
            else: selected.append({"feed_co2_fraction":float(y),"adaptation_level":3,"success":False})
        sdf=pd.DataFrame(selected)
        for level,label in [(0,"frozen"),(1,"single_or_less"),(2,"coordinated_or_less")]:
            tmp=pd.DataFrame({"feed_co2_fraction":ys,"eligible":pd.to_numeric(sdf["adaptation_level"],errors="coerce")<=level})
            lo,hi=_contiguous_interval_containing_nominal(tmp,settings.nominal_feed_co2_fraction)
            rows.append({"source_file":f.name,"equilibrium_source_role":"independent published parameterization","intervention_ceiling":label,"lower_feed_co2_fraction":lo,"upper_feed_co2_fraction":hi,"upper_boundary_censored":bool(np.isfinite(hi) and abs(hi-settings.feed_co2_max)<1e-12)})
    # independent scalar crosschecks are kept separate from fitting
    scalars=[]
    for f in sorted(folder.glob("*scalar*.csv")):
        try:
            q=pd.read_csv(f); q["source_file"]=f.name
            if fit_payload is not None and len(q):
                preds=[]
                for _,rr in q.iterrows():
                    gas=normalize_gas(rr.get("adsorbate","")); _fits=fit_payload.get("fits",fit_payload) if isinstance(fit_payload,Mapping) else fit_payload; desc=best_fit_descriptor(_fits,"CALF-20",gas)
                    T=pd.to_numeric(pd.Series([rr.get("temperature_K")]),errors="coerce").iloc[0]; P=pd.to_numeric(pd.Series([rr.get("pressure_bar")]),errors="coerce").iloc[0]
                    pred=np.nan
                    if desc is not None and np.isfinite(T) and np.isfinite(P): pred=float(model_loading(desc[0],np.array([float(P)]),np.array([float(T)]),desc[1])[0])
                    preds.append(pred)
                q["nominal_model_prediction_mol_kg"]=preds
                domain_status=[]; dT=[]; dP=[]
                for _,rr in q.iterrows():
                    gas=normalize_gas(rr.get("adsorbate","")); _fits=fit_payload.get("fits",fit_payload) if isinstance(fit_payload,Mapping) else fit_payload; desc=best_fit_descriptor(_fits,"CALF-20",gas)
                    T=float(pd.to_numeric(pd.Series([rr.get("temperature_K")]),errors="coerce").iloc[0]); P=float(pd.to_numeric(pd.Series([rr.get("pressure_bar")]),errors="coerce").iloc[0])
                    if desc is None or not np.isfinite(T) or not np.isfinite(P): domain_status.append("unresolved"); dT.append(np.nan); dP.append(np.nan); continue
                    dom=desc[2]; tlo=float(dom.get("T_min",dom.get("temperature_min_K",np.nan))); thi=float(dom.get("T_max",dom.get("temperature_max_K",np.nan))); plo=float(dom.get("p_min",dom.get("pressure_min_bar",np.nan))); phi=float(dom.get("p_max",dom.get("pressure_max_bar",np.nan)))
                    dt=(tlo-T if T<tlo else (T-thi if T>thi else 0.0)) if np.isfinite(tlo) and np.isfinite(thi) else np.nan
                    dp=(plo-P if P<plo else (P-phi if P>phi else 0.0)) if np.isfinite(plo) and np.isfinite(phi) else np.nan
                    dT.append(dt); dP.append(dp); domain_status.append("within_nominal_fit_domain" if (dt==0 and dp==0) else "external_extrapolation_crosscheck")
                q["nominal_fit_domain_status"]=domain_status; q["temperature_extrapolation_K"]=dT; q["pressure_extrapolation_bar"]=dP
                obs=pd.to_numeric(q.get("loading_mol_kg"),errors="coerce")
                q["nominal_minus_external_mol_kg"]=q["nominal_model_prediction_mol_kg"]-obs
                q["relative_loading_error"]=(q["nominal_minus_external_mol_kg"]/obs.replace(0,np.nan))
            scalars.append(q)
        except Exception: pass
    if logger and rows: logger.info("External equilibrium source-sensitivity active for %d independent source file(s)",len(set(r["source_file"] for r in rows)))
    return {"boundaries":pd.DataFrame(rows),"fit_summary":pd.concat(fitrows,ignore_index=True) if fitrows else pd.DataFrame(),"scalar_crosschecks":pd.concat(scalars,ignore_index=True) if scalars else pd.DataFrame()}


def _capacity_scaled_pair(pair, co2_scale: float=1.0, n2_scale: float=1.0):
    """Scale saturation capacities while preserving affinity/shape parameters.

    This is used only for non-probabilistic literature-discrepancy stress tests.
    Scaling qsat parameters reproduces a pure loading-amplitude discrepancy while
    retaining the nominal temperature and pressure dependence of each fitted
    model.  It is never used in the primary fit or bootstrap uncertainty.
    """
    out=[]
    for i,(model,theta,dom) in enumerate(pair):
        scale=float(co2_scale if i==0 else n2_scale)
        scale=max(scale,1.0e-6)
        th=np.asarray(theta,float).copy()
        if model=="dual_site_langmuir":
            th[0]+=math.log(scale); th[3]+=math.log(scale)
        elif model in {"langmuir","sips","toth"}:
            th[0]+=math.log(scale)
        else:
            raise ValueError(f"Unsupported capacity scaling for model {model}")
        out.append((model,th,dict(dom)))
    return tuple(out)


def _derive_discrepancy_scale(crosschecks: pd.DataFrame, gas: str, mode: str, ref_T: float, max_T_extrap: float) -> Tuple[float,int,str]:
    if mode=="unity": return 1.0,0,"unity"
    if crosschecks is None or crosschecks.empty: return np.nan,0,"no scalar crosschecks"
    d=crosschecks.copy()
    if "adsorbate" not in d: return np.nan,0,"adsorbate column absent"
    d=d[d["adsorbate"].astype(str).map(normalize_gas)==gas].copy()
    if d.empty: return np.nan,0,"no gas-matched scalar crosschecks"
    for c in ["temperature_K","loading_mol_kg","nominal_model_prediction_mol_kg","temperature_extrapolation_K"]:
        if c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    if "temperature_K" in d:
        d=d[np.abs(d["temperature_K"]-float(ref_T))<=1.0]
    if "temperature_extrapolation_K" in d:
        d=d[d["temperature_extrapolation_K"].fillna(np.inf)<=float(max_T_extrap)]
    d=d[(d["loading_mol_kg"]>0)&(d["nominal_model_prediction_mol_kg"]>0)].copy()
    if d.empty: return np.nan,0,"no eligible near-domain scalar crosschecks"
    ratios=(d["loading_mol_kg"]/d["nominal_model_prediction_mol_kg"]).replace([np.inf,-np.inf],np.nan).dropna()
    if ratios.empty: return np.nan,0,"no finite external/model ratios"
    if mode=="median_external_to_model_ratio": val=float(np.median(ratios))
    elif mode=="max_external_to_model_ratio": val=float(np.max(ratios))
    elif mode=="min_external_to_model_ratio": val=float(np.min(ratios))
    else: return np.nan,0,f"unknown scale mode {mode}"
    return val,int(len(ratios)),f"{mode}; n={len(ratios)}"


def thermodynamic_discrepancy_stress_test(config: RunConfig, fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], scalar_crosschecks: pd.DataFrame, settings: EngineeringSettings, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    """Propagate literature-observed batch/source loading discrepancies as stress cases.

    These cases are deliberately *not* assigned probabilities and are not merged
    with the bootstrap.  Each thermodynamic stress realization is conditioned
    back to the same nominal process purity/recovery anchor, so the exercise asks
    whether off-design intervention ordering survives plausible literature-level
    thermodynamic amplitude discrepancies rather than merely re-testing the
    nominal calibration point.
    """
    policy_path=_optional_data_root(config)/"external_equilibrium_sensitivity"/"CALF20_thermodynamic_discrepancy_stress_policy.csv"
    policy=_safe_read_csv(policy_path)
    if policy.empty:
        return {"scenarios":pd.DataFrame(),"boundaries":pd.DataFrame(),"selected":pd.DataFrame()}
    desc=[]
    for gas in ["CO2","N2"]:
        q=best_fit_descriptor(fit_payload.get("fits",{}),"CALF-20",gas)
        if q is None: return {"scenarios":pd.DataFrame(),"boundaries":pd.DataFrame(),"selected":pd.DataFrame()}
        desc.append(q)
    nominal_pair=tuple(desc)
    ys=np.linspace(settings.feed_co2_min,settings.feed_co2_max,LEVELS[config.level]["composition_n"])
    scenario_rows=[]; boundary_rows=[]; selected_rows=[]
    for _,pr in policy.iterrows():
        enabled=str(pr.get("enabled",True)).strip().lower() not in {"false","0","no"}
        if not enabled: continue
        sid=str(pr.get("scenario_id","stress")); refT=float(pd.to_numeric(pd.Series([pr.get("reference_temperature_K",298.0)]),errors="coerce").iloc[0]); maxext=float(pd.to_numeric(pd.Series([pr.get("max_temperature_extrapolation_K",10.0)]),errors="coerce").iloc[0])
        cs,nc,cwhy=_derive_discrepancy_scale(scalar_crosschecks,"CO2",str(pr.get("co2_scale_mode","unity")),refT,maxext)
        ns,nn,nwhy=_derive_discrepancy_scale(scalar_crosschecks,"N2",str(pr.get("n2_scale_mode","unity")),refT,maxext)
        ok=np.isfinite(cs) and np.isfinite(ns) and cs>0 and ns>0
        scenario_rows.append({**{c:pr.get(c,"") for c in policy.columns},"derived_co2_capacity_scale":cs,"derived_n2_capacity_scale":ns,"n_co2_scalar_points":nc,"n_n2_scalar_points":nn,"co2_scale_derivation":cwhy,"n2_scale_derivation":nwhy,"scenario_status":"evaluated" if ok else "skipped_missing_evidence"})
        if not ok: continue
        pair=_capacity_scaled_pair(nominal_pair,cs,ns)
        ft=_anchor_front_factor_table(evidence_payload.get("summary",pd.DataFrame()),pair,settings)
        cal=calibrate_nominal_surrogate_conditioned(pair,ft,settings,evidence_payload.get("process_anchors",pd.DataFrame()))
        # Fast grid-backed stress atlas. Unlike the primary analysis, this
        # non-probabilistic stress layer needs only the 0/1/2 control class and
        # its boundaries, not a continuously refined setpoint. Cache the
        # regeneration equilibrium states across feed compositions so the
        # literature-discrepancy check remains inexpensive even on one CPU.
        pressure_n=max(9,int(LEVELS[config.level]["pressure_n"]))
        p0=settings.nominal_regeneration_pressure_bar
        npg=max(9,int(math.sqrt(max(pressure_n,9))*2))
        nt=max(9,int(math.sqrt(max(pressure_n,9))*2))
        pgrid=np.unique(np.r_[np.geomspace(settings.regeneration_pressure_floor_bar,p0,npg),p0])
        tgrid=np.unique(np.r_[np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(15,pressure_n)),1.0])
        tjoint=np.unique(np.r_[np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt),1.0])
        reg_cache={float(pp):mixture_state(settings.product_reference_co2_fraction,float(pp),settings.nominal_temperature_K,pair) for pp in pgrid}

        def rel_from_states(ads, reg, ff):
            return {
                "q_ads_co2":ads["q_co2"],"q_ads_n2":ads["q_n2"],
                "q_reg_co2":reg["q_co2"],"q_reg_n2":reg["q_n2"],
                "dco2":max(ads["q_co2"]-reg["q_co2"],0.0)*ff,
                "dn2":max(ads["q_n2"]-reg["q_n2"],0.0)*ff,
                "alpha_ads":ads["alpha"],
                "iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"],
                "front_factor":ff,
            }

        per=[]
        for y in ys:
            y=float(y); ff=front_factor(y,ft)
            ads=mixture_state(y,settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)
            rel0=rel_from_states(ads,reg_cache[float(p0)],ff)
            fixed=_decorate_control_move(_process_metrics_from_release(y,p0,1.0,rel0,cal,settings),settings)
            if bool(fixed["meets_targets"]):
                rr={**fixed,"adaptation_level":0,"adaptation_name":"frozen_process","adaptation_variant":"frozen","success":True}
            else:
                one=[]
                for t in tgrid:
                    r=_process_metrics_from_release(y,p0,float(t),rel0,cal,settings)
                    if r["meets_targets"]:
                        r=_decorate_control_move(r,settings); r["adaptation_variant"]="timing_only"; one.append(r)
                for pp in pgrid:
                    rel=rel_from_states(ads,reg_cache[float(pp)],ff)
                    r=_process_metrics_from_release(y,float(pp),1.0,rel,cal,settings)
                    if r["meets_targets"]:
                        r=_decorate_control_move(r,settings); r["adaptation_variant"]="pressure_only"; one.append(r)
                one_best=_select_one_actuator_policy(one,y,settings) if one else None
                if one_best is not None:
                    rr={**one_best,"adaptation_level":1,"adaptation_name":"single_knob","success":True}
                else:
                    joint=[]
                    for pp in pgrid:
                        rel=rel_from_states(ads,reg_cache[float(pp)],ff)
                        for t in tjoint:
                            r=_process_metrics_from_release(y,float(pp),float(t),rel,cal,settings)
                            if r["meets_targets"]:
                                joint.append(_decorate_control_move(r,settings))
                    joint_best=_choose_best_feasible(joint,settings) if joint else None
                    if joint_best is not None:
                        rr={**joint_best,"adaptation_level":2,"adaptation_name":"coordinated_two_knob","adaptation_variant":"pressure_plus_timing","success":True}
                    else:
                        rr={**fixed,"adaptation_level":3,"adaptation_name":"outside_normal_control_envelope","adaptation_variant":"none_feasible","success":False}
            rr.update({"scenario_id":sid,"feed_co2_fraction":y,"co2_capacity_scale":cs,"n2_capacity_scale":ns})
            per.append(rr); selected_rows.append(rr)
        sdf=pd.DataFrame(per)
        for lev,label in [(0,"frozen"),(1,"single_or_less"),(2,"coordinated_or_less")]:
            tmp=pd.DataFrame({"feed_co2_fraction":ys,"eligible":pd.to_numeric(sdf["adaptation_level"],errors="coerce")<=lev})
            lo,hi=_contiguous_interval_containing_nominal(tmp,settings.nominal_feed_co2_fraction)
            boundary_rows.append({"scenario_id":sid,"scenario_label":pr.get("scenario_label",sid),"intervention_ceiling":label,"lower_feed_co2_fraction":lo,"upper_feed_co2_fraction":hi,"lower_boundary_resolved":bool(np.isfinite(lo) and abs(lo-settings.feed_co2_min)>1e-12),"upper_boundary_censored":bool(np.isfinite(hi) and abs(hi-settings.feed_co2_max)<1e-12),"co2_capacity_scale":cs,"n2_capacity_scale":ns,"nominal_conditioned_purity":cal.get("purity_anchor",np.nan),"nominal_conditioned_recovery":cal.get("recovery_anchor",np.nan),"stress_role":"non-probabilistic literature discrepancy stress"})
    if logger and boundary_rows:
        logger.info("Thermodynamic discrepancy stress evaluated for %d scenario(s)",len(set(r["scenario_id"] for r in boundary_rows)))
    return {"scenarios":pd.DataFrame(scenario_rows),"boundaries":pd.DataFrame(boundary_rows),"selected":pd.DataFrame(selected_rows)}


def ingest_css_validation(config: RunConfig, atlas_payload: Mapping[str,Any], settings: EngineeringSettings, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    """Ingest future rigorous CSS results without ever fabricating them.

    Accepted folders are data_root/css_validation or data_root/higher_fidelity_validation.
    Required fields: feed_co2_fraction, purity, recovery. Recommended: css_converged,
    p_reg_bar, timing_scale, intervention_class, case_id, energy_kJ_mol_CO2.
    """
    root=_optional_data_root(config); folders=[root/"css_validation",root/"higher_fidelity_validation"]
    parts=[]
    for fd in folders:
        z=_read_csvs_under(fd)
        if len(z): parts.append(z)
    if not parts: return {"raw":pd.DataFrame(),"comparison":pd.DataFrame(),"summary":pd.DataFrame()}
    d=pd.concat(parts,ignore_index=True,sort=False)
    aliases={"feed_co2_fraction":["feed_co2_fraction","y_co2","feed_y_co2"],"purity":["purity","purity_fraction","co2_purity"],"recovery":["recovery","recovery_fraction","co2_recovery"]}
    ren={}
    for k,als in aliases.items():
        c=_find_any_column(d.columns,als)
        if c is not None: ren[c]=k
    d=d.rename(columns=ren)
    if not {"feed_co2_fraction","purity","recovery"}.issubset(d.columns): return {"raw":d,"comparison":pd.DataFrame(),"summary":pd.DataFrame()}
    d["feed_co2_fraction"]=_fraction_series(d["feed_co2_fraction"]); d["purity"]=_fraction_series(d["purity"]); d["recovery"]=_fraction_series(d["recovery"])
    d["css_meets_targets"]=(d["purity"]>=settings.target_purity)&(d["recovery"]>=settings.target_recovery)
    sel=atlas_payload.get("selected",pd.DataFrame())
    comp=[]
    for _,r in d.iterrows():
        if sel.empty or not np.isfinite(r["feed_co2_fraction"]): pred=np.nan
        else:
            j=(sel["feed_co2_fraction"]-float(r["feed_co2_fraction"])).abs().idxmin(); pred=float(sel.loc[j,"adaptation_level"])
        actual=np.nan
        c=_find_any_column(d.columns,["intervention_class","adaptation_level","minimum_intervention_level"])
        if c is not None:
            try: actual=float(r[c])
            except Exception: actual=np.nan
        comp.append({"feed_co2_fraction":r["feed_co2_fraction"],"css_purity":r["purity"],"css_recovery":r["recovery"],"css_meets_targets":bool(r["css_meets_targets"]),"reduced_order_nearest_intervention_level":pred,"css_reported_intervention_level":actual,"case_id":r.get("case_id",r.get("source_id",""))})
    comparison=pd.DataFrame(comp)
    summary=pd.DataFrame([{"n_css_rows":len(d),"n_target_feasible":int(d["css_meets_targets"].sum()),"fraction_target_feasible":float(d["css_meets_targets"].mean()),"composition_min":float(d["feed_co2_fraction"].min()),"composition_max":float(d["feed_co2_fraction"].max()),"role":"independent high-fidelity falsification; not used to calibrate reduced-order model"}])
    if logger: logger.info("Higher-fidelity CSS validation data detected: %d rows",len(d))
    return {"raw":d,"comparison":comparison,"summary":summary}


# Wrap stage 05 to add mechanism/control-authority data products.
_stage_05_v130 = stage_05_feed_composition_robustness

def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_05_v130(config,dirs,fit_payload,evidence_payload,logger)
    mech=mechanism_and_control_authority(out,config.engineering)
    save_table_multi(mech,dirs["engineering"] / "mechanism_control_authority_map",SAVE_MODES[config.save_mode])
    out["mechanism"]=mech
    return out


# Replace uniform retained-model drawing with AICc-weighted model-form drawing.
def _draw_pair(fit_payload, uncertainty_payload, rng: np.random.Generator, allow_model_form: bool):
    fits=fit_payload["fits"]; out=[]
    for gas in ["CO2","N2"]:
        fs=fits.get(("CALF-20",gas),{})
        models=list(fs.get("retained_models",[])) if allow_model_form else [fs.get("best_model")]
        models=[m for m in models if m and fs.get("models",{}).get(m,{}).get("success")]
        if not models: return None
        if len(models)>1:
            a=np.array([fs["models"][m].get("aicc",np.nan) for m in models],float)
            if np.isfinite(a).any():
                amin=np.nanmin(a); w=np.exp(-0.5*np.where(np.isfinite(a),a-amin,50.0)); w=w/w.sum()
            else: w=np.full(len(models),1/len(models))
            model=str(rng.choice(models,p=w))
        else: model=models[0]
        samples=_successful_boot_theta(uncertainty_payload,"CALF-20",gas,model)
        theta=samples[int(rng.integers(0,len(samples)))] if samples else np.asarray(fs["models"][model]["theta"],dtype=float)
        out.append((model,theta,fs["domain"]))
    return tuple(out)


# Wrap stage 07 to activate optional external evidence and future CSS validation.
_stage_07_v130 = stage_07_validation

def stage_07_validation(config: RunConfig, dirs: Mapping[str,Path], harmonized_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    if config.level=="quickcheck":
        out={"blocked":pd.DataFrame(),"breakthrough":breakthrough_validation(evidence_payload,atlas_payload),"numerical":pd.DataFrame(),"boundary_grid":pd.DataFrame(),"falsification_portfolio":boundary_falsification_portfolio(atlas_payload,config.engineering),"model_sensitivity":model_form_sensitivity(fit_payload,evidence_payload,config.engineering)}
        for name,df in [("breakthrough_validation",out["breakthrough"]),("boundary_falsification_portfolio",out["falsification_portfolio"]),("model_form_process_sensitivity",out["model_sensitivity"])]:
            save_table_multi(df,dirs["validation"] / name,SAVE_MODES[config.save_mode])
    else:
        logger.info("Validation stage: core blocked/numerical/grid checks starting")
        out=_stage_07_v130(config,dirs,harmonized_payload,fit_payload,evidence_payload,atlas_payload,logger)
        logger.info("Validation stage: core checks complete")
    logger.info("Validation stage: optional external-pilot parser starting")
    pilot=external_pilot_validation(config,config.engineering,logger)
    logger.info("Validation stage: optional external-pilot parser complete")
    if config.level=="quickcheck":
        source={"boundaries":pd.DataFrame(),"fit_summary":pd.DataFrame(),"scalar_crosschecks":pd.DataFrame()}
    else:
        logger.info("Validation stage: optional independent-thermodynamics sensitivity starting")
        source=external_equilibrium_source_sensitivity(config,evidence_payload,config.engineering,fit_payload,logger)
        logger.info("Validation stage: optional independent-thermodynamics sensitivity complete")
    logger.info("Validation stage: literature discrepancy stress starting")
    stress=thermodynamic_discrepancy_stress_test(config,fit_payload,evidence_payload,source.get("scalar_crosschecks",pd.DataFrame()),config.engineering,logger) if config.level!="quickcheck" else {"scenarios":pd.DataFrame(),"boundaries":pd.DataFrame(),"selected":pd.DataFrame()}
    logger.info("Validation stage: literature discrepancy stress complete")
    logger.info("Validation stage: optional CSS-ingestion starting")
    css=ingest_css_validation(config,atlas_payload,config.engineering,logger)
    logger.info("Validation stage: optional CSS-ingestion complete")
    rec_parts=[]
    for label,df0 in [("frozen",atlas_payload.get("fixed",pd.DataFrame())),("selected",atlas_payload.get("selected",pd.DataFrame()))]:
        if isinstance(df0,pd.DataFrame) and len(df0):
            keep=[c for c in ["feed_co2_fraction","p_reg_bar","timing_scale","purity_estimate","recovery_estimate","co2_feed_mol_kg_cycle","co2_release_capacity_mol_kg_cycle","recovery_capacity_ratio_unbounded","recovery_feed_limited","mass_balance_recovery_violation","adaptation_level","adaptation_name","success"] if c in df0.columns]
            rr=df0[keep].copy(); rr.insert(0,"state_family",label); rec_parts.append(rr)
    recovery_audit=pd.concat(rec_parts,ignore_index=True,sort=False) if rec_parts else pd.DataFrame()
    for name,df in [("external_pilot_raw",pilot["raw"]),("external_pilot_composition_summary",pilot["summary"]),("external_pilot_adjusted_association",pilot["association"]),("external_equilibrium_source_boundary_sensitivity",source["boundaries"]),("external_equilibrium_fit_summary",source["fit_summary"]),("external_equilibrium_scalar_crosschecks",source["scalar_crosschecks"]),("thermodynamic_discrepancy_stress_scenarios",stress["scenarios"]),("thermodynamic_discrepancy_stress_boundaries",stress["boundaries"]),("thermodynamic_discrepancy_stress_selected",stress["selected"]),("css_validation_raw",css["raw"]),("css_validation_comparison",css["comparison"]),("css_validation_summary",css["summary"]),("recovery_mass_balance_audit",recovery_audit)]:
        save_table_multi(df,dirs["validation"] / name,SAVE_MODES[config.save_mode])
    out.update({"external_pilot":pilot,"external_source_sensitivity":source,"thermodynamic_discrepancy_stress":stress,"css_validation":css,"recovery_mass_balance_audit":recovery_audit})
    return out


# ---------------------------------------------------------------------------
# Publication tables
# ---------------------------------------------------------------------------
def configure_matplotlib() -> None:
    plt.rcParams.update({"font.size":9.5,"axes.titlesize":10.5,"axes.labelsize":9.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.0,"figure.titlesize":12.0,"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white","figure.facecolor":"white"})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12,1.05,label,transform=ax.transAxes,fontsize=11,fontweight="bold",va="top",ha="left")


def empty_axis(ax: plt.Axes, message: str) -> None:
    ax.axis("off"); ax.text(0.5,0.5,message,ha="center",va="center",transform=ax.transAxes,wrap=True)


def save_panel_source(df: pd.DataFrame, path: Path, save_opts: Mapping[str,Any]) -> None:
    if save_opts.get("panel_csv") and isinstance(df,pd.DataFrame): save_table_multi(df,path,save_opts)


def save_publication_table_artifacts(df: pd.DataFrame, stem: Path, preview_dir: Path, save_opts: Mapping[str,Any]) -> None:
    stem.parent.mkdir(parents=True,exist_ok=True)
    try: write_text(stem.with_suffix(".tex"),df.to_latex(index=False,escape=True,longtable=len(df)>35,float_format=lambda x:f"{x:.5g}"))
    except Exception: logging.getLogger("p10").exception("Could not write LaTeX table %s",stem)
    if df.empty: return
    raw_show=df.head(24).copy()
    if raw_show.shape[1]>12: raw_show=raw_show.iloc[:,:12]

    # Build a fresh plain-object display frame rather than mutating Arrow-backed
    # extension arrays in place.  In pandas/pyarrow combinations, astype(str)
    # can still expose missing scalars as float/NA objects to Series.map; calling
    # These payloads are mappings rather than row-oriented tables.
    display_cols: Dict[Any, List[str]] = {}
    for c in raw_show.columns:
        series = raw_show[c]
        numeric = pd.api.types.is_numeric_dtype(series)
        vals: List[str] = []
        for value in series.astype("object").tolist():
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            if missing:
                text = ""
            elif numeric:
                try:
                    text = f"{float(value):.4g}"
                except (TypeError, ValueError, OverflowError):
                    text = str(value)
            else:
                text = str(value)
            if len(text) > 34:
                text = text[:31] + "…"
            vals.append(text)
        display_cols[c] = vals
    show = pd.DataFrame(display_cols, index=raw_show.index, dtype=object)
    fig,ax=plt.subplots(figsize=(max(8,1.15*show.shape[1]),max(2.4,0.34*(len(show)+3)))); ax.axis("off")
    tab=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center"); tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.22)
    for (r,c),cell in tab.get_celld().items():
        if r==0: cell.set_text_props(weight="bold")
        cell.set_linewidth(0.35)
    ax.set_title(stem.name.replace("_"," "),fontsize=10.5,pad=10); preview_dir.mkdir(parents=True,exist_ok=True)
    save_figure(fig,preview_dir/stem.name,save_opts)


def source_registry(harmonized_payload,evidence_payload) -> pd.DataFrame:
    d=harmonized_payload["data"]
    rows=[]
    if len(d):
        for keys,g in d.groupby(["source_id","material","material_form","doi"],dropna=False):
            rows.append({"source_id":keys[0],"material":keys[1],"material_form":keys[2],"doi":keys[3],"data_role":"unary_equilibrium","gases":"; ".join(sorted(g["adsorbate"].astype(str).unique())),"temperature_min_K":g["temperature_K"].min(),"temperature_max_K":g["temperature_K"].max(),"pressure_min_bar":g["pressure_bar"].min(),"pressure_max_bar":g["pressure_bar"].max(),"n_rows":len(g)})
    if len(evidence_payload["summary"]):
        for _,r in evidence_payload["summary"].iterrows():
            rows.append({"source_id":r["source_id"],"material":"CALF-20","material_form":"see provenance","doi":"","data_role":"breakthrough_composition_anchor","gases":"CO2; N2","temperature_min_K":r["temperature_K"],"temperature_max_K":r["temperature_K"],"pressure_min_bar":r["total_pressure_bar"],"pressure_max_bar":r["total_pressure_bar"],"n_rows":r["n_trace_points"]})
    if len(evidence_payload.get("composition_anchors",pd.DataFrame())):
        for src,g in evidence_payload["composition_anchors"].groupby("source_id"):
            rows.append({"source_id":src,"material":"CALF-20","material_form":"see provenance","doi":"; ".join(sorted(set(g.get("doi",pd.Series(dtype=str)).dropna().astype(str)))) if "doi" in g else "","data_role":"published_composition_condition_anchor_no_trace","gases":"CO2; N2","temperature_min_K":pd.to_numeric(g.get("temperature_K"),errors="coerce").min(),"temperature_max_K":pd.to_numeric(g.get("temperature_K"),errors="coerce").max(),"pressure_min_bar":pd.to_numeric(g.get("total_pressure_bar"),errors="coerce").min(),"pressure_max_bar":pd.to_numeric(g.get("total_pressure_bar"),errors="coerce").max(),"n_rows":len(g)})
    if len(evidence_payload["process_anchors"]):
        for src,g in evidence_payload["process_anchors"].groupby("source_id"):
            doi="; ".join(sorted(set(g["doi"].dropna().astype(str)))) if "doi" in g else ""
            rows.append({"source_id":src,"material":"CALF-20","material_form":"structured/process sample","doi":doi,"data_role":"nominal_process_anchor","gases":"CO2; N2","temperature_min_K":np.nan,"temperature_max_K":np.nan,"pressure_min_bar":np.nan,"pressure_max_bar":np.nan,"n_rows":len(g)})
    return pd.DataFrame(rows)


def parameter_table(fit_payload,uncertainty_payload) -> pd.DataFrame:
    s=fit_payload.get("summary",pd.DataFrame()).copy()
    if s.empty: return s
    return s[s["material"].eq("CALF-20")].copy()


def headline_table(atlas_payload,robustness_payload,settings) -> pd.DataFrame:
    sel=atlas_payload["selected"]; rows=[]
    if len(sel):
        for y in [0.05,settings.nominal_feed_co2_fraction,0.30]:
            r=sel.iloc[(sel["feed_co2_fraction"]-y).abs().argsort()[:1]].iloc[0]
            rows.append({"finding":f"minimum intervention near yCO2={y:.2f}","feed_co2_fraction":r["feed_co2_fraction"],"robustness_class":r["robustness_class"],"p_reg_bar":r["p_reg_bar"],"timing_scale":r["timing_scale"],"purity_estimate":r["purity_estimate"],"recovery_estimate":r["recovery_estimate"],"specific_vacuum_work_kJ_mol_CO2":r["specific_vacuum_work_kJ_mol_CO2"],"composition_support":r["composition_support"]})
    b=robustness_payload.get("boundaries",pd.DataFrame())
    for _,r in b.iterrows():
        status=str(r.get("boundary_status",""))
        support=str(r.get("median_composition_support",""))
        interval=f"95% interval {r['q025']:.3g}-{r['q975']:.3g}" if np.isfinite(r["q025"]) else "not resolved"
        rows.append({"finding":r["boundary"],"feed_co2_fraction":r["median_feed_co2_fraction"],"robustness_class":"uncertainty_boundary","p_reg_bar":np.nan,"timing_scale":np.nan,"purity_estimate":np.nan,"recovery_estimate":np.nan,"specific_vacuum_work_kJ_mol_CO2":np.nan,"composition_support":" | ".join(x for x in [status,support,interval] if x)})
    return pd.DataFrame(rows)


def stage_08_tables(config: RunConfig, dirs: Mapping[str,Path], audit_payload, harmonized_payload, fit_payload, uncertainty_payload, evidence_payload, atlas_payload, robustness_payload, validation_payload, logger) -> Dict[str,Any]:
    parser_sr=source_registry(harmonized_payload,evidence_payload); root=_optional_data_root(config)
    package_sr=_safe_read_csv(root/"provenance"/"catalog"/"source_registry.csv")
    numeric_facts=_safe_read_csv(root/"provenance"/"catalog"/"numeric_fact_registry.csv")
    data_gaps=_safe_read_csv(root/"provenance"/"catalog"/"data_gap_register.csv")
    role_files=sorted((root/"literature_evidence").glob("CALF20_literature_evidence_roles_v*.csv")) if (root/"literature_evidence").exists() else []
    evidence_roles=_safe_read_csv(role_files[-1]) if role_files else pd.DataFrame()
    campaign_context=_safe_read_csv(root/"literature_evidence"/"CALF20_campaign_context.csv")
    lit_audit=_safe_read_csv(dirs["audit"]/"literature_data_audit.csv")
    sr=package_sr if len(package_sr) else parser_sr
    pt=parameter_table(fit_payload,uncertainty_payload); ht=headline_table(atlas_payload,robustness_payload,config.engineering)
    css_n=len(validation_payload.get("css_validation",{}).get("comparison",pd.DataFrame()))
    pilot_n=len(validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame()))
    src_n=len(validation_payload.get("external_source_sensitivity",{}).get("boundaries",pd.DataFrame()))
    claim_matrix=pd.DataFrame([
        {"claim_id":"C1","claim":"Nominal high performance does not imply broad frozen-feed robustness.","primary_evidence":"Figure 3; Figure 5; Table 3","validation_layer":"calibrated reduced-order process + uncertainty","status":"supported within declared reduced-order model"},
        {"claim_id":"C2","claim":"Successive control degrees of freedom expand the lean-feed operability boundary.","primary_evidence":"Figure 4; Figure 5; Table 3","validation_layer":"grid convergence + independent thermodynamic source","status":"supported; CSS boundary validation pending" if css_n==0 else "supported with CSS comparison"},
        {"claim_id":"C3","claim":"Dominant failure switches from dilution-side purity limitation to rich-side recovery limitation.","primary_evidence":"Figure 2; Figure 3","validation_layer":"mechanism diagnostics + external pilot direction","status":"supported; external pilot is directional, not matched-cycle" if pilot_n else "model-supported; external pilot absent"},
        {"claim_id":"C4","claim":"Intervention ordering is insensitive to an independent published CALF-20 thermodynamic parameterization.","primary_evidence":"Figure S16; Table S20b","validation_layer":"independent thermodynamic source","status":"supported" if src_n else "not evaluated"},
        {"claim_id":"C5","claim":"Exact 100% reduced-order recoveries arise only when release capacity exceeds feed CO2 and are mass-balance capped.","primary_evidence":"Table S20a","validation_layer":"explicit feed/release-capacity audit","status":"auditable reduced-order behavior; not a claim of perfect real-cycle capture"},
        {"claim_id":"C6","claim":"Rigorous non-isothermal CSS preserves the predicted boundary ordering.","primary_evidence":"future CSS validation","validation_layer":"high-fidelity cycle model","status":"not yet tested" if css_n==0 else "tested in supplied CSS cases"},
        {"claim_id":"C7","claim":"The ordering of minimum-control-authority boundaries is stress-tested against independent literature-scale thermodynamic discrepancies.","primary_evidence":"Table S20d2-S20d3; Figure S17","validation_layer":"non-probabilistic, calibration-conditioned literature discrepancy stress","status":"stress-tested; exact boundary values remain model dependent"},
    ])
    main={"Table_1_literature_datasets_and_provenance":sr,"Table_2_fitted_thermodynamic_parameters":pt,"Table_3_minimum_intervention_boundaries":ht}
    for name,df in main.items():
        stem=dirs["tables_main"]/name; save_table_multi(df,stem,SAVE_MODES[config.save_mode]); save_publication_table_artifacts(df,stem,dirs["table_fig_main"],SAVE_MODES[config.save_mode])
    panel_map=pd.DataFrame([
        ("Figure 1","A-D","composition evidence, unary equilibrium and independent pilot validation","harmonized adsorption + composition anchors + external pilot PVSA"),
        ("Figure 2","A-D","intrinsic composition response","intrinsic_composition_response"),
        ("Figure 3","A-D","frozen-process reduced-order estimates","fixed_process_offdesign"),
        ("Figure 4","A-D","minimum operating intervention","minimum_intervention_selected"),
        ("Figure 5","A-D","nested feed-composition robustness boundaries and uncertainty","nested_intervention_windows + uncertainty classification"),
    ],columns=["figure","panel","content","source_dataset"])
    si={
        "Table_S1_input_file_audit":audit_payload.get("audit",pd.DataFrame()),
        "Table_S1a_parser_detected_sources":parser_sr,
        "Table_S1b_literature_data_integrity_audit":lit_audit,
        "Table_S1c_numeric_fact_registry":numeric_facts,
        "Table_S1d_data_gap_registry":data_gaps,
        "Table_S1e_literature_evidence_roles":evidence_roles,
        "Table_S1f_literature_campaign_context":campaign_context,
        "Table_S1g_claim_evidence_matrix":claim_matrix,
        "Table_S2_input_missingness":audit_payload.get("missingness",pd.DataFrame()),
        "Table_S3_harmonized_data_domain":harmonized_payload.get("domain",pd.DataFrame()),"Table_S4_model_comparison":fit_payload.get("summary",pd.DataFrame()),
        "Table_S5_bootstrap_parameter_summary":uncertainty_payload.get("summary",pd.DataFrame()),"Table_S6_breakthrough_anchor_summary":evidence_payload.get("summary",pd.DataFrame()),"Table_S6b_published_composition_condition_anchors":evidence_payload.get("composition_anchors",pd.DataFrame()),
        "Table_S7_process_anchor_data":evidence_payload.get("process_anchors",pd.DataFrame()),"Table_S8_engineering_assumptions":pd.DataFrame([asdict(config.engineering)]),
        "Table_S9_intrinsic_composition_response":atlas_payload.get("intrinsic",pd.DataFrame()),"Table_S10_fixed_process_offdesign":atlas_payload.get("fixed",pd.DataFrame()),
        "Table_S11_all_intervention_candidates":atlas_payload.get("adaptation",pd.DataFrame()),"Table_S12_pressure_response_diagnostic":atlas_payload.get("pressure_atlas",pd.DataFrame()),"Table_S12b_nested_intervention_windows":atlas_payload.get("windows",pd.DataFrame()),
        "Table_S13_uncertainty_classification":robustness_payload.get("classification",pd.DataFrame()),"Table_S14_intervention_boundaries":robustness_payload.get("boundaries",pd.DataFrame()),
        "Table_S15_target_threshold_sensitivity_selected":robustness_payload.get("threshold_sensitivity",pd.DataFrame()),"Table_S16_target_threshold_sensitivity_windows":robustness_payload.get("threshold_windows",pd.DataFrame()),
        "Table_S17_adaptation_recovery_decomposition":atlas_payload.get("decomposition",pd.DataFrame()),"Table_S17b_intervention_boundary_diagnostics":atlas_payload.get("boundary_diagnostics",pd.DataFrame()),"Table_S17c_mechanism_control_authority_map":atlas_payload.get("mechanism",pd.DataFrame()),
        "Table_S18_blocked_validation":validation_payload.get("blocked",pd.DataFrame()),"Table_S19_breakthrough_validation":validation_payload.get("breakthrough",pd.DataFrame()),
        "Table_S20_model_form_sensitivity":validation_payload.get("model_sensitivity",pd.DataFrame()),"Table_S20a_recovery_mass_balance_audit":validation_payload.get("recovery_mass_balance_audit",pd.DataFrame()),"Table_S20b_external_equilibrium_source_boundary_sensitivity":validation_payload.get("external_source_sensitivity",{}).get("boundaries",pd.DataFrame()),"Table_S20c_external_equilibrium_fit_summary":validation_payload.get("external_source_sensitivity",{}).get("fit_summary",pd.DataFrame()),"Table_S20d_external_equilibrium_scalar_crosschecks":validation_payload.get("external_source_sensitivity",{}).get("scalar_crosschecks",pd.DataFrame()),"Table_S20d2_thermodynamic_discrepancy_stress_scenarios":validation_payload.get("thermodynamic_discrepancy_stress",{}).get("scenarios",pd.DataFrame()),"Table_S20d3_thermodynamic_discrepancy_stress_boundaries":validation_payload.get("thermodynamic_discrepancy_stress",{}).get("boundaries",pd.DataFrame()),"Table_S20e_external_pilot_composition_summary":validation_payload.get("external_pilot",{}).get("summary",pd.DataFrame()),"Table_S20f_external_pilot_adjusted_association":validation_payload.get("external_pilot",{}).get("association",pd.DataFrame()),"Table_S20g_css_validation_comparison":validation_payload.get("css_validation",{}).get("comparison",pd.DataFrame()),"Table_S21_numerical_verification":validation_payload.get("numerical",pd.DataFrame()),"Table_S21b_boundary_grid_convergence":validation_payload.get("boundary_grid",pd.DataFrame()),"Table_S21c_boundary_falsification_portfolio":validation_payload.get("falsification_portfolio",pd.DataFrame()),
        "Table_S22_panel_to_source_mapping":panel_map,
    }
    for name,df in si.items():
        stem=dirs["tables_si"]/name; save_table_multi(df,stem,SAVE_MODES[config.save_mode])
        if len(df)<=120 and df.shape[1]<=18: save_publication_table_artifacts(df,stem,dirs["table_fig_si"],SAVE_MODES[config.save_mode])
    return {"main":main,"si":si,"headline":ht,"panel_map":panel_map,"claim_evidence_matrix":claim_matrix}


# ---------------------------------------------------------------------------
# Main and SI figures
# ---------------------------------------------------------------------------
def _plot_fit(ax,data,fit_payload,gas,title):
    g=data[(data["material"]=="CALF-20")&(data["adsorbate"]==gas)]
    if g.empty: empty_axis(ax,f"No CALF-20 {gas} equilibrium data"); return pd.DataFrame()
    fs=fit_payload["fits"].get(("CALF-20",gas),{}); model=fs.get("best_model")
    if not model: empty_axis(ax,f"No fitted CALF-20 {gas} model"); return pd.DataFrame()
    th=fs["models"][model]["theta"]; rows=[]
    for T in sorted(g["temperature_K"].unique()):
        gg=g[g["temperature_K"]==T]; line,=ax.plot([],[],label=f"{T:g} K"); ax.scatter(gg["pressure_bar"],gg["loading_mol_kg"],s=16,color=line.get_color(),alpha=.8)
        pos=gg.loc[gg["pressure_bar"]>0,"pressure_bar"]; pmin=max(pos.min()*0.8,1e-5) if len(pos) else 1e-4; pmax=max(gg["pressure_bar"].max()*1.05,pmin*2); p=np.geomspace(pmin,pmax,160)
        q=model_loading(model,p,np.full_like(p,T),th); ax.plot(p,q,color=line.get_color(),lw=1.4)
        rows.extend({"gas":gas,"temperature_K":T,"pressure_bar":pp,"predicted_loading_mol_kg":qq,"model":model} for pp,qq in zip(p,q))
    ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Loading (mol kg$^{-1}$)"); ax.set_title(title); ax.grid(alpha=.18); ax.legend(frameon=False,ncol=2,fontsize=7)
    return pd.DataFrame(rows)


def figure_1(dirs,config,harmonized_payload,fit_payload,evidence_payload,validation_payload):
    data=harmonized_payload["data"]; traces=evidence_payload["traces"]
    pilot=validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame())
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.9),constrained_layout=True); src=[]; s=config.engineering

    # a) Evidence landscape: distinguish literature composition-condition anchors,
    # nominal process calibration point, and independent pilot compositions.
    ax=axs[0,0]; meta=evidence_payload.get("composition_anchors",pd.DataFrame())
    bt=sorted(meta["feed_co2_fraction"].dropna().unique()) if len(meta) and "feed_co2_fraction" in meta else []
    pp=sorted(pd.to_numeric(pilot.get("feed_co2_fraction",pd.Series(dtype=float)),errors="coerce").dropna().unique()) if len(pilot) else []
    ax.hlines(0.66,s.feed_co2_min,s.feed_co2_max,lw=1); ax.hlines(0.34,s.feed_co2_min,s.feed_co2_max,lw=1)
    if bt: ax.scatter(bt,[.66]*len(bt),s=60,label="published breakthrough conditions")
    if pp: ax.scatter(pp,[.34]*len(pp),s=55,marker="s",label="independent pilot PVSA")
    ax.scatter([s.nominal_feed_co2_fraction],[.66],s=115,facecolors="none",linewidths=1.5,label="nominal VSA anchor")
    ax.axvspan(s.primary_evidence_feed_min,s.primary_evidence_feed_max,alpha=.06); ax.axvline(s.nominal_feed_co2_fraction,ls="--",lw=.9)
    ax.set_ylim(.12,.88); ax.set_yticks([.34,.66],["pilot PVSA","BT / nominal"]); ax.set_xlabel("Feed CO$_2$ mole fraction"); ax.set_title("Composition evidence landscape"); ax.legend(frameon=False,fontsize=7,loc="lower right"); panel_label(ax,"a")

    # b) Thermodynamic foundation.
    c1=_plot_fit(axs[0,1],data,fit_payload,"CO2","CALF-20 unary CO$_2$ equilibrium"); panel_label(axs[0,1],"b"); src.append(c1)

    # c-d) Prefer independent pilot evidence when available. Exact breakthrough traces,
    # if supplied later, remain available in SI validation products rather than leaving
    # empty main-text panels.
    ax=axs[1,0]
    if len(pilot):
        q=pilot.copy(); q["feed_co2_fraction"]=pd.to_numeric(q["feed_co2_fraction"],errors="coerce"); q["purity_fraction"]=pd.to_numeric(q["purity_fraction"],errors="coerce"); q["recovery_fraction"]=pd.to_numeric(q["recovery_fraction"],errors="coerce"); q=q.dropna(subset=["feed_co2_fraction","purity_fraction","recovery_fraction"])
        for y,g in q.groupby("feed_co2_fraction"):
            ax.scatter(100*g["purity_fraction"],100*g["recovery_fraction"],s=28,alpha=.72,label=f"{100*y:.0f}% feed CO$_2$")
        ax.axvline(100*s.target_purity,ls="--",lw=.9); ax.axhline(100*s.target_recovery,ls="--",lw=.9); ax.set_xlabel("Pilot CO$_2$ purity (%)"); ax.set_ylabel("Pilot CO$_2$ recovery (%)"); ax.set_title("Independent 75-run pilot process evidence"); ax.legend(frameon=False,fontsize=7)
        src.append(q.assign(panel="1c_external_pilot"))
    elif not traces.empty:
        for y,g in traces.groupby("feed_co2_fraction"): ax.plot(g["time_s"],g["normalized_outlet"],label=f"{100*y:.0f}% CO$_2$")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("CO$_2$ outlet / inlet"); ax.set_title("Composition-dependent breakthrough traces"); ax.legend(frameon=False)
    else: empty_axis(ax,"No independent pilot or audited trace-level validation data supplied")
    panel_label(ax,"c")

    ax=axs[1,1]
    if len(pilot):
        q=pilot.copy(); q["feed_co2_fraction"]=pd.to_numeric(q["feed_co2_fraction"],errors="coerce"); q["recovery_fraction"]=pd.to_numeric(q["recovery_fraction"],errors="coerce"); q=q.dropna(subset=["feed_co2_fraction","recovery_fraction"])
        group=["feed_co2_fraction"]+(["contactor_type"] if "contactor_type" in q.columns else [])
        for key,g in q.groupby(group,dropna=False):
            label=(f"{key[1]}" if isinstance(key,tuple) else "pilot")
            ax.scatter(100*g["feed_co2_fraction"],100*g["recovery_fraction"],s=24,alpha=.45,label=label)
        med=q.groupby("feed_co2_fraction",as_index=False)["recovery_fraction"].median(); ax.plot(100*med["feed_co2_fraction"],100*med["recovery_fraction"],marker="o",lw=1.6,label="all-run median")
        ax.axhline(100*s.target_recovery,ls="--",lw=.9); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Pilot recovery (%)"); ax.set_title("External composition-response check");
        # Deduplicate legend entries from repeated contactor/feed groups.
        h,l=ax.get_legend_handles_labels(); seen={}; [(seen.setdefault(lbl,hh)) for hh,lbl in zip(h,l)]; ax.legend(seen.values(),seen.keys(),frameon=False,fontsize=7)
    else: empty_axis(ax,"External pilot composition-response data not supplied")
    panel_label(ax,"d")
    save_figure(fig,dirs["fig_main"] / "Figure_1_experimental_composition_foundation",SAVE_MODES[config.save_mode]); save_panel_source(pd.concat(src,ignore_index=True,sort=False) if src else pd.DataFrame(),dirs["source_main"] / "Figure_1_source_data",SAVE_MODES[config.save_mode])


def figure_2(dirs,config,atlas_payload):
    d=atlas_payload["intrinsic"]; fig,axs=plt.subplots(2,2,figsize=(11.2,7.7),constrained_layout=True)
    if d.empty:
        [empty_axis(ax,"No CALF-20 CO2/N2 fitted pair") for ax in axs.flat]
    else:
        x=100*d["feed_co2_fraction"]
        axs[0,0].plot(x,d["q_co2_mol_kg"],label="CO$_2$"); axs[0,0].plot(x,d["q_n2_mol_kg"],label="N$_2$"); axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)"); axs[0,0].legend(frameon=False)
        axs[0,1].plot(x,d["adsorbed_x_co2"]); axs[0,1].plot(x,d["feed_co2_fraction"],ls="--",lw=1,label="gas-phase y"); axs[0,1].set_ylabel("Adsorbed CO$_2$ mole fraction"); axs[0,1].legend(frameon=False)
        axs[1,0].semilogy(x,d["iast_selectivity_co2_n2"]); axs[1,0].set_ylabel("IAST selectivity CO$_2$/N$_2$")
        axs[1,1].plot(x,d["working_capacity_co2_mol_kg"],label="CO$_2$"); axs[1,1].plot(x,d["working_capacity_n2_mol_kg"],label="N$_2$"); axs[1,1].set_ylabel("Reduced-order working capacity (mol kg$^{-1}$)"); axs[1,1].legend(frameon=False)
        for i,ax in enumerate(axs.flat): ax.set_xlabel("Feed CO$_2$ (%)"); ax.grid(alpha=.18); ax.axvspan(100*config.engineering.primary_evidence_feed_min,100*config.engineering.primary_evidence_feed_max,alpha=.06); ax.axvline(100*config.engineering.nominal_feed_co2_fraction,ls="--",lw=.8); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Competitive mixture loading"); axs[0,1].set_title("Competitive displacement"); axs[1,0].set_title("Equilibrium separation strength"); axs[1,1].set_title("Regeneration-relevant capacity")
    save_figure(fig,dirs["fig_main"] / "Figure_2_intrinsic_composition_response",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_2_source_data",SAVE_MODES[config.save_mode])


def figure_3(dirs,config,atlas_payload):
    d=atlas_payload["fixed"].copy(); s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.2,7.7),constrained_layout=True)
    if d.empty: [empty_axis(ax,"No frozen-process response") for ax in axs.flat]
    else:
        x=100*d["feed_co2_fraction"]
        # A zero/negative recoverable CO2 swing is an infeasible reduced-order state,
        # not a physical prediction of exactly zero product purity. Mask it accordingly.
        valid=np.ones(len(d),dtype=bool)
        if "co2_product_mol_kg_cycle" in d: valid=pd.to_numeric(d["co2_product_mol_kg_cycle"],errors="coerce").to_numpy(float)>EPS
        pur=pd.to_numeric(d["purity_estimate"],errors="coerce").where(valid,np.nan); rec=pd.to_numeric(d["recovery_estimate"],errors="coerce").where(valid,np.nan)
        axs[0,0].plot(x,100*pur); axs[0,0].axhline(100*s.target_purity,ls="--",lw=1); axs[0,0].set_ylabel("CO$_2$ purity estimate (%)")
        axs[0,1].plot(x,100*rec); axs[0,1].axhline(100*s.target_recovery,ls="--",lw=1); axs[0,1].set_ylabel("CO$_2$ recovery estimate (%)")
        nom=d.iloc[(d["feed_co2_fraction"]-s.nominal_feed_co2_fraction).abs().argsort()[:1]].iloc[0]
        prod=pd.to_numeric(d["productivity_mol_kg_h"],errors="coerce").where(valid,np.nan); work=pd.to_numeric(d["specific_vacuum_work_kJ_mol_CO2"],errors="coerce").where(valid,np.nan)
        axs[1,0].plot(x,prod/max(float(nom["productivity_mol_kg_h"]),EPS)); axs[1,0].axhline(1,ls="--",lw=.8); axs[1,0].set_ylabel("Productivity / nominal")
        axs[1,1].plot(x,work); axs[1,1].set_ylabel("Idealized evacuation-work proxy (kJ mol$^{-1}$ CO$_2$)")
        if (~valid).any():
            xmax=float(np.nanmax(x[~valid]));
            for ax in axs.flat: ax.axvspan(float(np.nanmin(x)),xmax,alpha=.035,hatch="///",label="no positive frozen CO$_2$ swing")
        for i,ax in enumerate(axs.flat): ax.set_xlabel("Feed CO$_2$ (%)"); ax.grid(alpha=.18); ax.axvspan(100*s.primary_evidence_feed_min,100*s.primary_evidence_feed_max,alpha=.06); ax.axvline(100*s.nominal_feed_co2_fraction,ls="--",lw=.8); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Frozen process: purity"); axs[0,1].set_title("Frozen process: recovery"); axs[1,0].set_title("Frozen process: productivity"); axs[1,1].set_title("Frozen process: resource burden")
    save_figure(fig,dirs["fig_main"] / "Figure_3_offdesign_fixed_process",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_3_source_data",SAVE_MODES[config.save_mode])


def figure_4(dirs,config,atlas_payload):
    d=atlas_payload["selected"]; s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.2,7.7),constrained_layout=True)
    if d.empty: [empty_axis(ax,"No intervention results") for ax in axs.flat]
    else:
        d=d.copy(); failed=~d["success"].astype(bool); d.loc[failed,["p_reg_bar","timing_scale","specific_vacuum_work_kJ_mol_CO2","productivity_mol_kg_h"]]=np.nan
        x=100*d["feed_co2_fraction"]; axs[0,0].plot(x,d["p_reg_bar"]); axs[0,0].axhline(s.nominal_regeneration_pressure_bar,ls="--",lw=.8); axs[0,0].set_yscale("log"); axs[0,0].set_ylabel("Required regeneration pressure (bar)")
        axs[0,1].plot(x,d["timing_scale"]); axs[0,1].axhline(1,ls="--",lw=.8); axs[0,1].set_ylabel("Adsorption-time scale")
        fixed=atlas_payload["fixed"]; base=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"]); axs[1,0].plot(x,d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(base,EPS)); axs[1,0].axhline(1,ls="--",lw=.8); axs[1,0].set_ylabel("Adapted / frozen vacuum-work proxy")
        basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"]); axs[1,1].plot(x,d["productivity_mol_kg_h"]/np.maximum(basep,EPS)); axs[1,1].axhline(1,ls="--",lw=.8); axs[1,1].set_ylabel("Adapted / frozen productivity")
        for i,ax in enumerate(axs.flat): ax.set_xlabel("Feed CO$_2$ (%)"); ax.grid(alpha=.18); ax.axvspan(100*s.primary_evidence_feed_min,100*s.primary_evidence_feed_max,alpha=.06); ax.axvline(100*s.nominal_feed_co2_fraction,ls="--",lw=.8); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Regeneration-pressure intervention"); axs[0,1].set_title("Adsorption-timing intervention"); axs[1,0].set_title("Resource consequence"); axs[1,1].set_title("Productivity consequence")
    save_figure(fig,dirs["fig_main"] / "Figure_4_minimum_operating_intervention",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_4_source_data",SAVE_MODES[config.save_mode])


def _heatmap(ax,df,value,title,cbar_label):
    if df.empty: empty_axis(ax,"No atlas data"); return
    piv=df.pivot_table(index="p_reg_bar",columns="feed_co2_fraction",values=value,aggfunc="mean").sort_index()
    im=ax.imshow(piv.to_numpy(),origin="lower",aspect="auto",extent=[100*piv.columns.min(),100*piv.columns.max(),piv.index.min(),piv.index.max()]); ax.set_yscale("log"); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Regeneration pressure (bar)"); ax.set_title(title); plt.colorbar(im,ax=ax,label=cbar_label)


def figure_5(dirs,config,atlas_payload,robustness_payload):
    sel=atlas_payload.get("selected",pd.DataFrame()); win=atlas_payload.get("windows",pd.DataFrame()); cl=robustness_payload.get("classification",pd.DataFrame()); bounds=robustness_payload.get("boundaries",pd.DataFrame())
    fig,axs=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
    ax=axs[0,0]
    if sel.empty: empty_axis(ax,"No minimum-intervention classification")
    else:
        x=100*sel["feed_co2_fraction"]; ax.step(x,sel["adaptation_level"].clip(upper=3),where="mid"); ax.set_yticks([0,1,2,3],["frozen","one knob","two knobs","outside normal envelope"]); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_title("Minimum operating intervention")
    ax=axs[0,1]
    if cl.empty: empty_axis(ax,"No uncertainty classification")
    else:
        x=100*cl["feed_co2_fraction"]
        ax.stackplot(x,cl["p_robust_frozen"],cl["p_single_knob_recoverable"],cl["p_coordinated_recoverable"],cl["p_outside_control_envelope"],labels=["frozen","one knob","two knobs","outside"])
        ax.set_ylim(0,1); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Probability"); ax.set_title("Uncertainty-aware intervention class"); ax.legend(frameon=False,fontsize=7)
    ax=axs[1,0]
    if win.empty: empty_axis(ax,"No nested intervention windows")
    else:
        yloc=np.arange(len(win)); lo=100*win["lower_feed_co2_fraction"].to_numpy(float); hi=100*win["upper_feed_co2_fraction"].to_numpy(float)
        for i,(a,b) in enumerate(zip(lo,hi)):
            if not (np.isfinite(a) and np.isfinite(b)): continue
            ax.plot([a,b],[i,i],lw=5,solid_capstyle="butt")
            ls=str(win.iloc[i].get("lower_boundary_status","")); us=str(win.iloc[i].get("upper_boundary_status",""))
            if "left_censored" in ls: ax.plot(a,i,marker="<",ms=8)
            if "right_censored" in us: ax.plot(b,i,marker=">",ms=8)
        ax.axvline(100*config.engineering.nominal_feed_co2_fraction,ls="--",lw=.8); ax.set_yticks(yloc,win["intervention_ceiling"].str.replace("_"," ")); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_title("Nested windows (arrows = domain-censored edge)")
    ax=axs[1,1]
    if bounds.empty: empty_axis(ax,"No uncertainty boundary intervals")
    else:
        b=bounds.copy(); y=np.arange(len(b)); med=100*b["median_feed_co2_fraction"].to_numpy(float); lo=100*b["q025"].to_numpy(float); hi=100*b["q975"].to_numpy(float)
        for j,r in b.reset_index(drop=True).iterrows():
            if not np.isfinite(med[j]): continue
            status=str(r.get("boundary_status",""))
            left=max(med[j]-lo[j],0) if np.isfinite(lo[j]) else 0
            right=max(hi[j]-med[j],0) if np.isfinite(hi[j]) else 0
            marker="o"
            if "left_censored" in status: marker="<"
            elif "right_censored" in status: marker=">"
            elif status=="insufficient_valid_draw_support": marker="x"
            ax.errorbar([med[j]],[j],xerr=np.array([[left],[right]]),fmt=marker,capsize=3)
        ax.set_yticks(y,b["boundary"].str.replace("_"," "),fontsize=7); ax.set_xlabel("Feed CO$_2$ boundary (%)"); ax.set_title("Boundary uncertainty: o resolved, </> censored, x low support")
    for i,ax in enumerate(axs.flat):
        panel_label(ax,"abcd"[i]); ax.grid(alpha=.15)
    save_figure(fig,dirs["fig_main"] / "Figure_5_nested_feed_composition_robustness_boundaries",SAVE_MODES[config.save_mode]); save_panel_source(sel,dirs["source_main"] / "Figure_5_selected_intervention_source_data",SAVE_MODES[config.save_mode]); save_panel_source(cl,dirs["source_main"] / "Figure_5_uncertainty_source_data",SAVE_MODES[config.save_mode]); save_panel_source(bounds,dirs["source_main"] / "Figure_5_boundary_source_data",SAVE_MODES[config.save_mode])


def make_si_figures(dirs,config,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload):
    if not SAVE_MODES[config.save_mode]["si_figures"]: return
    # S1 raw equilibrium overlays
    data=harmonized_payload["data"]; fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if data.empty: empty_axis(ax,"No equilibrium data")
    else:
        for (gas,T),g in data[data["material"].eq("CALF-20")].groupby(["adsorbate","temperature_K"]): ax.scatter(g["pressure_bar"],g["loading_mol_kg"],s=12,label=f"{gas}, {T:g} K")
        ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Loading (mol kg$^{-1}$)"); ax.set_title("Figure S1. Raw CALF-20 equilibrium overlays"); ax.legend(frameon=False,fontsize=6,ncol=2)
    save_figure(fig,dirs["fig_si"] / "Figure_S1_raw_source_overlays",SAVE_MODES[config.save_mode])
    # S2 residuals
    r=fit_payload.get("residuals",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if r.empty: empty_axis(ax,"No fit residuals")
    else:
        rr=r[r["material"].eq("CALF-20")]; ax.scatter(rr["predicted_mol_kg"],rr["residual_mol_kg"],s=12,alpha=.6); ax.axhline(0,ls="--",lw=.8); ax.set_xlabel("Predicted loading"); ax.set_ylabel("Residual (mol kg$^{-1}$)"); ax.set_title("Figure S2. Isotherm-fit residuals")
    save_figure(fig,dirs["fig_si"] / "Figure_S2_fit_residuals",SAVE_MODES[config.save_mode])
    # S3 model comparison
    ms=fit_payload.get("summary",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if ms.empty: empty_axis(ax,"No model-comparison data")
    else:
        mm=ms[ms["material"].eq("CALF-20")]; labels=mm["adsorbate"].astype(str)+" | "+mm["model"].astype(str); ax.barh(np.arange(len(mm)),mm["aicc"]); ax.set_yticks(np.arange(len(mm)),labels,fontsize=7); ax.set_xlabel("AICc"); ax.set_title("Figure S3. Alternative isotherm models")
    save_figure(fig,dirs["fig_si"] / "Figure_S3_alternative_isotherms",SAVE_MODES[config.save_mode])
    # S4 bootstrap distributions from summaries
    bs=uncertainty_payload.get("summary",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if bs.empty: empty_axis(ax,"No bootstrap summary")
    else:
        bb=bs[(bs["material"].eq("CALF-20"))].head(18); x=np.arange(len(bb)); ax.errorbar(x,bb["median"],yerr=[bb["median"]-bb["q025"],bb["q975"]-bb["median"]],fmt="o",capsize=2); ax.set_xticks(x,(bb["adsorbate"]+" "+bb["parameter"]).tolist(),rotation=55,ha="right",fontsize=7); ax.set_title("Figure S4. Bootstrap parameter intervals")
    save_figure(fig,dirs["fig_si"] / "Figure_S4_bootstrap_distributions",SAVE_MODES[config.save_mode])
    # S5 support/extrapolation map
    intr=atlas_payload["intrinsic"]; fig,ax=plt.subplots(figsize=(8.6,4.5),constrained_layout=True)
    if intr.empty: empty_axis(ax,"No composition support map")
    else:
        codes={
            "direct_breakthrough_anchor":0,
            "interpolation_between_breakthrough_anchors":1,
            "literature_composition_anchor_no_trace":2,
            "literature_anchor_envelope_no_trace":3,
            "modest_extrapolation_from_breakthrough_anchors":4,
            "exploratory_extended_envelope":5,
            "severe_extrapolation_from_breakthrough_anchors":6,
            "outside_declared_composition_envelope":7,
        }
        z=intr["composition_support"].map(codes).fillna(7)
        ax.scatter(100*intr["feed_co2_fraction"],np.zeros(len(intr)),c=z,s=35)
        ax.axvspan(100*config.engineering.primary_evidence_feed_min,100*config.engineering.primary_evidence_feed_max,alpha=.06)
        ax.set_yticks([]); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_title("Figure S5. Composition support / extrapolation audit")
    save_figure(fig,dirs["fig_si"] / "Figure_S5_extrapolation_map",SAVE_MODES[config.save_mode])
    # S6 engineering sensitivity: model form
    mf=validation_payload.get("model_sensitivity",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if mf.empty: empty_axis(ax,"No model-form sensitivity")
    else:
        for key,g in mf.groupby(["co2_model","n2_model"]): ax.plot(100*g["feed_co2_fraction"],g["recovery_estimate"],marker="o",label=f"{key[0]} / {key[1]}")
        ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Recovery estimate"); ax.set_title("Figure S6. Model-form sensitivity"); ax.legend(frameon=False,fontsize=6)
    save_figure(fig,dirs["fig_si"] / "Figure_S6_engineering_sensitivity",SAVE_MODES[config.save_mode])
    # S7 leave-one-source/temperature validation
    bv=validation_payload.get("blocked",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if bv.empty: empty_axis(ax,"Blocked validation disabled/unavailable")
    else:
        ax.bar(np.arange(len(bv)),bv["rmse_mol_kg"]); ax.set_xlabel("Blocked validation case"); ax.set_ylabel("RMSE (mol kg$^{-1}$)"); ax.set_title("Figure S7. Leave-one-condition/source-out validation")
    save_figure(fig,dirs["fig_si"] / "Figure_S7_leave_one_source_out",SAVE_MODES[config.save_mode])
    # S8 uncertainty classes
    cl=robustness_payload.get("classification",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if cl.empty: empty_axis(ax,"No uncertainty classification")
    else:
        ax.stackplot(100*cl["feed_co2_fraction"],cl["p_robust_frozen"],cl["p_single_knob_recoverable"],cl["p_coordinated_recoverable"],cl["p_outside_control_envelope"],labels=["frozen","one knob","two knobs","outside"]); ax.set_ylim(0,1); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Probability"); ax.set_title("Figure S8. Uncertainty classification"); ax.legend(frameon=False)
    save_figure(fig,dirs["fig_si"] / "Figure_S8_uncertainty_classification",SAVE_MODES[config.save_mode])
    # S9 full-domain adaptation level
    sel=atlas_payload["selected"]; fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if sel.empty: empty_axis(ax,"No adaptation map")
    else:
        ax.step(100*sel["feed_co2_fraction"],sel["adaptation_level"],where="mid"); ax.set_yticks([0,1,2,3],["frozen","one knob","two knobs","outside normal envelope"]); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_title("Figure S9. Full-domain minimum intervention class")
    save_figure(fig,dirs["fig_si"] / "Figure_S9_full_domain_map",SAVE_MODES[config.save_mode])
    # S10 numerical verification
    nv=validation_payload.get("numerical",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    if nv.empty: empty_axis(ax,"No numerical verification")
    else:
        for n,g in nv.groupby("pressure_grid_n"): ax.plot(100*g["feed_co2_fraction"],g["largest_feasible_p_reg_bar"],marker="o",label=f"n={n}")
        ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Largest feasible $P_{reg}$ (bar)"); ax.set_title("Figure S10. Pressure-grid convergence"); ax.legend(frameon=False)
    save_figure(fig,dirs["fig_si"] / "Figure_S10_numerical_verification",SAVE_MODES[config.save_mode])
    # S11 audited input tables
    au=audit_payload.get("audit",pd.DataFrame()); fig,ax=plt.subplots(figsize=(9,5.4),constrained_layout=True)
    if au.empty: empty_axis(ax,"No input-file audit records")
    else:
        aa=au.copy(); aa["label"]=aa["relative_path"].astype(str)+" | "+aa["subtable"].astype(str); ax.barh(np.arange(len(aa)),aa["rows"]); ax.set_yticks(np.arange(len(aa)),aa["label"],fontsize=6); ax.set_xlabel("Rows"); ax.set_title("Figure S11. Source-data audit")
    if not au.empty: save_figure(fig,dirs["fig_si"] / "Figure_S11_source_data_validation",SAVE_MODES[config.save_mode])
    else: plt.close(fig)
    # S12 optional breakthrough dynamic validation
    bt=validation_payload.get("breakthrough",pd.DataFrame()); fig,ax=plt.subplots(figsize=(8.6,5.4),constrained_layout=True)
    gg=bt[bt["validation_type"].eq("leave_one_composition_out")] if len(bt) else pd.DataFrame()
    if gg.empty: empty_axis(ax,"Optional dynamic validation appears when >=2 breakthrough anchors are supplied")
    else:
        ax.scatter(100*gg["feed_co2_fraction"],gg["residual_s"],s=35); ax.axhline(0,ls="--",lw=.8); ax.set_xlabel("Held-out feed CO$_2$ (%)"); ax.set_ylabel("Predicted - observed $t_{50}$ (s)"); ax.set_title("Figure S12. Leave-one-composition-out breakthrough check")
    if not gg.empty: save_figure(fig,dirs["fig_si"] / "Figure_S12_optional_dynamic_validation",SAVE_MODES[config.save_mode])
    else: plt.close(fig)
    # S13 target-threshold sensitivity
    tw=robustness_payload.get("threshold_windows",pd.DataFrame()); fig,ax=plt.subplots(figsize=(9.0,5.6),constrained_layout=True)
    if tw.empty:
        empty_axis(ax,"No target-threshold sensitivity results")
    else:
        g=tw[tw["intervention_ceiling"].eq("coordinated_two_knob_or_less")].copy()
        for rt,gg in g.groupby("threshold_recovery"):
            gg=gg.sort_values("threshold_purity")
            ax.plot(100*gg["threshold_purity"],100*gg["lower_feed_co2_fraction"],marker="o",label=f"recovery ≥ {100*rt:.0f}%")
        ax.set_xlabel("Purity target (%)"); ax.set_ylabel("Lean-side lower boundary (feed CO$_2$ %)")
        ax.set_title("Figure S13. Target sensitivity of coordinated-control lower boundary"); ax.legend(frameon=False)
    save_figure(fig,dirs["fig_si"] / "Figure_S13_target_threshold_sensitivity",SAVE_MODES[config.save_mode])
    # S14 recoverable versus residual off-design deficits
    dec=atlas_payload.get("decomposition",pd.DataFrame()); fig,axs=plt.subplots(1,2,figsize=(10.2,4.8),constrained_layout=True)
    if dec.empty:
        [empty_axis(ax,"No adaptation-recovery decomposition") for ax in axs]
    else:
        x=100*dec["feed_co2_fraction"]
        axs[0].plot(x,100*dec["purity_deficit_frozen"],label="frozen deficit")
        axs[0].plot(x,100*dec["purity_deficit_residual"],label="residual after adaptation")
        axs[0].set_xlabel("Feed CO$_2$ (%)"); axs[0].set_ylabel("Purity target deficit (percentage points)"); axs[0].legend(frameon=False)
        axs[1].plot(x,100*dec["recovery_deficit_frozen"],label="frozen deficit")
        axs[1].plot(x,100*dec["recovery_deficit_residual"],label="residual after adaptation")
        axs[1].set_xlabel("Feed CO$_2$ (%)"); axs[1].set_ylabel("Recovery target deficit (percentage points)"); axs[1].legend(frameon=False)
        for ax in axs:
            ax.axvspan(100*config.engineering.primary_evidence_feed_min,100*config.engineering.primary_evidence_feed_max,alpha=.06)
            ax.axvline(100*config.engineering.nominal_feed_co2_fraction,ls="--",lw=.8); ax.grid(alpha=.15)
        axs[0].set_title("Purity: recoverable vs residual limitation"); axs[1].set_title("Recovery: recoverable vs residual limitation")
    save_figure(fig,dirs["fig_si"] / "Figure_S14_adaptation_recovery_decomposition",SAVE_MODES[config.save_mode])



# ------------------------- publication graphics -------------------------
PUB_COLORS={"frozen":"#315B7D","single":"#D89B3C","coordinated":"#3F8C7A","outside":"#B45A55","indeterminate":"#B7B7B7","data":"#404040","accent":"#5A4E8C"}

def _decorate_composition_axis(ax,s: EngineeringSettings):
    ax.axvspan(100*s.primary_evidence_feed_min,100*s.primary_evidence_feed_max,color="#DCE6EC",alpha=.28,zorder=-10)
    ax.axvline(100*s.nominal_feed_co2_fraction,color="#333333",ls="--",lw=.9,zorder=1)
    ax.grid(alpha=.12,lw=.6)
    ax.set_xlabel("Feed CO$_2$ (%)")

def _evidence_landscape(ax,evidence_payload,validation_payload,s):
    pts=[]
    ca=evidence_payload.get("composition_anchors",pd.DataFrame())
    if len(ca):
        for _,r in ca.iterrows(): pts.append((100*float(r["feed_co2_fraction"]),1,"breakthrough / composition anchor"))
    pa=evidence_payload.get("process_anchors",pd.DataFrame())
    if len(pa):
        for _,r in pa.iterrows():
            if np.isfinite(pd.to_numeric(pd.Series([r.get("feed_co2_fraction")]),errors="coerce").iloc[0]): pts.append((100*float(r["feed_co2_fraction"]),2,"laboratory process anchor"))
    pv=validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame())
    if len(pv):
        for y in sorted(pv["feed_co2_fraction"].dropna().unique()): pts.append((100*float(y),3,"independent pilot PVSA"))
    if not pts: return empty_axis(ax,"No composition-resolved external evidence loaded")
    q=pd.DataFrame(pts,columns=["x","y","role"])
    markers={1:"o",2:"s",3:"D"}; labels={1:"breakthrough condition",2:"lab VSA anchor",3:"pilot PVSA evidence"}
    for yy,g in q.groupby("y"):
        ax.scatter(g["x"],np.repeat(yy,len(g)),marker=markers[int(yy)],s=45,label=labels[int(yy)],zorder=4)
    ax.set_yticks([1,2,3],["Breakthrough / composition","Laboratory VSA","Pilot PVSA"],fontsize=7); ax.set_ylim(.5,3.5); _decorate_composition_axis(ax,s); ax.set_title("Composition-resolved evidence landscape"); ax.legend(frameon=False,fontsize=6.5,loc="lower right")


def figure_1(dirs,config,harmonized_payload,fit_payload,evidence_payload,validation_payload):
    s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.5,7.8),constrained_layout=True)
    src=[]
    _evidence_landscape(axs[0,0],evidence_payload,validation_payload,s); panel_label(axs[0,0],"a")
    # compact equilibrium panel: data + fits at selected temperatures
    data=harmonized_payload.get("data",pd.DataFrame()); ax=axs[0,1]
    if data.empty: empty_axis(ax,"No equilibrium data")
    else:
        for gas,ls in [("CO2","-"),("N2","--")]:
            g=data[(data["material"]=="CALF-20")&(data["adsorbate"]==gas)]
            if g.empty: continue
            fs=fit_payload["fits"].get(("CALF-20",gas),{}); m=fs.get("best_model")
            if not m: continue
            th=fs["models"][m]["theta"]
            Ts=sorted(g["temperature_K"].unique()); Ts=[Ts[0],Ts[len(Ts)//2],Ts[-1]] if len(Ts)>=3 else Ts
            for T in Ts:
                gg=g[np.isclose(g["temperature_K"],T)]; ax.scatter(gg["pressure_bar"],gg["loading_mol_kg"],s=10,alpha=.55)
                pp=np.geomspace(max(float(gg["pressure_bar"].min()),1e-4),float(gg["pressure_bar"].max()),150); ax.plot(pp,model_loading(m,pp,np.full_like(pp,T),th),ls=ls,lw=1.1,label=f"{gas}, {T:g} K")
        ax.set_xscale("log"); ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Loading (mol kg$^{-1}$)"); ax.set_title("Thermodynamic foundation"); ax.grid(alpha=.12); ax.legend(frameon=False,fontsize=6,ncol=2); panel_label(ax,"b")
    # external pilot composition-response summary when available; the full cloud stays in SI.
    ax=axs[1,0]; pv=validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame())
    if len(pv):
        rows=[]
        for y,g in pv.groupby("feed_co2_fraction"):
            rr=pd.to_numeric(g["recovery_fraction"],errors="coerce").dropna()
            if len(rr): rows.append((100*float(y),100*float(rr.median()),100*float(rr.quantile(.25)),100*float(rr.quantile(.75)),len(rr)))
        rows=sorted(rows)
        xx=np.array([r[0] for r in rows]); med=np.array([r[1] for r in rows]); q1=np.array([r[2] for r in rows]); q3=np.array([r[3] for r in rows])
        ax.errorbar(xx,med,yerr=np.vstack([med-q1,q3-med]),fmt="o-",capsize=4,lw=1.5)
        for x0,m0,_,_,n0 in rows: ax.text(x0,m0+1.6,f"n={n0}",ha="center",fontsize=7)
        ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Pilot recovery median (%, IQR)"); ax.set_title("Independent pilot-scale composition response"); ax.grid(alpha=.12)
    else: empty_axis(ax,"Independent pilot-scale dataset not yet supplied\n(optional; pipeline remains runnable)")
    panel_label(ax,"c")
    # conceptual intervention ladder using computed windows
    ax=axs[1,1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    boxes=[(.03,.58,.20,.20,"Frozen\nprocess",PUB_COLORS["frozen"]),(.29,.58,.20,.20,"One\nactuator",PUB_COLORS["single"]),(.55,.58,.20,.20,"Two coordinated\nactuators",PUB_COLORS["coordinated"]),(.81,.58,.16,.20,"Outside normal\ncontrol",PUB_COLORS["outside"])]
    for x,y,w,h,txt,c in boxes:
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc=c,ec="none",alpha=.16); ax.add_patch(patch); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9)
    for a,b in [(.23,.29),(.49,.55),(.75,.81)]: ax.annotate("",xy=(b,.68),xytext=(a,.68),arrowprops=dict(arrowstyle="->",lw=1.2,color="#555555"))
    ax.text(.5,.34,"Question: how many operating degrees of freedom are required\nto preserve fixed purity/recovery constraints as feed composition drifts?",ha="center",va="center",fontsize=9)
    ax.text(.5,.14,r"$A(y)=\min\{\dim(u):\;P(y,u)\geq P^*,\;R(y,u)\geq R^*\}$",ha="center",va="center",fontsize=10)
    ax.set_title("Minimum-control-authority framework"); panel_label(ax,"d")
    save_figure(fig,dirs["fig_main"] / "Figure_1_evidence_and_control_authority_framework",SAVE_MODES[config.save_mode])
    if len(pv): src.append(pv)
    save_panel_source(pd.concat(src,ignore_index=True,sort=False) if src else pd.DataFrame(),dirs["source_main"] / "Figure_1_source_data",SAVE_MODES[config.save_mode])


def figure_2(dirs,config,atlas_payload):
    s=config.engineering; d=atlas_payload.get("mechanism",pd.DataFrame()); intrinsic=atlas_payload.get("intrinsic",pd.DataFrame()); fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty or intrinsic.empty: [empty_axis(ax,"No mechanism data") for ax in axs.flat]
    else:
        x=100*intrinsic["feed_co2_fraction"]
        axs[0,0].plot(x,intrinsic["q_co2_mol_kg"],label="CO$_2$"); axs[0,0].plot(x,intrinsic["q_n2_mol_kg"],label="N$_2$"); axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)"); axs[0,0].legend(frameon=False)
        if "adsorbed_x_co2" in intrinsic:
            axs[0,1].plot(x,intrinsic["adsorbed_x_co2"]); axs[0,1].plot(x,intrinsic["feed_co2_fraction"],ls="--",lw=.8,color="#555555"); axs[0,1].text(.55,.10,r"$x_{CO_2}=y_{CO_2}$ (no enrichment)",transform=axs[0,1].transAxes,fontsize=7); axs[0,1].set_ylabel("Adsorbed-phase CO$_2$ fraction")
        else: axs[0,1].plot(x,intrinsic.get("iast_selectivity_co2_n2",np.nan)); axs[0,1].set_ylabel("IAST selectivity")
        xd=100*d["feed_co2_fraction"]; axs[1,0].plot(xd,d["working_capacity_co2_mol_kg"],label="CO$_2$ working capacity",lw=1.7); axs[1,0].set_ylabel("CO$_2$ working capacity (mol kg$^{-1}$)")
        axr=axs[1,0].twinx(); contam=d["working_capacity_n2_mol_kg"]/np.maximum(d["working_capacity_co2_mol_kg"],EPS); axr.plot(xd,contam,ls="--",lw=1.4,label="N$_2$/CO$_2$ capacity ratio"); axr.set_ylabel("N$_2$ carryover / CO$_2$ working capacity")
        lines=axs[1,0].get_lines()+axr.get_lines(); axs[1,0].legend(lines,[z.get_label() for z in lines],frameon=False,fontsize=7,loc="best")
        axs[1,1].plot(xd,d["thermodynamic_pressure_ratio"],lw=1.7); axs[1,1].axhline(1,ls="--",lw=.9,color="#333333"); axs[1,1].fill_between(xd,np.maximum(d["thermodynamic_pressure_ratio"],1e-6),1,where=d["thermodynamic_pressure_ratio"]<1,alpha=.12); axs[1,1].set_yscale("log"); axs[1,1].set_ylabel(r"$\Pi_{reg}=y_{CO_2}P_{ads}/(y_{prod}P_{reg})$"); axs[1,1].text(.03,.08,r"$\Pi_{reg}<1$: weak regeneration driving force",transform=axs[1,1].transAxes,fontsize=7)
        titles=["Competitive mixture loading","Adsorbed-phase enrichment","Capacity and contamination competition","Regeneration driving-force indicator"]
        for i,ax in enumerate(axs.flat): _decorate_composition_axis(ax,s); ax.set_title(titles[i]); panel_label(ax,"abcd"[i])
    save_figure(fig,dirs["fig_main"] / "Figure_2_mechanism_and_constraint_switching",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_2_source_data",SAVE_MODES[config.save_mode])


def figure_3(dirs,config,atlas_payload):
    d=atlas_payload.get("fixed",pd.DataFrame()).copy(); mech=atlas_payload.get("mechanism",pd.DataFrame()); s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty: [empty_axis(ax,"No frozen-process response") for ax in axs.flat]
    else:
        x=100*d["feed_co2_fraction"]
        valid=(d["working_capacity_co2_mol_kg"]>1e-8)&np.isfinite(d["specific_vacuum_work_kJ_mol_CO2"])
        win=atlas_payload.get("windows",pd.DataFrame()); flo=fhi=np.nan
        if len(win):
            ww=win[win["intervention_ceiling"].astype(str).eq("frozen_safe")]
            if len(ww): flo=float(ww.iloc[0]["lower_feed_co2_fraction"]); fhi=float(ww.iloc[0]["upper_feed_co2_fraction"])
        for ax in axs.flat:
            if (~valid).any(): ax.axvspan(float(x[~valid].min()),float(x[~valid].max()),color="#D9D9D9",alpha=.35,hatch="///",zorder=-5)
            if np.isfinite(flo) and np.isfinite(fhi): ax.axvspan(100*flo,100*fhi,color=PUB_COLORS["frozen"],alpha=.10,zorder=-4)
        axs[0,0].plot(x,np.where(valid,100*d["purity_estimate"],np.nan)); axs[0,0].axhline(100*s.target_purity,ls="--",lw=1,color="#333333"); axs[0,0].set_ylabel("CO$_2$ purity (%)")
        axs[0,1].plot(x,np.where(valid,100*d["recovery_estimate"],np.nan)); axs[0,1].axhline(100*s.target_recovery,ls="--",lw=1,color="#333333"); axs[0,1].set_ylabel("CO$_2$ recovery (%)")
        nom=d.iloc[(d["feed_co2_fraction"]-s.nominal_feed_co2_fraction).abs().argsort()[:1]].iloc[0]
        axs[1,0].plot(x,np.where(valid,d["productivity_mol_kg_h"]/max(nom["productivity_mol_kg_h"],EPS),np.nan)); axs[1,0].axhline(1,ls="--",lw=.8,color="#333333"); axs[1,0].set_ylabel("Productivity / nominal")
        axs[1,1].plot(x,np.where(valid,d["specific_vacuum_work_kJ_mol_CO2"],np.nan)); axs[1,1].set_ylabel("Idealized evacuation-work proxy\n(kJ mol$^{-1}$ CO$_2$)")
        for i,ax in enumerate(axs.flat): _decorate_composition_axis(ax,s); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Frozen process: purity constraint"); axs[0,1].set_title("Frozen process: recovery constraint"); axs[1,0].set_title("Frozen process: productivity"); axs[1,1].set_title("Frozen process: resource burden")
        if len(mech):
            # Directly label dominant failure mode near the two sides when available.
            pur=mech[mech["active_constraint"]=="purity_limited"]; rec=mech[mech["active_constraint"]=="recovery_limited"]
            if len(pur): axs[0,0].text(.03,.08,"dilution-side\npurity limitation",transform=axs[0,0].transAxes,fontsize=8)
            if len(rec): axs[0,1].text(.70,.08,"rich-side\nrecovery limitation",transform=axs[0,1].transAxes,fontsize=8,ha="center")
        if np.isfinite(flo) and np.isfinite(fhi): axs[0,0].text(.52,.88,f"frozen-safe: {100*flo:.2f}–{100*fhi:.2f}%",transform=axs[0,0].transAxes,fontsize=7.5,ha="center")
    save_figure(fig,dirs["fig_main"] / "Figure_3_frozen_process_offdesign_failure",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_3_source_data",SAVE_MODES[config.save_mode])


def _class_strip(ax,d):
    """Draw contiguous semantic control-authority blocks, not one stripe per grid point."""
    if d.empty: return
    x=100*pd.to_numeric(d["feed_co2_fraction"],errors="coerce").to_numpy(float)
    lev=pd.to_numeric(d["adaptation_level"],errors="coerce").fillna(3).clip(0,3).to_numpy(int)
    cols=[PUB_COLORS["frozen"],PUB_COLORS["single"],PUB_COLORS["coordinated"],PUB_COLORS["outside"]]
    if len(x)==1: ax.axvspan(x[0]-.2,x[0]+.2,ymin=.94,ymax=1.0,color=cols[lev[0]],alpha=.6,transform=ax.get_xaxis_transform(),clip_on=False); return
    edges=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and lev[j]==lev[i]: j+=1
        ax.axvspan(edges[i],edges[j],ymin=.94,ymax=1.0,color=cols[lev[i]],alpha=.58,transform=ax.get_xaxis_transform(),clip_on=False)
        i=j


def figure_4(dirs,config,atlas_payload):
    d=atlas_payload.get("selected",pd.DataFrame()).copy(); s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty: [empty_axis(ax,"No intervention results") for ax in axs.flat]
    else:
        failed=~d["success"].astype(bool); d.loc[failed,["p_reg_bar","timing_scale","specific_vacuum_work_kJ_mol_CO2","productivity_mol_kg_h"]]=np.nan
        x=100*d["feed_co2_fraction"]
        axs[0,0].plot(x,d["p_reg_bar"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3); axs[0,0].axhline(s.nominal_regeneration_pressure_bar,ls="--",lw=.8,color="#333333"); axs[0,0].set_yscale("log"); axs[0,0].set_ylabel("Required regeneration pressure (bar)")
        axs[0,1].plot(x,d["timing_scale"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3); axs[0,1].axhline(1,ls="--",lw=.8,color="#333333"); axs[0,1].set_ylabel("Required adsorption-time scale")
        fixed=atlas_payload["fixed"]; base=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"]); ratio=100*(d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(base,EPS)-1.0); ratio[~np.isfinite(ratio)]=np.nan; axs[1,0].plot(x,ratio,lw=1.8); axs[1,0].axhline(0,ls="--",lw=.8,color="#333333"); axs[1,0].set_ylabel("Change in evacuation-work proxy (%)")
        basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"]); pr=100*(d["productivity_mol_kg_h"]/np.maximum(basep,EPS)-1.0); pr[~np.isfinite(pr)]=np.nan; axs[1,1].plot(x,pr,lw=1.8); axs[1,1].axhline(0,ls="--",lw=.8,color="#333333"); axs[1,1].set_ylabel("Change in productivity (%)")
        for i,ax in enumerate(axs.flat): _decorate_composition_axis(ax,s); _class_strip(ax,d); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Minimum pressure intervention"); axs[0,1].set_title("Minimum timing intervention"); axs[1,0].set_title("Resource consequence"); axs[1,1].set_title("Productivity consequence")
    save_figure(fig,dirs["fig_main"] / "Figure_4_minimum_intervention_policy",SAVE_MODES[config.save_mode]); save_panel_source(d,dirs["source_main"] / "Figure_4_source_data",SAVE_MODES[config.save_mode])


def figure_5(dirs,config,atlas_payload,robustness_payload):
    sel=atlas_payload.get("selected",pd.DataFrame()); win=atlas_payload.get("windows",pd.DataFrame()); cl=robustness_payload.get("classification",pd.DataFrame()); bounds=robustness_payload.get("boundaries",pd.DataFrame()); s=config.engineering
    fig,axs=plt.subplots(2,2,figsize=(11.6,8.0),constrained_layout=True)
    ax=axs[0,0]
    if sel.empty: empty_axis(ax,"No minimum-intervention classification")
    else:
        x=100*sel["feed_co2_fraction"]; lev=sel["adaptation_level"].clip(upper=3); ax.step(x,lev,where="mid",lw=2.2,color="#333333"); ax.set_yticks([0,1,2,3],["frozen","one actuator","two actuators","outside normal envelope"]); _decorate_composition_axis(ax,s); ax.set_title("Minimum control authority")
        if len(win):
            for _,rr in win.iterrows():
                yy=float(rr.get("lower_feed_co2_fraction",np.nan));
                if np.isfinite(yy): ax.axvline(100*yy,lw=.7,ls=":",color="#555555"); ax.text(100*yy,2.75,f"{100*yy:.2f}%",rotation=90,va="top",ha="right",fontsize=6.5)
    ax=axs[0,1]
    if cl.empty: empty_axis(ax,"No uncertainty classification")
    else:
        x=100*cl["feed_co2_fraction"]; ax.stackplot(x,cl["p_robust_frozen"],cl["p_single_knob_recoverable"],cl["p_coordinated_recoverable"],cl["p_outside_control_envelope"],labels=["frozen","one actuator","two actuators","outside"],colors=[PUB_COLORS["frozen"],PUB_COLORS["single"],PUB_COLORS["coordinated"],PUB_COLORS["outside"]],alpha=.78)
        ax.plot(x,cl["max_class_probability"],color="#222222",lw=1.1,label="max class probability"); ax.axhline(s.class_probability_threshold,color="#222222",ls="--",lw=.8)
        ind=cl["classification"].astype(str).eq("indeterminate").to_numpy();
        if ind.any():
            for xx in x[ind]: ax.axvspan(xx-.15,xx+.15,color=PUB_COLORS["indeterminate"],alpha=.12,hatch="//")
        ax.set_ylim(0,1); ax.set_ylabel("Probability"); _decorate_composition_axis(ax,s); ax.set_xlim(max(100*s.feed_co2_min,9),min(100*s.feed_co2_max,17)); ax.set_title("Calibration-conditioned uncertainty (boundary zoom)"); ax.legend(frameon=False,fontsize=6,ncol=2)
    ax=axs[1,0]
    if win.empty: empty_axis(ax,"No nested intervention windows")
    else:
        yloc=np.arange(len(win)); labels=[]
        for i,r in win.reset_index(drop=True).iterrows():
            a=100*float(r["lower_feed_co2_fraction"]); b=100*float(r["upper_feed_co2_fraction"]); c=[PUB_COLORS["frozen"],PUB_COLORS["single"],PUB_COLORS["coordinated"]][min(i,2)]
            ax.plot([a,b],[i,i],lw=7,solid_capstyle="butt",color=c)
            if "left_censored" in str(r.get("lower_boundary_status","")): ax.plot(a,i,marker="<",ms=8,color=c)
            if "right_censored" in str(r.get("upper_boundary_status","")): ax.plot(b,i,marker=">",ms=8,color=c)
            labels.append(str(r["intervention_ceiling"]).replace("_"," "))
        ax.set_yticks(yloc,labels); _decorate_composition_axis(ax,s); ax.set_title("Nested operability windows")
    ax=axs[1,1]
    if bounds.empty: empty_axis(ax,"No uncertainty boundary intervals")
    else:
        b=bounds.copy(); y=np.arange(len(b)); med=100*b["median_feed_co2_fraction"].to_numpy(float); lo=100*b["q025"].to_numpy(float); hi=100*b["q975"].to_numpy(float)
        for j,r in b.reset_index(drop=True).iterrows():
            if not np.isfinite(med[j]): continue
            status=str(r.get("boundary_status","")); marker="o" if status=="interior_uncertainty_boundary" else ("x" if status=="insufficient_valid_draw_support" else ("<" if "left_censored" in status else ">" if "right_censored" in status else "o"))
            left=max(med[j]-lo[j],0) if np.isfinite(lo[j]) else 0; right=max(hi[j]-med[j],0) if np.isfinite(hi[j]) else 0
            ax.errorbar([med[j]],[j],xerr=np.array([[left],[right]]),fmt=marker,capsize=3,color="#333333")
        ax.set_yticks(y,b["boundary"].str.replace("_"," "),fontsize=7); ax.set_xlabel("Feed CO$_2$ boundary (%)"); ax.set_title("Boundary uncertainty and censoring"); ax.grid(alpha=.12)
    for i,ax in enumerate(axs.flat): panel_label(ax,"abcd"[i])
    save_figure(fig,dirs["fig_main"] / "Figure_5_nested_operability_boundaries",SAVE_MODES[config.save_mode]); save_panel_source(sel,dirs["source_main"] / "Figure_5_selected_intervention_source_data",SAVE_MODES[config.save_mode]); save_panel_source(cl,dirs["source_main"] / "Figure_5_uncertainty_source_data",SAVE_MODES[config.save_mode]); save_panel_source(bounds,dirs["source_main"] / "Figure_5_boundary_source_data",SAVE_MODES[config.save_mode])


def _extra_v2_si_figures(dirs,config,atlas_payload,robustness_payload,validation_payload):
    if not SAVE_MODES[config.save_mode]["si_figures"]: return
    # Target sensitivity as heat maps rather than overlapping curves.
    w=robustness_payload.get("threshold_windows",pd.DataFrame())
    if len(w) and {"threshold_purity","threshold_recovery","intervention_ceiling","lower_feed_co2_fraction","upper_feed_co2_fraction"}.issubset(w.columns):
        fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
        groups=list(w.groupby("intervention_ceiling",sort=False)); vals=[]
        for _,g in groups:
            gg=g.copy(); gg["lower_boundary_pct"]=100*gg["lower_feed_co2_fraction"]; vals.extend(pd.to_numeric(gg["lower_boundary_pct"],errors="coerce").dropna().tolist())
        vmin=min(vals) if vals else 0.0; vmax=max(vals) if vals else 1.0; im=None
        for ax,(name,g) in zip(axs,groups):
            gg=g.copy(); gg["lower_boundary_pct"]=100*gg["lower_feed_co2_fraction"]; piv=gg.pivot_table(index="threshold_recovery",columns="threshold_purity",values="lower_boundary_pct",aggfunc="first").sort_index()
            arr=piv.to_numpy(); im=ax.imshow(arr,origin="lower",aspect="auto",vmin=vmin,vmax=vmax)
            for ii in range(arr.shape[0]):
                for jj in range(arr.shape[1]):
                    if np.isfinite(arr[ii,jj]): ax.text(jj,ii,f"{arr[ii,jj]:.1f}",ha="center",va="center",fontsize=7)
                    else: ax.text(jj,ii,"×",ha="center",va="center",fontsize=9)
            ax.set_xticks(range(len(piv.columns)),[f"{100*x:g}%" for x in piv.columns]); ax.set_yticks(range(len(piv.index)),[f"{100*x:g}%" for x in piv.index]); ax.set_xlabel("Purity target"); ax.set_ylabel("Recovery target"); ax.set_title(str(name).replace("_"," "))
        if im is not None: fig.colorbar(im,ax=axs.ravel().tolist(),label="Lean-side lower boundary (feed CO$_2$ %)",shrink=.9)
        save_figure(fig,dirs["fig_si"] / "Figure_S13b_target_sensitivity_heatmaps",SAVE_MODES[config.save_mode])
    # External pilot evidence summary.
    pv=validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame())
    if len(pv):
        fig,axs=plt.subplots(1,2,figsize=(9.2,3.8),constrained_layout=True)
        for y,g in pv.groupby("feed_co2_fraction"):
            axs[0].scatter(100*g["feed_co2_fraction"],100*g["recovery_fraction"],s=20,alpha=.6,label=f"{100*y:g}%")
            axs[1].scatter(100*g["recovery_fraction"],100*g["purity_fraction"],s=20,alpha=.6,label=f"{100*y:g}%")
        axs[0].set_xlabel("Feed CO$_2$ (%)"); axs[0].set_ylabel("Pilot recovery (%)"); axs[0].set_title("Independent pilot recovery distribution")
        axs[1].axvline(100*config.engineering.target_recovery,ls="--",lw=.8,color="#333333"); axs[1].axhline(100*config.engineering.target_purity,ls="--",lw=.8,color="#333333"); axs[1].set_xlabel("Pilot recovery (%)"); axs[1].set_ylabel("Pilot purity (%)"); axs[1].set_title("Pilot purity-recovery cloud")
        for ax in axs: ax.grid(alpha=.12); ax.legend(frameon=False,fontsize=7)
        save_figure(fig,dirs["fig_si"] / "Figure_S15_external_pilot_validation",SAVE_MODES[config.save_mode])
    src=validation_payload.get("external_source_sensitivity",{}).get("boundaries",pd.DataFrame())
    if len(src):
        fig,ax=plt.subplots(figsize=(8.8,4.6),constrained_layout=True)
        nom=atlas_payload.get("windows",pd.DataFrame()).copy(); labels=[]; yy=[]
        mapping={"frozen_safe":"frozen","single_knob_or_less":"single_or_less","coordinated_two_knob_or_less":"coordinated_or_less"}
        for i,key in enumerate(["frozen","single_or_less","coordinated_or_less"]):
            gg=src[src["intervention_ceiling"].astype(str).eq(key)]
            if gg.empty: continue
            r=gg.iloc[0]; y0=len(yy); yy.append(y0); labels.append(key.replace("_"," "))
            ax.plot(100*float(r["lower_feed_co2_fraction"]),y0,"o",label="independent source" if y0==0 else None)
            nn=nom[nom["intervention_ceiling"].astype(str).map(mapping).eq(key)] if len(nom) else pd.DataFrame()
            if len(nn): ax.plot(100*float(nn.iloc[0]["lower_feed_co2_fraction"]),y0,"s",label="nominal source" if y0==0 else None)
        ax.set_yticks(yy,labels,fontsize=8); ax.set_xlabel("Lean-side lower boundary (feed CO$_2$ %)"); ax.set_title("Independent thermodynamic-source boundary sensitivity"); ax.grid(alpha=.12)
        handles,legend_labels=ax.get_legend_handles_labels()
        if handles: ax.legend(frameon=False,fontsize=7)
        save_figure(fig,dirs["fig_si"] / "Figure_S16_external_thermodynamic_source_sensitivity",SAVE_MODES[config.save_mode])
    stress=validation_payload.get("thermodynamic_discrepancy_stress",{}).get("boundaries",pd.DataFrame())
    # Retain the single-mode plot; the conditioned/unconditioned stress comparison is rendered separately.
    if len(stress) and "conditioning_mode" not in stress.columns:
        fig,ax=plt.subplots(figsize=(9.5,5.1),constrained_layout=True)
        piv=stress.pivot_table(index="scenario_label",columns="intervention_ceiling",values="lower_feed_co2_fraction",aggfunc="first")*100
        cols=[c for c in ["frozen","single_or_less","coordinated_or_less"] if c in piv.columns]
        for c in cols: ax.plot(piv[c].to_numpy(float),np.arange(len(piv)),marker="o",ls="-",label=c.replace("_"," "))
        ax.set_yticks(np.arange(len(piv)),piv.index,fontsize=7); ax.set_xlabel("Lean-side lower boundary (feed CO$_2$ %)"); ax.set_title("Literature discrepancy stress: boundary topology"); ax.grid(alpha=.12); ax.legend(frameon=False,fontsize=7)
        save_figure(fig,dirs["fig_si"] / "Figure_S17_thermodynamic_discrepancy_stress",SAVE_MODES[config.save_mode])


def stage_09_figures(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger):
    configure_matplotlib(); figure_1(dirs,config,harmonized_payload,fit_payload,evidence_payload,validation_payload); figure_2(dirs,config,atlas_payload); figure_3(dirs,config,atlas_payload); figure_4(dirs,config,atlas_payload); figure_5(dirs,config,atlas_payload,robustness_payload)
    make_si_figures(dirs,config,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload)
    _extra_v2_si_figures(dirs,config,atlas_payload,robustness_payload,validation_payload)
    logger.info("Generated five main P10 composite figures%s"," plus SI figures" if SAVE_MODES[config.save_mode]["si_figures"] else "")
    return {"main_figures":sorted(p.name for p in dirs["fig_main"].glob("Figure_*")),"si_figures":sorted(p.name for p in dirs["fig_si"].glob("Figure_*"))}


# ---------------------------------------------------------------------------
# Final reusable bundle
# ---------------------------------------------------------------------------
def stage_10_finalize(config,dirs,input_hash,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,tables_payload,figures_payload,logger):
    manifest={
        "pipeline":"P10 CALF-20 feed-composition operability and minimum control authority","pipeline_version":PIPELINE_VERSION,"created_utc":utc_now_iso(),"input_folder":str(Path(config.input_folder).resolve()),"output_folder":str(Path(config.output_folder).resolve()),"input_fingerprint_sha256":input_hash,
        "configuration":{"save_mode":config.save_mode,"ram_mode":config.ram_mode,"level":config.level,"requested_n_jobs":config.n_jobs,"resolved_n_jobs":resolve_n_jobs(config.n_jobs,config.ram_mode),"seed":config.seed,"engineering":asdict(config.engineering)},
        "environment":{"python":sys.version,"platform":platform.platform(),"packages":package_versions()},
        "scientific_scope_note":"Single-material CALF-20 feed-composition operability framed as minimum control authority. The primary literature-anchored composition envelope is 5--30% CO2; 2--40% remains exploratory. The code distinguishes frozen, one-actuator, coordinated two-actuator and outside-normal-control regimes; identifies purity- versus recovery-limited failure; conditions uncertainty realizations on the nominal experimental anchor; and can ingest independent pilot-scale, alternative-thermodynamic-source and higher-fidelity CSS validation without using those datasets for nominal calibration. Off-design purity/recovery/productivity/energy remain reduced-order estimates unless independently validated. Domain-edge windows are censored, not physical boundaries.",
        "main_figures":figures_payload.get("main_figures",[]),"si_figures":figures_payload.get("si_figures",[]),
    }
    atomic_json(manifest,dirs["final"] / "reproducibility_manifest.json")
    bundle={"manifest":manifest,"audit":audit_payload,"harmonized":harmonized_payload,"fits":fit_payload,"uncertainty_summary":{k:v for k,v in uncertainty_payload.items() if k!="samples"},"composition_evidence":evidence_payload,"feed_composition_robustness":atlas_payload,"robustness":robustness_payload,"validation":validation_payload,"tables":tables_payload}
    if SAVE_MODES[config.save_mode]["bootstrap_samples"]: bundle["uncertainty_samples"]=uncertainty_payload.get("samples",{})
    atomic_pickle(bundle,dirs["final"] / "P10_feed_composition_robustness_complete_results.pkl")
    rows=[]
    for p in sorted(Path(config.output_folder).rglob("*")):
        if p.is_file(): rows.append({"relative_path":str(p.relative_to(Path(config.output_folder))),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    catalog=pd.DataFrame(rows); save_table_multi(catalog,dirs["final"] / "output_file_catalog",SAVE_MODES[config.save_mode])
    write_text(dirs["final"] / "RUN_SUMMARY.txt",f"""P10 feed-composition robustness run completed successfully
================================
Pipeline version: {PIPELINE_VERSION}
Completed UTC: {manifest['created_utc']}
Input fingerprint: {input_hash}
Save mode: {config.save_mode}
RAM mode: {config.ram_mode}
Comprehensiveness: {config.level}
Resolved n_jobs: {manifest['configuration']['resolved_n_jobs']}

Scientific scope:
The 5/95, 15/85 and 30/70 literature conditions define the primary 5--30% CO2
composition envelope.  The wider 2--40% grid is exploratory and is labelled
accordingly.  The 15/85 VSA data may calibrate the nominal point.  Off-design
purity, recovery, productivity and vacuum-work outputs are reduced-order
estimates and should not be presented as full cyclic-steady-state VSA
predictions without independent higher-fidelity validation.  An apparent window edge at
2% or 40% CO2 is domain-censored and must not be reported as a resolved physical
transition.

This small-paper pipeline intentionally does NOT introduce cross-material
source-flexibility, generalist/specialist MOF classes, ML ranking, or universal
structure--property rules; those concepts are reserved for a future flagship
multi-material study.

Key reusable outputs:
- final/P10_feed_composition_robustness_complete_results.pkl
- data/harmonized_adsorption_data.csv
- data/breakthrough_traces_harmonized.csv
- data/composition_condition_anchors.csv
- models/fitted_models_complete.pkl
- engineering/intrinsic_composition_response.csv
- engineering/frozen_process_offdesign.csv
- engineering/intervention_candidates.csv
- engineering/nested_intervention_windows.csv
- engineering/intervention_boundary_diagnostics.csv
- engineering/adaptation_recovery_decomposition.csv
- engineering/mechanism_control_authority_map.csv
- engineering/pressure_response_diagnostic.csv
- uncertainty/feed_composition_robustness_uncertainty_classification.csv
- uncertainty/target_threshold_sensitivity_windows.csv
- validation/boundary_grid_convergence.csv
- validation/boundary_falsification_portfolio.csv
- validation/external_pilot_composition_summary.csv (when supplied)
- validation/external_equilibrium_source_boundary_sensitivity.csv (when supplied)
- validation/css_validation_comparison.csv (when supplied)
- tables/main, tables/SI, figures/main, figures/SI and source_data/
""")
    logger.info("Final P10 feed-composition robustness reusable bundle written")
    return {"manifest":manifest,"catalog":catalog}


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------
def create_input_folder_guidance(input_folder: Path) -> None:
    input_folder.mkdir(parents=True,exist_ok=True)
    # Never pollute an established scientific input directory.  Templates are
    # created only for a new/empty default folder; all schema templates are also
    # excluded from input hashing/discovery as a second safety layer.
    if any(input_folder.iterdir()):
        return
    schema=input_folder/"INPUT_SCHEMA_TEMPLATE.csv"
    if not schema.exists(): pd.DataFrame(columns=["source_id","material","material_form","adsorbate","temperature_K","pressure_bar","loading_mol_kg","uncertainty","doi","table_figure_si_location","extraction_method","allowed_use","notes"]).to_csv(schema,index=False)
    bts=input_folder/"BREAKTHROUGH_SCHEMA_TEMPLATE.csv"
    if not bts.exists(): pd.DataFrame(columns=["source_id","feed_co2_fraction","time_s","outlet_co2_fraction","normalized_outlet","temperature_K","total_pressure_bar","flow_mol_s","bed_mass_g","doi","notes"]).to_csv(bts,index=False)
    pas=input_folder/"PROCESS_ANCHOR_SCHEMA_TEMPLATE.csv"
    if not pas.exists(): pd.DataFrame(columns=["source_id","feed_co2_fraction","purity","recovery","productivity","specific_energy","regeneration_pressure_bar","adsorption_time_s","doi","notes"]).to_csv(pas,index=False)
    readme=input_folder/"README_INPUT_FOLDER.md"
    if not readme.exists(): write_text(readme,"""P10 input_folder
================
Place permitted CALF-20 equilibrium, breakthrough and process-anchor files here.
Raw inputs are never modified.

Preferred equilibrium long format:
source_id, material, material_form, adsorbate, temperature_K, pressure_bar,
loading_mol_kg, uncertainty, doi, table_figure_si_location, extraction_method,
allowed_use, notes

Preferred breakthrough format:
source_id, feed_co2_fraction, time_s, outlet_co2_fraction (or normalized_outlet),
temperature_K, total_pressure_bar, flow_mol_s, bed_mass_g, doi, notes

Preferred metadata-only composition-anchor format:
source_id, feed_co2_fraction, trace_status, figure_location, doi, notes

Preferred nominal-process anchor format:
source_id, feed_co2_fraction, purity, recovery, productivity, specific_energy,
regeneration_pressure_bar, adsorption_time_s, doi, notes

The parser also recognizes many common column-name variants and can infer the
CALF-20 5/95, 15/85 and 30/70 mixtures from filenames.  Percentages may be
entered either as fractions (0.15) or percentages (15).  Keep DOI/provenance.
Do not put results inside input_folder.  Byte-identical duplicate files are
hashed and analyzed only once.
""")



# ===========================================================================
# Publication analysis refinements
# ===========================================================================
# This release deliberately starts from fresh scientific checkpoints.  The
# overrides below implement the final publication-closure recommendations:
# adaptive boundary refinement, explicit single-actuator degeneracy, bounded
# contamination metrics, conditioned/unconditioned literature stress, refined
# falsification planning, and per-panel source-data export.

def _v3_safe_clean_output_folder(path: Path) -> None:
    """Safely remove a designated P10 run folder before a clean run.

    Refuses filesystem roots, the package directory, and the input data tree.
    """
    path=Path(path).resolve()
    script_dir=Path(__file__).resolve().parent
    protected={Path(path.anchor).resolve() if path.anchor else path, script_dir, DATA_ROOT}
    if path in protected or path==path.parent or len(path.parts)<3:
        raise ValueError(f"Refusing to clean unsafe output path: {path}")
    if path.exists():
        shutil.rmtree(path)


# --- One-actuator policy: retain both candidate solutions and expose degeneracy.
def _select_one_actuator_policy(candidates: List[Dict[str,Any]], y: float, settings: EngineeringSettings, relative_tie_tolerance: float=0.05) -> Optional[Dict[str,Any]]:
    if not candidates:
        return None
    cand=[dict(c) for c in candidates]
    for c in cand:
        mv=_control_move_components(c,settings)
        c.update(mv)
        c["control_effort_selector"]=float(mv["control_move_linf"]+1.0e-3*mv["control_move_l1"])
    cand.sort(key=lambda r:(r["control_effort_selector"],r["specific_vacuum_work_kJ_mol_CO2"],-r["productivity_mol_kg_h"]))
    best=cand[0]
    b=float(best["control_effort_selector"])
    tol=max(1.0e-6,relative_tie_tolerance*max(b,1.0e-6))
    near=[c for c in cand if float(c["control_effort_selector"])<=b+tol]
    pressure=next((c for c in cand if str(c.get("adaptation_variant",""))=="pressure_only"),None)
    timing=next((c for c in cand if str(c.get("adaptation_variant",""))=="timing_only"),None)
    pe=float(pressure["control_effort_selector"]) if pressure is not None else np.nan
    te=float(timing["control_effort_selector"]) if timing is not None else np.nan
    degenerate=bool(np.isfinite(pe) and np.isfinite(te) and abs(pe-te)<=max(1.0e-6,relative_tie_tolerance*max(min(pe,te),1.0e-6)))
    preference="pressure_only" if y<settings.nominal_feed_co2_fraction-1e-10 else ("timing_only" if y>settings.nominal_feed_co2_fraction+1e-10 else "")
    chosen=best
    if degenerate and preference:
        pref=[c for c in near if str(c.get("adaptation_variant",""))==preference]
        if pref:
            pref.sort(key=lambda r:(r["control_effort_selector"],r["specific_vacuum_work_kJ_mol_CO2"],-r["productivity_mol_kg_h"]))
            chosen=pref[0]
    chosen=dict(chosen)
    chosen.update({
        "policy_selection_rule":"minimum normalized control effort; mechanism-side tie-break only inside explicit 5% degeneracy band",
        "policy_preferred_actuator":preference or "none",
        "policy_near_tie_count":int(len(near)),
        "policy_effort_best":b,
        "policy_effort_selected":float(chosen["control_effort_selector"]),
        "actuator_choice_degenerate":degenerate,
        "pressure_candidate_feasible":pressure is not None,
        "timing_candidate_feasible":timing is not None,
        "pressure_candidate_effort":pe,
        "timing_candidate_effort":te,
        "actuator_effort_delta_pressure_minus_timing":pe-te if np.isfinite(pe) and np.isfinite(te) else np.nan,
        "pressure_candidate_p_reg_bar":float(pressure.get("p_reg_bar",np.nan)) if pressure is not None else np.nan,
        "pressure_candidate_specific_work":float(pressure.get("specific_vacuum_work_kJ_mol_CO2",np.nan)) if pressure is not None else np.nan,
        "pressure_candidate_productivity":float(pressure.get("productivity_mol_kg_h",np.nan)) if pressure is not None else np.nan,
        "timing_candidate_timing_scale":float(timing.get("timing_scale",np.nan)) if timing is not None else np.nan,
        "timing_candidate_specific_work":float(timing.get("specific_vacuum_work_kJ_mol_CO2",np.nan)) if timing is not None else np.nan,
        "timing_candidate_productivity":float(timing.get("productivity_mol_kg_h",np.nan)) if timing is not None else np.nan,
        "actuator_preference_class":"near_degenerate" if degenerate else ("pressure" if pe<te else "timing" if te<pe else str(chosen.get("adaptation_variant",""))),
    })
    return chosen


def _v3_min_success_level(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 3
    d=df.copy()
    ok=d["success"].astype(bool) if "success" in d else pd.Series(False,index=d.index)
    lev=pd.to_numeric(d.get("adaptation_level",pd.Series(np.nan,index=d.index)),errors="coerce")
    vals=lev[ok & lev.notna()]
    return int(vals.min()) if len(vals) else 3


def _v3_grid_level(y: float, pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int,
                   *, front_multiplier: float=1.0, contamination_multiplier: float=1.0,
                   vacuum_efficiency: Optional[float]=None) -> int:
    d=_solve_adaptation_levels_grid_v130(
        float(y),pair,calibration,front_table,settings,pressure_n,
        front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,
        vacuum_efficiency=vacuum_efficiency
    )
    return _v3_min_success_level(d)


def _v3_transition_bracket(ys: np.ndarray, eligible: np.ndarray, nominal_y: float, side: str) -> Tuple[float,float,str]:
    ys=np.asarray(ys,float); eligible=np.asarray(eligible,bool)
    order=np.argsort(ys); ys=ys[order]; eligible=eligible[order]
    j=int(np.argmin(np.abs(ys-float(nominal_y))))
    if not eligible[j]:
        return np.nan,np.nan,"nominal_not_eligible"
    if side=="lower":
        k=j
        while k>0 and eligible[k-1]:
            k-=1
        if k==0:
            return ys[0],ys[0],"left_censored_by_feed_domain_limit"
        return ys[k-1],ys[k],"interior_transition"
    k=j
    while k<len(ys)-1 and eligible[k+1]:
        k+=1
    if k==len(ys)-1:
        return ys[-1],ys[-1],"right_censored_by_feed_domain_limit"
    return ys[k],ys[k+1],"interior_transition"


def _v3_bisect_boundary(lo: float, hi: float, eligible_fn, side: str, tol_fraction: float=5e-4, max_iter: int=18) -> Tuple[float,float,int]:
    """Bisection with a false/true lower bracket or true/false upper bracket."""
    lo=float(lo); hi=float(hi)
    n=0
    while (hi-lo)>tol_fraction and n<max_iter:
        mid=0.5*(lo+hi); em=bool(eligible_fn(mid)); n+=1
        if side=="lower":
            # expected lo=False, hi=True
            if em: hi=mid
            else: lo=mid
        else:
            # expected lo=True, hi=False
            if em: lo=mid
            else: hi=mid
    boundary=hi if side=="lower" else lo
    return float(boundary),float(hi-lo),n


def _v3_refine_windows(selected: pd.DataFrame, pair, calibration, front_table, settings: EngineeringSettings,
                       pressure_n: int, *, front_multiplier: float=1.0, contamination_multiplier: float=1.0,
                       vacuum_efficiency: Optional[float]=None, tol_fraction: float=5e-4) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if selected is None or selected.empty:
        return pd.DataFrame(),pd.DataFrame()
    d=selected.sort_values("feed_co2_fraction").reset_index(drop=True)
    ys=d["feed_co2_fraction"].to_numpy(float)
    levs=pd.to_numeric(d["adaptation_level"],errors="coerce").fillna(3).to_numpy(int)
    success=d["success"].astype(bool).to_numpy() if "success" in d else (levs<=2)
    rows=[]; audit=[]
    labels={0:"frozen_safe",1:"single_knob_or_less",2:"coordinated_two_knob_or_less"}
    for level,label in labels.items():
        eligible=success & (levs<=level)
        out={"intervention_ceiling":label,"max_adaptation_level":level}
        for side in ("lower","upper"):
            a,b,status=_v3_transition_bracket(ys,eligible,settings.nominal_feed_co2_fraction,side)
            boundary=np.nan; width=np.nan; nit=0
            if status=="interior_transition":
                cache={}
                def efun(y):
                    key=round(float(y),10)
                    if key not in cache:
                        cache[key]=_v3_grid_level(
                            float(y),pair,calibration,front_table,settings,pressure_n,
                            front_multiplier=front_multiplier,contamination_multiplier=contamination_multiplier,
                            vacuum_efficiency=vacuum_efficiency
                        )<=level
                    return cache[key]
                boundary,width,nit=_v3_bisect_boundary(a,b,efun,side,tol_fraction=tol_fraction)
                bstatus="interior_adaptive_transition"
                resolved=True
            else:
                boundary=float(a) if np.isfinite(a) else np.nan
                bstatus=status
                resolved=False
            out[f"{side}_feed_co2_fraction"]=boundary
            out[f"{side}_boundary_status"]=bstatus
            out[f"{side}_boundary_resolved"]=resolved
            out[f"{side}_boundary_method"]="adaptive_bisection" if resolved else "domain_censoring"
            out[f"{side}_boundary_bracket_width_fraction"]=width
            audit.append({
                "intervention_ceiling":label,"max_adaptation_level":level,"side":side,
                "coarse_bracket_a":a,"coarse_bracket_b":b,"refined_boundary_feed_co2_fraction":boundary,
                "final_bracket_width_fraction":width,"iterations":nit,"status":bstatus,
                "tolerance_fraction":tol_fraction,"tolerance_percentage_points":100*tol_fraction,
            })
        lo=out["lower_feed_co2_fraction"]; hi=out["upper_feed_co2_fraction"]
        out["window_width_fraction_observed"]=hi-lo if np.isfinite(lo) and np.isfinite(hi) else np.nan
        out["lower_shift_percentage_points"]=100*(lo-settings.nominal_feed_co2_fraction) if np.isfinite(lo) else np.nan
        out["upper_shift_percentage_points"]=100*(hi-settings.nominal_feed_co2_fraction) if np.isfinite(hi) else np.nan
        rows.append(out)
    return pd.DataFrame(rows),pd.DataFrame(audit)


# Replace coarse headline windows with
# adaptive-bisection boundaries.  The full coarse atlas is still retained.
_stage_05_v230 = stage_05_feed_composition_robustness
def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_05_v230(config,dirs,fit_payload,evidence_payload,logger)
    pair=calf20_pair(fit_payload)
    if pair is not None and not out.get("selected",pd.DataFrame()).empty:
        rw,ra=_v3_refine_windows(
            out["selected"],pair,out.get("calibration",{}),out.get("front_table",pd.DataFrame()),
            config.engineering,int(LEVELS[config.level]["pressure_n"])
        )
        if len(rw):
            out["windows_coarse"]=out.get("windows",pd.DataFrame()).copy()
            out["windows"]=rw
            out["boundary_refinement"]=ra
            save_table_multi(rw,dirs["engineering"]/"nested_intervention_windows_refined",SAVE_MODES[config.save_mode])
            save_table_multi(ra,dirs["engineering"]/"adaptive_boundary_refinement_audit",SAVE_MODES[config.save_mode])
    sel=out.get("selected",pd.DataFrame())
    if len(sel):
        cols=[c for c in [
            "feed_co2_fraction","adaptation_level","adaptation_variant","actuator_choice_degenerate",
            "actuator_preference_class","pressure_candidate_feasible","timing_candidate_feasible",
            "pressure_candidate_effort","timing_candidate_effort","actuator_effort_delta_pressure_minus_timing",
            "pressure_candidate_p_reg_bar","timing_candidate_timing_scale","policy_selection_rule"
        ] if c in sel.columns]
        deg=sel[cols].copy()
        save_table_multi(deg,dirs["engineering"]/"one_actuator_policy_degeneracy",SAVE_MODES[config.save_mode])
        out["policy_degeneracy"]=deg
    # Recompute mechanism after final windows/policy metadata are attached.
    mech=mechanism_and_control_authority(out,config.engineering)
    if len(mech):
        co=pd.to_numeric(mech.get("working_capacity_co2_mol_kg",np.nan),errors="coerce")
        nn=pd.to_numeric(mech.get("working_capacity_n2_mol_kg",np.nan),errors="coerce")
        denom=(co.clip(lower=0)+nn.clip(lower=0))
        mech["bounded_n2_contamination_fraction"]=np.where(denom>EPS,nn.clip(lower=0)/denom,np.nan)
        mech["bounded_co2_release_fraction"]=np.where(denom>EPS,co.clip(lower=0)/denom,np.nan)
        save_table_multi(mech,dirs["engineering"]/"mechanism_control_authority_map",SAVE_MODES[config.save_mode])
    out["mechanism"]=mech
    return out


def _v3_draw_classifier(pair, calibration, front_table, settings: EngineeringSettings, pgrid, tgrid,
                        reg_cache, *, front_multiplier: float, contamination_multiplier: float,
                        vacuum_efficiency: float):
    cache={}
    def classify(y: float) -> str:
        key=round(float(y),10)
        if key in cache:
            return cache[key]
        y=float(y)
        ads=mixture_state(y,settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)
        ff=front_factor(y,front_table)*front_multiplier
        def rel_for(p):
            reg=reg_cache[float(p)]
            return {
                "dco2":max(ads["q_co2"]-reg["q_co2"],0.0)*ff,
                "dn2":max(ads["q_n2"]-reg["q_n2"],0.0)*ff,
                "alpha_ads":ads["alpha"],
                "iast_status_ads":ads["iast_status"],"iast_status_reg":reg["iast_status"],
                "front_factor":ff
            }
        p0=settings.nominal_regeneration_pressure_bar
        r0=_process_metrics_from_release(y,p0,1.0,rel_for(p0),calibration,settings,
                                         contamination_multiplier=contamination_multiplier,
                                         vacuum_efficiency=vacuum_efficiency)
        if r0["meets_targets"]:
            out="robust_frozen"
        else:
            timing_ok=any(
                _process_metrics_from_release(y,p0,float(t),rel_for(p0),calibration,settings,
                                              contamination_multiplier=contamination_multiplier,
                                              vacuum_efficiency=vacuum_efficiency)["meets_targets"]
                for t in tgrid
            )
            pressure_ok=False
            if not timing_ok:
                for p in pgrid:
                    if _process_metrics_from_release(y,float(p),1.0,rel_for(float(p)),calibration,settings,
                                                     contamination_multiplier=contamination_multiplier,
                                                     vacuum_efficiency=vacuum_efficiency)["meets_targets"]:
                        pressure_ok=True; break
            if timing_ok or pressure_ok:
                out="single_knob_recoverable"
            else:
                coordinated=False
                for p in pgrid:
                    rel=rel_for(float(p))
                    for t in tgrid:
                        if _process_metrics_from_release(y,float(p),float(t),rel,calibration,settings,
                                                         contamination_multiplier=contamination_multiplier,
                                                         vacuum_efficiency=vacuum_efficiency)["meets_targets"]:
                            coordinated=True; break
                    if coordinated: break
                out="coordinated_recoverable" if coordinated else "outside_control_envelope"
        cache[key]=out
        return out
    return classify


def _v3_refine_draw_boundary(ys: np.ndarray, classes: Sequence[str], nominal_y: float, level: int,
                             classifier, side: str, tol_fraction: float=5e-4) -> Tuple[float,str,float,int]:
    rank={"robust_frozen":0,"single_knob_recoverable":1,"coordinated_recoverable":2,"outside_control_envelope":3}
    eligible=np.array([rank.get(str(c),3)<=level for c in classes],bool)
    a,b,status=_v3_transition_bracket(np.asarray(ys,float),eligible,nominal_y,side)
    if status!="interior_transition":
        return (float(a) if np.isfinite(a) else np.nan,status,np.nan,0)
    def ef(y):
        return rank.get(classifier(float(y)),3)<=level
    val,width,nit=_v3_bisect_boundary(a,b,ef,side,tol_fraction=tol_fraction)
    return val,"interior_adaptive_transition",width,nit


def _v3_refine_threshold_windows(threshold_selected: pd.DataFrame, threshold_windows: pd.DataFrame,
                                 pair, calibration, front_table, settings: EngineeringSettings,
                                 pressure_n: int, tol_fraction: float=5e-4) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if threshold_windows is None or threshold_windows.empty or threshold_selected is None or threshold_selected.empty:
        return threshold_windows,pd.DataFrame()
    rows=[]; audits=[]
    level_map={"frozen_safe":0,"single_knob_or_less":1,"coordinated_two_knob_or_less":2}
    for _,w in threshold_windows.iterrows():
        pt=float(w["threshold_purity"]); rt=float(w["threshold_recovery"]); label=str(w["intervention_ceiling"]); lev=level_map.get(label,int(w.get("max_adaptation_level",2)))
        sub=threshold_selected[(np.isclose(threshold_selected["threshold_purity"],pt))&(np.isclose(threshold_selected["threshold_recovery"],rt))].copy()
        if sub.empty:
            rows.append(w.to_dict()); continue
        sub=sub.sort_values("feed_co2_fraction")
        ys=sub["feed_co2_fraction"].to_numpy(float)
        eligible=sub["threshold_success"].astype(bool).to_numpy() & (pd.to_numeric(sub["adaptation_level"],errors="coerce").fillna(3).to_numpy(int)<=lev)
        a,b,status=_v3_transition_bracket(ys,eligible,settings.nominal_feed_co2_fraction,"lower")
        rr=w.to_dict()
        if status=="interior_transition":
            s2=dataclasses.replace(settings,target_purity=pt,target_recovery=rt)
            cache={}
            def ef(y):
                key=round(float(y),10)
                if key not in cache:
                    cache[key]=_v3_grid_level(float(y),pair,calibration,front_table,s2,pressure_n)<=lev
                return cache[key]
            val,width,nit=_v3_bisect_boundary(a,b,ef,"lower",tol_fraction=tol_fraction)
            rr["lower_feed_co2_fraction"]=val
            rr["lower_boundary_method"]="adaptive_bisection"
            rr["lower_boundary_bracket_width_fraction"]=width
            audits.append({"threshold_purity":pt,"threshold_recovery":rt,"intervention_ceiling":label,"coarse_a":a,"coarse_b":b,"refined_lower":val,"width":width,"iterations":nit})
        else:
            rr["lower_boundary_method"]=status
        rows.append(rr)
    return pd.DataFrame(rows),pd.DataFrame(audits)


# Conditional Monte Carlo uncertainty with draw-wise
# lower/upper transitions are locally bisected rather than reported at grid cells.
def stage_06_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], uncertainty_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    if atlas_payload["intrinsic"].empty:
        return {"classification":pd.DataFrame(),"boundaries":pd.DataFrame(),"draw_boundaries":pd.DataFrame(),"threshold_sensitivity":pd.DataFrame(),"threshold_windows":pd.DataFrame(),"threshold_boundary_refinement":pd.DataFrame()}
    settings=config.engineering; levelcfg=LEVELS[config.level]
    ys=atlas_payload["intrinsic"]["feed_co2_fraction"].to_numpy(float)
    n_draw=int(config.uncertainty_draws_override if config.uncertainty_draws_override is not None else levelcfg["uncertainty_draws_atlas"])
    allow_model_form=levelcfg["bootstrap_models"]=="retained"
    class_order=["robust_frozen","single_knob_recoverable","coordinated_recoverable","outside_control_envelope"]
    per_y={float(y):{k:0 for k in class_order} for y in ys}; draw_bounds=[]
    partial_path=dirs["uncertainty"]/"feed_composition_robustness_draws_partial.pkl"
    partial_key=stable_json_hash({"scientific":config.scientific_hash_payload(),"n_draw":n_draw,"ys":ys.tolist(),"adaptive_boundary_tolerance_fraction":5e-4,"fit_summary":fit_payload.get("summary",pd.DataFrame()).to_dict("records")})
    start_draw=0
    if partial_path.exists():
        try:
            st=safe_read_pickle(partial_path)
            if st.get("key")==partial_key:
                start_draw=int(st.get("next_draw",0)); per_y=st.get("per_y",per_y); draw_bounds=st.get("draw_bounds",[])
                logger.info("uncertainty resume: %d/%d draws already completed",start_draw,n_draw)
        except Exception:
            logger.exception("Could not read partial uncertainty checkpoint; rebuilding")
    npg=max(7,min(13,int(levelcfg["pressure_n"]//2+3))); nt=max(7,min(13,int(levelcfg["pressure_n"]//2+3)))
    pgrid=np.geomspace(settings.regeneration_pressure_floor_bar,settings.nominal_regeneration_pressure_bar,npg)
    tgrid=np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt)
    batch_draws=max(1,int(RAM_MODES[config.ram_mode]["batch"]))
    for d in range(start_draw,n_draw):
        drng=np.random.default_rng(config.seed+6006+d*104729)
        pair=_draw_pair(fit_payload,uncertainty_payload,drng,allow_model_form)
        if pair is None: continue
        fmult=float(drng.uniform(*settings.front_factor_range)); cmult=float(drng.uniform(*settings.contamination_scale_range)); eta=float(drng.uniform(*settings.vacuum_efficiency_range))
        cal=calibrate_nominal_surrogate_conditioned(pair,atlas_payload["front_table"],settings,evidence_payload["process_anchors"],front_multiplier=fmult,contamination_multiplier=cmult)
        reg_cache={float(p):mixture_state(settings.product_reference_co2_fraction,float(p),settings.nominal_temperature_K,pair) for p in pgrid}
        reg_cache[float(settings.nominal_regeneration_pressure_bar)]=mixture_state(settings.product_reference_co2_fraction,settings.nominal_regeneration_pressure_bar,settings.nominal_temperature_K,pair)
        classifier=_v3_draw_classifier(pair,cal,atlas_payload["front_table"],settings,pgrid,tgrid,reg_cache,
                                       front_multiplier=fmult,contamination_multiplier=cmult,vacuum_efficiency=eta)
        classes=[]
        for y in ys:
            cls=classifier(float(y)); classes.append(cls); per_y[float(y)][cls]+=1
        rec={"draw":d}
        for lev,prefix in [(0,"frozen"),(1,"single_or_less"),(2,"coordinated_or_less")]:
            for side in ("lower","upper"):
                val,status,width,nit=_v3_refine_draw_boundary(ys,classes,settings.nominal_feed_co2_fraction,lev,classifier,side,tol_fraction=5e-4)
                rec[f"{prefix}_{side}_y"]=val
                rec[f"{prefix}_{side}_status"]=status
                rec[f"{prefix}_{side}_bracket_width_fraction"]=width
                rec[f"{prefix}_{side}_iterations"]=nit
        draw_bounds.append(rec)
        if (d+1)%batch_draws==0 or (d+1)==n_draw:
            atomic_pickle({"key":partial_key,"next_draw":d+1,"per_y":per_y,"draw_bounds":draw_bounds},partial_path)
        if (d+1)%max(1,n_draw//8)==0:
            logger.info("adaptive-boundary uncertainty: %d/%d draws",d+1,n_draw)
    records=[]
    for y in ys:
        counts=per_y[float(y)]; total=max(sum(counts.values()),1); probs={k:counts[k]/total for k in class_order}; kmax=max(probs,key=probs.get); maxp=probs[kmax]; final=kmax if maxp>=settings.class_probability_threshold else "indeterminate"
        records.append({"feed_co2_fraction":float(y),"p_robust_frozen":probs["robust_frozen"],"p_single_knob_recoverable":probs["single_knob_recoverable"],"p_coordinated_recoverable":probs["coordinated_recoverable"],"p_outside_control_envelope":probs["outside_control_envelope"],"classification":final,"max_class_probability":maxp,"composition_support":composition_support_class(float(y),evidence_payload["summary"],settings,evidence_payload.get("composition_anchors"))})
    clsdf=pd.DataFrame(records); db=pd.DataFrame(draw_bounds); bound_rows=[]
    for col,label in [("frozen_lower_y","frozen_safe_lower"),("frozen_upper_y","frozen_safe_upper"),("single_or_less_lower_y","single_knob_lower"),("single_or_less_upper_y","single_knob_upper"),("coordinated_or_less_lower_y","coordinated_lower"),("coordinated_or_less_upper_y","coordinated_upper")]:
        vals=pd.to_numeric(db.get(col,pd.Series(dtype=float)),errors="coerce").dropna().to_numpy(float)
        med=float(np.nanmedian(vals)) if len(vals) else np.nan; q025=float(np.nanquantile(vals,.025)) if len(vals) else np.nan; q975=float(np.nanquantile(vals,.975)) if len(vals) else np.nan
        support=composition_support_class(med,evidence_payload["summary"],settings,evidence_payload.get("composition_anchors")) if np.isfinite(med) else "unresolved"
        min_valid=max(3,int(math.ceil(.5*n_draw))); vf=float(len(vals)/n_draw) if n_draw else 0.0
        if len(vals)<min_valid: status="insufficient_valid_draw_support"
        elif label.endswith("lower") and any("left_censored" in str(x) for x in db.get(col.replace("_y","_status"),pd.Series(dtype=str))): status="mixed_or_left_censored"
        elif label.endswith("upper") and any("right_censored" in str(x) for x in db.get(col.replace("_y","_status"),pd.Series(dtype=str))): status="mixed_or_right_censored"
        else: status="interior_adaptive_uncertainty_boundary" if np.isfinite(med) else "unresolved"
        bound_rows.append({
            "boundary":label,"median_feed_co2_fraction":med,"q025":q025,"q975":q975,
            "n_valid_draws":len(vals),"valid_draw_fraction":vf,"minimum_valid_draws_for_resolved_boundary":min_valid,
            "median_composition_support":support,
            "ci_entirely_inside_primary_evidence_envelope":bool(np.isfinite(q025) and np.isfinite(q975) and q025>=settings.primary_evidence_feed_min-EPS and q975<=settings.primary_evidence_feed_max+EPS),
            "boundary_status":status,"boundary_resolved":status=="interior_adaptive_uncertainty_boundary",
            "boundary_method":"draw-wise adaptive bisection","boundary_tolerance_percentage_points":0.05
        })
    boundaries=pd.DataFrame(bound_rows)
    save_table_multi(clsdf,dirs["uncertainty"]/"feed_composition_robustness_uncertainty_classification",SAVE_MODES[config.save_mode])
    save_table_multi(boundaries,dirs["uncertainty"]/"feed_composition_robustness_intervention_boundaries",SAVE_MODES[config.save_mode])
    if SAVE_MODES[config.save_mode]["bootstrap_samples"]:
        save_table_multi(db,dirs["uncertainty"]/"feed_composition_robustness_draw_boundaries",SAVE_MODES[config.save_mode])
    pair0=calf20_pair(fit_payload)
    if pair0 is not None:
        threshold_settings=settings if config.level!="quickcheck" else dataclasses.replace(settings,purity_threshold_sensitivity=(settings.target_purity,),recovery_threshold_sensitivity=(settings.target_recovery,))
        ts,tw=threshold_sensitivity_recalculate(ys,pair0,atlas_payload.get("calibration",{}),atlas_payload.get("front_table",pd.DataFrame()),threshold_settings,int(levelcfg["pressure_n"]))
        tw_ref,aud=_v3_refine_threshold_windows(ts,tw,pair0,atlas_payload.get("calibration",{}),atlas_payload.get("front_table",pd.DataFrame()),threshold_settings,int(levelcfg["pressure_n"]))
    else:
        ts,tw_ref,aud=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    save_table_multi(ts,dirs["uncertainty"]/"target_threshold_sensitivity_selected",SAVE_MODES[config.save_mode])
    save_table_multi(tw_ref,dirs["uncertainty"]/"target_threshold_sensitivity_windows",SAVE_MODES[config.save_mode])
    save_table_multi(aud,dirs["uncertainty"]/"target_threshold_adaptive_boundary_audit",SAVE_MODES[config.save_mode])
    return {"classification":clsdf,"boundaries":boundaries,"draw_boundaries":db,"threshold_sensitivity":ts,"threshold_windows":tw_ref,"threshold_boundary_refinement":aud}


def _v3_min_row_from_grid(d: pd.DataFrame) -> Dict[str,Any]:
    if d is None or d.empty:
        return {}
    ok=d[d["success"].astype(bool)].copy()
    if ok.empty:
        r=d.iloc[0].to_dict(); r.update({"adaptation_level":3,"adaptation_name":"outside_normal_control_envelope","adaptation_variant":"none_feasible","success":False}); return r
    m=int(pd.to_numeric(ok["adaptation_level"],errors="coerce").min())
    cand=ok[pd.to_numeric(ok["adaptation_level"],errors="coerce")==m]
    if m==1:
        rows=[r.to_dict() for _,r in cand.iterrows()]
        c=_select_one_actuator_policy(rows,float(cand.iloc[0].get("feed_co2_fraction",np.nan)) if "feed_co2_fraction" in cand else np.nan,config_dummy_settings) if False else None
    return cand.iloc[0].to_dict()


def _v3_stress_atlas(pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int, ys: np.ndarray,
                     *, conditioning_mode: str, scenario_id: str, scenario_label: str, co2_scale: float, n2_scale: float) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    per=[]
    for y in ys:
        d=_solve_adaptation_levels_grid_v130(float(y),pair,calibration,front_table,settings,pressure_n)
        if d.empty:
            continue
        ok=d[d["success"].astype(bool)].copy()
        if ok.empty:
            r=d.iloc[0].to_dict(); r.update({"adaptation_level":3,"adaptation_name":"outside_normal_control_envelope","adaptation_variant":"none_feasible","success":False})
        else:
            m=int(pd.to_numeric(ok["adaptation_level"],errors="coerce").min())
            cand=ok[pd.to_numeric(ok["adaptation_level"],errors="coerce")==m]
            # For the stress atlas class topology is the scientific object; choose
            # the closest state within the minimum class only for diagnostics.
            cc=[_decorate_control_move(r.to_dict(),settings) for _,r in cand.iterrows()]
            cc.sort(key=lambda r:(r.get("control_move_linf",np.inf),r.get("control_move_l1",np.inf),r.get("specific_vacuum_work_kJ_mol_CO2",np.inf)))
            r=cc[0]
        r.update({"feed_co2_fraction":float(y),"scenario_id":scenario_id,"scenario_label":scenario_label,
                  "conditioning_mode":conditioning_mode,"co2_capacity_scale":co2_scale,"n2_capacity_scale":n2_scale})
        per.append(r)
    sdf=pd.DataFrame(per)
    if sdf.empty:
        return sdf,pd.DataFrame(),pd.DataFrame()
    rw,ra=_v3_refine_windows(sdf,pair,calibration,front_table,settings,pressure_n,tol_fraction=5e-4)
    if len(rw):
        rw=rw.copy()
        rw["scenario_id"]=scenario_id; rw["scenario_label"]=scenario_label; rw["conditioning_mode"]=conditioning_mode
        rw["co2_capacity_scale"]=co2_scale; rw["n2_capacity_scale"]=n2_scale
        rw["stress_role"]="non-probabilistic literature discrepancy stress"
    if len(ra):
        ra=ra.copy(); ra["scenario_id"]=scenario_id; ra["conditioning_mode"]=conditioning_mode
    return sdf,rw,ra


def thermodynamic_discrepancy_stress_test(config: RunConfig, fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], scalar_crosschecks: pd.DataFrame, settings: EngineeringSettings, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    """Evaluate both nominally conditioned and raw/unconditioned source discrepancy.

    Conditioned cases isolate off-design topology after every thermodynamic
    realization is recalibrated to the same 15% experimental anchor.
    Unconditioned cases retain the nominal calibration/front-factor mapping and
    therefore quantify direct batch/source sensitivity.  Neither family is
    assigned a probability.
    """
    root=_optional_data_root(config)/"external_equilibrium_sensitivity"
    candidates=[root/"CALF20_thermodynamic_discrepancy_stress_policy.csv"]
    policy=next((_safe_read_csv(p) for p in candidates if p.exists()),pd.DataFrame())
    if policy.empty:
        return {"scenarios":pd.DataFrame(),"boundaries":pd.DataFrame(),"selected":pd.DataFrame(),"boundary_refinement":pd.DataFrame()}
    desc=[]
    for gas in ["CO2","N2"]:
        q=best_fit_descriptor(fit_payload.get("fits",{}),"CALF-20",gas)
        if q is None:
            return {"scenarios":pd.DataFrame(),"boundaries":pd.DataFrame(),"selected":pd.DataFrame(),"boundary_refinement":pd.DataFrame()}
        desc.append(q)
    nominal_pair=tuple(desc)
    ft_nom=_anchor_front_factor_table(evidence_payload.get("summary",pd.DataFrame()),nominal_pair,settings)
    cal_nom=calibrate_nominal_surrogate_conditioned(nominal_pair,ft_nom,settings,evidence_payload.get("process_anchors",pd.DataFrame()))
    ys=np.linspace(settings.feed_co2_min,settings.feed_co2_max,LEVELS[config.level]["composition_n"])
    pressure_n=max(9,int(LEVELS[config.level]["pressure_n"]))
    scenario_rows=[]; all_bounds=[]; all_selected=[]; all_ref=[]
    for _,pr in policy.iterrows():
        if str(pr.get("enabled",True)).strip().lower() in {"false","0","no"}:
            continue
        sid=str(pr.get("scenario_id","stress")); label=str(pr.get("scenario_label",sid))
        refT=float(pd.to_numeric(pd.Series([pr.get("reference_temperature_K",298.0)]),errors="coerce").iloc[0])
        maxext=float(pd.to_numeric(pd.Series([pr.get("max_temperature_extrapolation_K",10.0)]),errors="coerce").iloc[0])
        cs,nc,cwhy=_derive_discrepancy_scale(scalar_crosschecks,"CO2",str(pr.get("co2_scale_mode","unity")),refT,maxext)
        ns,nn,nwhy=_derive_discrepancy_scale(scalar_crosschecks,"N2",str(pr.get("n2_scale_mode","unity")),refT,maxext)
        ok=np.isfinite(cs) and np.isfinite(ns) and cs>0 and ns>0
        base={**{c:pr.get(c,"") for c in policy.columns},"derived_co2_capacity_scale":cs,"derived_n2_capacity_scale":ns,
              "n_co2_scalar_points":nc,"n_n2_scalar_points":nn,"co2_scale_derivation":cwhy,"n2_scale_derivation":nwhy}
        if not ok:
            scenario_rows.append({**base,"conditioning_mode":"not_evaluated","scenario_status":"skipped_missing_evidence"}); continue
        pair=_capacity_scaled_pair(nominal_pair,cs,ns)
        # Conditioned: thermodynamics can differ, but the nominal experiment is reproduced.
        ft_cond=_anchor_front_factor_table(evidence_payload.get("summary",pd.DataFrame()),pair,settings)
        cal_cond=calibrate_nominal_surrogate_conditioned(pair,ft_cond,settings,evidence_payload.get("process_anchors",pd.DataFrame()))
        for mode,ft,cal in [("nominally_conditioned",ft_cond,cal_cond),("unconditioned_raw_source_sensitivity",ft_nom,cal_nom)]:
            scenario_rows.append({**base,"conditioning_mode":mode,"scenario_status":"evaluated",
                                  "nominal_conditioned_purity":cal.get("purity_anchor",np.nan),
                                  "nominal_conditioned_recovery":cal.get("recovery_anchor",np.nan),
                                  "interpretation":"off-design topology after re-anchoring" if mode=="nominally_conditioned" else "direct batch/source sensitivity without re-anchoring"})
            sdf,bdf,rdf=_v3_stress_atlas(pair,cal,ft,settings,pressure_n,ys,conditioning_mode=mode,scenario_id=sid,scenario_label=label,co2_scale=cs,n2_scale=ns)
            all_selected.append(sdf); all_bounds.append(bdf); all_ref.append(rdf)
    boundaries=pd.concat(all_bounds,ignore_index=True,sort=False) if all_bounds else pd.DataFrame()
    selected=pd.concat(all_selected,ignore_index=True,sort=False) if all_selected else pd.DataFrame()
    refinement=pd.concat(all_ref,ignore_index=True,sort=False) if all_ref else pd.DataFrame()
    if logger and len(boundaries):
        logger.info("thermodynamic discrepancy stress: %d scenario-mode combinations",boundaries[["scenario_id","conditioning_mode"]].drop_duplicates().shape[0])
    return {"scenarios":pd.DataFrame(scenario_rows),"boundaries":boundaries,"selected":selected,"boundary_refinement":refinement}


def boundary_falsification_portfolio(atlas_payload: Mapping[str,Any], settings: EngineeringSettings) -> pd.DataFrame:
    """Publication-closure CSS portfolio based on refined deterministic boundaries."""
    win=atlas_payload.get("windows",pd.DataFrame()); sel=atlas_payload.get("selected",pd.DataFrame())
    if win is None or win.empty:
        return pd.DataFrame()
    rows=[]; seen=set()
    def nearest_meta(y):
        if sel is None or sel.empty:
            return {}
        dist=(pd.to_numeric(sel["feed_co2_fraction"],errors="coerce")-float(y)).abs()
        j=dist.idxmin(); m=sel.loc[j].to_dict(); m["nearest_coarse_distance_fraction"]=float(dist.loc[j])
        return m
    def add(y,role,question,priority="must_run"):
        if not np.isfinite(y): return
        y=float(np.clip(y,settings.feed_co2_min,settings.feed_co2_max)); key=round(y,5)
        if key in seen: return
        seen.add(key); m=nearest_meta(y)
        coarse_dist=float(m.get("nearest_coarse_distance_fraction",np.nan)) if m else np.nan
        on_coarse=np.isfinite(coarse_dist) and coarse_dist<=0.0015
        rows.append({
            "case_id":f"CSS_{len(rows)+1:02d}","feed_co2_fraction":y,"feed_co2_percent":100*y,
            "role":role,"validation_question":question,"priority":priority,
            "nearest_coarse_distance_fraction":coarse_dist,
            "nearest_coarse_level":m.get("adaptation_level",np.nan) if on_coarse else np.nan,
            "nearest_coarse_variant":m.get("adaptation_variant","") if on_coarse else "off_grid_recheck_required",
            "suggested_p_reg_bar":m.get("p_reg_bar",np.nan) if on_coarse else np.nan,
            "suggested_timing_scale":m.get("timing_scale",np.nan) if on_coarse else np.nan,
            "reduced_order_prediction_basis":"nearest coarse-grid diagnostic" if on_coarse else "refined-boundary target; recompute directly in CSS",
            "composition_support":composition_support_class(y,pd.DataFrame(),settings,None),
            "required_outputs":"CSS convergence; purity; recovery; pressure/temperature transients; energy; active constraint; intervention level",
            "calibration_role":"independent falsification only; never fit reduced-order model"
        })
    # Refined lean boundaries with ±0.32 percentage-point outside checks, matching
    # the old coarse spacing while retaining the new exact boundary itself.
    delta=0.0032
    for label,role in [("coordinated_two_knob_or_less","coordinated"),("single_knob_or_less","single"),("frozen_safe","frozen")]:
        w=win[win["intervention_ceiling"].astype(str).eq(label)]
        if w.empty: continue
        y=float(w.iloc[0]["lower_feed_co2_fraction"])
        add(y-delta,f"just outside {role} lean boundary",f"Does rigorous CSS reject feasibility just below the {role} boundary?")
        add(y,f"{role} lean boundary",f"Does rigorous CSS preserve the predicted {role} transition and active constraint?")
    add(settings.nominal_feed_co2_fraction,"nominal experimental anchor","Does rigorous CSS reproduce the nominal 15% experimental operating state?")
    add(settings.nominal_feed_co2_fraction+0.003,"just rich of frozen region","Does rich-side CSS show recovery-limited frozen failure and a one-actuator remedy?")
    add(0.30,"published rich-feed composition anchor","Does the predicted rich-feed intervention direction survive at the 30% literature-supported composition?")
    return pd.DataFrame(rows).sort_values("feed_co2_fraction").reset_index(drop=True)


# Preserve the external-validation calculation and save the additional stress
# refinement audit and the refined falsification plan.
_stage_07_v230 = stage_07_validation
def stage_07_validation(config: RunConfig, dirs: Mapping[str,Path], harmonized_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_07_v230(config,dirs,harmonized_payload,fit_payload,evidence_payload,atlas_payload,logger)
    stress=out.get("thermodynamic_discrepancy_stress",{})
    if isinstance(stress,dict):
        ref=stress.get("boundary_refinement",pd.DataFrame())
        if isinstance(ref,pd.DataFrame):
            save_table_multi(ref,dirs["validation"]/"thermodynamic_discrepancy_stress_boundary_refinement",SAVE_MODES[config.save_mode])
    plan=boundary_falsification_portfolio(atlas_payload,config.engineering)
    save_table_multi(plan,dirs["validation"]/"boundary_falsification_portfolio",SAVE_MODES[config.save_mode])
    out["falsification_portfolio"]=plan
    return out


def figure_2(dirs,config,atlas_payload):
    s=config.engineering; d=atlas_payload.get("mechanism",pd.DataFrame()).copy(); intrinsic=atlas_payload.get("intrinsic",pd.DataFrame()).copy()
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty or intrinsic.empty:
        [empty_axis(ax,"No mechanism data") for ax in axs.flat]
    else:
        x=100*intrinsic["feed_co2_fraction"]
        axs[0,0].plot(x,intrinsic["q_co2_mol_kg"],label="CO$_2$")
        axs[0,0].plot(x,intrinsic["q_n2_mol_kg"],label="N$_2$")
        axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)"); axs[0,0].legend(frameon=False)
        if "adsorbed_x_co2" in intrinsic:
            axs[0,1].plot(x,intrinsic["adsorbed_x_co2"])
            axs[0,1].plot(x,intrinsic["feed_co2_fraction"],ls="--",lw=.8,color="#555555")
            axs[0,1].text(.53,.09,r"$x_{CO_2}=y_{CO_2}$ (no enrichment)",transform=axs[0,1].transAxes,fontsize=7)
            axs[0,1].set_ylabel("Adsorbed-phase CO$_2$ fraction")
        else:
            axs[0,1].plot(x,intrinsic.get("iast_selectivity_co2_n2",np.nan)); axs[0,1].set_ylabel("IAST selectivity")
        xd=100*d["feed_co2_fraction"]
        axs[1,0].plot(xd,d["working_capacity_co2_mol_kg"],label="CO$_2$ working capacity",lw=1.7)
        axs[1,0].set_ylabel("CO$_2$ working capacity (mol kg$^{-1}$)")
        frac=d.get("bounded_n2_contamination_fraction",pd.Series(np.nan,index=d.index))
        axr=axs[1,0].twinx()
        axr.plot(xd,frac,ls="--",lw=1.5,label="N$_2$ contamination fraction")
        axr.set_ylim(-.02,1.02); axr.set_ylabel("N$_2$ fraction of total released capacity")
        lines=axs[1,0].get_lines()+axr.get_lines()
        axs[1,0].legend(lines,[z.get_label() for z in lines],frameon=False,fontsize=7,loc="best")
        axs[1,1].plot(xd,d["thermodynamic_pressure_ratio"],lw=1.7)
        axs[1,1].axhline(1,ls="--",lw=.9,color="#333333")
        axs[1,1].fill_between(xd,np.maximum(d["thermodynamic_pressure_ratio"],1e-6),1,where=d["thermodynamic_pressure_ratio"]<1,alpha=.12)
        axs[1,1].set_yscale("log"); axs[1,1].set_ylabel(r"$\Pi_{reg}=y_{CO_2}P_{ads}/(y_{prod}P_{reg})$")
        axs[1,1].text(.03,.08,r"$\Pi_{reg}<1$: weak regeneration driving force",transform=axs[1,1].transAxes,fontsize=7)
        titles=["Competitive mixture loading","Adsorbed-phase enrichment","Capacity and bounded contamination","Regeneration driving-force indicator"]
        for i,ax in enumerate(axs.flat):
            _decorate_composition_axis(ax,s); ax.set_title(titles[i]); panel_label(ax,"abcd"[i])
    save_figure(fig,dirs["fig_main"]/"Figure_2_mechanism_and_constraint_switching",SAVE_MODES[config.save_mode])
    save_panel_source(d,dirs["source_main"]/"Figure_2_source_data",SAVE_MODES[config.save_mode])


def _v3_shade_degenerate(ax,d):
    if d is None or d.empty or "actuator_choice_degenerate" not in d:
        return
    x=100*pd.to_numeric(d["feed_co2_fraction"],errors="coerce").to_numpy(float)
    deg=np.asarray([bool(v) if pd.notna(v) else False for v in d["actuator_choice_degenerate"].tolist()],dtype=bool)
    if len(x)<2: return
    edges=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
    i=0
    while i<len(x):
        if not deg[i]: i+=1; continue
        j=i+1
        while j<len(x) and deg[j]: j+=1
        ax.axvspan(edges[i],edges[j],color="#777777",alpha=.08,zorder=-3)
        i=j


def figure_4(dirs,config,atlas_payload):
    d=atlas_payload.get("selected",pd.DataFrame()).copy(); s=config.engineering
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty:
        [empty_axis(ax,"No intervention results") for ax in axs.flat]
    else:
        failed=~d["success"].astype(bool)
        for c in ["p_reg_bar","timing_scale","specific_vacuum_work_kJ_mol_CO2","productivity_mol_kg_h"]:
            if c in d: d.loc[failed,c]=np.nan
        x=100*d["feed_co2_fraction"]
        axs[0,0].plot(x,d["p_reg_bar"],lw=1.2,label="selected policy")
        if "pressure_candidate_p_reg_bar" in d:
            axs[0,0].plot(x,d["pressure_candidate_p_reg_bar"],ls="--",lw=1.5,label="best pressure-only candidate")
        axs[0,0].axhline(s.nominal_regeneration_pressure_bar,ls=":",lw=.9,color="#333333")
        axs[0,0].set_yscale("log"); axs[0,0].set_ylabel("Regeneration pressure (bar)"); axs[0,0].legend(frameon=False,fontsize=7)
        axs[0,1].plot(x,d["timing_scale"],lw=1.2,label="selected policy")
        if "timing_candidate_timing_scale" in d:
            axs[0,1].plot(x,d["timing_candidate_timing_scale"],ls="--",lw=1.5,label="best timing-only candidate")
        axs[0,1].axhline(1,ls=":",lw=.9,color="#333333")
        axs[0,1].set_ylabel("Adsorption-time scale"); axs[0,1].legend(frameon=False,fontsize=7)
        fixed=atlas_payload.get("fixed",pd.DataFrame())
        base=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"])
        ratio=100*(d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(base,EPS)-1.0); ratio[~np.isfinite(ratio)]=np.nan
        axs[1,0].plot(x,ratio,lw=1.8); axs[1,0].axhline(0,ls="--",lw=.8,color="#333333"); axs[1,0].set_ylabel("Change in evacuation-work proxy (%)")
        basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"])
        pr=100*(d["productivity_mol_kg_h"]/np.maximum(basep,EPS)-1.0); pr[~np.isfinite(pr)]=np.nan
        axs[1,1].plot(x,pr,lw=1.8); axs[1,1].axhline(0,ls="--",lw=.8,color="#333333"); axs[1,1].set_ylabel("Change in productivity (%)")
        for i,ax in enumerate(axs.flat):
            _decorate_composition_axis(ax,s); _class_strip(ax,d); _v3_shade_degenerate(ax,d); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Pressure candidate and selected policy")
        axs[0,1].set_title("Timing candidate and selected policy")
        axs[1,0].set_title("Resource consequence")
        axs[1,1].set_title("Productivity consequence")
        if "actuator_choice_degenerate" in d and any(bool(v) if pd.notna(v) else False for v in d["actuator_choice_degenerate"].tolist()):
            axs[0,1].text(.53,.07,"grey = near-degenerate\nsingle-actuator choice",transform=axs[0,1].transAxes,fontsize=7,ha="center")
    save_figure(fig,dirs["fig_main"]/"Figure_4_minimum_intervention_policy",SAVE_MODES[config.save_mode])
    save_panel_source(d,dirs["source_main"]/"Figure_4_source_data",SAVE_MODES[config.save_mode])


def _v3_export_panel_sources(dirs, harmonized_payload, evidence_payload, atlas_payload, robustness_payload, validation_payload):
    outdir=dirs["source_main"]; outdir.mkdir(parents=True,exist_ok=True)
    def put(name,df):
        if isinstance(df,pd.DataFrame):
            atomic_csv(df,outdir/f"{name}.csv",index=False)
    put("Figure_1a",evidence_payload.get("composition_anchors",pd.DataFrame()))
    put("Figure_1b",harmonized_payload.get("data",pd.DataFrame()))
    put("Figure_1c",validation_payload.get("external_pilot",{}).get("summary",pd.DataFrame()))
    put("Figure_1d",atlas_payload.get("windows",pd.DataFrame()))
    put("Figure_2a",atlas_payload.get("intrinsic",pd.DataFrame()))
    put("Figure_2b",atlas_payload.get("intrinsic",pd.DataFrame()))
    put("Figure_2c",atlas_payload.get("mechanism",pd.DataFrame()))
    put("Figure_2d",atlas_payload.get("mechanism",pd.DataFrame()))
    for panel in "abcd": put(f"Figure_3{panel}",atlas_payload.get("fixed",pd.DataFrame()))
    for panel in "abcd": put(f"Figure_4{panel}",atlas_payload.get("selected",pd.DataFrame()))
    put("Figure_5a",atlas_payload.get("selected",pd.DataFrame()))
    put("Figure_5b",robustness_payload.get("classification",pd.DataFrame()))
    put("Figure_5c",atlas_payload.get("windows",pd.DataFrame()))
    put("Figure_5d",robustness_payload.get("boundaries",pd.DataFrame()))


_stage_08_v230 = stage_08_tables
def stage_08_tables(config: RunConfig, dirs: Mapping[str,Path], audit_payload, harmonized_payload, fit_payload, uncertainty_payload, evidence_payload, atlas_payload, robustness_payload, validation_payload, logger) -> Dict[str,Any]:
    out=_stage_08_v230(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger)
    extras={
        "Table_S12c_adaptive_deterministic_boundary_refinement":atlas_payload.get("boundary_refinement",pd.DataFrame()),
        "Table_S12d_one_actuator_policy_degeneracy":atlas_payload.get("policy_degeneracy",pd.DataFrame()),
        "Table_S16b_target_threshold_adaptive_boundary_refinement":robustness_payload.get("threshold_boundary_refinement",pd.DataFrame()),
        "Table_S20d4_thermodynamic_discrepancy_boundary_refinement":validation_payload.get("thermodynamic_discrepancy_stress",{}).get("boundary_refinement",pd.DataFrame()),
    }
    for name,df in extras.items():
        save_table_multi(df,dirs["tables_si"]/name,SAVE_MODES[config.save_mode]); out["si"][name]=df
    cm=out.get("claim_evidence_matrix",pd.DataFrame()).copy()
    if len(cm):
        cm.loc[cm["claim_id"].eq("C2"),"validation_layer"]="adaptive deterministic boundaries + draw-wise adaptive uncertainty + grid convergence + independent thermodynamic source"
        cm.loc[cm["claim_id"].eq("C7"),"validation_layer"]="non-probabilistic conditioned and unconditioned literature-discrepancy stress"
        cm=pd.concat([cm,pd.DataFrame([
            {"claim_id":"C8","claim":"Reported composition boundaries are not limited to the global feed-grid spacing.","primary_evidence":"Table S12c; Table S16b","validation_layer":"local adaptive bisection to 0.05 percentage-point tolerance","status":"supported"},
            {"claim_id":"C9","claim":"Near-degenerate pressure-only and timing-only solutions are identified rather than hidden as a unique rich-side policy.","primary_evidence":"Figure 4; Table S12d","validation_layer":"explicit normalized intervention-effort comparison","status":"supported within reduced-order model"},
        ])],ignore_index=True)
        save_table_multi(cm,dirs["tables_si"]/"Table_S1g_claim_evidence_matrix",SAVE_MODES[config.save_mode])
        out["claim_evidence_matrix"]=cm; out["si"]["Table_S1g_claim_evidence_matrix"]=cm
    return out


def _v3_figure_s17(dirs,config,validation_payload):
    stress=validation_payload.get("thermodynamic_discrepancy_stress",{}).get("boundaries",pd.DataFrame()).copy()
    if stress is None or stress.empty:
        return
    map_label={"frozen_safe":"Frozen","single_knob_or_less":"<=1 control","coordinated_two_knob_or_less":"<=2 controls"}
    scenario_order=list(dict.fromkeys(stress["scenario_label"].astype(str).tolist()))
    fig,ax=plt.subplots(figsize=(10.4,5.8),constrained_layout=True)
    ybase=np.arange(len(scenario_order),dtype=float)
    offsets={"frozen_safe":-0.18,"single_knob_or_less":0.0,"coordinated_two_knob_or_less":0.18}
    markers={"frozen_safe":"o","single_knob_or_less":"s","coordinated_two_knob_or_less":"D"}
    source_rows=[]
    for ceiling in ["frozen_safe","single_knob_or_less","coordinated_two_knob_or_less"]:
        for mode,filled,dx in [("nominally_conditioned",True,-0.035),("unconditioned_raw_source_sensitivity",False,0.035)]:
            xs=[]; ys=[]
            for i,sc in enumerate(scenario_order):
                g=stress[(stress["scenario_label"].astype(str)==sc)&(stress["intervention_ceiling"].astype(str)==ceiling)&(stress["conditioning_mode"].astype(str)==mode)]
                if g.empty: continue
                r=g.iloc[0]; x=pd.to_numeric(pd.Series([r.get("lower_feed_co2_fraction",np.nan)]),errors="coerce").iloc[0]
                if np.isfinite(x):
                    xs.append(100*float(x)); ys.append(i+offsets[ceiling]+dx)
                    source_rows.append(r.to_dict())
            if xs:
                label=f"{map_label[ceiling]} — {'conditioned' if filled else 'raw'}"
                if filled:
                    ax.scatter(xs,ys,marker=markers[ceiling],s=45,label=label,zorder=3)
                else:
                    ax.scatter(xs,ys,marker=markers[ceiling],s=45,facecolors='none',edgecolors='#333333',label=label,zorder=3)
    # Explicitly annotate mode/scenario combinations without a resolved nominal-eligible boundary.
    for i,sc in enumerate(scenario_order):
        g=stress[(stress["scenario_label"].astype(str)==sc)&(stress["conditioning_mode"].astype(str)=="unconditioned_raw_source_sensitivity")]
        if not g.empty and pd.to_numeric(g["lower_feed_co2_fraction"],errors="coerce").notna().sum()==0:
            ax.text(0.01,i+0.34,"raw: nominal infeasible",fontsize=6.5,va="center",transform=ax.get_yaxis_transform())
    ax.set_yticks(ybase,scenario_order,fontsize=7.5)
    ax.set_xlabel("Lean-side lower boundary (feed CO$_2$ %)")
    ax.set_title("Conditioned vs unconditioned literature-discrepancy stress")
    ax.grid(axis="x",alpha=.15); ax.legend(frameon=False,fontsize=6.5,ncol=2,loc="best")
    save_figure(fig,dirs["fig_si"] / "Figure_S17_thermodynamic_discrepancy_stress",SAVE_MODES[config.save_mode])
    atomic_csv(stress,dirs["source_si"] / "Figure_S17_source_data.csv",index=False)


_stage_09_v230 = stage_09_figures
def stage_09_figures(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger):
    out=_stage_09_v230(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger)
    _v3_figure_s17(dirs,config,validation_payload)
    _v3_export_panel_sources(dirs,harmonized_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload)
    out["si_figures"]=sorted(p.name for p in dirs["fig_si"].glob("Figure_*"))
    return out


_stage_10_v230 = stage_10_finalize
def stage_10_finalize(config,dirs,input_hash,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,tables_payload,figures_payload,logger):
    out=_stage_10_v230(config,dirs,input_hash,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,tables_payload,figures_payload,logger)
    css=validation_payload.get("css_validation",{}).get("comparison",pd.DataFrame())
    plan=validation_payload.get("falsification_portfolio",pd.DataFrame())
    gaps=pd.DataFrame([
        {"gap":"Rigorous non-isothermal CSS confirmation at refined minimum-control-authority boundaries","status":"OPEN" if css.empty else "PARTIALLY/CLOSED","blocking_for_reduced_order_paper":False,"blocking_for_highest_ceiling_claim":True,"next_action":"Run cases in validation/boundary_falsification_portfolio.csv and place genuine CSS outputs under data/css_validation."},
        {"gap":"Published raw breakthrough transient coordinates at exact P10 cases","status":"OPEN","blocking_for_reduced_order_paper":False,"blocking_for_highest_ceiling_claim":False,"next_action":"Use only if auditable raw/digitized traces can be obtained; never fabricate."},
        {"gap":"High-fidelity transient 15->12->18->15% feed-disturbance demonstration","status":"OPTIONAL_HIGH_CEILING","blocking_for_reduced_order_paper":False,"blocking_for_highest_ceiling_claim":False,"next_action":"Recommended only after CSS kernel is available."},
    ])
    save_table_multi(gaps,dirs["final"]/"OPEN_VALIDATION_GAPS",SAVE_MODES[config.save_mode])
    if isinstance(plan,pd.DataFrame): save_table_multi(plan,dirs["final"]/"CSS_FALSIFICATION_PLAN",SAVE_MODES[config.save_mode])
    cm=tables_payload.get("claim_evidence_matrix",pd.DataFrame())
    if isinstance(cm,pd.DataFrame): save_table_multi(cm,dirs["final"]/"CLAIM_EVIDENCE_MATRIX",SAVE_MODES[config.save_mode])
    summary=f"""# P10 analysis status

## What this clean run establishes
- Experimentally anchored nominal CALF-20 VSA state at 15% CO2.
- Minimum-control-authority topology from frozen to one-control to coordinated-control operation.
- Adaptive deterministic and Monte Carlo boundary refinement below the global feed-grid spacing.
- Explicit pressure-vs-timing near-degeneracy diagnostics rather than a forced unique actuator policy.
- Independent thermodynamic-source sensitivity.
- Conditioned and unconditioned literature-discrepancy stress tests.
- External 75-run pilot-scale directional consistency evidence.
- Exact per-panel source-data exports and a claim-evidence matrix.

## Highest-ceiling item still requiring genuinely new calculation
Rigorous non-isothermal cyclic-steady-state (CSS) falsification at the cases listed in
`CSS_FALSIFICATION_PLAN.csv`. These values are intentionally not fabricated by the package.

## Interpretation guardrail
Reduced-order boundaries are publication-ready as model-derived operability results, but the
highest-ceiling cross-fidelity claim must remain pending until genuine CSS calculations are supplied.
"""
    write_text(dirs["final"]/"PUBLICATION_CLOSURE_STATUS.md",summary)
    return out



# Cache regeneration equilibrium states for the stress atlas.
def _v3_fast_stress_classifier(pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int):
    p0=settings.nominal_regeneration_pressure_bar
    npg=max(9,int(math.sqrt(max(pressure_n,9))*2)); nt=max(9,int(math.sqrt(max(pressure_n,9))*2))
    pgrid=np.unique(np.r_[np.geomspace(settings.regeneration_pressure_floor_bar,p0,npg),p0])
    tgrid=np.unique(np.r_[np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(15,pressure_n)),1.0])
    tjoint=np.unique(np.r_[np.linspace(settings.timing_min_scale,settings.timing_max_scale,nt),1.0])
    reg_cache={float(pp):mixture_state(settings.product_reference_co2_fraction,float(pp),settings.nominal_temperature_K,pair) for pp in pgrid}
    cache={}
    def eval_y(y):
        key=round(float(y),10)
        if key in cache: return cache[key]
        y=float(y); ff=front_factor(y,front_table); ads=mixture_state(y,settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)
        def rel(pp):
            rg=reg_cache[float(pp)]
            return {"dco2":max(ads["q_co2"]-rg["q_co2"],0.0)*ff,"dn2":max(ads["q_n2"]-rg["q_n2"],0.0)*ff,
                    "alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":rg["iast_status"],"front_factor":ff}
        fixed=_decorate_control_move(_process_metrics_from_release(y,p0,1.0,rel(p0),calibration,settings),settings)
        if fixed["meets_targets"]:
            rr={**fixed,"adaptation_level":0,"adaptation_name":"frozen_process","adaptation_variant":"frozen","success":True}
        else:
            one=[]
            for t in tgrid:
                r=_process_metrics_from_release(y,p0,float(t),rel(p0),calibration,settings)
                if r["meets_targets"]:
                    r=_decorate_control_move(r,settings); r["adaptation_variant"]="timing_only"; one.append(r)
            for pp in pgrid:
                r=_process_metrics_from_release(y,float(pp),1.0,rel(float(pp)),calibration,settings)
                if r["meets_targets"]:
                    r=_decorate_control_move(r,settings); r["adaptation_variant"]="pressure_only"; one.append(r)
            best1=_select_one_actuator_policy(one,y,settings) if one else None
            if best1 is not None:
                rr={**best1,"adaptation_level":1,"adaptation_name":"single_knob","success":True}
            else:
                joint=[]
                for pp in pgrid:
                    rp=rel(float(pp))
                    for t in tjoint:
                        r=_process_metrics_from_release(y,float(pp),float(t),rp,calibration,settings)
                        if r["meets_targets"]: joint.append(_decorate_control_move(r,settings))
                b2=_choose_best_feasible(joint,settings) if joint else None
                if b2 is not None:
                    rr={**b2,"adaptation_level":2,"adaptation_name":"coordinated_two_knob","adaptation_variant":"pressure_plus_timing","success":True}
                else:
                    rr={**fixed,"adaptation_level":3,"adaptation_name":"outside_normal_control_envelope","adaptation_variant":"none_feasible","success":False}
        cache[key]=rr
        return rr
    return eval_y

def _v3_stress_atlas(pair, calibration, front_table, settings: EngineeringSettings, pressure_n: int, ys: np.ndarray,
                     *, conditioning_mode: str, scenario_id: str, scenario_label: str, co2_scale: float, n2_scale: float):
    evaluator=_v3_fast_stress_classifier(pair,calibration,front_table,settings,pressure_n)
    per=[]
    for y in ys:
        rr=dict(evaluator(float(y)))
        rr.update({"feed_co2_fraction":float(y),"scenario_id":scenario_id,"scenario_label":scenario_label,
                   "conditioning_mode":conditioning_mode,"co2_capacity_scale":co2_scale,"n2_capacity_scale":n2_scale})
        per.append(rr)
    sdf=pd.DataFrame(per)
    labels={0:"frozen_safe",1:"single_knob_or_less",2:"coordinated_two_knob_or_less"}
    wr=[]; ar=[]
    levs=pd.to_numeric(sdf["adaptation_level"],errors="coerce").fillna(3).to_numpy(int); yarr=sdf["feed_co2_fraction"].to_numpy(float)
    for lev,label in labels.items():
        eligible=levs<=lev
        rec={"intervention_ceiling":label,"max_adaptation_level":lev}
        for side in ("lower","upper"):
            a,b,status=_v3_transition_bracket(yarr,eligible,settings.nominal_feed_co2_fraction,side)
            if status=="interior_transition":
                def ef(y): return int(evaluator(float(y)).get("adaptation_level",3))<=lev
                val,width,nit=_v3_bisect_boundary(a,b,ef,side,tol_fraction=5e-4)
                rec[f"{side}_feed_co2_fraction"]=val; rec[f"{side}_boundary_status"]="interior_adaptive_transition"; rec[f"{side}_boundary_resolved"]=True
                ar.append({"intervention_ceiling":label,"side":side,"coarse_a":a,"coarse_b":b,"refined_boundary":val,"width":width,"iterations":nit})
            else:
                rec[f"{side}_feed_co2_fraction"]=a if np.isfinite(a) else np.nan; rec[f"{side}_boundary_status"]=status; rec[f"{side}_boundary_resolved"]=False
        rec.update({"scenario_id":scenario_id,"scenario_label":scenario_label,"conditioning_mode":conditioning_mode,
                    "co2_capacity_scale":co2_scale,"n2_capacity_scale":n2_scale,"stress_role":"non-probabilistic literature discrepancy stress"})
        wr.append(rec)
    rdf=pd.DataFrame(ar)
    if len(rdf): rdf["scenario_id"]=scenario_id; rdf["conditioning_mode"]=conditioning_mode
    return sdf,pd.DataFrame(wr),rdf



# Enrich the selected-policy rows while preserving the classified atlas.
# Re-evaluate only
# one-actuator compositions with the continuous solver; class boundaries do not change.
_stage_05_v3_prepolicy = stage_05_feed_composition_robustness
def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_05_v3_prepolicy(config,dirs,fit_payload,evidence_payload,logger)
    pair=calf20_pair(fit_payload); sel=out.get("selected",pd.DataFrame()).copy()
    if pair is not None and len(sel):
        pressure_n=int(LEVELS[config.level]["pressure_n"])
        for idx,row in sel.iterrows():
            if int(pd.to_numeric(pd.Series([row.get("adaptation_level",3)]),errors="coerce").iloc[0])!=1:
                continue
            y=float(row["feed_co2_fraction"])
            dd=solve_adaptation_levels(y,pair,out.get("calibration",{}),out.get("front_table",pd.DataFrame()),config.engineering,pressure_n)
            rr=dd[(dd["adaptation_level"]==1)&(dd["success"].astype(bool))]
            if len(rr):
                r=rr.iloc[0].to_dict()
                for k,v in r.items(): sel.at[idx,k]=v
                sel.at[idx,"robustness_class"]="single_knob_recoverable"
        out["selected"]=sel
        save_table_multi(sel,dirs["engineering"]/"minimum_intervention_selected",SAVE_MODES[config.save_mode])
        cols=[c for c in [
            "feed_co2_fraction","adaptation_level","adaptation_variant","actuator_choice_degenerate",
            "actuator_preference_class","pressure_candidate_feasible","timing_candidate_feasible",
            "pressure_candidate_effort","timing_candidate_effort","actuator_effort_delta_pressure_minus_timing",
            "pressure_candidate_p_reg_bar","timing_candidate_timing_scale","policy_selection_rule"
        ] if c in sel.columns]
        deg=sel[cols].copy()
        out["policy_degeneracy"]=deg
        save_table_multi(deg,dirs["engineering"]/"one_actuator_policy_degeneracy",SAVE_MODES[config.save_mode])
        mech=mechanism_and_control_authority(out,config.engineering)
        if len(mech):
            co=pd.to_numeric(mech.get("working_capacity_co2_mol_kg",np.nan),errors="coerce")
            nn=pd.to_numeric(mech.get("working_capacity_n2_mol_kg",np.nan),errors="coerce")
            denom=co.clip(lower=0)+nn.clip(lower=0)
            mech["bounded_n2_contamination_fraction"]=np.where(denom>EPS,nn.clip(lower=0)/denom,np.nan)
            mech["bounded_co2_release_fraction"]=np.where(denom>EPS,co.clip(lower=0)/denom,np.nan)
        out["mechanism"]=mech
        save_table_multi(mech,dirs["engineering"]/"mechanism_control_authority_map",SAVE_MODES[config.save_mode])
    return out



def _v3_fast_one_actuator_candidates(y: float, pair, calibration, front_table, settings: EngineeringSettings, n: int):
    p0=settings.nominal_regeneration_pressure_bar
    ff=front_factor(float(y),front_table)
    ads=mixture_state(float(y),settings.adsorption_pressure_bar,settings.nominal_temperature_K,pair)
    pvals=np.geomspace(settings.regeneration_pressure_floor_bar,p0,max(17,min(41,n)))
    tvals=np.linspace(settings.timing_min_scale,settings.timing_max_scale,max(21,min(61,n+8)))
    reg_cache={}
    def rel(pp):
        pp=float(pp)
        if pp not in reg_cache:
            rg=mixture_state(settings.product_reference_co2_fraction,pp,settings.nominal_temperature_K,pair)
            reg_cache[pp]={"dco2":max(ads["q_co2"]-rg["q_co2"],0.0)*ff,"dn2":max(ads["q_n2"]-rg["q_n2"],0.0)*ff,
                           "alpha_ads":ads["alpha"],"iast_status_ads":ads["iast_status"],"iast_status_reg":rg["iast_status"],"front_factor":ff}
        return reg_cache[pp]
    timing=[_process_metrics_from_release(float(y),p0,float(t),rel(p0),calibration,settings) for t in tvals]
    pressure=[_process_metrics_from_release(float(y),float(p),1.0,rel(float(p)),calibration,settings) for p in pvals]
    tb=_choose_best_feasible(timing,settings); pb=_choose_best_feasible(pressure,settings)
    c=[]
    if pb is not None: pb=dict(pb); pb["adaptation_variant"]="pressure_only"; c.append(pb)
    if tb is not None: tb=dict(tb); tb["adaptation_variant"]="timing_only"; c.append(tb)
    return _select_one_actuator_policy(c,float(y),settings)

# Replace the slow policy-enrichment wrapper with a grid-cached diagnostic that
# preserves the class topology but is practical for comprehensive single-CPU runs.
def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_05_v3_prepolicy(config,dirs,fit_payload,evidence_payload,logger)
    pair=calf20_pair(fit_payload); sel=out.get("selected",pd.DataFrame()).copy()
    if pair is not None and len(sel):
        n=int(LEVELS[config.level]["pressure_n"])
        for idx,row in sel.iterrows():
            if int(pd.to_numeric(pd.Series([row.get("adaptation_level",3)]),errors="coerce").iloc[0])!=1: continue
            r=_v3_fast_one_actuator_candidates(float(row["feed_co2_fraction"]),pair,out.get("calibration",{}),out.get("front_table",pd.DataFrame()),config.engineering,n)
            if r is not None:
                for k,v in r.items(): sel.at[idx,k]=v
                sel.at[idx,"adaptation_level"]=1; sel.at[idx,"adaptation_name"]="single_knob"; sel.at[idx,"success"]=True; sel.at[idx,"robustness_class"]="single_knob_recoverable"
        out["selected"]=sel
        save_table_multi(sel,dirs["engineering"]/"minimum_intervention_selected",SAVE_MODES[config.save_mode])
        cols=[c for c in ["feed_co2_fraction","adaptation_level","adaptation_variant","actuator_choice_degenerate","actuator_preference_class","pressure_candidate_feasible","timing_candidate_feasible","pressure_candidate_effort","timing_candidate_effort","actuator_effort_delta_pressure_minus_timing","pressure_candidate_p_reg_bar","timing_candidate_timing_scale","policy_selection_rule"] if c in sel.columns]
        deg=sel[cols].copy(); out["policy_degeneracy"]=deg
        save_table_multi(deg,dirs["engineering"]/"one_actuator_policy_degeneracy",SAVE_MODES[config.save_mode])
        mech=mechanism_and_control_authority(out,config.engineering)
        if len(mech):
            co=pd.to_numeric(mech.get("working_capacity_co2_mol_kg",np.nan),errors="coerce"); nn=pd.to_numeric(mech.get("working_capacity_n2_mol_kg",np.nan),errors="coerce"); denom=co.clip(lower=0)+nn.clip(lower=0)
            mech["bounded_n2_contamination_fraction"]=np.where(denom>EPS,nn.clip(lower=0)/denom,np.nan)
            mech["bounded_co2_release_fraction"]=np.where(denom>EPS,co.clip(lower=0)/denom,np.nan)
        out["mechanism"]=mech; save_table_multi(mech,dirs["engineering"]/"mechanism_control_authority_map",SAVE_MODES[config.save_mode])
    return out


# ===========================================================================
# Final publication analysis refinements
# ===========================================================================
# This final layer intentionally narrows the remaining work to publication
# closure: continuous one-actuator effort, dimensionless constraint geometry,
# adaptive independent-source boundaries, audited CSS validation/falsification,
# direct amplification uncertainty, and complete publication outputs.


def _v4_valid_css_rows_from_folder(folder: Path) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Return only genuine CSS result rows; priors/configuration CSVs never count."""
    rows=[]; audit=[]
    if not folder.exists(): return pd.DataFrame(),pd.DataFrame()
    for f in sorted(folder.rglob("*.csv")):
        try: d=pd.read_csv(f)
        except Exception as exc:
            audit.append({"file":str(f),"status":"unreadable","reason":str(exc)}); continue
        aliases={"feed_co2_fraction":["feed_co2_fraction","y_co2","feed_y_co2"],"purity":["purity","purity_fraction","co2_purity"],"recovery":["recovery","recovery_fraction","co2_recovery"]}
        ren={}
        for k,als in aliases.items():
            c=_find_any_column(d.columns,als)
            if c is not None: ren[c]=k
        q=d.rename(columns=ren)
        if not {"feed_co2_fraction","purity","recovery"}.issubset(q.columns):
            audit.append({"file":str(f),"status":"ignored_non_result","reason":"missing feed/purity/recovery result fields"}); continue
        # A genuine high-fidelity CSS result must also identify itself as a cyclic
        # simulation result (normally through CSS convergence / solver metadata).
        # This prevents experimental validation-target tables that happen to contain
        # feed/purity/recovery columns from being miscounted as completed CSS evidence.
        css_markers={"css_converged","solver_status","source_or_model","intervention_class","adaptation_level"}
        if not css_markers.intersection(set(map(str,q.columns))):
            audit.append({"file":str(f),"status":"ignored_non_result","reason":"missing CSS-result marker (e.g. css_converged/solver_status/source_or_model)"}); continue
        for c in ["feed_co2_fraction","purity","recovery"]: q[c]=_fraction_series(q[c])
        q=q.dropna(subset=["feed_co2_fraction","purity","recovery"]).copy()
        if q.empty:
            audit.append({"file":str(f),"status":"ignored_non_result","reason":"no complete numeric result rows"}); continue
        q["css_result_source_file"]=f.name; rows.append(q)
        audit.append({"file":str(f),"status":"accepted_css_result","reason":f"{len(q)} complete result row(s)"})
    return (pd.concat(rows,ignore_index=True,sort=False) if rows else pd.DataFrame(),pd.DataFrame(audit))


_audit_optional_literature_data_v3 = audit_optional_literature_data
def audit_optional_literature_data(config: RunConfig, dirs: Mapping[str,Path], logger: Optional[logging.Logger]=None) -> Dict[str,Any]:
    out=_audit_optional_literature_data_v3(config,dirs,logger)
    root=_optional_data_root(config)
    css_parts=[]; aud=[]
    for fd in [root/"css_validation",root/"higher_fidelity_validation"]:
        d,a=_v4_valid_css_rows_from_folder(fd); css_parts.append(len(d));
        if len(a): aud.append(a)
    dfa=out.get("audit",pd.DataFrame()).copy()
    if len(dfa) and "check" in dfa.columns:
        m=dfa["check"].astype(str).eq("high_fidelity_css_rows")
        if m.any():
            n=sum(css_parts); dfa.loc[m,"observed"]=str(n); dfa.loc[m,"status"]="INFO" if n==0 else "PASS"; dfa.loc[m,"note"]="Only files with complete feed/purity/recovery result fields count; CSS priors/configurations are excluded."
    out["audit"]=dfa; out["css_result_file_audit"]=pd.concat(aud,ignore_index=True) if aud else pd.DataFrame()
    if len(dfa): save_table_multi(dfa,dirs["audit"]/"literature_data_audit",SAVE_MODES[config.save_mode])
    if len(out["css_result_file_audit"]): save_table_multi(out["css_result_file_audit"],dirs["audit"]/"css_result_file_audit",SAVE_MODES[config.save_mode])
    return out


def ingest_css_validation(config: RunConfig, atlas_payload: Mapping[str,Any], settings: EngineeringSettings, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    root=_optional_data_root(config); parts=[]; audits=[]
    for fd in [root/"css_validation",root/"higher_fidelity_validation"]:
        d,a=_v4_valid_css_rows_from_folder(fd)
        if len(d): parts.append(d)
        if len(a): audits.append(a)
    if not parts: return {"raw":pd.DataFrame(),"comparison":pd.DataFrame(),"summary":pd.DataFrame(),"file_audit":pd.concat(audits,ignore_index=True) if audits else pd.DataFrame()}
    d=pd.concat(parts,ignore_index=True,sort=False)
    d["css_meets_targets"]=(d["purity"]>=settings.target_purity)&(d["recovery"]>=settings.target_recovery)
    sel=atlas_payload.get("selected",pd.DataFrame()); comp=[]
    for _,r in d.iterrows():
        y=float(r["feed_co2_fraction"]); pred=np.nan
        if len(sel): pred=float(sel.loc[(pd.to_numeric(sel["feed_co2_fraction"],errors="coerce")-y).abs().idxmin(),"adaptation_level"])
        actual=pd.to_numeric(pd.Series([r.get("intervention_class",r.get("adaptation_level",np.nan))]),errors="coerce").iloc[0]
        comp.append({"feed_co2_fraction":y,"css_purity":r["purity"],"css_recovery":r["recovery"],"css_meets_targets":bool(r["css_meets_targets"]),"reduced_order_nearest_intervention_level":pred,"css_reported_intervention_level":actual,"case_id":r.get("case_id",r.get("source_id","")),"css_result_source_file":r.get("css_result_source_file","")})
    comparison=pd.DataFrame(comp)
    summary=pd.DataFrame([{"n_css_rows":len(d),"n_target_feasible":int(d["css_meets_targets"].sum()),"fraction_target_feasible":float(d["css_meets_targets"].mean()),"composition_min":float(d["feed_co2_fraction"].min()),"composition_max":float(d["feed_co2_fraction"].max()),"role":"independent high-fidelity falsification; never used to calibrate reduced-order model"}])
    if logger: logger.info("External CSS result files detected: %d genuine result rows",len(d))
    return {"raw":d,"comparison":comparison,"summary":summary,"file_audit":pd.concat(audits,ignore_index=True) if audits else pd.DataFrame()}


def _v4_eval_at(y,p,t,pair,cal,front_table,settings,cache):
    k=round(float(p),10)
    if k not in cache:
        ff=front_factor(float(y),front_table); rr=release_state(float(y),float(p),pair,settings,ff); rr["front_factor"]=ff; cache[k]=rr
    return _decorate_control_move(_process_metrics_from_release(float(y),float(p),float(t),cache[k],cal,settings),settings)


def _v4_refine_single_actuator(y,pair,cal,front_table,settings):
    cache={}; p0=settings.nominal_regeneration_pressure_bar
    def margin(r): return min(float(r["purity_estimate"])-settings.target_purity,float(r["recovery_estimate"])-settings.target_recovery)
    # pressure: search from nominal toward deeper vacuum, first feasible crossing
    pscan=np.geomspace(p0,settings.regeneration_pressure_floor_bar,31)
    pr=[]; prev=_v4_eval_at(y,p0,1.0,pair,cal,front_table,settings,cache)
    if prev["meets_targets"]: pr=prev
    else:
        for p in pscan[1:]:
            cur=_v4_eval_at(y,p,1.0,pair,cal,front_table,settings,cache)
            if cur["meets_targets"]:
                a,b=float(p),float(prev["p_reg_bar"])
                for _ in range(22):
                    m=math.sqrt(a*b); rr=_v4_eval_at(y,m,1.0,pair,cal,front_table,settings,cache)
                    if rr["meets_targets"]: a=m
                    else: b=m
                pr=_v4_eval_at(y,a,1.0,pair,cal,front_table,settings,cache); break
            prev=cur
    # timing: search both directions away from 1; refine the closest feasible crossing
    tcands=[]
    for end in [settings.timing_min_scale,settings.timing_max_scale]:
        logs=np.linspace(0,math.log(end),31); prev=_v4_eval_at(y,p0,1.0,pair,cal,front_table,settings,cache)
        for lv in logs[1:]:
            t=float(math.exp(lv)); cur=_v4_eval_at(y,p0,t,pair,cal,front_table,settings,cache)
            if cur["meets_targets"]:
                a=float(prev["timing_scale"]); b=t
                # bisection in log between near-nominal infeasible and farther feasible
                lo,hi=math.log(a),math.log(b)
                for _ in range(22):
                    mid=.5*(lo+hi); rr=_v4_eval_at(y,p0,math.exp(mid),pair,cal,front_table,settings,cache)
                    if rr["meets_targets"]: hi=mid
                    else: lo=mid
                tcands.append(_v4_eval_at(y,p0,math.exp(hi),pair,cal,front_table,settings,cache)); break
            prev=cur
    tr=min(tcands,key=lambda r:(r["control_move_linf"],r["control_move_l1"])) if tcands else None
    return (dict(pr) if isinstance(pr,Mapping) else None),(dict(tr) if isinstance(tr,Mapping) else None)


_stage_05_v3_closure_base=stage_05_feed_composition_robustness
def stage_05_feed_composition_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_05_v3_closure_base(config,dirs,fit_payload,evidence_payload,logger)
    pair=calf20_pair(fit_payload); sel=out.get("selected",pd.DataFrame()).copy()
    if pair is None or sel.empty: return out
    rows=[]; cal=out.get("calibration",{}); ft=out.get("front_table",pd.DataFrame()); s=config.engineering
    for i,r in sel.iterrows():
        y=float(r["feed_co2_fraction"]); pr,tr=_v4_refine_single_actuator(y,pair,cal,ft,s)
        pe=float(pr["control_move_linf"]) if pr else np.nan; te=float(tr["control_move_linf"]) if tr else np.nan
        denom=max(min([x for x in [pe,te] if np.isfinite(x)] or [1.0]),1e-8)
        deg=bool(np.isfinite(pe) and np.isfinite(te) and abs(pe-te)<=0.05*denom)
        if np.isfinite(pe) and np.isfinite(te): pref="near_degenerate" if deg else ("pressure" if pe<te else "timing")
        elif np.isfinite(pe): pref="pressure"
        elif np.isfinite(te): pref="timing"
        else: pref="none"
        rr={"feed_co2_fraction":y,"pressure_candidate_feasible":pr is not None,"timing_candidate_feasible":tr is not None,"pressure_candidate_effort_continuous":pe,"timing_candidate_effort_continuous":te,"actuator_effort_delta_pressure_minus_timing_continuous":pe-te if np.isfinite(pe) and np.isfinite(te) else np.nan,"pressure_candidate_p_reg_bar_continuous":pr.get("p_reg_bar",np.nan) if pr else np.nan,"timing_candidate_timing_scale_continuous":tr.get("timing_scale",np.nan) if tr else np.nan,"actuator_choice_degenerate_continuous":deg,"actuator_preference_continuous":pref,"effort_definition":"normalized lexicographic control movement E*=min within minimum authority A*"}
        rows.append(rr)
        # replace selected level-1 solution only when a continuous candidate exists
        if int(r.get("adaptation_level",9))==1:
            cand=None
            if pref in {"pressure","near_degenerate"} and pr is not None: cand=pr
            if pref=="timing" and tr is not None: cand=tr
            if cand is not None:
                for k,v in cand.items(): sel.at[i,k]=v
                sel.at[i,"adaptation_variant"]="pressure_only" if cand is pr else "timing_only"
                sel.at[i,"policy_selection_rule"]="lexicographic: minimum authority A*, then continuously refined normalized control effort E*; 5% near-degeneracy retained"
    eff=pd.DataFrame(rows); out["continuous_one_actuator_effort"]=eff
    sel=sel.merge(eff,on="feed_co2_fraction",how="left",suffixes=("","_refined")); out["selected"]=sel
    save_table_multi(eff,dirs["engineering"]/"continuous_one_actuator_effort",SAVE_MODES[config.save_mode]); save_table_multi(sel,dirs["engineering"]/"minimum_intervention_selected",SAVE_MODES[config.save_mode])
    # dimensionless constraint-space trajectories
    f=out.get("fixed",pd.DataFrame()).copy(); a=sel.copy(); cs=[]
    cmax=(1-s.target_purity)/s.target_purity
    for role,d in [("frozen",f),("minimum_authority",a)]:
        if d.empty: continue
        for _,r in d.iterrows():
            pur=float(r.get("purity_estimate",np.nan)); rec=float(r.get("recovery_estimate",np.nan))
            if not np.isfinite(pur) or pur<=0: continue
            contam=(1-pur)/pur
            cs.append({"trajectory":role,"feed_co2_fraction":float(r["feed_co2_fraction"]),"contamination_ratio_n2_to_co2":contam,"normalized_contamination_constraint":contam/max(cmax,EPS),"recovery_fraction":rec,"normalized_recovery_constraint":rec/max(s.target_recovery,EPS),"feasible":bool(contam<=cmax+1e-12 and rec>=s.target_recovery-1e-12)})
    out["constraint_space"]=pd.DataFrame(cs); save_table_multi(out["constraint_space"],dirs["engineering"]/"dimensionless_constraint_space",SAVE_MODES[config.save_mode])
    return out


_stage_06_v3_closure_base=stage_06_robustness
def stage_06_robustness(config: RunConfig, dirs: Mapping[str,Path], fit_payload: Mapping[str,Any], uncertainty_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_06_v3_closure_base(config,dirs,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,logger)
    db=out.get("draw_boundaries",pd.DataFrame()).copy(); s=config.engineering
    rows=[]
    if len(db):
        fl=pd.to_numeric(db.get("frozen_lower_y"),errors="coerce"); sl=pd.to_numeric(db.get("single_or_less_lower_y"),errors="coerce"); cl=pd.to_numeric(db.get("coordinated_or_less_lower_y"),errors="coerce")
        tf=s.nominal_feed_co2_fraction-fl; ts=s.nominal_feed_co2_fraction-sl; tc=s.nominal_feed_co2_fraction-cl
        tmp=pd.DataFrame({"draw":db.get("draw",np.arange(len(db))),"frozen_lean_latitude":tf,"single_lean_latitude":ts,"coordinated_lean_latitude":tc,"single_authority_amplification":ts/tf.where(tf>1e-8),"coordinated_authority_amplification":tc/tf.where(tf>1e-8)})
        out["control_authority_amplification_draws"]=tmp
        for c in ["single_authority_amplification","coordinated_authority_amplification"]:
            v=pd.to_numeric(tmp[c],errors="coerce").dropna()
            if len(v): rows.append({"metric":c,"median":float(v.median()),"q025":float(v.quantile(.025)),"q975":float(v.quantile(.975)),"n_valid_draws":len(v),"uncertainty_method":"directly derived within each Monte Carlo draw"})
        save_table_multi(tmp,dirs["uncertainty"]/"control_authority_amplification_draws",SAVE_MODES[config.save_mode])
    out["control_authority_amplification_summary"]=pd.DataFrame(rows); save_table_multi(out["control_authority_amplification_summary"],dirs["uncertainty"]/"control_authority_amplification_summary",SAVE_MODES[config.save_mode])
    return out


_external_source_v3_base=external_equilibrium_source_sensitivity
def external_equilibrium_source_sensitivity(config: RunConfig, evidence_payload: Mapping[str,Any], settings: EngineeringSettings, fit_payload: Optional[Mapping[str,Any]]=None, logger: Optional[logging.Logger]=None) -> Dict[str,pd.DataFrame]:
    out=_external_source_v3_base(config,evidence_payload,settings,fit_payload,logger)
    folder=_optional_data_root(config)/"external_equilibrium_sensitivity"; rows=[]; audits=[]
    files=[f for f in sorted(folder.glob("*.csv")) if "scalar" not in f.name.lower() and "stress_policy" not in f.name.lower()]
    for j,f in enumerate(files):
        try: d=pd.read_csv(f)
        except Exception: continue
        if not {"adsorbate","temperature_K","pressure_bar","loading_mol_kg"}.issubset(d.columns): continue
        pair,fr=_fit_external_equilibrium_pair(d,LEVELS[config.level],config.seed+8800+j*101)
        if pair is None: continue
        ft=_anchor_front_factor_table(evidence_payload.get("summary",pd.DataFrame()),pair,settings); cal=calibrate_nominal_surrogate(pair,ft,settings,evidence_payload.get("process_anchors",pd.DataFrame()))
        ys=np.linspace(settings.feed_co2_min,settings.feed_co2_max,LEVELS[config.level]["composition_n"]); selected=[]
        for y in ys:
            ad=_solve_adaptation_levels_grid_v130(float(y),pair,cal,ft,settings,LEVELS[config.level]["pressure_n"]); ok=ad[ad["success"]==True]
            selected.append((ok.sort_values("adaptation_level").iloc[0].to_dict() if len(ok) else {"adaptation_level":3,"success":False}) | {"feed_co2_fraction":float(y)})
        rw,ra=_v3_refine_windows(pd.DataFrame(selected),pair,cal,ft,settings,int(LEVELS[config.level]["pressure_n"]))
        if len(rw):
            rw["source_file"]=f.name; rw["equilibrium_source_role"]="independent published parameterization; adaptively refined"; rows.append(rw)
        if len(ra): ra["source_file"]=f.name; audits.append(ra)
    if rows: out["boundaries"]=pd.concat(rows,ignore_index=True,sort=False)
    out["boundary_refinement"]=pd.concat(audits,ignore_index=True,sort=False) if audits else pd.DataFrame()
    return out


def _v4_css_config_from_data(config: RunConfig):
    from css_solver import CSSModelConfig
    p=_optional_data_root(config)/"css_model"/"CALF20_css_model_priors.csv"; kw={}
    if p.exists():
        d=pd.read_csv(p)
        for _,r in d.iterrows():
            name=str(r.get("parameter","")); val=pd.to_numeric(pd.Series([r.get("value")]),errors="coerce").iloc[0]
            if name in CSSModelConfig.__dataclass_fields__ and np.isfinite(val): kw[name]=float(val)
    # tune numerical depth by pipeline level; science gates are unchanged
    if config.level in {"rough","quickcheck"}: kw.update({"n_cells":4,"min_css_cycles":3,"max_css_cycles":6,"css_consecutive_cycles":2,"max_step_s":8.0,"max_pressure_step_s":1200.0,"open_boundary_velocity_cap_m_s":0.03,"integration_chunk_s":20.0,"step_wall_time_s":20.0,"case_wall_time_s":45.0})
    elif config.level=="standard": kw.update({"n_cells":6,"min_css_cycles":5,"max_css_cycles":14,"css_consecutive_cycles":2,"max_step_s":4.0,"max_pressure_step_s":1200.0,"open_boundary_velocity_cap_m_s":0.03,"integration_chunk_s":20.0,"step_wall_time_s":45.0,"case_wall_time_s":150.0})
    else: kw.update({"n_cells":8,"min_css_cycles":8,"max_css_cycles":24,"css_consecutive_cycles":3,"max_step_s":2.5,"max_pressure_step_s":1200.0,"open_boundary_velocity_cap_m_s":0.03,"integration_chunk_s":15.0,"step_wall_time_s":60.0,"case_wall_time_s":240.0})
    kw["ambient_temperature_K"]=config.engineering.nominal_temperature_K; kw["vacuum_efficiency"]=config.engineering.vacuum_efficiency
    return CSSModelConfig(**kw)


def _v4_policy_for_css_y(y,fit_payload,atlas_payload,settings):
    pair=calf20_pair(fit_payload); cal=atlas_payload.get("calibration",{}); ft=atlas_payload.get("front_table",pd.DataFrame())
    ad=solve_adaptation_levels(float(y),pair,cal,ft,settings,max(31,LEVELS["comprehensive"]["pressure_n"]))
    ok=ad[ad["success"]==True].copy()
    if len(ok): r=ok.sort_values(["adaptation_level","control_move_linf","control_move_l1"]).iloc[0].to_dict()
    else: r={"adaptation_level":3,"p_reg_bar":settings.regeneration_pressure_floor_bar,"timing_scale":settings.timing_max_scale,"success":False}
    return r


def _v4_run_css_closure(config,dirs,fit_payload,evidence_payload,atlas_payload,logger):
    from css_solver import EquilibriumTable,NonIsothermalCSSSolver,CycleOperatingPoint,run_validation_suite,assess_validation_gate
    if config.css_mode=="off": return {"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":{"status":"NOT_RUN"},"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    pair=calf20_pair(fit_payload)
    if pair is None: return {"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":{"status":"FAIL","reason":"CALF-20 equilibrium pair unavailable"},"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    eq=EquilibriumTable(lambda y,p,t: mixture_state(y,p,t,pair),p_range=(0.008,1.10),t_range=(285,315),n_y=13 if config.level in {"quickcheck","rough"} else 21,n_p=16 if config.level in {"quickcheck","rough"} else 24,n_t=5 if config.level in {"quickcheck","rough"} else 7)
    cfg=_v4_css_config_from_data(config); valp=_optional_data_root(config)/"css_model"/"Nguyen2025_same_rig_dry_validation_cases.csv"; vals=pd.read_csv(valp) if valp.exists() else pd.DataFrame()
    mesh=(3,4,5) if config.level in {"quickcheck","rough"} else ((4,6,8) if config.level=="standard" else (6,8,10))
    summ,qa,hist=run_validation_suite(eq,cfg,vals,mesh_cells=mesh,tolerance_factors=(2.0,1.0,0.5),logger=logger); gate=assess_validation_gate(summ,qa)
    save_table_multi(summ,dirs["validation"]/"css_same_rig_validation_summary",SAVE_MODES[config.save_mode]); save_table_multi(qa,dirs["validation"]/"css_numerical_qa",SAVE_MODES[config.save_mode]); atomic_json(gate,dirs["validation"]/"css_validation_gate.json")
    for k,h in hist.items(): save_table_multi(h,dirs["validation"]/f"css_history_{normalize_token(k) or 'case'}",SAVE_MODES[config.save_mode])
    logger.info("CSS same-rig validation gate: %s",gate.get("status"))
    fals=[]; fhist={}; plan=boundary_falsification_portfolio(atlas_payload,config.engineering)
    allow=gate.get("status")=="PASS" or config.css_run_on_gate_fail
    if config.css_mode in {"validate_and_falsify","full_closure"} and allow:
        solver=NonIsothermalCSSSolver(eq,cfg); b0=vals[vals["case_id"].astype(str).eq("NGUYEN2025_DRY_B0")].iloc[0] if len(vals) else None
        ratio=float(b0["intermediate_pressure_bar"]/b0["low_pressure_bar"]) if b0 is not None else 1.5; vfeed=float(b0["feed_superficial_velocity_m_s"]) if b0 is not None else 0.024
        for j,r in plan.iterrows():
            y=float(r["feed_co2_fraction"]); pol=_v4_policy_for_css_y(y,fit_payload,atlas_payload,config.engineering); preg=float(pol.get("p_reg_bar",config.engineering.nominal_regeneration_pressure_bar)); ts=float(pol.get("timing_scale",1.0)); pi=min(0.30,max(preg+0.005,ratio*preg))
            op=CycleOperatingPoint(case_id=f"CSS-{j+1:02d}",feed_co2_fraction=y,high_pressure_bar=config.engineering.adsorption_pressure_bar,intermediate_pressure_bar=pi,low_pressure_bar=preg,adsorption_time_s=config.engineering.nominal_adsorption_time_s,feed_superficial_velocity_m_s=vfeed,timing_scale=ts,source_role="P10 falsification; B0 schedule-shape prior")
            try:
                sol=solver.run_to_css(op); sm=sol["summary"]; pur=float(sm["purity_fraction"]); rec=float(sm["recovery_fraction"]); meets=pur>=config.engineering.target_purity and rec>=config.engineering.target_recovery
                fals.append({**sm,"portfolio_role":r.get("case_role",""),"portfolio_question":r.get("falsification_question",""),"reduced_predicted_level":pol.get("adaptation_level",np.nan),"reduced_policy_p_reg_bar":preg,"reduced_policy_timing_scale":ts,"css_meets_targets":meets,"validation_gate_status":gate.get("status"),"solver_status":"OK","solver_error":"","intermediate_pressure_mapping":"B0 same-rig PI/PL ratio; explicit schedule-shape prior, not exact 2023 point setting"})
                fhist[op.case_id]=sol["history"]
            except Exception as exc:
                fals.append({"case_id":op.case_id,"feed_co2_fraction":y,"portfolio_role":r.get("case_role",""),"portfolio_question":r.get("falsification_question",""),"reduced_predicted_level":pol.get("adaptation_level",np.nan),"reduced_policy_p_reg_bar":preg,"reduced_policy_timing_scale":ts,"css_meets_targets":False,"validation_gate_status":gate.get("status"),"solver_status":"FAIL","solver_error":repr(exc),"intermediate_pressure_mapping":"B0 same-rig PI/PL ratio; explicit schedule-shape prior, not exact 2023 point setting"})
        fals=pd.DataFrame(fals); save_table_multi(fals,dirs["validation"]/"css_falsification_nine_cases",SAVE_MODES[config.save_mode])
        for k,h in fhist.items(): save_table_multi(h,dirs["validation"]/f"css_falsification_history_{k}",SAVE_MODES[config.save_mode])
    else: fals=pd.DataFrame()
    # One optional cyclic disturbance sequence, only after a passing validation gate
    disturbance=[]
    if config.css_mode=="full_closure" and gate.get("status")=="PASS":
        solver=NonIsothermalCSSSolver(eq,cfg); state=None
        for seg,y in enumerate([0.15,0.12,0.18,0.15],1):
            pol=_v4_policy_for_css_y(y,fit_payload,atlas_payload,config.engineering); preg=float(pol.get("p_reg_bar",config.engineering.nominal_regeneration_pressure_bar)); ts=float(pol.get("timing_scale",1.0)); pi=min(0.30,max(preg+0.005,1.5*preg))
            op=CycleOperatingPoint(f"DIST-{seg}",y,config.engineering.adsorption_pressure_bar,pi,preg,config.engineering.nominal_adsorption_time_s,0.024,ts,source_role="dynamic feed-disturbance demonstration")
            try:
                sol=solver.run_to_css(op,Y0=state); state=sol["final_state"]; h=sol["history"].copy(); h["segment"]=seg; h["feed_co2_fraction"]=y; h["control_authority"]=pol.get("adaptation_level",np.nan); h["solver_status"]="OK"; h["solver_error"]=""; disturbance.append(h)
            except Exception as exc:
                disturbance.append(pd.DataFrame([{"segment":seg,"feed_co2_fraction":y,"control_authority":pol.get("adaptation_level",np.nan),"solver_status":"FAIL","solver_error":repr(exc)}])); break
        disturbance=pd.concat(disturbance,ignore_index=True) if disturbance else pd.DataFrame(); save_table_multi(disturbance,dirs["validation"]/"css_dynamic_15_12_18_15",SAVE_MODES[config.save_mode])
    else: disturbance=pd.DataFrame()
    return {"validation_summary":summ,"validation_qa":qa,"gate":gate,"falsification":fals,"disturbance":disturbance,"histories":hist|fhist}


def _v4_run_css_closure_isolated(config,dirs,fit_payload,evidence_payload,atlas_payload,logger):
    """Run the optional CSS closure in a separate Python process with a hard wall-time guard.

    This prevents a pathological stiff solve from blocking the primary P10 publication
    analysis indefinitely on Windows. The CSS gate is marked FAIL on timeout/crash;
    reduced-order results are retained and no high-fidelity support is claimed.
    """
    if config.css_mode=="off":
        return {"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":{"status":"NOT_RUN","reason":"css_mode_off"},"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    import pickle
    inp=Path(dirs["checkpoints"])/"_css_worker_input.pkl"
    outp=Path(dirs["checkpoints"])/"_css_worker_result.pkl"
    if outp.exists(): outp.unlink()
    payload={
        "config":{
            "input_folder":config.input_folder,"output_folder":config.output_folder,"save_mode":config.save_mode,
            "ram_mode":config.ram_mode,"level":config.level,"n_jobs":config.n_jobs,"seed":config.seed,
            "force_stage":None,"restart":False,"preflight_only":False,
            "uncertainty_draws_override":config.uncertainty_draws_override,"engineering":asdict(config.engineering),
            "css_mode":config.css_mode,"css_run_on_gate_fail":config.css_run_on_gate_fail,
        },
        "dirs":{k:str(v) for k,v in dirs.items()},
        "fit_payload":fit_payload,"evidence_payload":evidence_payload,"atlas_payload":atlas_payload,
    }
    with inp.open("wb") as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    worker=Path(__file__).resolve().with_name("css_worker.py")
    timeout={"quickcheck":30,"rough":60,"standard":300,"comprehensive":900,"exhaustive":1200}.get(config.level,1200)
    logger.info("Launching isolated CSS closure worker (hard wall-time guard %d s)",timeout)
    proc=subprocess.Popen([sys.executable,str(worker),"--input",str(inp),"--output",str(outp)],
                          cwd=str(Path(__file__).resolve().parent),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.monotonic()+timeout
    timed_out=False
    while proc.poll() is None:
        if time.monotonic()>=deadline:
            timed_out=True
            try:
                if os.name=="nt":
                    subprocess.run(["taskkill","/F","/T","/PID",str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=False)
                else:
                    proc.kill()
            except Exception:
                try: proc.kill()
                except Exception: pass
            break
        time.sleep(1.0)
    if timed_out:
        gate={"status":"FAIL","reason":"css_worker_timeout","wall_time_limit_s":timeout,
              "note":"CSS validation exceeded the hard runtime guard. Primary reduced-order P10 results are retained; no CSS support is claimed."}
        atomic_json(gate,dirs["validation"]/"css_validation_gate.json")
        logger.error("CSS closure worker exceeded %d s and was terminated; CSS gate marked FAIL",timeout)
        return {"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":gate,"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    rc=proc.returncode if proc.returncode is not None else -999
    if rc!=0 or not outp.exists():
        gate={"status":"FAIL","reason":"css_worker_process_error","returncode":int(rc),
              "note":"CSS worker failed. Inspect logs/P10_css_worker.log. Primary reduced-order P10 results are retained; no CSS support is claimed."}
        atomic_json(gate,dirs["validation"]/"css_validation_gate.json")
        logger.error("CSS closure worker failed with return code %s; CSS gate marked FAIL",rc)
        return {"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":gate,"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    with outp.open("rb") as f: result=pickle.load(f)
    logger.info("Isolated CSS closure worker completed with gate=%s",result.get("gate",{}).get("status","UNKNOWN"))
    return result


_stage_07_v3_closure_base=stage_07_validation
def stage_07_validation(config: RunConfig, dirs: Mapping[str,Path], harmonized_payload: Mapping[str,Any], fit_payload: Mapping[str,Any], evidence_payload: Mapping[str,Any], atlas_payload: Mapping[str,Any], logger: logging.Logger) -> Dict[str,Any]:
    out=_stage_07_v3_closure_base(config,dirs,harmonized_payload,fit_payload,evidence_payload,atlas_payload,logger)
    try:
        cssclosure=_v4_run_css_closure_isolated(config,dirs,fit_payload,evidence_payload,atlas_payload,logger)
    except Exception as exc:
        # CSS is an independent falsification/validation layer. A numerical CSS
        # failure must be explicit, but it must not discard the completed primary
        # reduced-order publication analysis.
        logger.exception("CSS closure layer failed; preserving reduced-order results and marking CSS gate FAIL")
        gate={"status":"FAIL","reason":"css_solver_exception","solver_error":repr(exc),"note":"Primary reduced-order P10 outputs remain valid; no CSS support is claimed."}
        atomic_json(gate,dirs["validation"]/"css_validation_gate.json")
        cssclosure={"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"gate":gate,"falsification":pd.DataFrame(),"disturbance":pd.DataFrame(),"histories":{}}
    out["css_closure"]=cssclosure
    # Preserve external-file CSS ingest separately; package-generated solver results are a distinct validation layer.
    return out


def figure_1(dirs,config,harmonized_payload,fit_payload,evidence_payload,validation_payload):
    s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.5,7.8),constrained_layout=True)
    _evidence_landscape(axs[0,0],evidence_payload,validation_payload,s); panel_label(axs[0,0],"a")
    data=harmonized_payload.get("data",pd.DataFrame()); ax=axs[0,1]
    if data.empty: empty_axis(ax,"No equilibrium data")
    else:
        for gas,ls in [("CO2","-"),("N2","--")]:
            g=data[(data["material"]=="CALF-20")&(data["adsorbate"]==gas)]; fs=fit_payload["fits"].get(("CALF-20",gas),{}); m=fs.get("best_model")
            if g.empty or not m: continue
            th=fs["models"][m]["theta"]; Ts=sorted(g["temperature_K"].unique()); Ts=[Ts[0],Ts[len(Ts)//2],Ts[-1]] if len(Ts)>=3 else Ts
            for T in Ts:
                gg=g[np.isclose(g["temperature_K"],T)]; ax.scatter(gg["pressure_bar"],gg["loading_mol_kg"],s=9,alpha=.45); pp=np.geomspace(max(float(gg["pressure_bar"].min()),1e-4),float(gg["pressure_bar"].max()),150); ax.plot(pp,model_loading(m,pp,np.full_like(pp,T),th),ls=ls,lw=1.05,label=f"{gas}, {T:g} K")
        ax.set_xscale("log"); ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Loading (mol kg$^{-1}$)"); ax.set_title("Thermodynamic foundation"); ax.legend(frameon=False,fontsize=6,ncol=2); panel_label(ax,"b")
    ax=axs[1,0]; pv=validation_payload.get("external_pilot",{}).get("raw",pd.DataFrame()).copy()
    if len(pv):
        lam=pv[pv.get("contactor_type",pd.Series("",index=pv.index)).astype(str).str.lower().eq("laminate")].copy(); rows=[]
        for y,g in lam.groupby("feed_co2_fraction"):
            rr=pd.to_numeric(g["recovery_fraction"],errors="coerce").dropna(); rows.append((100*y,100*rr.median(),100*rr.quantile(.25),100*rr.quantile(.75),len(rr)))
        rows=sorted(rows); xx=np.array([r[0] for r in rows]); med=np.array([r[1] for r in rows]); q1=np.array([r[2] for r in rows]); q3=np.array([r[3] for r in rows]); ax.errorbar(xx,med,yerr=np.vstack([med-q1,q3-med]),fmt="o-",capsize=4,lw=1.5,label="laminate")
        pac=pv[pv.get("contactor_type",pd.Series("",index=pv.index)).astype(str).str.lower().eq("packed")]
        if len(pac): ax.scatter(100*pac["feed_co2_fraction"].median(),100*pac["recovery_fraction"].median(),facecolors="none",edgecolors="#333333",s=45,label="packed, 8% context")
        ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Pilot recovery median (%, IQR)"); ax.set_title("Independent pilot composition response"); ax.legend(frameon=False,fontsize=7)
    else: empty_axis(ax,"Independent pilot evidence unavailable")
    panel_label(ax,"c")
    ax=axs[1,1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    levels=[(.11,.18,.78,.66,"$\\mathcal{U}_2$  coordinated controls",PUB_COLORS["coordinated"]),(.20,.30,.60,.42,"$\\mathcal{U}_1$  one actuator",PUB_COLORS["single"]),(.31,.42,.38,.18,"$\\mathcal{U}_0$  frozen",PUB_COLORS["frozen"])]
    for x,y,w,h,txt,c in levels:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc=c,ec="none",alpha=.13)); ax.text(x+w/2,y+h-0.055,txt,ha="center",va="top",fontsize=9)
    ax.text(.5,.10,r"$A^*(y)=\min\{\dim(u):P\geq P^*,R\geq R^*\}$",ha="center",fontsize=9.5); ax.text(.5,.04,r"$E^*(y)=\min_{u\in\mathcal{U}_{A^*}}\|u-u_0\|_{\mathrm{norm}}$",ha="center",fontsize=9.5); ax.set_title("Nested control authority and effort"); panel_label(ax,"d")
    save_figure(fig,dirs["fig_main"]/"Figure_1_evidence_and_control_authority_framework",SAVE_MODES[config.save_mode]); save_panel_source(pv,dirs["source_main"]/"Figure_1_source_data",SAVE_MODES[config.save_mode])


def figure_2(dirs,config,atlas_payload):
    s=config.engineering; intrinsic=atlas_payload.get("intrinsic",pd.DataFrame()); mech=atlas_payload.get("mechanism",pd.DataFrame()); cs=atlas_payload.get("constraint_space",pd.DataFrame()); fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if intrinsic.empty: [empty_axis(ax,"No mechanism data") for ax in axs.flat]
    else:
        x=100*intrinsic["feed_co2_fraction"]; axs[0,0].plot(x,intrinsic["q_co2_mol_kg"],label="CO$_2$"); axs[0,0].plot(x,intrinsic["q_n2_mol_kg"],label="N$_2$"); axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)"); axs[0,0].legend(frameon=False)
        axs[0,1].plot(x,intrinsic["adsorbed_x_co2"]); axs[0,1].plot(x,intrinsic["feed_co2_fraction"],ls="--",lw=.8,color="#555555"); axs[0,1].set_ylabel("Adsorbed-phase CO$_2$ fraction")
        ax=axs[1,0]
        if len(cs):
            for role,ls in [("frozen","-"),("minimum_authority","--")]:
                g=cs[cs["trajectory"]==role].sort_values("feed_co2_fraction"); ax.plot(g["normalized_contamination_constraint"],g["normalized_recovery_constraint"],ls=ls,lw=1.6,label=role.replace("_"," "))
            ax.axvline(1,ls=":",color="#444444"); ax.axhline(1,ls=":",color="#444444"); ax.fill_between([0,1],[1,1],[max(1.8,ax.get_ylim()[1] if ax.get_ylim()[1]>1 else 1.8)]*2,alpha=.06); ax.set_xlabel(r"normalized contamination $[(1-P)/P]/[(1-P^*)/P^*]$"); ax.set_ylabel(r"normalized recovery $R/R^*$"); ax.set_title("Dimensionless process-constraint space"); ax.legend(frameon=False,fontsize=7)
        else: empty_axis(ax,"Constraint-space data unavailable")
        ax=axs[1,1]
        if len(mech):
            xd=100*mech["feed_co2_fraction"]; ax.plot(xd,mech["thermodynamic_pressure_ratio"],lw=1.7); ax.axhline(1,ls="--",lw=.9,color="#333333"); ax.set_yscale("log"); ax.set_ylabel(r"regeneration indicator $\Pi_{reg}$")
        for i,ax in enumerate(axs.flat):
            if i!=2: _decorate_composition_axis(ax,s)
            panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Competitive mixture loading"); axs[0,1].set_title("Adsorbed-phase enrichment"); axs[1,1].set_title("Regeneration driving-force indicator")
    save_figure(fig,dirs["fig_main"]/"Figure_2_mechanism_and_constraint_switching",SAVE_MODES[config.save_mode]); save_panel_source(cs if len(cs) else mech,dirs["source_main"]/"Figure_2_source_data",SAVE_MODES[config.save_mode])


def figure_4(dirs,config,atlas_payload):
    d=atlas_payload.get("selected",pd.DataFrame()).copy(); e=atlas_payload.get("continuous_one_actuator_effort",pd.DataFrame()).copy(); s=config.engineering; fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if d.empty or e.empty: [empty_axis(ax,"No control-effort results") for ax in axs.flat]
    else:
        x=100*e["feed_co2_fraction"]; axs[0,0].plot(x,e["pressure_candidate_effort_continuous"],label="pressure-only $E_P$"); axs[0,0].plot(x,e["timing_candidate_effort_continuous"],label="timing-only $E_t$"); axs[0,0].set_ylabel("Normalized single-actuator effort"); axs[0,0].legend(frameon=False,fontsize=7)
        delta=e["actuator_effort_delta_pressure_minus_timing_continuous"]; axs[0,1].plot(x,delta,lw=1.5); axs[0,1].axhline(0,ls="--",lw=.8,color="#444444"); axs[0,1].fill_between(x,-0.05,0.05,alpha=.08); axs[0,1].set_ylabel(r"$E_P-E_t$"); axs[0,1].set_title("Actuator competition and degeneracy")
        fixed=atlas_payload.get("fixed",pd.DataFrame()); base=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"]); wr=100*(d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(base,EPS)-1); axs[1,0].plot(100*d["feed_co2_fraction"],wr); axs[1,0].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,0].set_ylabel("Change in evacuation-work proxy (%)")
        basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"]); pr=100*(d["productivity_mol_kg_h"]/np.maximum(basep,EPS)-1); axs[1,1].plot(100*d["feed_co2_fraction"],pr); axs[1,1].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,1].set_ylabel("Change in productivity (%)")
        for i,ax in enumerate(axs.flat): _decorate_composition_axis(ax,s); _class_strip(ax,d); panel_label(ax,"abcd"[i])
        axs[0,0].set_title("Continuous one-actuator control effort"); axs[1,0].set_title("Resource consequence"); axs[1,1].set_title("Productivity consequence")
    save_figure(fig,dirs["fig_main"]/"Figure_4_minimum_intervention_policy",SAVE_MODES[config.save_mode]); save_panel_source(e.merge(d,on="feed_co2_fraction",how="left",suffixes=("","_selected")) if len(e) else d,dirs["source_main"]/"Figure_4_source_data",SAVE_MODES[config.save_mode])


_stage_08_v4_base=stage_08_tables
def stage_08_tables(config: RunConfig, dirs: Mapping[str,Path], audit_payload, harmonized_payload, fit_payload, uncertainty_payload, evidence_payload, atlas_payload, robustness_payload, validation_payload, logger) -> Dict[str,Any]:
    out=_stage_08_v4_base(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger)
    css=validation_payload.get("css_closure",{})
    extras={"Table_S12e_continuous_one_actuator_effort":atlas_payload.get("continuous_one_actuator_effort",pd.DataFrame()),"Table_S12f_dimensionless_constraint_space":atlas_payload.get("constraint_space",pd.DataFrame()),"Table_S16c_control_authority_amplification_uncertainty":robustness_payload.get("control_authority_amplification_summary",pd.DataFrame()),"Table_S20b2_external_source_adaptive_refinement":validation_payload.get("external_source_sensitivity",{}).get("boundary_refinement",pd.DataFrame()),"Table_S22_css_same_rig_validation":css.get("validation_summary",pd.DataFrame()),"Table_S23_css_numerical_qa":css.get("validation_qa",pd.DataFrame()),"Table_S24_css_nine_falsification_cases":css.get("falsification",pd.DataFrame()),"Table_S25_css_dynamic_disturbance":css.get("disturbance",pd.DataFrame())}
    for name,df in extras.items(): save_table_multi(df,dirs["tables_si"]/name,SAVE_MODES[config.save_mode]); out["si"][name]=df
    # Publication claims are governed by the same-rig CSS validation gate, never merely
    # by the presence of generated solver rows. Exploratory cases after a failed gate
    # remain explicitly non-validating evidence.
    cm=out.get("claim_evidence_matrix",pd.DataFrame()).copy()
    gate=css.get("gate",{"status":"NOT_RUN"}); gs=str(gate.get("status","NOT_RUN")).upper()
    fals=css.get("falsification",pd.DataFrame()); dist=css.get("disturbance",pd.DataFrame())
    if len(cm):
        m=cm["claim_id"].astype(str).eq("C6")
        if gs=="PASS" and isinstance(fals,pd.DataFrame) and len(fals):
            cm.loc[m,"primary_evidence"]="Table S22-S24; Figures S18-S19"
            cm.loc[m,"validation_layer"]="same-rig non-isothermal CSS validation gate + targeted cross-fidelity falsification"
            cm.loc[m,"status"]="cross-fidelity falsification evaluated after PASS gate; interpret exact class/order from Table S24"
        elif gs=="PASS":
            cm.loc[m,"primary_evidence"]="Table S22-S23; Figure S18"
            cm.loc[m,"validation_layer"]="same-rig non-isothermal CSS validation gate"
            cm.loc[m,"status"]="same-rig CSS gate passed; nine-case falsification not yet evaluated"
        elif gs=="FAIL":
            cm.loc[m,"primary_evidence"]="Table S22-S23; Figure S18"
            cm.loc[m,"validation_layer"]="same-rig non-isothermal CSS validation gate"
            cm.loc[m,"status"]="NOT SUPPORTED: CSS validation gate failed; any boundary-case simulations are exploratory only"
        else:
            cm.loc[m,"status"]="not tested by validated CSS"
        extra_claims=[
            {"claim_id":"C10","claim":"The non-isothermal CSS solver meets the predeclared same-rig validation and numerical-independence gate.","primary_evidence":"Table S22-S23; CSS_VALIDATION_GATE.json","validation_layer":"published same-rig dry cycles + CSS convergence + mass closure + mesh/tolerance tests","status":"supported" if gs=="PASS" else ("NOT SUPPORTED: validation gate failed" if gs=="FAIL" else "not run")},
            {"claim_id":"C11","claim":"A feed-aware minimum-authority policy remains effective during a 15→12→18→15% disturbance sequence.","primary_evidence":"Table S25; Figure S20","validation_layer":"non-isothermal dynamic CSS disturbance demonstration","status":"evaluated after PASS gate" if isinstance(dist,pd.DataFrame) and len(dist) and gs=="PASS" else "not supported/not run; requires PASS CSS gate"},
        ]
        cm=cm[~cm["claim_id"].astype(str).isin(["C10","C11"])]
        cm=pd.concat([cm,pd.DataFrame(extra_claims)],ignore_index=True)
        save_table_multi(cm,dirs["tables_si"]/"Table_S1g_claim_evidence_matrix",SAVE_MODES[config.save_mode])
        out["claim_evidence_matrix"]=cm; out["si"]["Table_S1g_claim_evidence_matrix"]=cm
    return out


def _v4_css_figures(dirs,config,validation_payload):
    css=validation_payload.get("css_closure",{}); summ=css.get("validation_summary",pd.DataFrame()); fals=css.get("falsification",pd.DataFrame()); dist=css.get("disturbance",pd.DataFrame())
    if len(summ):
        fig,axs=plt.subplots(1,2,figsize=(9.8,4.3),constrained_layout=True)
        for ax,k,label in [(axs[0],"purity_fraction","Purity"),(axs[1],"recovery_fraction","Recovery")]:
            ex=pd.to_numeric(summ[f"exp_{k}"],errors="coerce"); si=pd.to_numeric(summ[f"sim_{k}"],errors="coerce"); ax.scatter(100*ex,100*si,s=45); lo=min(100*ex.min(),100*si.min())-2; hi=max(100*ex.max(),100*si.max())+2; ax.plot([lo,hi],[lo,hi],ls="--",color="#555555"); ax.set_xlabel(f"Experimental {label} (%)"); ax.set_ylabel(f"CSS {label} (%)"); ax.set_title(f"Same-rig dry validation: {label.lower()}")
        save_figure(fig,dirs["fig_si"]/"Figure_S18_css_same_rig_validation",SAVE_MODES[config.save_mode]); atomic_csv(summ,dirs["source_si"]/"Figure_S18_source_data.csv",index=False)
    if len(fals):
        fig,ax=plt.subplots(figsize=(9.8,4.8),constrained_layout=True); x=100*pd.to_numeric(fals["feed_co2_fraction"],errors="coerce"); ax.scatter(x,100*fals["purity_fraction"],label="CSS purity"); ax.scatter(x,100*fals["recovery_fraction"],label="CSS recovery"); ax.axhline(100*config.engineering.target_purity,ls="--",lw=.8); ax.axhline(100*config.engineering.target_recovery,ls=":",lw=.8); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("KPI (%)"); ax.set_title("Nine targeted CSS falsification cases"); ax.legend(frameon=False); save_figure(fig,dirs["fig_si"]/"Figure_S19_css_falsification",SAVE_MODES[config.save_mode]); atomic_csv(fals,dirs["source_si"]/"Figure_S19_source_data.csv",index=False)
    if len(dist):
        fig,axs=plt.subplots(2,1,figsize=(9.8,6.0),constrained_layout=True,sharex=True); xx=np.arange(len(dist)); axs[0].plot(xx,100*dist["purity_fraction"],label="purity"); axs[0].plot(xx,100*dist["recovery_fraction"],label="recovery"); axs[0].axhline(100*config.engineering.target_purity,ls="--",lw=.8); axs[0].axhline(100*config.engineering.target_recovery,ls=":",lw=.8); axs[0].legend(frameon=False); axs[0].set_ylabel("KPI (%)"); axs[1].step(xx,100*dist["feed_co2_fraction"],where="mid"); axs[1].set_ylabel("Feed CO$_2$ (%)"); axs[1].set_xlabel("Cycle index across disturbance sequence"); save_figure(fig,dirs["fig_si"]/"Figure_S20_css_dynamic_disturbance",SAVE_MODES[config.save_mode]); atomic_csv(dist,dirs["source_si"]/"Figure_S20_source_data.csv",index=False)


_stage_09_v4_base=stage_09_figures
def stage_09_figures(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger):
    out=_stage_09_v4_base(config,dirs,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,logger); _v4_css_figures(dirs,config,validation_payload); out["si_figures"]=sorted(p.name for p in dirs["fig_si"].glob("Figure_*")); return out


_stage_10_v4_base=stage_10_finalize
def stage_10_finalize(config,dirs,input_hash,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,tables_payload,figures_payload,logger):
    out=_stage_10_v4_base(config,dirs,input_hash,audit_payload,harmonized_payload,fit_payload,uncertainty_payload,evidence_payload,atlas_payload,robustness_payload,validation_payload,tables_payload,figures_payload,logger)
    css=validation_payload.get("css_closure",{}); gate=css.get("gate",{"status":"NOT_RUN"}); atomic_json(gate,dirs["final"]/"CSS_VALIDATION_GATE.json")
    for name,df in [("CSS_SAME_RIG_VALIDATION",css.get("validation_summary",pd.DataFrame())),("CSS_NUMERICAL_QA",css.get("validation_qa",pd.DataFrame())),("CSS_NINE_FALSIFICATION_CASES",css.get("falsification",pd.DataFrame())),("CSS_DYNAMIC_15_12_18_15",css.get("disturbance",pd.DataFrame())),("CONTROL_AUTHORITY_AMPLIFICATION_UNCERTAINTY",robustness_payload.get("control_authority_amplification_summary",pd.DataFrame()))]: save_table_multi(df,dirs["final"]/name,SAVE_MODES[config.save_mode])
    status="# P10 analysis status\n\n"+f"CSS validation gate: **{gate.get('status','NOT_RUN')}**\n\n"+"The reduced-order science is frozen in this release. High-fidelity claims are permitted only if the CSS validation gate passes; otherwise CSS outputs remain exploratory and the claim-evidence matrix must state the open gap.\n"
    atomic_text(dirs["final"]/"PUBLICATION_ANALYSIS_STATUS.md",status)
    return out



def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    script_dir=Path(__file__).resolve().parent; p=argparse.ArgumentParser(description="Restartable P10 CALF-20 feed-composition robustness and minimal-adaptation pipeline.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-folder",default=str(DATA_ROOT/"input_folder")); p.add_argument("--output-folder",default=str(script_dir/"results_local")); p.add_argument("--save-mode",choices=list(SAVE_MODES),default="balanced"); p.add_argument("--ram-mode",choices=list(RAM_MODES),default="light"); p.add_argument("--level",choices=list(LEVELS),default="comprehensive"); p.add_argument("--n-jobs",type=int,default=1); p.add_argument("--seed",type=int,default=20260921); p.add_argument("--force-stage",choices=STAGES,default=None); p.add_argument("--restart",action="store_true"); p.add_argument("--preflight-only",action="store_true",help="Audit parser-ready inputs, evidence anchors and output writability, then stop before fitting")
    p.add_argument("--uncertainty-draws",type=int,default=None,help="Override the feed-composition uncertainty draw count; use 1000-2000 for the final frozen manuscript run"); p.add_argument("--fresh-run",action="store_true",help="Delete only this run\'s designated output folder before starting. Use for a clean from-scratch calculation.")
    p.add_argument("--nominal-temperature-K",type=float,default=295.15); p.add_argument("--adsorption-pressure-bar",type=float,default=0.97); p.add_argument("--nominal-y-co2",type=float,default=0.15); p.add_argument("--feed-y-min",type=float,default=0.02); p.add_argument("--feed-y-max",type=float,default=0.40)
    p.add_argument("--primary-evidence-y-min",type=float,default=0.05); p.add_argument("--primary-evidence-y-max",type=float,default=0.30)
    p.add_argument("--purity-threshold-sensitivity",default="0.90,0.95,0.97",help="Comma-separated purity thresholds for robustness analysis")
    p.add_argument("--recovery-threshold-sensitivity",default="0.85,0.88,0.90",help="Comma-separated recovery thresholds for robustness analysis")
    p.add_argument("--p-reg-nominal-bar",type=float,default=0.05); p.add_argument("--p-reg-floor-bar",type=float,default=0.01); p.add_argument("--adsorption-time-s",type=float,default=120.0); p.add_argument("--nonfeed-cycle-time-s",type=float,default=180.0); p.add_argument("--nominal-purity-anchor",type=float,default=0.955); p.add_argument("--nominal-recovery-anchor",type=float,default=0.881); p.add_argument("--target-purity",type=float,default=0.95); p.add_argument("--target-recovery",type=float,default=0.88); p.add_argument("--vacuum-efficiency",type=float,default=0.60); p.add_argument("--class-probability-threshold",type=float,default=0.80)
    p.add_argument("--css-mode",choices=["off","validate","validate_and_falsify","full_closure"],default="off",help="Optional high-fidelity CSS diagnostic layer. The primary analysis defaults to off because the same-rig CSS branch remains numerically/scientifically unresolved; full_closure is opt-in only.")
    p.add_argument("--css-run-on-gate-fail",action="store_true",help="Run nine falsification cases even when the independent same-rig CSS validation gate fails; outputs remain labelled exploratory.")
    return p.parse_args(argv)


def _parse_threshold_tuple(value: str, name: str) -> Tuple[float, ...]:
    try:
        vals=tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    except Exception as exc:
        raise ValueError(f"Could not parse {name}: {value}") from exc
    if not vals or any((v<=0 or v>1) for v in vals):
        raise ValueError(f"{name} values must all lie in (0,1]")
    return vals


def build_config(args) -> RunConfig:
    if not (0<args.feed_y_min<args.feed_y_max<1): raise ValueError("Feed-composition bounds must satisfy 0 < min < max < 1")
    if not (args.feed_y_min<=args.nominal_y_co2<=args.feed_y_max): raise ValueError("Nominal yCO2 must lie inside the feed-composition grid")
    if not (args.feed_y_min<=args.primary_evidence_y_min<args.primary_evidence_y_max<=args.feed_y_max):
        raise ValueError("Primary evidence window must lie inside the exploratory feed-composition envelope")
    if not (args.primary_evidence_y_min<=args.nominal_y_co2<=args.primary_evidence_y_max):
        raise ValueError("Nominal yCO2 must lie inside the primary evidence window")
    if not (0<args.p_reg_floor_bar<=args.p_reg_nominal_bar<args.adsorption_pressure_bar): raise ValueError("Require 0 < p_reg_floor <= p_reg_nominal < adsorption pressure")
    for v,name in [(args.target_purity,"target purity"),(args.target_recovery,"target recovery"),(args.nominal_purity_anchor,"nominal purity"),(args.nominal_recovery_anchor,"nominal recovery")]:
        if not (0<v<=1): raise ValueError(f"{name} must be in (0,1]")
    pgrid=_parse_threshold_tuple(args.purity_threshold_sensitivity,"purity threshold sensitivity")
    rgrid=_parse_threshold_tuple(args.recovery_threshold_sensitivity,"recovery threshold sensitivity")
    e=EngineeringSettings(nominal_temperature_K=args.nominal_temperature_K,adsorption_pressure_bar=args.adsorption_pressure_bar,nominal_feed_co2_fraction=args.nominal_y_co2,feed_co2_min=args.feed_y_min,feed_co2_max=args.feed_y_max,primary_evidence_feed_min=args.primary_evidence_y_min,primary_evidence_feed_max=args.primary_evidence_y_max,nominal_regeneration_pressure_bar=args.p_reg_nominal_bar,regeneration_pressure_floor_bar=args.p_reg_floor_bar,nominal_adsorption_time_s=args.adsorption_time_s,fixed_nonfeed_cycle_time_s=args.nonfeed_cycle_time_s,nominal_purity_anchor=args.nominal_purity_anchor,nominal_recovery_anchor=args.nominal_recovery_anchor,target_purity=args.target_purity,target_recovery=args.target_recovery,purity_threshold_sensitivity=pgrid,recovery_threshold_sensitivity=rgrid,vacuum_efficiency=args.vacuum_efficiency,class_probability_threshold=args.class_probability_threshold)
    if args.uncertainty_draws is not None and args.uncertainty_draws < 8:
        raise ValueError("--uncertainty-draws must be >= 8 when supplied")
    return RunConfig(input_folder=str(Path(args.input_folder).resolve()),output_folder=str(Path(args.output_folder).resolve()),save_mode=args.save_mode,ram_mode=args.ram_mode,level=args.level,n_jobs=args.n_jobs,seed=args.seed,force_stage=args.force_stage,restart=bool(args.restart),preflight_only=bool(args.preflight_only),uncertainty_draws_override=args.uncertainty_draws,engineering=e,css_mode=str(args.css_mode),css_run_on_gate_fail=bool(args.css_run_on_gate_fail))


def _run_or_load(cp,stage,fn,*args,summary_fn=None,**kwargs):
    if cp.is_done(stage): return cp.load(stage)
    payload=fn(*args,**kwargs); cp.save(stage,payload,summary=summary_fn(payload) if summary_fn else {}); return payload


def main(argv: Optional[Sequence[str]]=None) -> int:
    args=parse_args(argv)
    if bool(getattr(args,"fresh_run",False)):
        _v3_safe_clean_output_folder(Path(args.output_folder).resolve())
    config=build_config(args); input_folder=Path(config.input_folder); output_folder=Path(config.output_folder); create_input_folder_guidance(input_folder)
    try:
        output_folder.relative_to(input_folder); raise ValueError("The output folder must not be inside input_folder.")
    except ValueError as exc:
        if str(exc).startswith("The output folder"): raise
    dirs=ensure_output_tree(output_folder); logger=setup_logging(dirs["logs"]); logger.info("P10 reduced-order analysis started (engine %s)",PIPELINE_VERSION); logger.info("Modes save=%s RAM=%s level=%s n_jobs=%d resolved=%d",config.save_mode,config.ram_mode,config.level,config.n_jobs,resolve_n_jobs(config.n_jobs,config.ram_mode))
    files=list_input_files(input_folder); _,input_hash=build_input_fingerprint(files,input_folder)
    scientific_payload = config.scientific_hash_payload()
    config_hash = stable_json_hash({"scientific": scientific_payload, "save_mode": config.save_mode})
    legacy_science_hashes: List[str] = []
    for legacy in LEGACY_SCIENCE_RELEASES:
        legacy_payload = dict(scientific_payload)
        legacy_payload["pipeline_version"] = legacy["pipeline_version"]
        legacy_payload["pipeline_source_sha256"] = legacy["source_sha256"]
        legacy_science_hashes.append(stable_json_hash({"scientific": legacy_payload, "save_mode": config.save_mode}))
    cp=CheckpointManager(dirs,config_hash,input_hash,logger,legacy_science_config_hashes=legacy_science_hashes)
    if config.restart: cp.invalidate_from(STAGES[0])
    elif config.force_stage: cp.invalidate_from(config.force_stage)
    t0=time.time()
    try:
        audit=_run_or_load(cp,"00_audit",stage_00_audit,config,dirs,logger,summary_fn=lambda x:{"n_files":len(x["fingerprints"]),"input_hash":x["input_hash"]})
        literature_audit=audit_optional_literature_data(config,dirs,logger)
        if literature_audit.get("summary",{}).get("status")=="FAIL": raise ValueError("Matched literature-data integrity audit failed; inspect audit/literature_data_audit.csv before continuing.")
        harm=_run_or_load(cp,"01_harmonize",stage_01_harmonize,config,dirs,audit,logger,summary_fn=lambda x:{"n_rows":len(x["data"])})
        evidence=_run_or_load(cp,"04_composition_evidence",stage_04_composition_evidence,config,dirs,audit,logger,summary_fn=lambda x:{"n_breakthrough_points":len(x["traces"]),"n_anchor_traces":len(x["summary"]),"n_metadata_composition_anchors":len(x.get("composition_anchors",[])),"n_process_anchors":len(x["process_anchors"])})
        config.engineering = resolve_engineering_settings_from_anchors(config.engineering, evidence.get("process_anchors", pd.DataFrame()), logger)
        if config.preflight_only:
            preflight={
                "pipeline_version":PIPELINE_VERSION,"input_folder":str(input_folder),"output_folder":str(output_folder),
                "input_files":len(files),"harmonized_equilibrium_rows":len(harm.get("data",pd.DataFrame())),
                "metadata_composition_anchors":len(evidence.get("composition_anchors",pd.DataFrame())),
                "breakthrough_trace_rows":len(evidence.get("traces",pd.DataFrame())),"process_anchor_rows":len(evidence.get("process_anchors",pd.DataFrame())),
                "literature_data_audit_status":literature_audit.get("summary",{}).get("status","NOT_RUN"),"literature_data_audit_checks":literature_audit.get("summary",{}).get("n_checks",0),
                "status":"PASS" if len(harm.get("data",pd.DataFrame()))>0 and len(evidence.get("process_anchors",pd.DataFrame()))>0 and literature_audit.get("summary",{}).get("status","PASS")!="FAIL" else "CHECK_INPUTS",
            }
            atomic_json(preflight,dirs["audit"] / "preflight_summary.json")
            logger.info("Preflight-only mode complete: %s",preflight["status"]); print(f"\nPreflight complete. Summary: {dirs['audit']/'preflight_summary.json'}"); return 0 if preflight["status"]=="PASS" else 2
        fits=_run_or_load(cp,"02_fit_models",stage_02_fit_models,config,dirs,harm,logger,summary_fn=lambda x:{"n_fitted_groups":len(x["fits"])})
        unc=_run_or_load(cp,"03_uncertainty",stage_03_uncertainty,config,dirs,harm,fits,logger,summary_fn=lambda x:{"n_bootstrap_sets":len(x["samples"])})
        atlas=_run_or_load(cp,"05_feed_composition_robustness",stage_05_feed_composition_robustness,config,dirs,fits,evidence,logger,summary_fn=lambda x:{"n_composition_states":len(x["intrinsic"]),"n_intervention_rows":len(x["adaptation"])})
        robust=_run_or_load(cp,"06_robustness",stage_06_robustness,config,dirs,fits,unc,evidence,atlas,logger,summary_fn=lambda x:{"n_classified":len(x["classification"])})
        validation=_run_or_load(cp,"07_validation",stage_07_validation,config,dirs,harm,fits,evidence,atlas,logger,summary_fn=lambda x:{"n_breakthrough_validation":len(x["breakthrough"])})
        if config.level=="quickcheck":
            qc={"pipeline_version":PIPELINE_VERSION,"status":"PASS","note":"quickcheck validates scientific stages 00-07 only; publication tables/figures are intentionally skipped","output_folder":str(output_folder)}
            atomic_json(qc,dirs["final"] / "quickcheck_summary.json")
            logger.info("Quickcheck completed successfully through stage 07_validation")
            print(f"\nQuickcheck complete. Summary: {dirs['final']/'quickcheck_summary.json'}"); return 0
        tables=_run_or_load(cp,"08_tables",stage_08_tables,config,dirs,audit,harm,fits,unc,evidence,atlas,robust,validation,logger,summary_fn=lambda x:{"n_main":len(x["main"]),"n_si":len(x["si"])})
        figures=_run_or_load(cp,"09_figures",stage_09_figures,config,dirs,audit,harm,fits,unc,evidence,atlas,robust,validation,logger,summary_fn=lambda x:{"n_main_files":len(x["main_figures"]),"n_si_files":len(x["si_figures"])})
        _run_or_load(cp,"10_finalize",stage_10_finalize,config,dirs,input_hash,audit,harm,fits,unc,evidence,atlas,robust,validation,tables,figures,logger,summary_fn=lambda x:{"completed":True})
    except Exception:
        logger.exception("P10 pipeline failed. Completed checkpoints remain reusable."); logger.error("Fix the issue and rerun the identical command to resume."); return 1
    logger.info("P10 completed successfully in %.1f s",time.time()-t0)
    if len(files)==0: logger.warning("No scientific input data were present; templates/audits were created.")
    print(f"\nCompleted. Results: {output_folder}"); print(f"Reusable bundle: {dirs['final']/'P10_feed_composition_robustness_complete_results.pkl'}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
