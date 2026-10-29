#!/usr/bin/env python
"""Run the optional high-fidelity cyclic validation workflow."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE_DIR = ROOT / "src" / "p10"
PATH = MODULE_DIR / "css_validation.py"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
spec = importlib.util.spec_from_file_location("p10_css_validation_engine", PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {PATH}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    raise SystemExit(mod.main())
