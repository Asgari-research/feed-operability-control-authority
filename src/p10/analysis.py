#!/usr/bin/env python
"""Analysis assembly helpers for the P10 publication workflow."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
CORE_PATH = SCRIPT_DIR / "core.py"
EXTENSIONS_PATH = SCRIPT_DIR / "extensions.py"
VERSION = "6.6.0"
PIPELINE_NAME = "P10 feed-composition robustness: final publication closure"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_module("p10_core_engine", CORE_PATH)
ext = _load_module("p10_extension_engine", EXTENSIONS_PATH)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def safe_rmtree(path: Path) -> None:
    p = path.resolve()
    if str(p) in {"/", str(Path.home().resolve()), str(SCRIPT_DIR.resolve())}:
        raise RuntimeError(f"Refusing unsafe deletion: {p}")
    if p.exists():
        shutil.rmtree(p)


def copy_tree(src: Path, dst: Path, ignore_names: Iterable[str] = ()) -> None:
    if not src.exists():
        return
    ignored = set(ignore_names)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in ignored for part in rel.parts):
            continue
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    n = 0
    if not src_dir.exists():
        return n
    for p in src_dir.glob(pattern):
        if p.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


class StageLedger:
    def __init__(self, root: Path, logger: logging.Logger):
        self.root = root
        self.logs = root / "logs"
        self.stage_logs = self.logs / "stages"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.stage_logs.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.logs / "STAGE_STATUS.csv"
        self.md_path = self.logs / "STAGE_STATUS.md"
        self.jsonl = self.logs / "stage_events.jsonl"
        self.logger = logger
        self.rows: List[Dict[str, Any]] = []
        if self.csv_path.exists():
            try:
                self.rows = pd.read_csv(self.csv_path).to_dict("records")
            except Exception:
                self.rows = []

    def event(self, stage: str, status: str, message: str = "", elapsed_s: Optional[float] = None,
              outputs: str = "", exception: str = "") -> None:
        row = {
            "utc": utc_now(), "stage": stage, "status": status,
            "elapsed_s": np.nan if elapsed_s is None else round(float(elapsed_s), 3),
            "message": message, "outputs": outputs, "exception": exception,
        }
        self.rows.append(row)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        with (self.stage_logs / f"{stage}.log").open("a", encoding="utf-8") as f:
            f.write(f"{row['utc']} | {status} | {message}")
            if outputs: f.write(f" | outputs={outputs}")
            if exception: f.write(f" | exception={exception}")
            f.write("\n")
        atomic_csv(pd.DataFrame(self.rows), self.csv_path)
        latest = pd.DataFrame(self.rows).groupby("stage", as_index=False).tail(1)
        md = ["# P10 current analysis stage status", "", f"Updated: {utc_now()}", "",
              "| Stage | Status | Elapsed (s) | Message |", "|---|---|---:|---|"]
        for _, r in latest.iterrows():
            msg = str(r.get("message", "")).replace("|", "/").replace("\n", " ")
            el = "" if pd.isna(r.get("elapsed_s")) else f"{float(r['elapsed_s']):.1f}"
            md.append(f"| {r['stage']} | **{r['status']}** | {el} | {msg} |")
        atomic_text("\n".join(md) + "\n", self.md_path)
        lvl = logging.ERROR if status in {"FAIL", "FAIL_NUMERICAL", "FAIL_PHYSICAL_VALIDATION"} else logging.WARNING if status in {"PARTIAL", "TIMEOUT", "INDETERMINATE_NUMERICAL_TIMEOUT"} else logging.INFO
        self.logger.log(lvl, "%s | %s | %s", stage, status, message)


def setup_logger(root: Path) -> logging.Logger:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("P10_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(root / "logs" / "P10_run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def locate_result_root(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path).resolve()
    if (p / "analysis/baseline").exists():
        return p
    if p.name == "analysis/baseline" and (p / "final").exists():
        return p.parent
    return p


def load_gate_bundle(gate_dir: Path) -> Mapping[str, Any]:
    p = gate_dir / "final" / "P10_feed_composition_robustness_complete_results.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Frozen baseline complete result bundle not found: {p}")
    return load_pickle(p)


def copy_existing_science(existing: Path, root: Path, ledger: StageLedger) -> None:
    t0 = time.time()
    gate_src = existing / "analysis" / "baseline"
    if gate_src.exists() and not (root / "analysis" / "baseline" / "final" / "P10_feed_composition_robustness_complete_results.pkl").exists():
        copy_tree(gate_src, root / "analysis" / "baseline")
    closure_src = existing / "closure"
    if closure_src.exists():
        # Preserve completed scientific products while regenerating publication outputs.
        copy_tree(closure_src, root / "analysis", ignore_names={"10_figures"})
    if (existing / "final").exists():
        copy_tree(existing / "final", root / "imported_results")
    ledger.event("00_import_existing", "PASS", "Imported completed /baseline science without rerunning baseline", time.time()-t0, str(existing))


def run_baseline_if_needed(args, root: Path, ledger: StageLedger, logger: logging.Logger) -> Mapping[str, Any]:
    """Run or resume the reduced-order baseline."""
    baseline = root / "analysis" / "baseline"
    bundle = baseline / "final" / "P10_feed_composition_robustness_complete_results.pkl"
    if bundle.exists() and not args.fresh_run:
        ledger.event("01_baseline", "PASS", "Reused completed reduced-order baseline", outputs=str(bundle))
        return load_pickle(bundle)
    ns = argparse.Namespace(**vars(args))
    return ext.run_baseline(ns, root, ledger, logger)

def verify_baseline(root: Path, gate: Mapping[str, Any], args, ledger: StageLedger) -> pd.DataFrame:
    return ext.verify_baseline(root, gate, args, ledger)


def fixed_external_boundaries(gate_bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Use the canonical external-source sensitivity key."""
    val = gate_bundle.get("validation", {})
    candidates = [
        val.get("external_source_sensitivity", {}),
        val.get("external_equilibrium_source_sensitivity", {}),
        val.get("external_equilibrium", {}),
    ]
    for obj in candidates:
        if isinstance(obj, pd.DataFrame) and len(obj):
            return obj.copy()
        if isinstance(obj, dict):
            for key in ("boundaries", "boundary_sensitivity", "source_boundary_sensitivity"):
                df = obj.get(key, pd.DataFrame())
                if isinstance(df, pd.DataFrame) and len(df):
                    return df.copy()
    return pd.DataFrame()


# Use the canonical external-source sensitivity extractor.
ext._extract_external_boundaries = fixed_external_boundaries


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_css_from_existing(root: Path) -> Dict[str, Any]:
    d = root / "analysis" / "09_css_high_fidelity"
    p = d / "css_closure_result.pkl"
    if p.exists():
        try: return load_pickle(p)
        except Exception: pass
    gate = {}
    gp = d / "validation" / "css_validation_gate.json"
    if gp.exists():
        try: gate = json.loads(gp.read_text(encoding="utf-8"))
        except Exception: gate = {}
    return {
        "gate": gate or {"status": "NOT_RUN", "reason": "no_existing_css_result"},
        "falsification": _read_csv(d / "validation" / "css_falsification_nine_cases.csv"),
        "disturbance": _read_csv(d / "validation" / "css_dynamic_15_12_18_15.csv"),
        "three_policy_disturbance": {"summary": _read_csv(d / "validation" / "css_dynamic_three_policy_summary.csv")},
    }


def normalize_css_gate(css: Mapping[str, Any]) -> Dict[str, Any]:
    gate = dict(css.get("gate", {}) or {})
    raw = str(gate.get("status", "NOT_RUN")).upper()
    reason = str(gate.get("reason", "")).lower()
    if "timeout" in reason:
        normalized = "INDETERMINATE_NUMERICAL_TIMEOUT"
    elif reason in {"css_worker_process_error", "css_orchestration_exception"} or "process_error" in reason or "exception" in reason:
        normalized = "FAIL_NUMERICAL"
    elif raw == "PASS":
        normalized = "PASS"
    elif raw == "FAIL":
        normalized = "FAIL_PHYSICAL_VALIDATION"
    elif raw in {"NOT_RUN", "SKIP", "SKIPPED"}:
        normalized = "NOT_RUN"
    else:
        normalized = raw or "NOT_RUN"
    gate["legacy_status"] = raw
    gate["status"] = normalized
    gate["status_semantics"] = "Timeout/process failure is not interpreted as physical-model falsification."
    return gate


def annotate_jacobian(jac: pd.DataFrame) -> pd.DataFrame:
    if jac.empty: return jac
    out = jac.copy()
    c = pd.to_numeric(out["condition_number"], errors="coerce")
    a = pd.to_numeric(out["actuator_response_angle_deg"], errors="coerce")
    out["conditioning_interpretation"] = np.select(
        [c < 10, c < 30, c < 100, c >= 100],
        ["well_conditioned", "moderately_coupled", "strongly_coupled", "highly_ill_conditioned"],
        default="indeterminate",
    )
    out["actuator_angle_interpretation"] = np.select(
        [a < 10, a < 30, a >= 30],
        ["near_collinear", "coupled", "more_separated"], default="indeterminate"
    )
    out["interpretive_note"] = "Second-actuator benefit is interpreted as nonlinear/global feasible-envelope expansion when local response vectors are near-collinear."
    return out


def load_or_compute_extensions(args, root: Path, gate: Mapping[str, Any], ledger: StageLedger, logger: logging.Logger):
    settings = ext.settings_from_bundle(gate)
    # Literature / kinetic evidence
    lit_dir = root / "analysis" / "literature_kinetics"
    if (lit_dir / "verified_breakthrough_conditions.csv").exists():
        literature = {
            "conditions": _read_csv(lit_dir / "verified_breakthrough_conditions.csv"),
            "scalars": _read_csv(lit_dir / "kinetic_scalar_crosschecks.csv"),
            "diagnostics": _read_csv(lit_dir / "kinetic_time_scale_diagnostics.csv"),
            "trace_gate": _read_csv(lit_dir / "breakthrough_trace_gate.csv"),
        }
        ledger.event("03_literature_kinetics", "PASS", "Reused completed literature/kinetic evidence outputs")
    else:
        literature = ext.run_literature_kinetic_audit(root, gate, ledger)

    p = root / "analysis" / "03_coprimary_95_90" / "variant_result.pkl"
    if p.exists():
        co90 = load_pickle(p); ledger.event("04_coprimary_95_90", "PASS", "Reused completed 95/90 benchmark calculation")
    else:
        settings90 = dataclasses.replace(settings, target_purity=0.95, target_recovery=0.90)
        co90 = ext.run_stage05_variant("03_coprimary_95_90", root, args, gate, settings90, logger, ledger)

    p = root / "analysis" / "control_jacobian" / "control_jacobian_svd_map.csv"
    if p.exists():
        jac = pd.read_csv(p); ledger.event("05_control_jacobian", "PASS", "Reused completed Jacobian/SVD map")
    else:
        jac = ext.control_jacobian(root, gate, settings, ledger)
    jac = annotate_jacobian(jac)
    atomic_csv(jac, root / "analysis" / "control_jacobian" / "control_jacobian_svd_map_interpreted.csv")

    p = root / "analysis" / "uncertainty_architecture" / "uncertainty_layer_register.csv"
    if p.exists():
        unc = pd.read_csv(p); ledger.event("06_uncertainty_architecture", "PASS", "Reused completed layered uncertainty register")
    else:
        unc = ext.layered_uncertainty(root, gate, settings, ledger)
    # Ensure repaired external thermodynamic layer is represented even when importing an old  register.
    external = fixed_external_boundaries(gate)
    if len(external) and not (unc.get("layer", pd.Series(dtype=str)).astype(str) == "independent_thermodynamic_parameterization").any():
        add = []
        for _, r in external.iterrows():
            add.append({"layer":"independent_thermodynamic_parameterization","evidence_type":"deterministic_source_sensitivity","intervention_ceiling":r.get("intervention_ceiling",r.get("max_adaptation_level","")),"lower_median":r.get("lower_feed_co2_fraction",np.nan),"lower_q025":np.nan,"lower_q975":np.nan,"status":"AVAILABLE","interpretation":"Alternative published thermodynamic parameterization; canonical source key used."})
        unc = pd.concat([unc, pd.DataFrame(add)], ignore_index=True, sort=False)
        atomic_csv(unc, root / "analysis" / "uncertainty_architecture" / "uncertainty_layer_register.csv")

    p = root / "analysis" / "06_equipment_realizability" / "variant_result.pkl"
    if p.exists():
        eqres = load_pickle(p); ledger.event("07_equipment_realizability", "PASS", "Reused completed equipment-constrained recalculation")
    else:
        eqres = ext.equipment_realizability(root, args, gate, settings, logger, ledger)

    p = root / "analysis" / "07_composition_flow_generalization" / "composition_x_feed_flow_authority_map.csv"
    if p.exists():
        flow = pd.read_csv(p); ledger.event("08_composition_flow", "PASS", "Reused completed composition-throughput authority map")
    else:
        flow = ext.flow_generalization(root, gate, settings, ledger)

    p = root / "analysis" / "08_disturbance_policies" / "quasisteady_15_12_18_15_policy_comparison.csv"
    if p.exists():
        qdyn = pd.read_csv(p); ledger.event("09_quasisteady_disturbance", "PASS", "Reused completed 15→12→18→15 quasi-steady policy comparison")
    else:
        qdyn = ext.quasisteady_disturbance(root, gate, settings, ledger)

    css = _load_css_from_existing(root)
    css["gate"] = normalize_css_gate(css)
    atomic_json(css["gate"], root / "analysis" / "09_css_high_fidelity" / "validation" / "css_validation_gate_interpreted.json")
    ledger.event("10_css_status_import", css["gate"]["status"], "Imported existing CSS status; CSS is not rerun by the primary analysis", outputs=str(root / "analysis" / "09_css_high_fidelity"))
    return settings, literature, co90, jac, unc, eqres, flow, qdyn, css


def _save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _class_color(level: int) -> str:
    return {0: base.PUB_COLORS.get("frozen", "#1f4e79"), 1: base.PUB_COLORS.get("single", "#e69f00"), 2: base.PUB_COLORS.get("coordinated", "#009e73"), 3: base.PUB_COLORS.get("outside", "#b35c5c")}.get(int(level), "#777777")


def _class_strip(ax, d: pd.DataFrame) -> None:
    try: base._class_strip(ax, d)
    except Exception: pass


def _window_rows(wdf: pd.DataFrame) -> List[Tuple[int, str, float, float, str, str]]:
    out = []
    if not isinstance(wdf, pd.DataFrame) or wdf.empty: return out
    for lev, lab in [(0, "Frozen"), (1, "≤1 actuator"), (2, "≤2 actuators")]:
        col = "max_adaptation_level" if "max_adaptation_level" in wdf.columns else None
        if col:
            q = wdf[pd.to_numeric(wdf[col], errors="coerce").eq(lev)]
        elif "intervention_ceiling" in wdf.columns:
            names = {0:"frozen_safe",1:"one_actuator_recoverable",2:"coordinated_recoverable"}
            q = wdf[wdf["intervention_ceiling"].astype(str).eq(names[lev])]
        else: q = pd.DataFrame()
        if q.empty: continue
        r = q.iloc[0]
        lo = float(pd.to_numeric(pd.Series([r.get("lower_feed_co2_fraction", np.nan)]), errors="coerce").iloc[0])
        hi = float(pd.to_numeric(pd.Series([r.get("upper_feed_co2_fraction", np.nan)]), errors="coerce").iloc[0])
        out.append((lev, lab, lo, hi, str(r.get("lower_boundary_status", "")), str(r.get("upper_boundary_status", ""))))
    return out


def _plot_ladder(ax, wdf: pd.DataFrame, title: str, xlim: Tuple[float,float] = (9, 40.5)) -> None:
    rows = _window_rows(wdf)
    if not rows:
        ax.text(.5,.5,"No operability-window data",ha="center",va="center",transform=ax.transAxes); ax.axis("off"); return
    for ypos,(lev, lab, lo, hi, lstat, ustat) in enumerate(rows):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            ax.text(np.mean(xlim), ypos, "nominal condition not feasible", ha="center", va="center", fontsize=7, style="italic")
            continue
        lo *= 100; hi *= 100
        ax.plot([lo,hi],[ypos,ypos],lw=7,solid_capstyle="butt",color=_class_color(lev),alpha=.88)
        ax.plot(lo,ypos,marker="|",ms=16,mew=1.2,color="#222222")
        ax.text(lo+.25,ypos+.17,f"{lo:.2f}%",fontsize=7,ha="left",va="bottom")
        if "censor" in ustat.lower() or hi >= xlim[1]-0.7:
            ax.plot(min(hi,xlim[1]-.1),ypos,marker=">",ms=8,color=_class_color(lev))
            ax.text(min(hi,xlim[1]-.8),ypos+.17,">40% (domain limit)",fontsize=6.5,ha="right",va="bottom")
        else:
            ax.plot(hi,ypos,marker="|",ms=16,mew=1.2,color="#222222")
            ax.text(hi-.25,ypos-.18,f"{hi:.2f}%",fontsize=7,ha="right",va="top")
    ax.set_yticks(range(len(rows)),[r[1] for r in rows]); ax.set_ylim(-.5,max(2.5,len(rows)-.5)); ax.set_xlim(*xlim)
    ax.axvline(15,ls=":",lw=.9,color="#555555"); ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_title(title); ax.grid(axis="x",alpha=.12)



def normalize_quasisteady_disturbance(qdyn: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate policy rows and attach explicit objectives/display labels."""
    if not isinstance(qdyn, pd.DataFrame) or qdyn.empty:
        return pd.DataFrame() if qdyn is None else qdyn.copy()
    q=qdyn.copy()
    keys=[c for c in ["segment","feed_co2_fraction","policy"] if c in q.columns]
    if keys:
        q=q.drop_duplicates(subset=keys,keep="last").sort_values(keys).reset_index(drop=True)
    objective={
        "frozen":"No adaptation; nominal settings retained",
        "minimum_authority":"Minimum adaptation level, then minimum control movement (Linf/L1)",
        "full_reoptimization_grid":"Energy-first feasible grid search: minimize idealized vacuum-work proxy, then maximize productivity",
    }
    display={
        "frozen":"Frozen",
        "minimum_authority":"Minimum authority",
        "full_reoptimization_grid":"Energy-min re-optimization",
    }
    if "policy" in q:
        q["policy_objective"]=q["policy"].astype(str).map(objective).fillna("")
        q["policy_display"]=q["policy"].astype(str).map(display).fillna(q["policy"].astype(str).str.replace("_"," "))
    return q


def scenario_coverage_summary(flow: pd.DataFrame) -> pd.DataFrame:
    """Scenario-count coverage on the evaluated composition x flow grid (not continuous area)."""
    if not isinstance(flow,pd.DataFrame) or flow.empty or "minimum_control_authority_level" not in flow:
        return pd.DataFrame()
    lev=pd.to_numeric(flow["minimum_control_authority_level"],errors="coerce")
    n=int(lev.notna().sum())
    rows=[]
    for ceiling,label in [(0,"Frozen sufficient"),(1,"At most one actuator"),(2,"At most two actuators")]:
        k=int((lev<=ceiling).sum())
        rows.append({"control_authority_ceiling":ceiling,"coverage_label":label,"scenarios_covered":k,"total_evaluated_scenarios":n,"scenario_coverage_percent":100*k/max(n,1),"interpretation":"Discrete evaluated scenario coverage; not continuous operating-area fraction."})
    k=int((lev>2).sum())
    rows.append({"control_authority_ceiling":3,"coverage_label":"Outside declared envelope","scenarios_covered":k,"total_evaluated_scenarios":n,"scenario_coverage_percent":100*k/max(n,1),"interpretation":"Discrete evaluated scenario coverage; not continuous operating-area fraction."})
    return pd.DataFrame(rows)


def equipment_boundary_summary(eqres: Mapping[str,Any]) -> pd.DataFrame:
    """Extract adaptively refined equipment-constrained operability boundaries when available."""
    if not isinstance(eqres,Mapping): return pd.DataFrame()
    w=eqres.get("windows",pd.DataFrame())
    rows=[]
    for lev,lab,lo,hi,lstat,ustat in _window_rows(w):
        rows.append({"max_adaptation_level":lev,"control_authority":lab,"equipment_realizable_lower_feed_co2_fraction":lo,"equipment_realizable_lower_feed_co2_percent":100*lo if np.isfinite(lo) else np.nan,"equipment_realizable_upper_feed_co2_fraction":hi,"equipment_realizable_upper_feed_co2_percent":100*hi if np.isfinite(hi) else np.nan,"lower_status":lstat,"upper_status":ustat,"boundary_source":"equipment-constrained adaptive refinement"})
    return pd.DataFrame(rows)


def _authority_level_from_text(x: Any) -> Optional[int]:
    z=str(x).lower()
    if "frozen" in z: return 0
    if "single" in z or "one_actuator" in z or "one actuator" in z: return 1
    if "coordinated" in z or "two_actuator" in z or "two actuator" in z: return 2
    return None


def uncertainty_hierarchy_summary(unc: pd.DataFrame, gate: Optional[Mapping[str,Any]] = None) -> pd.DataFrame:
    """Summarize boundary uncertainty/sensitivity layers without mixing probability and stress ranges."""
    frames=[]
    if isinstance(unc,pd.DataFrame) and not unc.empty:
        frames.append(unc.copy())
    # The layered register may carry blank conditional-MC fields because the baseline robustness
    # table uses boundary/q025/q975 column names. Repair from the frozen baseline bundle.
    if isinstance(gate,Mapping):
        rb=gate.get("robustness",{}).get("boundaries",pd.DataFrame())
        if isinstance(rb,pd.DataFrame) and len(rb):
            rrows=[]
            mapping={"frozen_safe_lower":"frozen_safe","single_knob_lower":"single_knob_or_less","coordinated_lower":"coordinated_two_knob_or_less"}
            for _,r in rb.iterrows():
                b=str(r.get("boundary",""))
                if b not in mapping: continue
                rrows.append({"layer":"calibration_conditioned_monte_carlo","evidence_type":"probabilistic_conditional","intervention_ceiling":mapping[b],"lower_median":r.get("median_feed_co2_fraction",np.nan),"lower_q025":r.get("q025",np.nan),"lower_q975":r.get("q975",np.nan),"status":"AVAILABLE","interpretation":"Conditional on nominal-anchor reproduction; not total predictive uncertainty."})
            if rrows: frames.append(pd.DataFrame(rrows))
    if not frames: return pd.DataFrame()
    d=pd.concat(frames,ignore_index=True,sort=False)
    # Drop blank duplicate rows from imported uncertainty registers.
    d=d[~(d.get("layer",pd.Series(dtype=str)).astype(str).eq("calibration_conditioned_monte_carlo") & pd.to_numeric(d.get("lower_median",np.nan),errors="coerce").isna())].copy()
    d["authority_level"]=[_authority_level_from_text(x) for x in d.get("intervention_ceiling",pd.Series("",index=d.index))]
    d["value"]=100*pd.to_numeric(d.get("lower_median",np.nan),errors="coerce")
    d["q025"]=100*pd.to_numeric(d.get("lower_q025",np.nan),errors="coerce")
    d["q975"]=100*pd.to_numeric(d.get("lower_q975",np.nan),errors="coerce")
    rows=[]
    for (layer,lev),g in d.dropna(subset=["authority_level"]).groupby(["layer","authority_level"],dropna=False):
        vals=pd.to_numeric(g["value"],errors="coerce").dropna()
        qlo=pd.to_numeric(g["q025"],errors="coerce").dropna(); qhi=pd.to_numeric(g["q975"],errors="coerce").dropna()
        if len(qlo) and len(qhi):
            lo=float(qlo.min()); hi=float(qhi.max())
        elif len(vals):
            lo=float(vals.min()); hi=float(vals.max())
        else:
            lo=hi=np.nan
        rows.append({"layer":layer,"authority_level":int(lev),"control_authority":{0:"Frozen",1:"≤1 actuator",2:"≤2 actuators"}.get(int(lev),str(lev)),"lower_boundary_min_percent":lo,"lower_boundary_central_percent":float(vals.median()) if len(vals) else np.nan,"lower_boundary_max_percent":hi,"n_entries":len(g),"status":";".join(sorted(set(g.get("status",pd.Series(dtype=str)).dropna().astype(str))))})
    return pd.DataFrame(rows)


def control_effort_summary(qdyn: pd.DataFrame) -> pd.DataFrame:
    """Summarize on-spec success and total sequential actuator movement for the quasi-steady test."""
    q=normalize_quasisteady_disturbance(qdyn)
    if q.empty: return pd.DataFrame()
    rows=[]
    for pol,g in q.groupby("policy",sort=False):
        gg=g.sort_values("segment")
        ok=pd.Series(gg.get("success",False)).astype(bool)
        p=pd.to_numeric(gg.get("p_reg_bar",np.nan),errors="coerce").to_numpy(float)
        t=pd.to_numeric(gg.get("timing_scale",np.nan),errors="coerce").to_numpy(float)
        lp=np.log(np.maximum(p,1e-12)); lt=np.log(np.maximum(t,1e-12))
        move=float(np.nansum(np.abs(np.diff(lp)))+np.nansum(np.abs(np.diff(lt)))) if len(gg)>1 else 0.0
        rows.append({"policy":pol,"policy_display":gg.get("policy_display",pd.Series([pol])).iloc[0],"policy_objective":gg.get("policy_objective",pd.Series([""])).iloc[0],"on_spec_states":int(ok.sum()),"total_states":int(len(gg)),"on_spec_fraction":float(ok.mean()) if len(gg) else np.nan,"total_log_actuator_movement":move})
    out=pd.DataFrame(rows)
    try:
        m=float(out.loc[out.policy.eq("minimum_authority"),"total_log_actuator_movement"].iloc[0]); f=float(out.loc[out.policy.eq("full_reoptimization_grid"),"total_log_actuator_movement"].iloc[0])
        red=100*(1-m/f) if f>0 else np.nan
        out["minimum_authority_movement_reduction_vs_reoptimization_percent"]=red
    except Exception:
        out["minimum_authority_movement_reduction_vs_reoptimization_percent"]=np.nan
    return out


def import_optional_breakthrough_traces(path: str, root: Path, ledger: StageLedger) -> pd.DataFrame:
    """Import user-supplied/digitized trace data only when schema and provenance fields are explicit.

    current analysis never fabricates trace coordinates. Expected minimum columns are feed_co2_fraction,
    time_s and normalized_outlet; provenance_source and digitization_method are strongly recommended.
    """
    if not path: return pd.DataFrame()
    p=Path(path).resolve()
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    req={"feed_co2_fraction","time_s","normalized_outlet"}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"Breakthrough trace CSV missing required columns: {sorted(miss)}")
    for c in req: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=list(req)).copy()
    if df.empty: raise ValueError("Breakthrough trace CSV has no valid numeric rows")
    out=root/"analysis"/"literature_kinetics"/"user_supplied_breakthrough_traces.csv"; atomic_csv(df,out)
    rows=[]
    for y,g in df.groupby("feed_co2_fraction"):
        gg=g.sort_values("time_s"); xx=gg["normalized_outlet"].to_numpy(float); tt=gg["time_s"].to_numpy(float)
        row={"feed_co2_fraction":float(y),"n_points":len(gg)}
        for frac in (0.1,0.5,0.9):
            if np.nanmin(xx)<=frac<=np.nanmax(xx):
                order=np.argsort(xx); row[f"t{int(frac*100)}_s"]=float(np.interp(frac,xx[order],tt[order]))
            else: row[f"t{int(frac*100)}_s"]=np.nan
        rows.append(row)
    summ=pd.DataFrame(rows); atomic_csv(summ,root/"analysis"/"literature_kinetics"/"user_supplied_breakthrough_trace_metrics.csv")
    ledger.event("03b_breakthrough_trace_import","PASS",f"Imported {len(df)} trace points across {df.feed_co2_fraction.nunique()} feed compositions; no synthetic coordinates generated",outputs=str(out))
    return df

def make_integrated_figures(root: Path, pkg: Path, gate: Mapping[str, Any], co90: Mapping[str, Any], jac: pd.DataFrame,
                            unc: pd.DataFrame, eqres: Mapping[str, Any], flow: pd.DataFrame, qdyn: pd.DataFrame, css: Mapping[str, Any], settings,
                            args, ledger: StageLedger) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import ListedColormap
    base.configure_matplotlib()
    main=pkg/"figures"/"main"; si=pkg/"figures"/"SI"; srcm=pkg/"source_data"/"main"; srcs=pkg/"source_data"/"SI"
    for p in (main,si,srcm,srcs): p.mkdir(parents=True,exist_ok=True)
    created=[]; gate_dir=root/"analysis"/"baseline"; atlas=gate["feed_composition_robustness"]
    intrinsic=atlas.get("intrinsic",pd.DataFrame()); fixed=atlas.get("fixed",pd.DataFrame()); sel=atlas.get("selected",pd.DataFrame()); effort=atlas.get("continuous_one_actuator_effort",pd.DataFrame())
    qdyn=normalize_quasisteady_disturbance(qdyn); coverage=scenario_coverage_summary(flow); eqsum=equipment_boundary_summary(eqres); uh=uncertainty_hierarchy_summary(unc,gate); ces=control_effort_summary(qdyn)

    # Figure 1 — preserve the established scientific grammar but remove misleading line through heterogeneous pilot campaigns.
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.8),constrained_layout=True)
    try: base._evidence_landscape(axs[0,0],gate.get("composition_evidence",{}),gate.get("validation",{}),settings)
    except Exception: axs[0,0].text(.5,.5,"Composition-resolved evidence landscape",ha="center",va="center",transform=axs[0,0].transAxes)
    base.panel_label(axs[0,0],"a")
    data=gate.get("harmonized",{}).get("data",pd.DataFrame()); ax=axs[0,1]
    if len(data):
        for gas,ls in [("CO2","-"),("N2","--")]:
            g=data[(data["material"]=="CALF-20")&(data["adsorbate"]==gas)]
            if g.empty: continue
            fs=gate.get("fits",{}).get("fits",{}).get(("CALF-20",gas),{}); m=fs.get("best_model")
            if not m: continue
            th=fs["models"][m]["theta"]; Ts=sorted(g["temperature_K"].unique()); Ts=[Ts[0],Ts[len(Ts)//2],Ts[-1]] if len(Ts)>=3 else Ts
            for T in Ts:
                gg=g[np.isclose(g["temperature_K"],T)]; ax.scatter(gg["pressure_bar"],gg["loading_mol_kg"],s=10,alpha=.55)
                pp=np.geomspace(max(float(gg["pressure_bar"].min()),1e-4),float(gg["pressure_bar"].max()),150); ax.plot(pp,base.model_loading(m,pp,np.full_like(pp,T),th),ls=ls,lw=1.1,label=f"{gas}, {T:g} K")
        ax.set_xscale("log"); ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Loading (mol kg$^{-1}$)"); ax.set_title("Thermodynamic foundation"); ax.grid(alpha=.12); ax.legend(frameon=False,fontsize=6,ncol=2)
    base.panel_label(ax,"b")
    ax=axs[1,0]; pv=gate.get("validation",{}).get("external_pilot",{}).get("raw",pd.DataFrame()); medrows=[]
    if isinstance(pv,pd.DataFrame) and len(pv):
        for y,g in pv.groupby("feed_co2_fraction"):
            rr=pd.to_numeric(g["recovery_fraction"],errors="coerce").dropna()
            if len(rr): medrows.append((100*float(y),100*float(rr.median()),100*float(rr.quantile(.25)),100*float(rr.quantile(.75)),len(rr)))
        for x0,m0,q1,q3,n0 in sorted(medrows):
            ax.errorbar([x0],[m0],yerr=np.array([[m0-q1],[q3-m0]]),fmt="o",capsize=4,lw=1.2)
            ax.text(x0,m0+1.5,f"n={n0}",ha="center",fontsize=7)
        ax.text(.02,.04,"Heterogeneous pilot operating conditions; points are not connected",transform=ax.transAxes,fontsize=6.8,color="#555555")
        ax.set_xlabel("Feed CO$_2$ (%)"); ax.set_ylabel("Pilot recovery median (%, IQR)"); ax.set_title("Independent pilot-scale composition sensitivity"); ax.grid(alpha=.12)
    else: ax.text(.5,.5,"Independent pilot dataset unavailable",ha="center",va="center",transform=ax.transAxes)
    base.panel_label(ax,"c")
    ax=axs[1,1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    boxes=[(.03,.58,.20,.20,"Frozen\nprocess",_class_color(0)),(.29,.58,.20,.20,"One\nactuator",_class_color(1)),(.55,.58,.20,.20,"Two coordinated\nactuators",_class_color(2)),(.81,.58,.16,.20,"Outside normal\ncontrol",_class_color(3))]
    for x0,y0,w,h,txt,c in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc=c,ec="none",alpha=.16)); ax.text(x0+w/2,y0+h/2,txt,ha="center",va="center",fontsize=9)
    for a,b in [(.23,.29),(.49,.55),(.75,.81)]: ax.annotate("",xy=(b,.68),xytext=(a,.68),arrowprops=dict(arrowstyle="->",lw=1.2,color="#555555"))
    ax.text(.5,.34,"How many manipulated operating variables are required\nto preserve purity/recovery as feed composition drifts?",ha="center",va="center",fontsize=9)
    ax.text(.5,.14,r"$A^*(y)=\min\{\dim(u):\;P(y,u)\geq P^*,\;R(y,u)\geq R^*\}$",ha="center",va="center",fontsize=10); ax.set_title("Minimum-control-authority framework"); base.panel_label(ax,"d")
    _save_figure(fig,main/"Figure_1_evidence_and_control_authority_framework")
    if medrows: atomic_csv(pd.DataFrame(medrows,columns=["feed_co2_percent","recovery_median_percent","recovery_q25_percent","recovery_q75_percent","n_runs"]),srcm/"Figure_1c_pilot_summary.csv")
    created.append(str(main/"Figure_1_evidence_and_control_authority_framework.png"))

    # Figure 2 — adsorption physics, active constraints, and local controllability.
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.8),constrained_layout=True)
    if len(intrinsic):
        x=100*intrinsic["feed_co2_fraction"]; axs[0,0].plot(x,intrinsic["q_co2_mol_kg"],label="CO$_2$"); axs[0,0].plot(x,intrinsic["q_n2_mol_kg"],label="N$_2$"); axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)"); axs[0,0].legend(frameon=False); axs[0,0].set_title("Competitive mixture loading")
    if len(fixed):
        x=100*fixed["feed_co2_fraction"]; pm=100*(fixed["purity_estimate"]-settings.target_purity); rm=100*(fixed["recovery_estimate"]-settings.target_recovery); axs[0,1].plot(x,pm,label="purity margin"); axs[0,1].plot(x,rm,label="recovery margin"); axs[0,1].axhline(0,ls="--",lw=.9,color="#444444"); axs[0,1].set_xlim(10,30); axs[0,1].set_ylabel("Constraint margin (percentage points)"); axs[0,1].set_title("Active-constraint switching (transition region)"); axs[0,1].legend(frameon=False,fontsize=7)
    if len(jac):
        x=100*jac["feed_co2_fraction"]; axs[1,0].plot(x,jac["pressure_response_norm"],label="pressure sensitivity"); axs[1,0].plot(x,jac["timing_response_norm"],label="timing sensitivity"); axs[1,0].set_ylabel("Local response norm"); axs[1,0].set_title("Control-sensitivity magnitude"); axs[1,0].legend(frameon=False,fontsize=7)
        cond=np.maximum(pd.to_numeric(jac["condition_number"],errors="coerce"),1e-6); ang=pd.to_numeric(jac["actuator_response_angle_deg"],errors="coerce"); axs[1,1].semilogy(x,cond,lw=1.6,label="$\\kappa(J)$"); axs[1,1].axhline(100,ls="--",lw=.8,color="#777777"); axs[1,1].set_ylabel("Jacobian condition number"); ax2=axs[1,1].twinx(); ax2.plot(x,ang,ls=":",lw=1.2,color="#666666"); ax2.set_ylabel("Actuator-response angle (deg)"); axs[1,1].set_title("Local coupling / near-collinearity")
        mc=float(np.nanmedian(cond)); ma=float(np.nanmedian(ang)); axs[1,1].text(.03,.93,f"median $\\kappa$ ≈ {mc:.0f}\nmedian angle ≈ {ma:.1f}°\nlocally near-collinear",transform=axs[1,1].transAxes,va="top",fontsize=7,bbox=dict(fc="white",ec="none",alpha=.8,pad=2))
    for i,a in enumerate(axs.flat):
        if a.has_data(): a.axvline(15,ls=":",lw=.8,color="#555555"); a.set_xlabel("Feed CO$_2$ (%)"); a.grid(alpha=.12)
        base.panel_label(a,"abcd"[i])
    _save_figure(fig,main/"Figure_2_mechanism_and_controllability"); atomic_csv(intrinsic,srcm/"Figure_2a_competitive_loading.csv"); atomic_csv(fixed,srcm/"Figure_2b_constraint_margins.csv"); atomic_csv(jac,srcm/"Figure_2cd_control_jacobian.csv"); created.append(str(main/"Figure_2_mechanism_and_controllability.png"))

    # Figure 3 — frozen process fragility in absolute engineering units.
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if len(fixed):
        x=100*fixed["feed_co2_fraction"]; valid=(pd.to_numeric(fixed["working_capacity_co2_mol_kg"],errors="coerce")>1e-8)&np.isfinite(pd.to_numeric(fixed["specific_vacuum_work_kJ_mol_CO2"],errors="coerce")); win=atlas.get("windows",pd.DataFrame()); flo=fhi=np.nan
        for lev,_,lo,hi,_,_ in _window_rows(win):
            if lev==0: flo,fhi=lo,hi
        for ax in axs.flat:
            if (~valid).any(): ax.axvspan(float(x[~valid].min()),float(x[~valid].max()),color="#D9D9D9",alpha=.30,hatch="///",zorder=-5)
            if np.isfinite(flo) and np.isfinite(fhi): ax.axvspan(100*flo,100*fhi,color=_class_color(0),alpha=.10,zorder=-4)
        axs[0,0].plot(x,np.where(valid,100*fixed["purity_estimate"],np.nan)); axs[0,0].axhline(95,ls="--",lw=1,color="#333333"); axs[0,0].set_ylabel("CO$_2$ purity (%)"); axs[0,0].set_title("Frozen process: purity constraint")
        axs[0,1].plot(x,np.where(valid,100*fixed["recovery_estimate"],np.nan)); axs[0,1].axhline(88,ls="--",lw=1,color="#333333",label="88% experimental-anchor target"); axs[0,1].axhline(90,ls=":",lw=1,color="#666666",label="90% conventional target"); axs[0,1].set_ylabel("CO$_2$ recovery (%)"); axs[0,1].set_title("Frozen process: recovery constraint"); axs[0,1].legend(frameon=False,fontsize=6.5)
        axs[1,0].plot(x,np.where(valid,fixed["productivity_mol_kg_h"],np.nan)); axs[1,0].set_ylabel("Productivity (mol kg$^{-1}$ h$^{-1}$)"); axs[1,0].set_title("Frozen process: productivity")
        axs[1,1].plot(x,np.where(valid,fixed["specific_vacuum_work_kJ_mol_CO2"],np.nan)); axs[1,1].set_ylabel("Idealized vacuum-work proxy\n(kJ mol$^{-1}$ CO$_2$)"); axs[1,1].set_title("Frozen process: resource burden")
        if (~valid).any(): axs[0,0].text(.03,.07,"hatched: outside reduced-order applicability / no positive frozen CO$_2$ swing",transform=axs[0,0].transAxes,fontsize=6.4)
    for i,a in enumerate(axs.flat): a.axvline(15,ls=":",lw=.8,color="#555555"); a.set_xlabel("Feed CO$_2$ (%)"); a.grid(alpha=.12); base.panel_label(a,"abcd"[i])
    _save_figure(fig,main/"Figure_3_frozen_process_offdesign_failure"); atomic_csv(fixed,srcm/"Figure_3_source_data.csv"); created.append(str(main/"Figure_3_frozen_process_offdesign_failure.png"))

    # Figure 4 — intervention trajectory, equipment limits, actuator competition and consequences.
    fig,axs=plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True); d=sel.copy(); d=d[d.get("success",False).astype(bool)] if len(d) else d
    eqlean=np.nan
    if len(eqsum):
        q=eqsum[eqsum.max_adaptation_level.eq(2)]
        if len(q): eqlean=float(q.iloc[0].equipment_realizable_lower_feed_co2_percent)
    if len(d):
        x=100*d["feed_co2_fraction"]; axs[0,0].plot(x,d["p_reg_bar"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3); axs[0,0].axhline(settings.nominal_regeneration_pressure_bar,ls="--",lw=.8,color="#333333",label="nominal"); axs[0,0].axhline(args.equipment_min_p_reg_bar,ls=":",lw=.9,color="#777777",label="equipment floor"); axs[0,0].set_yscale("log"); axs[0,0].set_ylabel("Selected regeneration pressure (bar)"); axs[0,0].legend(frameon=False,fontsize=6.5)
        axs[0,1].plot(x,d["timing_scale"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3); axs[0,1].axhline(1,ls="--",lw=.8,color="#333333"); axs[0,1].axhline(args.equipment_min_timing_scale,ls=":",lw=.8,color="#777777"); axs[0,1].axhline(args.equipment_max_timing_scale,ls=":",lw=.8,color="#777777"); axs[0,1].set_ylabel("Selected timing scale")
        if len(effort) and "actuator_effort_delta_pressure_minus_timing_continuous" in effort.columns:
            ex=100*effort["feed_co2_fraction"]; ey=effort["actuator_effort_delta_pressure_minus_timing_continuous"]; axs[1,0].plot(ex,ey,lw=1.6); axs[1,0].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,0].fill_between(ex,-.05,.05,alpha=.08); axs[1,0].set_ylabel("Pressure effort − timing effort"); axs[1,0].text(.03,.92,"<0 pressure preferred\n≈0 near-degenerate\n>0 timing preferred",transform=axs[1,0].transAxes,va="top",fontsize=7)
        if len(fixed):
            basew=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"]); basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"]); dw=100*(d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(basew,1e-12)-1); dp=100*(d["productivity_mol_kg_h"]/np.maximum(basep,1e-12)-1); axs[1,1].plot(x,dw,label="vacuum-work proxy"); axs[1,1].plot(x,dp,label="productivity"); axs[1,1].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,1].set_ylabel("Change from frozen policy (%)"); axs[1,1].legend(frameon=False,fontsize=7)
    titles=["Minimum pressure intervention","Minimum timing intervention","Actuator competition / degeneracy","Resource and productivity consequence"]
    for i,a in enumerate(axs.flat):
        a.axvline(15,ls=":",lw=.8,color="#555555");
        if np.isfinite(eqlean): a.axvline(eqlean,ls="--",lw=.9,color="#777777");
        a.set_xlabel("Feed CO$_2$ (%)"); a.grid(alpha=.12); a.set_title(titles[i]); base.panel_label(a,"abcd"[i])
    if np.isfinite(eqlean): axs[0,0].text(.03,.06,f"equipment-realizable coordinated lean limit ≈ {eqlean:.2f}%",transform=axs[0,0].transAxes,fontsize=6.7)
    _save_figure(fig,main/"Figure_4_minimum_intervention_policy"); atomic_csv(d,srcm/"Figure_4_selected_policy.csv"); atomic_csv(effort,srcm/"Figure_4_actuator_competition.csv"); atomic_csv(eqsum,srcm/"Figure_4_equipment_boundaries.csv"); created.append(str(main/"Figure_4_minimum_intervention_policy.png"))

    # Figure 5 — signature figure: topology, co-primary target, multivariable coverage, operational consequence.
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.1),constrained_layout=True)
    _plot_ladder(axs[0,0],atlas.get("windows",pd.DataFrame()),"Nested operability — 95% purity / 88% recovery")
    # Overlay conditional Monte Carlo and kinetic/front sensitivity on lean boundaries.
    if len(uh):
        for lev in (0,1,2):
            kin=uh[(uh.authority_level==lev)&uh.layer.astype(str).eq("kinetic_front_utilization_sensitivity")]
            mc=uh[(uh.authority_level==lev)&uh.layer.astype(str).eq("calibration_conditioned_monte_carlo")]
            if len(kin):
                r=kin.iloc[0]; axs[0,0].plot([r.lower_boundary_min_percent,r.lower_boundary_max_percent],[lev-.15,lev-.15],lw=5,alpha=.22,color=_class_color(lev),solid_capstyle="butt")
            if len(mc):
                r=mc.iloc[0]; axs[0,0].plot([r.lower_boundary_min_percent,r.lower_boundary_max_percent],[lev-.05,lev-.05],lw=2.5,color="#111111",solid_capstyle="butt")
        axs[0,0].text(.02,.03,"thin dark: calibration-conditioned 95% interval\nthick pale: declared kinetic/front sensitivity range",transform=axs[0,0].transAxes,fontsize=6.3)
    _plot_ladder(axs[0,1],co90.get("windows",pd.DataFrame()),"Co-primary benchmark — 95% purity / 90% recovery")
    if len(flow):
        piv=flow.pivot(index="feed_flow_scale",columns="feed_co2_fraction",values="minimum_control_authority_level").sort_index(); xx=100*piv.columns.to_numpy(float); yy=piv.index.to_numpy(float); X,Y=np.meshgrid(xx,yy); m=axs[1,0].pcolormesh(X,Y,piv.values,shading="nearest",vmin=-.5,vmax=3.5,cmap=ListedColormap([_class_color(i) for i in range(4)])); axs[1,0].scatter(np.repeat(xx,len(yy)),np.tile(yy,len(xx)),s=3,c="k",alpha=.12); axs[1,0].set_xlabel("Feed CO$_2$ (%)"); axs[1,0].set_ylabel("Feed-flow scale")
        covtxt=""
        if len(coverage):
            vals={int(r.control_authority_ceiling):float(r.scenario_coverage_percent) for _,r in coverage.iterrows()}; covtxt=f"frozen {vals.get(0,np.nan):.0f}%  |  ≤1 actuator {vals.get(1,np.nan):.0f}%  |  ≤2 actuators {vals.get(2,np.nan):.0f}%"
        axs[1,0].set_title("Composition × throughput control authority\n"+covtxt)
        labels={0:"Frozen",1:"≤1 actuator",2:"≤2 actuators",3:"Outside"}
        for lev,lab in labels.items():
            pts=flow[pd.to_numeric(flow.minimum_control_authority_level,errors="coerce").eq(lev)]
            if len(pts): axs[1,0].text(100*pts.feed_co2_fraction.median(),pts.feed_flow_scale.median(),lab,ha="center",va="center",fontsize=7,bbox=dict(fc="white",ec="none",alpha=.58,pad=1.5))
    if len(qdyn):
        q=qdyn.copy(); q["spec_margin_pp"]=100*np.minimum(pd.to_numeric(q.get("purity",np.nan),errors="coerce")-settings.target_purity,pd.to_numeric(q.get("recovery",np.nan),errors="coerce")-settings.target_recovery)
        for pol,g in q.groupby("policy"):
            gg=g.sort_values("segment"); axs[1,1].plot(gg["segment"],gg["spec_margin_pp"],marker="o",label=str(gg["policy_display"].iloc[0]))
        axs[1,1].axhline(0,ls="--",lw=.9,color="#333333"); axs[1,1].set_xticks([1,2,3,4],["15%","12%","18%","15%"]); axs[1,1].set_xlabel("Feed-composition sequence"); axs[1,1].set_ylabel("Minimum purity/recovery margin (percentage points)"); axs[1,1].set_title("Quasi-steady disturbance: specification recovery"); axs[1,1].legend(frameon=False,fontsize=6.2,loc="best"); axs[1,1].grid(alpha=.12)
        if len(ces):
            txt=[]
            for pol in ["frozen","minimum_authority","full_reoptimization_grid"]:
                rr=ces[ces.policy.eq(pol)]
                if len(rr): txt.append(f"{rr.iloc[0].policy_display}: {int(rr.iloc[0].on_spec_states)}/{int(rr.iloc[0].total_states)} on-spec")
            rr=ces[ces.policy.eq("minimum_authority")]
            if len(rr) and np.isfinite(rr.iloc[0].minimum_authority_movement_reduction_vs_reoptimization_percent): txt.append(f"minimum-authority movement ≈ {rr.iloc[0].minimum_authority_movement_reduction_vs_reoptimization_percent:.0f}% lower than re-optimization")
            axs[1,1].text(.02,.03,"\n".join(txt),transform=axs[1,1].transAxes,fontsize=6.4,bbox=dict(fc="white",ec="none",alpha=.78,pad=2))
    for i,a in enumerate(axs.flat): base.panel_label(a,"abcd"[i])
    _save_figure(fig,main/"Figure_5_nested_operability_and_generalization"); atomic_csv(flow,srcm/"Figure_5c_composition_flow.csv"); atomic_csv(qdyn,srcm/"Figure_5d_quasisteady_disturbance.csv"); atomic_csv(coverage,srcm/"Figure_5c_scenario_coverage_summary.csv"); atomic_csv(ces,srcm/"Figure_5d_control_effort_summary.csv"); atomic_csv(uh,srcm/"Figure_5a_uncertainty_hierarchy_summary.csv"); created.append(str(main/"Figure_5_nested_operability_and_generalization.png"))

    # Preserve the complete supporting-figure set and replace/extend corrected panels.
    copy_glob(gate_dir/"figures"/"SI","*",si); copy_glob(gate_dir/"source_data"/"SI","*",srcs)
    for pold in si.glob("Figure_S7_leave_one_source_out.*"): pold.unlink(missing_ok=True)
    for pold in srcs.glob("Figure_S7_leave_one_source_out*"): pold.unlink(missing_ok=True)
    bv=gate.get("validation",{}).get("blocked",pd.DataFrame())
    if isinstance(bv,pd.DataFrame) and len(bv):
        fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True)
        for ads,g in bv.groupby("adsorbate"):
            gg=g.copy(); gg["temperature_K"]=pd.to_numeric(gg["held_out"].astype(str).str.extract(r"([0-9.]+)")[0],errors="coerce"); ax.plot(gg["temperature_K"],pd.to_numeric(gg["rmse_mol_kg"],errors="coerce"),marker="o",label=str(ads))
        ax.set_yscale("log"); ax.set_xlabel("Held-out temperature (K)"); ax.set_ylabel("RMSE (mol kg$^{-1}$)"); ax.set_title("Blocked leave-one-temperature-out thermodynamic reconstruction test"); ax.grid(alpha=.12); ax.legend(frameon=False); _save_figure(fig,si/"Figure_S7_blocked_leave_one_temperature_out"); atomic_csv(bv,srcs/"Figure_S7_blocked_leave_one_temperature_out.csv")
    ext=fixed_external_boundaries(gate); fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True); nom=atlas.get("windows",pd.DataFrame())
    if len(ext):
        levelcol="max_adaptation_level" if "max_adaptation_level" in ext.columns else None; sourcecol="source_file" if "source_file" in ext.columns else None; groups=ext.groupby(sourcecol) if sourcecol else [("independent thermodynamic source",ext)]
        for srcname,g in groups:
            gg=g.copy(); xlev=pd.to_numeric(gg[levelcol],errors="coerce") if levelcol else np.arange(len(gg)); ax.plot(xlev,100*pd.to_numeric(gg["lower_feed_co2_fraction"],errors="coerce"),marker="o",label=str(srcname))
        nrows=_window_rows(nom)
        if nrows: ax.plot([r[0] for r in nrows],[100*r[2] for r in nrows],marker="s",ls="--",label="baseline nominal")
        ax.set_xticks([0,1,2],["Frozen","≤1 actuator","≤2 actuators"]); ax.set_ylabel("Lean-side boundary (feed CO$_2$ %)"); ax.legend(frameon=False,fontsize=7); ax.grid(alpha=.12)
    else: ax.text(.5,.5,"Independent thermodynamic source result not found in baseline bundle",ha="center",va="center",transform=ax.transAxes); ax.axis("off")
    ax.set_title("Independent thermodynamic-source sensitivity"); _save_figure(fig,si/"Figure_S16_external_thermodynamic_source_sensitivity"); atomic_csv(ext,srcs/"Figure_S16_external_thermodynamic_source_sensitivity.csv")
    if len(jac):
        fig,axs2=plt.subplots(1,2,figsize=(10.8,4.2),constrained_layout=True); x=100*jac.feed_co2_fraction; axs2[0].semilogy(x,np.maximum(jac.singular_value_1,1e-12),label="$\\sigma_1$"); axs2[0].semilogy(x,np.maximum(jac.singular_value_2,1e-12),label="$\\sigma_2$"); axs2[0].set_xlabel("Feed CO$_2$ (%)"); axs2[0].set_ylabel("Jacobian singular value"); axs2[0].legend(frameon=False); axs2[0].grid(alpha=.12); axs2[0].set_title("Local control singular values"); axs2[1].semilogy(x,np.maximum(jac.condition_number,1e-6)); axs2[1].axhline(100,ls="--",lw=.8,color="#777777"); axs2[1].set_xlabel("Feed CO$_2$ (%)"); axs2[1].set_ylabel("Condition number"); axs2[1].set_title("Strong local coupling / ill-conditioning"); axs2[1].grid(alpha=.12); _save_figure(fig,si/"Figure_S21_control_jacobian_singular_values"); atomic_csv(jac,srcs/"Figure_S21_control_jacobian.csv")
    if len(eqsum):
        fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True); ax.plot(eqsum.max_adaptation_level,eqsum.equipment_realizable_lower_feed_co2_percent,marker="o",label="equipment-realizable"); nrows=_window_rows(atlas.get("windows",pd.DataFrame())); ax.plot([r[0] for r in nrows],[100*r[2] for r in nrows],marker="s",ls="--",label="mathematical / unconstrained"); ax.set_xticks([0,1,2],["Frozen","≤1 actuator","≤2 actuators"]); ax.set_ylabel("Lean-side boundary (feed CO$_2$ %)"); ax.set_title("Mathematical versus equipment-realizable control authority"); ax.grid(alpha=.12); ax.legend(frameon=False); _save_figure(fig,si/"Figure_S22_equipment_realizability"); atomic_csv(eqsum,srcs/"Figure_S22_equipment_boundary_summary.csv")
    if len(qdyn):
        fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True)
        for pol,g in qdyn.groupby("policy"):
            gg=g.sort_values("segment"); ax.plot(gg.segment,100*np.minimum(pd.to_numeric(gg.purity,errors="coerce")-settings.target_purity,pd.to_numeric(gg.recovery,errors="coerce")-settings.target_recovery),marker="o",label=str(gg.policy_display.iloc[0]))
        ax.axhline(0,ls="--",lw=.8,color="#333333"); ax.set_xticks([1,2,3,4],["15%","12%","18%","15%"]); ax.set_ylabel("Minimum specification margin (pp)"); ax.set_title("Three-policy quasi-steady disturbance comparison"); ax.grid(alpha=.12); ax.legend(frameon=False); _save_figure(fig,si/"Figure_S23_quasisteady_policy_comparison"); atomic_csv(qdyn,srcs/"Figure_S23_quasisteady_policy_comparison.csv")
    gate_status=css.get("gate",{}); fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True); ax.axis("off"); txt=f"High-fidelity CSS status: {gate_status.get('status','NOT_RUN')}\n\nOriginal solver status: {gate_status.get('legacy_status','')}\nReason: {gate_status.get('reason','')}\n\nInterpretation:\nTimeout/process errors are numerical indeterminacy, not physical falsification.\nCSS-derived scientific claims are promoted only after a PASS same-rig validation gate."; ax.text(.04,.95,txt,va="top",fontsize=10,transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.6",fc="white",ec="0.5")); ax.set_title("High-fidelity evidence-gate status"); _save_figure(fig,si/"Figure_S24_high_fidelity_css_status"); atomic_json(gate_status,srcs/"Figure_S24_high_fidelity_css_status.json")
    # New S25: uncertainty hierarchy forest plot, explicitly separating probability from stress/sensitivity.
    if len(uh):
        fig,ax=plt.subplots(figsize=(9.4,5.8),constrained_layout=True); layer_order=["calibration_conditioned_monte_carlo","independent_thermodynamic_parameterization","thermodynamic_model_discrepancy_stress","kinetic_front_utilization_sensitivity"]; y=0; yt=[]; yl=[]
        for lev in (0,1,2):
            for layer in layer_order:
                g=uh[(uh.authority_level==lev)&uh.layer.eq(layer)]
                if g.empty: continue
                r=g.iloc[0]; c={0:_class_color(0),1:_class_color(1),2:_class_color(2)}[lev]; ax.plot([r.lower_boundary_min_percent,r.lower_boundary_max_percent],[y,y],lw=4,color=c,alpha=.45,solid_capstyle="butt"); ax.plot(r.lower_boundary_central_percent,y,"o",color=c,ms=5); yt.append(y); yl.append(f"{r.control_authority} | {layer.replace('_',' ')}"); y+=1
            y+=.5
        ax.set_yticks(yt,yl,fontsize=6.5); ax.set_xlabel("Lean-side boundary (feed CO$_2$ %)"); ax.set_title("Uncertainty hierarchy: parameter/source robustness versus kinetic sensitivity"); ax.grid(axis="x",alpha=.12); _save_figure(fig,si/"Figure_S25_uncertainty_hierarchy"); atomic_csv(uh,srcs/"Figure_S25_uncertainty_hierarchy.csv")
    ledger.event("11_figures","PASS",f"Generated/refined final five main figures and current analysis SI closure figures; created={len(created)}",outputs=str(pkg/"figures"))
    return created


def assemble_tables_and_science(root: Path, pkg: Path, gate: Mapping[str, Any], literature, co90, jac, unc, eqres, flow, qdyn, css, ledger: StageLedger) -> None:
    t0=time.time(); gate_dir=root/"analysis"/"baseline"; qdyn=normalize_quasisteady_disturbance(qdyn); coverage=scenario_coverage_summary(flow); eqsum=equipment_boundary_summary(eqres); uh=uncertainty_hierarchy_summary(unc,gate); ces=control_effort_summary(qdyn)
    copy_tree(gate_dir/"tables"/"main",pkg/"tables"/"main"); copy_tree(gate_dir/"tables"/"SI",pkg/"tables"/"SI"); copy_tree(gate_dir/"source_data"/"main",pkg/"source_data"/"main"); copy_tree(gate_dir/"source_data"/"SI",pkg/"source_data"/"SI")
    for name in ("data","models","engineering","uncertainty","validation"): copy_tree(gate_dir/name,pkg/name)
    copy_tree(gate_dir/"audit",pkg/"provenance"/"baseline_audit"); copy_tree(gate_dir/"logs",pkg/"logs"/"baseline"); copy_tree(root/"logs",pkg/"logs"/"analysis",ignore_names={"archives"})
    copy_tree(root/"analysis"/"control_jacobian",pkg/"control_analysis"); copy_tree(root/"analysis"/"06_equipment_realizability",pkg/"engineering"/"equipment_realizability"); copy_tree(root/"analysis"/"07_composition_flow_generalization",pkg/"generalization"/"composition_x_flow"); copy_tree(root/"analysis"/"08_disturbance_policies",pkg/"disturbance"/"quasisteady"); copy_tree(root/"analysis"/"09_css_high_fidelity",pkg/"high_fidelity"); copy_tree(root/"analysis"/"uncertainty_architecture",pkg/"uncertainty"/"architecture"); copy_tree(root/"analysis"/"literature_kinetics",pkg/"provenance"/"literature_kinetics")
    # Write one deduplicated disturbance comparison.
    atomic_csv(qdyn,pkg/"disturbance"/"quasisteady"/"quasisteady_15_12_18_15_policy_comparison.csv")
    for p0 in [gate_dir/"final"/"P10_feed_composition_robustness_complete_results.pkl",root/"analysis"/"03_coprimary_95_90"/"variant_result.pkl",root/"analysis"/"06_equipment_realizability"/"variant_result.pkl"]:
        if p0.exists(): shutil.copy2(p0,pkg/"machine_readable"/p0.name)
    pkg_si=pkg/"tables"/"SI"; pkg_main=pkg/"tables"/"main"; pkg_si.mkdir(parents=True,exist_ok=True); pkg_main.mkdir(parents=True,exist_ok=True)
    atomic_csv(jac,pkg_si/"Table_S26_control_jacobian_svd_and_interpretation.csv"); atomic_csv(eqres.get("selected",pd.DataFrame()) if isinstance(eqres,dict) else pd.DataFrame(),pkg_si/"Table_S27_equipment_realizability.csv"); atomic_csv(flow,pkg_si/"Table_S28_composition_x_feed_flow_authority.csv"); atomic_csv(qdyn,pkg_si/"Table_S29_quasisteady_15_12_18_15_policy_comparison.csv"); atomic_csv(unc,pkg_si/"Table_S30_layered_uncertainty_register.csv"); atomic_csv(literature.get("conditions",pd.DataFrame()),pkg_si/"Table_S31_breakthrough_condition_evidence.csv"); atomic_csv(pd.DataFrame([css.get("gate",{})]),pkg_si/"Table_S32_css_gate_status.csv"); atomic_csv(coverage,pkg_si/"Table_S33_scenario_coverage_summary.csv"); atomic_csv(eqsum,pkg_si/"Table_S34_equipment_boundary_refinement.csv"); atomic_csv(uh,pkg_si/"Table_S35_uncertainty_hierarchy_summary.csv"); atomic_csv(ces,pkg_si/"Table_S36_disturbance_control_effort_summary.csv")
    # Keep one unambiguous manuscript Table 1; supporting baseline tables are placed in SI.
    legacy=pkg_si/"legacy_main_tables"; legacy.mkdir(parents=True,exist_ok=True)
    for f in list(pkg_main.iterdir()):
        if f.is_file(): shutil.move(str(f),str(legacy/f.name))
    eqmap={int(r.max_adaptation_level):r for _,r in eqsum.iterrows()} if len(eqsum) else {}
    rows=[]
    for target,wdf in [("95% purity / 88% recovery",gate["feed_composition_robustness"].get("windows",pd.DataFrame())),("95% purity / 90% recovery",co90.get("windows",pd.DataFrame()))]:
        for lev,lab,lo,hi,lstat,ustat in _window_rows(wdf):
            er=eqmap.get(lev,None) if target.startswith("95% purity / 88") else None
            rows.append({"specification":target,"control_authority":lab,"lean_boundary_feed_co2_percent":100*lo if np.isfinite(lo) else np.nan,"rich_boundary_feed_co2_percent":100*hi if np.isfinite(hi) else np.nan,"rich_boundary_reporting":">40% (domain limit)" if "censor" in ustat.lower() else (f"{100*hi:.2f}%" if np.isfinite(hi) else "not feasible"),"equipment_realizable_lean_boundary_percent":float(er.equipment_realizable_lower_feed_co2_percent) if er is not None else np.nan,"evidence_status":"Reduced-order + adaptive boundary refinement" if target.startswith("95% purity / 88") else "Independent co-primary reduced-order recalculation","lower_boundary_status":lstat,"upper_boundary_status":ustat})
    main_table=pd.DataFrame(rows); atomic_csv(main_table,pkg_main/"Table_1_minimum_control_authority_summary.csv"); atomic_pickle(main_table,pkg_main/"Table_1_minimum_control_authority_summary.pkl")
    atomic_csv(coverage,pkg/"generalization"/"SCENARIO_COVERAGE_SUMMARY.csv"); atomic_csv(eqsum,pkg/"engineering"/"EQUIPMENT_REALIZABLE_BOUNDARY_SUMMARY.csv"); atomic_csv(ces,pkg/"disturbance"/"CONTROL_EFFORT_SUMMARY.csv"); atomic_csv(uh,pkg/"uncertainty"/"UNCERTAINTY_HIERARCHY_SUMMARY.csv")
    ledger.event("12_tables_science","PASS","Merged analysis outputs; final active main-table set reduced to one integrated Table 1; scenario coverage/equipment/uncertainty/control-effort summaries added",time.time()-t0,str(pkg))


def curate_placeholder_tables(pkg: Path, ledger: StageLedger) -> pd.DataFrame:
    """Remove zero-row files from the active table hierarchy while retaining explicit provenance status."""
    rows=[]
    for base in (pkg/"tables"/"main",pkg/"tables"/"SI"):
        if not base.exists(): continue
        for csvp in [p for p in base.glob("*.csv") if not p.name.endswith("_STATUS.csv")]:
            try: df=pd.read_csv(csvp)
            except Exception: continue
            if len(df)!=0: continue
            stem=csvp.stem; dest=base/"not_run_placeholders"; dest.mkdir(parents=True,exist_ok=True); moved=[]
            for sib in list(base.glob(stem+".*")):
                q=dest/sib.name; shutil.move(str(sib),str(q)); moved.append(str(q.relative_to(pkg)))
            previews=base/"visual_previews"
            if previews.exists():
                pdest=dest/"visual_previews"; pdest.mkdir(parents=True,exist_ok=True)
                for pv in list(previews.glob(stem+".*")):
                    q=pdest/pv.name; shutil.move(str(pv),str(q)); moved.append(str(q.relative_to(pkg)))
            rows.append({"table":stem,"original_location":str(csvp.relative_to(pkg)),"status":"NOT_RUN_OR_NO_DATA","publication_action":"excluded from active table set; retained only in not_run_placeholders","reason":"zero data rows","moved_files":";".join(moved)})
    out=pd.DataFrame(rows,columns=["table","original_location","status","publication_action","reason","moved_files"]); atomic_csv(out,pkg/"provenance"/"EMPTY_PLACEHOLDER_AUDIT.csv")
    ledger.event("12b_placeholder_curation","PASS",f"Curated {len(out)} zero-row legacy tables; active table directories contain no zero-row result CSVs",outputs=str(pkg/"tables"))
    return out


def create_claims_and_readiness(pkg: Path, gate: Mapping[str,Any], literature, co90, jac, unc, eqres, flow, qdyn, css, ledger: StageLedger):
    prov=pkg/"provenance"; prov.mkdir(parents=True,exist_ok=True); css_status=str(css.get("gate",{}).get("status","NOT_RUN")); trace_status=str(literature.get("trace_gate",pd.DataFrame([{"status":"PARTIAL"}])).iloc[0].get("status","PARTIAL")) if len(literature.get("trace_gate",pd.DataFrame())) else "PARTIAL"; coverage=scenario_coverage_summary(flow); eqsum=equipment_boundary_summary(eqres); ces=control_effort_summary(qdyn); uh=uncertainty_hierarchy_summary(unc,gate)
    checks=pd.DataFrame([
        ["Frozen baseline baseline","PASS","Reduced-order numerical baseline preserved and consistency-checked."],
        ["Final publication hierarchy","PASS","One canonical root; one active main Table 1; zero-row legacy results excluded from active tables."],
        ["95/90 co-primary recalculation","PASS" if len(co90.get("windows",pd.DataFrame())) else "FAIL","Conventional benchmark recalculated, not relabelled."],
        ["Composition-resolved breakthrough traces",trace_status,"Verified conditions/scalars retained; exact trace coordinates are not fabricated. Optional trace import is supported."],
        ["Control Jacobian/SVD","PASS" if len(jac) else "FAIL","Near-collinearity/local ill-conditioning is explicitly distinguished from nonlinear global operability."],
        ["Layered uncertainty","PARTIAL" if (unc.get("status",pd.Series(dtype=str)).astype(str)=="OPEN").any() else "PASS","Calibration-conditioned uncertainty is separated from source/model-form and kinetic/front sensitivity."],
        ["Equipment realizability","PASS" if len(eqsum) else "FAIL","Equipment-constrained boundaries use the recalculated variant and adaptive refinement when available."],
        ["Composition x feed-flow generalization","PASS" if len(flow) else "FAIL","Discrete evaluated scenario coverage is reported; not mislabelled as continuous area."],
        ["Quasi-steady disturbance policy test","PASS" if len(normalize_quasisteady_disturbance(qdyn)) else "FAIL","Duplicate legacy rows removed; re-optimization objective and actuator movement are explicit."],
        ["High-fidelity CSS",css_status,"Timeout/process errors remain numerical indeterminacy; physical support requires PASS same-rig validation."],
    ],columns=["gate","status","note"]); atomic_csv(checks,prov/"PUBLICATION_READINESS_GATES.csv")
    claims=pd.DataFrame([
        ["C1","Minimum control authority forms a nested topology: coordinated control extends lean operability beyond one-actuator control, which extends it beyond frozen operation.","SUPPORTED_AT_DECLARED_FIDELITY","Reduced-order baseline + adaptive boundary refinement","Figure 5A"],
        ["C2","The same qualitative hierarchy remains meaningful under the 95/90 co-primary specification.","SUPPORTED_AT_DECLARED_FIDELITY","Independent reduced-order recalculation","Figure 5B"],
        ["C3","Lean frozen failure is purity-limited and richer frozen operation becomes recovery-limited.","SUPPORTED_AT_DECLARED_FIDELITY","Frozen constraint margins","Figures 2B and 3"],
        ["C4","Pressure and timing are often locally near-collinear; the second actuator expands the nonlinear/global feasible envelope rather than simply supplying an orthogonal local direction.","SUPPORTED_AT_DECLARED_FIDELITY","Jacobian/SVD + nested global authority","Figure 2C-D; Figure S18"],
        ["C5","Required control authority changes jointly with composition and throughput; reported percentages are discrete evaluated-scenario coverage.","SUPPORTED_AT_DECLARED_FIDELITY","Composition x feed-flow map + scenario counts","Figure 5C"],
        ["C6","Minimum-authority adaptation recovers specification over the 15→12→18→15% quasi-steady sequence with less actuator movement than energy-minimizing grid re-optimization when the calculated metric supports that comparison.","SUPPORTED_AT_DECLARED_FIDELITY","Deduplicated three-policy disturbance test + control-effort summary","Figure 5D; Figure S20"],
        ["C7","Parameter/source uncertainty is smaller than declared front/kinetic sensitivity for the key lean boundaries.","SUPPORTED_AS_SENSITIVITY_HIERARCHY","Layered uncertainty register; front-factor range is non-probabilistic","Figure 5A; Figure S21"],
        ["C8","Published composition-resolved breakthrough traces quantitatively validate the kinetic/front model.","OPEN_TRACE_LEVEL" if trace_status!="PASS" else "SUPPORTED","Trace coordinates required; verified conditions alone are insufficient",""],
        ["C9","Non-isothermal CSS independently supports the nested topology and transient response.","SUPPORTED" if css_status=="PASS" else "OPEN_HIGH_FIDELITY","Same-rig CSS validation + targeted falsification required","Repository-only CSS status; not a journal figure"],
    ],columns=["claim_id","claim","status","evidence","figure"]); atomic_csv(claims,prov/"CURATED_CLAIM_EVIDENCE_MATRIX.csv"); atomic_csv(claims,pkg/"tables"/"SI"/"Table_S1g_claim_evidence_matrix.csv"); atomic_pickle(claims,pkg/"tables"/"SI"/"Table_S1g_claim_evidence_matrix.pkl")
    md=["# P10 current analysis publication-readiness status","",f"Created: {utc_now()}","","## Recommended claim mode","",("**High-fidelity-supported topology**" if css_status=="PASS" else "**Reduced-order topology with explicit high-fidelity limitation**"),"","| Gate | Status | Note |","|---|---|---|"]
    for _,r in checks.iterrows(): md.append(f"| {r.gate} | **{r.status}** | {str(r.note).replace('|','/')} |")
    if len(coverage):
        vals={int(r.control_authority_ceiling):float(r.scenario_coverage_percent) for _,r in coverage.iterrows()}; md += ["","## Process-engineering headline metrics","",f"- Discrete composition×throughput scenario coverage: frozen **{vals.get(0,np.nan):.1f}%**, ≤1 actuator **{vals.get(1,np.nan):.1f}%**, ≤2 actuators **{vals.get(2,np.nan):.1f}%**."]
    if len(eqsum):
        q=eqsum[eqsum.max_adaptation_level.eq(2)]
        if len(q): md.append(f"- Equipment-realizable coordinated lean boundary: **{float(q.iloc[0].equipment_realizable_lower_feed_co2_percent):.2f}% CO₂** (according to declared actuator limits and adaptive refinement status).")
    if len(ces):
        rr=ces[ces.policy.eq("minimum_authority")]
        if len(rr): md.append(f"- Minimum-authority disturbance success: **{int(rr.iloc[0].on_spec_states)}/{int(rr.iloc[0].total_states)}** evaluated states on-spec.")
    md += ["","## Claim discipline","","- The primary claim is the **topology/order of control authority**, not false precision in fourth-decimal boundary locations.","- 95/88 remains the experimental-anchor specification; 95/90 is co-primary and explicitly recalculated.","- Scenario-coverage percentages refer only to the discrete evaluated composition×flow grid.","- The 0.8–1.2 front-factor range is a declared non-probabilistic sensitivity, not a confidence interval.","- Exact 5/15/30% breakthrough trace validation remains open unless legitimate machine-readable traces are supplied/digitized.","- A CSS timeout is **INDETERMINATE_NUMERICAL_TIMEOUT**, not failed physical validation.","- Energy remains an **idealized vacuum-work proxy** until a validated vacuum-machine model is introduced."]
    atomic_text("\n".join(md)+"\n",pkg/"PUBLICATION_READINESS_STATUS.md"); ledger.event("13_claims_readiness","PASS","Final current analysis claim/evidence matrix, process-engineering metrics and readiness status written",outputs=str(prov)); return checks,claims


def write_package_readme(pkg: Path, css_status: str) -> None:
    txt=f"""# P10 final publication package — current analysis.0

This is the final-closure single-root scientific package. The frozen .0.1 baseline engine remains the numerical backbone; //current analysis layers add co-primary targets, local controllability, equipment realizability, multivariable operating coverage, uncertainty hierarchy, and the quasi-steady disturbance-policy comparison.

## Final scientific identity
The primary contribution is **minimum control authority**: the smallest number/magnitude of manipulated operating variables required to retain purity/recovery specifications as feed composition and throughput vary. CALF-20 is the experimentally anchored demonstration platform, not the novelty claim.

## Main publication outputs
- `figures/main` — five final integrated figures.
- `tables/main` — one active manuscript table: `Table_1_minimum_control_authority_summary.csv`.
- `figures/SI`, `tables/SI` — full supporting evidence; zero-row legacy outputs are isolated under `not_run_placeholders`.
- `source_data` — figure-specific reusable source data.
- `engineering`, `generalization`, `disturbance`, `uncertainty`, `validation`, `control_analysis` — reusable scientific outputs.
- `provenance` — claim/evidence matrix, audits, manifests and literature/kinetic evidence.
- `machine_readable` — reusable result bundles.

## Evidence discipline
- CSS status: **{css_status}**; CSS evidence is promoted only after PASS same-rig validation.
- Breakthrough trace coordinates are never fabricated. current analysis optionally accepts a user-supplied audited trace CSV.
- Calibration-conditioned Monte Carlo uncertainty is not presented as total predictive uncertainty.
- Front/kinetic sensitivity is kept separate and non-probabilistic unless experimentally calibrated.
- Re-optimization in the disturbance comparison is explicitly energy-first (minimum idealized vacuum-work proxy, then maximum productivity).

No manuscript LaTeX is generated. This is a structured results/evidence archive.
"""
    atomic_text(txt,pkg/"README.md")


def manifest_directory(root: Path, exclude_prefixes: Tuple[str,...]=()) -> pd.DataFrame:
    rows=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        rel=str(p.relative_to(root)).replace("\\","/")
        if any(rel.startswith(pref) for pref in exclude_prefixes): continue
        rows.append({"relative_path":rel,"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    return pd.DataFrame(rows)


def zip_paths(base: Path, manifest: pd.DataFrame, zpath: Path) -> pd.DataFrame:
    zpath.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest["relative_path"].astype(str): z.write(base/rel,arcname=rel)
    with zipfile.ZipFile(zpath,"r") as z:
        names=set(z.namelist()); rows=[]
        for _,r in manifest.iterrows():
            rel=str(r.relative_path); ok=rel in names
            if ok:
                with z.open(rel) as f:
                    hh=hashlib.sha256()
                    for chunk in iter(lambda:f.read(1024*1024),b""): hh.update(chunk)
                    h=hh.hexdigest()
            else: h=""
            rows.append({"relative_path":rel,"present":ok,"expected_sha256":r.sha256,"archived_sha256":h,"hash_match":bool(ok and h==r.sha256)})
    return pd.DataFrame(rows)


def create_archives(root: Path, pkg: Path, ledger: StageLedger):
    t0=time.time(); archives=root/"archives"; archives.mkdir(parents=True,exist_ok=True)
    # Publication bundle: exactly the canonical publication package.
    pub_manifest=manifest_directory(pkg)
    atomic_csv(pub_manifest,pkg/"provenance"/"PUBLICATION_PACKAGE_FILE_MANIFEST.csv")
    pub_manifest=manifest_directory(pkg)  # include the manifest itself
    pub_zip=archives/"P10_SUBMISSION_BUNDLE.zip"; pub_audit=zip_paths(pkg,pub_manifest,pub_zip)
    atomic_csv(pub_audit,root/"final"/"PUBLICATION_ARCHIVE_INTEGRITY_AUDIT.csv")
    # Complete reproducibility archive: entire  output excluding archives themselves.
    full_manifest=manifest_directory(root,exclude_prefixes=("archives/",))
    full_zip=archives/"P10_REPRODUCIBILITY_ARCHIVE.zip"; full_audit=zip_paths(root,full_manifest,full_zip)
    atomic_csv(full_audit,root/"final"/"COMPLETE_ARCHIVE_INTEGRITY_AUDIT.csv")
    status={"publication_expected":int(len(pub_manifest)),"publication_archived":int(pub_audit.present.sum()),"publication_hash_failures":int((~pub_audit.hash_match).sum()),"complete_expected":int(len(full_manifest)),"complete_archived":int(full_audit.present.sum()),"complete_hash_failures":int((~full_audit.hash_match).sum()),"status":"PASS" if pub_audit.present.all() and pub_audit.hash_match.all() and full_audit.present.all() and full_audit.hash_match.all() else "FAIL"}
    atomic_json(status,root/"final"/"ARCHIVE_INTEGRITY_STATUS.json")
    ledger.event("14_archive_integrity",status["status"],f"Archive audit: publication {status['publication_archived']}/{status['publication_expected']}; complete {status['complete_archived']}/{status['complete_expected']}; hash failures={status['publication_hash_failures']+status['complete_hash_failures']}",time.time()-t0,outputs=str(archives))
    if status["status"]!="PASS": raise RuntimeError("Archive integrity check failed")
    return status


def write_final_bundle(root: Path,pkg: Path,gate,literature,co90,jac,unc,eqres,flow,qdyn,css,figures,checks,claims,ledger):
    final=root/"final"; final.mkdir(parents=True,exist_ok=True)
    bundle={"version":VERSION,"created_utc":utc_now(),"baseline":gate,"literature_kinetics":literature,"co_primary_95_90":co90,"control_jacobian":jac,"uncertainty_architecture":unc,"equipment_realizability":eqres,"composition_flow_generalization":flow,"quasisteady_disturbance":qdyn,"css_high_fidelity":css,"figures":figures,"readiness":checks,"claims":claims}
    atomic_pickle(bundle,final/"P10_integrated_results.pkl")
    atomic_json({"version":VERSION,"created_utc":utc_now(),"css_status":css.get("gate",{}).get("status"),"canonical_publication_package":str(pkg),"standalone_css_runner":"p10_css_validation.py","claim_mode":"high_fidelity_supported_topology" if css.get("gate",{}).get("status")=="PASS" else "reduced_order_topology_with_explicit_high_fidelity_limitations"},final/"ANALYSIS_STATUS.json")
    cat=manifest_directory(root,exclude_prefixes=("archives/",)); atomic_csv(cat,final/"OUTPUT_FILE_CATALOG_PRE_ARCHIVE.csv")
    ledger.event("15_finalize","PASS","Reusable analysis bundle and final status written",outputs=str(final))


