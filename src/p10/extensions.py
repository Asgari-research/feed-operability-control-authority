#!/usr/bin/env python
"""Control, uncertainty, equipment and disturbance extensions for the P10 analysis."""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
CORE_PATH = SCRIPT_DIR / "core.py"
ENGINE_VERSION = "6.6.0"
PIPELINE_NAME = "P10 feed-composition robustness extensions"


def _load_core():
    spec = importlib.util.spec_from_file_location("p10_core_engine", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import reduced-order engine: {CORE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_core()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _json_default(x: Any):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, Path): return str(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    if dataclasses.is_dataclass(x): return dataclasses.asdict(x)
    raise TypeError(type(x).__name__)


def safe_rmtree(path: Path) -> None:
    p = path.resolve()
    if str(p) in {"/", str(Path.home().resolve()), str(SCRIPT_DIR.resolve())}:
        raise RuntimeError(f"Refusing unsafe deletion: {p}")
    if p.exists(): shutil.rmtree(p)


class StageLedger:
    """Single source of truth for stage diagnostics."""
    def __init__(self, root: Path, logger: logging.Logger):
        self.root = root
        self.logs = root / "logs"
        self.stage_logs = self.logs / "stages"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.stage_logs.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.logs / "stage_events.jsonl"
        self.csv_path = self.logs / "STAGE_STATUS.csv"
        self.md_path = self.logs / "STAGE_STATUS.md"
        self.logger = logger
        self.rows: List[Dict[str, Any]] = []
        if self.csv_path.exists():
            try: self.rows = pd.read_csv(self.csv_path).to_dict("records")
            except Exception: self.rows = []

    def event(self, stage: str, status: str, message: str = "", elapsed_s: Optional[float] = None,
              exception: str = "", outputs: str = "") -> None:
        row = {
            "utc": utc_now(), "stage": stage, "status": status,
            "elapsed_s": np.nan if elapsed_s is None else round(float(elapsed_s), 3),
            "message": message, "exception": exception, "outputs": outputs,
        }
        self.rows.append(row)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")
        with (self.stage_logs / f"{stage}.log").open("a", encoding="utf-8") as f:
            f.write(f"{row['utc']} | {status} | {message}")
            if exception: f.write(f" | exception={exception}")
            if outputs: f.write(f" | outputs={outputs}")
            f.write("\n")
        atomic_csv(pd.DataFrame(self.rows), self.csv_path)
        latest = pd.DataFrame(self.rows).groupby("stage", as_index=False).tail(1)
        md = ["# P10 stage status", "", f"Updated: {utc_now()}", "", "| Stage | Status | Elapsed (s) | Message |", "|---|---|---:|---|"]
        for _, r in latest.iterrows():
            msg = str(r.get("message", "")).replace("|", "/").replace("\n", " ")
            el = "" if pd.isna(r.get("elapsed_s")) else f"{float(r['elapsed_s']):.1f}"
            md.append(f"| {r['stage']} | **{r['status']}** | {el} | {msg} |")
        atomic_text("\n".join(md) + "\n", self.md_path)
        level = logging.ERROR if status == "FAIL" else logging.WARNING if status == "PARTIAL" else logging.INFO
        self.logger.log(level, "%s | %s | %s", stage, status, message)

    @contextmanager
    def stage(self, stage: str, description: str):
        t0 = time.time(); self.event(stage, "START", description)
        try:
            yield
        except Exception as exc:
            self.event(stage, "FAIL", f"{description} failed", time.time()-t0, repr(exc))
            with (self.stage_logs / f"{stage}.log").open("a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")
            raise
        else:
            self.event(stage, "PASS", description, time.time()-t0)


def setup_logger(out: Path) -> logging.Logger:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("P10_extensions")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out / "logs" / "P10_extensions.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f: return pickle.load(f)


def make_base_config(args, output: Path, settings, css_mode: str = "off"):
    return base.RunConfig(
        input_folder=str(Path(args.input_folder).resolve()), output_folder=str(output.resolve()),
        save_mode=args.save_mode, ram_mode=args.ram_mode, level=args.level,
        n_jobs=args.n_jobs, seed=args.seed, force_stage=None, restart=False,
        preflight_only=False, uncertainty_draws_override=args.uncertainty_draws,
        engineering=settings, css_mode=css_mode,
        css_run_on_gate_fail=bool(args.css_run_on_gate_fail),
    )


def settings_from_bundle(bundle: Mapping[str, Any]):
    d = dict(bundle["manifest"]["configuration"]["engineering"])
    # JSON/pickle preserves tuples, but accept lists as well.
    for k in ("purity_threshold_sensitivity", "recovery_threshold_sensitivity", "vacuum_efficiency_range", "bulk_density_range_kg_m3", "front_factor_range", "contamination_scale_range"):
        if k in d: d[k] = tuple(d[k])
    return base.EngineeringSettings(**d)


def run_baseline(args, root: Path, ledger: StageLedger, logger: logging.Logger) -> Mapping[str, Any]:
    gate = root / "analysis" / "baseline"
    bundle_path = gate / "final" / "P10_feed_composition_robustness_complete_results.pkl"
    if bundle_path.exists() and not args.fresh_run:
        ledger.event("00_baseline", "PASS", "Reused completed reduced-order baseline", outputs=str(bundle_path))
        return load_pickle(bundle_path)
    cmd = [
        sys.executable, str(CORE_PATH), "--input-folder", str(Path(args.input_folder).resolve()),
        "--output-folder", str(gate.resolve()), "--save-mode", args.save_mode,
        "--ram-mode", args.ram_mode, "--level", args.level, "--n-jobs", str(args.n_jobs),
        "--seed", str(args.seed), "--uncertainty-draws", str(args.uncertainty_draws),
        "--target-purity", "0.95", "--target-recovery", "0.88", "--css-mode", "off",
    ]
    if args.fresh_run: cmd.append("--fresh-run")
    logp = root / "logs" / "stages" / "00_baseline_subprocess.log"
    ledger.event("00_baseline", "START", "Running reduced-order baseline", outputs=str(logp))
    t0 = time.time()
    with logp.open("a", encoding="utf-8") as f:
        f.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not bundle_path.exists():
        ledger.event("00_baseline", "FAIL", f"Reduced-order baseline subprocess failed (return code {proc.returncode}); inspect stage log", time.time()-t0, outputs=str(logp))
        raise RuntimeError("Reduced-order baseline run failed")
    ledger.event("00_baseline", "PASS", "Reduced-order baseline completed", time.time()-t0, outputs=str(bundle_path))
    return load_pickle(bundle_path)


def _window_value(w: pd.DataFrame, level: int, side: str) -> float:
    q = w[pd.to_numeric(w["max_adaptation_level"], errors="coerce").eq(level)]
    if q.empty: return np.nan
    return float(pd.to_numeric(q.iloc[0][f"{side}_feed_co2_fraction"], errors="coerce"))


def verify_baseline(root: Path, bundle: Mapping[str, Any], args, ledger: StageLedger) -> pd.DataFrame:
    refs = pd.read_csv(DATA_ROOT / "baseline" / "reference_operability_boundaries.csv")
    w = bundle["feed_composition_robustness"].get("windows", pd.DataFrame())
    got = {
        "frozen_lower": _window_value(w, 0, "lower"),
        "frozen_upper": _window_value(w, 0, "upper"),
        "one_actuator_lower": _window_value(w, 1, "lower"),
        "two_actuator_lower": _window_value(w, 2, "lower"),
    }
    rows=[]
    for _, r in refs.iterrows():
        name=str(r.metric); exp=float(r.value_fraction); obs=got.get(name,np.nan)
        strict_tol=float(r.tolerance_percentage_points)/100.0
        # The frozen manuscript baseline is a comprehensive-level reference. Lower
        # levels are diagnostic/resolution-reduced runs and therefore receive a
        # slightly wider numerical smoke-test tolerance; comprehensive/exhaustive
        # production runs retain the strict frozen tolerance.
        tol=strict_tol if getattr(args,"level","comprehensive") in {"comprehensive","exhaustive"} else max(strict_tol,0.0020)
        ok=bool(np.isfinite(obs) and abs(obs-exp)<=tol)
        rows.append({"metric":name,"expected_fraction":exp,"observed_fraction":obs,"absolute_difference_fraction":abs(obs-exp) if np.isfinite(obs) else np.nan,"strict_manuscript_tolerance_fraction":strict_tol,"applied_tolerance_fraction":tol,"run_level":getattr(args,"level","comprehensive"),"status":"PASS" if ok else "FAIL"})
    out=pd.DataFrame(rows); p=root/"analysis"/"baseline_consistency"/"baseline_consistency_audit.csv"; atomic_csv(out,p)
    overall="PASS" if (out.status=="PASS").all() else "FAIL"
    ledger.event("01_baseline_consistency",overall,"Baseline headline-boundary consistency audit",outputs=str(p))
    if overall=="FAIL" and not args.allow_baseline_drift:
        raise RuntimeError("Baseline headline boundaries drifted beyond the reference tolerance. Use --allow-baseline-drift only for diagnosis, not manuscript production.")
    return out


def run_literature_kinetic_audit(root: Path, bundle: Mapping[str, Any], ledger: StageLedger) -> Dict[str,pd.DataFrame]:
    outdir=root/"analysis"/"literature_kinetics"; outdir.mkdir(parents=True,exist_ok=True)
    conditions=pd.read_csv(DATA_ROOT/"literature_evidence"/"CALF20_breakthrough_condition_registry.csv")
    scalars=pd.read_csv(DATA_ROOT/"literature_evidence"/"CALF20_kinetic_scalar_crosschecks.csv")
    drow=scalars[scalars.quantity.eq("D_over_R2")].iloc[0]
    d=float(drow.value); tads=float(bundle["manifest"]["configuration"]["engineering"]["nominal_adsorption_time_s"])
    diagnostics=pd.DataFrame([
        {"diagnostic":"diffusion_time_scale_1_over_D_R2","value":1.0/d,"unit":"s","interpretation":"independent ZLC scalar; different state point"},
        {"diagnostic":"simple_first_order_fraction_over_nominal_adsorption_time","value":1.0-math.exp(-d*tads),"unit":"fraction","interpretation":"diagnostic only; not a full packed-bed front model"},
        {"diagnostic":"spherical_LDF_rate_15D_R2","value":15.0*d,"unit":"s^-1","interpretation":"ideal spherical LDF mapping used only as transport-scale context"},
        {"diagnostic":"spherical_LDF_fraction_over_nominal_adsorption_time","value":1.0-math.exp(-15.0*d*tads),"unit":"fraction","interpretation":"diagnostic only"},
    ])
    traces=bundle.get("composition_evidence",{}).get("traces",pd.DataFrame())
    trace_status=pd.DataFrame([{
        "gate":"composition_resolved_breakthrough_trace_validation",
        "status":"PASS" if isinstance(traces,pd.DataFrame) and len(traces)>0 else "PARTIAL",
        "n_machine_readable_trace_rows":len(traces) if isinstance(traces,pd.DataFrame) else 0,
        "note":"Verified 5/15/30% experimental conditions are packaged. Exact trace coordinates are not fabricated; trace-level calibration remains open until legitimately acquired/digitized source data are available."
    }])
    atomic_csv(conditions,outdir/"verified_breakthrough_conditions.csv"); atomic_csv(scalars,outdir/"kinetic_scalar_crosschecks.csv"); atomic_csv(diagnostics,outdir/"kinetic_time_scale_diagnostics.csv"); atomic_csv(trace_status,outdir/"breakthrough_trace_gate.csv")
    ledger.event("02_literature_kinetics",str(trace_status.iloc[0].status),trace_status.iloc[0].note,outputs=str(outdir))
    return {"conditions":conditions,"scalars":scalars,"diagnostics":diagnostics,"trace_gate":trace_status}


def run_stage05_variant(name: str, root: Path, args, gate_bundle: Mapping[str,Any], settings, logger: logging.Logger, ledger: StageLedger) -> Dict[str,Any]:
    outroot=root/"analysis"/name; dirs=base.ensure_output_tree(outroot)
    cfg=make_base_config(args,outroot,settings,css_mode="off")
    t0=time.time(); ledger.event(name,"START",f"Running reduced-order variant: {name}")
    try:
        res=base.stage_05_feed_composition_robustness(cfg,dirs,gate_bundle["fits"],gate_bundle["composition_evidence"],logger)
        p=outroot/"variant_result.pkl"; atomic_pickle(res,p)
        ledger.event(name,"PASS",f"Completed reduced-order variant: {name}",time.time()-t0,outputs=str(p))
        return res
    except Exception as exc:
        ledger.event(name,"FAIL",f"Variant {name} failed",time.time()-t0,repr(exc),str(outroot)); raise


def control_jacobian(root: Path, gate_bundle: Mapping[str,Any], settings, ledger: StageLedger, log_h: float=0.02) -> pd.DataFrame:
    outdir=root/"analysis"/"control_jacobian"; outdir.mkdir(parents=True,exist_ok=True)
    atlas=gate_bundle["feed_composition_robustness"]; pair=base.calf20_pair(gate_bundle["fits"]); cal=atlas["calibration"]; ft=atlas.get("front_table",pd.DataFrame()); sel=atlas.get("selected",pd.DataFrame())
    rows=[]
    for _,r in sel.iterrows():
        if not bool(r.get("success",False)) or int(pd.to_numeric(pd.Series([r.get("adaptation_level",3)]),errors="coerce").fillna(3).iloc[0])>2: continue
        y=float(r.feed_co2_fraction); p=float(r.get("p_reg_bar",settings.nominal_regeneration_pressure_bar)); t=float(r.get("timing_scale",1.0))
        pminus=max(settings.regeneration_pressure_floor_bar,p*math.exp(-log_h)); pplus=min(settings.adsorption_pressure_bar*0.95,p*math.exp(log_h))
        tminus=max(settings.timing_min_scale,t*math.exp(-log_h)); tplus=min(settings.timing_max_scale,t*math.exp(log_h))
        a=base.process_surrogate(y,pminus,t,pair,cal,ft,settings); b=base.process_surrogate(y,pplus,t,pair,cal,ft,settings); c=base.process_surrogate(y,p,tminus,pair,cal,ft,settings); d=base.process_surrogate(y,p,tplus,pair,cal,ft,settings)
        dp=max(math.log(pplus)-math.log(pminus),1e-12); dt=max(math.log(tplus)-math.log(tminus),1e-12)
        J=np.array([[(b["purity_estimate"]-a["purity_estimate"])/dp,(d["purity_estimate"]-c["purity_estimate"])/dt],[(b["recovery_estimate"]-a["recovery_estimate"])/dp,(d["recovery_estimate"]-c["recovery_estimate"])/dt]],float)
        sv=np.linalg.svd(J,compute_uv=False); cond=float(sv[0]/sv[-1]) if sv[-1]>1e-12 else np.inf
        vp=J[:,0]; vt=J[:,1]; denom=np.linalg.norm(vp)*np.linalg.norm(vt); angle=float(np.degrees(np.arccos(np.clip(np.dot(vp,vt)/denom,-1,1)))) if denom>1e-14 else np.nan
        clipped=bool(abs(pminus-settings.regeneration_pressure_floor_bar)<1e-12 or abs(pplus-settings.adsorption_pressure_bar*0.95)<1e-12 or abs(tminus-settings.timing_min_scale)<1e-12 or abs(tplus-settings.timing_max_scale)<1e-12)
        extreme=bool((not np.isfinite(cond)) or cond>1e6 or np.nanmax(np.abs(J))>1e3)
        rows.append({"feed_co2_fraction":y,"policy_adaptation_level":int(r.get("adaptation_level",3)),"p_reg_bar":p,"timing_scale":t,"d_purity_d_lnpreg":J[0,0],"d_purity_d_lntiming":J[0,1],"d_recovery_d_lnpreg":J[1,0],"d_recovery_d_lntiming":J[1,1],"singular_value_1":sv[0],"singular_value_2":sv[-1],"condition_number":cond,"actuator_response_angle_deg":angle,"pressure_response_norm":float(np.linalg.norm(vp)),"timing_response_norm":float(np.linalg.norm(vt)),"derivative_stencil_clipped":clipped,"jacobian_quality_flag":"CHECK_NONSMOOTH_OR_BOUNDARY" if extreme else ("BOUNDARY_STENCIL" if clipped else "OK")})
    df=pd.DataFrame(rows); atomic_csv(df,outdir/"control_jacobian_svd_map.csv")
    ledger.event("04_control_jacobian","PASS" if len(df) else "FAIL",f"Computed Jacobian/SVD at {len(df)} feasible composition states",outputs=str(outdir))
    return df


def _extract_external_boundaries(gate_bundle: Mapping[str,Any]) -> pd.DataFrame:
    val=gate_bundle.get("validation",{})
    obj=val.get("external_equilibrium_source_sensitivity",val.get("external_equilibrium",{}))
    if isinstance(obj,dict): return obj.get("boundaries",pd.DataFrame()).copy()
    return pd.DataFrame()


def layered_uncertainty(root: Path, gate_bundle: Mapping[str,Any], settings, ledger: StageLedger) -> pd.DataFrame:
    outdir=root/"analysis"/"uncertainty_architecture"; outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    rob=gate_bundle.get("robustness",{})
    b=rob.get("boundaries",pd.DataFrame())
    if isinstance(b,pd.DataFrame) and len(b):
        for _,r in b.iterrows():
            rows.append({"layer":"calibration_conditioned_monte_carlo","evidence_type":"probabilistic_conditional","intervention_ceiling":r.get("intervention_ceiling",r.get("class","")),"lower_median":r.get("lower_median",r.get("median_lower_y",np.nan)),"lower_q025":r.get("lower_q025",r.get("q025_lower_y",np.nan)),"lower_q975":r.get("lower_q975",r.get("q975_lower_y",np.nan)),"status":"AVAILABLE","interpretation":"Conditional on nominal-anchor reproduction; not total predictive uncertainty."})
    ext=_extract_external_boundaries(gate_bundle)
    for _,r in ext.iterrows():
        rows.append({"layer":"independent_thermodynamic_parameterization","evidence_type":"deterministic_source_sensitivity","intervention_ceiling":r.get("intervention_ceiling",""),"lower_median":r.get("lower_feed_co2_fraction",np.nan),"lower_q025":np.nan,"lower_q975":np.nan,"status":"AVAILABLE","interpretation":"Alternative published thermodynamic parameterization; source lineage should be reported."})
    stress=gate_bundle.get("validation",{}).get("thermodynamic_discrepancy_stress",{})
    if isinstance(stress,dict): stress=stress.get("boundaries",pd.DataFrame())
    if isinstance(stress,pd.DataFrame):
        for _,r in stress.iterrows():
            rows.append({"layer":"thermodynamic_model_discrepancy_stress","evidence_type":"declared_stress_test","intervention_ceiling":r.get("intervention_ceiling",""),"lower_median":r.get("lower_feed_co2_fraction",np.nan),"lower_q025":np.nan,"lower_q975":np.nan,"status":"AVAILABLE","interpretation":str(r.get("conditioning_mode","declared stress test"))})
    # Kinetic/front-utilization sensitivity is kept separate from probability-based
    # uncertainty because the 0.8–1.2 range is a declared engineering stress range,
    # not an experimentally identified distribution.
    atlas=gate_bundle["feed_composition_robustness"]; pair=base.calf20_pair(gate_bundle["fits"]); cal=atlas.get("calibration",{}); ft=atlas.get("front_table",pd.DataFrame())
    ys=np.linspace(settings.feed_co2_min,settings.feed_co2_max,41); frows=[]
    for fm in (float(settings.front_factor_range[0]),1.0,float(settings.front_factor_range[1])):
        selected=[]
        for y in ys:
            d=base._solve_adaptation_levels_grid_v130(float(y),pair,cal,ft,settings,21,front_multiplier=fm)
            ok=d[d["success"].astype(bool)].copy() if len(d) else pd.DataFrame()
            if len(ok):
                lev=int(pd.to_numeric(ok["adaptation_level"],errors="coerce").min()); rr=ok[pd.to_numeric(ok["adaptation_level"],errors="coerce").eq(lev)].iloc[0].to_dict(); rr["success"]=True
            else:
                rr={"adaptation_level":3,"success":False}
            rr["feed_co2_fraction"]=float(y); selected.append(rr)
        rw,_=base._v3_refine_windows(pd.DataFrame(selected),pair,cal,ft,settings,21,front_multiplier=fm)
        if len(rw):
            rw=rw.copy(); rw["front_multiplier"]=fm; frows.append(rw)
            for _,r in rw.iterrows():
                rows.append({"layer":"kinetic_front_utilization_sensitivity","evidence_type":"declared_nonprobabilistic_sensitivity","intervention_ceiling":r.get("intervention_ceiling",""),"lower_median":r.get("lower_feed_co2_fraction",np.nan),"lower_q025":np.nan,"lower_q975":np.nan,"status":"AVAILABLE","interpretation":f"front_multiplier={fm:.2f}; declared sensitivity only, not trace-calibrated probability"})
    fsens=pd.concat(frows,ignore_index=True,sort=False) if frows else pd.DataFrame(); atomic_csv(fsens,outdir/"front_factor_boundary_sensitivity.csv")
    # Explicitly declare missing layers rather than manufacturing probability distributions.
    for layer,note in [
        ("composition_resolved_kinetics","Trace-level 5/15/30 breakthrough coordinates not yet legitimately machine-readable; independent kinetic scalars and declared front-factor sensitivity are available, but trace-level probability remains open."),
        ("high_fidelity_cycle_model","Populated only if the independent CSS validation gate passes."),
        ("nominal_process_anchor_measurement","No unsupported probability law is imposed on the 95.5/88.1 anchor; add reported experimental uncertainty if primary source supplies it."),
    ]:
        rows.append({"layer":layer,"evidence_type":"open_uncertainty_layer","intervention_ceiling":"","lower_median":np.nan,"lower_q025":np.nan,"lower_q975":np.nan,"status":"OPEN","interpretation":note})
    df=pd.DataFrame(rows); atomic_csv(df,outdir/"uncertainty_layer_register.csv")
    atomic_text("# Uncertainty architecture\n\nThe analysis separates calibration-conditioned Monte Carlo uncertainty from source/model-form, kinetic/transport, high-fidelity cycle-model, and experimental-anchor uncertainty. Open layers are reported as OPEN rather than being assigned invented distributions.\n",outdir/"README.md")
    ledger.event("05_uncertainty_architecture","PARTIAL" if (df.status=="OPEN").any() else "PASS","Layered uncertainty register written; open evidence layers are explicit rather than hidden",outputs=str(outdir))
    return df


def equipment_realizability(root: Path, args, gate_bundle: Mapping[str,Any], base_settings, logger, ledger):
    eqs=dataclasses.replace(base_settings,regeneration_pressure_floor_bar=args.equipment_min_p_reg_bar,timing_min_scale=args.equipment_min_timing_scale,timing_max_scale=args.equipment_max_timing_scale)
    res=run_stage05_variant("06_equipment_realizability",root,args,gate_bundle,eqs,logger,ledger)
    sel=res.get("selected",pd.DataFrame()).copy()
    if len(sel):
        sel["equipment_realizable"]=(pd.to_numeric(sel.get("p_reg_bar"),errors="coerce")>=args.equipment_min_p_reg_bar-1e-12)&(pd.to_numeric(sel.get("timing_scale"),errors="coerce")>=args.equipment_min_timing_scale-1e-12)&(pd.to_numeric(sel.get("timing_scale"),errors="coerce")<=args.equipment_max_timing_scale+1e-12)&sel.get("success",False).astype(bool)
        # Vacuum-work efficiency envelope: same process state, change only engineering efficiency interpretation.
        base_work=pd.to_numeric(sel.get("specific_vacuum_work_kJ_mol_CO2"),errors="coerce")
        eta0=float(base_settings.vacuum_efficiency)
        for eta in (0.45,0.60,0.75): sel[f"idealized_vacuum_work_proxy_eta_{eta:.2f}"]=base_work*eta0/eta
        atomic_csv(sel,root/"analysis"/"06_equipment_realizability"/"equipment_realizable_selected_policies.csv")
    return res


def flow_generalization(root: Path, gate_bundle: Mapping[str,Any], settings, ledger, n_comp: int=41) -> pd.DataFrame:
    outdir=root/"analysis"/"07_composition_flow_generalization"; outdir.mkdir(parents=True,exist_ok=True)
    atlas=gate_bundle["feed_composition_robustness"]; pair=base.calf20_pair(gate_bundle["fits"]); ft=atlas.get("front_table",pd.DataFrame()); basecal=atlas["calibration"]
    ys=np.linspace(max(0.05,settings.primary_evidence_feed_min),min(0.30,settings.primary_evidence_feed_max),n_comp)
    scales=[0.75,0.90,1.00,1.10,1.25]; rows=[]
    n=max(17,min(31,int(base.LEVELS.get("comprehensive",{}).get("pressure_n",41))))
    for fs in scales:
        cal=copy.deepcopy(basecal); cal["nominal_feed_mol_kg_cycle"]=float(basecal["nominal_feed_mol_kg_cycle"])*fs
        for y in ys:
            lev=int(base._v3_grid_level(float(y),pair,cal,ft,settings,n))
            rows.append({"feed_co2_fraction":float(y),"feed_flow_scale":fs,"minimum_control_authority_level":lev,"authority_label":{0:"frozen",1:"one_actuator",2:"two_actuator",3:"outside_envelope"}.get(lev,"outside_envelope"),"scope":"reduced-order composition x feed-throughput generalization"})
    df=pd.DataFrame(rows); atomic_csv(df,outdir/"composition_x_feed_flow_authority_map.csv")
    ledger.event("07_composition_flow_generalization","PASS",f"Computed {len(df)} composition x feed-flow states",outputs=str(outdir))
    return df


def _full_reopt_grid(y: float,pair,cal,ft,settings) -> Dict[str,Any]:
    """Energy-first 2-D policy search with cached pressure-release states.

    `process_surrogate` otherwise repeats the competitive equilibrium solve for every
    timing value. Caching the pressure-dependent release state makes this closure
    diagnostic fast enough for routine runs without changing its mathematics.
    """
    pgrid=np.geomspace(settings.nominal_regeneration_pressure_bar,settings.regeneration_pressure_floor_bar,21)
    tgrid=np.linspace(settings.timing_min_scale,settings.timing_max_scale,21)
    ff=base.front_factor(float(y),ft)
    best=None
    for p in pgrid:
        rel=base.release_state(float(y),float(p),pair,settings,ff); rel["front_factor"]=ff
        for t in tgrid:
            r=base._process_metrics_from_release(float(y),float(p),float(t),rel,cal,settings)
            if r["purity_estimate"]+1e-12<settings.target_purity or r["recovery_estimate"]+1e-12<settings.target_recovery: continue
            key=(float(r["specific_vacuum_work_kJ_mol_CO2"]),-float(r["productivity_mol_kg_h"]))
            if best is None or key<best[0]: best=(key,r)
    return {} if best is None else dict(best[1])


def quasisteady_disturbance(root: Path, gate_bundle: Mapping[str,Any], settings, ledger) -> pd.DataFrame:
    outdir=root/"analysis"/"08_disturbance_policies"; outdir.mkdir(parents=True,exist_ok=True)
    atlas=gate_bundle["feed_composition_robustness"]; pair=base.calf20_pair(gate_bundle["fits"]); cal=atlas["calibration"]; ft=atlas.get("front_table",pd.DataFrame())
    rows=[]
    for seg,y in enumerate([0.15,0.12,0.18,0.15],1):
        frozen=base.process_surrogate(y,settings.nominal_regeneration_pressure_bar,1.0,pair,cal,ft,settings)
        ad=base.solve_adaptation_levels(y,pair,cal,ft,settings,31); ok=ad[ad.success.astype(bool)].copy()
        minpol=ok.sort_values(["adaptation_level","control_move_linf","control_move_l1"]).iloc[0].to_dict() if len(ok) else {}
        full=_full_reopt_grid(y,pair,cal,ft,settings)
        for pol,r in [("frozen",frozen),("minimum_authority",minpol),("full_reoptimization_grid",full)]:
            if not r:
                rows.append({"segment":seg,"feed_co2_fraction":y,"policy":pol,"success":False}); continue
            rows.append({"segment":seg,"feed_co2_fraction":y,"policy":pol,"success":bool(r.get("meets_targets",r.get("success",False))),"purity":r.get("purity_estimate",np.nan),"recovery":r.get("recovery_estimate",np.nan),"p_reg_bar":r.get("p_reg_bar",np.nan),"timing_scale":r.get("timing_scale",np.nan),"productivity_mol_kg_h":r.get("productivity_mol_kg_h",np.nan),"idealized_vacuum_work_proxy_kJ_mol_CO2":r.get("specific_vacuum_work_kJ_mol_CO2",np.nan),"purity_shortfall":max(0.0,settings.target_purity-float(r.get("purity_estimate",0.0))),"recovery_shortfall":max(0.0,settings.target_recovery-float(r.get("recovery_estimate",0.0))),"fidelity":"quasi-steady reduced-order; not transient CSS"})
    df=pd.DataFrame(rows); atomic_csv(df,outdir/"quasisteady_15_12_18_15_policy_comparison.csv")
    ledger.event("08_quasisteady_disturbance","PASS","Three-policy 15→12→18→15% quasi-steady disturbance comparison completed; transient CSS remains separately gated",outputs=str(outdir))
    return df


