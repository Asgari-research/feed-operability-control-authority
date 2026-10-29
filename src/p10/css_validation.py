#!/usr/bin/env python
"""Run the optional high-fidelity cyclic validation independently from the primary analysis."""
from __future__ import annotations
import argparse, dataclasses, json, logging, os, pickle, signal, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, Mapping
import pandas as pd

import core as base

SCRIPT_DIR=Path(__file__).resolve().parent
WORKER=SCRIPT_DIR/"css_worker.py"


def atomic_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8"); os.replace(tmp,path)

def atomic_text(text,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(text,encoding="utf-8"); os.replace(tmp,path)

def load_pickle(p):
    with open(p,"rb") as f:return pickle.load(f)

def find_gate_root(src:Path)->Path:
    if (src/"analysis/baseline"/"final"/"P10_feed_composition_robustness_complete_results.pkl").exists(): return src/"analysis/baseline"
    if src.name=="analysis/baseline" and (src/"final"/"P10_feed_composition_robustness_complete_results.pkl").exists(): return src
    raise FileNotFoundError("Could not find analysis/baseline/final/P10_feed_composition_robustness_complete_results.pkl")

def settings_from_bundle(bundle):
    d=dict(bundle["manifest"]["configuration"]["engineering"])
    for k in ("purity_threshold_sensitivity","recovery_threshold_sensitivity","vacuum_efficiency_range","bulk_density_range_kg_m3","front_factor_range","contamination_scale_range"):
        if k in d:d[k]=tuple(d[k])
    return base.EngineeringSettings(**d)

def kill_tree(proc):
    try:
        if os.name=="nt": subprocess.run(["taskkill","/F","/T","/PID",str(proc.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=False)
        else:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: proc.kill()
    except Exception:
        try: proc.kill()
        except Exception: pass

def tail(path:Path,n=6)->str:
    if not path.exists(): return "worker log not created yet"
    try:return "\n".join(path.read_text(encoding="utf-8",errors="replace").splitlines()[-n:])
    except Exception:return "could not read worker log"

def make_config(args,out,settings,mode):
    return base.RunConfig(input_folder=str((SCRIPT_DIR.parents[1]/"data"/"input_folder").resolve()),output_folder=str(out.resolve()),save_mode=args.save_mode,ram_mode=args.ram_mode,level=args.level,n_jobs=args.n_jobs,seed=args.seed,force_stage=None,restart=False,preflight_only=False,uncertainty_draws_override=0,engineering=settings,css_mode=mode,css_run_on_gate_fail=bool(args.run_on_gate_fail))

def run_worker(args,bundle,out,mode,wall_time_s):
    dirs=base.ensure_output_tree(out); cfg=make_config(args,out,settings_from_bundle(bundle),mode)
    inp=dirs["checkpoints"]/f"_css_worker_input_{mode}.pkl"; outp=dirs["checkpoints"]/f"_css_worker_result_{mode}.pkl"
    payload={"config":{"input_folder":cfg.input_folder,"output_folder":cfg.output_folder,"save_mode":cfg.save_mode,"ram_mode":cfg.ram_mode,"level":cfg.level,"n_jobs":cfg.n_jobs,"seed":cfg.seed,"force_stage":None,"restart":False,"preflight_only":False,"uncertainty_draws_override":cfg.uncertainty_draws_override,"engineering":dataclasses.asdict(cfg.engineering),"css_mode":cfg.css_mode,"css_run_on_gate_fail":cfg.css_run_on_gate_fail},"dirs":{k:str(v) for k,v in dirs.items()},"fit_payload":bundle["fits"],"evidence_payload":bundle["composition_evidence"],"atlas_payload":bundle["feed_composition_robustness"]}
    with inp.open("wb") as f:pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    launcher=dirs["logs"]/f"P10_css_launcher_{mode}.log"; workerlog=dirs["logs"]/"P10_css_worker.log"; heartbeat=dirs["logs"]/f"CSS_HEARTBEAT_{mode}.md"
    with launcher.open("a",encoding="utf-8") as lf: lf.write(f"{time.ctime()} | START | mode={mode} | wall_time_s={wall_time_s}\n")
    popen_kw={"cwd":str(SCRIPT_DIR),"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL}
    if os.name!="nt":popen_kw["start_new_session"]=True
    proc=subprocess.Popen([sys.executable,str(WORKER),"--input",str(inp),"--output",str(outp)],**popen_kw)
    start=time.monotonic(); last_hb=0
    while proc.poll() is None:
        elapsed=time.monotonic()-start
        if elapsed-last_hb>=args.heartbeat_s:
            txt=f"# CSS heartbeat\n\nMode: `{mode}`  \nPID: `{proc.pid}`  \nElapsed: `{elapsed/60:.1f} min`  \nWall-time guard: `{wall_time_s/3600:.2f} h`\n\n## Latest worker log\n\n```text\n{tail(workerlog,8)}\n```\n"
            atomic_text(txt,heartbeat); print(f"[{mode}] {elapsed/60:.1f} min | {tail(workerlog,1)}",flush=True); last_hb=elapsed
        if elapsed>=wall_time_s:
            kill_tree(proc); gate={"status":"INDETERMINATE_NUMERICAL_TIMEOUT","legacy_status":"FAIL","reason":"css_worker_timeout","mode":mode,"wall_time_limit_s":wall_time_s,"note":"Numerical timeout is not interpreted as physical-model falsification."}; atomic_json(gate,dirs["validation"]/"css_validation_gate.json"); return {"gate":gate,"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"falsification":pd.DataFrame(),"disturbance":pd.DataFrame()},False
        time.sleep(2)
    rc=proc.returncode
    if rc!=0 or not outp.exists():
        gate={"status":"FAIL_NUMERICAL","legacy_status":"FAIL","reason":"css_worker_process_error","returncode":rc,"mode":mode,"note":"Worker/process failure is numerical, not a physical validation result."}; atomic_json(gate,dirs["validation"]/"css_validation_gate.json"); return {"gate":gate,"validation_summary":pd.DataFrame(),"validation_qa":pd.DataFrame(),"falsification":pd.DataFrame(),"disturbance":pd.DataFrame()},False
    result=load_pickle(outp); legacy=str(result.get("gate",{}).get("status","UNKNOWN")).upper(); gate=dict(result.get("gate",{})); gate["legacy_status"]=legacy; gate["status"]="PASS" if legacy=="PASS" else "FAIL_PHYSICAL_VALIDATION" if legacy=="FAIL" else legacy; gate["mode"]=mode; gate["note"]="Only completed validation evidence can produce FAIL_PHYSICAL_VALIDATION; timeout/process failure uses distinct numerical statuses."; result["gate"]=gate; atomic_json(gate,dirs["validation"]/"css_validation_gate.json"); return result,gate["status"]=="PASS"

def main():
    ap=argparse.ArgumentParser(description="P10 high-fidelity CSS validation")
    ap.add_argument("--source-results",required=True,help="Completed P10 result root containing analysis/baseline")
    ap.add_argument("--output-folder",required=True)
    ap.add_argument("--mode",choices=["validate","validate_and_falsify","full_closure"],default="validate")
    ap.add_argument("--level",choices=list(base.LEVELS),default="comprehensive"); ap.add_argument("--save-mode",choices=list(base.SAVE_MODES),default="rich"); ap.add_argument("--ram-mode",choices=list(base.RAM_MODES),default="light"); ap.add_argument("--n-jobs",type=int,default=1); ap.add_argument("--seed",type=int,default=20260921)
    ap.add_argument("--validation-wall-time-s",type=int,default=21600,help="Default 6 h guard for same-rig validation")
    ap.add_argument("--closure-wall-time-s",type=int,default=43200,help="Default 12 h guard for falsification/full closure after PASS validation")
    ap.add_argument("--heartbeat-s",type=int,default=60); ap.add_argument("--run-on-gate-fail",action="store_true")
    a=ap.parse_args(); src=Path(a.source_results).resolve(); out=Path(a.output_folder).resolve(); out.mkdir(parents=True,exist_ok=True)
    gate_root=find_gate_root(src); bundle=load_pickle(gate_root/"final"/"P10_feed_composition_robustness_complete_results.pkl")
    status={"started":time.ctime(),"source_gate_a":str(gate_root),"requested_mode":a.mode}
    try:
        print("Stage HF1: same-rig validation only",flush=True); val,ok=run_worker(a,bundle,out/"HF1_same_rig_validation","validate",a.validation_wall_time_s); status["validation_gate"]=val.get("gate",{})
        if not ok:
            status["final_status"]=val.get("gate",{}).get("status","UNKNOWN"); status["next_action"]="Repair/diagnose same-rig numerical or physical validation before running the falsification portfolio."; atomic_json(status,out/"CSS_FINAL_STATUS.json"); atomic_text(json.dumps(status,indent=2,default=str)+"\n",out/"CSS_FINAL_STATUS.md"); return 0
        if a.mode=="validate": status["final_status"]="PASS"; status["next_action"]="Validation passed; optionally run validate_and_falsify or full_closure."; atomic_json(status,out/"CSS_FINAL_STATUS.json"); atomic_text(json.dumps(status,indent=2)+"\n",out/"CSS_FINAL_STATUS.md"); return 0
        print(f"Stage HF2+: {a.mode} after PASS same-rig gate",flush=True); res,ok2=run_worker(a,bundle,out/f"HF2_{a.mode}",a.mode,a.closure_wall_time_s); status["closure_gate"]=res.get("gate",{}); status["final_status"]=res.get("gate",{}).get("status","UNKNOWN"); status["falsification_rows"]=len(res.get("falsification",pd.DataFrame())); status["disturbance_rows"]=len(res.get("disturbance",pd.DataFrame())); status["next_action"]="Promote CSS evidence only if final_status=PASS and the targeted falsification/dynamic outputs are populated and numerically auditable."
        atomic_json(status,out/"CSS_FINAL_STATUS.json"); atomic_text(json.dumps(status,indent=2,default=str)+"\n",out/"CSS_FINAL_STATUS.md"); return 0
    except Exception as exc:
        status["final_status"]="FAIL_NUMERICAL"; status["exception"]=repr(exc); status["traceback"]=traceback.format_exc(); atomic_json(status,out/"CSS_FINAL_STATUS.json"); print(traceback.format_exc()); return 1

if __name__=="__main__": raise SystemExit(main())
