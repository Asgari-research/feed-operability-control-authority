#!/usr/bin/env python
"""Worker process for the optional cyclic validation calculation."""
from __future__ import annotations
import argparse, pickle, logging
from pathlib import Path
import pandas as pd
from core import (
    RunConfig, EngineeringSettings, _v4_run_css_closure
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args()
    with open(a.input,"rb") as f: payload=pickle.load(f)
    c=dict(payload["config"]); c["engineering"]=EngineeringSettings(**c["engineering"])
    config=RunConfig(**c)
    dirs={k:Path(v) for k,v in payload["dirs"].items()}
    logger=logging.getLogger("P10_css_worker"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh=logging.FileHandler(dirs["logs"]/"P10_css_worker.log",mode="a",encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")); logger.addHandler(fh)
    try:
        result=_v4_run_css_closure(config,dirs,payload["fit_payload"],payload["evidence_payload"],payload["atlas_payload"],logger)
        with open(a.output,"wb") as f: pickle.dump(result,f,protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    except Exception:
        logger.exception("Isolated CSS worker failed")
        return 1

if __name__=="__main__":
    raise SystemExit(main())
