#!/usr/bin/env python
"""Worker for the optional three-policy cyclic disturbance comparison."""
from __future__ import annotations
import argparse, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from core import (
    RunConfig, EngineeringSettings, calf20_pair, mixture_state, _v4_css_config_from_data, _optional_data_root
)
from css_solver import EquilibriumTable, NonIsothermalCSSSolver, CycleOperatingPoint


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    with open(a.input,'rb') as f: z=pickle.load(f)
    c=dict(z['config']); c['engineering']=EngineeringSettings(**c['engineering']); config=RunConfig(**c)
    outdir=Path(z['output_dir']); outdir.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('P10_css_three_policy'); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh=logging.FileHandler(outdir/'P10_css_three_policy_worker.log',mode='a',encoding='utf-8'); fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')); logger.addHandler(fh)
    pair=calf20_pair(z['fit_payload'])
    if pair is None: raise RuntimeError('CALF-20 equilibrium pair unavailable')
    eq=EquilibriumTable(lambda y,p,t: mixture_state(y,p,t,pair),p_range=(0.008,1.10),t_range=(285,315),n_y=21,n_p=24,n_t=7)
    cfg=_v4_css_config_from_data(config)
    vals_path=_optional_data_root(config)/'css_model'/'Nguyen2025_same_rig_dry_validation_cases.csv'
    vals=pd.read_csv(vals_path) if vals_path.exists() else pd.DataFrame()
    b0=vals[vals['case_id'].astype(str).eq('NGUYEN2025_DRY_B0')].iloc[0] if len(vals) else None
    ratio=float(b0['intermediate_pressure_bar']/b0['low_pressure_bar']) if b0 is not None else 1.5
    vfeed=float(b0['feed_superficial_velocity_m_s']) if b0 is not None else 0.024
    policies=z['policies'].copy(); summaries=[]; histories=[]
    for policy in ['frozen','minimum_authority','full_reoptimization_grid']:
        state=None; solver=NonIsothermalCSSSolver(eq,cfg)
        pp=policies[policies['policy'].eq(policy)].sort_values('segment')
        for _,r in pp.iterrows():
            seg=int(r['segment']); y=float(r['feed_co2_fraction']); preg=float(r['p_reg_bar']); ts=float(r['timing_scale']); pi=min(0.30,max(preg+0.005,ratio*preg))
            op=CycleOperatingPoint(case_id=f'POLICY-{policy}-{seg}',feed_co2_fraction=y,high_pressure_bar=config.engineering.adsorption_pressure_bar,intermediate_pressure_bar=pi,low_pressure_bar=preg,adsorption_time_s=config.engineering.nominal_adsorption_time_s,feed_superficial_velocity_m_s=vfeed,timing_scale=ts,source_role='three-policy dynamic comparison; B0 schedule-shape prior')
            logger.info('START policy=%s segment=%d y=%.4f preg=%.5f timing=%.4f',policy,seg,y,preg,ts)
            try:
                sol=solver.run_to_css(op,Y0=state); state=sol['final_state']; sm=dict(sol['summary'])
                sm.update({'policy':policy,'segment':seg,'feed_co2_fraction':y,'solver_status':'OK','solver_error':'','meets_95_88':bool(float(sm.get('purity_fraction',np.nan))>=config.engineering.target_purity and float(sm.get('recovery_fraction',np.nan))>=config.engineering.target_recovery)})
                summaries.append(sm); h=sol['history'].copy(); h['policy']=policy; h['segment']=seg; h['feed_co2_fraction']=y; histories.append(h)
                logger.info('PASS policy=%s segment=%d purity=%s recovery=%s',policy,seg,sm.get('purity_fraction'),sm.get('recovery_fraction'))
            except Exception as exc:
                summaries.append({'policy':policy,'segment':seg,'feed_co2_fraction':y,'p_reg_bar':preg,'timing_scale':ts,'solver_status':'FAIL','solver_error':repr(exc),'meets_95_88':False}); logger.exception('FAIL policy=%s segment=%d',policy,seg); break
    result={'summary':pd.DataFrame(summaries),'history':pd.concat(histories,ignore_index=True,sort=False) if histories else pd.DataFrame()}
    with open(a.output,'wb') as f: pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    return 0

if __name__=='__main__': raise SystemExit(main())
