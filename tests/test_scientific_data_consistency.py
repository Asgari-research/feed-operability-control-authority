"""Independent numerical and provenance sanity checks for the archived P10 data.

These do not establish independent dynamic or cyclic-steady-state validation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
R=8.31446261815324

@pytest.mark.parametrize('data_file,param_file,heat_unit',[
 ('input_folder/equilibrium.csv','CALF20_DSL_Liske_Rajendran_2026.csv','kJ'),
 ('external_equilibrium_sensitivity/CALF20_BASF_pilot_DSL_reconstruction.csv','CALF20_DSL_Wilkins_pilot_2025.csv','J'),
])
def test_reconstructed_isotherms_against_published_dsl_parameters(data_file,param_file,heat_unit):
    rows=pd.read_csv(ROOT/'data'/data_file)
    pars=pd.read_csv(ROOT/'data/reference_parameters'/param_file)
    assert len(rows) in (228,240)
    assert not rows.duplicated(['adsorbate','temperature_K','pressure_bar']).any()
    assert (rows.loading_mol_kg>=0).all()
    for (_,temperature),part in rows.groupby(['adsorbate','temperature_K']):
        assert (part.sort_values('pressure_bar').loading_mol_kg.diff().dropna()>0).all()
    predicted=[]
    for row in rows.itertuples():
        c=row.pressure_bar*1e5/(R*row.temperature_K)
        q=0.
        for p in pars[pars.species.eq(row.adsorbate)].itertuples():
            heat=getattr(p, 'minus_deltaU_'+heat_unit+'_mol')*(1000 if heat_unit=='kJ' else 1)
            b=p.b0_m3_mol*np.exp(heat/(R*row.temperature_K))
            q+=p.q_sat_mol_kg*b*c/(1+b*c)
        predicted.append(q)
    assert np.max(np.abs(np.array(predicted)-rows.loading_mol_kg.to_numpy())) < 1.e-7


def test_external_pilot_campaign_structure_and_physical_bounds():
    p=pd.read_csv(ROOT/'data/external_validation/CALF20_pilot_PVSA_75_experiments_2025.csv')
    assert len(p)==75
    assert p.contactor_type.value_counts().to_dict()=={'Laminate':61,'Packed':14}
    assert p.source_table.value_counts().to_dict()=={'S3-1':20,'S3-2':16,'S5':14,'S4':12,'S2':7,'S1':6}
    assert set(np.round(p.feed_co2_fraction,2))=={.08,.12,.16}
    assert np.max(np.abs(p.cycle_time_s-p.cycle_time_recomputed_s))<1.e-10
    for column in ['feed_co2_fraction','purity_fraction','recovery_fraction']:
        assert p[column].between(0,1,inclusive='both').all()
    for column in ['purity_abs_uncertainty','recovery_abs_uncertainty','productivity_abs_uncertainty_tpd_m3']:
        assert p[column].ge(0).all()
    assert p.productivity_tpd_m3.gt(0).all()
    assert p.publication_doi.eq('10.1016/j.cej.2026.174238').all()


def test_primary_process_anchor_is_not_invented():
    p=pd.read_csv(ROOT/'data/input_folder/process_anchor.csv')
    assert len(p)==1
    r=p.iloc[0]
    assert np.isclose(r.feed_co2_fraction,.15)
    assert np.isclose(r.purity,.955)
    assert np.isclose(r.recovery,.881)
    assert all(pd.isna(r[col]) for col in ['adsorption_time_s','regeneration_pressure_bar','productivity','specific_energy'])


def test_reference_boundaries_and_main_table_agree():
    base=pd.read_csv(ROOT/'data/baseline/reference_operability_boundaries.csv')
    vals=dict(zip(base.metric,base.value_percent))
    assert np.isclose(vals['frozen_lower'],14.190)
    assert np.isclose(vals['frozen_upper'],15.075)
    assert np.isclose(vals['one_actuator_lower'],13.400)
    assert np.isclose(vals['two_actuator_lower'],10.590)
    t=pd.read_csv(ROOT/'results/final_submission/tables/main/Table_1_minimum_control_authority_summary.csv')
    assert len(t)==6
    for name,label in [('frozen_lower','Frozen'),('one_actuator_lower','≤1 actuator'),('two_actuator_lower','≤2 actuators')]:
        s=t[(t['Specification']=='95% purity / 88% recovery') & (t['Control authority']==label)].iloc[0]
        # Published table reports 2 decimal places; the baseline contains 3.
        assert abs(float(str(s['Lean boundary (feed CO2)']).strip('%'))-vals[name])<.02


def test_scenario_coverage_and_control_effort_arithmetic():
    t=pd.read_csv(ROOT/'results/final_submission/tables/SI/Table_S55_scenario_coverage_summary.csv')
    t=t.set_index('control_authority_ceiling')
    assert all(int(t.loc[k,'scenarios_covered'])==v for k,v in [(0,37),(1,136),(2,157),(3,48)])
    assert t.total_evaluated_scenarios.eq(205).all()
    for k in (0,1,2,3):
        assert np.isclose(t.loc[k,'scenario_coverage_percent'],100*t.loc[k,'scenarios_covered']/205)
    assert t.loc[2,'scenarios_covered']+t.loc[3,'scenarios_covered']==205
    e=pd.read_csv(ROOT/'results/final_submission/tables/SI/Table_S58_disturbance_control_effort_summary.csv').set_index('policy')
    full=e.loc['full_reoptimization_grid','total_log_actuator_movement']
    minimum=e.loc['minimum_authority','total_log_actuator_movement']
    reduction=100*(1-minimum/full)
    assert np.isclose(reduction,30.06201347329339,rtol=1e-9)
    assert e.loc['minimum_authority','on_spec_states']==4


def test_active_si_tables_have_numeric_records_and_figure_files_exist():
    si=sorted((ROOT/'results/final_submission/tables/SI').glob('Table_S*.csv'))
    assert len(si)==59
    assert sorted(int(p.stem.split('_')[1][1:]) for p in si)==list(range(1,60))
    assert all(len(pd.read_csv(p))>0 for p in si)
    assert len(list((ROOT/'results/final_submission/figures/main').glob('Figure_*.pdf')))==5
    assert len(list((ROOT/'results/final_submission/figures/SI').glob('Figure_S*.pdf')))==21


def test_independent_batch_scalar_values():
    d=pd.read_csv(ROOT/'data/external_equilibrium_sensitivity/CALF20_independent_scalar_crosschecks.csv')
    # Pereira et al. (2025), Inorg. Chem. Table 2, historical source_id begins CALDERON.
    p=d[d.doi.eq('10.1021/acs.inorgchem.4c04727')]
    assert len(p)==2
    r=p.set_index('temperature_K')
    assert np.isclose(r.loc[273.0,'loading_mol_kg'],4.65)
    assert np.isclose(r.loc[298.0,'loading_mol_kg'],3.38)
    assert np.allclose(r.qst_kJ_mol,39.09)


def test_publication_boundary_label_uses_half_up_rounding():
    # Regression test for plot-text precision, not a new scientific prediction.
    import sys
    sys.path.insert(0, str(ROOT))
    from src.p10.analysis import format_two_decimal_percent
    assert format_two_decimal_percent(15.075) == '15.08'
    assert format_two_decimal_percent(14.1917) == '14.19'
