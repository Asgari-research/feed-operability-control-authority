# CSS validation hook — deliberately empty in the literature data release

No cyclic-steady-state (CSS) rows are fabricated here. The literature sources audited for this
available public sources do not report the exact falsification cases at the predicted intervention
boundaries. The code therefore treats this folder as optional.

After a rigorous non-isothermal cycle model is run, save one or more CSVs here with at least:
`feed_co2_fraction,purity,recovery`.
Recommended fields are:
`case_id,css_converged,p_reg_bar,timing_scale,intervention_class,energy_kJ_mol_CO2,source_or_model,notes`.

The pipeline will ingest these files automatically without using them for nominal calibration.
