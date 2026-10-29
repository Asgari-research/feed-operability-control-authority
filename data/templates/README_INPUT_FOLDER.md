P10 input_folder
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

Preferred nominal-process anchor format:
source_id, feed_co2_fraction, purity, recovery, productivity, specific_energy,
regeneration_pressure_bar, adsorption_time_s, doi, notes

The parser also recognizes many common column-name variants and can infer the
CALF-20 5/95, 15/85 and 30/70 mixtures from filenames.  Percentages may be
entered either as fractions (0.15) or percentages (15).  Keep DOI/provenance.
Do not put results inside input_folder.  Byte-identical duplicate files are
hashed and analyzed only once.
