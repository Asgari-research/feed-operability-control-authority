# P10 manuscript / source-data alignment (29 October 2026)

This follow-up is an **editorial + provenance release** of the independently audited code package. Only the figure-label formatter in `src/p10/analysis.py` was changed, to use conventional half-up two-decimal formatting when the figure is regenerated. No model equations, numerical solver, canonical inputs or numeric CSV files were changed.

- The saved frozen rich edge is **15.075% CO₂** in the relevant source table (and 14.1917% for the lean edge). The paper rounds to **14.19–15.08%**, using conventional two-decimal presentation; the exact value remains in SI Table S19 and data CSVs. Main Figure 5's visible annotation was updated from **15.07%** to **15.08%**, with other plotted content retained; the plotting source was updated to regenerate the same consistent label. This is not a recomputation.
- Pereira et al. (2025), DOI 10.1021/acs.inorgchem.4c04727, replaces the erroneous historical author attribution in bibliography metadata; the legacy machine source ID remains unchanged for data continuity. See `data/provenance/source_notes/Calderon_2025_InorgChem_notes.md` for crosswalk details.
- The 650 kg/m³ assumed engineering bulk density and the 570 kg/m³ parameter in the Liske & Rajendran paper have different evidence roles. See `data/ENGINEERING_DENSITY_CONVENTION.md`. No density substitution or simulation rerun was performed.
- Both Overleaf documents have been recompiled with their citation and figure references. The SI now names the Pereira source as an independent cross-batch check, without treating it as nominal-model calibration or experimental validation.
- Validation boundary remains unchanged: 24 suite tests, including one new plot-precision regression test; 8-draw quickcheck from the prior audit, **not** a 2,000-draw independent full-study reproduction or dynamic/CSS validation.

**Release review needed:** author approvals, rights/licence review, full 2,000-draw clean rerun, and primary-page audit of all third-party numerical transcriptions before journal/data deposition.
