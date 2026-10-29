# P10 CALF-20 — independent data, provenance and GitHub-package audit

**Audit date:** 29 October 2026  
**Package inspected:** `P10_02_GITHUB_REPOSITORY.zip`, compared with `P10_03_INDEPENDENT_RERUN.zip`, the three-author manuscript/SI archive and Overleaf archive.  
**Software version:** 6.6.0.  
**Scope:** independent checking of archive/file integrity, selected primary-source facts, deterministic isotherm reconstructions, numerical table arithmetic, basic software operation and editorial/data crosswalks. This is **not** a full scientific re-computation or independent validation of high-fidelity VSA performance.

## Executive verdict

**CONDITIONALLY READY for a private GitHub release, collaboration and independent rerun.** The package is structurally sound and internally consistent for the checks described below. A fresh smoke test and 23 regression/data-integrity checks pass in the audited edition. Its scientific result should be described as a *reduced-order, calibration-conditioned operability study*. Final journal-level reproducibility, complete row-by-row literature transcription and independent cyclic-steady-state validation are **not certified** by this audit.

## 1. Checks completed and evidence

| Area | Check and result | Status |
|---|---|---|
| Archive | Original ZIP CRC check: no errors; 236 archive members. | PASS |
| Provenance manifest | All 235 original manifest-listed files matched SHA256 and byte counts. The manifest excludes itself. | PASS |
| Code | Python source syntax compilation successful; no obvious hard-coded secret detected in a targeted text scan. | PASS, limited security scope |
| Existing tests | 15/15 supplied pytest tests passed. Many are structural or archive regression tests and are not full physical-model verification. | PASS |
| Additional audit tests | 8 scientific/data integrity tests added; resulting suite 23/23 passed. | PASS |
| End-to-end smoke test | Fresh `--level quickcheck`, 8 uncertainty draws, completed successfully (stages 00–07 only). | PASS, smoke test only |
| CSV readability | All 158 CSV files in the inspected original GitHub tree parsed successfully. Several intentionally contain missing values for infeasible cases or unobserved source fields. | PASS |
| Figure files | All 5 main + 21 SI figure PDFs readable; the 26 corresponding files are byte-identical in GitHub, paper and Overleaf archives. | PASS |
| Supporting tables | 59 consecutively numbered SI CSVs, each nonempty; main table and selected provenance/source-data reconciled. | PASS |
| Archive concordance | Main entry scripts and inspected canonical inputs byte-identical between GitHub and independent rerun archives. | PASS |
| Full publication rerun | Not performed; 2,000-draw comprehensive execution can be resource-intensive. | NOT VERIFIED |
| Dynamic/CSS experimental validation | Not present as a completed independently validated result. | NOT VERIFIED |

## 2. Literature data and reconstructed equilibrium

**Nominal 2023 process anchor.** The package uses 15 mol% CO2 feed, 95.5% CO2 purity and 88.1% recovery. Those numbers and the DOI are consistent with Nguyen, Shimizu & Rajendran, *Chemical Engineering Journal* 452 (2023), 139550, DOI 10.1016/j.cej.2022.139550. The input file properly leaves unavailable point-specific regeneration pressure, adsorption time, productivity and specific energy blank. The nominal pressure/timing settings used in the reduced-order model are therefore **assumptions/calibrations**, not experimentally identified cycle settings.

**Liske & Rajendran isotherm.** The 228-row canonical equilibrium grid (2 gases × 6 temperatures × 19 pressures) is explicitly labelled a *deterministic reconstruction*, not raw experimental observations. Recomputing the dual-site Langmuir equation independently from the bundled published parameters yields maximum absolute deviation **5.0e-9 mol/kg**, consistent with eight-decimal CSV rounding; loading is nonnegative and monotonically increasing with pressure at every gas/temperature setting. The coefficients (capacities 2.387/3.271 mol/kg; CO2 heats 35.06/28.95 kJ/mol; N2 primary heat 17.96 kJ/mol) match Liske & Rajendran, DOI 10.1002/cjce.70453.

**Wilkins alternative isotherm.** The 240-row independent equilibrium sensitivity grid (2 gases × 8 temperatures × 15 pressures) similarly reproduces its stored DSL parameters with maximum absolute deviation **5.2e-9 mol/kg**, and monotonic/nonnegative loading. This is a *separate CALF-20 sample parameterization* and is not a second set of direct observations for the nominal calibration.

**Independent pilot dataset.** All 75 rows are populated for the reported process variables; contactor counts are **61 laminate and 14 packed**, with feed fractions **0.08, 0.12, 0.16**. The source-table group counts (S1 6; S2 7; S3-1 20; S3-2 16; S4 12; S5 14) sum to 75. Every stored cycle time equals its stored step-time sum, and all recorded purities/recoveries lie between zero and one. The campaign counts and headline results agree with Wilkins et al. (2026), DOI 10.1016/j.cej.2026.174238, and the open CC BY 4.0 preprint DOI 10.26434/chemrxiv-2025-zz73j. **This is not a page-by-page human comparison of all 75 individual entries against the primary SI PDF**; that remains a required independent audit before claiming exhaustive transcription verification.

**Pereira et al. correction.** One historical source identifier, `CALDERON2025_INORGCHEM_CALF20`, incorrectly names the first author. The DOI **10.1021/acs.inorgchem.4c04727 actually belongs to Pereira et al. (2025)**, *Enhancing CO2 Capture Via Fast Microwave-Assisted Synthesis of the CALF-20 Metal-Organic Framework*. The numerical values transcribed for the separate CALF-20-SOL batch are nevertheless correct against published Table 2: 4.65 mol/kg at 273 K, 3.38 mol/kg at 298 K (1 bar), and zero-loading adsorption heat 39.09 kJ/mol. The audited release corrects bibliographic author/title metadata while retaining the legacy machine identifier for backwards compatibility. This is **provenance correction, not a change to the scientific numeric inputs**.

**Other context sources checked:** Nguyen et al. (2022), DOI 10.1016/j.cej.2022.136263; Tang et al. (2025), DOI 10.1021/acs.iecr.5c02322; Dilipkumar et al. (2026), DOI 10.1016/j.cej.2026.177413. These support background/transport context. Not every referenced third-party numeric fact has been independently checked against a primary page or table in this limited audit.

## 3. Manuscript/results consistency

* The frozen-boundary source CSV reports **14.190–15.075% CO2**; the paper gives **14.19–15.07%** (a small 0.005 percentage-point rounding convention difference). Consider using **15.075%** in text and tables, or consistently **15.08%** for two decimals, in the next editorial update. This does not alter the actual numerical result.
* One-actuator and two-actuator lean boundaries agree across the source table and manuscript at **13.400%** and approximately **10.590%**. The declared equipment-realizable coordinated boundary is approximately **10.95%**, and the conventional 95%/90% specification has **no frozen feasible interval**.
* The separate composition–throughput grid contains **205** scenarios. Stored success counts are **37/205=18.0488%** frozen, **136/205=66.3415%** at most one actuator, and **157/205=76.5854%** at most two actuators. They are discrete grid coverage, *not continuous feasible volume*.
* Four-state disturbance comparisons are numerical quasi-steady policy evaluations (not closed-loop transient tests). The control-effort figures are **1.47345** for energy-first re-optimization and **1.03050** for minimum-authority policy, a **30.0620%** reduction by the declared log-movement definition. Both on-spec policies have 4/4 on-spec states in the saved table.
* The paper/SI and figure asset snapshots are mutually aligned. SI figure numbering S01–S21 is an editorial crosswalk from historical calculation source IDs; retain `docs/PUBLICATION_FIGURE_CROSSWALK.md` during figure regeneration.

## 4. Software and data-package refinements introduced

1. The original README advised a fresh venv then `python -m pytest` without installing pytest. The audited README now installs pytest explicitly and includes the new audit location.
2. Added `requirements-dev.txt` and eight independent checks in `tests/test_scientific_data_consistency.py` covering DSL reconstruction, source-data bounds, main/table boundaries, scenario/effort arithmetic and the Pereira scalar values.
3. Corrected the inaccurate Pereira bibliography name/title and corresponding source note. The historical source ID is left untouched to preserve all source crosswalks.
4. Clarified that `data/audit/literature_data_integrity_audit.csv` contains a historical data-pipeline version **2.3.0**, not the current executable research-code version **6.6.0**. The archived record itself remains unchanged.
5. Preserved every original `src/p10/*.py` scientific calculation module and every original numeric `.csv` file **byte-for-byte**. No simulation result was recomputed or silently altered for this editorial/data-integrity release.

## 5. Remaining release gates and risk assessment

**Required before asserting numerical reproducibility:** run the *full* `--level comprehensive --uncertainty-draws 2000` workflow into a clean output folder and compare every main figure source/table with tolerances and a discrepancy log. The 8-draw quickcheck and 23 small tests do not replace this.

**Required before claiming experimental/process validation:** obtain and verify the appropriate cyclic performance transients, exact operating history and column/valve/thermal parameters. The nominal 2023 experimental purity/recovery is a scalar anchor, not proof of off-design cycle response. The optional CSS model and external 75-run pilot data are not experiment-matched independent validation of this surrogate.

**Engineering-density convention check:** The reduced-order settings use 650 kg/m3 as an assumed adsorbent bulk density, whereas the Liske reference parameter table lists 570 kg/m3 as a separate material-density parameter. Because these values arise from different data roles/samples, they are not directly interchangeable; the co-authors should confirm whether each denotes crystalline, particle or packed-bed density before making volumetric scale-up claims. The model explicitly varies its density assumption, but that does not independently establish the physical definition.

**Required before public publication/deposition:** confirm source licences (including third-party table reproduction), add an agreed code licence, inspect author/affiliation/funding declarations, verify all source facts used in the manuscript at primary pages, and review detailed figure readability at journal print scale. The paper currently communicates these scientific limitations appropriately.

**Release recommendation:** use `P10_GITHUB_AUDITED_2026-10-29.zip` for the private GitHub repository, with the existing Overleaf manuscript archive unchanged until the co-authors reconcile the 15.075% editorial precision and complete their independent full rerun. No substantial numerical flaw has been demonstrated by the completed checks; neither has the entire physical model been independently validated.

## Primary references used for this audit

- Nguyen et al. 2023: https://doi.org/10.1016/j.cej.2022.139550
- Nguyen et al. 2022: https://doi.org/10.1016/j.cej.2022.136263
- Liske & Rajendran 2026: https://doi.org/10.1002/cjce.70453
- Wilkins et al. 2026: https://doi.org/10.1016/j.cej.2026.174238
- Open pilot SI (CC BY 4.0): https://doi.org/10.26434/chemrxiv-2025-zz73j
- Pereira et al. 2025: https://doi.org/10.1021/acs.inorgchem.4c04727
- Tang et al. 2025: https://doi.org/10.1021/acs.iecr.5c02322
- Dilipkumar et al. 2026: https://doi.org/10.1016/j.cej.2026.177413
