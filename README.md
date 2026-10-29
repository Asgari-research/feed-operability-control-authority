# CALF-20 feed-composition operability and control authority

**GitHub repository:** `Asgari-research/feed-operability-control-authority`

**Suggested GitHub description:** Reproducible reduced-order analysis of CALF-20 CO2/N2 VSA feed-composition operability, minimum actuator authority, equipment constraints and publication source data.

This research code assesses the feasible feed envelope of a fixed nominal cycle and the additional operation recovered by adjusting regeneration pressure, adsorption-time scale or both. It uses literature-derived equilibrium inputs and a published nominal process anchor. **It does not constitute independently validated off-design cyclic-steady-state performance.**

## Files

- `p10_feed_composition_robustness.py`: main command-line entry point.
- `src/p10/`: calculation, analysis, plotting and finalisation modules.
- `data/`: canonical three-file inputs and supporting provenance, source conditions and assumptions.
- `results/final_submission/`: curated reference figures (PDF), figure source data, tables and provenance CSVs.
- `tests/`: regression and package-integrity checks.
- `docs/`: methodological limits and software notes.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q tests
python .\p10_feed_composition_robustness.py --input-folder .\data\input_folder --output-folder .\runs\quickcheck --save-mode light --ram-mode ultra_light --level quickcheck --n-jobs 1 --uncertainty-draws 8 --fresh-run
```

Full calculation (more expensive):

```powershell
python .\p10_feed_composition_robustness.py --input-folder .\data\input_folder --output-folder .\runs\production --save-mode rich --ram-mode light --level comprehensive --n-jobs 1 --uncertainty-draws 2000 --fresh-run
```

`--fresh-run` deletes the output folder **only**, so use a dedicated `runs` path. Omit this switch when resuming a checkpoint-compatible interrupted run. Set the random seed explicitly if comparing against published outputs. Output directories are never inside the version-controlled reference snapshot.

The optional `p10_css_validation.py` is independent and should not be represented as validation of the main model unless its experiment-matched gates pass.

See `RUN_INSTRUCTIONS.md`, `docs/` and the provenance registry. Source citations and licences require a human evidence audit before public release; not all source metadata are proof of downloadable numerical observations.

## Manuscript and source-data precision

The archived frozen upper boundary is 15.075% inlet CO2; the updated manuscript and Figure 5 display 15.08% to two decimal places. The distinction between the 650 kg/m3 engineering density assumption and the 570 kg/m3 literature model parameter is documented in `data/ENGINEERING_DENSITY_CONVENTION.md`. For paper/code cross-references see `docs/OVERLEAF_GITHUB_ALIGNMENT_2026-10-29.md`. Only the figure-label rounding formatter was changed in `src/p10/analysis.py`; calculation equations and numeric CSVs were not altered.

## Scientific and provenance audit

See `docs/INDEPENDENT_PACKAGE_AUDIT_2026-10-29.md` and `tests/test_scientific_data_consistency.py`. The audit verifies the package, certain source values and the deterministic reconstructions; it is not a full 2000-draw process-model rerun, experimental validation or a row-by-row transcription check against every original literature table. The source identifier `CALDERON2025_INORGCHEM_CALF20` is retained as a historical internal key but corresponds to **Pereira et al. (2025)**, DOI `10.1021/acs.inorgchem.4c04727`.
