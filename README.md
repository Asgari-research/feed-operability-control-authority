# CALF-20 feed-composition operability and control authority

**Suggested GitHub repository:** `calf20-feed-operability-control-authority`

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
