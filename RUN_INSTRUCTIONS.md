# Command sequence and checks

From the repository root, preferably in a fresh Windows PowerShell environment:

1. `py -3.11 -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt pytest`
5. `python -m pytest -q tests`
6. `python .\p10_feed_composition_robustness.py --help`
7. Run the `quickcheck` command in the README (smoke test only).
8. Run `comprehensive` into a fresh `runs/production` folder; use 2000 uncertainty draws to match the reference settings.
9. Preserve `run_summary`, `logs`, `publication/provenance`, and the output manifest. Compare files and numerical tolerances against `results/final_submission`.
10. Produce a documented PASS / DIFFERENCE EXPLAINED / FAIL table for each main claim.

The complete run can be resource-intensive and its runtime depends on the machine. Do not substitute the quickcheck outputs for published results. See the separate student verification assignment for source-level auditing.
